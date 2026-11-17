"""Freeze a replacement stock and assemble independent-stock results safely."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from main import read_summary
from event_sources import candidates

MODEL_CODE = ['event_data.py','event_sources.py','event_models.py','event_training.py',
              'main.py','features.py','compact_features.py']
RECIPE = ['seeds','folds','test_size','val_size','embargo','window','max_brokers','width',
          'batch_size','epochs','patience','lr','no_amp','no_ablation','permutation_repeats',
          'fee_bps','slippage_bps']
TABLES = ['metrics','calibration','predictions','training','history','gates','gate_samples',
          'importance','ablation','samples','backtest','trades','equity','regime_metrics']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compatible(original,replacement):
    original,replacement=map(Path,[original,replacement])
    old=json.loads((original/'config.json').read_text());new=json.loads((replacement/'config.json').read_text())
    for key in RECIPE:
        if old[key]!=new[key]:raise ValueError(f'Recipe mismatch: {key}')
    for name in MODEL_CODE:
        if (original/'sources'/name).read_bytes()!=(replacement/'sources'/name).read_bytes():
            raise ValueError(f'Model/preprocessing code differs: {name}; cannot silently reuse fits')
    return old,new


def freeze(ticker, out, base, reference):
    """Copy actual source bytes, including overlap, before looking at test scores."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    base=Path(base);root=out/'raw-source';records=[]
    def copy(path, relative):
        data=Path(path).read_bytes();dest=root/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        records.append(dict(path=str(relative),original=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    reference=Path(reference)
    copy(reference,Path('Daily/Summary/ADRO_Summary.csv'))
    calendar=read_summary(reference)[0].index
    copy(base/'Daily/Summary'/f'{ticker}_Summary.csv',Path('Daily/Summary')/f'{ticker}_Summary.csv')
    for day in calendar:
        for investor in ['All','Foreign']:
            for path in candidates(base/'Daily'/investor,ticker,day,investor):
                if path.exists():copy(path,path.relative_to(base))
    for path in (base/'Monthly/Ownership/lbrpe').glob(f'*/{ticker}.json'):
        copy(path,path.relative_to(base))
    (out/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(dict(ticker=ticker,files=len(records),reference_sessions=len(calendar),
        reference_start=str(calendar.min().date()),reference_end=str(calendar.max().date()),root=str(root)),indent=2))


def assemble(original, replacement, replacement_source, out):
    original,replacement,out=map(Path,[original,replacement,out])
    old,new=compatible(original,replacement)
    ticker=new['tickers'][0]
    if len(new['tickers'])!=1 or ticker in ['ADRO','BBRI']:
        raise ValueError('Replacement run must contain exactly one new stock')
    # Recipes are matched, but code/source hashes retain per-run provenance.
    stocks=['ADRO',ticker,'BBRI'];out.mkdir(parents=True,exist_ok=False)
    lineage=[]
    roots={'ADRO':original,'BBRI':original,ticker:replacement}
    for stock,source in roots.items():
        shutil.copytree(source/stock,out/stock)
        for suffix in ['daily.npz','data-audit.json','ohlcv-snapshot.csv']:
            shutil.copy2(source/f'{stock}-{suffix}',out/f'{stock}-{suffix}')
        for checkpoint in sorted((out/stock).glob('fold-*/*/best.pt')):
            relative=checkpoint.relative_to(out)
            if sha(checkpoint)!=sha(source/relative):raise ValueError('Checkpoint changed during copying')
            lineage.append(dict(ticker=stock,checkpoint=str(relative),sha256=sha(checkpoint),
                                origin=str(source/relative),reused=source==original))
    for table in TABLES:
        parts=[]
        for source,chosen in [(original,['ADRO','BBRI']),(replacement,[ticker])]:
            frame=pd.read_csv(source/f'{table}.csv');parts.append(frame[frame.ticker.isin(chosen)])
        pd.concat(parts,ignore_index=True).to_csv(out/f'{table}.csv',index=False)
    predictions=pd.read_csv(out/'predictions.csv')
    if predictions.duplicated(['ticker','fold','model','seed','date','horizon']).any():
        raise ValueError('Duplicate predictions in assembled cohort')
    if set(predictions.ticker)!=set(stocks):raise ValueError('Wrong stock membership')
    samples=pd.read_csv(out/'samples.csv')
    reference=samples[(samples.ticker=='ADRO')].drop(columns='ticker').sort_values(['fold','split','calendar_index']).reset_index(drop=True)
    for stock in stocks[1:]:
        other=samples[samples.ticker==stock].drop(columns='ticker').sort_values(['fold','split','calendar_index']).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference,other)
    # Copy ONLY selected stocks' immutable raw files. The ADRO calendar remains
    # identical; it supplies trading dates, not ADRO features to other models.
    raw=out/'raw-source';source_map={s:Path(old['summary']).parents[1] for s in ['ADRO','BBRI']}
    source_map[ticker]=Path(replacement_source)
    for stock,source in source_map.items():
        for path in source.rglob('*'):
            if path.is_file() and (path.name.startswith(stock+'_') or path.name.startswith(stock+'-')
                                   or path.name in [stock+'.json',stock+'.xls']):
                dest=raw/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() and sha(dest)!=sha(path):raise ValueError('Raw snapshot collision')
                shutil.copy2(path,dest)
    shutil.copytree(original/'sources',out/'sources-original')
    shutil.copytree(replacement/'sources',out/'sources')
    config=dict(old,tickers=stocks,out=str(out),summary=str(raw/'Daily/Summary'),brokers=str(raw/'Daily/All'),
                foreign=str(raw/'Daily/Foreign'),ownership=str(raw/'Monthly/Ownership'),
                calendar_summary=str(raw/'Daily/Summary/ADRO_Summary.csv'),
                code_hash=None,code_hash_definition='Assembled independent fits; per-stock training code hashes retained below.',
                source_training_code_hashes={s:old['code_hash'] if s!=ticker else new['code_hash'] for s in stocks},
                matched_model_code_sha256={name:sha(original/'sources'/name) for name in MODEL_CODE},
                assembly=dict(original=str(original),replacement=str(replacement),new_fits=sum(not x['reused'] for x in lineage),
                              reused_fits=sum(x['reused'] for x in lineage),note='Independent per-stock models; no pooled training or averaging of model weights.'))
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'lineage.json').write_text(json.dumps(lineage,indent=2),encoding='utf-8')
    for name in ['verification.json']:
        shutil.copy2(original/name,out/'original-verification.json');shutil.copy2(replacement/name,out/'replacement-verification.json')
    frame=pd.read_csv(out/'metrics.csv');keys=['mae','rmse','accuracy','balanced_accuracy','roc_auc','brier','calibration_ece','vol_mae','vol_rmse']
    frame.groupby(['model','horizon'])[keys].mean().reset_index().to_csv(out/'comparison.csv',index=False)
    frame.groupby('model')[keys].mean().reset_index().to_csv(out/'macro_summary.csv',index=False)
    (out/'progress.json').write_text(json.dumps(dict(status='complete',trained_fits=len(lineage),**config['assembly'])),encoding='utf-8')
    from dashboard import build_event_dashboard
    from event_report import write_report
    build_event_dashboard(out);write_report(out)
    print(json.dumps(config['assembly'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze');f.add_argument('--ticker',required=True);f.add_argument('--out',required=True)
    f.add_argument('--base',default='data');f.add_argument('--reference',required=True)
    a=sub.add_parser('assemble');a.add_argument('--original',required=True);a.add_argument('--replacement',required=True)
    a.add_argument('--replacement-source',required=True);a.add_argument('--out',required=True)
    args=vars(p.parse_args());action=args.pop('action')
    (freeze if action=='freeze' else assemble)(**args)
