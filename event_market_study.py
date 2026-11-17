"""Paired local-input versus Yahoo-context experiment; freeze all selections before test."""
import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from event_data import preprocess
from event_market_data import augment,sha,dump
from event_objective_study import runtime
from event_training import fit,infer,decoded,diagnostics
from event_predict import restore
from event_evaluation import metrics,backtest
from event_uncertainty import intervals

ARMS=['local_only','market_context']


def load_raw(source,ticker):
    cache=np.load(source/f'{ticker}-daily.npz',allow_pickle=False)
    meta=json.loads((source/f'{ticker}-data-audit.json').read_text())
    data={k:cache[k] for k in cache.files};data['dates']=pd.DatetimeIndex(data['dates'])
    data.update(ticker=ticker,names=meta['features'],groups=meta['groups'],source_hash=meta['source_hash'])
    data['frame']=pd.read_csv(source/f'{ticker}-ohlcv-snapshot.csv',index_col=0,parse_dates=True)
    assert data['frame'].index.equals(data['dates'])
    return data


def save_tables(root,tables):
    for name,rows in tables.items():
        if rows:pd.DataFrame(rows).to_csv(root/f'{name}.csv',index=False)


def market_regime(raw,rows,train):
    value=raw['x'][:,raw['names'].index('ihsg_volatility20')]
    missing=raw['x'][:,raw['names'].index('ihsg_volatility20_missing')]>0
    good=train[~missing[train]]
    if not len(good):return np.full(len(rows),'unknown'),None
    cutoff=float(np.median(value[good]))
    return np.where(missing[rows],'unknown',np.where(value[rows]>cutoff,'high_market_vol','low_market_vol')),cutoff


def run(args):
    args.device=runtime(args.device);root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    source=Path(args.source).resolve();market=Path(args.market).resolve()
    old=json.loads((source/'config.json').read_text());stocks=args.tickers or old['tickers'];seeds=args.seeds or old['seeds']
    folds=list(range(1,args.folds+1))
    config=dict(study_type='market_context',source=str(source),market_source=str(market),out=str(root),
        tickers=stocks,seeds=seeds,folds=folds,arms=ARMS,device=args.device,
        window=old['window'],width=old['width'],batch_size=old['batch_size'],lr=old['lr'],no_amp=old['no_amp'],
        epochs=args.epochs,patience=args.patience,embargo=old['embargo'],selection_metric='return_mae',
        training_objective='multitask',planned_fits=len(stocks)*len(seeds)*len(folds)*2,
        bootstrap_repeats=args.bootstrap_repeats,permutation_repeats=args.permutation_repeats,
        fee_bps=15.,slippage_bps=10.,torch=torch.__version__,
        availability='source daily date D +1 calendar day at12:00WIB; backward asof at query EOD23:59; stale7days after cutoff',
        controls='Same input dimension and initial weights, market columns zero in control; all arms trained afresh',
        test_status='Historical exploratory reuse; no untouched confirmation test',
        sector='Rejected: each Yahoo candidate has only1bar',
        original_config_sha256=sha(source/'config.json'),market_manifest_sha256=sha(market/'manifest.json'))
    dump(root/'config.json',config);shutil.copytree(market,root/'market-source');(root/'sources').mkdir()
    code=['event_market_study.py','event_market_data.py','event_market_verify.py','event_market_results.py',
        'event_training.py','event_models.py','event_data.py','event_sources.py','event_predict.py','event_evaluation.py',
        'event_objective_study.py','event_uncertainty.py','main.py','features.py','compact_features.py','dashboard.py',
        'requirements.txt','requirements-market.txt','EVENT-MARKET-PROTOCOL.md','test_event_market.py']
    for name in code:shutil.copy2(Path(__file__).parent/name,root/'sources'/name)
    dump(root/'code_manifest.json',{name:sha(root/'sources'/name) for name in code})
    tables={key:[] for key in ['training','history','gates','importance','pairs']};fits=[];manifest=[]
    settings=SimpleNamespace(**{k:config[k] for k in ['device','window','width','batch_size','lr','no_amp','epochs','patience','permutation_repeats']},
        out=str(root),training_objective='multitask',selection_metric='return_mae',log_grad_norm=True,importance_metric='return_mae')
    for ticker in stocks:
        raw=load_raw(source,ticker);full,receipts=augment(raw,root/'market-source',raw['frame'].Close)
        for receipt in receipts:
            joins=receipt.pop('joins');joins.to_csv(root/f'{ticker}-{receipt["key"]}-availability.csv',index=False)
        dump(root/f'{ticker}-market-audit.json',dict(receipts=receipts,features=full['names'],groups=full['groups']))
        for name in [f'{ticker}-daily.npz',f'{ticker}-data-audit.json',f'{ticker}-ohlcv-snapshot.csv']:
            shutil.copy2(source/name,root/name)
        np.savez_compressed(root/f'{ticker}-augmented.npz',x=full['x'])
        for fold in folds:
            origin=source/ticker/f'fold-{fold}';old_arrays=np.load(origin/'prepared.npz',allow_pickle=False)
            split={key:old_arrays[key+'_indices'] for key in ['train','val','test']};arms={}
            for arm in ARMS:
                current=dict(full);current['x']=full['x'].copy()
                if arm=='local_only':current['x'][:,raw['x'].shape[1]:]=0
                data,state=preprocess(current,split['train'],config['window'])
                for key in ['tokens','ids','mask','y']:np.testing.assert_array_equal(data[key],old_arrays[key])
                np.testing.assert_array_equal(data['x'][:,:raw['x'].shape[1]],old_arrays['x'])
                folder=root/ticker/f'fold-{fold}'/arm;folder.mkdir(parents=True)
                np.savez_compressed(folder/'prepared.npz',**{k:data[k] for k in ['x','tokens','ids','mask','y']},
                    **{key+'_indices':value for key,value in split.items()})
                torch.save(dict(preprocess=state,features=data['names'],groups=data['groups'],source_hash=data['source_hash']),folder/'preprocess.pt')
                manifest.append(dict(ticker=ticker,fold=fold,input_arm=arm,path=str(folder.relative_to(root)),
                    prepared_sha256=sha(folder/'prepared.npz'),original_prepared_sha256=sha(origin/'prepared.npz'),
                    old_feature_count=raw['x'].shape[1],counts={key:len(value) for key,value in split.items()}))
                arms[arm]=(data,state,folder)
            for seed in seeds:
                hashes=[]
                for arm in ARMS:
                    data,state,folder=arms[arm];destination=folder/f'none-seed{seed}'
                    print(f'FIT {ticker} fold={fold} seed={seed} arm={arm}',flush=True)
                    model,record,history,gates=fit(data,split,state,settings,'none',seed,destination)
                    context=dict(ticker=ticker,fold=fold,seed=seed,backbone='none',input_arm=arm,model='none/'+arm,
                                 training_objective='multitask')
                    tables['training'].append(dict(context,**record));tables['history'].extend([{**context,**r} for r in history])
                    tables['gates'].extend([{**context,**r} for r in gates]);hashes.append(record['initial_weights_sha256'])
                    if arm=='market_context':
                        regimes,cutoff=market_regime(full,split['val'],split['train'])
                        for reg in ['all']+list(np.unique(regimes)):
                            subset=split if reg=='all' else dict(split,val=split['val'][regimes==reg])
                            values=diagnostics(model,data,subset,state,settings,seed)
                            tables['importance'].extend([{**context,'regime':reg,'metric':'validation_return_mae',
                                'market_vol_cutoff':cutoff,'supported_in_train':not state['constant_columns'][data['groups'][r['group']]].all(),**r} for r in values])
                    checkpoint=destination/'best.pt'
                    fits.append(dict(context,path=str(checkpoint.relative_to(root)),checkpoint_sha256=sha(checkpoint),
                        prepared_sha256=sha(folder/'prepared.npz'),best_epoch=record['best_epoch'],validation_return_mae=record['val_loss']))
                    del model
                if len(set(hashes))!=1:raise ValueError('Paired initialization mismatch')
                tables['pairs'].append(dict(ticker=ticker,fold=fold,seed=seed,initial_weights_sha256=hashes[0],equal_initialization=True))
                save_tables(root,tables)
                from dashboard import build_event_dashboard
                build_event_dashboard(root)
    if len(fits)!=config['planned_fits']:raise ValueError('Incomplete fit grid')
    dump(root/'data_manifest.json',manifest);dump(root/'selected_checkpoints.json',fits)
    dump(root/'phase1_frozen.json',{name:sha(root/name) for name in ['config.json','code_manifest.json','data_manifest.json','selected_checkpoints.json']})
    evaluate(root)


def evaluate(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());device=runtime(config['device'])
    frozen=json.loads((root/'phase1_frozen.json').read_text())
    for name,digest in frozen.items():assert sha(root/name)==digest
    fits=json.loads((root/'selected_checkpoints.json').read_text())
    tables={name:[] for name in ['metrics','calibration','predictions','validation_metrics','backtest','trades','equity','regime_metrics']}
    for item in fits:
        checkpoint=root/item['path'];assert sha(checkpoint)==item['checkpoint_sha256']
        model,saved=restore(checkpoint,device);folder=checkpoint.parents[1]
        assert sha(folder/'prepared.npz')==item['prepared_sha256']
        arrays=np.load(folder/'prepared.npz',allow_pickle=False);data={k:arrays[k] for k in ['x','tokens','ids','mask','y']}
        state=saved['preprocess'];settings=SimpleNamespace(window=saved['window'],batch_size=config['batch_size'],device=device)
        meta=json.loads((root/f'{item["ticker"]}-data-audit.json').read_text());dates=meta['dates']
        context={key:item[key] for key in ['ticker','fold','seed','backbone','input_arm','model']}
        for split in ['val','test']:
            rows=arrays[split+'_indices'];out,_=infer(model,data,rows,settings,state);returns,p,vol=decoded(out,state)
            scores,bins=metrics(data['y'][rows],returns,p,vol)
            tables['validation_metrics' if split=='val' else 'metrics'].extend([{**context,**r} for r in scores])
            if split=='val':
                loss=float(abs(returns-data['y'][rows,:3]).mean())
                np.testing.assert_allclose(loss,saved['record']['val_loss'],atol=2e-6,rtol=2e-5)
                continue
            tables['calibration'].extend([{**context,**r} for r in bins])
            for k,index in enumerate(rows):
                for j,h in enumerate([5,10,20]):
                    tables['predictions'].append(dict(context,date=dates[index],calendar_index=int(index),horizon=h,
                        actual=float(data['y'][index,j]),prediction=float(returns[k,j]),probability=float(p[k,j]),
                        actual_vol=float(np.exp(data['y'][index,j+3])),predicted_vol=float(vol[k,j])))
        del model
    source=pd.read_csv(Path(config['source'])/'predictions.csv')
    controls=source[source.ticker.isin(config['tickers'])&source.fold.isin(config['folds'])&source.model.isin(['zero','train_mean','lightgbm_manual'])]
    controls.to_csv(root/'baseline_source_predictions.csv',index=False)
    for key,part in controls.groupby(['ticker','fold','seed','model']):
        ticker,fold,seed,name=key;context=dict(ticker=ticker,fold=fold,seed=seed,model=name,backbone='baseline',input_arm='baseline')
        actual=np.column_stack([part[part.horizon==h].sort_values('calendar_index').actual for h in [5,10,20]]+
            [np.log(part[part.horizon==h].sort_values('calendar_index').actual_vol) for h in [5,10,20]])
        r,p,v=[np.column_stack([part[part.horizon==h].sort_values('calendar_index')[c] for h in [5,10,20]])
               for c in ['prediction','probability','predicted_vol']]
        scores,bins=metrics(actual,r,p,v);tables['metrics'].extend([{**context,**row} for row in scores])
        tables['calibration'].extend([{**context,**row} for row in bins])
        tables['predictions'].extend([{**row,**context} for row in part.to_dict('records')])
    prediction=pd.DataFrame(tables['predictions'])
    for (ticker,model,seed,horizon),part in prediction.groupby(['ticker','model','seed','horizon']):
        frame=pd.read_csv(root/f'{ticker}-ohlcv-snapshot.csv',index_col=0,parse_dates=True)
        for scenario,mult in [('base',1),('stress',2)]:
            stats,trades,curve=backtest(frame,part,int(horizon),config['fee_bps']*mult,config['slippage_bps']*mult)
            context=dict(ticker=ticker,model=model,seed=seed,horizon=horizon,cost_scenario=scenario)
            tables['backtest'].append(dict(context,**stats));tables['trades'].extend([{**context,**r} for r in trades])
            tables['equity'].extend([{**context,**r} for r in curve])
    save_tables(root,tables)
    frame=pd.DataFrame(tables['metrics']);columns=['mae','rmse','accuracy','balanced_accuracy','roc_auc','brier','calibration_ece','vol_mae','vol_rmse']
    frame.groupby(['ticker','horizon','model'])[columns].mean().reset_index().to_csv(root/'by_stock_horizon.csv',index=False)
    frame.groupby('model')[columns].mean().reset_index().to_csv(root/'macro_summary.csv',index=False)
    frame.groupby(['model','horizon'])[columns].mean().reset_index().to_csv(root/'comparison.csv',index=False)
    statistics(root,config)
    dump(root/'progress.json',dict(status='complete',new_fits=len(fits),pairs=len(fits)//2,test_status=config['test_status']))
    from dashboard import build_event_dashboard
    build_event_dashboard(root);print('COMPLETE',root,flush=True)


def statistics(root,config):
    frame=pd.read_csv(root/'predictions.csv');frame['loss']=abs(frame.prediction-frame.actual)
    losses=frame.groupby(['date','ticker','horizon','model']).loss.mean().unstack('model')
    matrices=[];keys=[]
    for treatment,control in [('none/market_context','none/local_only'),('none/market_context','zero'),('none/local_only','zero')]:
        wide=(losses[control]-losses[treatment]).unstack(['ticker','horizon']).sort_index()
        if not np.isfinite(wide.to_numpy()).all():raise ValueError('Unequal test coverage')
        matrices.append(np.column_stack([wide,wide.mean(axis=1)]))
        keys.extend([(treatment,control,*key) for key in wide.columns]);keys.append((treatment,control,'ALL','ALL'))
    matrix=np.column_stack(matrices);rows=[]
    for block in [20,40]:
        bounds=intervals(matrix,block=block,repeats=config['bootstrap_repeats'])
        for j,(treatment,control,ticker,horizon) in enumerate(keys):
            rows.append(dict(treatment=treatment,comparator=control,ticker=ticker,horizon=horizon,block_sessions=block,
                dates=len(matrix),gain_mae=bounds[0][j],simultaneous_low=bounds[1][j],simultaneous_high=bounds[2][j],
                nominal_low=bounds[3][j],nominal_high=bounds[4][j],family_columns=matrix.shape[1]))
    pd.DataFrame(rows).to_csv(root/'paired_bands.csv',index=False)
    metric=pd.read_csv(root/'metrics.csv');paired=[]
    for (ticker,horizon),part in metric[metric.backbone=='none'].groupby(['ticker','horizon']):
        pivot=part.pivot(index=['fold','seed'],columns='input_arm',values='mae');gain=pivot.local_only-pivot.market_context
        paired.append(dict(ticker=ticker,horizon=horizon,gain_mae=gain.mean(),local_mae=pivot.local_only.mean(),
            market_mae=pivot.market_context.mean(),folds_won=int((gain.groupby(level='fold').mean()>0).sum()),
            seeds_won=int((gain.groupby(level='seed').mean()>0).sum())))
    pd.DataFrame(paired).to_csv(root/'paired_summary.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--source',default='runs/event-adro-antm-bbri-20261116')
    p.add_argument('--market',default='runs/event-market-source-network-20261117')
    p.add_argument('--tickers',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--folds',type=int,default=3);p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--patience',type=int,default=6);p.add_argument('--bootstrap-repeats',type=int,default=2000)
    p.add_argument('--permutation-repeats',type=int,default=3);p.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    run(p.parse_args())
