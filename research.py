"""Controlled per-stock feature ablation and purged expanding walk-forward."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge

from main import BASE, HORIZONS, Encoder, broker_row, read_summary
from features import expanded_features

VARIANTS=('ohlcv','broker_retail','combined')
PRICE_FEATURES=set(BASE[:7]) | {'price_return_20','price_return_20_missing','price_return_40','price_return_40_missing'}


def digest_state(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_stock(ticker, summary_path, brokers_path):
    df,good=read_summary(Path(summary_path)/f'{ticker}_Summary.csv')
    directories=sorted(p.name for p in Path(brokers_path).iterdir() if p.is_dir() and len(p.name)==10 and p.name[4]=='-')
    df=df.loc[:directories[-1]];good=good.reindex(df.index)
    cache=[];extras=[];snapshots=[];quality=[]
    for day in df.index:
        directory=Path(brokers_path)/f'{day:%Y-%m-%d}'
        path=directory/f'{ticker}_ALL_{day:%Y-%m-%d}.json'
        if not path.exists():path=directory/f'{ticker}.xls'
        if not path.exists():
            cache.append({});extras.append([1,0,0,0]);snapshots.append(None)
            quality.append({'date':str(day.date()),'issue':'missing broker file'})
            continue
        try:
            values,extra,detail=broker_row(path,ticker,day,details=True)
        except (ValueError,KeyError,TypeError) as error:
            cache.append({});extras.append([1,0,0,0]);snapshots.append(None)
            quality.append({'date':str(day.date()),'issue':str(error)})
            continue
        cache.append(values);extras.append(extra);snapshots.append(detail)
    return df,good,cache,np.array(extras),snapshots,quality


def make_folds(n_dates,folds=3,test_size=14,val_size=10,window=20):
    last_exclusive=n_dates-max(HORIZONS)
    first_test=last_exclusive-folds*test_size
    result=[]
    for fold in range(folds):
        start=first_test+fold*test_size;stop=start+test_size
        val_start=start-max(HORIZONS)-val_size
        train_end=val_start-max(HORIZONS)-1
        if train_end-(40+window-1)+1<30:
            raise ValueError('Insufficient history: need >=30 train windows per fold after purging')
        result.append({'fold':fold+1,'val_start':val_start,'test_start':start,'test_stop':stop,'train_end':train_end})
    return result


def fold_vocabulary(cache, train_end):
    return sorted(set().union(*(set(row) for row in cache[:train_end+1])))


def build_fold(loaded, fold,window=20):
    df,good,cache,extras,snapshots,quality=loaded
    eligible_train=[i for i in range(40+window-1,fold['train_end']+1)
                    if good.iloc[i-window+1:i+max(HORIZONS)+1].all()]
    if not eligible_train:raise ValueError('No eligible training input dates')
    vocabulary_end=max(eligible_train)
    vocabulary=fold_vocabulary(cache,vocabulary_end)
    ret=np.log(df.Close).diff()
    base=np.column_stack([ret,np.log(df.Open/df.Close.shift()),np.log(df.High/df.Low),
                          np.log(df.Close/df.Open),np.log1p(df.Volume).diff(),
                          np.log((df.Volume+1)/(df.Volume.rolling(20,min_periods=5).median()+1)),
                          ret.rolling(20,min_periods=5).std(),extras])
    known=set(vocabulary)
    net=np.array([[row.get(c,0) for c in vocabulary]+[sum(v for c,v in row.items() if c not in known)] for row in cache])
    expanded,names,coverage=expanded_features(df,vocabulary,snapshots)
    names=BASE+[f'net_share_{c}' for c in vocabulary]+['net_share_unseen']+names
    matrix=np.column_stack([base,net,expanded]).astype('float32')
    x=[];y=[];metadata=[];splits=[]
    for i in range(40+window-1,len(df)-max(HORIZONS)):
        bx=matrix[i-window+1:i+1]
        if not good.iloc[i-window+1:i+max(HORIZONS)+1].all() or not np.isfinite(bx).all():continue
        if i+max(HORIZONS)<fold['val_start']:split='train'
        elif i>=fold['val_start'] and i+max(HORIZONS)<fold['test_start']:split='val'
        elif fold['test_start']<=i<fold['test_stop']:split='test'
        else:continue
        x.append(bx);y.append([np.log(df.Close.iloc[i+h]/df.Close.iloc[i]) for h in HORIZONS])
        splits.append(split)
        metadata.append({'date':str(df.index[i].date()),'label_end':str(df.index[i+max(HORIZONS)].date()),'calendar_index':i})
    meta=pd.DataFrame(metadata)
    masks={s:np.array(splits)==s for s in ['train','val','test']}
    if any(not m.any() for m in masks.values()):raise ValueError('Empty fold split')
    x=np.array(x,dtype='float32');y=np.array(y,dtype='float32')
    price=np.array([name in PRICE_FEATURES for name in names])
    groups={'ohlcv':price,'broker_retail':~price,'combined':np.ones(len(names),dtype=bool)}
    audit={'fold':fold['fold'],'vocabulary':vocabulary,'features':names,
           'val_start':str(df.index[fold['val_start']].date()),'test_start':str(df.index[fold['test_start']].date()),
           'test_end':str(df.index[fold['test_stop']-1].date()),
           'vocabulary_last_input_date':str(df.index[vocabulary_end].date()),
           'counts':{s:int(m.sum()) for s,m in masks.items()},
           'feature_counts':{s:int(m.sum()) for s,m in groups.items()},'source_quality':quality,
           'samples':[{**row,'split':split} for row,split in zip(metadata,splits)]}
    return x,y,meta,masks,groups,audit


def normalize_fold(x,y,masks):
    train=masks['train']
    mean=x[train].mean(axis=(0,1));std=x[train].std(axis=(0,1)).clip(.01)
    ym=y[train].mean(axis=0);ys=y[train].std(axis=0).clip(.001)
    return np.clip((x-mean)/std,-10,10), (y-ym)/ys, mean,std,ym,ys


def fit_variant(xn,yn,y,masks,allowed,seed,epochs,device,save_path=None,return_state=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    x=xn.copy();x[:,:,~allowed]=0
    # Match convolution input energy despite zeroed channels. Reconstruction
    # targets remain the unamplified standardized values for equal loss scaling.
    input_gain=float(np.sqrt(x.shape[2]/int(allowed.sum())))
    model=Encoder(x.shape[2],x.shape[1],channels=16,compact_decoder=True).to(device)
    initial_hash=digest_state(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[masks['train']]),torch.from_numpy(yn[masks['train']])),
                      batch_size=32,shuffle=True,generator=generator)
    val=torch.from_numpy(x[masks['val']]).to(device);target=torch.from_numpy(yn[masks['val']]).to(device)
    feature_mask=torch.tensor(allowed,device=device).reshape(1,1,-1)
    best=float('inf');state=None;best_epoch=None;history=[]
    for epoch in range(epochs):
        model.train();total=0
        for bx,by in loader:
            bx=bx.to(device);by=by.to(device)
            hidden=torch.rand(len(bx),bx.shape[1],1,device=device)<.2
            # Guarantee at least one masked day even for the smallest final batch.
            if not hidden.any():hidden[0,0,0]=True
            _,pred,reconstruction=model(bx.masked_fill(hidden,0)*input_gain)
            loss=(pred-by).square().mean()+.2*(reconstruction-bx).square()[hidden & feature_mask].mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            total+=float(loss.detach())*len(bx)
        model.eval()
        with torch.no_grad():vl=float((model(val*input_gain)[1]-target).square().mean())
        history.append({'epoch':epoch+1,'train_loss':total/len(loader.dataset),'val_mse_standardized':vl})
        if vl<best:best=vl;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    if state is None:raise ValueError('Training produced no finite validation checkpoint')
    model.load_state_dict(state);model.eval()
    with torch.no_grad():prediction=model(torch.from_numpy(x[masks['test']]).to(device)*input_gain)[1].cpu().numpy()
    record={'seed':seed,'best_epoch':best_epoch,'val_mse_standardized':best,'history':history,
            'initial_weights_sha256':initial_hash,'best_weights_sha256':digest_state(state),
            'parameters':sum(p.numel() for p in model.parameters()),'active_features':int(allowed.sum()),'input_gain':input_gain}
    if save_path:
        save_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':{k:v.cpu() for k,v in state.items()},'allowed_features':allowed,'record':record},save_path)
    if return_state:
        return prediction,record,{k:v.detach().cpu().clone() for k,v in state.items()}
    return prediction,record


def fit_ridge(xn,yn,masks,allowed):
    flat=xn[:,:,allowed].reshape(len(xn),-1)
    best=float('inf');chosen=None;alpha_chosen=None
    for alpha in [1.,100.,10000.]:
        model=Ridge(alpha=alpha).fit(flat[masks['train']],yn[masks['train']])
        score=float(np.mean((model.predict(flat[masks['val']])-yn[masks['val']])**2))
        if score<best:best=score;chosen=model;alpha_chosen=alpha
    return chosen.predict(flat[masks['test']]),alpha_chosen


def summarize(output):
    rows=[]
    for (ticker,variant,seed,horizon),g in output.groupby(['ticker','variant','seed','horizon']):
        actual=g.actual.to_numpy();prediction=g.prediction.to_numpy()
        up=actual>0;correct=(prediction>0)==up
        rows.append({'ticker':ticker,'variant':variant,'seed':int(seed),'horizon':int(horizon),
                     'n_dates':len(g),'mae':float(np.mean(abs(actual-prediction))),
                     'ridge_mae':float(np.mean(abs(actual-g.ridge_prediction))),
                     'train_mean_mae':float(np.mean(abs(actual-g.train_mean_prediction))),
                     'direction_accuracy':float(correct.mean()),
                     'balanced_direction_accuracy':float((correct[up].mean()+correct[~up].mean())/2) if up.any() and (~up).any() else None})
    summary=pd.DataFrame(rows)
    average=summary.groupby(['ticker','horizon','variant'],as_index=False)[['mae','ridge_mae','train_mean_mae','direction_accuracy','balanced_direction_accuracy']].mean()
    comparison=average.pivot(index=['ticker','horizon'],columns='variant',values='mae').reset_index()
    comparison.columns.name=None
    comparison['combined_improvement_vs_ohlcv']=1-comparison.combined/comparison.ohlcv
    comparison['best_variant']=comparison[list(VARIANTS)].idxmin(axis=1)
    fold_rows=[]
    for (ticker,fold,variant,seed,horizon),g in output.groupby(['ticker','fold','variant','seed','horizon']):
        fold_rows.append({'ticker':ticker,'fold':fold,'variant':variant,'seed':seed,'horizon':horizon,
                          'mae':float(np.mean(abs(g.actual-g.prediction)))})
    by_fold=pd.DataFrame(fold_rows).groupby(['ticker','fold','horizon','variant'],as_index=False).mae.mean()
    fold_comparison=by_fold.pivot(index=['ticker','fold','horizon'],columns='variant',values='mae').reset_index()
    stability=fold_comparison.groupby(['ticker','horizon']).apply(lambda g:int((g.combined<g.ohlcv).sum()),include_groups=False).rename('folds_combined_beats_ohlcv').reset_index()
    comparison=comparison.merge(stability,on=['ticker','horizon'])
    macro=summary.copy();macro['relative_mae_to_train_mean']=macro.mae/macro.train_mean_mae
    macro=macro.groupby('variant',as_index=False).agg(mean_relative_mae=('relative_mae_to_train_mean','mean'),
          mean_direction_accuracy=('direction_accuracy','mean'),mean_balanced_accuracy=('balanced_direction_accuracy','mean'))
    return summary,comparison,macro,pd.DataFrame(fold_rows)


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    stock_manifest=json.loads((Path(args.input)/'manifest.json').read_text())
    tickers=args.tickers or [r['ticker'] for r in stock_manifest]
    predictions=[];runs=[];audits=[];fold_parameters={}
    for stock_number,ticker in enumerate(tickers,1):
        loaded=load_stock(ticker,args.summary,args.brokers)
        folds=make_folds(len(loaded[0]),args.folds,args.test_size,args.val_size)
        for fold in folds:
            x,y,meta,masks,groups,audit=build_fold(loaded,fold)
            audit['ticker']=ticker;audits.append(audit)
            xn,yn,mean,std,ym,ys=normalize_fold(x,y,masks)
            fold_parameters[(ticker,fold['fold'])]=(len(x),x.shape[2])
            refs=[]
            for variant in VARIANTS:
                allowed=groups[variant]
                ridge,alpha=fit_ridge(xn,yn,masks,allowed);ridge=ridge*ys+ym
                for seed in args.seeds:
                    model_path=root/'models'/ticker/f'fold-{fold["fold"]}'/f'{variant}-{seed}.pt' if args.save_models else None
                    pred,record=fit_variant(xn,yn,y,masks,allowed,seed,args.epochs,device,model_path)
                    pred=pred*ys+ym
                    record.update(ticker=ticker,fold=fold['fold'],variant=variant,ridge_alpha=alpha)
                    runs.append(record)
                    testmeta=meta.loc[masks['test']].reset_index(drop=True);actual=y[masks['test']]
                    for i,row in testmeta.iterrows():
                        for j,horizon in enumerate(HORIZONS):
                            predictions.append({'ticker':ticker,'fold':fold['fold'],'variant':variant,'seed':seed,
                              'date':row.date,'horizon':horizon,'actual':float(actual[i,j]),
                              'prediction':float(pred[i,j]),'ridge_prediction':float(ridge[i,j]),'train_mean_prediction':float(ym[j])})
                    print(f'[{stock_number}/{len(tickers)}] {ticker} fold={fold["fold"]} {variant} seed={seed} best_epoch={record["best_epoch"]}',flush=True)
            # Paired variants must start from identical weights and parameter counts.
            for seed in args.seeds:
                paired=[r for r in runs if r['ticker']==ticker and r['fold']==fold['fold'] and r['seed']==seed]
                if len({r['initial_weights_sha256'] for r in paired})!=1 or len({r['parameters'] for r in paired})!=1:
                    raise ValueError('Ablation architecture/initialization mismatch')
            pd.DataFrame(predictions).to_csv(root/'predictions.csv',index=False)
            (root/'runs.json').write_text(json.dumps(runs,indent=2,allow_nan=False),encoding='utf-8')
            (root/'fold_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    output=pd.DataFrame(predictions)
    summary,comparison,macro,by_fold=summarize(output)
    summary.to_csv(root/'metrics_by_seed.csv',index=False)
    comparison.to_csv(root/'comparison.csv',index=False)
    macro.to_csv(root/'macro_summary.csv',index=False)
    by_fold.to_csv(root/'metrics_by_fold.csv',index=False)
    config=vars(args).copy();config.update(device=str(device),tickers=tickers,
        feature_policy='Per-fold vocabulary and scaler fitted exclusively on training inputs; inactive groups zeroed after scaling.',
        architecture='Identical full-width Conv1D(16)-Conv1D(16)-embedding(32)-compact decoder for paired groups.',
        input_energy='Enabled standardized channels multiplied by sqrt(total channels / enabled channels) before encoder; reconstruction targets unamplified.',
        objective='Standardized next-return MSE + 0.2 masked reconstruction MSE on enabled features only.',
        unique_test_dates_per_stock=int(output.groupby('ticker').date.nunique().min()),
        fitted_models=len(runs),forecast_units='log return',
        limitations=['Overlapping horizon labels and correlated folds; counts are not independent samples.',
                     'Two seeds and short historical period cannot establish statistical significance.',
                     'Prior exploratory holdout was viewed in this session; these historical data are not a fresh untouched confirmation set.',
                     'No transaction execution, costs, slippage, or profit backtest.',
                     'Vendor price adjustment, revisions and universe survivorship are unverified.'])
    (root/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    write_report(root,comparison,macro,config)
    print(macro.to_string(index=False),flush=True)
    print('Completed fits:',len(runs),flush=True)


def write_report(root,comparison,macro,config):
    wins=int((comparison.combined<comparison.ohlcv).sum());n=len(comparison)
    stable=int((comparison.folds_combined_beats_ohlcv==config['folds']).sum())
    combined_ratio=float(macro.loc[macro.variant=='combined','mean_relative_mae'].iloc[0])
    lines=['# Hasil ablation dan walk-forward terkontrol','',
      f'Ringkasan: penambahan broker + Retail menurunkan error dibanding OHLCV saja pada {wins}/{n} pasangan saham-horizon; hanya {stable}/{n} yang membaik pada seluruh fold.',
      f'Error relatif rata-rata model gabungan terhadap baseline rata-rata train adalah {combined_ratio:.3f}. Angka di atas 1 berarti kalah dari baseline.',
      'Hasil ini belum menunjukkan keunggulan yang konsisten. Seluruh perbandingan dirata-rata atas dua seed; bukan bukti profit transaksi.',
      '',
      f"{len(config['tickers'])} independent stocks; {config['folds']} expanding folds; seeds {config['seeds']}; {config['fitted_models']} fresh CNN fits.",
      f"{config['unique_test_dates_per_stock']} nonduplicated query dates per stock. Forecast horizons: 5, 10, 20 sessions.",
      'Same eligible windows, labels, full-width architecture, initialization, minibatch order and hyperparameters for all three variants.',
      'OHLCV-only: price/volume-derived features, including returns 20/40. Broker+Retail-only: all broker-derived features and non-OHLC Summary features. Combined: both.',
      'Disabled feature channels are zeroed after train-only scaling; reconstruction loss averages enabled channels only.',
      'Enabled inputs are multiplied by sqrt(total/enabled channels) to avoid attenuating OHLCV solely because of zeroed channels; reconstruction targets retain their original standardized scale.',
      'Per-fold broker vocabulary uses training-input dates only. Labels must end strictly before validation/test boundaries. Checkpoints/alpha chosen only on validation.',
      '',f'Combined has lower MAE than OHLCV-only in {wins}/{n} stock-horizon pairs, averaged over seeds.',
      f'Combined beats OHLCV-only in every fold in {stable}/{n} stock-horizon pairs (fold metrics averaged over seeds).',
      '','## Macro comparison','',
      'Relative MAE = model MAE / train-mean baseline MAE for the same stock, seed and horizon, then equally averaged. Lower is better; 1 equals baseline.',
      '','| Variant | Mean relative MAE | Mean direction accuracy | Mean balanced accuracy |','|---|---:|---:|---:|']
    for r in macro.to_dict('records'):
        lines.append(f"| {r['variant']} | {r['mean_relative_mae']:.3f} | {100*r['mean_direction_accuracy']:.1f}% | {100*r['mean_balanced_accuracy']:.1f}% |")
    lines.extend(['','## Per-stock comparison','',
                  '| Stock | Horizon | OHLCV MAE | Broker+Retail MAE | Combined MAE | Combined improvement vs OHLCV | Folds improved |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for r in comparison.to_dict('records'):
        lines.append(f"| {r['ticker']} | {r['horizon']} | {r['ohlcv']:.4f} | {r['broker_retail']:.4f} | {r['combined']:.4f} | {100*r['combined_improvement_vs_ohlcv']:+.1f}% | {r['folds_combined_beats_ohlcv']}/{config['folds']} |")
    lines.extend(['','## Limits','']+config['limitations']+['',
       'This is a forecasting evaluation, not evidence of profitable trading. Results from this already-explored historical period need future unseen data for confirmation.',
       'predictions.csv preserves paired out-of-sample forecasts; fold_audit.json preserves boundaries, feature schema, vocabulary and sample identities. runs.json preserves validation histories and weight hashes.',
       'Models from earlier production-style runs are unchanged. Research weights are saved only when --save-models is specified.'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='runs/expanded-v2')
    parser.add_argument('--out',default='runs/research-v1')
    parser.add_argument('--summary',default=r'data/Daily/Summary')
    parser.add_argument('--brokers',default=r'data/Daily/All')
    parser.add_argument('--tickers',nargs='+')
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--test-size',type=int,default=14)
    parser.add_argument('--val-size',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,123])
    parser.add_argument('--save-models',action='store_true')
    args=parser.parse_args()
    if min(args.folds,args.test_size,args.val_size,args.epochs)<1:parser.error('Positive experiment sizes required')
    if len(set(args.seeds))!=len(args.seeds):parser.error('Use unique seeds')
    if args.tickers:
        args.tickers=[t.upper() for t in args.tickers]
        if len(set(args.tickers))!=len(args.tickers) or any(len(t)!=4 or not t.isalpha() for t in args.tickers):parser.error('Use unique four-letter tickers')
    run(args)
