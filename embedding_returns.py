"""Purged price-return transfer test; independent BRPT, MDKA and ADRO models.

Frozen masked-broker encoders are trained strictly inside each outer training
prefix. Existing later-dated relation checkpoints are deliberately NOT reused.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from research import load_stock
from broker_relation_test import align_loaded_calendar,relation_tokens,mask_inputs,fast_fit,SEEDS
from adro_token_embedding import EventEmbedding
from relation_io import source_fingerprint

HORIZONS=[1,5,10]
ALPHAS=[.1,1.,10.,100.,1000.,10000.]
LAYOUTS=[(1,),(1,5),(1,5,20)]
METHODS=['ohlcv','broker_aggregates','broker_direct','broker_onehot','learned_embedding','random_embedding']
CONTROLS=['ohlcv','broker_aggregates','broker_direct','broker_onehot','random_embedding','zero','training_median']
PRICE_NAMES=['return_1','return_5','return_20','return_40','open_gap','range_log','body_log','volume_log1p','volume_relative20','volatility20']
AGG_NAMES=['retail_signed_log1p','buyer_fraction','buyer_seller_log_ratio','absolute_net_value_log1p']


def make_return_folds(n):
    edges=np.linspace(int(n*.6),n-max(HORIZONS),4,dtype=int)
    result=[]
    for k,(start,stop) in enumerate(zip(edges[:-1],edges[1:]),1):
        val_start=int(start-10-21)
        train_stop=val_start-10  # train t+10 < val_start
        val_stop=int(start-10)  # val t+10 < test_start
        if train_stop-40<30:raise ValueError('Need >=30 train dates after 40-session warmup and purging')
        result.append(dict(fold=k,train_stop=train_stop,val_start=val_start,val_stop=val_stop,
                           test_start=int(start),test_stop=int(stop)))
    return result


def samples(loaded,fold):
    df,good,_,_,snapshots,_=loaded
    good=good.to_numpy(dtype=bool); indices=[];splits=[]
    for t in range(40,fold['test_stop']):
        if not good[t-40:t+11].all() or any(s is None for s in snapshots[t-19:t+1]):continue
        if t<fold['train_stop']:split='train'
        elif fold['val_start']<=t<fold['val_stop']:split='val'
        elif fold['test_start']<=t<fold['test_stop']:split='test'
        else:continue
        indices.append(t);splits.append(split)
    ids=np.array(indices,dtype=int);split=np.array(splits)
    if any(not (split==s).any() for s in ['train','val','test']):raise ValueError('Empty return split')
    y=np.column_stack([np.log(df.Close.iloc[ids+h].to_numpy()/df.Close.iloc[ids].to_numpy()) for h in HORIZONS])
    return ids,split,y


def price_features(loaded):
    df=loaded[0].copy();df.loc[~loaded[1],['Open','High','Low','Close','Volume']]=np.nan
    r=np.log(df.Close/df.Close.shift(1))
    out=pd.DataFrame({
        'return_1':r,'return_5':np.log(df.Close/df.Close.shift(5)),
        'return_20':np.log(df.Close/df.Close.shift(20)),'return_40':np.log(df.Close/df.Close.shift(40)),
        'open_gap':np.log(df.Open/df.Close.shift(1)),'range_log':np.log(df.High/df.Low),
        'body_log':np.log(df.Close/df.Open),'volume_log1p':np.log1p(df.Volume),
        'volume_relative20':np.log((df.Volume+1)/(df.Volume.rolling(20,min_periods=20).median()+1)),
        'volatility20':r.rolling(20,min_periods=20).std()})
    return out[PRICE_NAMES].to_numpy()


def broker_arrays(loaded,tokenizer,states):
    codes=tokenizer['brokers'];n=len(states);b=len(codes)
    onehot=np.eye(5)[states.clip(min=0)].reshape(n,b*5)
    onehot[(states<0).any(1)]=np.nan
    shares=np.full((n,b),np.nan);agg=np.full((n,4),np.nan)
    retail=pd.to_numeric(loaded[0]['Retail'],errors='coerce').to_numpy() if 'Retail' in loaded[0] else np.full(n,np.nan)
    agg[:,0]=np.sign(retail)*np.log1p(np.abs(retail))
    for i,snapshot in enumerate(loaded[4]):
        if snapshot is None:continue
        values=snapshot['values'];denominator=max(sum(abs(v) for v in values.values()),1.)
        shares[i]=[values.get(c,0)/denominator for c in codes]
        buys=sum(v>0 for v in values.values());sells=sum(v<0 for v in values.values())
        agg[i,1:]=[buys/max(buys+sells,1),np.log1p(buys)-np.log1p(sells),np.log1p(sum(abs(v) for v in values.values()))]
    state_names=[f'{code}_{state}' for code in codes for state in ['NET_ZERO','BUY_NORMAL','BUY_LARGE','SELL_NORMAL','SELL_LARGE']]
    return dict(aggregate=agg,onehot=onehot,direct=np.column_stack([shares,onehot])),dict(
        aggregate=AGG_NAMES,onehot=state_names,direct=[f'net_share_{c}' for c in codes]+state_names)


def pool_encoder(model,daily,states,brokers):
    model.eval()
    with torch.no_grad():
        vectors=model.embedding(torch.as_tensor(daily[:,:brokers],dtype=torch.long)).mean(1)
        latent=model.transform(vectors).numpy()
    latent[(states<0).any(1)]=np.nan
    return latent


def train_encoder(loaded,fold,ticker,seed,epochs=300,patience=40):
    outer_stop=fold['train_stop'];inner_stop=int(outer_stop*.8)
    tokenizer,daily,states,valid=relation_tokens(loaded,inner_stop,12,ticker,True)
    ids=np.flatnonzero(valid[:outer_stop]&(states[:outer_stop]>=0).all(1))
    split=np.where(ids<inner_stop,'train','val')
    if min((split=='train').sum(),(split=='val').sum())<10:raise ValueError('Insufficient inner embedding dates')
    x=mask_inputs(daily[ids,:len(tokenizer['brokers'])],len(tokenizer['brokers']),'brokers_only')
    model,record=fast_fit(x,states[ids],split,len(tokenizer['tokens']),8,seed,epochs,patience)
    torch.manual_seed(seed)
    random_model=EventEmbedding(len(tokenizer['tokens']),len(tokenizer['brokers']),8,1)
    record.update(inner_train_stop=inner_stop,outer_train_stop=outer_stop,
                  last_embedding_validation_date=str(loaded[0].index[ids[-1]].date()),
                  last_tokenizer_date=str(loaded[0].index[inner_stop-1].date()),
                  train_dates=int((split=='train').sum()),val_dates=int((split=='val').sum()))
    return tokenizer,daily,states,model,random_model,record


def rolling(values,window):
    if window==1:return values.copy()
    return pd.DataFrame(values).rolling(window,min_periods=window).mean().to_numpy()


def design(price,aggregate,branch,branch_names,layout):
    arrays=[price];names=list(PRICE_NAMES)
    for window in layout:
        arrays.append(rolling(aggregate,window));names += [f'broker_{window}d_{c}' for c in AGG_NAMES]
        if branch is not None:
            arrays.append(rolling(branch,window));names += [f'broker_{window}d_{c}' for c in branch_names]
    return np.column_stack(arrays),names


def fit_preprocessor(x):
    x=np.where(np.isfinite(x),x,np.nan)
    med=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in x.T])
    filled=np.where(np.isfinite(x),x,med)
    augmented=np.column_stack([filled,~np.isfinite(x)]).astype(float)
    center=augmented.mean(0);scale=augmented.std(0);scale[scale<1e-10]=1.
    return dict(imputation=med,center=center,scale=scale)


def transform(x,state):
    missing=~np.isfinite(x);filled=np.where(missing,state['imputation'],x)
    return (np.column_stack([filled,missing]).astype(float)-state['center'])/state['scale']


def select_head(options,ids,split,target):
    """Only training/validation target values are ever read for selection."""
    tr=split=='train';va=split=='val';best=None
    y=target[tr];intercept=float(y.mean())
    for layout,(matrix,names) in options.items():
        state=fit_preprocessor(matrix[ids[tr]])
        tx=transform(matrix[ids[tr]],state);vx=transform(matrix[ids[va]],state)
        u,s,vt=np.linalg.svd(tx,full_matrices=False)
        uy=u.T@(y-intercept)
        for alpha in ALPHAS:
            coef=vt.T@((s/(s*s+alpha))*uy)
            error=float(np.abs(vx@coef+intercept-target[va]).mean())
            # Ties prefer the earlier, simpler layout/alpha in the fixed grid.
            if best is None or error<best['validation_mae']-1e-12:
                best=dict(**state,coefficient=coef,intercept=intercept,alpha=alpha,layout=list(layout),
                          validation_mae=error,features=names)
    return best


def head_predict(matrix,head):
    return transform(matrix,head)@head['coefficient']+head['intercept']


def save_head(path,head):
    np.savez_compressed(path,**{k:np.array(v) for k,v in head.items()})


def run_stock(ticker,args):
    start=time.monotonic();folder=Path(args.out)/ticker;folder.mkdir(parents=True,exist_ok=True)
    loaded=align_loaded_calendar(load_stock(ticker,args.summary,args.brokers),args.brokers)
    source=loaded[0];price=price_features(loaded)
    np.savez_compressed(folder/'source_snapshot.npz',dates=source.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                        close=source.Close.to_numpy(),open=source.Open.to_numpy(),good=loaded[1].to_numpy())
    predictions=[];selections=[];embedding_records=[];audit=[]
    for fold in make_return_folds(len(source)):
        fdir=folder/f"fold-{fold['fold']}";fdir.mkdir(exist_ok=True)
        ids,split,y=samples(loaded,fold)
        np.savez_compressed(fdir/'targets.npz',indices=ids,splits=split,targets=y)
        test=split=='test'
        for i,t in enumerate(ids):
            audit.append(dict(fold=fold['fold'],split=split[i],feature_index=int(t),feature_date=str(source.index[t].date()),
                              last_target_index=int(t+10),last_target_date=str(source.index[t+10].date())))
        options_by_method={}
        for seed in SEEDS:
            tokenizer,daily,states,encoder,random_encoder,record=train_encoder(loaded,fold,ticker,seed,args.epochs,args.patience)
            if seed==SEEDS[0]:
                (fdir/'tokenizer.json').write_text(json.dumps(tokenizer,indent=2),encoding='utf-8')
                np.savez_compressed(fdir/'token_snapshot.npz',daily=daily,states=states)
                broker,names=broker_arrays(loaded,tokenizer,states)
                np.savez_compressed(fdir/'feature_snapshot.npz',price=price,**broker)
                (fdir/'feature_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
                options_by_method['ohlcv',-1]={(): (price,PRICE_NAMES)}
                for method,branch in [('broker_aggregates',None),('broker_direct','direct'),('broker_onehot','onehot')]:
                    options_by_method[method,-1]={layout:design(price,broker['aggregate'],None if branch is None else broker[branch],
                        [] if branch is None else names[branch],layout) for layout in LAYOUTS}
            torch.save(dict(state_dict=encoder.state_dict(),seed=seed,tokenizer=tokenizer,dimension=8,context_days=1,
                            record=record),fdir/f'encoder-seed{seed}.pt')
            record.update(fold=fold['fold'],seed=seed);embedding_records.append(record)
            for method,model in [('learned_embedding',encoder),('random_embedding',random_encoder)]:
                latent=pool_encoder(model,daily,states,len(tokenizer['brokers']))
                options_by_method[method,seed]={layout:design(price,broker['aggregate'],latent,[f'latent_{i+1}' for i in range(8)],layout)
                                               for layout in LAYOUTS}
            print(ticker,'fold',fold['fold'],'encoder',seed,'epoch',record['best_epoch'],flush=True)
        for (method,seed),options in options_by_method.items():
            for j,horizon in enumerate(HORIZONS):
                head=select_head(options,ids,split,y[:,j]);layout=tuple(head['layout']);matrix=options[layout][0]
                pred=head_predict(matrix[ids[test]],head)
                save_head(fdir/f'{method}-h{horizon}-seed{seed}.npz',head)
                selections.append(dict(fold=fold['fold'],method=method,seed=seed,horizon=horizon,
                    alpha=head['alpha'],layout=head['layout'],features=len(head['features']),validation_mae=head['validation_mae']))
                for t,actual,p in zip(ids[test],y[test,j],pred):
                    predictions.append(dict(ticker=ticker,fold=fold['fold'],method=method,seed=seed,horizon=horizon,
                        date=str(source.index[t].date()),feature_index=int(t),target_date=str(source.index[t+horizon].date()),
                        actual=float(actual),prediction=float(p)))
        for method in ['zero','training_median']:
            for j,horizon in enumerate(HORIZONS):
                p=0. if method=='zero' else float(np.median(y[split=='train',j]))
                for t,actual in zip(ids[test],y[test,j]):
                    predictions.append(dict(ticker=ticker,fold=fold['fold'],method=method,seed=-1,horizon=horizon,
                        date=str(source.index[t].date()),feature_index=int(t),target_date=str(source.index[t+horizon].date()),
                        actual=float(actual),prediction=p))
        print(ticker,'fold',fold['fold'],'price heads complete',dict(zip(*np.unique(split,return_counts=True))),flush=True)
    table=pd.DataFrame(predictions)
    table['absolute_error']=np.abs(table.actual-table.prediction)
    table['squared_error']=(table.actual-table.prediction)**2
    table['direction_correct']=(np.sign(table.actual)==np.sign(table.prediction)).astype(float)
    table.to_csv(folder/'test_predictions.csv',index=False)
    daily=table.groupby(['ticker','fold','horizon','date','method'],as_index=False)[['absolute_error','squared_error','direction_correct']].mean()
    daily.to_csv(folder/'daily_test_scores.csv',index=False)
    summary=daily.groupby(['ticker','horizon','method'],as_index=False)[['absolute_error','squared_error','direction_correct']].mean()
    summary['rmse']=np.sqrt(summary.squared_error)
    summary.to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(selections).to_csv(folder/'selected_heads.csv',index=False)
    pd.DataFrame(audit).to_csv(folder/'sample_audit.csv',index=False)
    (folder/'embedding_training.json').write_text(json.dumps(embedding_records,indent=2),encoding='utf-8')
    config=dict(ticker=ticker,source_days=len(source),source_start=str(source.index[0].date()),source_end=str(source.index[-1].date()),
        source_fingerprint=source_fingerprint(ticker,args.summary,args.brokers),folds=make_return_folds(len(source)),
        seeds=SEEDS,horizons=HORIZONS,alphas=ALPHAS,layouts=LAYOUTS,epochs=args.epochs,patience=args.patience,
        summary_path=args.summary,brokers_path=args.brokers,source_quality=loaded[-1],
        representation='broker-only masked same-day encoder, 8D hidden mean transformed by frozen encoder; retrained inside purged price train prefix',
        controls=CONTROLS,price_features=PRICE_NAMES,shared_broker_aggregates=AGG_NAMES,
        target='raw unadjusted close-to-close log return; no dividend/total-return adjustment',
        selection='outer validation MAE selects alpha and broker lookbacks separately per stock/fold/horizon/method/seed; no refit on validation',
        test_use='historical exploratory reuse; BRPT/MDKA selected from earlier historical state tests, ADRO comparator',
        timing='features known after session close, not an executable trade at that close',seconds=time.monotonic()-start)
    (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(ticker,'finished',round(config['seconds'],1),'seconds',flush=True)


def main(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    (root/'experiment_plan.json').write_text(json.dumps(dict(vars(args),horizons=HORIZONS,seeds=SEEDS,alphas=ALPHAS,
        broker_lookback_candidates=LAYOUTS,primary_metric='mean absolute log-return error; losses averaged across seeds before dates',
        comparison_family=f'all {len(args.tickers)} stocks x 3 horizons x 7 controls = {len(args.tickers)*21}; synchronized chronological test-date bootstrap',
        bootstrap_blocks=[10,15,20],bootstrap_repeats=20000,
        pass_rule='all 7 simultaneous lower gain bounds positive at all 3 block lengths, >=2 positive folds/control; experimental, not confirmation',
        execution_cost_scenario='only evaluate an illustrative trade simulation if the predeclared return criterion passes; fees/slippage must be explicit'),indent=2),encoding='utf-8')
    with threadpool_limits(limits=1):
        for ticker in args.tickers:run_stock(ticker,args)
    from return_results import analyze
    analyze(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers',nargs='+',default=['BRPT','MDKA','ADRO'])
    parser.add_argument('--out',default='runs/embedding-returns-v1')
    parser.add_argument('--summary',default=r'data/Daily/Summary')
    parser.add_argument('--brokers',default=r'data/Daily/All')
    parser.add_argument('--epochs',type=int,default=300);parser.add_argument('--patience',type=int,default=40)
    args=parser.parse_args()
    if min(args.epochs,args.patience)<1 or len(set(args.tickers))!=len(args.tickers):parser.error('Positive epochs and distinct tickers required')
    if any(len(t)!=4 or not t.isalpha() or t!=t.upper() for t in args.tickers):parser.error('Use uppercase stock codes')
    main(args)
