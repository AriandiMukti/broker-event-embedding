"""Train-only tokenization of ADRO broker actions and OHLCV context."""
import numpy as np
import pandas as pd

STATES=['NET_ZERO','BUY_NORMAL','BUY_LARGE','SELL_NORMAL','SELL_LARGE']
CONTEXT_FEATURES=['return_1','return_20','return_40','volume_relative20','range','body','retail_share','buyer_fraction']


def context_values(loaded):
    df,good,cache,extras,snapshots,quality=loaded
    amounts=pd.Series([sum(abs(v) for v in s['values'].values()) if s is not None else np.nan for s in snapshots],index=df.index)
    lot_totals=pd.Series([sum(abs(v) for v in s['lots'].values()) if s is not None else np.nan for s in snapshots],index=df.index)
    fractions=pd.Series([sum(v>0 for v in s['values'].values())/max(1,sum(v!=0 for v in s['values'].values()))
                         if s is not None else np.nan for s in snapshots],index=df.index)
    retail=pd.to_numeric(df['Retail'],errors='coerce') if 'Retail' in df else pd.Series(np.nan,index=df.index)
    values=pd.DataFrame({'return_1':df.Close/df.Close.shift()-1,'return_20':df.Close/df.Close.shift(20)-1,
        'return_40':df.Close/df.Close.shift(40)-1,
        'volume_relative20':(df.Volume+1)/(df.Volume.rolling(20,min_periods=5).median()+1),
        'range':df.High/df.Low-1,'body':df.Close/df.Open-1,
        'retail_share':retail/lot_totals.where(lot_totals>0),'buyer_fraction':fractions})
    return values


def fit_tokenizer(loaded,train_stop,top_brokers=12,min_active_days=20):
    df,good,cache,extras,snapshots,quality=loaded
    scores={};active={}
    for snapshot in snapshots[:train_stop]:
        if snapshot is None:continue
        for code,value in snapshot['values'].items():
            if value:
                scores[code]=scores.get(code,0)+abs(value)
                active[code]=active.get(code,0)+1
    brokers=sorted((c for c in scores if active[c]>=min_active_days),key=lambda c:(-scores[c],c))[:top_brokers]
    if len(brokers)<2:raise ValueError('Insufficient brokers with training activity')
    large={}
    for code in brokers:
        shares=[]
        for snapshot in snapshots[:train_stop]:
            if snapshot is None:continue
            denominator=sum(abs(v) for v in snapshot['values'].values())
            value=abs(snapshot['values'].get(code,0))
            if value>0 and denominator>0:shares.append(value/denominator)
        large[code]=float(np.quantile(shares,2/3))
    continuous=context_values(loaded)
    cuts={}
    for name in CONTEXT_FEATURES:
        valid=continuous[name].iloc[:train_stop].replace([np.inf,-np.inf],np.nan).dropna()
        cuts[name]=np.quantile(valid,[1/3,2/3]).tolist() if len(valid) else [0.,0.]
    names=['PAD']+[f'{code}_{state}' for code in brokers for state in STATES]
    names += [f'{name.upper()}_{level}' for name in CONTEXT_FEATURES for level in ['LOW','MID','HIGH','MISSING']]
    return {'ticker':'ADRO','brokers':brokers,'large_share_threshold':large,'context_cutoffs':cuts,
            'tokens':names,'train_stop':int(train_stop),'train_cutoff':str(df.index[train_stop-1].date()),
            'broker_activity_days':{c:active[c] for c in brokers},'large_definition':'absolute net value / total absolute net value > training active-share 2/3 quantile per broker',
            'context_definition':'LOW/MID/HIGH are training quantile bins, not absolute up/down claims; MISSING is explicit.'}


def transform_tokens(loaded,tokenizer):
    df,good,cache,extras,snapshots,quality=loaded
    continuous=context_values(loaded);lookup={name:i for i,name in enumerate(tokenizer['tokens'])}
    brokers=tokenizer['brokers'];daily=[];labels=[];valid=[]
    for i,snapshot in enumerate(snapshots):
        states=[];tokens=[]
        if snapshot is None:
            daily.append([0]*(len(brokers)+len(CONTEXT_FEATURES)));labels.append([-1]*len(brokers));valid.append(False);continue
        denominator=sum(abs(v) for v in snapshot['values'].values())
        for code in brokers:
            value=snapshot['values'].get(code,0)
            large=abs(value)/max(denominator,1)>tokenizer['large_share_threshold'][code]
            state=0 if value==0 else ((2 if large else 1) if value>0 else (4 if large else 3))
            states.append(state);tokens.append(lookup[f'{code}_{STATES[state]}'])
        for name in CONTEXT_FEATURES:
            value=float(continuous.iloc[i][name])
            level='MISSING' if not np.isfinite(value) else ['LOW','MID','HIGH'][int(np.searchsorted(tokenizer['context_cutoffs'][name],value,side='right'))]
            tokens.append(lookup[f'{name.upper()}_{level}'])
        daily.append(tokens);labels.append(states);valid.append(bool(good.iloc[i]))
    return np.array(daily,dtype='int64'),np.array(labels,dtype='int64'),np.array(valid,dtype=bool)


def make_samples(loaded,tokenizer,context_days=3,val_stop=None):
    daily,states,valid=transform_tokens(loaded,tokenizer)
    n=len(daily);train_stop=tokenizer['train_stop'];val_stop=int(n*.8) if val_stop is None else val_stop
    x=[];y=[];previous=[];target_indices=[];splits=[]
    for target in range(40+context_days,n):
        start=target-context_days
        if not valid[start:target].all() or np.any(states[target]<0):continue
        x.append(daily[start:target]);y.append(states[target]);previous.append(states[target-1]);target_indices.append(target)
        splits.append('train' if target<train_stop else ('val' if target<val_stop else 'test'))
    masks={s:np.array(splits)==s for s in ['train','val','test']}
    if any(not m.any() for m in masks.values()):raise ValueError('Empty token-model split')
    return np.array(x),np.array(y),np.array(previous),np.array(target_indices),masks,daily,states,valid
