"""Compact candidate pool; final feature subsets are chosen separately per stock."""
import numpy as np
import pandas as pd
from main import BASE
from features import expanded_features

PRICE=['return','open_gap','range','body','log_volume_change','volume_relative20','volatility20','price_return_20','price_return_40']
GROUPS={
 'ohlcv':PRICE,
 'retail':['summary_retail_raw','summary_retail_signed_log1p','summary_retail_share_abs_net_lot','summary_retail_cumulative_share_5','summary_retail_cumulative_share_20'],
 'buyer_seller':['broker_buy_count','broker_sell_count','buyer_seller_ratio','broker_buyer_fraction','buyer_seller_ratio_delta_5','buyer_seller_ratio_delta_20'],
 'concentration':['buy_hhi','sell_hhi','lot_buy_hhi','lot_sell_hhi','summary_net_top3_all_lot_share','summary_net_top5_all_lot_share','summary_net_top5_all_lot_cumulative_20'],
 'intensity':['broker_abs_net_value_log1p','broker_abs_net_lot_log1p','broker_abs_net_value_relative20','broker_abs_net_lot_relative20'],
 'broker_price':['broker_price_spread'],
 'persistence':['persistent_buyer_share_5','persistent_seller_share_5','persistent_buyer_share_20','persistent_seller_share_20','flow_turnover_1','flow_turnover_mean20']}
NUMERIC=[name for group in GROUPS.values() for name in group]


def align_loaded(loaded,calendar):
    df,good,cache,extras,snapshots,quality=loaded
    lookup={day:i for i,day in enumerate(df.index)}
    aligned_cache=[];aligned_extra=[];aligned_snapshots=[];issues=list(quality)
    for day in calendar:
        if day in lookup:
            i=lookup[day];aligned_cache.append(cache[i]);aligned_extra.append(extras[i]);aligned_snapshots.append(snapshots[i])
        else:
            aligned_cache.append({});aligned_extra.append([1,0,0,0]);aligned_snapshots.append(None)
            issues.append({'date':str(day.date()),'issue':'missing Summary row; no forward filling'})
    for day in df.index[~good]:
        issues.append({'date':str(day.date()),'issue':'invalid OHLCV bounds','ohlcv':df.loc[day,['Open','High','Low','Close','Volume']].to_dict()})
    aligned_df=df.reindex(calendar).copy();aligned_good=good.reindex(calendar,fill_value=False)
    aligned_df.loc[~aligned_good,['Open','High','Low','Close','Volume']]=np.nan
    return aligned_df,aligned_good,aligned_cache,np.array(aligned_extra),aligned_snapshots,issues


def compact_matrix(loaded):
    df,good,cache,extras,snapshots,quality=loaded
    ret=np.log(df.Close).diff()
    basic=np.column_stack([ret,np.log(df.Open/df.Close.shift()),np.log(df.High/df.Low),np.log(df.Close/df.Open),
        np.log1p(df.Volume).diff(),np.log((df.Volume+1)/(df.Volume.rolling(20,min_periods=5).median()+1)),
        ret.rolling(20,min_periods=5).std(),extras])
    values={name:basic[:,i] for i,name in enumerate(BASE)}
    expanded,names,_=expanded_features(df,[],snapshots)
    values.update({name:expanded[:,i] for i,name in enumerate(names)})
    for h in (5,20):
        for side,label in [(1,'buyer'),(-1,'seller')]:
            rows=[]
            for i,current in enumerate(snapshots):
                history=snapshots[max(0,i-h+1):i+1]
                if len(history)<h or any(s is None for s in history):rows.append(np.nan);continue
                active={c:v for c,v in current['values'].items() if np.sign(v)==side}
                total=sum(abs(v) for v in active.values())
                persistent=0.
                for code,value in active.items():
                    fraction=sum(np.sign(s['values'].get(code,0))==side for s in history)/h
                    if fraction>=.6:persistent+=abs(value)
                rows.append(persistent/total if total else np.nan)
            values[f'persistent_{label}_share_{h}']=np.array(rows)
    turns=[]
    for i in range(len(df)):
        if i==0 or snapshots[i] is None or snapshots[i-1] is None:turns.append(np.nan);continue
        a,b=cache[i],cache[i-1]
        turns.append(.5*sum(abs(a.get(c,0)-b.get(c,0)) for c in set(a)|set(b)))
    values['flow_turnover_1']=np.array(turns)
    values['flow_turnover_mean20']=pd.Series(turns).rolling(20,min_periods=20).mean().to_numpy()
    columns=[];column_names=[];pairs={}
    for name in NUMERIC:
        raw=np.asarray(values[name],dtype=float)
        if name+'_missing' in values:missing=np.asarray(values[name+'_missing'],dtype=float)>0
        else:missing=~np.isfinite(raw)
        pairs[name]=[len(columns),len(columns)+1]
        columns.extend([np.where(missing,0,raw),missing.astype(float)])
        column_names.extend([name,name+'_missing'])
    matrix=np.column_stack(columns).astype('float32')
    if not np.isfinite(matrix).all():raise ValueError('Non-finite compact input')
    return matrix,column_names,pairs


def feature_mask(selected,pairs,width):
    result=np.zeros(width,dtype=bool)
    for name in selected:result[pairs[name]]=True
    return result


def dataset(loaded,matrix,fold,latest=False,window=20):
    df,good,*_=loaded;x=[];y=[];rows=[];splits=[]
    for i in range(40+window-1,len(df)):
        if not good.iloc[i-window+1:i+1].all():continue
        if latest and i==len(df)-1:
            split='latest';label=[np.nan]*3;end=None
        else:
            if i+20>=len(df) or not all(bool(good.iloc[i+h]) for h in (5,10,20)):continue
            if i+20<fold['val_start']:split='train'
            elif i>=fold['val_start'] and i+20<fold['test_start']:split='val'
            elif not latest and fold['test_start']<=i<fold['test_stop']:split='test'
            else:continue
            label=[np.log(df.Close.iloc[i+h]/df.Close.iloc[i]) for h in (5,10,20)]
            end=str(df.index[i+20].date())
        x.append(matrix[i-window+1:i+1]);y.append(label);splits.append(split)
        rows.append({'date':str(df.index[i].date()),'label_end':end,'split':split,'calendar_index':i})
    keys=['train','val','latest' if latest else 'test']
    masks={s:np.array(splits)==s for s in keys}
    if any(not masks[s].any() for s in ['train','val']):raise ValueError('Empty training or validation split')
    if not latest and not masks['test'].any():raise ValueError('Empty adaptive test split')
    return np.array(x,dtype='float32'),np.array(y,dtype='float32'),pd.DataFrame(rows),masks
