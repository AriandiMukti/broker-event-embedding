"""Freeze Yahoo daily context, conservative availability and train-ready features."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS={'ihsg':'^JKSE','usd_idr':'IDR=X','sector_finance':'IDXFINANCE.JK',
         'sector_energy':'IDXENERGY.JK','sector_basic':'IDXBASIC.JK'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf-8')


def download(out,start='2020-01-01',end='2026-11-17'):
    import yfinance as yf
    root=Path(out);root.mkdir(parents=True,exist_ok=False);audit=[]
    for key,symbol in SYMBOLS.items():
        item=dict(group=key,symbol=symbol,start_inclusive=start,end_exclusive=end,
                  retrieved_at=datetime.now(timezone.utc).isoformat(),yfinance=yf.__version__)
        print('DOWNLOAD',key,symbol,flush=True)
        try:
            ticker=yf.Ticker(symbol)
            frame=ticker.history(start=start,end=end,interval='1d',auto_adjust=False,back_adjust=False,
                                 repair=False,actions=False,keepna=True,timeout=20,raise_errors=True)
            metadata=ticker.get_history_metadata()
            dump(root/f'{key}-metadata.json',metadata)
            if frame.empty:raise ValueError('Empty Yahoo history')
            if not isinstance(frame.index,pd.DatetimeIndex):raise ValueError('No dated daily bars')
            normalized=frame.copy();normalized.insert(0,'source_timestamp',[x.isoformat() for x in frame.index])
            normalized.insert(0,'date',[str(x.date()) for x in frame.index]);normalized.reset_index(drop=True,inplace=True)
            path=root/f'{key}.csv';normalized.to_csv(path,index=False)
            prices=normalized[['Open','High','Low','Close']].apply(pd.to_numeric,errors='coerce')
            invalid=(~np.isfinite(prices).all(axis=1)|(prices<=0).any(axis=1)
                     |(prices.High<prices[['Open','Close','Low']].max(axis=1))
                     |(prices.Low>prices[['Open','Close','High']].min(axis=1)))
            dup=int(normalized.date.duplicated().sum());valid=int((~invalid).sum())
            instrument=metadata.get('instrumentType');exchange=metadata.get('exchangeName')
            accepted=(valid>=500 and dup==0 and
                      (instrument=='CURRENCY' if key=='usd_idr' else instrument=='INDEX'))
            item.update(status='accepted' if accepted else 'rejected',rows=len(frame),valid_rows=valid,
                invalid_rows=int(invalid.sum()),duplicate_dates=dup,first=normalized.date.min(),last=normalized.date.max(),
                instrument_type=instrument,exchange=exchange,name=metadata.get('longName',metadata.get('shortName')),
                sha256=sha(path),invalid_dates=normalized.loc[invalid,'date'].tolist())
        except Exception as error:
            item.update(status='failed',error=f'{type(error).__name__}: {error}')
        audit.append(item);dump(root/'download-audit.json',audit);print(item,flush=True)
    dump(root/'manifest.json',{p.name:sha(p) for p in root.iterdir() if p.is_file()})
    return audit


def series_features(path,calendar,stock_close=None):
    """Daily date D is usable only at D+1 calendar day 12:00 Asia/Jakarta.

    No interpolated observations, no backward fill, age limit7 calendar days.
    Both local and UTC FX midnight conventions finish before this assumed cutoff.
    This is a conservative research lag, not proof of historical vendor release.
    """
    source=pd.read_csv(path);dates=pd.to_datetime(source.date)
    if dates.duplicated().any():raise ValueError('Duplicate source dates')
    price=source[['Open','High','Low','Close']].apply(pd.to_numeric,errors='coerce')
    good=(np.isfinite(price).all(axis=1)&(price>0).all(axis=1)
          &(price.High>=price[['Open','Close','Low']].max(axis=1))
          &(price.Low<=price[['Open','Close','High']].min(axis=1)))
    close=pd.Series(np.where(good,price.Close,np.nan),index=pd.DatetimeIndex(dates))
    log=np.log(close);ret=log.diff();values=pd.DataFrame(index=close.index)
    values['return1']=ret;values['return5']=log.diff(5);values['return20']=log.diff(20)
    values['volatility20']=ret.rolling(20,min_periods=20).std()
    if stock_close is not None:
        stock_log=np.log(stock_close.reindex(close.index))
        values['relative_strength1']=stock_log.diff()-log.diff()
        values['relative_strength5']=stock_log.diff(5)-log.diff(5)
        values['relative_strength20']=stock_log.diff(20)-log.diff(20)
    values['source_date']=values.index
    values['available_at']=values.index+pd.Timedelta(days=1,hours=12)
    values=values.reset_index(drop=True).sort_values('available_at')
    query=pd.DataFrame(dict(query_date=calendar,available_at=calendar+pd.Timedelta(hours=23,minutes=59)))
    joined=pd.merge_asof(query,values,on='available_at',direction='backward',tolerance=pd.Timedelta(days=7))
    # merge_asof keeps the left join key, so retain an explicit source cutoff.
    joined['source_available_at']=joined.source_date+pd.Timedelta(days=1,hours=12)
    observed=joined.source_date.notna()
    assert (joined.loc[observed,'source_available_at']<=joined.loc[observed,'available_at']).all()
    columns=[];names=[]
    for name in [c for c in values if c not in ['source_date','available_at']]:
        array=joined[name].to_numpy(dtype=float);missing=~np.isfinite(array)
        columns.extend([np.where(missing,0,array),missing.astype(float)]);names.extend([name,name+'_missing'])
    ages=(joined.query_date-joined.source_date).dt.total_seconds()/86400
    columns.append(ages.fillna(8).clip(0,8).to_numpy());names.append('age_calendar_days')
    return np.column_stack(columns).astype('float32'),names,joined[['query_date','source_date','source_available_at']]


def augment(raw,market_root,stock_close):
    root=Path(market_root);audits=json.loads((root/'download-audit.json').read_text())
    result=dict(raw);result['x']=raw['x'].copy();result['names']=list(raw['names'])
    result['groups']={key:list(value) for key,value in raw['groups'].items()};receipts=[]
    accepted={row['group']:row for row in audits if row['status']=='accepted'}
    if not {'ihsg','usd_idr'}<=set(accepted):raise ValueError('Verified IHSG and USD/IDR histories required')
    for key,row in accepted.items():
        path=root/f'{key}.csv'
        if sha(path)!=row['sha256']:raise ValueError('Yahoo snapshot changed after audit')
        extra,names,joined=series_features(path,raw['dates'],stock_close if key=='ihsg' else None)
        group='sector' if key.startswith('sector_') else key
        start=result['x'].shape[1];result['groups'].setdefault(group,[]).extend(range(start,start+extra.shape[1]))
        result['x']=np.column_stack([result['x'],extra]);result['names'].extend([key+'_'+name for name in names])
        receipts.append(dict(key=key,source=row,coverage_dates=int(joined.source_date.notna().sum()),
            missing_cells=int(extra[:,[i for i,name in enumerate(names) if name.endswith('_missing')]].sum()),
            joins=joined))
    result['source_hash']=hashlib.sha256((raw['source_hash']+sha(root/'manifest.json')).encode()).hexdigest()
    return result,receipts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    parser.add_argument('--start',default='2020-01-01');parser.add_argument('--end',default='2026-11-17')
    args=parser.parse_args();download(args.out,args.start,args.end)
