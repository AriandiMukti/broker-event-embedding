"""Expanded causal stock features. No backward filling or future observations."""
import numpy as np
import pandas as pd


def signed_streak(series):
    result=[]; previous=0; run=0
    for value in series:
        sign=int(np.sign(value)) if np.isfinite(value) else 0
        run=run+1 if sign and sign==previous else (1 if sign else 0)
        result.append(sign*min(run,40)/40)
        previous=sign
    return np.array(result)


def expanded_features(df, vocabulary, snapshots):
    index=df.index; count=len(df); columns={}; missing_counts={}
    if len(snapshots)!=count:
        raise ValueError('Snapshot count does not match market calendar')
    def add(name, values, mask=True):
        a=np.asarray(values,dtype='float64')
        if a.shape!=(count,):
            raise ValueError(f'Invalid feature shape {name}: {a.shape}')
        missing=~np.isfinite(a)
        columns[name]=np.where(missing,0,a)
        missing_counts[name]=int(missing.sum())
        if mask:
            columns[name+'_missing']=missing.astype(float)
    def series(values):
        return pd.Series(values,index=index,dtype='float64')
    def safe_ratio(numerator, denominator):
        return numerator/denominator.where(denominator!=0)
    def source(name):
        if name not in df:
            return series(np.full(count,np.nan))
        return pd.to_numeric(df[name],errors='coerce')

    # Arithmetic percentage changes represented as fractions: 0.10 means +10%.
    for h in (20,40):
        add(f'price_return_{h}',df.Close/df.Close.shift(h)-1)
    lots={};values={};prices={};raw_lot={code:[] for code in vocabulary}
    raw_price={code:[] for code in vocabulary}
    buyers=[];sellers=[];amounts=[];lot_totals=[];price_reference=[];price_spread=[]
    price_coverage=[];lot_coverage=[];lot_buy_hhi=[];lot_sell_hhi=[]
    for snapshot in snapshots:
        if snapshot is None:
            buyers.append(np.nan);sellers.append(np.nan);amounts.append(np.nan);lot_totals.append(np.nan)
            price_reference.append(np.nan);price_spread.append(np.nan)
            price_coverage.append(np.nan);lot_coverage.append(np.nan);lot_buy_hhi.append(np.nan);lot_sell_hhi.append(np.nan)
            for code in vocabulary:
                raw_lot[code].append(np.nan);raw_price[code].append(np.nan)
            continue
        values=snapshot['values'];lots=snapshot['lots'];prices=snapshot['prices']
        active={code:value for code,value in values.items() if value!=0}
        buyers.append(sum(v>0 for v in active.values()));sellers.append(sum(v<0 for v in active.values()))
        amounts.append(sum(abs(v) for v in values.values()))
        lot_ok=all(np.isfinite(lots.get(code,np.nan)) for code in active)
        total=sum(abs(lots.get(code,0)) for code in active) if lot_ok else np.nan
        lot_totals.append(total)
        lot_coverage.append(sum(np.isfinite(lots.get(code,np.nan)) for code in active)/len(active) if active else 1)
        valid_prices={code:prices.get(code,np.nan) for code in active
                      if np.isfinite(prices.get(code,np.nan)) and prices.get(code,0)>0}
        price_coverage.append(len(valid_prices)/len(active) if active else 1)
        # Compare prices within the same raw-price source. Never divide by adjusted OHLC.
        weights={code:abs(values[code]) for code in valid_prices}
        weight_sum=sum(weights.values())
        reference=sum(valid_prices[c]*weights[c] for c in valid_prices)/weight_sum if weight_sum else np.nan
        price_reference.append(reference)
        def weighted_side(sign):
            codes=[c for c in valid_prices if np.sign(values[c])==sign]
            denominator=sum(weights[c] for c in codes)
            return sum(valid_prices[c]*weights[c] for c in codes)/denominator if denominator else np.nan
        bp,sp=weighted_side(1),weighted_side(-1)
        price_spread.append((bp-sp)/reference if reference>0 else np.nan)
        def hhi(sign):
            a=np.array([abs(lots.get(c,np.nan)) for c in active if np.sign(values[c])==sign])
            if not np.isfinite(a).all():
                return np.nan
            return float(np.sum((a/a.sum())**2)) if a.sum()>0 else 0.0
        lot_buy_hhi.append(hhi(1));lot_sell_hhi.append(hhi(-1))
        for code in vocabulary:
            # Absent broker is a genuine zero if the day's snapshot is available.
            raw_lot[code].append(lots.get(code,0.0))
            raw_price[code].append((prices[code]/reference-1)
                if code in valid_prices and reference>0 else (0.0 if code not in active else np.nan))
    buyers=series(buyers);sellers=series(sellers);total_brokers=buyers+sellers
    lot_totals=series(lot_totals);amounts=series(amounts)
    add('broker_buy_count',buyers);add('broker_sell_count',sellers)
    ratio=safe_ratio(buyers,sellers)
    add('buyer_seller_ratio',ratio)
    add('seller_count_zero',(sellers==0).astype(float))
    add('broker_buyer_fraction',safe_ratio(buyers,total_brokers))
    add('broker_count_balance',safe_ratio(buyers-sellers,total_brokers))
    for h in (5,20):
        add(f'buyer_seller_ratio_delta_{h}',ratio-ratio.shift(h))
    for name,data in [('broker_abs_net_value',amounts),('broker_abs_net_lot',lot_totals)]:
        add(name+'_log1p',np.log1p(data))
        add(name+'_relative20',safe_ratio(data,data.rolling(20,min_periods=20).median()))
    add('broker_price_spread',price_spread)
    add('broker_price_coverage',price_coverage);add('broker_lot_coverage',lot_coverage)
    add('lot_buy_hhi',lot_buy_hhi);add('lot_sell_hhi',lot_sell_hhi)

    for h in (5,10,20):
        add(f'broker_snapshot_coverage_{h}',series([s is not None for s in snapshots]).rolling(h,min_periods=h).mean())
    for name in ['Total Buyer','Total Seller']:
        add('summary_'+name.lower().replace(' ','_'),source(name))
    add('summary_buyer_seller_ratio',safe_ratio(source('Total Buyer'),source('Total Seller')))
    for name in ['Net Top3 All Lot','Net Top5 All Lot']:
        key=name.lower().replace(' ','_')
        raw=source(name)
        add('summary_'+key+'_share',safe_ratio(raw,lot_totals))
        for h in (5,10,20):
            add(f'summary_{key}_cumulative_{h}',safe_ratio(raw.rolling(h,min_periods=h).sum(),lot_totals.rolling(h,min_periods=h).sum()))

    # Retail is consumed as supplied, without claiming it identifies all retail investors.
    retail=source('Retail')
    add('summary_retail_raw',retail)
    add('summary_retail_signed_log1p',np.sign(retail)*np.log1p(abs(retail)))
    add('summary_retail_share_abs_net_lot',safe_ratio(retail,lot_totals))
    for h in (5,10,20):
        add(f'summary_retail_cumulative_{h}',retail.rolling(h,min_periods=h).sum())
        add(f'summary_retail_cumulative_share_{h}',safe_ratio(retail.rolling(h,min_periods=h).sum(),lot_totals.rolling(h,min_periods=h).sum()))
    for code in vocabulary:
        lot=series(raw_lot[code]);share=safe_ratio(lot,lot_totals)
        add(f'net_lot_share_{code}',share,mask=False)
        add(f'net_lot_share_delta_{code}',share.diff(),mask=False)
        for h in (5,10,20):
            add(f'net_lot_cumulative_share_{h}_{code}',safe_ratio(lot.rolling(h,min_periods=h).sum(),lot_totals.rolling(h,min_periods=h).sum()),mask=False)
            add(f'broker_persistence_{h}_{code}',np.sign(lot).rolling(h,min_periods=h).mean(),mask=False)
        add(f'broker_signed_streak_{code}',signed_streak(lot),mask=False)
        flips=((np.sign(lot)*np.sign(lot.shift()))==-1).astype(float)
        flips[lot.isna()|lot.shift().isna()]=np.nan
        add(f'broker_side_flip_rate20_{code}',flips.rolling(20,min_periods=20).mean(),mask=False)
        add(f'broker_price_deviation_{code}',raw_price[code])
    names=list(columns)
    matrix=np.column_stack([columns[k] for k in names]).astype('float32')
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite expanded features after missing-value handling')
    coverage={'missing_by_feature':missing_counts,
              'price_comparison':'raw broker averages vs abs-net-value-weighted raw broker reference; not adjusted Close',
              'retail_definition':'Retail column used as supplied; investor identity unverified',
              'lot_units':'Net lot as stored in blot/slot or NBLot/NSLot; not gross turnover',
              'warmup_sessions':40,
              'missing_policy':'Scalar features and broker price deviations carry explicit missing flags; broker lot rolling features use day-level lot/snapshot coverage'}
    return matrix,names,coverage
