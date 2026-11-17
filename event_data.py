"""Point-in-time daily broker sets; reuse the existing parsers/features.

Inputs are EOD snapshots, available for a next-session open decision only.
Net-mode source fields are NEVER interpreted as gross buy/sell turnover.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xlrd
from torch.utils.data import Dataset

from main import broker_row, broker_xls, read_summary
from compact_features import compact_matrix, GROUPS, PRICE, align_loaded
from event_sources import market_calendar, resolve, read_broker

HORIZONS = (5, 10, 20)
TOKEN_NAMES = ['signed_log_net_value', 'net_value_share', 'signed_log_net_lot',
               'net_lot_missing', 'consistency_5', 'share_acceleration',
               'positive_net_value_share', 'negative_net_value_share']


def fingerprint(paths):
    h = hashlib.sha256()
    for path in sorted(set(map(Path, paths))):
        h.update(str(path).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def ownership_asof(root, ticker, calendar):
    """Conservative local-availability join; no latest overrides or registry mix.

    KSEI as_of is a position date, not release time, so it is excluded.
    LBRPE fetched_at guards against retrospective parsing/selection revisions.
    """
    rows, paths = [], []
    for path in sorted((Path(root)/'lbrpe').glob(f'*/{ticker}.json')):
        paths.append(path)
        obj = json.loads(path.read_text(encoding='utf-8-sig'))
        stamps = [pd.to_datetime(obj.get(key), errors='coerce')
                  for key in ['announced_at', 'fetched_at']]
        if any(pd.isna(s) for s in stamps):
            continue
        local = [pd.Timestamp(s).tz_convert('Asia/Jakarta').tz_localize(None)
                 if pd.Timestamp(s).tzinfo is not None else pd.Timestamp(s) for s in stamps]
        available = max(local)
        rows.append(dict(available_at=available,
                         free_float=pd.to_numeric(obj.get('free_float'), errors='coerce')/100,
                         log_sid=np.log1p(pd.to_numeric(obj.get('sid'), errors='coerce'))))
    result = np.full((len(calendar), 4), np.nan, dtype='float32')
    if rows:
        source = pd.DataFrame(rows).sort_values('available_at').drop_duplicates('available_at', keep='last')
        queries = pd.DataFrame({'available_at': calendar + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)})
        joined = pd.merge_asof(queries, source, on='available_at', direction='backward',
                               tolerance=pd.Timedelta(days=100))
        for j, key in enumerate(['free_float', 'log_sid']):
            value = joined[key].to_numpy()
            result[:, 2*j] = np.nan_to_num(value)
            result[:, 2*j+1] = ~np.isfinite(value)
    else:
        result[:] = [0, 1, 0, 1]
    return result, paths, rows


def external_asof(path, calendar):
    """Optional real CSV: available_at plus ihsg_close/usd_idr/sector_close.

    One file per ticker is required to avoid mixing sector assignments.
    Publication timestamps must carry Asia/Jakarta local wall time, no timezone.
    Missing/stale (>7 calendar days) observations retain missing flags.
    """
    source = pd.read_csv(path)
    source['available_at'] = pd.to_datetime(source['available_at'], errors='raise')
    if source.available_at.dt.tz is not None:
        raise ValueError('External available_at must use naive Asia/Jakarta local time')
    if source.available_at.duplicated().any():
        raise ValueError('Duplicate external availability times')
    query = pd.DataFrame({'available_at': calendar+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)})
    joined = pd.merge_asof(query, source.sort_values('available_at'), on='available_at',
                           direction='backward', tolerance=pd.Timedelta(days=7))
    columns, names, groups = [], [], {}
    for group, key in [('ihsg', 'ihsg_close'), ('usd_idr', 'usd_idr'), ('sector', 'sector_close')]:
        if key not in joined:
            continue
        raw = pd.to_numeric(joined[key], errors='coerce').where(lambda x: x > 0)
        change = np.log(raw).diff().to_numpy()
        groups[group] = [len(columns), len(columns)+1]
        columns.extend([np.nan_to_num(change), (~np.isfinite(change)).astype(float)])
        names.extend([key+'_log_change', key+'_missing'])
    return (np.column_stack(columns).astype('float32') if columns else np.empty((len(calendar), 0))), names, groups


def load_daily(ticker, summary, brokers, foreign=None, ownership=None, external=None, max_brokers=96, calendar_summary=None):
    broker_root = Path(brokers)
    calendar = market_calendar(broker_root)
    discovered = calendar
    if calendar_summary:
        reference, _ = read_summary(Path(calendar_summary))
        # Reference is frozen before training, common to every stock. Reindexing
        # below still preserves stock-specific missing rows on this calendar.
        calendar = reference.index[(reference.index >= discovered.min()) & (reference.index <= discovered.max())]
    path = Path(summary)/f'{ticker}_Summary.csv'
    df, good = read_summary(path)
    # Deduplicated union of compact CSV and ISO JSON/XLS daily directories.
    # Retain Summary holes; they are not silently compressed out of horizons.
    calendar = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
    df = df.reindex(calendar); good = good.reindex(calendar, fill_value=False)
    df.loc[~good, ['Open', 'High', 'Low', 'Close', 'Volume']] = np.nan
    cache, extras, snapshots, quality, sources = [], [], [], [], [path]
    if calendar_summary:
        sources.append(Path(calendar_summary))
    foreign_rows = []; formats = {}; foreign_formats = {}; lot_sign_anomalies = {'all': 0, 'foreign': 0}
    for day in calendar:
        path = resolve(broker_root, ticker, day)
        if path.exists():
            sources.append(path)
        try:
            flow, extra, detail = read_broker(path, ticker, day)
            formats[path.suffix] = formats.get(path.suffix, 0)+1
            lot_sign_anomalies['all'] += detail.get('lot_sign_anomalies', 0)
        except (OSError, ValueError, KeyError, TypeError, xlrd.XLRDError) as error:
            flow, extra, detail = {}, [1, 0, 0, 0], None
            quality.append({'date': str(day.date()), 'source': 'all', 'error': str(error)})
        cache.append(flow); extras.append(extra); snapshots.append(detail)
        fpath = resolve(foreign, ticker, day, 'Foreign') if foreign else None
        try:
            if fpath is None:
                raise FileNotFoundError('foreign source not configured')
            sources.append(fpath) if fpath.exists() else None
            _, _, fdetail = read_broker(fpath, ticker, day, 'Foreign')
            foreign_formats[fpath.suffix] = foreign_formats.get(fpath.suffix, 0)+1
            lot_sign_anomalies['foreign'] += fdetail.get('lot_sign_anomalies', 0)
            net = sum(fdetail['values'].values())
            absval = sum(abs(v) for v in fdetail['values'].values())
            lots = list(fdetail['lots'].values())
            lot = sum(lots) if all(np.isfinite(lots)) else np.nan
            foreign_rows.append([np.sign(net)*np.log1p(abs(net)), net/max(absval, 1),
                                 np.sign(lot)*np.log1p(abs(lot)), float(not np.isfinite(lot)), 0])
        except (OSError, ValueError, KeyError, TypeError, xlrd.XLRDError) as error:
            foreign_rows.append([0, 0, 0, 1, 1])
            quality.append({'date': str(day.date()), 'source': 'foreign', 'error': str(error)})
    loaded = (df, good, cache, np.asarray(extras), snapshots, quality)
    matrix, names, pairs = compact_matrix(loaded)
    groups = {'ohlcv': [k for name in PRICE for k in pairs[name]],
              'broker': [k for name in pairs if name not in PRICE for k in pairs[name]]}
    if foreign:
        groups['foreign'] = list(range(matrix.shape[1], matrix.shape[1]+5))
        matrix = np.column_stack([matrix, np.nan_to_num(foreign_rows)]).astype('float32')
        names += ['foreign_signed_log_net_value', 'foreign_net_share', 'foreign_signed_log_net_lot',
                  'foreign_lot_missing', 'foreign_snapshot_missing']
    own_records = []
    if ownership:
        own, paths, own_records = ownership_asof(ownership, ticker, calendar)
        sources += paths
        if own_records:
            groups['ownership'] = list(range(matrix.shape[1], matrix.shape[1]+4))
            matrix = np.column_stack([matrix, own]).astype('float32')
            names += ['ownership_free_float', 'ownership_free_float_missing', 'ownership_log_sid', 'ownership_log_sid_missing']
    if external:
        ext, enames, egroups = external_asof(external, calendar)
        offset = matrix.shape[1]
        groups.update({key: [i+offset for i in ids] for key, ids in egroups.items()})
        matrix = np.column_stack([matrix, ext]); names += enames; sources.append(Path(external))
        # Relative strength is a past stock return minus past available IHSG return.
        if 'ihsg' in egroups:
            market = ext[:, egroups['ihsg'][0]]
            rs = np.log(df.Close).diff().to_numpy()-market
            missing = ~np.isfinite(rs) | (ext[:, egroups['ihsg'][1]] > 0)
            groups['ihsg'] += [matrix.shape[1], matrix.shape[1]+1]
            matrix = np.column_stack([matrix, np.where(missing, 0, rs), missing]).astype('float32')
            names += ['relative_strength_1', 'relative_strength_missing']
    tokens = np.zeros((len(df), max_brokers, len(TOKEN_NAMES)), dtype='float32')
    codes = np.full((len(df), max_brokers), '', dtype='<U8')
    mask = np.zeros((len(df), max_brokers), dtype=bool); truncated = 0
    for i, snapshot in enumerate(snapshots):
        if snapshot is None:
            continue
        selected = sorted(snapshot['values'], key=lambda c: (-abs(snapshot['values'][c]), c))
        truncated += max(0, len(selected)-max_brokers)
        for j, code in enumerate(selected[:max_brokers]):
            value = snapshot['values'][code]; lot = snapshot['lots'].get(code, np.nan)
            share = cache[i].get(code, 0)
            history = cache[max(0, i-4):i+1]
            consistency = np.mean([np.sign(x.get(code, 0)) for x in history])
            previous = cache[i-1].get(code, 0) if i else 0
            previous2 = cache[i-2].get(code, 0) if i > 1 else 0
            tokens[i, j] = [np.sign(value)*np.log1p(abs(value)), share,
                           np.sign(lot)*np.log1p(abs(lot)) if np.isfinite(lot) else 0,
                           not np.isfinite(lot), consistency, share-2*previous+previous2, max(share, 0), min(share, 0)]
            codes[i, j] = code; mask[i, j] = True
    logret = np.log(df.Close).diff().to_numpy()
    targets = np.full((len(df), 6), np.nan, dtype='float32')
    for i in range(len(df)-max(HORIZONS)):
        if not good.iloc[i:i+max(HORIZONS)+1].all():
            continue
        for j, horizon in enumerate(HORIZONS):
            targets[i, j] = np.log(df.Close.iloc[i+horizon]/df.Close.iloc[i])
            vol = np.sqrt(np.mean(logret[i+1:i+horizon+1]**2)*252)
            targets[i, j+3] = np.log(max(vol, 1e-6))
    return dict(ticker=ticker, dates=calendar, frame=df, good=good.to_numpy(),
                x=matrix.astype('float32'), tokens=tokens, codes=codes, mask=mask, y=targets,
                names=names, groups=groups, source_hash=fingerprint(sources),
                audit=dict(quality=quality, truncated_broker_tokens=truncated,
                           all_formats=formats, foreign_formats=foreign_formats,
                           csv_lot_sign_anomalies_preserved=lot_sign_anomalies,
                           summary_missing_or_invalid=int((~good).sum()),
                           calendar_reference=str(calendar_summary),
                           broker_dates_outside_reference=[str(d.date()) for d in discovered.difference(calendar)] if calendar_summary else [],
                           ownership_records=len(own_records), source_files=len(sources),
                           missing_groups=[g for g in ['ihsg', 'usd_idr', 'sector', 'ownership', 'foreign'] if g not in groups]))


def make_folds(n, folds=3, test_size=7, val_size=8, embargo=20, window=20):
    """Forward folds: label_end + embargo < following split's first query.

    The extra embargo is separate from 20-session label purging. Later folds
    may learn from earlier test periods once their labels are fully mature.
    """
    result = []
    first_test = n-max(HORIZONS)-folds*test_size
    for f in range(folds):
        test_start = first_test+f*test_size
        val_start = test_start-max(HORIZONS)-embargo-val_size
        train_stop = val_start-max(HORIZONS)-embargo
        if train_stop-(40+window-1) < 12:
            raise ValueError('Insufficient history for purge + embargo; reduce fold/test/val/window sizes explicitly')
        result.append(dict(fold=f+1, train_stop=train_stop, val_start=val_start,
                           val_stop=test_start-max(HORIZONS)-embargo,
                           test_start=test_start, test_stop=test_start+test_size, embargo=embargo))
    return result


def indices(data, fold, window):
    available = np.array([i for i in range(40+window-1, len(data['dates'])-20)
                          if data['good'][i-window+1:i+1].all() and np.isfinite(data['y'][i]).all()])
    result = dict(train=available[available < fold['train_stop']],
                  val=available[(available >= fold['val_start']) & (available < fold['val_stop'])],
                  test=available[(available >= fold['test_start']) & (available < fold['test_stop'])])
    if any(len(v) == 0 for v in result.values()):
        raise ValueError(f'Empty split for {data["ticker"]}: {result}')
    return result


def preprocess(data, train, window):
    days = np.unique(np.concatenate([np.arange(i-window+1, i+1) for i in train]))
    mean = data['x'][days].mean(0); std = data['x'][days].std(0).clip(.01)
    constant = np.ptp(data['x'][days], axis=0) < 1e-8
    observed = data['tokens'][days][data['mask'][days]]
    tm = observed.mean(0) if len(observed) else np.zeros(len(TOKEN_NAMES))
    ts = observed.std(0).clip(.01) if len(observed) else np.ones(len(TOKEN_NAMES))
    vocabulary = sorted(set(data['codes'][days][data['mask'][days]].tolist()))
    lookup = {code: i+2 for i, code in enumerate(vocabulary)}  # 0 padding; 1 unknown
    ids = np.array([[lookup.get(c, 1) if c else 0 for c in row] for row in data['codes']], dtype='int64')
    ym = data['y'][train].mean(0); ys = data['y'][train].std(0).clip(.001)
    prepared = dict(data, x=np.clip((data['x']-mean)/std, -10, 10).astype('float32'),
                    tokens=np.where(data['mask'][..., None], np.clip((data['tokens']-tm)/ts, -10, 10), 0).astype('float32'),
                    ids=ids)
    # A feature never observed varying in train cannot acquire a meaningful
    # weight from a later test-only release. Preserve it in source/cache, freeze
    # to zero for this fold; reconsider with the next train window only.
    prepared['x'][:, constant] = 0
    state = dict(mean=mean, std=std, token_mean=tm, token_std=ts, target_mean=ym,
                 target_std=ys, vocabulary=vocabulary, feature_names=data['names'], groups=data['groups'],
                 constant_columns=constant, fit_days=days)
    return prepared, state


class Windows(Dataset):
    def __init__(self, data, rows, window, state, drop=None):
        self.data, self.rows, self.window, self.state, self.drop = data, rows, window, state, drop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i = self.rows[k]; sl = slice(i-self.window+1, i+1)
        x = self.data['x'][sl].copy(); token = self.data['tokens'][sl].copy()
        ids = self.data['ids'][sl].copy(); mask = self.data['mask'][sl].copy()
        if self.drop:
            x[:, self.data['groups'][self.drop]] = 0
            if self.drop == 'broker':
                token[:] = 0; ids[:] = 0; mask[:] = False
        y = self.data['y'][i]
        return tuple(torch.from_numpy(a) for a in [x, token, ids, mask,
            ((y-self.state['target_mean'])/self.state['target_std']).astype('float32'),
            (y[:3] > 0).astype('float32')])
