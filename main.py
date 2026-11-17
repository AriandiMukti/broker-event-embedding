"""Local broker/OHLC embedding research prototype. No network/data downloads."""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xlrd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge

HORIZONS = (5, 10, 20)
BASE = ['return', 'open_gap', 'range', 'body', 'log_volume_change',
        'volume_relative20', 'volatility20', 'broker_missing',
        'buy_hhi', 'sell_hhi', 'active_brokers']


def num(x):
    try:
        v = float(str(x).replace(',', ''))
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_summary(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='raise')
    if df['Date'].duplicated().any():
        raise ValueError(f'Duplicate dates: {path}')
    df = df.sort_values('Date').set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    good = (np.isfinite(df[['Open', 'High', 'Low', 'Close', 'Volume']]).all(axis=1)
            & (df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
            & (df.Volume >= 0) & (df.High >= df.Low)
            & (df.High + 0.01 >= df[['Open', 'Close']].max(axis=1))
            & (df.Low - 0.01 <= df[['Open', 'Close']].min(axis=1)))
    return df, good


def normalize_flows(flows, buys, sells):
    denom = max(sum(abs(v) for v in flows.values()), 1.0)
    def hhi(values):
        a = np.array(values)
        return float(np.sum((a / a.sum()) ** 2)) if a.sum() > 0 else 0.0
    return {k: v / denom for k, v in flows.items()}, [
        0.0, hhi(buys), hhi(sells), len(flows) / 100.0]


def broker_xls(path, ticker, day, details=False, investor='All', preserve_lot_sign=False, preserve_value_sign=False):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)
    header, settings = sheet.row_values(0), sheet.row_values(1)
    if (header[1] != ticker or header[3] != f'{day:%Y-%m-%d}'
            or header[5] != f'{day:%Y-%m-%d}' or header[7] != 'Net'
            or settings[1] != investor or settings[3] != 'RG'):
        raise ValueError(f'XLS metadata mismatch (require daily {investor}/RG/Net): {path}')
    columns = sheet.row_values(2)
    if columns[:3] != ['NBY', 'NBLot', 'NBVal'] or columns[5:8] != ['NSL', 'NSLot', 'NSVal']:
        raise ValueError(f'Unknown XLS column schema: {path}')
    flows, buys, sells, lots, prices = {}, [], [], {}, {}
    for i in range(3, sheet.nrows):
        row = sheet.row_values(i)
        for code_col, value_col, sign, values in [(0, 2, 1, buys), (5, 7, -1, sells)]:
            code = str(row[code_col]).strip()
            if not code:
                continue
            if len(code) != 2 or not code.isalpha():
                raise ValueError(f'Unexpected broker code {code}: {path}')
            value = abs(num(row[value_col]))
            flows[code] = flows.get(code, 0) + (num(row[value_col]) if preserve_value_sign else sign * value)
            lot = optional_number(row[1 if sign == 1 else 6])
            lots[code] = (lot if preserve_lot_sign else sign * abs(lot)) if np.isfinite(lot) else np.nan
            prices[code] = optional_number(row[3 if sign == 1 else 8])
            values.append(value)
    book.release_resources()
    normalized, fields = normalize_flows(flows, buys, sells)
    return (normalized, fields, {'values':flows, 'lots':lots, 'prices':prices}) if details else (normalized, fields)


def optional_number(value):
    try:
        value = float(str(value).replace(',', ''))
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def broker_row(path, ticker, day, details=False, preserve_lot_sign=False, preserve_value_sign=False):
    if path.suffix.lower() == '.xls':
        return broker_xls(path, ticker, day, details=details, preserve_lot_sign=preserve_lot_sign, preserve_value_sign=preserve_value_sign)
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    data = obj['data']
    summary = data['broker_summary']
    flows, lots, prices = {}, {}, {}
    buys, sells = [], []
    expected = day.strftime('%Y%m%d')
    for side, value_key in [('brokers_buy', 'bval'), ('brokers_sell', 'sval')]:
        for row in summary.get(side, []):
            if str(row.get('netbs_stock_code')) != ticker:
                raise ValueError(f'Ticker mismatch: {path}')
            if str(row.get('netbs_date')) != expected:
                raise ValueError(f'Date mismatch: {path}')
            value = abs(num(row.get(value_key)))
            signed = value if side == 'brokers_buy' else -value
            if preserve_value_sign:
                signed = num(row.get(value_key))
            code = row['netbs_broker_code']
            flows[code] = flows.get(code, 0) + signed
            lot = optional_number(row.get('blot' if side == 'brokers_buy' else 'slot'))
            lots[code] = (lot if preserve_lot_sign else (1 if side == 'brokers_buy' else -1)*abs(lot)) if np.isfinite(lot) else np.nan
            prices[code] = optional_number(row.get('netbs_buy_avg_price' if side == 'brokers_buy' else 'netbs_sell_avg_price'))
            (buys if side == 'brokers_buy' else sells).append(value)
    normalized, fields = normalize_flows(flows, buys, sells)
    return (normalized, fields, {'values':flows, 'lots':lots, 'prices':prices}) if details else (normalized, fields)


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    universe, invalid = {}, {}
    only_ticker = getattr(args, 'only_ticker', None)
    if only_ticker:
        only_ticker = only_ticker.upper()
        if not only_ticker.isalpha() or len(only_ticker) != 4:
            raise ValueError('Use a four-letter ticker for --only-ticker')
        paths = [Path(args.summary) / f'{only_ticker}_Summary.csv']
    else:
        paths = sorted(Path(args.summary).glob('*_Summary.csv'))
    for path in paths:
        ticker = path.name.removesuffix('_Summary.csv')
        df, good = read_summary(path)
        universe[ticker] = (df, good)
        invalid[ticker] = int((~good).sum())
    if not universe:
        raise ValueError('No *_Summary.csv found')
    calendar = pd.DatetimeIndex(sorted(set().union(*(set(x[0].index) for x in universe.values()))))
    # Do not use observations beyond the latest available daily broker directory.
    broker_dates = sorted(p.name for p in Path(args.brokers).iterdir()
                          if p.is_dir() and len(p.name) == 10 and p.name[4] == '-')
    calendar = calendar[calendar <= pd.Timestamp(broker_dates[-1])]
    if len(calendar) < 120:
        raise ValueError('At least 120 common market-calendar dates required')
    val_start = calendar[int(len(calendar) * .60)]
    test_start = calendar[int(len(calendar) * .80)]
    ranks = []
    for ticker, (df, good) in universe.items():
        train = df.loc[(df.index < val_start) & good]
        if len(train) >= args.window + 40:
            # Ranking based only on the training period, not the latest volume.
            ranks.append((float((train.Close * train.Volume).median()), ticker))
    tickers = [t for _, t in sorted(ranks, reverse=True)[:args.stocks]]
    if not tickers:
        raise ValueError('No ticker with sufficient clean training history')
    records, vocabulary, quality = {}, set(), []
    for n, ticker in enumerate(tickers):
        df, good = universe[ticker]
        df = df.reindex(calendar)
        good = good.reindex(calendar, fill_value=False)
        cache, extra, detail_rows = [], [], []
        missing = malformed = 0
        for day in calendar:
            path = Path(args.brokers) / day.strftime('%Y-%m-%d') / f'{ticker}_ALL_{day:%Y-%m-%d}.json'
            if not path.exists():
                path = path.parent / f'{ticker}.xls'
            if not path.exists():
                cache.append({}); extra.append([1, 0, 0, 0]); missing += 1
                detail_rows.append(None)
                continue
            try:
                flows, fields, detail = broker_row(path, ticker, day, details=True)
            except (ValueError, KeyError, TypeError, xlrd.XLRDError) as exc:
                cache.append({}); extra.append([1, 0, 0, 0]); malformed += 1
                detail_rows.append(None)
                quality.append({'ticker': ticker, 'date': str(day.date()), 'error': str(exc)})
                continue
            cache.append(flows); extra.append(fields); detail_rows.append(detail)
            if day < val_start:
                vocabulary.update(flows)
        records[ticker] = (df, good, cache, extra, detail_rows)
        quality.append({'ticker': ticker, 'missing_broker_days': missing,
                        'malformed_broker_days': malformed, 'invalid_ohlc_rows': invalid[ticker]})
        print(f'Prepared {n+1}/{len(tickers)} {ticker}', flush=True)
    vocabulary = sorted(vocabulary)
    features = BASE + [f'net_share_{c}' for c in vocabulary] + ['net_share_unseen']
    samples, labels, metadata, latest_x, latest_meta = [], [], [], [], []
    coverage = {}
    feature_set = getattr(args, 'feature_set', 'basic')
    for ticker, (df, good, cache, extra, detail_rows) in records.items():
        ret = np.log(df.Close).diff()
        base = np.column_stack([
            ret, np.log(df.Open / df.Close.shift()), np.log(df.High / df.Low),
            np.log(df.Close / df.Open), np.log1p(df.Volume).diff(),
            np.log((df.Volume + 1) / (df.Volume.rolling(20, min_periods=5).median() + 1)),
            ret.rolling(20, min_periods=5).std(), np.array(extra)])
        broker = np.array([[flow.get(code, 0) for code in vocabulary] +
                           [sum(v for k, v in flow.items() if k not in vocabulary)]
                           for flow in cache])
        matrix = np.column_stack([base, broker]).astype('float32')
        if feature_set == 'expanded_v2':
            from features import expanded_features
            additional, additional_names, feature_coverage = expanded_features(df, vocabulary, detail_rows)
            matrix = np.column_stack([matrix, additional]).astype('float32')
            features = BASE + [f'net_share_{c}' for c in vocabulary] + ['net_share_unseen'] + additional_names
            coverage[ticker] = feature_coverage
        for i in range(args.window - 1, len(calendar)):
            x = matrix[i-args.window+1:i+1]
            if feature_set == 'expanded_v2' and i-args.window+1 < 40:
                continue  # Every day in the window has a full 40-session price lookback.
            if not good.iloc[i-args.window+1:i+1].all() or not np.isfinite(x).all():
                continue
            meta = {'ticker': ticker, 'date': str(calendar[i].date())}
            if i == len(calendar)-1:
                latest_x.append(x); latest_meta.append(meta)
            if i + max(HORIZONS) >= len(calendar):
                continue
            if not good.iloc[i:i+max(HORIZONS)+1].all():
                continue
            y = [np.log(df.Close.iloc[i+h] / df.Close.iloc[i]) for h in HORIZONS]
            label_end = calendar[i + max(HORIZONS)]
            if label_end < val_start:
                split = 'train'
            elif calendar[i] >= val_start and label_end < test_start:
                split = 'val'
            elif calendar[i] >= test_start:
                split = 'test'
            else:
                continue  # Purge labels spanning a temporal boundary.
            samples.append(x); labels.append(y)
            metadata.append(dict(meta, split=split, label_end=str(label_end.date())))
    x = np.array(samples, dtype='float32'); y = np.array(labels, dtype='float32')
    meta = pd.DataFrame(metadata)
    for split in ['train', 'val', 'test']:
        if not (meta.split == split).any():
            raise ValueError(f'No {split} samples after purging')
    np.savez_compressed(out/'dataset.npz', x=x, y=y, latest_x=np.array(latest_x, dtype='float32'))
    meta.to_csv(out/'samples.csv', index=False)
    pd.DataFrame(latest_meta).to_csv(out/'latest.csv', index=False)
    config = {'feature_set': feature_set, 'feature_coverage':coverage,
              'scope': 'single_stock' if only_ticker else 'pooled',
              'only_ticker': only_ticker,
              'brokers': str(Path(args.brokers).resolve()), 'summary': str(Path(args.summary).resolve()),
              'window': args.window, 'features': features, 'vocabulary': vocabulary, 'tickers': tickers,
              'horizons': HORIZONS, 'universe_files': len(universe), 'calendar_days': len(calendar),
              'start': str(calendar[0].date()), 'end': str(calendar[-1].date()),
              'val_start': str(val_start.date()), 'test_start': str(test_start.date()),
              'counts': meta.split.value_counts().to_dict(), 'quality': quality}
    (out/'data_report.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(config['counts'], 'features:', len(features), flush=True)


class Encoder(nn.Module):
    def __init__(self, features, window, channels=48, compact_decoder=False):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(features, channels, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.GELU(),
                                  nn.Flatten())
        self.embedding = nn.Sequential(nn.Linear(channels*window, 32), nn.LayerNorm(32), nn.GELU())
        self.decode = (nn.Sequential(nn.Linear(32, window*8), nn.Unflatten(1,(window,8)), nn.Linear(8,features))
                       if compact_decoder else nn.Linear(32, features*window))
        self.head = nn.Linear(32, len(HORIZONS))
        self.features, self.window = features, window

    def forward(self, x):
        z = self.embedding(self.conv(x.transpose(1, 2)))
        return z, self.head(z), self.decode(z).reshape(-1, self.window, self.features)


def metrics(y, pred, meta):
    result = {}
    for j, horizon in enumerate(HORIZONS):
        frame = meta[['date']].copy()
        frame['y'], frame['p'] = y[:, j], pred[:, j]
        daily_ics = [g.y.corr(g.p, method='spearman') for _, g in frame.groupby('date')
                     if len(g) >= 5 and g.y.nunique() > 1 and g.p.nunique() > 1]
        up = y[:, j] > 0
        correct = (pred[:, j] > 0) == up
        balanced = float((correct[up].mean()+correct[~up].mean())/2) if up.any() and (~up).any() else None
        temporal = [g.y.corr(g.p, method='spearman') for _, g in
                    pd.DataFrame({'ticker': meta.ticker.to_numpy(), 'y': y[:, j], 'p': pred[:, j]}).groupby('ticker')
                    if len(g) >= 3 and g.y.nunique() > 1 and g.p.nunique() > 1]
        result[str(horizon)] = {'mae_log_return': float(np.mean(abs(y[:, j]-pred[:, j]))),
                               'direction_accuracy': float(np.mean((y[:, j]>0) == (pred[:, j]>0))),
                               'balanced_direction_accuracy': balanced,
                               'actual_up_fraction': float(up.mean()),
                               'mean_time_series_rank_correlation': float(np.mean(temporal)) if temporal else None,
                               'mean_daily_rank_ic': float(np.nanmean(daily_ics)) if daily_ics else None,
                               'ic_dates': len(daily_ics)}
    return result


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = Path(args.out)
    pack = np.load(out/'dataset.npz')
    meta = pd.read_csv(out/'samples.csv')
    config = json.loads((out/'data_report.json').read_text(encoding='utf-8'))
    if config.get('scope') == 'single_stock':
        if set(meta.ticker) != {config['only_ticker']} or config['tickers'] != [config['only_ticker']]:
            raise ValueError('Single-stock isolation violated')
    masks = {s: (meta.split == s).to_numpy() for s in ['train', 'val', 'test']}
    x, y = pack['x'], pack['y']
    mean = x[masks['train']].mean(axis=(0, 1)); std = x[masks['train']].std(axis=(0, 1)).clip(.01)
    ym = y[masks['train']].mean(axis=0); ys = y[masks['train']].std(axis=0).clip(.001)
    xn = np.clip((x-mean)/std, -10, 10); yn = (y-ym)/ys
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channels = getattr(args, 'channels', 48)
    compact_decoder = config.get('feature_set') == 'expanded_v2'
    model = Encoder(x.shape[2], x.shape[1], channels=channels, compact_decoder=compact_decoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    loader = DataLoader(TensorDataset(torch.from_numpy(xn[masks['train']]),
                                     torch.from_numpy(yn[masks['train']])),
                        batch_size=args.batch, shuffle=True)
    def infer(array):
        zs, ps = [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(array), args.batch):
                z, p, _ = model(torch.from_numpy(array[start:start+args.batch]).to(device))
                zs.append(z.cpu().numpy()); ps.append(p.cpu().numpy())
        return np.concatenate(zs), np.concatenate(ps)*ys+ym
    best, history, best_state = float('inf'), [], None
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            hidden = torch.rand(bx.shape[0], bx.shape[1], 1, device=device) < .20
            _, p, reconstruction = model(bx.masked_fill(hidden, 0))
            loss = ((p-by)**2).mean() + .2 * ((reconstruction-bx)**2)[hidden.expand_as(bx)].mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()*len(bx)
        _, vp = infer(xn[masks['val']])
        validation = float(np.mean(((vp-y[masks['val']])/ys)**2))
        history.append({'epoch': epoch+1, 'train_loss': total/len(loader.dataset), 'val_mse_standardized': validation})
        if validation < best:
            best = validation; best_state = copy.deepcopy(model.state_dict())
        print(history[-1], flush=True)
    model.load_state_dict(best_state)
    z, pred = infer(xn)
    # Ridge sees the same full window and features; tune alpha on validation only.
    flat = xn.reshape(len(xn), -1)
    best_ridge, best_ridge_loss, best_alpha = None, float('inf'), None
    for alpha in (1.0, 100.0, 10000.0):
        baseline = Ridge(alpha=alpha).fit(flat[masks['train']], yn[masks['train']])
        loss = float(np.mean((baseline.predict(flat[masks['val']])-yn[masks['val']])**2))
        if loss < best_ridge_loss:
            best_ridge, best_ridge_loss, best_alpha = baseline, loss, alpha
    ridge_pred = best_ridge.predict(flat)*ys+ym
    report = {'scope': config.get('scope', 'pooled'), 'tickers': config['tickers'],
              'feature_set':config.get('feature_set','basic'), 'compact_decoder':compact_decoder,
              'channels': channels, 'device': str(device), 'seed': args.seed, 'history': history,
              'best_epoch': min(history, key=lambda h:h['val_mse_standardized'])['epoch'],
              'ridge_alpha': best_alpha, 'parameters': sum(p.numel() for p in model.parameters()),
              'limitations': ['Single chronological holdout, not a walk-forward trading backtest.',
                  'Overlapping labels; sample counts are not independent observations.',
                  'Current available universe; survivorship bias may exist.',
                  'Vendor price adjustment and historical revisions are unverified.',
                  'Broker identities do not identify underlying investors.',
                  'Features are available after daily file completion; no same-close execution assumption.']}
    for split in ['val', 'test']:
        m = masks[split]
        report[split] = {'samples': int(m.sum()), 'dates': int(meta.loc[m, 'date'].nunique()),
                         'cnn': metrics(y[m], pred[m], meta.loc[m]),
                         'ridge': metrics(y[m], ridge_pred[m], meta.loc[m]),
                         'train_mean': metrics(y[m], np.tile(ym, (m.sum(), 1)), meta.loc[m])}
    (out/'training_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    torch.save({'state_dict': model.cpu().state_dict(), 'features': x.shape[2], 'window': x.shape[1],
                'channels': channels, 'compact_decoder':compact_decoder,
                'feature_set':config.get('feature_set','basic'),
                'tickers': config['tickers'], 'scope': config.get('scope', 'pooled'),
                'mean': mean, 'std': std, 'target_mean': ym, 'target_std': ys}, out/'model.pt')
    latest = pd.read_csv(out/'latest.csv')
    lx = np.clip((pack['latest_x']-mean)/std, -10, 10)
    model.to(device)
    lz, lp = infer(lx)
    np.savez_compressed(out/'embeddings.npz', z=z, latest_z=lz)
    for j, h in enumerate(HORIZONS):
        latest[f'predicted_log_return_{h}'] = lp[:, j]
    latest.to_csv(out/'latest_predictions.csv', index=False)
    print(json.dumps({'best_epoch':report['best_epoch'], 'test':report['test']}, indent=2), flush=True)


def neighbors(args):
    out = Path(args.out)
    latest = pd.read_csv(out/'latest_predictions.csv')
    meta = pd.read_csv(out/'samples.csv')
    config = json.loads((out/'data_report.json').read_text(encoding='utf-8'))
    if config.get('scope') == 'single_stock' and set(meta.ticker) != {args.ticker.upper()}:
        raise ValueError('Query does not match the single-stock model')
    pack = np.load(out/'embeddings.npz'); data = np.load(out/'dataset.npz')
    matches = np.where(latest.ticker.to_numpy() == args.ticker.upper())[0]
    if not len(matches):
        raise ValueError('Ticker unavailable. See latest_predictions.csv / increase --stocks.')
    query = pack['latest_z'][matches[0]]
    z = pack['z']
    similarity = z @ query / np.maximum(np.linalg.norm(z, axis=1)*np.linalg.norm(query), 1e-8)
    # Train-only reference bank: embeddings and outcomes were available before validation.
    order = np.argsort(-similarity)
    chosen, last_dates = [], {}
    for i in order:
        row = meta.iloc[i]
        if row['split'] != 'train':
            continue
        date = pd.Timestamp(row.date)
        if any(abs((date-old).days) < 28 for old in last_dates.get(row.ticker, [])):
            continue
        chosen.append(i); last_dates.setdefault(row.ticker, []).append(date)
        if len(chosen) == args.k:
            break
    result = meta.iloc[chosen].copy()
    result['cosine_similarity'] = similarity[chosen]
    for j, h in enumerate(HORIZONS):
        result[f'realized_return_{h}'] = np.expm1(data['y'][chosen, j])
    result.to_csv(out/f'{args.ticker.upper()}_neighbors.csv', index=False)
    print(result.to_string(index=False))
    print('Historical analogues are descriptive; neighbor outcomes do not imply forecast probabilities.')


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'train', 'neighbors'])
    p.add_argument('--brokers', default=r'data/Daily/All')
    p.add_argument('--summary', default=r'data/Daily/Summary')
    p.add_argument('--out', default='runs/local')
    p.add_argument('--stocks', type=int, default=40)
    p.add_argument('--only-ticker', help='Read and train on exactly one stock; excludes every other Summary')
    p.add_argument('--channels', type=int, default=48)
    p.add_argument('--feature-set', choices=['basic','expanded_v2'], default='expanded_v2')
    p.add_argument('--window', type=int, default=20)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ticker', default='ADRO')
    p.add_argument('--k', type=int, default=10)
    args = p.parse_args()
    if min(args.stocks, args.window, args.epochs, args.batch, args.k, args.channels) < 1:
        p.error('Numeric options must be positive')
    {'prepare': prepare, 'train': train, 'neighbors': neighbors}[args.command](args)


if __name__ == '__main__':
    cli()
