"""Freeze and audit every available daily format BEFORE long-history fitting."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from event_sources import market_calendar, candidates, read_broker
from event_data import load_daily, make_folds, indices


def run(out, base, tickers, snapshot=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    base = Path(base); frozen = Path(snapshot) if snapshot else out/'raw-source'; manifest = []
    def copy(path):
        data = path.read_bytes(); rel = path.relative_to(base)
        target = frozen/rel; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest.append(dict(path=str(rel), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    calendar = market_calendar(base/'Daily'/'All')
    if snapshot is None:
        for ticker in tickers:
            copy(base/'Daily'/'Summary'/f'{ticker}_Summary.csv')
            for day in calendar:
                for investor in ['All', 'Foreign']:
                    for path in candidates(base/'Daily'/investor, ticker, day, investor):
                        if path.exists(): copy(path)
            for path in (base/'Monthly'/'Ownership'/'lbrpe').glob(f'*/{ticker}.json'):
                copy(path)
        (out/'snapshot-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    overlap = []
    for ticker in tickers:
        for investor in ['All', 'Foreign']:
            for day in calendar:
                paths = [p for p in candidates(frozen/'Daily'/investor, ticker, day, investor) if p.exists()]
                csv = next((p for p in paths if p.suffix == '.csv'), None)
                modern = next((p for p in paths if p.suffix in ['.json', '.xls']), None)
                if csv is None or modern is None: continue
                try:
                    a = read_broker(csv, ticker, day, investor)[2]
                    b = read_broker(modern, ticker, day, investor)[2]
                    codes = set(a['values'])|set(b['values'])
                    different = [c for c in codes if abs(a['values'].get(c,0)-b['values'].get(c,0))
                                 > a['value_rounding'].get(c,0)+max(1,abs(b['values'].get(c,0)))*1e-6]
                    lot_different = [c for c in codes if not np.isclose(a['lots'].get(c,0), b['lots'].get(c,0), equal_nan=True)]
                    overlap.append(dict(ticker=ticker, investor=investor, date=str(day.date()),
                        different_values=len(different), different_lots=len(lot_different), error=None))
                except (ValueError, KeyError, OSError) as error:
                    overlap.append(dict(ticker=ticker, investor=investor, date=str(day.date()), error=str(error)))
    pd.DataFrame(overlap).to_csv(out/'overlap.csv', index=False)
    report = []
    for ticker in tickers:
        print('Audit', ticker, flush=True)
        data = load_daily(ticker, frozen/'Daily'/'Summary', frozen/'Daily'/'All',
                          frozen/'Daily'/'Foreign', frozen/'Monthly'/'Ownership',
                          calendar_summary=frozen/'Daily'/'Summary'/'ADRO_Summary.csv')
        folds = make_folds(len(data['dates']), 3, 126, 126, 20, 20)
        audit = dict(data['audit'], ticker=ticker, start=str(data['dates'][0].date()), end=str(data['dates'][-1].date()),
            calendar_sessions=len(data['dates']), finite_targets=int(np.isfinite(data['y']).all(1).sum()),
            folds=[dict(f, counts={k:len(v) for k,v in indices(data,f,20).items()}) for f in folds])
        (out/f'{ticker}-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        report.append({k:v for k,v in audit.items() if k != 'quality'})
    (out/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', default='data')
    parser.add_argument('--tickers', nargs='+', default=['ADRO','BBCA','BBRI'])
    parser.add_argument('--snapshot', default=None)
    args = parser.parse_args(); run(args.out, args.base, args.tickers, args.snapshot)
