"""Daily broker source discovery for both YYYYMMDD CSV and ISO JSON/XLS."""
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from main import broker_row, broker_xls, normalize_flows


def market_calendar(root):
    dates = set()
    for path in Path(root).iterdir():
        if not path.is_dir():
            continue
        if re.fullmatch(r'\d{8}', path.name):
            dates.add(pd.to_datetime(path.name, format='%Y%m%d'))
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}', path.name):
            dates.add(pd.to_datetime(path.name, format='%Y-%m-%d'))
    if not dates:
        raise ValueError(f'No dated broker directories: {root}')
    return pd.DatetimeIndex(sorted(dates))


def candidates(root, ticker, day, investor='All'):
    root = Path(root); iso = f'{day:%Y-%m-%d}'; compact = f'{day:%Y%m%d}'
    suffix = 'ALL' if investor == 'All' else 'FOREIGN'
    return [root/iso/f'{ticker}_{suffix}_{iso}.json', root/iso/f'{ticker}.xls',
            root/compact/f'{ticker}-{compact}.csv', root/iso/f'{ticker}-{compact}.csv']


def compact_number(raw):
    """Parse original rounded display numbers and return rounding half-unit.

    Commas are thousands separators. No value is reconstructed from prices.
    """
    text = str(raw).strip().replace(',', '')
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([KMBT]?)', text, re.I)
    if not match:
        raise ValueError(f'Invalid numeric broker field: {raw!r}')
    value = Decimal(match[1]); scale = Decimal({'': 1, 'K': 10**3, 'M': 10**6, 'B': 10**9, 'T': 10**12}[match[2].upper()])
    number = float(value*scale)
    if not np.isfinite(number):
        raise ValueError('Non-finite broker number')
    tolerance = float(Decimal(10)**value.as_tuple().exponent*scale/2) if match[2] else 0.
    return number, tolerance


def broker_csv(path, ticker, day, details=False):
    path = Path(path)
    if path.name != f'{ticker}-{day:%Y%m%d}.csv' or path.parent.name not in [f'{day:%Y%m%d}', f'{day:%Y-%m-%d}']:
        raise ValueError(f'CSV ticker/date path mismatch: {path}')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ['Buyer', 'B Lot', 'B Value', 'B Average', 'Seller', 'S Lot', 'S Value', 'S Average']
    if not set(required) <= set(frame):
        raise ValueError(f'CSV schema mismatch: {path}')
    flows, lots, prices, rounding, buy, sell, seen = {}, {}, {}, {}, [], [], set()
    lot_sign_anomalies = 0; value_sign_anomalies = 0
    for row in frame.to_dict('records'):
        for side, prefix, sign, values in [('Buyer', 'B', 1, buy), ('Seller', 'S', -1, sell)]:
            code = row[side].strip().upper()
            if code in ['', '-', '--', 'NAN']:
                continue
            if not re.fullmatch(r'[A-Z]{2}', code) or code in seen:
                # Net-mode needs disjoint sides. Duplicates would require
                # gross-flow semantics and cannot silently be treated as net.
                raise ValueError(f'Invalid/duplicate broker code {code}: {path}')
            seen.add(code)
            value, error = compact_number(row[prefix+' Value'])
            lot, _ = compact_number(row[prefix+' Lot'])
            value_sign_anomalies += int(sign*value < 0)
            lot_sign_anomalies += int(sign*lot < 0)
            try:
                price, _ = compact_number(row[prefix+' Average'])
            except ValueError:
                price = np.nan
            flows[code] = value; lots[code] = lot; prices[code] = price
            rounding[code] = error; values.append(abs(value))
    normalized, fields = normalize_flows(flows, [v for v in flows.values() if v > 0], [-v for v in flows.values() if v < 0])
    detail = dict(values=flows, lots=lots, prices=prices, value_rounding=rounding, lot_sign_anomalies=lot_sign_anomalies,
                  value_sign_anomalies=value_sign_anomalies)
    return (normalized, fields, detail) if details else (normalized, fields)


def read_broker(path, ticker, day, investor='All'):
    if Path(path).suffix.lower() == '.csv':
        return broker_csv(path, ticker, day, details=True)
    if Path(path).suffix.lower() == '.xls':
        result = broker_xls(path, ticker, day, details=True, investor=investor, preserve_lot_sign=True, preserve_value_sign=True)
    else:
        result = broker_row(Path(path), ticker, day, details=True, preserve_lot_sign=True, preserve_value_sign=True)
    detail = result[2]; flows = detail['values']
    normalized, fields = normalize_flows(flows, [v for v in flows.values() if v > 0], [-v for v in flows.values() if v < 0])
    return normalized, fields, detail


def resolve(root, ticker, day, investor='All'):
    paths = candidates(root, ticker, day, investor)
    return next((p for p in paths if p.exists()), paths[0])
