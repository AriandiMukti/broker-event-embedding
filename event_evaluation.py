"""Exploratory metrics and executable next-open, non-overlapping backtests."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from event_data import HORIZONS


def metrics(actual, returns, probability, volatility):
    rows, calibration = [], []
    for j, h in enumerate(HORIZONS):
        y = actual[:, j]; label = y > 0; p = np.clip(probability[:, j], 1e-6, 1-1e-6)
        error = returns[:, j]-y
        ece = 0.
        for low in np.arange(0, 1, .1):
            hit = (p >= low) & (p < low+.1)
            if hit.any():
                confidence = float(p[hit].mean()); frequency = float(label[hit].mean())
                ece += hit.mean()*abs(confidence-frequency)
                calibration.append(dict(horizon=h, bin_low=low, n=int(hit.sum()),
                                        probability=confidence, positive_rate=frequency))
        truevol = np.exp(actual[:, j+3]); predvol = volatility[:, j]
        rows.append(dict(horizon=h, n=len(y), mae=float(np.mean(abs(error))),
            rmse=float(np.sqrt(np.mean(error**2))), accuracy=float(accuracy_score(label, p > .5)),
            balanced_accuracy=float(balanced_accuracy_score(label, p > .5)) if len(np.unique(label)) == 2 else None,
            roc_auc=float(roc_auc_score(label, p)) if len(np.unique(label)) == 2 else None,
            brier=float(np.mean((p-label)**2)), calibration_ece=float(ece),
            vol_mae=float(np.mean(abs(predvol-truevol))), vol_rmse=float(np.sqrt(np.mean((predvol-truevol)**2)))))
    return rows, calibration


def temporal_permutation(data, rows, window, group, seed, block=5):
    """Permute chronological daily blocks, then rebuild overlapping windows.

    Every member of a feature group shares the same daily mapping; broker
    IDs, values and masks move together. No independent shuffle of windows.
    Diagnostic on validation only; discontinuities are not causal evidence.
    """
    result = dict(data)
    days = np.arange(min(rows)-window+1, max(rows)+1)
    chunks = [days[i:i+block] for i in range(0, len(days), block)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chunks))
    if len(chunks) > 1 and np.array_equal(order, np.arange(len(chunks))):
        order = np.roll(order, 1)
    source = np.concatenate([chunks[i] for i in order])
    result['x'] = data['x'].copy()
    cols = data['groups'][group]
    result['x'][np.ix_(days, cols)] = data['x'][np.ix_(source, cols)]
    if group == 'broker':
        for name in ['tokens', 'ids', 'mask']:
            result[name] = data[name].copy(); result[name][days] = data[name][source]
    return result


def backtest(frame, forecasts, horizon, fee_bps=15., slippage_bps=10.):
    """Long/cash; forecast at EOD t, entry t+1 open, exit t+h+1 open.

    Positions do not overlap for one stock/model/seed/horizon. Daily equity is
    marked to the next open (same price basis as trades), with both sides of
    costs; a signal at an exit date may re-enter. Period includes cash days.
    """
    forecasts = forecasts.sort_values('calendar_index')
    if forecasts.empty:
        return {}, [], []
    first = int(forecasts.calendar_index.min())+1
    last = min(int(forecasts.calendar_index.max())+horizon+1, len(frame)-1)
    equity = np.ones(last-first+1); daily_turnover = np.zeros(len(equity))
    signals = {int(row.calendar_index): row for row in forecasts.itertuples()}
    trades = []; blocked_until = -1; cost = (fee_bps+slippage_bps)/10000
    for query, row in sorted(signals.items()):
        entry, exit_ = query+1, query+horizon+1
        if entry < blocked_until or exit_ > last or row.probability <= .5 or row.prediction <= 0:
            continue
        segment = frame.iloc[entry:exit_+1]
        prices = segment.Open.to_numpy()
        if not np.isfinite(segment[['Open', 'High', 'Low', 'Close']].to_numpy()).all() or np.any(prices <= 0):
            continue
        # Independent consecutive trade factors, retained across later cash days.
        a, b = entry-first, exit_-first
        equity[a:] *= 1-cost
        for k in range(a+1, b+1):
            equity[k:] *= prices[k-a]/prices[k-a-1]
        equity[b:] *= 1-cost
        daily_turnover[a] += 1; daily_turnover[b] += 1
        blocked_until = exit_
        trades.append(dict(query=str(frame.index[query].date()), entry=str(frame.index[entry].date()),
                           exit=str(frame.index[exit_].date()), entry_index=entry, exit_index=exit_,
                           gross_return=float(prices[-1]/prices[0]-1),
                           net_return=float((prices[-1]/prices[0])*(1-cost)**2-1)))
    returns = np.diff(np.r_[1., equity])/np.r_[1., equity[:-1]]
    stdev = returns.std(ddof=1) if len(returns) > 1 else 0
    drawdown = equity/np.maximum.accumulate(np.r_[1., equity])[1:]-1
    stats = dict(sharpe=float(returns.mean()/stdev*np.sqrt(252)) if stdev > 1e-12 else None,
                 maximum_drawdown=float(drawdown.min()), total_return=float(equity[-1]-1),
                 turnover=float(daily_turnover.sum()), average_daily_turnover=float(daily_turnover.mean()),
                 trades=len(trades), sessions=len(equity), fee_bps=fee_bps, slippage_bps=slippage_bps)
    curve = [dict(date=str(frame.index[first+i].date()), equity=float(e),
                  daily_return=float(returns[i]), turnover=float(daily_turnover[i])) for i, e in enumerate(equity)]
    return stats, trades, curve


def regime(data, rows, train):
    """Past realized stock volatility regime; absent IHSG is NOT market regime."""
    values = data['frame'].Close.pipe(np.log).diff().rolling(20).std().to_numpy()
    cutoff = float(np.nanmedian(values[train]))
    return np.where(values[rows] > cutoff, 'high_stock_vol', 'low_stock_vol'), cutoff
