"""Post-fit synchronized date-block bootstrap; never used to choose models."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def intervals(values, block=20, repeats=2000, seed=8911):
    """Circular trading-date blocks, shared over stocks/horizons/models.

    Returns simultaneous bands over columns, conditional on the fitted models.
    This does not include training/architecture-selection uncertainty.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite synchronized daily loss-gain matrix')
    rng = np.random.default_rng(seed); n, width = values.shape
    actual = values.mean(0); draws = np.empty((repeats, width))
    for start in range(0, repeats, 64):
        size = min(64, repeats-start)
        starts = rng.integers(0, n, (size, int(np.ceil(n/block))))
        ids = ((starts[..., None]+np.arange(block)) % n).reshape(size, -1)[:, :n]
        draws[start:start+size] = values[ids].mean(1)
    sd = draws.std(0, ddof=1); safe = np.where(sd > 1e-12, sd, 1)
    critical = np.quantile(np.max(abs((draws-actual)/safe), 1), .95)
    return actual, actual-critical*sd, actual+critical*sd, np.quantile(draws, .025, axis=0), np.quantile(draws, .975, axis=0)


def run(root):
    root = Path(root); predictions = pd.read_csv(root/'predictions.csv')
    predictions['loss'] = abs(predictions.prediction-predictions.actual)
    losses = predictions.groupby(['date','ticker','horizon','model']).loss.mean().unstack('model')
    methods = [m for m in losses if m != 'train_mean']
    gains = losses[methods].rsub(losses['train_mean'], axis=0)
    matrix = gains.stack().unstack(['ticker','horizon','model']).sort_index()
    if matrix.isna().any().any():
        raise ValueError('Unequal test coverage: align explicitly rather than silently dropping dates')
    records = []
    for block in [20,40]:
        # One family contains ALL models x stocks x horizons, plus macro means.
        macro = pd.DataFrame({model:matrix.xs(model,level='model',axis=1).mean(1) for model in methods})
        combined = np.column_stack([matrix.to_numpy(), macro.to_numpy()])
        bounds = intervals(combined, block)
        keys = list(matrix.columns)+[('ALL','ALL',model) for model in methods]
        for i, (ticker, horizon, model) in enumerate(keys):
            records.append(dict(ticker=ticker,horizon=horizon,model=model,comparator='train_mean',
                block_sessions=block,dates=len(matrix),gain_mae=bounds[0][i],simultaneous_low=bounds[1][i],
                simultaneous_high=bounds[2][i],nominal_low=bounds[3][i],nominal_high=bounds[4][i],
                note='positive gain favors model; historical exploratory; conditional on fitted models'))
    pd.DataFrame(records).to_csv(root/'paired_uncertainty.csv', index=False)
    print(pd.DataFrame(records).query("ticker == 'ALL'").to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root')
    run(parser.parse_args().root)
