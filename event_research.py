"""Run the frozen exploratory event protocol; refuses to overwrite any run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from event_data import load_daily, make_folds, indices, preprocess, HORIZONS, fingerprint
from event_training import fit, infer, diagnostics, decoded, objective
from event_evaluation import metrics, backtest, regime


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        return str(x)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def run(args):
    torch.set_num_threads(2)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).parent/name for name in ['event_data.py', 'event_models.py',
        'event_training.py', 'event_evaluation.py', 'event_research.py', 'event_report.py', 'dashboard.py',
        'test_event_research.py', 'event_sources.py', 'event_history_audit.py', 'test_event_sources.py',
        'event_predict.py', 'event_verify.py', 'requirements.txt', 'main.py', 'features.py', 'compact_features.py']]
    config = dict(vars(args), torch=torch.__version__, python=platform.python_version(),
        code_hash=fingerprint(source_paths), test_status='exploratory historical reuse; no untouched confirmation set',
        target='log return; probability log return >0; annualized sqrt(mean future squared daily log return *252)',
        objective='train-standardized return MSE + 0.25 log-vol MSE + 0.25 direction BCE',
        selection='best validation objective; no test-driven choices; ablation/importance validation only',
        execution='fixed long/cash rule: probability>0.5 AND return>0; t+1 open to t+h+1 open; no overlap',
        embargo_definition='20-session label purge PLUS extra configured embargo before both later splits',
        ownership_definition='LBRPE max(announced_at,fetched_at); no KSEI position-date backfill; constant train columns frozen',
        regime='stock trailing 20-session volatility vs train median; no market regime without IHSG',
        missing_macro=['ihsg', 'usd_idr', 'sector'] if not args.external_dir else [])
    write_json(root/'config.json', config)
    (root/'sources').mkdir()
    for path in source_paths:
        (root/'sources'/path.name).write_bytes(path.read_bytes())
    tables = {name: [] for name in ['metrics', 'calibration', 'predictions', 'training', 'history', 'gates',
        'gate_samples', 'importance', 'ablation', 'samples', 'backtest', 'trades', 'equity', 'regime_metrics']}
    data_by_ticker = {}

    def flush():
        for name, rows in tables.items():
            if rows:
                pd.DataFrame(rows).to_csv(root/f'{name}.csv', index=False)
        if tables['metrics']:
            metric = pd.DataFrame(tables['metrics'])
            keys = ['mae', 'rmse', 'accuracy', 'balanced_accuracy', 'roc_auc', 'brier', 'calibration_ece', 'vol_mae', 'vol_rmse']
            metric.groupby(['model', 'horizon'])[keys].mean().reset_index().to_csv(root/'comparison.csv', index=False)
            metric.groupby('model')[keys].mean().reset_index().to_csv(root/'macro_summary.csv', index=False)
        from dashboard import build_event_dashboard
        build_event_dashboard(root)

    def add_predictions(data, split, state, fold, name, seed, returns, probs, vols):
        context = dict(ticker=data['ticker'], fold=fold['fold'], model=name, seed=seed)
        rows = split['test']; actual = data['y'][rows]
        scores, bins = metrics(actual, returns, probs, vols)
        tables['metrics'].extend([{**context, **r} for r in scores])
        tables['calibration'].extend([{**context, **r} for r in bins])
        regimes, cutoff = regime(data, rows, split['train'])
        for reg in np.unique(regimes):
            hit = regimes == reg
            scores, _ = metrics(actual[hit], returns[hit], probs[hit], vols[hit])
            tables['regime_metrics'].extend([{**context, 'regime': reg, 'threshold': cutoff, **r} for r in scores])
        for k, i in enumerate(rows):
            for j, h in enumerate(HORIZONS):
                tables['predictions'].append(dict(context, date=str(data['dates'][i].date()), calendar_index=int(i),
                    horizon=h, actual=float(actual[k, j]), prediction=float(returns[k, j]),
                    probability=float(probs[k, j]), actual_vol=float(np.exp(actual[k, j+3])),
                    predicted_vol=float(vols[k, j]), regime=regimes[k]))

    for ticker in args.tickers:
        print(f'Loading {ticker}', flush=True)
        external = Path(args.external_dir)/f'{ticker}.csv' if args.external_dir else None
        raw = load_daily(ticker, args.summary, args.brokers, args.foreign, args.ownership, external, args.max_brokers, args.calendar_summary)
        data_by_ticker[ticker] = raw
        raw['frame'].to_csv(root/f'{ticker}-ohlcv-snapshot.csv', index_label='Date')
        write_json(root/f'{ticker}-data-audit.json', dict(raw['audit'], source_hash=raw['source_hash'],
            dates=[str(d.date()) for d in raw['dates']], features=raw['names'], groups=raw['groups']))
        np.savez_compressed(root/f'{ticker}-daily.npz', x=raw['x'], tokens=raw['tokens'], codes=raw['codes'],
                            mask=raw['mask'], y=raw['y'], good=raw['good'], dates=np.asarray(raw['dates'].strftime('%Y-%m-%d'), dtype='U10'))
        folds = make_folds(len(raw['dates']), args.folds, args.test_size, args.val_size, args.embargo, args.window)
        for fold in folds:
            split = indices(raw, fold, args.window)
            data, state = preprocess(raw, split['train'], args.window)
            folder = root/ticker/f"fold-{fold['fold']}"; folder.mkdir(parents=True)
            state['window'] = args.window
            train_days = state['fit_days']
            unsupported = [g for g, cols in data['groups'].items() if state['constant_columns'][cols].all()]
            write_json(folder/'audit.json', dict(fold, counts={k: len(v) for k, v in split.items()},
                vocabulary=state['vocabulary'], fit_last_date=str(data['dates'][max(train_days)].date()),
                unsupported_groups=unsupported, constant_features=[n for n, c in zip(data['names'], state['constant_columns']) if c],
                splits={k: [str(data['dates'][i].date()) for i in v] for k, v in split.items()}))
            for key, rows in split.items():
                for i in rows:
                    tables['samples'].append(dict(ticker=ticker, fold=fold['fold'], split=key, calendar_index=int(i),
                        date=str(data['dates'][i].date()), label_end=str(data['dates'][i+20].date())))
            np.savez_compressed(folder/'prepared.npz', x=data['x'], tokens=data['tokens'], ids=data['ids'], mask=data['mask'], y=data['y'],
                                **{f'{k}_indices': v for k, v in split.items()})
            for name, seed, regression, probabilities in baselines(data, split, args):
                add_predictions(data, split, state, fold, name, seed, regression[:, :3], probabilities,
                                np.exp(np.clip(regression[:, 3:], -15, 6)))
            for seed in args.seeds:
                for mode in ['cnn', 'none', 'static', 'dynamic']:
                    name = {'cnn': 'legacy_cnn_matched', 'none': 'set_tcn', 'static': 'set_tcn_static', 'dynamic': 'set_tcn_dynamic'}[mode]
                    print(f'{ticker} fold={fold["fold"]} seed={seed} {name}', flush=True)
                    model, record, history, gates = fit(data, split, state, args, mode, seed, folder/f'{mode}-seed{seed}')
                    context = dict(ticker=ticker, fold=fold['fold'], seed=seed, model=name)
                    tables['training'].append(dict(context, **record))
                    tables['history'].extend([{**context, **r} for r in history])
                    tables['gates'].extend([{**context, **r} for r in gates])
                    output, _ = infer(model, data, split['test'], args, state)
                    add_predictions(data, split, state, fold, name, seed, *decoded(output, state))
                    valout, valgate = infer(model, data, split['val'], args, state)
                    valtarget = torch.from_numpy(((data['y'][split['val']]-state['target_mean'])/state['target_std']).astype('float32'))
                    positive = torch.from_numpy((data['y'][split['val'], :3] > 0).astype('float32'))
                    base_loss = float(objective(torch.from_numpy(valout), valtarget, positive))
                    regs, _ = regime(raw, split['val'], split['train'])
                    if mode != 'cnn':
                        for k, i in enumerate(split['val']):
                            for j, group in enumerate(data['groups']):
                                tables['gate_samples'].append(dict(context, date=str(data['dates'][i].date()), regime=regs[k],
                                    group=group, gate=float(valgate[k, -1, j])))
                        for reg in ['all']+list(np.unique(regs)):
                            subset = split if reg == 'all' else dict(split, val=split['val'][regs == reg])
                            values = diagnostics(model, data, subset, state, args, seed)
                            tables['importance'].extend([{**context, 'regime': reg, 'supported_in_train': r['group'] not in unsupported,
                                                          **r} for r in values])
                    # Refit ablations for the dynamic model on exactly the same
                    # folds/seeds. Validation diagnostics never select test recipe.
                    if mode == 'dynamic' and not args.no_ablation:
                        for group in data['groups']:
                            if group in unsupported:
                                tables['ablation'].append(dict(context, group=group, status='unavailable_in_train',
                                    base_validation_loss=base_loss, validation_loss=base_loss, delta_loss=0.))
                                continue
                            abmodel, abrecord, abhistory, abgates = fit(data, split, state, args, mode, seed,
                                folder/f'dynamic-without-{group}-seed{seed}', drop=group)
                            about, _ = infer(abmodel, data, split['val'], args, state, drop=group)
                            abloss = float(objective(torch.from_numpy(about), valtarget, positive))
                            tables['ablation'].append(dict(context, group=group, status='refit_validation_only',
                                base_validation_loss=base_loss, validation_loss=abloss, delta_loss=abloss-base_loss,
                                best_epoch=abrecord['best_epoch']))
                            tables['training'].append(dict(context, model=name+'-without-'+group, **abrecord))
                            tables['history'].extend([{**context, 'model': name+'-without-'+group, **r} for r in abhistory])
                            tables['gates'].extend([{**context, 'model': name+'-without-'+group, **r} for r in abgates])
                            del abmodel
                    del model
                flush()
    # Stitch predictions across folds BEFORE executing positions; no overlap
    # across retraining boundaries. Seeds remain separate strategies.
    predictions = pd.DataFrame(tables['predictions'])
    for (ticker, name, seed, horizon), part in predictions.groupby(['ticker', 'model', 'seed', 'horizon']):
        context = dict(ticker=ticker, model=name, seed=int(seed), horizon=int(horizon))
        for label, fees, slip in [('base', args.fee_bps, args.slippage_bps), ('stress', args.fee_bps*2, args.slippage_bps*2)]:
            stats, trades, curve = backtest(data_by_ticker[ticker]['frame'], part, int(horizon), fees, slip)
            context_cost = dict(context, cost_scenario=label)
            tables['backtest'].append(dict(context_cost, **stats))
            tables['trades'].extend([{**context_cost, **r} for r in trades])
            tables['equity'].extend([{**context_cost, **r} for r in curve])
    flush()
    write_json(root/'progress.json', dict(status='complete', trained_fits=len(tables['training']), test_status=config['test_status']))
    from event_report import write_report
    write_report(root)
    print(f'Complete: {root}', flush=True)


def baselines(data, split, args):
    train = data['y'][split['train']]; n = len(split['test'])
    probability = (train[:, :3] > 0).mean(0)
    constant = np.tile(train.mean(0), (n, 1))
    zero = constant.copy(); zero[:, :3] = 0
    result = [('zero', -1, zero, np.full((n, 3), .5)),
              ('train_mean', -1, constant, np.tile(probability, (n, 1)))]
    # Real manual daily features only; latest + temporal mean/std. Hyperparameters
    # are fixed before test, no synthetic rows and no test selection.
    from lightgbm import LGBMRegressor, LGBMClassifier
    def manual(rows):
        return np.array([np.r_[data['x'][i], data['x'][i-args.window+1:i+1].mean(0),
                               data['x'][i-args.window+1:i+1].std(0)] for i in rows])
    xtrain, xtest = manual(split['train']), manual(split['test'])
    params = dict(n_estimators=60, num_leaves=5, min_child_samples=5, learning_rate=.03,
                  reg_lambda=1., verbosity=-1, n_jobs=2, random_state=42, deterministic=True, force_col_wise=True)
    pred = np.column_stack([LGBMRegressor(**params).fit(xtrain, train[:, j]).predict(xtest) for j in range(6)])
    probabilities = []
    for j in range(3):
        label = train[:, j] > 0
        probabilities.append(np.full(n, label.mean()) if len(np.unique(label)) < 2 else
                             LGBMClassifier(**params).fit(xtrain, label).predict_proba(xtest)[:, 1])
    result.append(('lightgbm_manual', 42, pred, np.column_stack(probabilities)))
    return result


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', nargs='+', default=['ADRO', 'BBCA', 'BBRI'])
    parser.add_argument('--out', default='runs/event-v1')
    parser.add_argument('--summary', default='data/Daily/Summary')
    parser.add_argument('--brokers', default='data/Daily/All')
    parser.add_argument('--foreign', default='data/Daily/Foreign')
    parser.add_argument('--ownership', default='data/Monthly/Ownership')
    parser.add_argument('--external-dir', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 42, 123])
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--test-size', type=int, default=126)
    parser.add_argument('--val-size', type=int, default=126)
    parser.add_argument('--calendar-summary', default=None, help='Frozen common market-session Summary CSV; stock-specific missing rows retained')
    parser.add_argument('--embargo', type=int, default=20)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--max-brokers', type=int, default=96)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-ablation', action='store_true')
    parser.add_argument('--permutation-repeats', type=int, default=3)
    parser.add_argument('--fee-bps', type=float, default=15.)
    parser.add_argument('--slippage-bps', type=float, default=10.)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.permutation_repeats, args.batch_size) < 1 or args.embargo < 0:
        parser.error('Positive epochs/patience/repeats/batch required; nonnegative embargo')
    run(args)


if __name__ == '__main__':
    cli()
