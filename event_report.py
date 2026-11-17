"""Human-readable results from actual completed exports only."""
import json
from pathlib import Path
import pandas as pd


def write_report(root):
    root = Path(root)
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    summary = pd.read_csv(root/'macro_summary.csv')
    fits = pd.read_csv(root/'training.csv')
    samples = pd.read_csv(root/'samples.csv')
    counts = samples.groupby(['ticker', 'fold', 'split']).size().unstack().reset_index()
    importance = pd.read_csv(root/'importance.csv')
    ablation = pd.read_csv(root/'ablation.csv') if (root/'ablation.csv').exists() else pd.DataFrame()
    backtest = pd.read_csv(root/'backtest.csv')
    text = f'''# Broker Event Embedding — actual exploratory results

Run: `{root.resolve()}`. Device: {config['device']}. Seeds: {config['seeds']}.
Neural fits executed: {len(fits)} (including refit ablations); max epochs {config['epochs']}, early stopping patience {config['patience']}.
Best checkpoint picked by validation multitask loss. Peak allocated GPU memory: {fits.peak_gpu_mb.max():.1f} MiB; summed training time {fits.seconds.sum():.1f} seconds (excludes loading/diagnostics/export).

## Actual comparison

Equal mean over fold/stock/horizon/seed scores (not ensemble predictions). MAE/RMSE log return; volatility annualized fraction.
Probability outputs are uncalibrated sigmoid heads; Brier and ECE measure reliability, not proof of calibration. Baselines zero/mean use training priors; LightGBM has a fixed small-tree recipe.

{summary.round(6).to_markdown(index=False)}

## Data and time controls

{counts.to_markdown(index=False)}

Calendar from real broker sessions, preserving missing Summary rows. Input window {config['window']} sessions, warmup 40. Targets 5/10/20 sessions; return log, binary positive, realized RMS volatility (not demeaned standard deviation).
All scalers, vocabulary, regime thresholds and constant-column decisions fit on training inputs/labels only. Label purge 20 PLUS extra embargo {config['embargo']} before validation and test. Later folds train on expanded history only after target maturity. Test queries disjoint; outcomes overlap and are not independent.
Raw per-stock arrays and source SHA256 are saved; code snapshot and config precede all fits. Source/vendor revisions and corporate actions remain unverified.

## Architecture

A: broker-ID embedding + token projection, two masked set self-attention blocks, learned-query attention pooling per day, causal TCN dilation 1/2/4, multi-output heads. B adds sigmoid static group gates. C computes sigmoid gates from same-day available context before temporal processing. No broker position encoding; set permutation invariant. Padding/unknown broker IDs separated; empty sets handled explicitly.
CNN comparator reuses main.Encoder architecture/32D embedding, flattening per-broker tokens into train-vocabulary channels, with added probability/volatility head. All neural models use the same supervised objective; original reconstruction-trained CNN results/checkpoints remain separate. Decoder preserved but unused. This is a matched retraining, not a claim that new heads reproduce old checkpoints.
AdamW, ReduceLROnPlateau, early stopping, clipping 1.0, AMP on CUDA unless disabled, batch {config['batch_size']}; CPU fallback supported. Daily arrays cached once and windows materialized per batch.

## Feature contribution (validation only)

Static/dynamic gate history and per-stock/volatility-regime gates are in gates.csv and gate_samples.csv. Grouped temporal block permutation (block 5, {config['permutation_repeats']} repeats) moves synchronized daily groups and reconstructs overlapping windows. Loss gradients cover standardized continuous features; sums depend on group size and do not measure causal effects or categorical identity sensitivity.
Small gates alone do not identify noise. Negative permutation/ablation deltas can reflect sampling error, redundancy or unstable fits. No permanent feature removal or test-driven feature selection occurred.

{importance[(importance.method == 'temporal_permutation') & (importance.regime == 'all')].groupby(['model','group']).value.mean().round(6).to_markdown()}

Refit dynamic-model ablations (same folds/seeds/selection; positive delta means removing group hurts validation):

{ablation.groupby(['group','status']).delta_loss.mean().round(6).to_markdown() if not ablation.empty else 'Not executed in this smoke run.'}

IHSG/USDIDR/sector unavailable: ablations cannot be evaluated and no synthetic training features were created. Relative strength can only run when a real IHSG available_at file is provided. Foreign XLS validated Net/Foreign/RG; net flows, not gross values/volume. Ownership endpoint examined read-only: KSEI lacks release timestamps; latest registry and overrides excluded. LBRPE uses max(announced_at,fetched_at), not position date. Entirely constant/unavailable-in-train columns frozen per fold, retained in raw data. Ownership ablation marked unavailable when unsupported. Regimes are trailing STOCK volatility vs train median, not broad market regimes.

## Walk-forward backtest

Signals stitched across folds before position management. Fixed long/cash rule probability>0.5 AND return>0; next open entry, exit h sessions later at open. No overlapping positions across folds for a stock/seed/horizon. Both entry/exit costs; fee {config['fee_bps']} bp + slippage {config['slippage_bps']} bp per side, stress doubles both. Costs are assumptions, not broker-specific fees. Daily mark-to-open equity includes cash days, full round trips, Sharpe (risk-free=0), drawdown and turnover. Unadjusted prices, EOD availability and execution/liquidity are not verified: backtest illustrative, no tradability claim.

{backtest[backtest.cost_scenario == 'base'].groupby('model')[['total_return','sharpe','maximum_drawdown','turnover','trades']].mean().round(6).to_markdown()}

Full comparison.csv, metrics.csv, predictions.csv, calibration.csv, regime_metrics.csv, backtest.csv, equity.csv, trades.csv, history.csv and best.pt checkpoints preserve per-stock/fold/seed/horizon detail. index.html extends the existing Plotly dashboard exporter, with link to old CNN embedding maps. progress.json is updated every epoch; readers retry partial reads on Windows.

## Limits and next experiments

This historical period was previously explored; NO untouched confirmation test exists. Do not select a production model or infer statistical significance from these scores. Actual stock/fold sample sizes are shown above. Overlapping labels, per-stock models and multiple diagnostics limit effective evidence, even with several years of input history. Missing macro sources prevent a market-regime claim. More complexity can overfit.
Freeze this recipe and code hash, then use future point-in-time broker/OHLCV and dated ownership releases for a new untouched confirmation period. Use validated adjusted price/corporate-action data and actual execution constraints before economic conclusions. Add timestamped IHSG/USDIDR/sector only from real sources. Never compare run scores as if their stock membership or time splits were identical.
'''
    (root/'RESULTS.md').write_text(text, encoding='utf-8')
