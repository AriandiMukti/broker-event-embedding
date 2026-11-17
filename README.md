# Broker Event Embedding

A research framework for learning daily broker-event representations and testing their relationship to future market outcomes. It combines OHLCV context, masked broker sets, and causal temporal models, with explicit controls for temporal leakage and exploratory statistical analysis.

This public release contains source code and synthetic unit-test fixtures. Raw observations, trained weights, frozen research snapshots, logs, predictions, and historical reports are intentionally excluded. No historical performance claim is presented as independently reproduced by this release.

## Research methodology

Daily broker records are aligned to a trading-session calendar. Missing sessions and invalid OHLCV observations are retained as missing or rejected rather than silently repaired. Broker net values are treated as net flows, not gross turnover. Parsers support dated CSV, JSON, and legacy XLS inputs and validate ticker/date metadata.

The event pipeline creates signed flow, relative share, lot availability, consistency, and acceleration tokens for each broker. Vocabulary, normalization, and constant-column decisions are learned from training observations. Ownership and external context are joined using availability timestamps and backward as-of joins. Current downloads cannot establish historical vendor vintages.

Chronological folds apply horizon-aware purging and an embargo. The primary event targets cover 5, 10, and 20 sessions: log return, positive-return probability, and realized RMS volatility. Validation selects checkpoints; evaluation code includes baselines, calibration, and next-open backtests with transaction fees and slippage. Existing historical studies are exploratory; repeated use of the same history limits confirmatory interpretation.

## Architecture

```text
Validated daily files -> session alignment -> train-only preprocessing
                           |                       |
                    masked broker tokens      OHLCV/context
                           |                       |
                     Set Transformer ---- optional group gates
                                      |
                                  fusion layer
                                      |
                           causal dilated temporal CNN
                                      |
                    return / direction / volatility heads
```

`event_models.py` implements masked attention, permutation-invariant pooling, empty-set handling, and temporal blocks with left padding. The primary model supports no gates, static gates, and dynamic gates. Separate relation models mask target brokers and test next-session net-state prediction. Earlier convolutional, adaptive, and token-embedding approaches remain available for comparison.

## Experiments implemented

| Experiment | Entry point | Purpose |
| --- | --- | --- |
| Event representation | `event_research.py` | Set/temporal model, gating, ablation, baselines |
| Objective ablation | `event_objective_study.py` | Paired return-only versus multitask training |
| Market context | `event_market_study.py` | Local inputs versus additional market context |
| Frozen encoder probes | `event_broker_probe.py` | Predictive contribution beyond own history and raw peers |
| End-to-end relations | `event_relation_study.py` | Direct next-session broker-state objective |
| Association controls | `event_association_study.py` | Relation controls and paired uncertainty analysis |
| Diagnostics and replay | `event_diagnose.py`, `event_*_verify.py` | Leakage checks, receipt validation, checkpoint replay |

Secondary studies often require a completed frozen source run. Their default `runs/` references identify the original experiment recipes; those artifacts are not distributed. Inspect each script's `--help` and supply your own source directory.

## Installation

Use Python 3.10 or later. The publication verification environment uses Python 3.10; see `requirements-verified.txt` for exact direct dependency versions from that environment. Use a separate virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

For the dependency versions used during verification, consult `requirements-verified.txt`. This is an installed direct-dependency snapshot, not a fully resolved cross-platform lockfile or a verified fresh installation. Its CUDA-specific PyTorch build may require a matching package index; use a platform-appropriate CPU build for CPU-only machines. Optional Yahoo acquisition requires `python -m pip install -r requirements-market.txt` and network access; the unit suite does not require a download.

## Input data and reproduction

Acquire data you are authorized to use and keep it outside version control. Expected local layout:

```text
data/
  Daily/Summary/ADRO_Summary.csv
  Daily/All/YYYY-MM-DD/ADRO_ALL_YYYY-MM-DD.json
  Daily/All/YYYY-MM-DD/ADRO.xls
  Daily/All/YYYYMMDD/ADRO-YYYYMMDD.csv
  Daily/Foreign/...
  Monthly/Ownership/lbrpe/YYYY-MM/ADRO.json
```

Broker layouts above are alternatives. Summary CSVs require `Date` in `%d-%b-%y` format and numeric `Open`, `High`, `Low`, `Close`, `Volume` fields. Compact broker CSVs require `Buyer`, `B Lot`, `B Value`, `B Average`, `Seller`, `S Lot`, `S Value`, and `S Average`. JSON/XLS metadata and net-mode expectations are defined by `main.py` and `event_sources.py`; consult parser tests for synthetic examples. Ownership records need announcement and retrieval timestamps to participate in point-in-time joins.

Start with the primary pipeline after supplying sufficient real history for purging and folds:

```bash
python event_research.py --help
python event_research.py --tickers ADRO --summary data/Daily/Summary --brokers data/Daily/All --foreign data/Daily/Foreign --ownership data/Monthly/Ownership --out runs/event-local --device cpu --no-amp --no-ablation
python event_verify.py runs/event-local
```

Use fresh output directories. Archive input hashes, source snapshots, configuration, split boundaries, seeds, library versions, device, precision mode, selection receipts, and verification results privately. Fingerprints can include paths, so byte-identical relocated datasets may not produce the same fingerprint. Run on CPU without AMP for a portable reference, and expect hardware/version-dependent numeric differences. Verify paired initialization, training-only selection, purging, and execution timing before comparing experiments. Do not upload generated run folders: they can contain source data, local paths, and private metadata.

## Verified results and limitations

Publication validation is recorded in `VALIDATION.md`. Synthetic tests check mechanisms including causal behavior, masking, fold separation, as-of availability, training-only preprocessing, and replay/statistical controls. Passing these checks establishes software behavior on fixtures; it does not establish predictive accuracy, profitable trading, broker coordination, or causal relationships in real markets.

Historical result tables and trained models were excluded because they depend on private inputs and artifact lineage. Real-data experiments have not been rerun as part of publication. Financial performance cannot be independently reproduced from this repository alone.

## Repository map

- `main.py`, `features.py`, `compact_features.py`: parsing, features, early models.
- `event_data.py`, `event_sources.py`: point-in-time assembly and preprocessing.
- `event_models.py`, `event_training.py`, `event_evaluation.py`: architecture, training, metrics.
- `event_*_study.py`, `event_*_results.py`, `event_*_verify.py`: experiment runners, report generation, replay.
- `test_*.py`: synthetic fixtures and behavioral tests.
- `SECURITY.md`: publication exclusions and safe artifact handling.

Some legacy report generators retain their original Indonesian narrative. The public documentation is in English. No license has been selected for this release; publication alone does not grant a general reuse license.
