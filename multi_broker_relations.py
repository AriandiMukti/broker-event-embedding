"""Run separate relation models for the existing 25-stock research universe."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from broker_relation_test import run, SEEDS
from relation_io import source_fingerprint

STOCKS = ['ADRO','BBCA','BBRI','BMRI','TLKM','ANTM','PTBA','BREN','TPIA','BRPT',
          'PTRO','INKP','BUMI','BRMS','ISAT','DEWA','BUVA','CUAN','MDKA','ASII',
          'BBNI','BULL','TINS','GOTO','ENRG']


def worker(options):
    args = SimpleNamespace(**options)
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=True)
    fingerprint=source_fingerprint(args.ticker,args.summary,args.brokers)
    required = ['summary.csv','paired_block_bootstrap.csv','neighbors_by_seed.csv',
                'peer_ablation_by_seed.csv','summary_by_fold.csv','fold-3/next_day_peers.pt','config.json']
    if args.resume and all((folder/f).exists() for f in required):
        saved = json.loads((folder/'config.json').read_text())
        if saved.get('source_fingerprint')==fingerprint and all(saved.get(k) == options[k] for k in ['ticker','epochs','patience','seeds','dimension','clean_ohlcv','summary','brokers']):
            return dict(ticker=args.ticker, status='resumed')
    start = time.monotonic()
    try:
        with (folder/'training.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            run(args)
        config=json.loads((folder/'config.json').read_text());config['source_fingerprint']=fingerprint
        (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        return dict(ticker=args.ticker, status='completed', seconds=round(time.monotonic()-start,1))
    except Exception:
        detail = traceback.format_exc()
        (folder/'error.log').write_text(detail, encoding='utf-8')
        return dict(ticker=args.ticker, status='failed', error=detail)


def main(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config['stocks'] = args.tickers
    (root/'batch_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    statuses = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for ticker in args.tickers:
            options = dict(out=str(root/ticker), ticker=ticker, summary=args.summary, brokers=args.brokers,
                           epochs=args.epochs, patience=args.patience, dimension=8, seeds=SEEDS,
                           fast=True, clean_ohlcv=True, resume=args.resume)
            futures[pool.submit(worker, options)] = ticker
        for future in as_completed(futures):
            result = future.result(); statuses.append(result)
            (root/'batch_status.json').write_text(json.dumps(statuses, indent=2), encoding='utf-8')
            print(len(statuses), '/', len(futures), result, flush=True)
    if not args.no_analysis:
        from relation_maps import analyze
        analyze(root)
        from audit_relation_batch import audit
        audit(root)
    if any(r['status']=='failed' for r in statuses): raise SystemExit('Some stocks failed; see batch_status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', default=r'data/Daily/Summary')
    parser.add_argument('--brokers', default=r'data/Daily/All')
    parser.add_argument('--out', default='runs/broker-relations-25')
    parser.add_argument('--tickers', nargs='+', default=STOCKS)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no-analysis', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or min(args.epochs,args.patience)<1 or len(set(args.tickers))!=len(args.tickers):
        parser.error('Positive sizes and distinct stock codes required')
    if any(len(t)!=4 or not t.isalpha() or t!=t.upper() for t in args.tickers): parser.error('Use four-letter uppercase tickers')
    main(args)
