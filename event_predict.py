"""Inference using trusted locally generated checkpoints and frozen scalers."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
import torch

from event_data import load_daily, HORIZONS
from event_models import EventModel, LegacyCNN
from event_training import infer, decoded


def restore(checkpoint, device='cpu'):
    """Only load checkpoints produced by this project (pickle-based .pt)."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = saved['preprocess']; vocabulary = len(state['vocabulary'])
    model = (LegacyCNN(len(saved['features']), saved['window'], vocabulary, len(saved['token_names']))
             if saved['mode'] == 'cnn' else EventModel(len(saved['features']), len(saved['token_names']),
                 vocabulary, saved['groups'], saved['mode'], saved['width']))
    model.load_state_dict(saved['state_dict']); model.to(device); model.eval()
    return model, saved


def transform(raw, state):
    if raw['names'] != state['feature_names']:
        raise ValueError('Feature schema changed; use matching source configuration')
    lookup = {code: i+2 for i, code in enumerate(state['vocabulary'])}
    x = np.clip((raw['x']-state['mean'])/state['std'], -10, 10).astype('float32')
    x[:, state['constant_columns']] = 0
    tokens = np.where(raw['mask'][..., None],
        np.clip((raw['tokens']-state['token_mean'])/state['token_std'], -10, 10), 0).astype('float32')
    ids = np.array([[lookup.get(c, 1) if c else 0 for c in row] for row in raw['codes']], dtype='int64')
    return dict(raw, x=x, tokens=tokens, ids=ids)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--date', default=None)
    parser.add_argument('--summary', default='data/Daily/Summary')
    parser.add_argument('--brokers', default='data/Daily/All')
    parser.add_argument('--foreign', default='data/Daily/Foreign')
    parser.add_argument('--ownership', default='data/Monthly/Ownership')
    parser.add_argument('--external', default=None)
    parser.add_argument('--calendar-summary', default=None)
    parser.add_argument('--max-brokers', type=int, default=96)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    model, saved = restore(args.checkpoint, args.device)
    raw = load_daily(args.ticker, args.summary, args.brokers, args.foreign, args.ownership, args.external, args.max_brokers,
                     args.calendar_summary or saved.get('calendar_summary'))
    data = transform(raw, saved['preprocess'])
    i = len(data['dates'])-1 if args.date is None else data['dates'].get_loc(args.date)
    window = saved['window']
    if i < 40+window-1 or not data['good'][i-window+1:i+1].all():
        raise ValueError('Insufficient valid historical input for this date')
    runtime = SimpleNamespace(window=window, batch_size=1, device=args.device)
    output, gates = infer(model, data, np.array([i]), runtime, saved['preprocess'], drop=saved.get('drop'))
    returns, probabilities, vol = decoded(output, saved['preprocess'])
    result = dict(ticker=args.ticker, date=str(data['dates'][i].date()),
                  checkpoint=args.checkpoint, source_hash_matches_training=raw['source_hash'] == saved['source_hash'],
                  status='exploratory; frozen historical model, not latest refit',
                  forecasts=[dict(horizon=h, log_return=float(returns[0, j]),
                                  probability_positive=float(probabilities[0, j]), realized_volatility=float(vol[0, j]))
                             for j, h in enumerate(HORIZONS)])
    result['training_objective']=saved.get('training_objective','multitask')
    if result['training_objective']=='return_only':
        result['auxiliary_heads_status']='Untrained: probability and volatility outputs must not be used'
        for forecast in result['forecasts']:
            forecast['probability_positive']=None;forecast['realized_volatility']=None
    if saved['mode'] != 'cnn':
        result['gates'] = dict(zip(saved['groups'], gates[0, -1].astype(float).tolist()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    cli()
