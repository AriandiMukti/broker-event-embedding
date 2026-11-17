"""Replay exported predictions and ablation objectives from saved checkpoints."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from event_predict import restore
from event_training import infer, decoded, objective


def verify(root, source_root=None):
    root = Path(root); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    predictions = pd.read_csv(root/'predictions.csv')
    config = json.loads((root/'config.json').read_text())
    ablations = pd.read_csv(root/'ablation.csv') if (root/'ablation.csv').exists() else pd.DataFrame()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    count, abcount, max_error, cpu_error = 0, 0, 0., 0.
    names = dict(cnn='legacy_cnn_matched', none='set_tcn', static='set_tcn_static', dynamic='set_tcn_dynamic')
    for checkpoint in sorted(root.glob('*/fold-*/*/best.pt')):
        model, saved = restore(checkpoint, device)
        ticker = checkpoint.parents[2].name
        fold = int(checkpoint.parents[1].name.split('-')[1])
        arrays = np.load(checkpoint.parents[1]/'prepared.npz', allow_pickle=False)
        data = {k: arrays[k] for k in ['x', 'tokens', 'ids', 'mask', 'y']}
        data['groups'] = saved['groups']
        args = SimpleNamespace(window=saved['window'], batch_size=config['batch_size'], device=device)
        drop = saved.get('drop')
        rows = arrays['val_indices' if drop else 'test_indices']
        output, _ = infer(model, data, rows, args, saved['preprocess'], drop)
        if drop:
            state = saved['preprocess']
            target = torch.from_numpy(((data['y'][rows]-state['target_mean'])/state['target_std']).astype('float32'))
            positive = torch.from_numpy((data['y'][rows, :3] > 0).astype('float32'))
            loss = float(objective(torch.from_numpy(output), target, positive))
            match = ablations[(ablations.ticker == ticker) & (ablations.fold == fold)
                              & (ablations.seed == saved['seed']) & (ablations.group == drop)]
            np.testing.assert_allclose(loss, match.validation_loss.iloc[0], atol=2e-5, rtol=2e-5)
            abcount += 1
        else:
            returns, probabilities, volatility = decoded(output, saved['preprocess'])
            part = predictions[(predictions.ticker == ticker) & (predictions.fold == fold)
                               & (predictions.seed == saved['seed']) & (predictions.model == names[saved['mode']])]
            for column, values in [('prediction', returns), ('probability', probabilities), ('predicted_vol', volatility)]:
                expected = part.pivot(index='calendar_index', columns='horizon', values=column).reindex(rows).to_numpy()
                np.testing.assert_allclose(values, expected, atol=2e-5, rtol=2e-5)
                max_error = max(max_error, float(np.max(abs(values-expected))))
            count += 1
            if device == 'cuda':
                model.to('cpu'); args.device = 'cpu'; args.batch_size = 8
                cpu_output, _ = infer(model, data, rows, args, saved['preprocess'])
                for values, reference in zip(decoded(cpu_output, saved['preprocess']), [returns, probabilities, volatility]):
                    # Cross-device TF32/float32 convolution/attention roundoff;
                    # exact GPU replay above is checked independently.
                    np.testing.assert_allclose(values, reference, atol=2e-4, rtol=2e-4)
                    cpu_error = max(cpu_error, float(np.max(abs(values-reference))))
    samples = pd.read_csv(root/'samples.csv')
    config = json.loads((root/'config.json').read_text())
    for (_, _), part in samples.groupby(['ticker', 'fold']):
        train = part[part.split == 'train'].calendar_index
        val = part[part.split == 'val'].calendar_index
        test = part[part.split == 'test'].calendar_index
        assert train.max()+20+config['embargo'] < val.min()
        assert val.max()+20+config['embargo'] < test.min()
    test = samples[samples.split == 'test']
    assert not test.duplicated(['ticker', 'date']).any()
    frozen_verified = []
    frozen = Path(source_root) if source_root else root/'raw-source'
    if not frozen.exists() and config.get('calendar_summary'):
        frozen = Path(config['summary']).parents[1]
    if frozen.exists():
        from event_data import load_daily
        for ticker in config['tickers']:
            data = load_daily(ticker, frozen/'Daily'/'Summary', frozen/'Daily'/'All',
                              frozen/'Daily'/'Foreign', frozen/'Monthly'/'Ownership', max_brokers=config['max_brokers'],
                              calendar_summary=frozen/'Daily'/'Summary'/'ADRO_Summary.csv' if config.get('calendar_summary') else None)
            cached = np.load(root/f'{ticker}-daily.npz', allow_pickle=False)
            for key in ['x', 'tokens', 'y']:
                np.testing.assert_allclose(data[key], cached[key], atol=0, rtol=0, equal_nan=True)
            for key in ['mask', 'codes', 'good']:
                np.testing.assert_array_equal(data[key], cached[key])
            data['frame'].to_csv(root/f'{ticker}-ohlcv-snapshot.csv', index_label='Date')
            frozen_verified.append(ticker)
    result = dict(prediction_checkpoints_replayed=count, ablation_checkpoints_replayed=abcount,
                  replay_device=device, maximum_replay_vs_export_error=max_error,
                  maximum_cpu_vs_gpu_error=cpu_error, purge_and_embargo_verified=True,
                  disjoint_test_queries=True, note='trusted local checkpoints, cached inputs; not independent validation of vendor history')
    result['raw_source_snapshot_rebuild_matches_cached_arrays_exactly'] = frozen_verified
    (root/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--source-root', default=None, help='Relocated raw-source snapshot after extracting an archive')
    args = parser.parse_args()
    verify(args.root, args.source_root)
