"""Synthetic fixtures are unit-test only; production runners load real files."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from event_models import EventModel, CausalBlock, LegacyCNN
from event_data import make_folds, indices, preprocess, ownership_asof, external_asof, Windows
from event_training import fit, infer, decoded
from event_evaluation import temporal_permutation, backtest


def fixture(n=208):
    rng = np.random.default_rng(12)
    dates = pd.bdate_range('2025-01-01', periods=n)
    close = np.exp(np.cumsum(rng.normal(0, .01, n)))*100
    x = rng.normal(size=(n, 4)).astype('float32')
    y = rng.normal(size=(n, 6)).astype('float32'); y[:, :3] *= .04
    mask = np.ones((n, 3), bool); mask[:, -1] = False
    codes = np.tile(np.array(['AA', 'BB', '']), (n, 1))
    tokens = rng.normal(size=(n, 3, 8)).astype('float32')*mask[..., None]
    return dict(ticker='TEST', dates=dates, x=x, y=y, tokens=tokens, codes=codes, mask=mask,
                frame=pd.DataFrame(dict(Open=close, High=close+1, Low=close-1, Close=close), index=dates),
                good=np.ones(n, bool), groups={'ohlcv': [0, 1], 'broker': [2, 3]},
                names=['a', 'b', 'c', 'd'], source_hash='unit-test-only')


class EventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_set_invariance_padding_empty_and_all_modes_backprop(self):
        torch.manual_seed(42)
        x = torch.randn(2, 5, 4); tokens = torch.randn(2, 5, 4, 8)
        ids = torch.tensor([2, 3, 0, 0]).expand(2, 5, 4)
        mask = ids > 0; mask[0, 0] = False
        permutation = [3, 1, 0, 2]
        for mode in ['none', 'static', 'dynamic']:
            model = EventModel(4, 8, 3, {'ohlcv': [0, 1], 'broker': [2, 3]}, mode, 8).eval()
            actual, gates = model(x, tokens, ids, mask)
            swapped = model(x, tokens[:, :, permutation], ids[:, :, permutation], mask[:, :, permutation])[0]
            torch.testing.assert_close(actual, swapped, atol=1e-5, rtol=1e-5)
            dirty = tokens.clone(); dirty[~mask] = 1e5
            torch.testing.assert_close(actual, model(x, dirty, ids, mask)[0], atol=1e-5, rtol=1e-5)
            extra = torch.cat([tokens, torch.randn(2, 5, 2, 8)], 2)
            expanded = model(x, extra, torch.cat([ids, torch.zeros(2, 5, 2, dtype=torch.long)], 2),
                             torch.cat([mask, torch.zeros(2, 5, 2, dtype=torch.bool)], 2))[0]
            torch.testing.assert_close(actual, expanded, atol=1e-5, rtol=1e-5)
            empty = model(x, tokens, ids, torch.zeros_like(mask))[0]
            self.assertTrue(torch.isfinite(empty).all())
            actual.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(gates.shape, (2, 5, 2))

    def test_tcn_is_strictly_causal(self):
        block = CausalBlock(8, 4).eval()
        x = torch.randn(2, 8, 20); changed = x.clone(); changed[:, :, 12:] += 100
        torch.testing.assert_close(block(x)[:, :, :12], block(changed)[:, :, :12])

    def test_purge_plus_embargo_and_disjoint_tests(self):
        folds = make_folds(208)
        data = fixture(); alltest = []
        for fold in folds:
            split = indices(data, fold, 20)
            self.assertLess(max(split['train'])+20+20, min(split['val']))
            self.assertLess(max(split['val'])+20+20, min(split['test']))
            alltest.extend(split['test'])
        self.assertEqual(len(alltest), len(set(alltest)))
        with self.assertRaises(ValueError):
            make_folds(120)

    def test_scalers_ids_constant_decisions_and_targets_ignore_future(self):
        raw = fixture(); split = indices(raw, make_folds(208)[0], 20)
        raw['x'][:, 0] = 0; raw['x'][150:, 0] = 42
        prepared, state = preprocess(raw, split['train'], 20)
        self.assertTrue(state['constant_columns'][0]); self.assertTrue((prepared['x'][:, 0] == 0).all())
        other = copy.deepcopy(raw); other['x'][120:] = 999; other['tokens'][120:] = 999
        other['codes'][120:, 0] = 'ZZ'; other['y'][120:] = 999
        _, changed = preprocess(other, split['train'], 20)
        for key in ['mean', 'std', 'target_mean', 'target_std', 'token_mean', 'token_std', 'constant_columns']:
            np.testing.assert_array_equal(state[key], changed[key])
        self.assertEqual(state['vocabulary'], changed['vocabulary'])
        self.assertNotIn('ZZ', state['vocabulary'])
        from event_predict import transform
        restored = transform(raw, state)
        for key in ['x', 'tokens', 'ids']:
            np.testing.assert_array_equal(prepared[key], restored[key])
        ablated = Windows(prepared, split['val'], 20, state, 'broker')[0]
        self.assertFalse(ablated[3].any()); self.assertFalse(ablated[2].any()); self.assertFalse(ablated[1].any())

    def test_ownership_waits_for_fetch_and_release_not_position_month(self):
        calendar = pd.bdate_range('2026-08-01', '2026-09-08')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'lbrpe'/'2026-09'; folder.mkdir(parents=True)
            (folder/'ADRO.json').write_text(json.dumps(dict(free_float=25, sid=10000,
                period='July', announced_at='2026-08-10T20:00:00', fetched_at='2026-09-03T20:00:00')))
            value, _, _ = ownership_asof(tmp, 'ADRO', calendar)
            before = calendar < pd.Timestamp('2026-09-03')
            self.assertTrue((value[before, 1] == 1).all())
            self.assertTrue((value[~before, 0] == .25).all())
            self.assertTrue((value[~before, 1] == 0).all())

    def test_external_uses_availability_and_never_backfills_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'macro.csv'
            pd.DataFrame(dict(available_at=['2026-01-05T18:00:00', '2026-01-07T18:00:00'],
                              ihsg_close=[7000, 7100])).to_csv(path, index=False)
            dates = pd.bdate_range('2026-01-02', '2026-01-08')
            x, _, groups = external_asof(path, dates)
            self.assertEqual(x[0, 1], 1)
            self.assertAlmostEqual(x[3, 0], np.log(7100/7000), places=6)
            self.assertEqual(set(groups), {'ihsg'})

    def test_temporal_permutation_moves_one_mapping_and_preserves_other_groups(self):
        raw = fixture(); prepared, state = preprocess(raw, np.arange(59, 80), 20)
        rows = np.arange(119, 127)
        for i in range(208):
            prepared['x'][i, 2] = i; prepared['tokens'][i, 0, 0] = i; prepared['ids'][i, 0] = i
        result = temporal_permutation(prepared, rows, 20, 'broker', 42)
        np.testing.assert_array_equal(result['x'][:, :2], prepared['x'][:, :2])
        np.testing.assert_array_equal(result['x'][:, 2], result['tokens'][:, 0, 0])
        np.testing.assert_array_equal(result['x'][:, 2], result['ids'][:, 0])
        self.assertFalse(np.array_equal(result['x'][:, 2], prepared['x'][:, 2]))

    def test_execution_next_open_costs_no_overlap_and_constant_equity_cash(self):
        dates = pd.bdate_range('2025-01-01', periods=20)
        prices = np.arange(100., 120.)
        frame = pd.DataFrame(dict(Open=prices, High=prices+1, Low=prices-1, Close=prices), index=dates)
        signals = pd.DataFrame(dict(calendar_index=np.arange(10), probability=.7, prediction=.01))
        stats, trades, curve = backtest(frame, signals, 5, 15, 10)
        self.assertEqual(trades[0]['entry_index'], 1); self.assertEqual(trades[0]['exit_index'], 6)
        self.assertGreaterEqual(trades[1]['entry_index'], trades[0]['exit_index'])
        expected = 106/101*(1-.0025)**2-1
        self.assertAlmostEqual(trades[0]['net_return'], expected)
        self.assertEqual(stats['turnover'], 2*len(trades))
        factor = np.prod([1+t['net_return'] for t in trades])
        self.assertAlmostEqual(curve[-1]['equity'], factor)
        signals.probability = .2
        empty, trades, curve = backtest(frame, signals, 5)
        self.assertEqual(empty['trades'], 0); self.assertIsNone(empty['sharpe'])
        self.assertTrue(all(row['equity'] == 1 for row in curve))

    def test_test_changes_do_not_change_fit_checkpoint_and_checkpoint_replays(self):
        raw = fixture(); split = indices(raw, make_folds(208)[0], 20)
        data, state = preprocess(raw, split['train'], 20)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(window=20, batch_size=8, device='cpu', width=8, lr=.001,
                                   no_amp=True, epochs=2, patience=2, out=tmp)
            model, record, _, _ = fit(data, split, state, args, 'dynamic', 42, Path(tmp)/'a')
            changed = copy.deepcopy(data)
            changed['x'][split['test']] = 1e3; changed['y'][split['test']] = 1e3
            other, _, _, _ = fit(changed, split, state, args, 'dynamic', 42, Path(tmp)/'b')
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
            saved = torch.load(Path(tmp)/'a'/'best.pt', weights_only=False)
            replay = EventModel(4, 8, len(state['vocabulary']), raw['groups'], 'dynamic', 8)
            replay.load_state_dict(saved['state_dict'])
            actual, _ = infer(model, data, split['test'], args, state)
            exported, _ = infer(replay, data, split['test'], args, state)
            np.testing.assert_allclose(actual, exported)
            r, p, v = decoded(actual, state)
            self.assertTrue(np.isfinite(v).all()); self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_real_cached_targets_and_source_calendar(self):
        root = Path('runs/event-v1-20261116')
        if not (root/'config.json').exists():
            self.skipTest('Real event run has not been executed')
        config = json.loads((root/'config.json').read_text())
        for ticker in config['tickers']:
            cached = np.load(root/f'{ticker}-daily.npz', allow_pickle=False)
            audit = json.loads((root/f'{ticker}-data-audit.json').read_text())
            dates = pd.DatetimeIndex(audit['dates'])
            self.assertGreaterEqual(dates.min(), pd.Timestamp('2026-01-02'))
            # Verify frozen real inputs, not a live CSV which may be appended
            # or revised by the independent daily downloader during this test.
            ret = cached['x'][:, audit['features'].index('return')]
            for i in [59, 80, 119, 167, 187]:
                for j, h in enumerate([5, 10, 20]):
                    self.assertAlmostEqual(float(cached['y'][i, j]), float(ret[i+1:i+h+1].sum()), places=6)
                    vol = np.sqrt(np.mean(ret[i+1:i+h+1]**2)*252)
                    self.assertAlmostEqual(float(cached['y'][i, j+3]), np.log(vol), places=6)


if __name__ == '__main__':
    unittest.main()
