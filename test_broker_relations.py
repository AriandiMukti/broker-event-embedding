import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from broker_relation_test import (relation_tokens, dataset, folds, mask_inputs,
    shift_peers, fit, predict, baseline, losses, block_bootstrap, STATES)
import test_broker_tokens as fixture_module


class RelationTests(unittest.TestCase):
    def test_masked_target_is_absent_and_aggregates_excluded(self):
        loaded = fixture_module.BrokerTokenTests().fixture()
        _, daily, states, valid = relation_tokens(loaded, 103, 2)
        fold = folds(len(daily))[0]
        for task in ['same_day_masked', 'next_day_peers']:
            base, y, prev, targets, source, split = dataset(daily, states, valid, fold, task)
            self.assertEqual(base.shape[1], 2+6)
            self.assertTrue(np.all(targets-source == (0 if task == 'same_day_masked' else 1)))
            x = mask_inputs(base, 2, 'full')
            for b in range(2):
                self.assertTrue((x[:, b, b] == 0).all())
                self.assertFalse(np.isin(x[:, b], np.arange(1+5*b, 6+5*b)).any())
            np.testing.assert_array_equal(y, states[targets])

    def test_target_value_cannot_change_peer_state_or_ohlcv_input(self):
        loaded = fixture_module.BrokerTokenTests().fixture()
        tokenizer, daily, _, _ = relation_tokens(loaded, 103, 2)
        changed = copy.deepcopy(loaded)
        code = tokenizer['brokers'][0]
        changed[4][170]['values'][code] = 1e15
        other, daily2, _, _ = relation_tokens(changed, 103, 2)
        self.assertEqual(tokenizer, other)
        np.testing.assert_array_equal(daily[170, 1:2+6], daily2[170, 1:2+6])
        self.assertNotIn('large_share_threshold', tokenizer)

    def test_shift_preserves_split_marginals_and_ohlcv(self):
        loaded = fixture_module.BrokerTokenTests().fixture()
        _, daily, states, valid = relation_tokens(loaded, 103, 2)
        base, _, _, _, _, split = dataset(daily, states, valid, folds(len(daily))[0], 'same_day_masked')
        shifted, audit = shift_peers(base, split, 2, 42)
        np.testing.assert_array_equal(base[:, 2:], shifted[:, 2:])
        for s in ['train', 'val', 'test']:
            for b in range(2):
                np.testing.assert_array_equal(np.sort(base[split == s, b]), np.sort(shifted[split == s, b]))
                self.assertGreaterEqual(audit[s][b], 5)
        masked = mask_inputs(shifted, 2, 'shifted_peers')
        self.assertTrue((masked[:, 0, 0] == 0).all())

    def test_folds_are_chronological_and_test_dates_disjoint(self):
        seen = set()
        for f in folds(208):
            self.assertLess(f['train_stop'], f['val_stop'])
            test = set(range(f['val_stop'], f['test_stop']))
            self.assertFalse(seen & test); seen |= test
        self.assertEqual(len(seen), 84)

    def test_test_labels_cannot_change_checkpoint(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(4)
        base = rng.integers(11, 20, (60, 8)); base[:, :2] = rng.integers(1, 11, (60, 2))
        x = mask_inputs(base, 2, 'full'); y = rng.integers(0, 5, (60, 2))
        split = np.array(['train']*30+['val']*15+['test']*15)
        model, record = fit(x, y, split, 20, 4, 42, 8, 4)
        changed = y.copy(); changed[split == 'test'] = (changed[split == 'test']+1)%5
        other, record2 = fit(x, changed, split, 20, 4, 42, 8, 4)
        self.assertEqual(record, record2)
        for key, v in model.state_dict().items():
            torch.testing.assert_close(v, other.state_dict()[key], atol=0, rtol=0)
        np.testing.assert_allclose(predict(model, x).sum(-1), 1, atol=1e-6)

    def test_fixed_embedding_really_remains_at_initial_weights(self):
        from adro_token_embedding import EventEmbedding
        torch.set_num_threads(1); torch.manual_seed(42)
        expected = EventEmbedding(11, 2, 4, 1).embedding.weight.detach().clone()
        rng = np.random.default_rng(8)
        base = np.zeros((50,8), dtype=int)
        base[:,0] = rng.integers(1,6,50); base[:,1] = rng.integers(6,11,50)
        y = rng.integers(0,5,(50,2)); split = np.array(['train']*25+['val']*15+['test']*10)
        model, _ = fit(mask_inputs(base, 2, 'full'), y, split, 11, 4, 42, 10, 4, freeze=True)
        torch.testing.assert_close(model.embedding.weight, expected, atol=0, rtol=0)
        self.assertFalse(model.embedding.weight.requires_grad)

    def test_synthetic_relation_is_learned_and_shuffle_breaks_it(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(12); n = 300
        # Two brokers always have opposite directions; no price information.
        a = rng.integers(0, 2, n)
        y = np.column_stack([np.where(a, 2, 4), np.where(a, 4, 2)])
        base = np.zeros((n, 8), dtype=int)
        base[:, 0] = 1+y[:, 0]; base[:, 1] = 6+y[:, 1]
        split = np.array(['train']*180+['val']*60+['test']*60)
        x = mask_inputs(base, 2, 'full')
        model, _ = fit(x, y, split, 11, 4, 42, 180, 25)
        p = predict(model, x[split == 'test'])
        freq, _ = baseline(y, y, split)
        real = losses(y[split == 'test'], p).mean()
        frequency = losses(y[split == 'test'], freq[split == 'test']).mean()
        shifted, _ = shift_peers(base, split, 2, 42)
        nullx = mask_inputs(shifted, 2, 'full')
        null, _ = fit(nullx, y, split, 11, 4, 42, 180, 25)
        null_loss = losses(y[split == 'test'], predict(null, nullx[split == 'test'])).mean()
        self.assertLess(real, frequency-.3)
        self.assertLess(real, null_loss-.3)

    def test_bootstrap_positive_constant_difference(self):
        result = block_bootstrap(np.full(30, .1), np.repeat([1,2,3], 10), repeats=100)
        self.assertAlmostEqual(result['gain_nll'], .1)
        self.assertAlmostEqual(result['ci_low'], .1)
        self.assertEqual(result['dates'], 30)

    def test_saved_experiment_reproduces_selected_checkpoints(self):
        from research import load_stock
        from adro_token_embedding import EventEmbedding
        root = Path('runs/adro-relations-v1')
        if not (root/'config.json').exists(): self.skipTest('Run experiment first')
        config = json.loads((root/'config.json').read_text())
        loaded = load_stock('ADRO', config['summary'], config['brokers'])
        # Reproduce the saved experiment calendar even after Summary history
        # expands. This verifies old weights; never regenerate their artefacts.
        from compact_features import align_loaded
        dates = loaded[0].index
        calendar = dates[(dates >= config['source_start']) & (dates <= config['source_end'])]
        self.assertEqual(len(calendar), config['source_days'])
        loaded = align_loaded(loaded, calendar)
        exported = pd.read_csv(root/'test_predictions.csv')
        records = json.loads((root/'training_runs.json').read_text())
        self.assertEqual(len(records), 3*2*5*len(config['seeds']))
        daily_scores = pd.read_csv(root/'daily_test_scores.csv')
        for task in ['same_day_masked', 'next_day_peers']:
            self.assertEqual(daily_scores[(daily_scores.task==task)&(daily_scores.method=='full')].date.nunique(), 84)
        for fold in config['folds']:
            tokenizer, daily, states, valid = relation_tokens(loaded, fold['train_stop'], 12)
            saved_tokenizer = json.loads((root/f"fold-{fold['fold']}"/'tokenizer.json').read_text())
            self.assertEqual(tokenizer, saved_tokenizer)
            for task in ['same_day_masked','next_day_peers']:
                base, y, _, targets, _, split = dataset(daily, states, valid, fold, task)
                checkpoint = torch.load(root/f"fold-{fold['fold']}"/f'{task}.pt', map_location='cpu', weights_only=False)
                candidates = [r for r in records if r['task']==task and r['fold']==fold['fold'] and r['method']=='full']
                self.assertEqual(checkpoint['seed'], min(candidates,key=lambda r:r['validation_nll'])['seed'])
                model = EventEmbedding(len(tokenizer['tokens']), len(tokenizer['brokers']), config['dimension'], 1)
                model.load_state_dict(checkpoint['state_dict'])
                p = predict(model, mask_inputs(base, len(tokenizer['brokers']), 'full')[split=='test'])
                rows = exported[(exported.task==task)&(exported.fold==fold['fold'])&(exported.method=='full')&(exported.seed==checkpoint['seed'])]
                np.testing.assert_allclose(rows[STATES].to_numpy().reshape(p.shape), p, atol=1e-7)
                np.testing.assert_array_equal(rows.actual.to_numpy().reshape(y[split=='test'].shape), y[split=='test'])


if __name__ == '__main__': unittest.main()
