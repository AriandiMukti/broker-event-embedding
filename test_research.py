import unittest
from pathlib import Path
import json
import numpy as np
import torch

from research import make_folds,fold_vocabulary,normalize_fold,fit_variant,PRICE_FEATURES,VARIANTS


class ResearchTests(unittest.TestCase):
    def test_folds_have_purged_boundaries_and_nonoverlapping_tests(self):
        folds=make_folds(208)
        self.assertEqual([f['train_end']-59+1 for f in folds],[37,51,65])
        dates=[]
        for f in folds:
            self.assertLess(f['train_end']+20,f['val_start'])
            last_val=f['test_start']-20-1
            self.assertLess(last_val+20,f['test_start'])
            dates.extend(range(f['test_start'],f['test_stop']))
        self.assertEqual(len(dates),42)
        self.assertEqual(len(set(dates)),42)
        self.assertEqual(max(dates)+20,207)

    def test_future_broker_code_is_not_in_early_fold_vocabulary(self):
        cache=[{'AA':1} for _ in range(208)]
        cache[100]={'ZZ':1}
        folds=make_folds(208)
        self.assertEqual(fold_vocabulary(cache,folds[0]['train_end']),['AA'])
        self.assertEqual(fold_vocabulary(cache,folds[1]['train_end']),['AA','ZZ'])

    def test_scaler_is_unchanged_by_future_data(self):
        rng=np.random.default_rng(5)
        x=rng.normal(size=(12,20,5)).astype('float32');y=rng.normal(size=(12,3)).astype('float32')
        masks={'train':np.arange(12)<6,'val':(np.arange(12)>=6)&(np.arange(12)<9),'test':np.arange(12)>=9}
        original=normalize_fold(x,y,masks)
        x[6:]*=100;y[6:]*=100
        changed=normalize_fold(x,y,masks)
        for before,after in zip(original[2:],changed[2:]):np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(original[0][:6],changed[0][:6])

    def test_test_labels_and_disabled_features_do_not_affect_training(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(1);xn=rng.normal(size=(14,20,6)).astype('float32')
        yn=rng.normal(size=(14,3)).astype('float32')
        masks={'train':np.arange(14)<8,'val':(np.arange(14)>=8)&(np.arange(14)<11),'test':np.arange(14)>=11}
        allowed=np.array([True,True,False,False,False,False])
        p,r=fit_variant(xn,yn,yn,masks,allowed,42,2,torch.device('cpu'))
        changed=xn.copy();changed[:,:,~allowed]*=1000
        y=yn.copy();y[masks['test']]=1000
        p2,r2=fit_variant(changed,y,y,masks,allowed,42,2,torch.device('cpu'))
        np.testing.assert_array_equal(p,p2)
        self.assertEqual(r['best_weights_sha256'],r2['best_weights_sha256'])
        _,r3=fit_variant(xn,yn,yn,masks,~allowed,42,2,torch.device('cpu'))
        self.assertEqual(r['initial_weights_sha256'],r3['initial_weights_sha256'])
        self.assertEqual(r['parameters'],r3['parameters'])

    def test_completed_research_integrity(self):
        root=Path('runs/research-v1')
        if not (root/'config.json').exists():self.skipTest('Complete research run first')
        config=json.loads((root/'config.json').read_text())
        audits=json.loads((root/'fold_audit.json').read_text())
        records=json.loads((root/'runs.json').read_text())
        self.assertEqual(config['fitted_models'],270)
        self.assertEqual(len(records),270)
        for a in audits:
            train=[s for s in a['samples'] if s['split']=='train']
            val=[s for s in a['samples'] if s['split']=='val']
            test=[s for s in a['samples'] if s['split']=='test']
            self.assertTrue(all(s['label_end']<a['val_start'] for s in train))
            self.assertTrue(all(s['label_end']<a['test_start'] for s in val))
            self.assertTrue(all(s['date']>=a['test_start'] for s in test))
            self.assertLessEqual(a['vocabulary_last_input_date'],max(s['date'] for s in train))
            self.assertEqual(a['feature_counts']['ohlcv'],11)
            paired=[r for r in records if r['ticker']==a['ticker'] and r['fold']==a['fold']]
            for seed in config['seeds']:
                subset=[r for r in paired if r['seed']==seed]
                self.assertEqual({r['variant'] for r in subset},set(VARIANTS))
                self.assertEqual(len({r['initial_weights_sha256'] for r in subset}),1)
                self.assertEqual(len({r['parameters'] for r in subset}),1)


if __name__=='__main__':unittest.main()
