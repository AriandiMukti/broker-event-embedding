"""Meaningful guards for result reuse, seed accounting and paired comparisons."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from event_cohort import compatible,MODEL_CODE,RECIPE
from event_diagnose import seed_losses,paired_bands,stability


class DiagnosticTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for day in pd.bdate_range('2025-01-01',periods=80):
            for model,seed,prediction in [('zero',-1,0),('train_mean',-1,.2),('set_tcn',11,0),('set_tcn',42,2)]:
                rows.append(dict(date=str(day.date()),ticker='TEST',horizon=5,model=model,seed=seed,
                    prediction=prediction,actual=1.,probability=.5,actual_vol=.5,predicted_vol=.5))
        return pd.DataFrame(rows)

    def test_seed_loss_is_not_ensemble_loss(self):
        losses=seed_losses(self.fixture())
        model=losses[losses.model=='set_tcn']
        self.assertTrue((model.mean_prediction==1).all())
        self.assertTrue((model.return_loss==1).all())
        self.assertTrue((abs(model.mean_prediction-model.actual)==0).all())

    def test_target_mismatch_duplicates_and_probability_fail_closed(self):
        base=self.fixture()
        with self.assertRaises(ValueError):seed_losses(pd.concat([base,base.iloc[:1]]))
        bad=base.copy();bad.loc[0,'actual']=9
        with self.assertRaises(ValueError):seed_losses(bad)
        bad=base.copy();bad.loc[0,'probability']=1.1
        with self.assertRaises(ValueError):seed_losses(bad)

    def test_paired_bands_cover_all_heads_controls_and_no_silent_coverage_drop(self):
        bands=paired_bands(seed_losses(self.fixture()),repeats=100)
        self.assertEqual(set(bands.metric),{'return_mae','brier','vol_mae'})
        self.assertEqual(set(bands.comparator),{'zero','train_mean'})
        p=bands[(bands.model=='set_tcn')&(bands.comparator=='zero')&(bands.metric=='return_mae')]
        np.testing.assert_allclose(p.gain,0,atol=1e-12)
        np.testing.assert_allclose(p.simultaneous_low,0,atol=1e-12)
        p=bands[(bands.model=='set_tcn')&(bands.comparator=='train_mean')&(bands.metric=='return_mae')]
        np.testing.assert_allclose(p.gain,-.2,atol=1e-12)
        frame=self.fixture();drop=(frame.date==frame.date.iloc[0])&(frame.model=='set_tcn')
        with self.assertRaises(ValueError):paired_bands(seed_losses(frame[~drop]),repeats=100)

    def test_fold_and_seed_wins_are_separate_units(self):
        rows=[]
        for fold in [1,2,3]:
            rows.append(dict(ticker='TEST',horizon=5,model='zero',seed=-1,fold=fold,mae=.1,brier=.25,vol_mae=.1))
            rows.append(dict(ticker='TEST',horizon=5,model='train_mean',seed=-1,fold=fold,mae=.12,brier=.26,vol_mae=.12))
            for seed in [11,42,123]:
                value=.05 if fold<3 else .2
                rows.append(dict(ticker='TEST',horizon=5,model='set_tcn',seed=seed,fold=fold,mae=value,brier=value,vol_mae=value))
        p=stability(pd.DataFrame(rows))
        r=p[(p.model=='set_tcn')&(p.comparator=='zero')&(p.metric=='mae')].iloc[0]
        self.assertEqual(r.folds_won,2);self.assertEqual(r.folds,3);self.assertEqual(r.seeds,3)

    def test_reuse_rejects_changed_recipe_and_changed_model_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';recipe={key:1 for key in RECIPE}
            for root in [a,b]:
                (root/'sources').mkdir(parents=True)
                (root/'config.json').write_text(json.dumps(recipe))
                for name in MODEL_CODE:(root/'sources'/name).write_text('same model')
            compatible(a,b)
            changed=dict(recipe,batch_size=99);(b/'config.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):compatible(a,b)
            (b/'config.json').write_text(json.dumps(recipe));(b/'sources'/MODEL_CODE[0]).write_text('different preprocessing')
            with self.assertRaises(ValueError):compatible(a,b)


if __name__=='__main__':unittest.main()
