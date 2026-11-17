"""Objective isolation and checkpoint-selection contracts; synthetic fixtures only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from event_data import indices,make_folds,preprocess
from event_training import task_loss,selection_loss,fit,infer
from event_predict import restore
from test_event_research import fixture
from event_objective_study import statistics


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_return_only_ignores_auxiliary_labels_and_has_zero_auxiliary_gradient(self):
        out=torch.arange(27,dtype=torch.float32).reshape(3,9).requires_grad_()
        target=torch.ones(3,6);positive=torch.zeros(3,3)
        value=task_loss(out,target,positive,'return_only')
        other=target.clone();other[:,3:]=1000
        torch.testing.assert_close(value,task_loss(out,other,1-positive,'return_only'))
        value.backward()
        self.assertTrue((out.grad[:,3:]==0).all());self.assertTrue((out.grad[:,:3]!=0).any())

    def test_multitask_responds_to_auxiliary_labels(self):
        out=torch.ones(3,9,requires_grad=True);target=torch.zeros(3,6);positive=torch.zeros(3,3)
        value=task_loss(out,target,positive,'multitask');value.backward()
        self.assertTrue((out.grad[:,3:]!=0).all())
        self.assertNotEqual(float(value.detach()),float(task_loss(out,target,1-positive,'multitask').detach()))

    def test_common_selection_is_raw_return_mae(self):
        out=torch.tensor([[1.,2.,3.,100.,100.,100.,-9.,-9.,-9.]])
        target=torch.zeros(1,6);std=torch.tensor([.01,.1,1.,9.,9.,9.])
        loss=selection_loss(out,target,torch.zeros(1,3),std,'return_mae')
        self.assertAlmostEqual(float(loss),(.01+.2+3)/3,places=6)

    def test_invalid_objective_and_untrained_auxiliary_selection_rejected(self):
        with self.assertRaises(ValueError):task_loss(torch.zeros(1,9),torch.zeros(1,6),torch.zeros(1,3),'bad')
        raw=fixture();split=indices(raw,make_folds(208)[0],20);data,state=preprocess(raw,split['train'],20)
        args=SimpleNamespace(training_objective='return_only',selection_metric='joint_loss')
        with self.assertRaises(ValueError):fit(data,split,state,args,'none',11,'unused')

    def test_paired_initialization_replay_and_test_labels_do_not_select_checkpoint(self):
        raw=fixture();split=indices(raw,make_folds(208)[0],20);data,state=preprocess(raw,split['train'],20)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(window=20,batch_size=8,device='cpu',width=8,lr=.001,no_amp=True,
                epochs=2,patience=2,out=tmp,selection_metric='return_mae',log_grad_norm=True)
            records={}
            for task in ['multitask','return_only']:
                args.training_objective=task
                model,record,history,_=fit(data,split,state,args,'none',11,Path(tmp)/task)
                records[task]=record
                self.assertAlmostEqual(record['val_loss'],min(row['val_return_mae'] for row in history),places=5)
                self.assertTrue(all(row['nonfinite_gradient_steps']==0 for row in history))
            self.assertEqual(records['return_only']['initial_weights_sha256'],records['multitask']['initial_weights_sha256'])
            changed=copy.deepcopy(data);changed['x'][split['test']]=1e3;changed['y'][split['test']]=1e3
            other,record,_,_=fit(changed,split,state,args,'none',11,Path(tmp)/'changed')
            self.assertEqual(record['best_weights_sha256'],records['return_only']['best_weights_sha256'])
            restored,saved=restore(Path(tmp)/'return_only'/'best.pt')
            self.assertEqual(saved['training_objective'],'return_only')
            np.testing.assert_array_equal(infer(model,data,split['test'],args,state)[0],
                                          infer(restored,data,split['test'],args,state)[0])

    def statistic_fixture(self,root,missing=False):
        rows=[]
        for date in pd.bdate_range('2026-01-01',periods=6).strftime('%Y-%m-%d'):
            for task,predictions,backbone in [('return_only',[-1.,1.],'none'),('multitask',[.25,.25],'none'),
                                              ('zero',[0.],'baseline'),('train_mean',[.1],'baseline')]:
                if missing and task=='return_only' and date=='2026-01-01':continue
                for seed,prediction in enumerate(predictions):
                    rows.append(dict(date=date,ticker='TEST',horizon=5,fold=1,seed=seed,
                        backbone=backbone,training_objective=task,actual=0.,prediction=prediction))
        frame=pd.DataFrame(rows);frame.to_csv(root/'predictions.csv',index=False)
        frame['mae']=abs(frame.actual-frame.prediction)
        frame.groupby(['ticker','horizon','fold','seed','backbone','training_objective']).mae.mean().reset_index().to_csv(root/'metrics.csv',index=False)
        (root/'config.json').write_text(json.dumps(dict(modes=['none'],bootstrap_repeats=50)))

    def test_paired_statistics_average_seed_losses_not_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.statistic_fixture(root);statistics(root)
            bands=pd.read_csv(root/'paired_bands.csv')
            row=bands[(bands.treatment=='return_only')&(bands.comparator=='multitask')&(bands.ticker=='ALL')].iloc[0]
            self.assertAlmostEqual(row.gain_mae,-.75);self.assertEqual(row.family_columns,10)

    def test_paired_statistics_reject_unequal_date_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.statistic_fixture(root,missing=True)
            with self.assertRaisesRegex(ValueError,'Unequal date coverage'):statistics(root)


if __name__=='__main__':unittest.main()
