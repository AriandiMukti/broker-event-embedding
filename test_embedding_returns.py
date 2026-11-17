import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from embedding_returns import (make_return_folds,samples,price_features,train_encoder,rolling,
    select_head,head_predict,fit_preprocessor,transform,CONTROLS)
from return_results import bootstrap,classify,illustrative_trades
import test_broker_tokens as fixtures


class ReturnTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_purging_keeps_all_target_ends_before_next_split(self):
        loaded=fixtures.BrokerTokenTests().fixture();seen=set();count=0
        for fold in make_return_folds(208):
            ids,split,y=samples(loaded,fold)
            self.assertLess((ids[split=='train']+10).max(),ids[split=='val'].min())
            self.assertLess((ids[split=='val']+10).max(),ids[split=='test'].min())
            test=set(ids[split=='test']);self.assertFalse(seen&test);seen |= test
            count+=len(test)
            for j,h in enumerate([1,5,10]):
                np.testing.assert_allclose(y[:,j],np.log(loaded[0].Close.iloc[ids+h].to_numpy()/loaded[0].Close.iloc[ids].to_numpy()))
        self.assertEqual(count,74)

    def test_future_prices_and_brokers_cannot_change_encoder(self):
        loaded=fixtures.BrokerTokenTests().fixture();fold=make_return_folds(208)[0]
        changed=copy.deepcopy(loaded)
        changed[0].loc[changed[0].index[fold['test_start']:],['Open','High','Low','Close']]*=3
        for snapshot in changed[4][fold['test_start']:]:snapshot['values']['ZZ']=1e18
        before=train_encoder(loaded,fold,'ADRO',42,epochs=5,patience=3)
        after=train_encoder(changed,fold,'ADRO',42,epochs=5,patience=3)
        self.assertEqual(before[0],after[0]);self.assertEqual(before[-1],after[-1])
        self.assertNotIn('ZZ',before[0]['brokers'])
        for key,value in before[3].state_dict().items():
            torch.testing.assert_close(value,after[3].state_dict()[key],atol=0,rtol=0)
        last=before[-1]['last_embedding_validation_date']
        self.assertLessEqual(pd.Timestamp(last),loaded[0].index[fold['train_stop']-1])

    def test_future_test_targets_and_inputs_cannot_change_selected_head(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(80,4));target=rng.normal(0,.03,80)
        ids=np.arange(80);split=np.array(['train']*40+['val']*20+['test']*20)
        with threadpool_limits(limits=1):
            before=select_head({(): (x,['a','b','c','d'])},ids,split,target)
            altered=x.copy();altered[split=='test']*=1e8
            changed=target.copy();changed[split=='test']+=1e5
            after=select_head({(): (altered,['a','b','c','d'])},ids,split,changed)
        for key in ['alpha','layout','validation_mae','intercept','features']:self.assertEqual(before[key],after[key])
        for key in ['imputation','center','scale','coefficient']:np.testing.assert_array_equal(before[key],after[key])

    def test_rolling_and_price_features_are_causal_with_exact_40_session_return(self):
        loaded=fixtures.BrokerTokenTests().fixture();changed=copy.deepcopy(loaded)
        changed[0].iloc[150:]*=100
        before=price_features(loaded);after=price_features(changed)
        np.testing.assert_allclose(before[:150],after[:150],equal_nan=True)
        self.assertAlmostEqual(before[100,3],np.log(loaded[0].Close.iloc[100]/loaded[0].Close.iloc[60]))
        values=np.arange(30.)[:,None];r=rolling(values,5)
        self.assertTrue(np.isnan(r[:4]).all());self.assertEqual(r[4,0],2.)
        altered=values.copy();altered[20:]=1e9
        np.testing.assert_allclose(rolling(values,5)[:20],rolling(altered,5)[:20],equal_nan=True)

    def test_ridge_can_recover_a_real_synthetic_predictive_signal(self):
        rng=np.random.default_rng(33);x=rng.normal(size=(100,3));y=.02*x[:,0]-.01*x[:,1]
        split=np.array(['train']*60+['val']*20+['test']*20);ids=np.arange(100)
        with threadpool_limits(limits=1):head=select_head({(): (x,['a','b','c'])},ids,split,y)
        error=np.abs(head_predict(x[split=='test'],head)-y[split=='test']).mean()
        self.assertLess(error,np.abs(y[split=='test']).mean()*.05)

    def test_imputation_and_scale_use_training_only(self):
        x=np.array([[1.,np.nan],[3.,np.nan],[5.,np.nan]])
        state=fit_preprocessor(x);self.assertEqual(state['imputation'].tolist(),[3.,0.])
        z=transform(np.array([[1e9,4.]]),state)
        self.assertTrue(np.isfinite(z).all());self.assertEqual(state['imputation'].tolist(),[3.,0.])

    def test_joint_intervals_and_classification_cannot_treat_zero_gain_as_success(self):
        result=bootstrap(np.zeros((74,6)),10,repeats=1000)
        np.testing.assert_array_equal(result['low'],0)
        rows=pd.DataFrame([dict(comparator=c,block_sessions=b,sim_ci_low=0.,nominal_ci_low=0.)
                           for c in CONTROLS for b in [10,15,20]])
        folds=pd.DataFrame({c:[0.,0.,0.] for c in CONTROLS})
        self.assertEqual(classify(rows,folds),'belum')
        rows[['sim_ci_low','nominal_ci_low']]=.01
        folds.loc[:,:]=.02
        self.assertEqual(classify(rows,folds),'didukung')
        rows.loc[0,'sim_ci_low']=-.01
        self.assertEqual(classify(rows,folds),'indikasi')

    def test_illustrative_execution_uses_next_open_fees_and_nonoverlapping_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);dates=pd.bdate_range('2026-01-01',periods=10).strftime('%Y-%m-%d').to_numpy(dtype='U10')
            np.savez_compressed(folder/'source_snapshot.npz',dates=dates,open=np.full(10,110.),close=np.full(10,100.),good=np.ones(10,dtype=bool))
            pd.DataFrame([dict(horizon=3,method='learned_embedding',feature_index=t,prediction=.02) for t in range(8)]).to_csv(folder/'test_predictions.csv',index=False)
            trades,summary=illustrative_trades(folder,3,'learned_embedding')
        self.assertEqual([t['entry_index'] for t in trades],[1,4,7])
        self.assertEqual([t['exit_index'] for t in trades],[3,6,9])
        self.assertAlmostEqual(trades[0]['net_return'],(100/110)*.998*.997-1)
        self.assertLess(summary['net_compound_return'],0)


if __name__=='__main__':unittest.main()
