"""Frozen representation/temporal contracts; synthetic fixtures only."""
import copy
import unittest
import numpy as np
import pandas as pd
import torch
from event_models import SetEncoder
from event_broker_geometry import encoded,state_labels,select_brokers,subset_pool
from event_broker_probe import query_splits,fit_head,probabilities


def raw_fixture(n=140):
    rng=np.random.default_rng(17);codes=np.tile(np.array(['AA','BB','CC','DD']),(n,1))
    tok=rng.normal(size=(n,4,8)).astype('float32')
    tok[:,:,0]=rng.choice([-1,1],(n,4))*rng.uniform(10,20,(n,4))
    return dict(dates=pd.bdate_range('2020-01-01',periods=n),codes=codes,tokens=tok,mask=np.ones((n,4),bool))


class BrokerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_manual_encoder_matches_existing_forward_padding_and_empty(self):
        enc=SetEncoder(4,8,width=8).eval();t=torch.randn(3,4,8)
        ids=torch.tensor([[2,3,4,5]]*3);mask=torch.tensor([[1,1,0,0],[0,0,0,0],[1,1,1,1]],dtype=torch.bool)
        _,p=encoded(enc,t,ids,mask)
        torch.testing.assert_close(p,enc(t[:,None],ids[:,None],mask[:,None])[:,0],atol=1e-6,rtol=1e-6)
        self.assertTrue((p[1]==0).all())

    def test_masked_target_cannot_change_peer_representation(self):
        raw=raw_fixture();codes=['AA','BB','CC','DD']
        state=dict(vocabulary=codes,token_mean=np.zeros(8),token_std=np.ones(8))
        enc=SetEncoder(4,8,width=8).eval();rows=np.arange(10)
        a=subset_pool(enc,raw['tokens'],raw['mask'],codes,state,rows,0,'cpu')
        changed=raw['tokens'].copy();changed[:,0]=1e6
        b=subset_pool(enc,changed,raw['mask'],codes,state,rows,0,'cpu')
        np.testing.assert_array_equal(a,b)

    def test_broker_selection_and_state_cuts_ignore_future(self):
        raw=raw_fixture();train=np.arange(80);codes,_=select_brokers(raw,train,count=4)
        states,cuts,_,_=state_labels(raw,codes,train)
        other=copy.deepcopy(raw);other['tokens'][90:]=1e6;other['codes'][90:,0]='ZZ'
        new,_=select_brokers(other,train,count=4);self.assertEqual(codes,new)
        altered,cut2,_,_=state_labels(other,codes,train)
        np.testing.assert_array_equal(cuts,cut2);np.testing.assert_array_equal(states[:90],altered[:90])

    def test_next_day_calendar_gap_is_not_compressed_or_cross_split(self):
        raw=raw_fixture(200);raw['mask'][110]=False
        data={key+'_indices':np.arange(a,b) for key,a,b in [('train',10,90),('val',100,130),('test',150,190)]}
        splits=query_splits(data,raw)
        self.assertNotIn(109,splits['val']);self.assertNotIn(110,splits['val'])
        for key,rows in splits.items():
            self.assertTrue(set(rows+1)<=set(data[key+'_indices']))
            self.assertNotIn(data[key+'_indices'][-1],rows)

    def test_head_selection_and_scaler_ignore_test_and_unseen_class_probs_normalize(self):
        rng=np.random.default_rng(41);x=rng.normal(size=(170,6));y=rng.integers(0,3,170)
        split=np.array(['train']*100+['val']*30+['test']*40)
        a,trials=fit_head(x,y,split);changed=x.copy();changed[split=='test']=999
        labels=y.copy();labels[split=='test']=4
        b,_=fit_head(changed,labels,split)
        np.testing.assert_array_equal(a['scaler'].mean_,b['scaler'].mean_)
        np.testing.assert_array_equal(a['classifier'].coef_,b['classifier'].coef_)
        self.assertEqual(a['C'],b['C'])
        p=probabilities(a,x);np.testing.assert_allclose(p.sum(1),1)
        self.assertTrue((p>0).all());self.assertEqual(p.shape,(170,5))


if __name__=='__main__':unittest.main()
