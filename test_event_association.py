"""Leakage and feature-control tests; fixtures are only for unit testing."""
import unittest
import numpy as np
import torch
from event_association_study import local_peer_tokens,linear_features,prepare_data,assert_reuse
from event_relation_model import RelationModel
from test_event_relation_e2e import fixture


class AssociationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_target_local_mutation_does_not_change_other_peers(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(40,4,8)).astype('float32');m=np.ones((40,4),bool)
        a=local_peer_tokens(x,m);changed=x.copy();changed[:,1]=100
        b=local_peer_tokens(changed,m)
        np.testing.assert_array_equal(a[:,[0,2,3]],b[:,[0,2,3]])
        shared=x.copy();shared[:,:,[1,5,6,7]]=1e5
        np.testing.assert_array_equal(a,local_peer_tokens(shared,m))

    def test_temporal_features_never_look_forward_and_missing_is_zero(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(40,4,8)).astype('float32');m=np.ones((40,4),bool);m[4,2]=False
        a=local_peer_tokens(x,m);x[20:]=1e4;b=local_peer_tokens(x,m)
        np.testing.assert_array_equal(a[:20],b[:20]);np.testing.assert_array_equal(a[4,2],np.zeros(8))
        np.testing.assert_allclose(a[7,:,4],x[7,:,0]-x[6,:,0])

    def test_both_linear_controls_mask_target_entire_history(self):
        data=fixture();data['peer_net_log']=data['tokens'][...,0].copy()
        for method in ['raw_linear','aggregate_linear']:
            before=linear_features(data,1,method);changed={k:v.copy() for k,v in data.items()}
            changed['tokens'][:,:,1]=1000;changed['mask'][:,:,1]=~changed['mask'][:,:,1]
            changed['peer_net_log'][:,:,1]=1000
            np.testing.assert_array_equal(before,linear_features(changed,1,method))
            expected=np.column_stack([data['own'][:,1,::-1].reshape(60,70),data['context']])
            np.testing.assert_array_equal(before[:,:88],expected)

    def test_empty_peer_set_is_finite_for_linear_and_neural(self):
        data=fixture();data['peer_net_log']=data['tokens'][...,0].copy();data['mask'][:]=False
        for method in ['raw_linear','aggregate_linear']:
            self.assertTrue(np.isfinite(linear_features(data,0,method)).all())
        model=RelationModel(4,width=8).eval()
        tokens=torch.as_tensor(data['tokens'][:2,None].repeat(4,1))
        mask=torch.zeros((2,4,5,4),dtype=torch.bool);ids=torch.as_tensor(data['ids'])
        self.assertTrue((model.peer_representation(tokens,mask,ids)==0).all())

    def test_preparation_has_matching_reuse_and_train_only_scaler(self):
        old=fixture();old['queries']=np.arange(5,65)
        rng=np.random.default_rng(6);n=70;b=4
        raw=dict(tokens=rng.normal(size=(n,b,8)).astype('float32'),mask=np.ones((n,b),bool),
                 codes=np.broadcast_to(old['codes'],(n,b)).copy(),dates=np.arange(n))
        train=np.arange(35);a,sa=prepare_data(old,raw,train);assert_reuse(old,a)
        raw['tokens'][50:]=10000;b,sb=prepare_data(old,raw,train)
        self.assertEqual(sa,sb);np.testing.assert_array_equal(a['tokens'][:40],b['tokens'][:40])
        np.testing.assert_array_equal(a['queries'],old['queries']+1)
        self.assertEqual(a['tokens'].shape,(60,5,4,8))

    def test_reuse_rejects_changed_labels_or_own(self):
        old=fixture();data={k:v.copy() for k,v in old.items()};data['queries']+=1
        assert_reuse(old,data);data['y'][0,0]=(data['y'][0,0]+1)%5
        with self.assertRaises(AssertionError):assert_reuse(old,data)


if __name__=='__main__':unittest.main()
