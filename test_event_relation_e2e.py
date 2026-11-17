"""Temporal/masking/objective tests; random fixtures only for unit tests."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from event_relation_model import RelationModel,RelationDataset,MODES
from event_relation_training import fit,infer
from event_training import seed_all,weights_hash


def fixture():
    rng=np.random.default_rng(72);n=60;b=4
    return dict(own=rng.normal(size=(n,b,5,14)).astype('float32'),context=rng.normal(size=(n,18)).astype('float32'),
        tokens=rng.normal(size=(n,5,b,8)).astype('float32'),mask=rng.random((n,5,b))>.2,
        ids=np.arange(2,b+2,dtype='int64'),y=rng.integers(0,5,(n,b)),vocabulary_size=np.asarray(b),
        splits=np.array(['train']*30+['val']*15+['test']*15),queries=np.arange(60,120),codes=np.array(['AA','BB','CC','DD']))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def inputs(self,days=2):
        return (torch.randn(days,4,5,14),torch.randn(days,18),torch.randn(days,4,5,4,8),
                torch.ones(days,4,5,4,dtype=torch.bool),torch.arange(2,6))

    def test_target_is_masked_in_entire_peer_history(self):
        model=RelationModel(4,width=8).eval();own,ctx,tok,mask,ids=self.inputs()
        p=model.peer_representation(tok,mask,ids);other=tok.clone()
        for target in range(4):other[:,target,:,target]=1e6
        torch.testing.assert_close(p,model.peer_representation(other,mask,ids),atol=0,rtol=0)
        missing=model.peer_representation(tok,torch.zeros_like(mask),ids)
        self.assertTrue((missing==0).all())

    def test_own_control_ignores_peer_values(self):
        model=RelationModel(4,width=8,mode='own_only').eval()
        own,ctx,tok,mask,ids=self.inputs()
        torch.testing.assert_close(model(own,ctx,tok,mask,ids),model(own,ctx,tok*1000,~mask,ids),atol=0,rtol=0)

    def test_matched_initial_weights_and_frozen_encoder_flags(self):
        hashes=[]
        for mode in MODES:
            seed_all(11);model=RelationModel(4,width=8,mode=mode);hashes.append(weights_hash(model.state_dict()))
            self.assertEqual(all(not p.requires_grad for p in model.set_encoder.parameters()),mode in ['own_only','frozen_peers'])
        self.assertEqual(len(set(hashes)),1)

    def test_dataset_window_and_shift_control_do_not_change_labels_or_own(self):
        data=fixture();rows=np.arange(60)
        normal=RelationDataset(data,rows,'relational',11);shift=RelationDataset(data,rows,'shifted_peers',11)
        for index in [0,29,30,44,45,59]:
            a=normal[index];b=shift[index]
            for j in [0,1,4]:torch.testing.assert_close(a[j],b[j])
            for target,source in enumerate(shift.mapping[index]):
                self.assertEqual(data['splits'][index],data['splits'][source])
                np.testing.assert_array_equal(b[2][target].numpy(),data['tokens'][source])

    def test_fit_ignores_test_and_updates_only_allowed_encoder(self):
        data=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(device='cpu',width=8,batch_size=16,lr=.001,amp=False,epochs=2,patience=2,out=tmp)
            model,a,_=fit(data,config,'relational',11,Path(tmp)/'a')
            changed=copy.deepcopy(data);test=changed['splits']=='test';changed['own'][test]=1000
            changed['tokens'][test]=1000;changed['y'][test]=4
            other,b,_=fit(changed,config,'relational',11,Path(tmp)/'b')
            self.assertEqual(a['best_encoder_sha256'],b['best_encoder_sha256']);self.assertTrue(a['encoder_changed'])
            self.assertEqual(weights_hash(model.state_dict()),weights_hash(other.state_dict()))
            frozen,c,_=fit(data,config,'frozen_peers',11,Path(tmp)/'frozen')
            self.assertFalse(c['encoder_changed'])
            p=infer(model,data,np.flatnonzero(test),config,'relational',11)
            self.assertEqual(p.shape,(15,4,5));np.testing.assert_allclose(p.sum(-1),1)


if __name__=='__main__':unittest.main()
