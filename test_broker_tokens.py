import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from broker_tokens import fit_tokenizer,transform_tokens,make_samples,STATES
from adro_token_embedding import EventEmbedding,fit_model,probability,scores,baseline_probabilities


class BrokerTokenTests(unittest.TestCase):
    def test_saved_model_standalone_prediction_matches_export(self):
        root=Path('runs/adro-token-v1')
        if not (root/'model.pt').exists():self.skipTest('Run token model first')
        from predict_broker_tokens import predict
        torch.set_num_threads(1)
        result=predict()
        expected=json.loads((root/'latest_next_session.json').read_text())
        self.assertTrue(result['available'])
        self.assertEqual(result['context_end'],expected['context_end'])
        for actual,saved in zip(result['predictions'],expected['predictions']):
            self.assertEqual(actual['broker'],saved['broker'])
            for state in STATES:self.assertAlmostEqual(actual['probabilities'][state],saved['probabilities'][state],places=6)

    def fixture(self):
        n=208;index=pd.bdate_range('2025-01-01',periods=n)
        close=100+np.arange(n)*.1+np.sin(np.arange(n)*.4)
        df=pd.DataFrame({'Open':close,'High':close+2,'Low':close-2,'Close':close,'Volume':1000+np.arange(n),
                         'Retail':np.sin(np.arange(n))},index=index)
        snapshots=[];cache=[]
        for i in range(n):
            a=(1 if i%3 else -1)*(100+i%13);b=(1 if i%4 else -1)*(150+i%17)
            values={'AA':a,'BB':b}
            if i>=124:values['ZZ']=100000
            total=sum(abs(v) for v in values.values())
            cache.append({k:v/total for k,v in values.items()})
            snapshots.append({'values':values,'lots':{k:v/10 for k,v in values.items()},'prices':{k:100. for k in values}})
        return df,pd.Series(True,index=index),cache,np.zeros((n,4)),snapshots,[]

    def test_future_data_cannot_change_vocabulary_or_bins(self):
        loaded=self.fixture();before=fit_tokenizer(loaded,124,2)
        self.assertNotIn('ZZ',before['brokers'])
        changed=list(copy.deepcopy(loaded));changed[0].iloc[124:]*=10
        changed[4][130]['values']['ZZ']=1e12
        after=fit_tokenizer(changed,124,2)
        self.assertEqual(before,after)

    def test_context_precedes_target_and_splits_are_chronological(self):
        loaded=self.fixture();tokenizer=fit_tokenizer(loaded,124,2)
        x,y,previous,targets,masks,daily,states,valid=make_samples(loaded,tokenizer)
        self.assertEqual({s:int(m.sum()) for s,m in masks.items()},{'train':81,'val':42,'test':42})
        for i,t in enumerate(targets):
            np.testing.assert_array_equal(x[i],daily[t-3:t])
            np.testing.assert_array_equal(previous[i],states[t-1])
            np.testing.assert_array_equal(y[i],states[t])
        self.assertLess(targets[masks['train']].max(),targets[masks['val']].min())
        self.assertLess(targets[masks['val']].max(),targets[masks['test']].min())

    def test_model_is_unordered_within_day_and_has_tied_vectors(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        model=EventEmbedding(20,2,dimension=4,context_days=3)
        x=torch.randint(1,20,(6,3,7))
        p=model(x);reordered=model(x.flip(dims=[2]))
        torch.testing.assert_close(p,reordered,atol=1e-6,rtol=1e-6)
        self.assertEqual(set(dict(model.named_parameters())),{'embedding.weight','transform.weight','bias'})
        model(x).square().mean().backward()
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()),0)
        self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()),0)
        np.testing.assert_allclose(probability(model,x).sum(axis=2),1,atol=1e-6)

    def test_test_labels_do_not_affect_fit_or_checkpoint(self):
        torch.set_num_threads(1);rng=np.random.default_rng(7)
        x=rng.integers(1,15,size=(20,3,6));y=rng.integers(0,5,size=(20,2))
        masks={'train':np.arange(20)<12,'val':(np.arange(20)>=12)&(np.arange(20)<16),'test':np.arange(20)>=16}
        args=SimpleNamespace(dimension=4,context_days=3,epochs=5,patience=3)
        model,record=fit_model(x,y,masks,15,2,args,42)
        changed=y.copy();changed[masks['test']]=(changed[masks['test']]+1)%5
        other,record2=fit_model(x,changed,masks,15,2,args,42)
        self.assertEqual(record,record2)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],atol=0,rtol=0)
        _,shuffled=fit_model(x,y,masks,15,2,args,42,shuffle=True)
        self.assertEqual(record['initial_weights_sha256'],shuffled['initial_weights_sha256'])

    def test_baselines_only_fit_training_targets(self):
        loaded=self.fixture();tokenizer=fit_tokenizer(loaded,124,2)
        _,y,previous,_,masks,*_=make_samples(loaded,tokenizer)
        before=baseline_probabilities(y,previous,masks)
        changed=y.copy();changed[~masks['train']]=(changed[~masks['train']]+1)%5
        after=baseline_probabilities(changed,previous,masks)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        perfect=np.eye(5)[y]
        result=scores(y,perfect)
        self.assertEqual(result['nll'],0);self.assertEqual(result['brier'],0);self.assertEqual(result['state_accuracy'],1)

    def test_completed_adro_token_run(self):
        root=Path('runs/adro-token-v1')
        if not (root/'config.json').exists():self.skipTest('Run token model first')
        config=json.loads((root/'config.json').read_text());tokenizer=json.loads((root/'tokenizer.json').read_text())
        self.assertEqual(config['ticker'],'ADRO');self.assertEqual(config['counts'],{'train':81,'val':42,'test':42})
        self.assertEqual(len(config['brokers']),12);self.assertEqual(config['dimension'],8)
        runs=json.loads((root/'training_runs.json').read_text())
        self.assertEqual(len(runs),10)
        real=[r for r in runs if not r['shuffled_training_context']]
        self.assertEqual(config['best_seed_by_validation'],min(real,key=lambda r:r['validation_nll'])['seed'])
        for seed in config['seeds']:
            same=[r for r in runs if r['seed']==seed]
            self.assertEqual(len({r['initial_weights_sha256'] for r in same}),1)
        vectors=pd.read_csv(root/'token_vectors.csv')
        self.assertEqual(len(vectors),93)
        self.assertTrue(np.isfinite(vectors[[f'embedding_{i}' for i in range(1,9)]]).all().all())
        neighbors=pd.read_csv(root/'token_neighbors.csv')
        self.assertTrue((neighbors['query'].str.split('_').str[0]!=neighbors.neighbor.str.split('_').str[0]).all())
        self.assertTrue((neighbors.query_train_days>=5).all());self.assertTrue((neighbors.neighbor_train_days>=5).all())
        latest=json.loads((root/'latest_next_session.json').read_text())
        for row in latest['predictions']:
            self.assertAlmostEqual(sum(row['probabilities'].values()),1,places=6)
            self.assertEqual(set(row['probabilities']),set(STATES))
        checkpoint=torch.load(root/'model.pt',map_location='cpu',weights_only=False)
        model=EventEmbedding(len(tokenizer['tokens']),len(tokenizer['brokers']),8,3)
        model.load_state_dict(checkpoint['state_dict'])
        data=np.load(root/'token_dataset.npz');samples=pd.read_csv(root/'samples.csv')
        mask=(samples.split=='test').to_numpy();p=probability(model,data['x'][mask])
        actual=scores(data['y'][mask],p)
        self.assertAlmostEqual(actual['nll'],config['selected_seed_test']['nll'],places=6)


if __name__=='__main__':unittest.main()
