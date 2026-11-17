import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from compact_features import NUMERIC,PRICE,compact_matrix,feature_mask,dataset
from adaptive_research import select_ridge,choose_models,prune_state
from research import normalize_fold


class AdaptiveTests(unittest.TestCase):
    def test_missing_future_intermediate_day_only_excludes_missing_endpoints(self):
        from research import make_folds
        index=pd.bdate_range('2025-01-01',periods=208)
        df=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':1000.},index=index)
        good=pd.Series(True,index=index);good.iloc[188]=False
        df.iloc[188]=np.nan
        loaded=(df,good,[],[],[],[]);matrix=np.zeros((208,76),dtype='float32')
        _,_,meta,masks=dataset(loaded,matrix,make_folds(208)[2])
        self.assertEqual(int(masks['test'].sum()),12)
        self.assertNotIn(178,meta.loc[masks['test'],'calendar_index'].tolist())
        self.assertNotIn(183,meta.loc[masks['test'],'calendar_index'].tolist())
        _,_,_,latest_masks=dataset(loaded,matrix,{'val_start':178,'test_start':208,'test_stop':208},latest=True)
        self.assertEqual(int(latest_masks['val'].sum()),8)
        self.assertEqual(int(latest_masks['latest'].sum()),0)

    def test_pruning_removes_unused_inputs_without_changing_prediction(self):
        from main import Encoder
        torch.manual_seed(42)
        model=Encoder(12,20,channels=16,compact_decoder=True).eval()
        mask=np.array([True,False,True]+[False]*8+[True])
        x=torch.randn(5,20,12);x[:,:,~mask]=0
        pruned=Encoder(int(mask.sum()),20,channels=16,compact_decoder=True).eval()
        pruned.load_state_dict(prune_state(model.state_dict(),mask))
        with torch.no_grad():
            z,p,_=model(x);z2,p2,_=pruned(x[:,:,mask])
        torch.testing.assert_close(z,z2,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(p,p2,atol=1e-5,rtol=1e-5)

    def synthetic(self):
        rng=np.random.default_rng(5);n=40
        pairs={name:[2*i,2*i+1] for i,name in enumerate(NUMERIC)}
        numeric=rng.normal(size=(n,len(NUMERIC))).astype('float32')
        x=np.zeros((n,20,2*len(NUMERIC)),dtype='float32')
        x[:,:,::2]=numeric[:,None,:]
        masks={'train':np.arange(n)<26,'val':(np.arange(n)>=26)&(np.arange(n)<33),'test':np.arange(n)>=33}
        return x,pairs,masks

    def test_different_stocks_can_select_different_features(self):
        x,pairs,masks=self.synthetic()
        selected=[]
        for signal in ['return','summary_retail_raw']:
            y=np.tile(x[:,-1,pairs[signal][0]][:,None],(1,3))
            xn,yn,*_=normalize_fold(x,y,masks)
            chosen,_,_=select_ridge(xn,yn,masks,pairs,max_features=1)
            self.assertEqual(chosen[0],signal)
            selected.append(chosen)
        self.assertNotEqual(selected[0],selected[1])

    def test_no_feature_is_forced_when_validation_is_already_perfect(self):
        x,pairs,masks=self.synthetic();y=np.zeros((len(x),3),dtype='float32')
        selected,path,_=select_ridge(x,y,masks,pairs,max_features=2)
        self.assertEqual(selected,[]);self.assertEqual(path,[])

    def test_holdout_cannot_change_selection_or_weights(self):
        torch.set_num_threads(2)
        x,pairs,masks=self.synthetic();signal=pairs['summary_retail_raw'][0]
        y=np.tile(x[:,-1,signal][:,None],(1,3))
        args=SimpleNamespace(max_features=1,min_gain=.02,seeds=[42],epochs=2)
        a=choose_models(x,y,masks,pairs,args,torch.device('cpu'))
        changed=x.copy();changed[masks['test']]=10000
        other=y.copy();other[masks['test']]=9999
        b=choose_models(changed,other,masks,pairs,args,torch.device('cpu'))
        self.assertEqual(a[-1]['ridge_selected_features'],b[-1]['ridge_selected_features'])
        self.assertEqual(a[-1]['cnn_selected_features'],b[-1]['cnn_selected_features'])
        self.assertEqual(a[-1]['cnn_candidates'],b[-1]['cnn_candidates'])

    def test_compact_matrix_is_causal(self):
        from research import load_stock
        root=Path(r'data/Daily')
        if not (root/'Summary/ADRO_Summary.csv').exists():self.skipTest('Local source absent')
        loaded=load_stock('ADRO',root/'Summary',root/'All')
        original,names,pairs=compact_matrix(loaded)
        changed=list(copy.deepcopy(loaded))
        changed[0].iloc[170:]=changed[0].iloc[170:]*3
        changed[4][180]=None
        actual,_,_=compact_matrix(changed)
        np.testing.assert_array_equal(original[:170],actual[:170])
        self.assertEqual(len(NUMERIC),38);self.assertEqual(len(names),76)
        self.assertEqual(int(feature_mask(PRICE,pairs,76).sum()),18)

    def test_full_adaptive_outputs(self):
        root=Path('runs/adaptive-v1')
        if not (root/'config.json').exists():self.skipTest('Complete adaptive run first')
        config=json.loads((root/'config.json').read_text())
        universe=json.loads((root/'universe.json').read_text())
        latest=json.loads((root/'latest_manifest.json').read_text())
        selection=json.loads((root/'selection_audit.json').read_text())
        self.assertEqual(len(config['stocks']),25);self.assertEqual(len(set(config['stocks'])),25)
        self.assertEqual(len(universe['added']),10)
        self.assertEqual(len(selection),75);self.assertEqual(len(latest),25)
        self.assertEqual(len({r['model_sha256'] for r in latest}),25)
        for audit in selection:
            for row in audit['samples']:
                if row['split']=='train':self.assertLess(row['label_end'],audit['val_start'])
                if row['split']=='val':self.assertLess(row['label_end'],audit['test_start'])
            candidates=audit['cnn_candidates'];winner=min(candidates,key=lambda r:r['validation_mse'])
            self.assertEqual(audit['cnn_selected_features'],winner['features'])
            self.assertLessEqual(len(audit['ridge_selected_features']),8)
            self.assertTrue(set(audit['cnn_selected_features']).issubset(set(NUMERIC)))
        forecasts=pd.read_csv(root/'predictions.csv')
        self.assertEqual(set(forecasts.ticker),set(config['stocks']))
        self.assertFalse(forecasts.duplicated(['ticker','fold','model','seed','date','horizon']).any())
        counts=forecasts.groupby(['ticker','model','seed','horizon']).date.nunique()
        self.assertTrue((counts>0).all());self.assertTrue((counts<=42).all())
        self.assertEqual(len(pd.read_csv(root/'comparison.csv')),75)
        chart=json.loads((root/'latest/dashboard-config.json').read_text())
        self.assertEqual(len(chart['data']),75)
        self.assertEqual(len(chart['layout']['updatemenus'][0]['buttons']),25)
        for i,button in enumerate(chart['layout']['updatemenus'][0]['buttons']):
            self.assertEqual(button['args'][0]['visible'],[j//3==i for j in range(75)])
        for record in latest:
            path=root/'latest'/record['ticker']/'model.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(checkpoint['ticker'],record['ticker'])
            self.assertEqual(checkpoint['selected_features'],record['selected_features'])
            self.assertEqual(int(checkpoint['source_feature_mask'].sum()),2*len(record['selected_features']))
            self.assertEqual(len(checkpoint['feature_names']),2*len(record['selected_features']))
            self.assertEqual(checkpoint['state_dict']['conv.0.weight'].shape[1],len(checkpoint['feature_names']))

    def test_pruned_checkpoint_can_run_standalone_inference(self):
        path=Path('runs/adaptive-v1/latest_manifest.json')
        if not path.exists():self.skipTest('Train latest models first')
        from predict_adaptive import predict
        torch.set_num_threads(2)
        expected=next(r for r in json.loads(path.read_text()) if r['ticker']=='ADRO')
        actual=predict('ADRO')
        self.assertTrue(actual['available'])
        self.assertEqual(actual['selected_features'],expected['selected_features'])
        for h in ['5','10','20']:
            self.assertAlmostEqual(actual['predicted_returns'][h],expected['predictions'][h],delta=1e-5)


if __name__=='__main__':unittest.main()
