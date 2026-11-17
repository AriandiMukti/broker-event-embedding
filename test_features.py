import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features import expanded_features
from main import broker_row, Encoder
import torch


class ExpandedFeaturesTests(unittest.TestCase):
    def test_all_expanded_models_and_actual_latest_inputs(self):
        root=Path('runs/expanded-v2')
        if not (root/'manifest.json').exists():
            self.skipTest('Run expanded pipeline first')
        manifest=json.loads((root/'manifest.json').read_text())
        requested={'ANTM','PTBA','ADRO','BREN','TPIA','BRPT','PTRO','INKP','BUMI','BRMS','ISAT'}
        self.assertEqual(len(manifest),15)
        self.assertTrue(requested.issubset({r['ticker'] for r in manifest}))
        self.assertEqual(len({r['model_sha256'] for r in manifest}),15)
        for item in manifest:
            ticker=item['ticker'];folder=root/ticker
            report=json.loads((folder/'data_report.json').read_text())
            names=report['features'];pack=np.load(folder/'dataset.npz')
            self.assertEqual(report['feature_set'],'expanded_v2')
            self.assertEqual(report['tickers'],[ticker])
            self.assertEqual(report['universe_files'],1)
            self.assertEqual(set(pd.read_csv(folder/'samples.csv').ticker),{ticker})
            self.assertEqual(set(pd.read_csv(folder/f'{ticker}_neighbors.csv').ticker),{ticker})
            self.assertTrue(np.isfinite(pack['x']).all())
            self.assertTrue(np.isfinite(pack['latest_x']).all())
            summary=pd.read_csv(Path(report['summary'])/f'{ticker}_Summary.csv')
            summary['Date']=pd.to_datetime(summary.Date,format='%d-%b-%y')
            summary=summary.sort_values('Date').set_index('Date').loc[:report['end']]
            vector=pack['latest_x'][0,-1]
            for h in (20,40):
                expected=summary.Close.iloc[-1]/summary.Close.iloc[-1-h]-1
                self.assertAlmostEqual(float(vector[names.index(f'price_return_{h}')]),expected,places=6)
            raw=pd.to_numeric(summary.Retail,errors='coerce').iloc[-1]
            if np.isfinite(raw):
                self.assertAlmostEqual(float(vector[names.index('summary_retail_raw')]),float(raw),delta=max(1e-4,abs(float(raw))*1e-6))
            b=float(vector[names.index('broker_buy_count')]);s=float(vector[names.index('broker_sell_count')])
            if s>0:
                self.assertAlmostEqual(float(vector[names.index('buyer_seller_ratio')]),b/s,places=6)
            meta=pd.read_csv(folder/'samples.csv')
            self.assertTrue((meta.loc[meta.split=='train','label_end']<report['val_start']).all())
            self.assertTrue((meta.loc[meta.split=='val','label_end']<report['test_start']).all())
        checkpoint=torch.load(root/'ADRO/model.pt',map_location='cpu',weights_only=False)
        self.assertEqual(checkpoint['tickers'],['ADRO'])
        self.assertTrue(checkpoint['compact_decoder'])
        pack=np.load(root/'ADRO/dataset.npz');meta=pd.read_csv(root/'ADRO/samples.csv')
        np.testing.assert_allclose(checkpoint['mean'],pack['x'][meta.split=='train'].mean(axis=(0,1)),rtol=1e-6)
        config=json.loads((root/'dashboard-config.json').read_text())
        for i,button in enumerate(config['layout']['updatemenus'][0]['buttons']):
            self.assertEqual(button['args'][0]['visible'],[j//3==i for j in range(45)])
        self.assertEqual(len(config['data']),45)

    def fixture(self):
        n=100; index=pd.bdate_range('2025-01-01',periods=n)
        df=pd.DataFrame({'Open':np.arange(n)+100.,'High':np.arange(n)+101.,
                         'Low':np.arange(n)+99.,'Close':np.arange(n)+100.,'Volume':1000.,
                         'Retail':-50.,'Total Buyer':2.,'Total Seller':1.,
                         'Net Top3 All Lot':30.,'Net Top5 All Lot':20.},index=index)
        snapshots=[{'values':{'AA':300.,'BB':200.,'CC':-500.},
                    'lots':{'AA':30.,'BB':20.,'CC':-50.},
                    'prices':{'AA':100.,'BB':101.,'CC':99.}} for _ in range(n)]
        return df,snapshots

    def test_exact_returns_retail_counts_and_rolling_lots(self):
        df,snapshots=self.fixture()
        matrix,names,_=expanded_features(df,['AA','BB','CC'],snapshots)
        at=lambda name,i=60: float(matrix[i,names.index(name)])
        self.assertAlmostEqual(at('price_return_20'),160/140-1,places=6)
        self.assertAlmostEqual(at('price_return_40'),160/120-1,places=6)
        self.assertEqual(at('summary_retail_raw'),-50)
        self.assertAlmostEqual(at('summary_retail_share_abs_net_lot'),-.5)
        self.assertEqual(at('summary_retail_cumulative_5'),-250)
        self.assertEqual(at('broker_buy_count'),2)
        self.assertEqual(at('broker_sell_count'),1)
        self.assertEqual(at('buyer_seller_ratio'),2)
        self.assertAlmostEqual(at('broker_buyer_fraction'),2/3,places=6)
        self.assertAlmostEqual(at('net_lot_cumulative_share_20_AA'),.3,places=6)
        self.assertEqual(at('broker_persistence_20_AA'),1)
        self.assertEqual(at('broker_persistence_20_CC'),-1)

    def test_zero_sellers_and_missing_retail_are_explicit(self):
        df,snapshots=self.fixture();df.loc[df.index[60],'Retail']=np.nan
        snapshots[60]={'values':{'AA':300.},'lots':{'AA':30.},'prices':{'AA':100.}}
        matrix,names,_=expanded_features(df,['AA','BB','CC'],snapshots)
        self.assertEqual(matrix[60,names.index('buyer_seller_ratio_missing')],1)
        self.assertEqual(matrix[60,names.index('seller_count_zero')],1)
        self.assertEqual(matrix[60,names.index('summary_retail_raw_missing')],1)
        self.assertTrue(np.isfinite(matrix).all())

    def test_future_changes_do_not_change_earlier_features(self):
        df,snapshots=self.fixture()
        original,names,_=expanded_features(df,['AA','BB','CC'],snapshots)
        changed=df.copy();changed.iloc[80:]=changed.iloc[80:]*10
        later=list(snapshots);later[85]=None
        actual,_,_=expanded_features(changed,['AA','BB','CC'],later)
        np.testing.assert_array_equal(original[:80],actual[:80])

    def test_broker_price_features_are_invariant_to_raw_price_scale(self):
        df,snapshots=self.fixture()
        original,names,_=expanded_features(df,['AA','BB','CC'],snapshots)
        scaled=[dict(s,prices={k:v*10 for k,v in s['prices'].items()}) for s in snapshots]
        actual,_,_=expanded_features(df,['AA','BB','CC'],scaled)
        for name in ['broker_price_spread','broker_price_deviation_AA','broker_price_deviation_BB']:
            np.testing.assert_allclose(original[:,names.index(name)],actual[:,names.index(name)],rtol=1e-6)

    def test_compact_decoder_gradient(self):
        model=Encoder(200,20,channels=16,compact_decoder=True)
        z,p,reconstruction=model(torch.randn(3,20,200))
        self.assertEqual(reconstruction.shape,(3,20,200))
        (reconstruction.square().mean()+p.square().mean()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))



if __name__=='__main__':
    unittest.main()
