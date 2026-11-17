import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from main import Encoder, broker_row, read_summary, prepare


class PipelineTests(unittest.TestCase):
    def test_single_stock_does_not_read_other_summaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); summary=root/'summary';summary.mkdir()
            brokers=root/'brokers';brokers.mkdir()
            days=pd.bdate_range('2025-01-01',periods=160)
            (brokers/days[-1].strftime('%Y-%m-%d')).mkdir()
            close=100+np.sin(np.arange(len(days))*.2)
            pd.DataFrame({'Date':days.strftime('%d-%b-%y'),'Open':close,'High':close+1,
                          'Low':close-1,'Close':close,'Volume':1000}).to_csv(summary/'ADRO_Summary.csv',index=False)
            (summary/'BROKEN_Summary.csv').write_text('Not a valid stock file')
            output=root/'result'
            prepare(SimpleNamespace(out=str(output),summary=str(summary),brokers=str(brokers),
                                    only_ticker='ADRO',window=20,stocks=1))
            report=json.loads((output/'data_report.json').read_text())
            self.assertEqual(report['universe_files'],1)
            self.assertEqual(report['tickers'],['ADRO'])
            self.assertEqual(set(pd.read_csv(output/'samples.csv').ticker),{'ADRO'})

    def test_five_runs_are_independent(self):
        root=Path('runs/individual')
        if not (root/'manifest.json').exists():
            self.skipTest('Run individual.py first')
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(len(manifest),5)
        self.assertEqual(len({item['model_sha256'] for item in manifest}),5)
        for item in manifest:
            ticker=item['ticker']; folder=root/ticker
            self.assertEqual(item['scope'],'single_stock')
            self.assertEqual(item['source_summary_files'],1)
            self.assertEqual(item['dataset_tickers'],[ticker])
            meta=pd.read_csv(folder/'samples.csv')
            report=json.loads((folder/'data_report.json').read_text())
            self.assertEqual(set(meta.ticker),{ticker})
            self.assertEqual(set(pd.read_csv(folder/f'{ticker}_neighbors.csv').ticker),{ticker})
            self.assertTrue((meta.loc[meta.split=='train','label_end']<report['val_start']).all())
            self.assertTrue((meta.loc[meta.split=='val','label_end']<report['test_start']).all())
            self.assertTrue((meta.loc[meta.split=='test','date']>=report['test_start']).all())
            self.assertTrue(np.isfinite(np.load(folder/'dataset.npz')['x']).all())
            self.assertTrue((folder/'embedding-3d.html').exists())
            self.assertTrue((folder/'plotly.min.js').exists())

    def test_dashboard_selector_shows_one_stock_at_a_time(self):
        path=Path('runs/individual/dashboard-config.json')
        if not path.exists():
            self.skipTest('Build dashboard first')
        config=json.loads(path.read_text())
        tickers=['ADRO','BBCA','BBRI','BMRI','TLKM']
        self.assertEqual(len(config['data']),15)
        buttons=config['layout']['updatemenus'][0]['buttons']
        self.assertEqual(len(buttons),5)
        for i,button in enumerate(buttons):
            visible=button['args'][0]['visible']
            self.assertEqual(visible,[j//3==i for j in range(15)])
            self.assertEqual(button['label'],tickers[i])
            for trace,shown in zip(config['data'],visible):
                if shown:
                    self.assertIn(tickers[i],trace['name'])

    def test_signed_net_and_concentration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.json'
            shared = {'netbs_date': '20260102', 'netbs_stock_code': 'ADRO'}
            payload = {'data': {'broker_summary': {
                'brokers_buy': [dict(shared, netbs_broker_code='AA', bval='300', type='Lokal')],
                'brokers_sell': [dict(shared, netbs_broker_code='ZZ', sval='-100', type='Asing')]}}}
            path.write_text(json.dumps(payload))
            flows, extra = broker_row(path, 'ADRO', pd.Timestamp('2026-01-02'))
            self.assertEqual(flows, {'AA': .75, 'ZZ': -.25})
            self.assertEqual(extra[1], 1.0)
            self.assertEqual(extra[2], 1.0)
            with self.assertRaises(ValueError):
                broker_row(path, 'ADRO', pd.Timestamp('2026-01-05'))

    def test_invalid_ohlc_is_flagged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'summary.csv'
            path.write_text('Date,Open,High,Low,Close,Volume\n2-Jan-26,100,90,95,100,20\n')
            _, good = read_summary(path)
            self.assertFalse(good.iloc[0])


    def test_encoder_supports_backprop(self):
        model = Encoder(15, 20)
        z, prediction, reconstruction = model(torch.randn(8, 20, 15))
        self.assertEqual(z.shape, (8, 32))
        self.assertEqual(prediction.shape, (8, 3))
        (prediction.square().mean()+reconstruction.square().mean()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_real_dataset_has_purged_boundaries(self):
        root = Path('runs/local')
        if not (root/'samples.csv').exists():
            self.skipTest('Run prepare first')
        meta = pd.read_csv(root/'samples.csv')
        config = json.loads((root/'data_report.json').read_text())
        self.assertTrue((meta.loc[meta.split == 'train', 'label_end'] < config['val_start']).all())
        self.assertTrue((meta.loc[meta.split == 'val', 'date'] >= config['val_start']).all())
        self.assertTrue((meta.loc[meta.split == 'val', 'label_end'] < config['test_start']).all())
        self.assertTrue((meta.loc[meta.split == 'test', 'date'] >= config['test_start']).all())
        pack = np.load(root/'dataset.npz')
        self.assertTrue(np.isfinite(pack['x']).all())
        self.assertTrue(np.isfinite(pack['y']).all())


if __name__ == '__main__':
    unittest.main()
