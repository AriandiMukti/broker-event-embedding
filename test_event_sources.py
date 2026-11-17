import tempfile
import unittest
from pathlib import Path

import pandas as pd

from event_sources import market_calendar, broker_csv, compact_number, resolve


class SourceTests(unittest.TestCase):
    def test_calendar_deduplicates_formats_and_keeps_2020(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['20200602', '20260102', '2026-01-02', '2026-01-05', 'notes']:
                (Path(tmp)/name).mkdir()
            days = market_calendar(tmp)
            self.assertEqual(days.strftime('%Y-%m-%d').tolist(), ['2020-06-02', '2026-01-02', '2026-01-05'])

    def test_value_units_quantization_and_missing_rejection(self):
        for raw, expected in [('8.5B', 8.5e9), ('-17.4B', -17.4e9), ('2M', 2e6),
                              ('1.25K', 1250), ('1T', 1e12), ('1,234', 1234)]:
            self.assertEqual(compact_number(raw)[0], expected)
        self.assertEqual(compact_number('8.5B')[1], 5e7)
        with self.assertRaises(ValueError): compact_number('--')

    def test_csv_disjoint_sides_and_signs_and_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'20200602'; folder.mkdir()
            path = folder/'ADRO-20200602.csv'
            row = dict(Buyer='AA', **{'B Lot': '100', 'B Value': '8.5B', 'B Average': '1,149'},
                       Seller='BB', **{'S Lot': '-200', 'S Value': '-17.4B', 'S Average': '1,152'})
            pd.DataFrame([row]).to_csv(path,index=False)
            day = pd.Timestamp('2020-06-02')
            self.assertEqual(resolve(root,'ADRO',day),path)
            _, _, detail = broker_csv(path,'ADRO',day,True)
            self.assertEqual(detail['values']['BB'], -17.4e9)
            self.assertEqual(detail['lots']['AA'], 100)
            row['S Lot']='1'; pd.DataFrame([row]).to_csv(path,index=False)
            _, _, detail = broker_csv(path,'ADRO',day,True)
            self.assertEqual(detail['lots']['BB'],1)
            self.assertEqual(detail['lot_sign_anomalies'],1)
            row['S Value']='467.5K'; pd.DataFrame([row]).to_csv(path,index=False)
            _, _, detail = broker_csv(path,'ADRO',day,True)
            self.assertEqual(detail['values']['BB'],467500)
            self.assertEqual(detail['value_sign_anomalies'],1)
            row['Seller']='AA'; pd.DataFrame([row]).to_csv(path,index=False)
            with self.assertRaises(ValueError): broker_csv(path,'ADRO',day,True)


if __name__ == '__main__': unittest.main()
