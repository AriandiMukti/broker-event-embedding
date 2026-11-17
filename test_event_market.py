"""Temporal alignment/quality tests, synthetic fixtures only."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from event_market_data import series_features


class MarketTests(unittest.TestCase):
    def make(self,path,bad=False,future=False):
        dates=pd.bdate_range('2020-01-01',periods=50);price=np.exp(np.arange(50)*.01)*100
        frame=pd.DataFrame(dict(date=dates,Open=price,High=price+1,Low=price-1,Close=price))
        if bad:frame.loc[21,'High']=1
        if future:frame.loc[35:,'Close']*=100;frame.loc[35:,'High']=frame.loc[35:,'Close']+1
        frame.to_csv(path,index=False);return dates,price

    def test_no_same_date_bar_and_future_changes_do_not_change_past(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';dates,_=self.make(p)
            a,names,joins=series_features(p,dates)
            seen=joins.source_date.notna()
            self.assertTrue((joins.loc[seen,'source_date']<joins.loc[seen,'query_date']).all())
            self.make(p,future=True);b,_,_=series_features(p,dates)
            np.testing.assert_array_equal(a[:36],b[:36])

    def test_invalid_ohlc_masked_and_no_future_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';dates,_=self.make(p,bad=True)
            x,names,_=series_features(p,dates)
            self.assertEqual(x[0,names.index('return1_missing')],1)
            self.assertEqual(x[22,names.index('return1_missing')],1)
            self.assertEqual(x[22,names.index('return1')],0)

    def test_relative_strength_uses_matching_past_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';dates,price=self.make(p)
            x,names,_=series_features(p,dates,pd.Series(price*2,index=dates))
            for h in [1,5,20]:
                valid=x[:,names.index(f'relative_strength{h}_missing')]==0
                self.assertTrue(valid.any())
                np.testing.assert_allclose(x[valid,names.index(f'relative_strength{h}')],0,atol=1e-12)

    def test_stale_source_not_carried_indefinitely_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';dates,_=self.make(p)
            x,names,_=series_features(p,pd.DatetimeIndex([dates[-1]+pd.Timedelta(days=15)]))
            self.assertEqual(x[0,names.index('return1_missing')],1)
            frame=pd.read_csv(p);pd.concat([frame,frame.iloc[-1:]]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate'):series_features(p,dates)


if __name__=='__main__':unittest.main()
