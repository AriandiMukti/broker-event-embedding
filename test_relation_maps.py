import copy
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adro_token_embedding import EventEmbedding
from broker_relation_test import (tensor_inputs, logits_for, selected_logits, baseline,
                                  relation_tokens, dataset, folds, align_loaded_calendar)
from relation_maps import simultaneous_intervals, stable_pairs, classify, CONTROLS, same_day_cooccurrence,peer_summary
from relation_io import compact_predictions,read_predictions,source_fingerprint
import test_broker_tokens as fixtures


class MultiRelationTests(unittest.TestCase):
    def test_fast_head_matches_original_logits_and_gradients(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        model=EventEmbedding(31,4,8,1); other=copy.deepcopy(model)
        x=np.random.default_rng(5).integers(0,31,(6,4,10))
        full=logits_for(model,x); tx,ids=tensor_inputs(x)
        fast=selected_logits(other,tx,ids)
        torch.testing.assert_close(full,fast,atol=1e-7,rtol=1e-6)
        full.square().sum().backward(); fast.square().sum().backward()
        for key,value in model.named_parameters():
            torch.testing.assert_close(value.grad,dict(other.named_parameters())[key].grad,atol=1e-6,rtol=1e-5)

    def test_absolute_thresholds_are_stock_specific_and_bad_prices_masked(self):
        loaded=fixtures.BrokerTokenTests().fixture()
        changed=copy.deepcopy(loaded)
        for s in changed[4]:
            for c in s['values']: s['values'][c]*=100
        t,_,_,_=relation_tokens(loaded,103,2,ticker='ADRO',clean_ohlcv=True)
        u,_,_,_=relation_tokens(changed,103,2,ticker='BBCA',clean_ohlcv=True)
        self.assertEqual(u['ticker'],'BBCA')
        for c in t['brokers']:
            self.assertAlmostEqual(u['large_absolute_net_threshold'][c],100*t['large_absolute_net_threshold'][c])
        bad=copy.deepcopy(loaded)
        bad[0].iloc[170,bad[0].columns.get_loc('Close')]=1e20
        bad[1].iloc[170]=False
        _,daily,states,valid=relation_tokens(bad,103,2,clean_ohlcv=True)
        # return_1 at 171 cannot use invalid prior close: explicit MISSING token.
        self.assertEqual(daily[171,2],1+2*5+3)
        self.assertEqual(float(bad[0].Close.iloc[170]),1e20)

    def test_missing_previous_state_uses_frequency_not_negative_index(self):
        y=np.array([[0,1],[1,1],[0,2],[2,3]])
        previous=np.array([[0,1],[-1,1],[1,1],[-1,-1]])
        splits=np.array(['train','train','val','test'])
        freq,markov=baseline(y,previous,splits)
        np.testing.assert_array_equal(markov[-1],freq[-1])

    def test_missing_summary_session_does_not_shorten_next_session_lag(self):
        loaded=fixtures.BrokerTokenTests().fixture();missing=170
        source=list(loaded)
        source[0]=loaded[0].drop(loaded[0].index[missing])
        source[1]=loaded[1].drop(loaded[0].index[missing])
        source[2]=loaded[2][:missing]+loaded[2][missing+1:]
        source[3]=np.delete(loaded[3],missing,axis=0)
        source[4]=loaded[4][:missing]+loaded[4][missing+1:]
        with tempfile.TemporaryDirectory() as temp:
            for d in loaded[0].index: (Path(temp)/str(d.date())).mkdir()
            aligned=align_loaded_calendar(source,temp)
        self.assertEqual(len(aligned[0]),208)
        self.assertFalse(aligned[1].iloc[missing]);self.assertIsNone(aligned[4][missing])
        _,daily,states,valid=relation_tokens(aligned,103,2,clean_ohlcv=True)
        _,_,_,targets,src,_=dataset(daily,states,valid,folds(208)[2],'next_day_peers')
        self.assertNotIn(missing,targets);self.assertNotIn(missing+1,targets)
        self.assertTrue((targets-src==1).all())

    def test_simultaneous_intervals_cover_pointwise_intervals(self):
        rng=np.random.default_rng(4)
        x=rng.normal(0,.1,(60,12)); x[:,0]+=.3
        x[8,3]=np.nan
        mean,lo,hi,std,critical=simultaneous_intervals(x,np.repeat([1,2,3],20),repeats=1500)
        np.testing.assert_allclose(mean,np.nanmean(x,axis=0))
        self.assertTrue((lo<=mean).all());self.assertTrue((hi>=mean).all())
        self.assertGreater(critical,1.96)
        self.assertGreater(lo[0],0)

    def test_relevance_requires_all_controls_and_temporal_consistency(self):
        rows=pd.DataFrame([dict(comparator=c,block_sessions=b,sim_ci_low=.01,nominal_ci_low=.02)
                           for c in CONTROLS for b in [3,5,10]])
        fold=pd.DataFrame({c:[.1,.2,-.01] for c in CONTROLS})
        self.assertEqual(classify(rows,fold,84),'didukung')
        rows.loc[0,'sim_ci_low']=-.01
        self.assertEqual(classify(rows,fold,84),'indikasi')
        rows.loc[(rows.comparator=='frequency')&(rows.block_sessions==5),'nominal_ci_low']=-.01
        self.assertEqual(classify(rows,fold,84),'belum')
        self.assertEqual(classify(rows,fold,30),'belum')

    def test_neighbor_stability_requires_cross_fold_and_control_gap(self):
        rows=[]
        for fold in [1,2,3]:
            for seed in [11,42,123,2026,3050]:
                for method in ['full','shifted_peers']:
                    # Query and candidate are both eligible in every run.
                    neighbor='BB_BUY_LARGE' if method=='full' else 'CC_BUY_LARGE'
                    rows.append(dict(task='same_day_masked',fold=fold,seed=seed,method=method,
                                     query='AA_BUY_LARGE',neighbor=neighbor))
                    rows.append(dict(task='same_day_masked',fold=fold,seed=seed,method=method,
                                     query='BB_BUY_LARGE',neighbor='CC_BUY_LARGE'))
        result=stable_pairs(pd.DataFrame(rows))
        pair=result[(result['query']=='AA_BUY_LARGE')&(result.neighbor=='BB_BUY_LARGE')].iloc[0]
        self.assertTrue(pair.stable_geometry);self.assertEqual(pair.seed_support,1)
        self.assertEqual(pair.shifted_support,0);self.assertEqual(pair.eligible_folds,3)

    def test_joint_counts_deduplicate_seeds_and_compact_probabilities_roundtrip(self):
        records=[]
        for seed in [11,42]:
            for i in range(10):
                for code,state in [('AA',2 if i<5 else 4),('BB',4 if i<5 else 2)]:
                    records.append(dict(task='same_day_masked',fold=1,method='full',seed=seed,
                        date=str(pd.Timestamp('2026-01-01')+pd.Timedelta(days=i))[:10],broker=code,actual=state,
                        **{s:float(k==state) for k,s in enumerate(['NET_ZERO','BUY_NORMAL','BUY_LARGE','SELL_NORMAL','SELL_LARGE'])}))
        frame=pd.DataFrame(records)
        pair=pd.DataFrame([dict(task='same_day_masked',query='AA_BUY_LARGE',neighbor='BB_SELL_LARGE')])
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);frame.to_csv(folder/'test_predictions.csv',index=False)
            result=same_day_cooccurrence(folder,pair).iloc[0]
            self.assertEqual(result.eligible_test_days,10)
            self.assertEqual(result.joint_test_days,5)
            self.assertAlmostEqual(result.cooccurrence_lift,2)
            compact_predictions(folder);(folder/'test_predictions.csv').unlink()
            other=read_predictions(folder)
            self.assertEqual(len(other),len(frame))
            np.testing.assert_array_equal(other.actual,frame.actual)

    def test_source_fingerprint_detects_data_change(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);summary=folder/'summary';brokers=folder/'brokers'
            summary.mkdir();brokers.mkdir();(summary/'ADRO_Summary.csv').write_text('data')
            day=brokers/'2026-01-02';day.mkdir();file=day/'ADRO.xls';file.write_bytes(b'first')
            before=source_fingerprint('ADRO',summary,brokers)
            file.write_bytes(b'second')
            self.assertNotEqual(before,source_fingerprint('ADRO',summary,brokers))

    def test_peer_contribution_requires_periods_and_seed_agreement(self):
        rows=[dict(task='same_day_masked',fold=f,peer='AA',target='BB',seed=s,gain_nll=.01 if f<3 else -.005)
              for f in [1,2,3] for s in range(5)]
        result=peer_summary(pd.DataFrame(rows)).iloc[0]
        self.assertTrue(result.stable_contribution);self.assertEqual(result.positive_folds,2)


if __name__=='__main__':unittest.main()
