"""Reproduce selected checkpoints and preserve model inputs for all stocks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from research import load_stock
from adro_token_embedding import EventEmbedding
from broker_relation_test import align_loaded_calendar,relation_tokens,dataset,predict,mask_inputs,STATES,write_report
from relation_io import compact_predictions,read_predictions,source_fingerprint


def audit(root):
    root=Path(root); torch.set_num_threads(1)
    batch=json.loads((root/'batch_config.json').read_text()); records=[]; quality=[]
    for ticker in batch['stocks']:
        folder=root/ticker
        if not (folder/'config.json').exists(): raise AssertionError(f'Incomplete stock: {ticker}')
        config=json.loads((folder/'config.json').read_text())
        assert config['ticker']==ticker
        loaded=align_loaded_calendar(load_stock(ticker,config['summary'],config['brokers']),config['brokers'])
        config['source_fingerprint']=source_fingerprint(ticker,config['summary'],config['brokers'])
        config['calendar_aligned']=True
        config['invalid_summary_dates']=[str(d.date()) for d in loaded[0].index[~loaded[1]]]
        compact_predictions(folder); exported=read_predictions(folder)
        runs=json.loads((folder/'training_runs.json').read_text())
        assert len(runs)==3*2*5*len(config['seeds'])
        source=pd.DataFrame({'date':[str(d.date()) for d in loaded[0].index],
                             'summary_valid':loaded[1].to_numpy(),
                             'broker_snapshot_available':[s is not None for s in loaded[4]]})
        source.to_csv(folder/'source_calendar_audit.csv',index=False)
        missing=sum(q.get('issue')=='missing summary calendar session' for q in loaded[-1])
        quality.append(dict(ticker=ticker,source_sessions=len(source),missing_summary_sessions=missing,
                            invalid_ohlcv_rows=int((~loaded[1]).sum())-missing,
                            unavailable_broker_snapshots=int((~source.broker_snapshot_available).sum())))
        for fold in config['folds']:
            tokenizer,daily,states,valid=relation_tokens(loaded,fold['train_stop'],12,ticker,True)
            assert tokenizer==json.loads((folder/f"fold-{fold['fold']}"/'tokenizer.json').read_text()),f'{ticker}: tokenizer changed'
            np.savez_compressed(folder/f"fold-{fold['fold']}"/'tokens.npz',daily=daily,states=states,valid=valid,
                                dates=np.array([str(d.date()) for d in loaded[0].index]))
            for task in ['same_day_masked','next_day_peers']:
                base,y,_,targets,sources,split=dataset(daily,states,valid,fold,task)
                checkpoint=torch.load(folder/f"fold-{fold['fold']}"/f'{task}.pt',map_location='cpu',weights_only=False)
                choices=[r for r in runs if r['fold']==fold['fold'] and r['task']==task and r['method']=='full']
                assert checkpoint['seed']==min(choices,key=lambda r:r['validation_nll'])['seed']
                model=EventEmbedding(len(tokenizer['tokens']),len(tokenizer['brokers']),8,1)
                model.load_state_dict(checkpoint['state_dict'])
                x=mask_inputs(base,len(tokenizer['brokers']),'full')
                for b in range(len(tokenizer['brokers'])):
                    assert not np.isin(x[:,b],np.arange(1+5*b,6+5*b)).any()
                p=predict(model,x[split=='test'])
                rows=exported[(exported.fold==fold['fold'])&(exported.task==task)&(exported.method=='full')&(exported.seed==checkpoint['seed'])]
                np.testing.assert_array_equal(rows.actual.to_numpy().reshape(y[split=='test'].shape),y[split=='test'])
                np.testing.assert_allclose(rows[STATES].to_numpy().reshape(p.shape),p,atol=1e-7)
                assert (targets-sources==(0 if task=='same_day_masked' else 1)).all()
                records.append(dict(ticker=ticker,fold=fold['fold'],task=task,test_dates=int((split=='test').sum()),
                    brokers=len(tokenizer['brokers']),selected_seed=checkpoint['seed'],checkpoint_matches=True,
                    target_absent=True,calendar_lag_correct=True))
        (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        write_report(folder,config,pd.read_csv(folder/'summary.csv'),pd.read_csv(folder/'paired_block_bootstrap.csv'),
                     pd.read_csv(folder/'neighbor_stability.csv'),pd.read_csv(folder/'peer_ablation.csv'))
        print('Verified',ticker,flush=True)
    pd.DataFrame(records).to_csv(root/'checkpoint_audit.csv',index=False)
    pd.DataFrame(quality).to_csv(root/'data_quality.csv',index=False)
    (root/'verification.json').write_text(json.dumps(dict(stocks=len(batch['stocks']),checkpoints=len(records),
        checked_inputs='target broker absent, original chronological calendar, train-only tokenizer',
        predictions='selected full checkpoint matches exported probability and actual label; compact npz uses float32',
        unit_tests='See relations-25-tests.log'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='runs/broker-relations-25')
    audit(parser.parse_args().out)
