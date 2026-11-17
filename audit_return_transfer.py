"""Reproduce all return heads from frozen encoders and original/snapshot data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from embedding_returns import (price_features,broker_arrays,pool_encoder,design,head_predict,samples,
                                HORIZONS,PRICE_NAMES,ALPHAS,LAYOUTS)
from research import load_stock
from broker_relation_test import align_loaded_calendar,relation_tokens
from adro_token_embedding import EventEmbedding
from relation_io import source_fingerprint


def audit(root):
    root=Path(root);plan=json.loads((root/'experiment_plan.json').read_text())
    rows=[];encoders=[];baseline_count=0
    torch.set_num_threads(1)
    for ticker in plan['tickers']:
        folder=root/ticker;config=json.loads((folder/'config.json').read_text())
        assert config['source_fingerprint']==source_fingerprint(ticker,config['summary_path'],config['brokers_path'])
        loaded=align_loaded_calendar(load_stock(ticker,config['summary_path'],config['brokers_path']),config['brokers_path'])
        price=price_features(loaded);predictions=pd.read_csv(folder/'test_predictions.csv')
        heads=pd.read_csv(folder/'selected_heads.csv');snapshot=np.load(folder/'source_snapshot.npz',allow_pickle=False)
        np.testing.assert_array_equal(snapshot['close'],loaded[0].Close.to_numpy())
        for fold in config['folds']:
            fdir=folder/f"fold-{fold['fold']}";ids,split,y=samples(loaded,fold)
            target_snapshot=np.load(fdir/'targets.npz',allow_pickle=False)
            np.testing.assert_array_equal(ids,target_snapshot['indices']);np.testing.assert_allclose(y,target_snapshot['targets'],atol=0,rtol=0)
            assert int((ids[split=='train']+10).max())<int(ids[split=='val'].min())
            assert int((ids[split=='val']+10).max())<int(ids[split=='test'].min())
            tokenizer=json.loads((fdir/'tokenizer.json').read_text())
            expected,daily,states,_=relation_tokens(loaded,int(fold['train_stop']*.8),12,ticker,True)
            assert tokenizer==expected
            broker,names=broker_arrays(loaded,tokenizer,states)
            np.savez_compressed(fdir/'feature_snapshot.npz',price=price,**broker)
            (fdir/'feature_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
            models={};latent_cache={}
            for seed in config['seeds']:
                ckpt=torch.load(fdir/f'encoder-seed{seed}.pt',map_location='cpu',weights_only=False)
                record=ckpt['record']
                assert pd.Timestamp(record['last_embedding_validation_date'])<=loaded[0].index[fold['train_stop']-1]
                assert record['inner_train_stop']==int(fold['train_stop']*.8)
                model=EventEmbedding(len(tokenizer['tokens']),len(tokenizer['brokers']),8,1)
                model.load_state_dict(ckpt['state_dict'])
                latent_cache['learned_embedding',seed]=pool_encoder(model,daily,states,len(tokenizer['brokers']))
                torch.manual_seed(seed);random=EventEmbedding(len(tokenizer['tokens']),len(tokenizer['brokers']),8,1)
                latent_cache['random_embedding',seed]=pool_encoder(random,daily,states,len(tokenizer['brokers']))
                encoders.append(dict(ticker=ticker,fold=fold['fold'],seed=seed,last_embedding_date=record['last_embedding_validation_date'],
                                     first_price_validation_date=str(loaded[0].index[fold['val_start']].date()),past_only=True))
            for row in heads[heads.fold==fold['fold']].to_dict('records'):
                method=row['method'];seed=int(row['seed']);h=int(row['horizon'])
                file=fdir/f'{method}-h{h}-seed{seed}.npz'
                data=np.load(file,allow_pickle=False);head={k:data[k] for k in data.files}
                layout=tuple(int(v) for v in head['layout'])
                assert float(head['alpha']) in ALPHAS
                if method=='ohlcv':matrix=price;assert layout==()
                elif method in ['broker_aggregates','broker_direct','broker_onehot']:
                    key={'broker_direct':'direct','broker_onehot':'onehot'}.get(method)
                    matrix,_=design(price,broker['aggregate'],None if key is None else broker[key],[] if key is None else names[key],layout)
                else:
                    matrix,_=design(price,broker['aggregate'],latent_cache[method,seed],[f'latent_{i+1}' for i in range(8)],layout)
                p=head_predict(matrix[ids[split=='test']],head)
                exported=predictions[(predictions.fold==fold['fold'])&(predictions.method==method)&(predictions.seed==seed)&(predictions.horizon==h)]
                np.testing.assert_allclose(p,exported.prediction.to_numpy(),atol=1e-12)
                np.testing.assert_allclose(y[split=='test',HORIZONS.index(h)],exported.actual.to_numpy(),atol=1e-12)
                validation_mae=float(np.abs(head_predict(matrix[ids[split=='val']],head)-y[split=='val',HORIZONS.index(h)]).mean())
                assert abs(validation_mae-float(head['validation_mae']))<1e-12
                rows.append(dict(ticker=ticker,fold=fold['fold'],method=method,seed=seed,horizon=h,
                                 test_dates=len(p),prediction_matches=True,validation_score_matches=True,purged=True))
            for method in ['zero','training_median']:
                for j,h in enumerate(HORIZONS):
                    expected=0. if method=='zero' else np.median(y[split=='train',j])
                    exported=predictions[(predictions.fold==fold['fold'])&(predictions.method==method)&(predictions.horizon==h)]
                    np.testing.assert_allclose(exported.prediction.to_numpy(),expected,atol=1e-12)
                    baseline_count+=1
        print('Verified',ticker,flush=True)
    pd.DataFrame(rows).to_csv(root/'head_audit.csv',index=False)
    pd.DataFrame(encoders).to_csv(root/'encoder_time_audit.csv',index=False)
    verification=dict(stocks=len(plan['tickers']),frozen_encoders=len(encoders),return_heads=len(rows),baseline_checks=baseline_count,
        outcome='All saved head predictions/validation scores reproduce; purged price splits and encoder time limits verified',
        source='fingerprints match raw local data; feature/token/target snapshots retained for reproduction',
        note='Does not turn historically reused test data into new confirmation')
    (root/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(verification,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='runs/embedding-returns-v1')
    with threadpool_limits(limits=1):audit(parser.parse_args().out)
