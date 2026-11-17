"""Rebuild frozen representations, replay all heads and label timing."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from event_objective_study import runtime
from event_market_study import load_raw
from event_market_data import sha,dump
from event_training import weights_hash
from event_predict import restore
from event_broker_geometry import initial_model,select_brokers
from event_broker_probe import build_cache,read_cache,query_splits,all_features,probabilities,METHODS,BENCHMARKS
from broker_relation_test import baseline as legacy_baseline
from adro_token_embedding import scores


def verify(root,device='auto'):
    root=Path(root);device=runtime(device);config=json.loads((root/'config.json').read_text())
    for name,digest in json.loads((root/'phase1_frozen.json').read_text()).items():assert sha(root/name)==digest
    for name,digest in json.loads((root/'code_manifest.json').read_text()).items():assert sha(root/'sources'/name)==digest
    if (root/'encoder_source_manifest.json').exists():
        for name,digest in json.loads((root/'encoder_source_manifest.json').read_text()).items():
            assert sha(root/'encoder-source'/name)==digest
    manifest=json.loads((root/'encoder_manifest.json').read_text());heads=json.loads((root/'selected_heads.json').read_text())
    preds=pd.read_csv(root/'predictions.csv');metrics=pd.read_csv(root/'metrics.csv');trials=pd.read_csv(root/'head_trials.csv')
    assert not preds.duplicated(['ticker','fold','seed','method','date','broker']).any()
    maximum=0.;count=0;cases=0;cpu_error=0.;test_dates={};expected_heads=0
    for item in manifest:
        ticker,fold,seed=item['ticker'],item['fold'],item['seed'];checkpoint=root/item['path']
        assert sha(checkpoint)==item['sha256']
        model,saved=restore(checkpoint,device);random=initial_model(saved).to(device).eval()
        assert weights_hash(model.state_dict())==item['state_sha256']==saved['record']['best_weights_sha256']
        assert weights_hash(random.state_dict())==item['initial_state_sha256']==saved['record']['initial_weights_sha256']
        assert all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
        original=root/'encoder-source'/ticker/f'fold-{fold}'/'local_only'
        with np.load(original/'prepared.npz',allow_pickle=False) as z:data={key:z[key] for key in z.files}
        np.testing.assert_array_equal(saved['training_indices'],data['train_indices'])
        raw=load_raw(root/'encoder-source',ticker);split=query_splits(data,raw)
        queries=np.concatenate(list(split.values()));splits=np.concatenate([np.repeat(key,len(rows)) for key,rows in split.items()])
        codes,_=select_brokers(raw,data['train_indices']);destination=root/ticker/f'fold-{fold}'/f'seed-{seed}'
        expected_heads+=len(codes)*len(METHODS)
        assert sha(destination/'features.npz')==item['features_sha256']
        assert sha(destination.parent/'base_features.npz')==item['base_sha256']
        cached=read_cache(destination);rebuilt=build_cache(model,random,data,raw,codes,saved['preprocess'],queries,splits,seed,device)
        for name in cached:
            if cached[name].dtype.kind in 'f':np.testing.assert_allclose(cached[name],rebuilt[name],atol=2e-6,rtol=2e-6)
            else:np.testing.assert_array_equal(cached[name],rebuilt[name])
        assert split['train'].max()+40<split['val'].min()
        assert split['val'].max()+40<split['test'].min()
        test=splits=='test';val=splits=='val';y=cached['y'];method_p={}
        test_dates[ticker,fold]=set(str(raw['dates'][q+1].date()) for q in queries[test])
        for method in METHODS:
            values=[]
            for target,broker in enumerate(codes):
                record=next(row for row in heads if row['ticker']==ticker and row['fold']==fold and row['seed']==seed
                            and row['broker']==broker and row['method']==method)
                path=root/record['path'];assert sha(path)==record['sha256']
                with path.open('rb') as f:head=pickle.load(f)
                x=all_features(cached,queries,splits,target,method);p=probabilities(head,x)
                loss=float(-np.log(p[val][np.arange(val.sum()),y[val,target]]).mean())
                np.testing.assert_allclose(loss,record['validation_nll'],atol=1e-11,rtol=1e-10)
                attempts=trials[(trials.ticker==ticker)&(trials.fold==fold)&(trials.seed==seed)&
                    (trials.broker==broker)&(trials.method==method)]
                np.testing.assert_allclose(loss,attempts.nll.min(),atol=1e-11,rtol=1e-10)
                if head['constant'] is None:
                    np.testing.assert_allclose(head['scaler'].mean_,np.mean(x[splits=='train'],axis=0,dtype=float),atol=1e-10,rtol=1e-10)
                values.append(p[test]);count+=1
            method_p[method]=np.stack(values,axis=1)
        constant,markov=legacy_baseline(y,cached['previous'],splits)
        persistence=np.full((len(y),len(codes),5),.025);np.put_along_axis(persistence,cached['previous'][...,None],.9,axis=2)
        method_p.update(frequency=constant[test],markov=markov[test],persistence=persistence[test])
        for method,p in method_p.items():
            if method in BENCHMARKS and seed!=config['seeds'][0]:continue
            export_seed=seed if method in METHODS else -1
            part=preds[(preds.ticker==ticker)&(preds.fold==fold)&(preds.seed==export_seed)&(preds.method==method)]
            for j,broker in enumerate(codes):
                actual=part[part.broker==broker].sort_values('query_index')
                np.testing.assert_array_equal(actual.query_index.to_numpy(),queries[test])
                np.testing.assert_array_equal(actual.actual.to_numpy(),y[test,j])
                expected=p[:,j];export=actual[[f'p{k}' for k in range(5)]].to_numpy()
                np.testing.assert_allclose(expected,export,atol=1e-12,rtol=1e-10)
                maximum=max(maximum,float(abs(expected-export).max()))
            score=scores(y[test],p)
            row=metrics[(metrics.ticker==ticker)&(metrics.fold==fold)&(metrics.seed==export_seed)&(metrics.method==method)].iloc[0]
            for key,value in score.items():np.testing.assert_allclose(value,row[key],atol=1e-11,rtol=1e-10)
        if device=='cuda':
            from event_broker_geometry import subset_pool,dense_selected
            tokens,present=dense_selected(raw,codes,saved['preprocess']);model.to('cpu');sample=queries[test][:8]
            cpu=subset_pool(model.set_encoder,tokens,present,codes,saved['preprocess'],sample,0,'cpu')
            reference=cached['learned'][test][:8,0]
            np.testing.assert_allclose(cpu,reference,atol=2e-4,rtol=2e-4)
            cpu_error=max(cpu_error,float(abs(cpu-reference).max()))
        assert weights_hash(model.state_dict())==item['state_sha256'];cases+=1
    for ticker in config['tickers']:
        sets=[test_dates[ticker,f] for f in config['folds']];assert len(set.union(*sets))==sum(map(len,sets))
    assert count==len(heads)==expected_heads
    result=dict(encoder_cases_rebuilt=cases,selected_heads_replayed=count,maximum_probability_export_error=maximum,
        maximum_cpu_gpu_feature_error=cpu_error,device=device,all_backbones_unchanged=True,exact_initialization_verified=True,
        train_only_broker_selection_thresholds_scalers_verified=True,next_session_labels_same_split_and_disjoint_test_verified=True,
        all_head_choices_replay_validation_minimum=True,source_and_checkpoint_hashes_verified=True,test_status=config['test_status'])
    dump(root/'verification.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    args=p.parse_args();verify(args.root,args.device)
