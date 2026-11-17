"""Verify every paired checkpoint and independently rebuild Yahoo feature alignment."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from event_market_data import augment,sha,dump
from event_market_study import load_raw,ARMS
from event_objective_study import runtime
from event_training import infer,decoded,weights_hash
from event_predict import restore
from event_data import preprocess


def verify(root,device='auto'):
    root=Path(root);device=runtime(device);config=json.loads((root/'config.json').read_text())
    for name,digest in json.loads((root/'phase1_frozen.json').read_text()).items():assert sha(root/name)==digest
    for name,digest in json.loads((root/'code_manifest.json').read_text()).items():assert sha(root/'sources'/name)==digest
    for name,digest in json.loads((root/'market-source/manifest.json').read_text()).items():assert sha(root/'market-source'/name)==digest
    predictions=pd.read_csv(root/'predictions.csv');fits=json.loads((root/'selected_checkpoints.json').read_text())
    assert len(fits)==config['planned_fits']
    assert not predictions.duplicated(['ticker','fold','seed','model','date','horizon']).any()
    prepared_count=0;count=0;cpu_error=0.;error=0.;initial={};testsets={}
    for ticker in config['tickers']:
        raw=load_raw(root,ticker);full,receipts=augment(raw,root/'market-source',raw['frame'].Close)
        cache=np.load(root/f'{ticker}-augmented.npz');np.testing.assert_array_equal(full['x'],cache['x'])
        for receipt in receipts:
            joined=receipt['joins'];observed=joined.source_date.notna()
            assert (joined.loc[observed,'source_available_at']<=joined.loc[observed,'query_date']+pd.Timedelta(hours=23,minutes=59)).all()
        for fold in config['folds']:
            for arm in ARMS:
                folder=root/ticker/f'fold-{fold}'/arm;arrays=np.load(folder/'prepared.npz',allow_pickle=False)
                saved=torch.load(folder/'preprocess.pt',map_location='cpu',weights_only=False)
                current=dict(full);current['x']=full['x'].copy()
                if arm=='local_only':current['x'][:,len(raw['names']):]=0
                rebuilt,state=preprocess(current,arrays['train_indices'],config['window'])
                for key in ['x','tokens','ids','mask','y']:np.testing.assert_array_equal(rebuilt[key],arrays[key])
                for key in ['mean','std','token_mean','token_std','target_mean','target_std','constant_columns','fit_days']:
                    np.testing.assert_array_equal(state[key],saved['preprocess'][key])
                assert state['vocabulary']==saved['preprocess']['vocabulary'];prepared_count+=1
                train,val,test=[arrays[s+'_indices'] for s in ['train','val','test']]
                assert train.max()+20+config['embargo']<val.min() and val.max()+20+config['embargo']<test.min()
                testsets[ticker,fold]=set(map(int,test))
    for item in fits:
        path=root/item['path'];assert sha(path)==item['checkpoint_sha256']
        model,saved=restore(path,device);state=saved['preprocess']
        assert weights_hash(saved['state_dict'])==saved['record']['best_weights_sha256']
        assert all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values())
        arrays=np.load(path.parents[1]/'prepared.npz',allow_pickle=False)
        assert sha(path.parents[1]/'prepared.npz')==item['prepared_sha256']
        data={k:arrays[k] for k in ['x','tokens','ids','mask','y']}
        np.testing.assert_array_equal(arrays['train_indices'],saved['training_indices'])
        np.testing.assert_array_equal(arrays['val_indices'],saved['validation_indices'])
        initial.setdefault((item['ticker'],item['fold'],item['seed']),[]).append(saved['record']['initial_weights_sha256'])
        settings=SimpleNamespace(window=saved['window'],batch_size=config['batch_size'],device=device)
        val=arrays['val_indices'];out,_=infer(model,data,val,settings,state)
        np.testing.assert_allclose(abs(decoded(out,state)[0]-data['y'][val,:3]).mean(),saved['record']['val_loss'],atol=2e-6,rtol=2e-5)
        history=json.loads((path.parent/'history.json').read_text());best=float('inf');epoch=None
        for row in history:
            if row['val_loss']<best-1e-6:best=row['val_loss'];epoch=row['epoch']
        assert epoch==saved['record']['best_epoch'] and abs(best-saved['record']['val_loss'])<1e-9
        rows=arrays['test_indices'];out,_=infer(model,data,rows,settings,state);values=decoded(out,state)
        part=predictions[(predictions.ticker==item['ticker'])&(predictions.fold==item['fold'])
            &(predictions.seed==item['seed'])&(predictions.model==item['model'])]
        for column,value in zip(['prediction','probability','predicted_vol'],values):
            expected=part.pivot(index='calendar_index',columns='horizon',values=column).reindex(rows).to_numpy()
            np.testing.assert_allclose(value,expected,atol=2e-5,rtol=2e-5);error=max(error,float(abs(value-expected).max()))
        if device=='cuda':
            model.to('cpu');settings.device='cpu';settings.batch_size=8
            cpu=decoded(infer(model,data,rows,settings,state)[0],state)
            for value,expected in zip(cpu,values):
                np.testing.assert_allclose(value,expected,atol=2e-4,rtol=2e-4);cpu_error=max(cpu_error,float(abs(value-expected).max()))
        count+=1
    assert all(len(values)==2 and len(set(values))==1 for values in initial.values())
    for ticker in config['tickers']:
        sets=[testsets[ticker,fold] for fold in config['folds']]
        assert len(set.union(*sets))==sum(map(len,sets))
    history=pd.read_csv(root/'history.csv')
    assert history.nonfinite_gradient_steps.sum()==history.confirmed_scaler_skips.sum()
    result=dict(checkpoints_replayed=count,prepared_transforms_rebuilt_exactly=prepared_count,
        paired_initializations_verified=len(initial),maximum_replay_error=error,maximum_cpu_gpu_error=cpu_error,
        device=device,all_checkpoint_tensors_finite=True,train_only_preprocessing_verified=True,
        purge_embargo_disjoint_test_verified=True,conservative_availability_verified=True,
        frozen_source_code_and_checkpoint_hashes_verified=True,
        nonfinite_norm_minibatches=int(history.nonfinite_gradient_steps.sum()),
        confirmed_scaler_skips=int(history.confirmed_scaler_skips.sum()),
        test_status=config['test_status'])
    dump(root/'verification.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    args=p.parse_args();verify(args.root,args.device)
