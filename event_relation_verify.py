"""Rebuild temporal broker inputs and replay every end-to-end checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from event_relation_model import RelationModel,MODES
from event_relation_training import restore,infer,configure_runtime as runtime
from event_relation_study import load_arrays
from event_broker_geometry import dense_selected,state_labels
from event_market_study import load_raw
from event_market_data import sha,dump
from event_training import seed_all,weights_hash
from adro_token_embedding import scores


def verify(root,device='auto'):
    root=Path(root);config=json.loads((root/'config.json').read_text());config['device']=runtime(device)
    for name,digest in json.loads((root/'phase1_frozen.json').read_text()).items():assert sha(root/name)==digest
    for name,digest in json.loads((root/'code_manifest.json').read_text()).items():assert sha(root/'sources'/name)==digest
    for name,digest in json.loads((root/'input_source_manifest.json').read_text()).items():assert sha(root/name)==digest
    predictions=pd.read_csv(root/'predictions.csv');metrics=pd.read_csv(root/'metrics.csv')
    assert not predictions.duplicated(['ticker','fold','seed','method','date','broker']).any()
    rebuilt=0;cases=0;error=0.;cpu_error=0.;initials={};updated=0;frozen=0;datesets={}
    for ticker in config['tickers']:
        raw=load_raw(root/'encoder-source',ticker)
        for fold in config['folds']:
            data=load_arrays(root/ticker/f'fold-{fold}'/'dataset.npz')
            base=load_arrays(root/'source-base'/ticker/f'fold-{fold}'/'base_features.npz')
            saved=torch.load(next((root/'encoder-source'/ticker/f'fold-{fold}'/'local_only').glob('none-seed*/best.pt')),
                map_location='cpu',weights_only=False);state=saved['preprocess']
            tokens,present=dense_selected(raw,data['codes'],state);q=data['queries'];history=q[:,None]+np.arange(-4,1)
            np.testing.assert_array_equal(data['tokens'],tokens[history]);np.testing.assert_array_equal(data['mask'],present[history])
            states,thresholds,_,_=state_labels(raw,data['codes'],saved['training_indices'])
            np.testing.assert_array_equal(data['y'],states[q+1]);np.testing.assert_array_equal(thresholds,base['thresholds'])
            for target in range(len(data['codes'])):
                np.testing.assert_array_equal(data['own'][:,target],base[f'own_{target}'][:,:70].reshape(len(q),5,14)[:,::-1])
                np.testing.assert_array_equal(data['context'],base[f'own_{target}'][:,70:])
                np.testing.assert_array_equal(data['own'][:,target,:,5:13],tokens[history,target])
                expected=np.eye(5)[states[history,target]]*raw['mask'][history].any(-1)[...,None]
                np.testing.assert_array_equal(data['own'][:,target,:,:5],expected)
            for split in ['train','val','test']:
                local=q[data['splits']==split];eligible=set(map(int,saved['training_indices'] if split=='train' else
                    np.load(root/'encoder-source'/ticker/f'fold-{fold}'/'local_only'/'prepared.npz')[split+'_indices']))
                assert set(map(int,local+1))<=eligible
            train=q[data['splits']=='train'];val=q[data['splits']=='val'];test=q[data['splits']=='test']
            assert train.max()+40<val.min() and val.max()+40<test.min()
            datesets[ticker,fold]=set(map(int,test));rebuilt+=1
    fits=json.loads((root/'selected.json').read_text())
    for item in fits:
        path=root/item['path'];assert sha(path)==item['sha256'];assert sha(root/item['dataset'])==item['dataset_sha256']
        model,saved=restore(path,config['device']);record=saved['record'];data=load_arrays(root/item['dataset'])
        seed_all(item['seed']);initial=RelationModel(saved['vocabulary'],saved['context_features'],saved['width'],saved['mode'])
        assert weights_hash(initial.state_dict())==record['initial_sha256']
        assert weights_hash(initial.set_encoder.state_dict())==record['initial_encoder_sha256']
        assert weights_hash(model.set_encoder.state_dict())==record['best_encoder_sha256']
        assert all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
        changed=record['initial_encoder_sha256']!=record['best_encoder_sha256']
        if item['method'] in ['own_only','frozen_peers']:assert not changed;frozen+=1
        else:assert changed;updated+=1
        initials.setdefault((item['ticker'],item['fold'],item['seed']),[]).append(record['initial_sha256'])
        rows=np.flatnonzero(data['splits']=='val');p=infer(model,data,rows,config,item['method'],item['seed'])
        nll=float(-np.log(np.take_along_axis(p,data['y'][rows,...,None],axis=-1)).mean())
        np.testing.assert_allclose(nll,record['validation_nll'],atol=2e-6,rtol=2e-5)
        history=json.loads((path.parent/'history.json').read_text());best=float('inf');best_epoch=None
        for row in history:
            if row['validation_nll']<best-1e-6:best=row['validation_nll'];best_epoch=row['epoch']
        assert best_epoch==record['best_epoch'] and abs(best-record['validation_nll'])<1e-10
        rows=np.flatnonzero(data['splits']=='test');p=infer(model,data,rows,config,item['method'],item['seed'])
        part=predictions[(predictions.ticker==item['ticker'])&(predictions.fold==item['fold'])&
            (predictions.seed==item['seed'])&(predictions.method==item['method'])]
        for target,broker in enumerate(data['codes']):
            exported=part[part.broker==broker].sort_values('query_index')
            np.testing.assert_array_equal(exported.query_index.to_numpy(),data['queries'][rows])
            np.testing.assert_array_equal(exported.actual.to_numpy(),data['y'][rows,target])
            expected=exported[[f'p{k}' for k in range(5)]].to_numpy()
            np.testing.assert_allclose(p[:,target],expected,atol=2e-6,rtol=2e-5)
            error=max(error,float(abs(p[:,target]-expected).max()))
        row=metrics[(metrics.ticker==item['ticker'])&(metrics.fold==item['fold'])&
            (metrics.seed==item['seed'])&(metrics.method==item['method'])].iloc[0]
        for name,value in scores(data['y'][rows],p).items():np.testing.assert_allclose(value,row[name],atol=2e-6,rtol=2e-5)
        if config['device']=='cuda':
            model.to('cpu');cpu_config=dict(config,device='cpu',batch_size=8)
            cpu=infer(model,data,rows[:8],cpu_config,item['method'],item['seed'])
            # Shift mapping built across all rows, so even small CPU sample
            # preserves the exact offline-control alignment.
            np.testing.assert_allclose(cpu,p[:8],atol=2e-4,rtol=2e-4)
            cpu_error=max(cpu_error,float(abs(cpu-p[:8]).max()))
        cases+=1
    assert cases==config['planned_fits']
    assert all(len(values)==len(MODES) and len(set(values))==1 for values in initials.values())
    for ticker in config['tickers']:
        sets=[datesets[ticker,fold] for fold in config['folds']];assert len(set.union(*sets))==sum(map(len,sets))
    logs=pd.read_csv(root/'history.csv');assert logs.nonfinite_gradient_norms.sum()==logs.confirmed_scaler_skips.sum()
    result=dict(checkpoints_replayed=cases,datasets_rebuilt=rebuilt,matched_initial_groups=len(initials),
        updated_peer_encoders=updated,unchanged_control_encoders=frozen,maximum_export_error=error,
        maximum_cpu_gpu_probability_error=cpu_error,device=config['device'],
        full_target_history_mask_unit_tested=True,query_plus_one_label_verified=True,train_only_input_transform_verified=True,
        frozen_selections_and_hashes_verified=True,purge_embargo_disjoint_test_verified=True,
        confirmed_scaler_skips=int(logs.confirmed_scaler_skips.sum()),test_status=config['test_status'])
    dump(root/'verification.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    args=p.parse_args();verify(args.root,args.device)
