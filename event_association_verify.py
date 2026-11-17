"""Rebuild same-session association inputs, frozen controls and all predictions."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from event_association_study import prepare_data,assert_reuse,linear_features,NEW_MODES
from event_relation_model import RelationModel
from event_relation_training import restore,infer,configure_runtime
from event_relation_study import load_arrays
from event_broker_geometry import state_labels
from event_broker_probe import probabilities
from event_market_study import load_raw
from event_market_data import sha,dump
from event_training import seed_all,weights_hash
from adro_token_embedding import scores


def verify(root,device='auto'):
    root=Path(root);config=json.loads((root/'config.json').read_text());config['device']=configure_runtime(device)
    for n,h in json.loads((root/'phase1_frozen.json').read_text()).items():assert sha(root/n)==h
    for manifest,prefix in [('input_source_manifest.json',root),('data_manifest.json',root),('code_manifest.json',root/'sources')]:
        for n,h in json.loads((root/manifest).read_text()).items():assert sha(prefix/n)==h
    pred=pd.read_csv(root/'predictions.csv');metric=pd.read_csv(root/'metrics.csv')
    assert not pred.duplicated(['ticker','fold','seed','method','date','broker']).any()
    p=pred[[f'p{k}' for k in range(5)]].to_numpy()
    assert np.isfinite(p).all() and (p>0).all()
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-12,rtol=1e-12)
    pred['nll']=-np.log(p[np.arange(len(p)),pred.actual.to_numpy()])
    group=['ticker','fold','seed','method']
    joined=pred.groupby(group).nll.mean().rename('export_nll').reset_index().merge(metric,on=group,validate='one_to_one')
    np.testing.assert_allclose(joined.export_nll,joined.nll,atol=1e-11,rtol=1e-10)
    datasets=0;date_sets={};cache={}
    for ticker in config['tickers']:
        raw=load_raw(root/'encoder-source',ticker)
        for fold in config['folds']:
            folder=root/ticker/f'fold-{fold}';data=load_arrays(folder/'dataset.npz')
            old=load_arrays(root/'source-control'/ticker/f'fold-{fold}'/'dataset.npz');assert_reuse(old,data)
            origin=root/'encoder-source'/ticker/f'fold-{fold}'/'local_only'
            with np.load(origin/'prepared.npz',allow_pickle=False) as z:prepared={k:z[k] for k in z.files}
            saved=torch.load(next(origin.glob('none-seed*/best.pt')),map_location='cpu',weights_only=False)
            rebuilt,scaler=prepare_data(old,raw,prepared['train_indices'])
            for name in data:np.testing.assert_array_equal(data[name],rebuilt[name])
            assert scaler==json.loads((folder/'peer_scaler.json').read_text())
            states,cuts,_,_=state_labels(raw,data['codes'],prepared['train_indices'])
            np.testing.assert_array_equal(data['y'],states[data['queries']])
            # Independent calendar/history membership, including current peer date.
            assert (data['queries']>=5).all()
            for split in ['train','val','test']:
                q=data['queries'][data['splits']==split]
                assert set(map(int,q))<=set(map(int,prepared[split+'_indices']))
            train=data['queries'][data['splits']=='train'];val=data['queries'][data['splits']=='val'];test=data['queries'][data['splits']=='test']
            assert train.max()+40<val.min() and val.max()+40<test.min()
            date_sets[ticker,fold]=set(map(int,test));cache[ticker,fold]=data;datasets+=1
        local=pred[pred.ticker==ticker]
        dates=np.asarray(json.loads((root/'encoder-source'/f'{ticker}-data-audit.json').read_text())['dates'])
        np.testing.assert_array_equal(dates[local.query_index.to_numpy()],local.date.to_numpy())
    cases=0;updated=0;frozen=0;error=0.;cpu_error=0.;initials={}
    def check(context,data,rows,prob):
        nonlocal error
        part=pred
        for k,v in context.items():part=part[part[k]==v]
        for target,broker in enumerate(data['codes']):
            sub=part[part.broker==broker].sort_values('query_index')
            np.testing.assert_array_equal(sub.query_index,data['queries'][rows])
            np.testing.assert_array_equal(sub.actual,data['y'][rows,target])
            expected=sub[[f'p{k}' for k in range(5)]].to_numpy()
            np.testing.assert_allclose(prob[:,target],expected,atol=2e-6,rtol=2e-5)
            error=max(error,float(abs(prob[:,target]-expected).max()))
        record=metric
        for k,v in context.items():record=record[record[k]==v]
        assert len(record)==1
        for k,v in scores(data['y'][rows],prob).items():np.testing.assert_allclose(v,record.iloc[0][k],atol=2e-6,rtol=2e-5)
    for item in json.loads((root/'selected.json').read_text()):
        assert sha(root/item['path'])==item['sha256'];assert sha(root/item['dataset'])==item['dataset_sha256']
        data=cache[item['ticker'],item['fold']];model,saved=restore(root/item['path'],config['device']);r=saved['record']
        seed_all(item['seed']);initial=RelationModel(saved['vocabulary'],saved['context_features'],saved['width'],item['method'])
        assert weights_hash(initial.state_dict())==r['initial_sha256']
        assert weights_hash(initial.set_encoder.state_dict())==r['initial_encoder_sha256']
        assert weights_hash(model.set_encoder.state_dict())==r['best_encoder_sha256']
        changed=r['initial_encoder_sha256']!=r['best_encoder_sha256']
        if item['method'] in ['own_only','frozen_peers']:assert not changed;frozen+=1
        else:assert changed;updated+=1
        assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
        initials.setdefault((item['ticker'],item['fold'],item['seed']),[]).append(r['initial_sha256'])
        val=np.flatnonzero(data['splits']=='val');pv=infer(model,data,val,config,item['method'],item['seed'])
        nll=float(-np.log(np.take_along_axis(pv,data['y'][val,...,None],axis=-1)).mean())
        np.testing.assert_allclose(nll,r['validation_nll'],atol=2e-6,rtol=2e-5)
        if not item['reused']:
            history=json.loads((root/item['path']).with_name('history.json').read_text());best=float('inf');best_epoch=0
            for row in history:
                if row['validation_nll']<best-1e-6:best=row['validation_nll'];best_epoch=row['epoch']
            assert r['best_epoch']==best_epoch and abs(best-r['validation_nll'])<1e-10
        rows=np.flatnonzero(data['splits']=='test');prob=infer(model,data,rows,config,item['method'],item['seed'])
        check({k:item[k] for k in group},data,rows,prob)
        if config['device']=='cuda':
            model.to('cpu');cpu=infer(model,data,rows[:8],dict(config,device='cpu',batch_size=8),item['method'],item['seed'])
            np.testing.assert_allclose(cpu,prob[:8],atol=2e-4,rtol=2e-4);cpu_error=max(cpu_error,float(abs(cpu-prob[:8]).max()))
        cases+=1
    heads=json.loads((root/'selected_heads.json').read_text());head_count=0
    for ticker in config['tickers']:
        for fold in config['folds']:
            data=cache[ticker,fold];rows=np.flatnonzero(data['splits']=='test');val=data['splits']=='val'
            for method in ['raw_linear','aggregate_linear']:
                prob=[]
                for item in sorted([x for x in heads if (x['ticker'],x['fold'],x['method'])==(ticker,fold,method)],key=lambda x:x['target']):
                    assert sha(root/item['path'])==item['sha256']
                    with (root/item['path']).open('rb') as f:head=pickle.load(f)
                    x=linear_features(data,item['target'],method)
                    if head['scaler'] is not None:
                        np.testing.assert_allclose(head['scaler'].mean_,x[data['splits']=='train'].mean(0,dtype='float64'),atol=1e-10,rtol=1e-10)
                    vp=probabilities(head,x[val]);nll=-np.log(vp[np.arange(val.sum()),data['y'][val,item['target']]]).mean()
                    np.testing.assert_allclose(nll,item['validation_nll'],atol=1e-10,rtol=1e-10)
                    trials=pd.read_csv(root/'head_trials.csv')
                    trials=trials[(trials.ticker==ticker)&(trials.fold==fold)&(trials.method==method)&(trials.target==item['target'])]
                    np.testing.assert_allclose(nll,trials.nll.min(),atol=1e-10,rtol=1e-10)
                    prob.append(probabilities(head,x[rows]));head_count+=1
                check(dict(ticker=ticker,fold=fold,seed=-1,method=method),data,rows,np.stack(prob,axis=1))
    prior=pd.read_csv(root/'prior_predictions.csv');prior.query_index+=1
    reused=pred[pred.method.isin(prior.method.unique())]
    joined=prior.merge(reused,on=group+['date','broker','query_index'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(joined)==len(prior)==len(reused)
    for name in ['actual']+[f'p{k}' for k in range(5)]:
        np.testing.assert_allclose(joined[name+'_old'],joined[name+'_new'],atol=2e-6,rtol=2e-5)
    assert all(len(v)==4 and len(set(v))==1 for v in initials.values())
    for ticker in config['tickers']:
        sets=[date_sets[ticker,fold] for fold in config['folds']];assert len(set.union(*sets))==sum(map(len,sets))
    h=pd.read_csv(root/'history.csv');assert h.nonfinite_gradient_norms.sum()==h.confirmed_scaler_skips.sum()
    result=dict(checkpoints_replayed=cases,new_checkpoints=cases*3//4,reused_own_controls=cases//4,
        linear_heads_replayed=head_count,datasets_rebuilt=datasets,matched_initial_groups=len(initials),
        updated_peer_encoders=updated,unchanged_control_encoders=frozen,maximum_export_error=error,
        maximum_cpu_gpu_probability_error=cpu_error,confirmed_scaler_skips=int(h.confirmed_scaler_skips.sum()),
        target_date_and_train_only_scalers_verified=True,old_own_features_labels_and_predictions_reused_verified=True,
        no_global_share_peer_features=True,device=config['device'],test_status=config['test_status'])
    dump(root/'verification.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--device',default='auto')
    args=p.parse_args();verify(args.root,args.device)
