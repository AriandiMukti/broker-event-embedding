"""Replay objective-study checkpoints, common validation selection, and paired inputs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from event_objective_study import runtime,sha,write_json
from event_predict import restore
from event_training import infer,decoded,weights_hash


def verify(root,device='auto'):
    root=Path(root);device=runtime(device);config=json.loads((root/'config.json').read_text())
    frozen=json.loads((root/'phase1_frozen.json').read_text())
    for name,key in [('selected_checkpoints.json','selected_checkpoints_sha256'),('data_manifest.json','data_manifest_sha256'),
                     ('config.json','config_sha256'),('code_manifest.json','code_manifest_sha256')]:
        assert sha(root/name)==frozen[key],f'Freeze receipt mismatch: {name}'
    for name,digest in json.loads((root/'code_manifest.json').read_text()).items():
        assert sha(root/'sources'/name)==digest,name
    fits=json.loads((root/'selected_checkpoints.json').read_text())
    assert len(fits)==config['planned_fits']==frozen['fits']
    preds=pd.read_csv(root/'predictions.csv');pairs=pd.read_csv(root/'pair_proof.csv')
    assert not preds.duplicated(['ticker','fold','seed','horizon','backbone','training_objective','date']).any()
    maximum=0.;cpu_max=0.;count=0;test_sets={};groups={}
    for item in fits:
        checkpoint=root/item['checkpoint']/'best.pt'
        assert sha(checkpoint)==item['checkpoint_sha256']
        model,saved=restore(checkpoint,device);state=saved['preprocess'];record=saved['record']
        assert weights_hash(saved['state_dict'])==record['best_weights_sha256']
        assert saved['training_objective']==item['training_objective']
        assert saved['selection_metric']=='return_mae'
        folder=checkpoint.parents[1];assert sha(folder/'prepared.npz')==item['prepared_sha256']
        cache=np.load(folder/'prepared.npz',allow_pickle=False)
        data={k:cache[k] for k in ['x','tokens','ids','mask','y']};data['groups']=saved['groups']
        train,val,test=[cache[k+'_indices'] for k in ['train','val','test']]
        np.testing.assert_array_equal(saved['training_indices'],train)
        np.testing.assert_array_equal(saved['validation_indices'],val)
        assert train.max()+20+config['embargo']<val.min()
        assert val.max()+20+config['embargo']<test.min()
        np.testing.assert_allclose(state['target_mean'],data['y'][train].mean(0),atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(state['target_std'],data['y'][train].std(0).clip(.001),atol=1e-7,rtol=1e-6)
        key=(item['ticker'],item['fold'],item['seed'],item['backbone'])
        groups.setdefault(key,{})[item['training_objective']]=record['initial_weights_sha256']
        test_sets[(item['ticker'],item['fold'])]=set(map(int,test))
        history=json.loads((checkpoint.parent/'history.json').read_text());best=float('inf');best_epoch=None
        for row in history:
            assert abs(row['val_loss']-row['val_return_mae'])<1e-8
            if row['val_loss']<best-1e-6:best=row['val_loss'];best_epoch=row['epoch']
        assert best_epoch==record['best_epoch']==item['best_epoch']
        assert abs(best-record['val_loss'])<1e-9
        settings=SimpleNamespace(window=saved['window'],batch_size=config['batch_size'],device=device)
        output,_=infer(model,data,val,settings,state);returns=decoded(output,state)[0]
        mae=float(abs(returns-data['y'][val,:3]).mean())
        np.testing.assert_allclose(mae,record['val_loss'],atol=2e-6,rtol=2e-5)
        output,_=infer(model,data,test,settings,state);returns=decoded(output,state)[0]
        part=preds[(preds.ticker==item['ticker'])&(preds.fold==item['fold'])&(preds.seed==item['seed'])
            &(preds.backbone==item['backbone'])&(preds.training_objective==item['training_objective'])]
        expected=part.pivot(index='calendar_index',columns='horizon',values='prediction').reindex(test).to_numpy()
        actual=part.pivot(index='calendar_index',columns='horizon',values='actual').reindex(test).to_numpy()
        np.testing.assert_allclose(actual,data['y'][test,:3],atol=1e-15,rtol=1e-14)
        np.testing.assert_allclose(returns,expected,atol=2e-5,rtol=2e-5)
        maximum=max(maximum,float(abs(returns-expected).max()))
        if device=='cuda':
            model.to('cpu');settings.device='cpu';settings.batch_size=8
            cpu=decoded(infer(model,data,test,settings,state)[0],state)[0]
            np.testing.assert_allclose(cpu,returns,atol=2e-4,rtol=2e-4)
            cpu_max=max(cpu_max,float(abs(cpu-returns).max()))
        count+=1
    for key,initial in groups.items():
        assert set(initial)=={'multitask','return_only'} and len(set(initial.values()))==1
        ticker,fold,seed,mode=key
        pair=pairs[(pairs.ticker==ticker)&(pairs.fold==fold)&(pairs.seed==seed)&(pairs.backbone==mode)]
        assert len(pair)==1 and pair.iloc[0].initial_weights_sha256==initial['return_only']
    assert len(groups)==len(pairs)==frozen['paired_initializations']
    for ticker in config['tickers']:
        sets=[test_sets[ticker,fold] for fold in config['folds']]
        assert len(set.union(*sets))==sum(map(len,sets))
    baseline=pd.read_csv(root/'baseline_source_predictions.csv')
    provenance=json.loads((root/'baseline_provenance.json').read_text())
    assert sha(root/'baseline_source_predictions.csv')==provenance['snapshot_sha256']
    baseline_part=preds[preds.backbone=='baseline']
    assert len(baseline_part)==len(baseline)
    join=baseline_part.merge(baseline,left_on=['ticker','fold','seed','horizon','date','training_objective'],
        right_on=['ticker','fold','seed','horizon','date','model'],suffixes=('_new','_source'),validate='one_to_one')
    for column in ['actual','prediction']:
        np.testing.assert_allclose(join[column+'_new'],join[column+'_source'],atol=1e-14,rtol=1e-14)
    result=dict(checkpoints_replayed=count,paired_initializations_verified=len(groups),replay_device=device,
        maximum_replay_vs_export_error=maximum,maximum_cpu_vs_gpu_return_error=cpu_max,
        common_validation_selection_verified=True,train_only_target_statistics_verified=True,
        purge_embargo_disjoint_test_verified=True,frozen_manifest_and_checkpoint_hashes_verified=True,
        baseline_predictions_reused_exactly=True,return_only_auxiliary_heads_not_evaluated=True,
        source_raw_validation='Prepared arrays reused from previously raw-rebuilt verified immutable cohort; no new vendor history',
        test_status=config['test_status'])
    write_json(root/'verification.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    args=parser.parse_args();verify(args.root,args.device)
