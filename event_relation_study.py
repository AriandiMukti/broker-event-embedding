"""End-to-end relation objective, matched peer controls, all selections frozen before test."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from event_relation_model import MODES
from event_relation_training import fit,infer,restore,configure_runtime as runtime
from event_market_data import sha,dump
from event_market_study import load_raw
from event_broker_geometry import dense_selected
from event_broker_probe import prepare_encoder_source
from broker_relation_test import baseline
from adro_token_embedding import scores
from relation_maps import simultaneous_intervals

CONTROLS=['own_only','frozen_peers','shifted_peers','frequency','markov','persistence','prior_own_linear','prior_frozen_probe']


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key] for key in z.files}


def prepare_fold(source,ticker,fold,root,encoder_source=None):
    base=load_arrays(source/ticker/f'fold-{fold}'/'base_features.npz')
    encoder_source=Path(encoder_source) if encoder_source is not None else source/'encoder-source'
    origin=encoder_source/ticker/f'fold-{fold}'/'local_only'
    checkpoint=next(origin.glob('none-seed*/best.pt'));saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    raw=load_raw(encoder_source,ticker);state=saved['preprocess'];codes=base['codes'];queries=base['queries']
    tokens,present=dense_selected(raw,codes,state);history=queries[:,None]+np.arange(-4,1)
    own=np.stack([base[f'own_{i}'][:,:70].reshape(len(queries),5,14)[:,::-1].copy() for i in range(len(codes))],axis=1)
    context=base['own_0'][:,70:]
    for i in range(len(codes)):np.testing.assert_array_equal(context,base[f'own_{i}'][:,70:])
    lookup={code:i+2 for i,code in enumerate(state['vocabulary'])}
    data=dict(own=own,context=context,tokens=tokens[history],mask=present[history],
        ids=np.asarray([lookup[code] for code in codes],dtype='int64'),y=base['y'],splits=base['splits'],
        queries=queries,codes=codes,vocabulary_size=np.asarray(len(state['vocabulary'])))
    folder=root/ticker/f'fold-{fold}';folder.mkdir(parents=True)
    np.savez_compressed(folder/'dataset.npz',**data)
    dump(folder/'lineage.json',dict(base_path=str(source/ticker/f'fold-{fold}'/'base_features.npz'),
        base_sha256=sha(source/ticker/f'fold-{fold}'/'base_features.npz'),preprocessor_checkpoint=str(checkpoint),
        preprocessor_checkpoint_sha256=sha(checkpoint),tokens='frozen train-only transform, no backbone weights transferred'))
    return data,folder


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False);source=Path(args.source).resolve()
    old=json.loads((source/'config.json').read_text());device=runtime(args.device)
    config=dict(source=str(source),out=str(root),tickers=args.tickers or old['tickers'],folds=list(range(1,args.folds+1)),
        seeds=args.seeds or old['seeds'],modes=MODES,device=device,width=24,batch_size=32,lr=.001,amp=not args.no_amp,
        epochs=args.epochs,patience=args.patience,history_sessions=5,bootstrap_repeats=args.bootstrap_repeats,
        objective='next-session5state crossentropy, equal day/target average',selection='validation NLL, min_delta1e-6',
        test_status='Exploratory historical reuse; no untouched confirmation',torch=torch.__version__)
    config['fp32_evaluation']='TF32 disabled; AMP training unchanged'
    config['planned_fits']=len(config['tickers'])*len(config['folds'])*len(config['seeds'])*len(MODES)
    dump(root/'config.json',config)
    # Freeze the minimal old provider snapshot; no new downloads or live data.
    prepare_encoder_source(source/'encoder-source',root/'encoder-source',config)
    (root/'source-base').mkdir()
    for ticker in config['tickers']:
        for fold in config['folds']:
            folder=root/'source-base'/ticker/f'fold-{fold}';folder.mkdir(parents=True)
            shutil.copy2(source/ticker/f'fold-{fold}'/'base_features.npz',folder/'base_features.npz')
    dump(root/'input_source_manifest.json',{str(p.relative_to(root)):sha(p) for top in ['encoder-source','source-base']
        for p in (root/top).rglob('*') if p.is_file()})
    # Full project Python snapshot includes all transitive local imports.
    (root/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/name for name in ['requirements.txt','EVENT-RELATION-E2E-PROTOCOL.md']]
    for path in paths:shutil.copy2(path,root/'sources'/path.name)
    dump(root/'code_manifest.json',{path.name:sha(root/'sources'/path.name) for path in paths})
    prior=pd.read_csv(source/'predictions.csv')
    prior=prior[prior.ticker.isin(config['tickers'])&prior.fold.isin(config['folds'])&prior.seed.isin(config['seeds'])
        &prior.method.isin(['own_context','learned_encoder'])]
    prior.to_csv(root/'prior_predictions.csv',index=False)
    fits=[];training=[];history=[];pairs=[]
    # prepare_fold reads a provider layout; use an isolated provider clone.
    provider=root/'source-base'
    # Original raw/prepared data remain in root/encoder-source. prepare function
    # receives these explicitly through an optional argument below.
    for ticker in config['tickers']:
        for fold in config['folds']:
            data,folder=prepare_fold(provider,ticker,fold,root,root/'encoder-source')
            for seed in config['seeds']:
                initials=[]
                for mode in MODES:
                    print(f'FIT {ticker} fold={fold} seed={seed} mode={mode}',flush=True)
                    destination=folder/f'{mode}-seed{seed}'
                    model,record,rows=fit(data,dict(config,ticker=ticker,fold=fold),mode,seed,destination)
                    context=dict(ticker=ticker,fold=fold,seed=seed,method=mode)
                    training.append(dict(context,**record));history.extend([{**context,**row} for row in rows])
                    initials.append(record['initial_sha256'])
                    fits.append(dict(context,path=str((destination/'best.pt').relative_to(root)),
                        sha256=sha(destination/'best.pt'),dataset=str((folder/'dataset.npz').relative_to(root)),
                        dataset_sha256=sha(folder/'dataset.npz')))
                    pd.DataFrame(training).to_csv(root/'training.csv',index=False)
                    pd.DataFrame(history).to_csv(root/'history.csv',index=False)
                    from dashboard import build_event_relation_dashboard
                    build_event_relation_dashboard(root)
                    del model
                if len(set(initials))!=1:raise ValueError('Matched initial weights failed')
                pairs.append(dict(ticker=ticker,fold=fold,seed=seed,initial_sha256=initials[0],matched_modes=len(MODES)))
    if len(fits)!=config['planned_fits']:raise ValueError('Incomplete fit grid')
    dump(root/'selected.json',fits);pd.DataFrame(pairs).to_csv(root/'initial_pairs.csv',index=False)
    dump(root/'phase1_frozen.json',{name:sha(root/name) for name in
        ['config.json','input_source_manifest.json','code_manifest.json','selected.json','initial_pairs.csv','prior_predictions.csv']})
    evaluate(root)
    dump(root/'progress.json',dict(status='complete',new_fits=len(fits),objective=config['objective']))
    print('COMPLETE',root,flush=True)


def evaluate(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());config['device']=runtime(config['device'])
    for name,digest in json.loads((root/'phase1_frozen.json').read_text()).items():assert sha(root/name)==digest
    fits=json.loads((root/'selected.json').read_text());metrics=[];daily=[];predictions=[];peer_rows=[];geometries=[]
    def add(context,queries,codes,y,p,dates):
        metrics.append(dict(context,**scores(y,p)))
        losses=-np.log(np.take_along_axis(p,y[...,None],axis=-1)[:,:,0].clip(1e-12))
        for k,q in enumerate(queries):
            date=dates[int(q)+1]
            daily.append(dict(context,date=date,mean_nll=float(losses[k].mean())))
            for j,broker in enumerate(codes):
                predictions.append(dict(context,date=date,query_index=int(q),broker=broker,actual=int(y[k,j]),
                    **{f'p{c}':float(p[k,j,c]) for c in range(5)}))
    for item in fits:
        assert sha(root/item['path'])==item['sha256'];assert sha(root/item['dataset'])==item['dataset_sha256']
        data=load_arrays(root/item['dataset']);model,saved=restore(root/item['path'],config['device'])
        test=np.flatnonzero(data['splits']=='test');val=np.flatnonzero(data['splits']=='val')
        context={key:item[key] for key in ['ticker','fold','seed','method']}
        dates=json.loads((root/'encoder-source'/f'{item["ticker"]}-data-audit.json').read_text())['dates']
        p=infer(model,data,test,config,item['method'],item['seed'])
        add(context,data['queries'][test],data['codes'],data['y'][test],p,dates)
        pv=infer(model,data,val,config,item['method'],item['seed'])
        nll=float(-np.log(np.take_along_axis(pv,data['y'][val,...,None],axis=-1)).mean())
        np.testing.assert_allclose(nll,saved['record']['validation_nll'],atol=2e-6,rtol=2e-5)
        if item['method']=='relational':
            reference=-np.log(np.take_along_axis(pv,data['y'][val,...,None],axis=-1)[:,:,0])
            for peer,code in enumerate(data['codes']):
                removed=infer(model,data,val,config,item['method'],item['seed'],remove_peer=peer)
                gain=-np.log(np.take_along_axis(removed,data['y'][val,...,None],axis=-1)[:,:,0])-reference
                for target,broker in enumerate(data['codes']):
                    if target!=peer:peer_rows.append(dict(context,peer=code,broker=broker,mean_delta_nll=float(gain[:,target].mean())))
        geometries.append(dict(context,**saved['record']))
        del model
    # Identical simple baselines, once per stock/fold.
    for ticker in config['tickers']:
        dates=json.loads((root/'encoder-source'/f'{ticker}-data-audit.json').read_text())['dates']
        for fold in config['folds']:
            data=load_arrays(root/ticker/f'fold-{fold}'/'dataset.npz')
            base=load_arrays(root/'source-base'/ticker/f'fold-{fold}'/'base_features.npz')
            constant,markov=baseline(data['y'],base['previous'],data['splits'])
            persistence=np.full_like(constant,.025);np.put_along_axis(persistence,base['previous'][...,None],.9,axis=2)
            test=data['splits']=='test'
            for method,p in [('frequency',constant),('markov',markov),('persistence',persistence)]:
                add(dict(ticker=ticker,fold=fold,seed=-1,method=method),data['queries'][test],data['codes'],data['y'][test],p[test],dates)
    prior=pd.read_csv(root/'prior_predictions.csv')
    for key,part in prior.groupby(['ticker','fold','seed','method']):
        ticker,fold,seed,name=key;data=load_arrays(root/ticker/f'fold-{fold}'/'dataset.npz');test=data['splits']=='test'
        arrays=[]
        for broker in data['codes']:
            sub=part[part.broker==broker].sort_values('query_index')
            np.testing.assert_array_equal(sub.query_index.to_numpy(),data['queries'][test])
            arrays.append(sub[[f'p{k}' for k in range(5)]].to_numpy())
        p=np.stack(arrays,axis=1);method='prior_own_linear' if name=='own_context' else 'prior_frozen_probe'
        dates=json.loads((root/'encoder-source'/f'{ticker}-data-audit.json').read_text())['dates']
        add(dict(ticker=ticker,fold=fold,seed=seed,method=method),data['queries'][test],data['codes'],data['y'][test],p,dates)
    for name,rows in [('metrics',metrics),('daily_losses',daily),('predictions',predictions),('peer_removal_validation',peer_rows),('encoder_updates',geometries)]:
        pd.DataFrame(rows).to_csv(root/f'{name}.csv',index=False)
    fields=['nll','state_accuracy','direction_accuracy','macro_state_recall','brier'];table=pd.DataFrame(metrics)
    table.groupby(['ticker','method'])[fields].mean().reset_index().to_csv(root/'comparison.csv',index=False)
    table.groupby('method')[fields].mean().reset_index().to_csv(root/'macro_summary.csv',index=False)
    loss=pd.DataFrame(daily).groupby(['date','fold','ticker','method']).mean_nll.mean().unstack('method')
    matrices=[];keys=[];consistency=[]
    for control in CONTROLS:
        wide=(loss[control]-loss.relational).unstack('ticker').sort_index()
        matrices.append(wide);keys.extend([(ticker,control) for ticker in wide.columns])
        for ticker in config['tickers']:
            local=table[table.ticker==ticker];trained=local[local.method=='relational'].set_index(['fold','seed']).nll
            reference=local[local.method==control]
            if control in ['frequency','markov','persistence']:
                ref=reference.set_index('fold').nll;gain=pd.Series([ref.loc[f]-v for (f,s),v in trained.items()],index=trained.index)
            else:gain=reference.set_index(['fold','seed']).nll-trained
            consistency.append(dict(ticker=ticker,comparator=control,gain_nll=float(gain.mean()),
                folds_won=int((gain.groupby(level='fold').mean()>0).sum()),seeds_won=int((gain.groupby(level='seed').mean()>0).sum())))
    matrix=pd.concat(matrices,axis=1).to_numpy();folds=matrices[0].index.get_level_values('fold').to_numpy();bounds=[]
    for block in [3,5,10]:
        mean,low,high,sd,critical=simultaneous_intervals(matrix,folds,repeats=config['bootstrap_repeats'],block=block)
        for i,(ticker,control) in enumerate(keys):
            bounds.append(dict(ticker=ticker,comparator=control,block_sessions=block,gain_nll=mean[i],simultaneous_low=low[i],
                simultaneous_high=high[i],family_columns=matrix.shape[1],dates=int(np.isfinite(matrix[:,i]).sum())))
    pd.DataFrame(bounds).to_csv(root/'paired_intervals.csv',index=False)
    pd.DataFrame(consistency).to_csv(root/'fold_seed_stability.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--source',default='runs/event-broker-probe-20261117');p.add_argument('--tickers',nargs='+')
    p.add_argument('--folds',type=int,default=3);p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--patience',type=int,default=6)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--no-amp',action='store_true')
    p.add_argument('--bootstrap-repeats',type=int,default=2000);run(p.parse_args())
