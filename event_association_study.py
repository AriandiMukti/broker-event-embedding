"""Same-session masked association; reuse the next-session backbone and controls."""
import argparse
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from event_relation_model import RelationModel
from event_relation_training import fit,infer,restore,configure_runtime
from event_relation_study import load_arrays
from event_broker_probe import prepare_encoder_source,fit_head,probabilities
from event_broker_geometry import dense_selected
from event_market_study import load_raw
from event_market_data import sha,dump
from event_training import seed_all,weights_hash
from adro_token_embedding import scores
from relation_maps import simultaneous_intervals

NEW_MODES=['relational','frozen_peers','shifted_peers']
LINEAR=['raw_linear','aggregate_linear']
CONTROLS=['own_only','frozen_peers','shifted_peers','raw_linear','aggregate_linear',
          'prior_own_linear','frequency','markov','persistence']
TOKEN_NAMES=['signed_log_net_value','signed_log_net_lot','lot_missing','consistency_5',
             'delta_signed_log_net','acceleration_signed_log_net','net_positive','net_negative']


def local_peer_tokens(tokens,present):
    """No global shares/denominators, target activity cannot affect another peer."""
    value=tokens[...,0];past=np.vstack([np.zeros_like(value[:1]),value[:-1]])
    past2=np.vstack([np.zeros_like(value[:2]),value[:-2]])
    result=np.stack([value,tokens[...,2],tokens[...,3],tokens[...,4],value-past,
        value-2*past+past2,(value>0).astype(float),(value<0).astype(float)],axis=-1)
    return np.where(present[...,None],result,0).astype('float32')


def transformed_peer(raw,codes,train):
    raw_tokens,present=dense_selected(raw,codes)
    local=local_peer_tokens(raw_tokens,present)
    observed=local[train][present[train]]
    mean=observed.mean(0);std=observed.std(0).clip(.01)
    normalized=np.where(present[...,None],np.clip((local-mean)/std,-10,10),0).astype('float32')
    return normalized,present,dict(mean=mean.tolist(),std=std.tolist(),features=TOKEN_NAMES),raw_tokens


def prepare_data(old,raw,train):
    q=old['queries']+1
    tokens,present,scaler,raw_tokens=transformed_peer(raw,old['codes'],train)
    history=q[:,None]+np.arange(-4,1)
    data={k:v.copy() for k,v in old.items()}
    data.update(tokens=tokens[history],mask=present[history],queries=q,
                source_queries=old['queries'].copy(),peer_net_log=raw_tokens[history,:,0])
    # Reuse old own history and lagged OHLCV exactly: own is t-5..t-1.
    return data,scaler


def linear_features(data,target,method):
    own=np.column_stack([data['own'][:,target,::-1].reshape(len(data['y']),70),data['context']])
    tokens=data['tokens'].copy();mask=data['mask'].copy();mask[:,:,target]=False
    tokens=np.where(mask[...,None],tokens,0)
    if method=='raw_linear':
        extra=np.concatenate([tokens,mask[...,None]],axis=-1).reshape(len(own),-1)
    elif method=='aggregate_linear':
        count=mask.sum(-1,keepdims=True).clip(1)
        mean=tokens.sum(2)/count
        variance=((tokens-mean[:,:,None,:])**2*mask[...,None]).sum(2)/count
        logs=np.where(mask,data['peer_net_log'],0).astype('float64')
        net=np.sign(logs)*np.expm1(abs(logs))
        def signed_log(x):return np.sign(x)*np.log1p(abs(x))
        extra=np.concatenate([mean,np.sqrt(variance),mask.mean(2)[...,None],
            signed_log(net.sum(2))[...,None],np.log1p(abs(net).sum(2))[...,None]],axis=-1).reshape(len(own),-1)
    else:raise ValueError(method)
    return np.column_stack([own,extra]).astype('float32')


def assert_reuse(old,data):
    for key in ['own','context','y','ids','codes','splits','vocabulary_size']:
        np.testing.assert_array_equal(old[key],data[key])
    np.testing.assert_array_equal(old['queries']+1,data['queries'])


def build_dashboard(root):
    from dashboard import build_event_relation_dashboard
    build_event_relation_dashboard(root)
    path=Path(root)/'index.html';page=path.read_text(encoding='utf-8')
    page=page.replace('Relasi Broker — Training Objective Langsung','Asosiasi Broker — Masked Reconstruction')
    page=page.replace('Relasi Broker — Objective Langsung','Asosiasi Broker — Masked Reconstruction')
    page=page.replace('Target: net-state broker tepat sesi berikutnya. Model terpisah per saham; Set Transformer dan TCN dilatih untuk target broker.',
        'Target: net-state broker sesi t yang disembunyikan. Own history dan OHLCV sampai t-1; peer sampai t. Ini asosiasi EOD, bukan forecast sebelum penutupan.')
    page=page.replace('Test next-session net-state NLL','Test same-session masked net-state NLL')
    path.write_text(page,encoding='utf-8')


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False);source=Path(args.source).resolve()
    old_config=json.loads((source/'config.json').read_text())
    config=dict(source=str(source),out=str(root),tickers=args.tickers or old_config['tickers'],
        folds=list(range(1,args.folds+1)),seeds=args.seeds or old_config['seeds'],device=configure_runtime(args.device),
        width=24,batch_size=32,lr=.001,amp=not args.no_amp,epochs=args.epochs,patience=args.patience,
        bootstrap_repeats=args.bootstrap_repeats,objective='same-session masked 5-state crossentropy; peer through t, own/price through t-1',
        test_status='Exploratory historical reuse; no untouched confirmation',torch=torch.__version__,
        fp32_evaluation='TF32 disabled; AMP training',new_modes=NEW_MODES)
    config['token_features'] = TOKEN_NAMES
    config['planned_new_fits']=len(config['tickers'])*len(config['folds'])*len(config['seeds'])*3
    dump(root/'config.json',config)
    prepare_encoder_source(source/'encoder-source',root/'encoder-source',config)
    selected_old=json.loads((source/'selected.json').read_text())
    for ticker in config['tickers']:
        for fold in config['folds']:
            folder=root/'source-control'/ticker/f'fold-{fold}';folder.mkdir(parents=True)
            shutil.copy2(source/ticker/f'fold-{fold}'/'dataset.npz',folder/'dataset.npz')
            for seed in config['seeds']:
                item=next(x for x in selected_old if (x['ticker'],x['fold'],x['seed'],x['method'])==(ticker,fold,seed,'own_only'))
                assert sha(source/item['path'])==item['sha256']
                dest=folder/f'own_only-seed{seed}';dest.mkdir()
                shutil.copy2(source/item['path'],dest/'best.pt')
    prior=pd.read_csv(source/'predictions.csv')
    prior=prior[prior.ticker.isin(config['tickers'])&prior.fold.isin(config['folds'])&
        ((prior.seed.isin(config['seeds'])&prior.method.isin(['own_only','prior_own_linear']))|
         prior.method.isin(['frequency','markov','persistence']))]
    prior.to_csv(root/'prior_predictions.csv',index=False)
    dump(root/'input_source_manifest.json',{str(p.relative_to(root)):sha(p)
        for top in ['encoder-source','source-control'] for p in (root/top).rglob('*') if p.is_file()})
    (root/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/n for n in
        ['requirements.txt','EVENT-ASSOCIATION-AUDIT.md','EVENT-ASSOCIATION-PROTOCOL.md']]
    for path in paths:shutil.copy2(path,root/'sources'/path.name)
    dump(root/'code_manifest.json',{p.name:sha(root/'sources'/p.name) for p in paths})
    selected=[];heads=[];trials=[];training=[];history=[];pairs=[]
    with threadpool_limits(limits=2):
        for ticker in config['tickers']:
            raw=load_raw(root/'encoder-source',ticker)
            for fold in config['folds']:
                folder=root/ticker/f'fold-{fold}';folder.mkdir(parents=True)
                origin=root/'encoder-source'/ticker/f'fold-{fold}'/'local_only'
                with np.load(origin/'prepared.npz',allow_pickle=False) as z:train=z['train_indices'].copy()
                old=load_arrays(root/'source-control'/ticker/f'fold-{fold}'/'dataset.npz')
                data,scaler=prepare_data(old,raw,train);assert_reuse(old,data)
                np.savez_compressed(folder/'dataset.npz',**data);dump(folder/'peer_scaler.json',scaler)
                for seed in config['seeds']:
                    original=root/'source-control'/ticker/f'fold-{fold}'/f'own_only-seed{seed}'/'best.pt'
                    saved=torch.load(original,map_location='cpu',weights_only=False);record=saved['record']
                    seed_all(seed);initial=RelationModel(saved['vocabulary'],saved['context_features'],saved['width'],'own_only')
                    assert weights_hash(initial.state_dict())==record['initial_sha256']
                    context=dict(ticker=ticker,fold=fold,seed=seed,method='own_only')
                    selected.append(dict(context,path=str(original.relative_to(root)),sha256=sha(original),
                        dataset=str((folder/'dataset.npz').relative_to(root)),dataset_sha256=sha(folder/'dataset.npz'),reused=True))
                    training.append(dict(context,**record,reused=True))
                    initials=[record['initial_sha256']]
                    for mode in NEW_MODES:
                        print(f'FIT {ticker} fold={fold} seed={seed} mode={mode}',flush=True)
                        dest=folder/f'{mode}-seed{seed}'
                        model,record,rows=fit(data,dict(config,ticker=ticker,fold=fold),mode,seed,dest)
                        context=dict(ticker=ticker,fold=fold,seed=seed,method=mode)
                        training.append(dict(context,**record,reused=False));history.extend([{**context,**r} for r in rows])
                        initials.append(record['initial_sha256'])
                        selected.append(dict(context,path=str((dest/'best.pt').relative_to(root)),sha256=sha(dest/'best.pt'),
                            dataset=str((folder/'dataset.npz').relative_to(root)),dataset_sha256=sha(folder/'dataset.npz'),reused=False))
                        pd.DataFrame(training).to_csv(root/'training.csv',index=False)
                        pd.DataFrame(history).to_csv(root/'history.csv',index=False)
                        build_dashboard(root);del model
                    assert len(set(initials))==1
                    pairs.append(dict(ticker=ticker,fold=fold,seed=seed,matched_modes=4,initial_sha256=initials[0]))
                for target,broker in enumerate(data['codes']):
                    for method in LINEAR:
                        x=linear_features(data,target,method);head,attempts=fit_head(x,data['y'][:,target],data['splits'])
                        path=folder/f'{method}-{broker}.pkl'
                        with path.open('wb') as f:pickle.dump(head,f)
                        context=dict(ticker=ticker,fold=fold,target=target,broker=broker,method=method)
                        heads.append(dict(context,path=str(path.relative_to(root)),sha256=sha(path),
                            dataset=str((folder/'dataset.npz').relative_to(root)),C=head['C'],
                            validation_nll=float(-np.log(probabilities(head,x[data['splits']=='val'])[
                                np.arange((data['splits']=='val').sum()),data['y'][data['splits']=='val',target]]).mean())))
                        trials.extend([{**context,**r} for r in attempts])
                pd.DataFrame(trials).to_csv(root/'head_trials.csv',index=False)
    assert sum(not x['reused'] for x in selected)==config['planned_new_fits']
    dump(root/'selected.json',selected);dump(root/'selected_heads.json',heads)
    dump(root/'data_manifest.json',{str(p.relative_to(root)):sha(p) for ticker in config['tickers']
        for fold in config['folds'] for p in [root/ticker/f'fold-{fold}'/'dataset.npz',
            root/ticker/f'fold-{fold}'/'peer_scaler.json']})
    pd.DataFrame(pairs).to_csv(root/'initial_pairs.csv',index=False)
    dump(root/'phase1_frozen.json',{n:sha(root/n) for n in ['config.json','input_source_manifest.json',
        'code_manifest.json','data_manifest.json','selected.json','selected_heads.json','initial_pairs.csv','prior_predictions.csv']})
    evaluate(root)
    dump(root/'progress.json',dict(status='complete',new_fits=config['planned_new_fits'],
        reused_neural=len(selected)//4,new_linear_heads=len(heads),objective=config['objective']))
    build_dashboard(root);print('COMPLETE',root,flush=True)


def evaluate(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());config['device']=configure_runtime(config['device'])
    for name,digest in json.loads((root/'phase1_frozen.json').read_text()).items():assert sha(root/name)==digest
    metrics=[];daily=[];predictions=[];peer_rows=[]
    def add(context,data,rows,p):
        y=data['y'][rows];queries=data['queries'][rows]
        dates=json.loads((root/'encoder-source'/f'{context["ticker"]}-data-audit.json').read_text())['dates']
        metrics.append(dict(context,**scores(y,p)))
        loss=-np.log(np.take_along_axis(p,y[...,None],axis=-1)[:,:,0])
        for k,q in enumerate(queries):
            daily.append(dict(context,date=dates[int(q)],mean_nll=float(loss[k].mean())))
            for j,broker in enumerate(data['codes']):
                predictions.append(dict(context,date=dates[int(q)],query_index=int(q),broker=broker,actual=int(y[k,j]),
                    **{f'p{c}':float(p[k,j,c]) for c in range(5)}))
    for item in json.loads((root/'selected.json').read_text()):
        assert sha(root/item['path'])==item['sha256'];assert sha(root/item['dataset'])==item['dataset_sha256']
        data=load_arrays(root/item['dataset']);model,saved=restore(root/item['path'],config['device'])
        test=np.flatnonzero(data['splits']=='test');val=np.flatnonzero(data['splits']=='val')
        context={k:item[k] for k in ['ticker','fold','seed','method']}
        add(context,data,test,infer(model,data,test,config,item['method'],item['seed']))
        pv=infer(model,data,val,config,item['method'],item['seed'])
        nll=float(-np.log(np.take_along_axis(pv,data['y'][val,...,None],axis=-1)).mean())
        np.testing.assert_allclose(nll,saved['record']['validation_nll'],atol=2e-6,rtol=2e-5)
        if item['method']=='relational':
            ref=-np.log(np.take_along_axis(pv,data['y'][val,...,None],axis=-1)[:,:,0])
            for peer,code in enumerate(data['codes']):
                pp=infer(model,data,val,config,item['method'],item['seed'],remove_peer=peer)
                gain=-np.log(np.take_along_axis(pp,data['y'][val,...,None],axis=-1)[:,:,0])-ref
                for target,broker in enumerate(data['codes']):
                    if target!=peer:peer_rows.append(dict(context,peer=code,broker=broker,mean_delta_nll=float(gain[:,target].mean())))
        del model
    for ticker in config['tickers']:
        for fold in config['folds']:
            data=load_arrays(root/ticker/f'fold-{fold}'/'dataset.npz');rows=np.flatnonzero(data['splits']=='test')
            for method in LINEAR:
                pp=[]
                for item in json.loads((root/'selected_heads.json').read_text()):
                    if (item['ticker'],item['fold'],item['method'])!=(ticker,fold,method):continue
                    assert sha(root/item['path'])==item['sha256']
                    with (root/item['path']).open('rb') as f:head=pickle.load(f)
                    pp.append((item['target'],probabilities(head,linear_features(data,item['target'],method)[rows])))
                p=np.stack([v for k,v in sorted(pp)],axis=1)
                add(dict(ticker=ticker,fold=fold,seed=-1,method=method),data,rows,p)
    prior=pd.read_csv(root/'prior_predictions.csv')
    for (ticker,fold,seed,method),part in prior.groupby(['ticker','fold','seed','method']):
        if method=='own_only':continue # Independently replayed above; exact prior comparison in verifier.
        data=load_arrays(root/ticker/f'fold-{fold}'/'dataset.npz');rows=np.flatnonzero(data['splits']=='test');parts=[]
        for broker in data['codes']:
            sub=part[part.broker==broker].sort_values('query_index')
            np.testing.assert_array_equal(sub.query_index.to_numpy()+1,data['queries'][rows])
            np.testing.assert_array_equal(sub.actual.to_numpy(),data['y'][rows,list(data['codes']).index(broker)])
            parts.append(sub[[f'p{k}' for k in range(5)]].to_numpy())
        add(dict(ticker=ticker,fold=fold,seed=seed,method=method),data,rows,np.stack(parts,axis=1))
    for name,records in [('metrics',metrics),('daily_losses',daily),('predictions',predictions),('peer_removal_validation',peer_rows)]:
        pd.DataFrame(records).to_csv(root/f'{name}.csv',index=False)
    table=pd.DataFrame(metrics);fields=['nll','state_accuracy','direction_accuracy','macro_state_recall','brier']
    table.groupby(['ticker','method'])[fields].mean().reset_index().to_csv(root/'comparison.csv',index=False)
    table.groupby('method')[fields].mean().reset_index().to_csv(root/'macro_summary.csv',index=False)
    loss=pd.DataFrame(daily).groupby(['date','fold','ticker','method']).mean_nll.mean().unstack('method')
    matrices=[];keys=[];consistency=[]
    for control in CONTROLS:
        wide=(loss[control]-loss.relational).unstack('ticker').sort_index();matrices.append(wide)
        keys.extend([(ticker,control) for ticker in wide.columns])
        for ticker in config['tickers']:
            local=table[table.ticker==ticker];trained=local[local.method=='relational'].set_index(['fold','seed']).nll
            reference=local[local.method==control]
            if (reference.seed==-1).all():
                ref=reference.set_index('fold').nll;gain=pd.Series([ref.loc[f]-v for (f,s),v in trained.items()],index=trained.index)
            else:gain=reference.set_index(['fold','seed']).nll-trained
            consistency.append(dict(ticker=ticker,comparator=control,gain_nll=float(gain.mean()),
                folds_won=int((gain.groupby(level='fold').mean()>0).sum()),seeds_won=int((gain.groupby(level='seed').mean()>0).sum())))
    matrix=pd.concat(matrices,axis=1).to_numpy();folds=matrices[0].index.get_level_values('fold').to_numpy();bounds=[]
    for block in [3,5,10]:
        mean,low,high,_,_=simultaneous_intervals(matrix,folds,repeats=config['bootstrap_repeats'],block=block)
        for i,(ticker,control) in enumerate(keys):
            bounds.append(dict(ticker=ticker,comparator=control,block_sessions=block,gain_nll=mean[i],
                simultaneous_low=low[i],simultaneous_high=high[i],family_columns=matrix.shape[1],
                dates=int(np.isfinite(matrix[:,i]).sum())))
    pd.DataFrame(bounds).to_csv(root/'paired_intervals.csv',index=False)
    pd.DataFrame(consistency).to_csv(root/'fold_seed_stability.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--source',default='runs/event-relation-e2e-20261117');p.add_argument('--tickers',nargs='+')
    p.add_argument('--folds',type=int,default=3);p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--patience',type=int,default=6)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--no-amp',action='store_true')
    p.add_argument('--bootstrap-repeats',type=int,default=2000);run(p.parse_args())
