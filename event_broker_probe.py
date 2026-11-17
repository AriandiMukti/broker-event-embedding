"""Lightweight next-session net-state probes; every backbone remains frozen."""
import argparse
import json
import pickle
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from event_broker_geometry import select_brokers,state_labels,dense_selected,initial_model,subset_pool
from event_market_study import load_raw
from event_market_data import sha,dump
from event_objective_study import runtime
from event_predict import restore
from event_training import weights_hash
from broker_relation_test import baseline as legacy_baseline
from adro_token_embedding import scores
from relation_maps import simultaneous_intervals

METHODS=['own_context','raw_peers','random_encoder','learned_encoder','shifted_encoder']
BENCHMARKS=['frequency','markov','persistence']
CS=[.01,.1,1.]


def query_splits(prepared,raw):
    """Calendar t+1 must remain in the same original split, never next observed."""
    result={}
    observed=raw['mask'].any(1)
    for key in ['train','val','test']:
        rows=prepared[key+'_indices'];eligible=set(map(int,rows))
        result[key]=np.asarray([int(i) for i in rows if i>=4 and int(i)+1 in eligible
            and observed[i] and observed[i+1]],dtype='int64')
        if len(result[key])<10:raise ValueError(f'Insufficient valid next-day {key} queries')
    return result


def own_features(data,states,tokens,present,observed,queries,target,ohlcv):
    pieces=[]
    for lag in range(5):
        rows=queries-lag
        onehot=np.eye(5)[states[rows,target]]*observed[rows,None]
        pieces.extend([onehot,tokens[rows,target],present[rows,target,None].astype(float)])
    return np.column_stack(pieces+[data['x'][queries][:,ohlcv]]).astype('float32')


def raw_peer_features(tokens,present,queries,target):
    token=tokens[queries].copy();mask=present[queries].copy()
    token[:,target]=0;mask[:,target]=False
    return np.concatenate([token,mask[...,None]],axis=2).reshape(len(queries),-1).astype('float32')


def shift_mapping(splits,seed,target):
    rng=np.random.default_rng(seed+90117+target*997);mapping=np.arange(len(splits));offsets={}
    for key in ['train','val','test']:
        rows=np.flatnonzero(splits==key)
        offset=int(rng.integers(5,len(rows)-4))
        mapping[rows]=np.roll(rows,offset);offsets[key]=offset
    return mapping,offsets


def probabilities(head,x):
    if head['constant'] is not None:return np.tile(head['constant'],(len(x),1))
    predicted=head['classifier'].predict_proba(head['scaler'].transform(x))
    p=np.full((len(x),5),1e-8)
    p[:,head['classifier'].classes_.astype(int)]=predicted
    return p/p.sum(1,keepdims=True)


def fit_head(x,y,splits):
    train=splits=='train';val=splits=='val';ys=y[train]
    if len(np.unique(ys))<2:
        p=(np.bincount(ys,minlength=5)+1)/(len(ys)+5)
        return dict(constant=p,scaler=None,classifier=None,C=None),[dict(C=None,nll=float(-np.log(p[y[val]]).mean()),warnings=0)]
    scaler=StandardScaler().fit(x[train]);xt=scaler.transform(x[train]);best=None;best_loss=float('inf');trials=[]
    for c in CS:
        model=LogisticRegression(C=c,solver='lbfgs',max_iter=300,random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(xt,ys)
        head=dict(constant=None,scaler=scaler,classifier=model,C=c)
        loss=float(-np.log(probabilities(head,x[val])[np.arange(val.sum()),y[val]]).mean())
        count=sum(issubclass(row.category,ConvergenceWarning) for row in caught)
        trials.append(dict(C=c,nll=loss,warnings=count,n_iter=int(model.n_iter_.max())))
        if loss<best_loss:best_loss=loss;best=head
    return best,trials


def all_features(cache,queries,splits,target,method):
    own=cache[f'own_{target}']
    if method=='own_context':return own
    if method=='raw_peers':extra=cache[f'raw_{target}']
    elif method=='random_encoder':extra=cache['random'][:,target]
    elif method=='learned_encoder':extra=cache['learned'][:,target]
    elif method=='shifted_encoder':extra=cache['learned'][cache['shift_indices'][:,target],target]
    else:raise ValueError(method)
    return np.column_stack([own,extra]).astype('float32')


def build_cache(model,random,data,raw,codes,state,queries,splits,seed,device):
    states,thresholds,_,present=state_labels(raw,codes,data['train_indices'])
    tokens,_=dense_selected(raw,codes,state);learned=[];initial=[];cache={}
    observed=raw['mask'].any(1);shift=[]
    for target in range(len(codes)):
        cache[f'own_{target}']=own_features(data,states,tokens,present,observed,queries,target,model.groups['ohlcv'])
        cache[f'raw_{target}']=raw_peer_features(tokens,present,queries,target)
        learned.append(subset_pool(model.set_encoder,tokens,present,codes,state,queries,target,device))
        initial.append(subset_pool(random.set_encoder,tokens,present,codes,state,queries,target,device))
        mapping,_=shift_mapping(splits,seed,target);shift.append(mapping)
    cache.update(learned=np.stack(learned,axis=1),random=np.stack(initial,axis=1),
        shift_indices=np.stack(shift,axis=1),queries=queries,splits=splits,
        y=states[queries+1],previous=states[queries],codes=np.asarray(codes),thresholds=thresholds)
    return cache


def read_cache(destination):
    destination=Path(destination)
    with np.load(destination.parent/'base_features.npz',allow_pickle=False) as z:base={k:z[k] for k in z.files}
    with np.load(destination/'features.npz',allow_pickle=False) as z:base.update({k:z[k] for k in z.files})
    return base


def prepare_encoder_source(source,destination,config):
    destination.mkdir()
    shutil.copy2(source/'config.json',destination/'config.json')
    for ticker in config['tickers']:
        for name in [f'{ticker}-daily.npz',f'{ticker}-data-audit.json',f'{ticker}-ohlcv-snapshot.csv']:
            shutil.copy2(source/name,destination/name)
        for fold in config['folds']:
            origin=source/ticker/f'fold-{fold}'/'local_only';folder=destination/ticker/f'fold-{fold}'/'local_only'
            folder.mkdir(parents=True);shutil.copy2(origin/'prepared.npz',folder/'prepared.npz')
            for seed in config['seeds']:
                target=folder/f'none-seed{seed}';target.mkdir();shutil.copy2(origin/f'none-seed{seed}/best.pt',target/'best.pt')


def run(args):
    device=runtime(args.device);root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    source=Path(args.source).resolve();old=json.loads((source/'config.json').read_text())
    config=dict(source=str(source),tickers=args.tickers or old['tickers'],folds=list(range(1,args.folds+1)),
        seeds=args.seeds or old['seeds'],device=device,methods=METHODS,baselines=BENCHMARKS,C=CS,
        max_iter=300,head='train-standardized multinomial linear logistic',own_history_sessions=5,
        top_brokers=12,min_active_train_days=50,bootstrap_repeats=args.bootstrap_repeats,
        test_status='Exploratory historical reuse; no untouched confirmation set',backbone_updated=False,
        component='SetEncoder daily pooled24; TCN and original9outputs unused',
        control_information='12 train-selected brokers, target masked; identical peer information for raw/random/learned',
        snapshot_semantics='NET_ZERO means netzero/unreported on a valid snapshot, not gross inactivity')
    config.update(torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__)
    dump(root/'config.json',config);prepare_encoder_source(source,root/'encoder-source',config)
    dump(root/'encoder_source_manifest.json',{str(p.relative_to(root/'encoder-source')):sha(p)
        for p in (root/'encoder-source').rglob('*') if p.is_file()})
    (root/'sources').mkdir()
    code=['event_broker_probe.py','event_broker_geometry.py','event_broker_probe_verify.py','event_broker_probe_results.py',
          'event_models.py','event_training.py','event_data.py','event_predict.py','event_market_study.py','event_market_data.py',
          'event_objective_study.py','event_uncertainty.py','broker_relation_test.py','adro_token_embedding.py','broker_tokens.py',
          'relation_maps.py','relation_io.py','research.py','main.py','features.py','compact_features.py','event_sources.py',
          'dashboard.py','EVENT-RELATION-AUDIT.md','EVENT-RELATION-PROTOCOL.md','test_event_broker_probe.py','requirements.txt']
    for name in code:shutil.copy2(Path(__file__).parent/name,root/'sources'/name)
    dump(root/'code_manifest.json',{name:sha(root/'sources'/name) for name in code})
    selected=[];encoder_manifest=[];trials=[];samples=[];peer_rows=[];started=time.perf_counter()
    with threadpool_limits(limits=2):
        for ticker in config['tickers']:
            raw=load_raw(root/'encoder-source',ticker)
            for fold in config['folds']:
                origin=root/'encoder-source'/ticker/f'fold-{fold}'/'local_only'
                arrays=np.load(origin/'prepared.npz',allow_pickle=False);data={k:arrays[k] for k in arrays.files}
                split=query_splits(data,raw);queries=np.concatenate(list(split.values()))
                splits=np.concatenate([np.repeat(k,len(v)) for k,v in split.items()])
                codes,selection=select_brokers(raw,data['train_indices'])
                folder=root/ticker/f'fold-{fold}';folder.mkdir(parents=True)
                dump(folder/'labels.json',dict(codes=codes,selection=selection,split_counts={k:len(v) for k,v in split.items()}))
                for key,rows in split.items():
                    samples.extend([dict(ticker=ticker,fold=fold,split=key,query_index=int(i),target_index=int(i+1),
                        query_date=str(raw['dates'][i].date()),target_date=str(raw['dates'][i+1].date())) for i in rows])
                for seed in config['seeds']:
                    print(f'PROBE {ticker} fold={fold} seed={seed}',flush=True)
                    checkpoint=origin/f'none-seed{seed}/best.pt'
                    model,saved=restore(checkpoint,device);model.requires_grad_(False)
                    random=initial_model(saved).to(device).eval().requires_grad_(False)
                    before=weights_hash(model.state_dict());initial_before=weights_hash(random.state_dict())
                    destination=folder/f'seed-{seed}';destination.mkdir()
                    cache=build_cache(model,random,data,raw,codes,saved['preprocess'],queries,splits,seed,device)
                    shared={k:v for k,v in cache.items() if k not in ['learned','random','shift_indices']}
                    base_path=folder/'base_features.npz'
                    if not base_path.exists():np.savez_compressed(base_path,**shared)
                    else:
                        with np.load(base_path,allow_pickle=False) as old_base:
                            for key,value in shared.items():np.testing.assert_array_equal(value,old_base[key])
                    np.savez_compressed(destination/'features.npz',**{k:cache[k] for k in ['learned','random','shift_indices']})
                    y=cache['y'];val=splits=='val'
                    for target,code_name in enumerate(codes):
                        for method in METHODS:
                            x=all_features(cache,queries,splits,target,method)
                            head,attempts=fit_head(x,y[:,target],splits)
                            path=destination/f'{method}-{code_name}.pkl'
                            with path.open('wb') as f:pickle.dump(head,f)
                            context=dict(ticker=ticker,fold=fold,seed=seed,broker=code_name,target=target,method=method)
                            selected.append(dict(context,path=str(path.relative_to(root)),sha256=sha(path),
                                features_sha256=sha(destination/'features.npz'),base_sha256=sha(base_path),C=head['C'],
                                validation_nll=float(-np.log(probabilities(head,x[val])[np.arange(val.sum()),y[val,target]]).mean())))
                            trials.extend([{**context,**row} for row in attempts])
                            if method=='learned_encoder':
                                # Validation perturbation only. All directed pairs retained.
                                baseline=-np.log(probabilities(head,x[val])[np.arange(val.sum()),y[val,target]])
                                tokens,present=dense_selected(raw,codes,saved['preprocess'])
                                for peer,peer_code in enumerate(codes):
                                    if peer==target:continue
                                    masked=present.copy();masked[:,peer]=False
                                    extra=subset_pool(model.set_encoder,tokens,masked,codes,saved['preprocess'],queries[val],target,device)
                                    perturbed=np.column_stack([cache[f'own_{target}'][val],extra]).astype('float32')
                                    gain=-np.log(probabilities(head,perturbed)[np.arange(val.sum()),y[val,target]])-baseline
                                    peer_rows.append(dict(context,peer=peer_code,mean_delta_nll=float(gain.mean()),validation_dates=len(gain)))
                    after=weights_hash(model.state_dict());assert before==after
                    assert initial_before==weights_hash(random.state_dict())
                    encoder_manifest.append(dict(ticker=ticker,fold=fold,seed=seed,path=str(checkpoint.relative_to(root)),
                        sha256=sha(checkpoint),state_sha256=before,initial_state_sha256=initial_before,
                        features=str((destination/'features.npz').relative_to(root)),features_sha256=sha(destination/'features.npz'),
                        base_sha256=sha(base_path),
                        backbone_unchanged=True))
                    pd.DataFrame(trials).to_csv(root/'head_trials.csv',index=False)
                    pd.DataFrame(peer_rows).to_csv(root/'peer_removal_validation.csv',index=False)
                    dump(root/'progress.json',dict(stage='train_validation_heads',encoder_cases=len(encoder_manifest),
                        planned_cases=len(config['tickers'])*len(config['folds'])*len(config['seeds']),selected_heads=len(selected),
                        ticker=ticker,fold=fold,seed=seed))
                    del model,random
    dump(root/'selected_heads.json',selected);dump(root/'encoder_manifest.json',encoder_manifest)
    pd.DataFrame(samples).to_csv(root/'samples.csv',index=False)
    dump(root/'phase1_frozen.json',{name:sha(root/name) for name in
        ['config.json','code_manifest.json','selected_heads.json','encoder_manifest.json','encoder_source_manifest.json','samples.csv']})
    evaluate(root)
    dump(root/'progress.json',dict(status='complete',encoder_cases=len(encoder_manifest),new_heads=len(selected),
        backbone_updated=False,seconds=time.perf_counter()-started))
    print('COMPLETE',root,flush=True)


def evaluate(root):
    root=Path(root);config=json.loads((root/'config.json').read_text())
    for name,digest in json.loads((root/'phase1_frozen.json').read_text()).items():assert sha(root/name)==digest
    selected=json.loads((root/'selected_heads.json').read_text());records=[];daily=[];predictions=[]
    for ticker in config['tickers']:
        raw=load_raw(root/'encoder-source',ticker)
        for fold in config['folds']:
            for seed in config['seeds']:
                folder=root/ticker/f'fold-{fold}'/f'seed-{seed}';cache=read_cache(folder)
                queries=cache['queries'];splits=cache['splits'];y=cache['y'];test=splits=='test';codes=cache['codes']
                constant,markov=legacy_baseline(y,cache['previous'],splits)
                persistence=np.full((len(y),len(codes),5),.025)
                np.put_along_axis(persistence,cache['previous'][...,None],.9,axis=2)
                methods=dict(frequency=constant,markov=markov,persistence=persistence)
                for method in METHODS:
                    values=[]
                    for target,broker in enumerate(codes):
                        item=next(row for row in selected if row['ticker']==ticker and row['fold']==fold
                            and row['seed']==seed and row['method']==method and row['broker']==broker)
                        path=root/item['path'];assert sha(path)==item['sha256']
                        assert sha(folder/'features.npz')==item['features_sha256']
                        assert sha(folder.parent/'base_features.npz')==item['base_sha256']
                        with path.open('rb') as f:head=pickle.load(f)
                        x=all_features(cache,queries,splits,target,method)
                        values.append(probabilities(head,x))
                    methods[method]=np.stack(values,axis=1)
                for method,p in methods.items():
                    # Seed-independent benchmarks exported once per fold.
                    if method in BENCHMARKS and seed!=config['seeds'][0]:continue
                    actual=y[test];prob=p[test];context=dict(ticker=ticker,fold=fold,seed=seed if method in METHODS else -1,method=method)
                    records.append(dict(context,**scores(actual,prob)))
                    for k,index in enumerate(queries[test]):
                        losses=-np.log(prob[k,np.arange(len(codes)),actual[k]].clip(1e-12))
                        daily.append(dict(context,date=str(raw['dates'][index+1].date()),query_index=int(index),target_index=int(index+1),
                            mean_nll=float(losses.mean()),brokers=len(codes)))
                        for j,broker in enumerate(codes):
                            predictions.append(dict(context,date=str(raw['dates'][index+1].date()),query_index=int(index),
                                broker=broker,actual=int(actual[k,j]),**{f'p{c}':float(prob[k,j,c]) for c in range(5)}))
    pd.DataFrame(records).to_csv(root/'metrics.csv',index=False);pd.DataFrame(daily).to_csv(root/'daily_losses.csv',index=False)
    pd.DataFrame(predictions).to_csv(root/'predictions.csv',index=False)
    metric=pd.DataFrame(records);columns=['nll','state_accuracy','direction_accuracy','macro_state_recall','brier']
    metric.groupby(['ticker','method'])[columns].mean().reset_index().to_csv(root/'comparison.csv',index=False)
    metric.groupby('method')[columns].mean().reset_index().to_csv(root/'macro_summary.csv',index=False)
    statistical_summary(root,config)
    peer=pd.read_csv(root/'peer_removal_validation.csv')
    byfold=peer.groupby(['ticker','peer','broker','fold']).mean_delta_nll.mean().reset_index()
    pair=byfold.groupby(['ticker','peer','broker']).mean_delta_nll.agg(mean_delta_nll='mean',folds='count',
        positive_folds=lambda v:int((v>0).sum())).reset_index()
    fraction=peer.groupby(['ticker','peer','broker']).mean_delta_nll.apply(lambda v:float((v>0).mean())).rename('positive_fit_fraction').reset_index()
    pair=pair.merge(fraction,on=['ticker','peer','broker'])
    pair['stable_descriptive']=(pair.folds>=2)&(pair.positive_folds>=2)&(pair.positive_fit_fraction>=.6)&(pair.mean_delta_nll>0)
    pair.to_csv(root/'peer_candidates.csv',index=False)


def statistical_summary(root,config):
    frame=pd.read_csv(root/'daily_losses.csv')
    losses=frame.groupby(['date','fold','ticker','method']).mean_nll.mean().unstack('method')
    keys=[];columns=[]
    for comparator in BENCHMARKS+[m for m in METHODS if m!='learned_encoder']:
        gain=(losses[comparator]-losses.learned_encoder).unstack('ticker').sort_index()
        columns.append(gain);keys.extend([(ticker,comparator) for ticker in gain.columns])
    matrix=pd.concat(columns,axis=1).to_numpy();index=columns[0].index;folds=index.get_level_values('fold').to_numpy()
    rows=[]
    for block in [3,5,10]:
        mean,low,high,sd,critical=simultaneous_intervals(matrix,folds,repeats=config['bootstrap_repeats'],block=block)
        for i,(ticker,control) in enumerate(keys):
            rows.append(dict(ticker=ticker,comparator=control,block_sessions=block,gain_nll=float(mean[i]),
                simultaneous_low=float(low[i]),simultaneous_high=float(high[i]),standard_error=float(sd[i]),
                critical=float(critical),family_columns=matrix.shape[1],dates=int(np.isfinite(matrix[:,i]).sum())))
    pd.DataFrame(rows).to_csv(root/'paired_intervals.csv',index=False)
    metrics=pd.read_csv(root/'metrics.csv');stability=[]
    for ticker,part in metrics.groupby('ticker'):
        learned=part[part.method=='learned_encoder'].set_index(['fold','seed']).nll
        for comparator in BENCHMARKS+[m for m in METHODS if m!='learned_encoder']:
            control=part[part.method==comparator]
            if comparator in BENCHMARKS:
                baseline=control.set_index('fold').nll
                gain=pd.Series([baseline.loc[f]-value for (f,s),value in learned.items()],index=learned.index)
            else:gain=control.set_index(['fold','seed']).nll-learned
            stability.append(dict(ticker=ticker,comparator=comparator,gain_nll=float(gain.mean()),
                folds_won=int((gain.groupby(level='fold').mean()>0).sum()),folds=len(config['folds']),
                seeds_won=int((gain.groupby(level='seed').mean()>0).sum()),seeds=len(config['seeds'])))
    pd.DataFrame(stability).to_csv(root/'fold_seed_stability.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--source',default='runs/event-market-context-20261117');p.add_argument('--tickers',nargs='+')
    p.add_argument('--folds',type=int,default=3);p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);p.add_argument('--bootstrap-repeats',type=int,default=2000)
    run(p.parse_args())
