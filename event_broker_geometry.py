"""Read-only audit of broker-ID and contextual SetEncoder geometry."""
import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from event_models import EventModel
from event_training import seed_all,weights_hash
from event_market_data import sha,dump
from event_market_study import load_raw
from event_predict import restore
from event_objective_study import runtime


def select_brokers(raw,train,count=12,min_active=50):
    """Train-query activity count; ties by absolute share, then broker code."""
    totals={}
    for i in train:
        for code,value,share in zip(raw['codes'][i][raw['mask'][i]],raw['tokens'][i,raw['mask'][i],0],raw['tokens'][i,raw['mask'][i],1]):
            if value:
                n,size=totals.get(str(code),(0,0.));totals[str(code)]=(n+1,size+abs(float(share)))
    codes=sorted([c for c,(n,_) in totals.items() if n>=min_active],key=lambda c:(-totals[c][0],-totals[c][1],c))[:count]
    if len(codes)<4:raise ValueError('Fewer than four eligible brokers in train')
    return codes,{c:dict(active_train_days=totals[c][0],absolute_share_sum=totals[c][1]) for c in codes}


def dense_selected(raw,codes,state=None):
    n=len(raw['dates']);tokens=np.zeros((n,len(codes),8),dtype='float32');present=np.zeros((n,len(codes)),bool)
    for j,code in enumerate(codes):
        match=(raw['codes']==code)&raw['mask']
        if np.any(match.sum(1)>1):raise ValueError('Repeated broker in one daily set')
        rows,slots=np.where(match);tokens[rows,j]=raw['tokens'][rows,slots];present[rows,j]=True
    if state is not None:
        tokens=np.where(present[...,None],np.clip((tokens-state['token_mean'])/state['token_std'],-10,10),0).astype('float32')
    return tokens,present


def state_labels(raw,codes,train):
    tokens,present=dense_selected(raw,codes);values=tokens[:,:,0];cuts=[];states=np.zeros(values.shape,dtype='int64')
    for j in range(len(codes)):
        active=abs(values[train,j]);active=active[active>0]
        if not len(active):raise ValueError('No train activity for selected broker')
        cut=float(np.quantile(active,2/3));cuts.append(cut)
        large=abs(values[:,j])>cut
        states[:,j]=np.where(values[:,j]>0,np.where(large,2,1),np.where(values[:,j]<0,np.where(large,4,3),0))
    return states,np.asarray(cuts),tokens,present


def initial_model(saved):
    seed_all(saved['seed']);model=EventModel(len(saved['features']),len(saved['token_names']),len(saved['preprocess']['vocabulary']),
        saved['groups'],saved['mode'],saved['width'])
    if weights_hash(model.state_dict())!=saved['record']['initial_weights_sha256']:
        raise ValueError('Cannot reproduce exact initial architecture/weights')
    return model


def encoded(enc,tokens,ids,mask):
    """Expose contextual broker vectors; pooled output matches existing forward."""
    mask=mask.bool();present=mask.any(1);safe=mask.clone();safe[~present,0]=True
    tokens=tokens.masked_fill(~mask[...,None],0);ids=ids.masked_fill(~mask,0)
    hidden=enc.project(tokens)+enc.identity(ids)
    for block in enc.blocks:hidden=block(hidden,~safe)
    pool=enc.pool(enc.seed.expand(len(tokens),-1,-1),hidden,hidden,key_padding_mask=~safe,need_weights=False)[0][:,0]
    return hidden*mask[...,None],pool*present[:,None]


def subset_pool(enc,tokens,mask,codes,state,rows,target,device,batch=256):
    lookup={c:i+2 for i,c in enumerate(state['vocabulary'])};code_ids=np.asarray([lookup[c] for c in codes],dtype='int64')
    result=[];enc.eval()
    with torch.no_grad():
        for start in range(0,len(rows),batch):
            use=rows[start:start+batch];m=mask[use].copy();m[:,target]=False
            t=torch.as_tensor(tokens[use],device=device);ids=torch.as_tensor(np.broadcast_to(code_ids,(len(use),len(codes))).copy(),device=device)
            _,pool=encoded(enc,t,ids,torch.as_tensor(m,device=device));result.append(pool.cpu().numpy())
    return np.concatenate(result)


def cosine(v):
    v=np.asarray(v,float);unit=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    return unit@unit.T


def rank_similarity(a,b):
    ids=np.triu_indices(len(a),1)
    x,y=a[ids],b[ids];valid=np.isfinite(x)&np.isfinite(y);x,y=x[valid],y[valid]
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(spearmanr(x,y).statistic)


def native_centroids(enc,data,raw,codes,train,device,batch=256):
    sums=np.zeros((len(codes),enc.identity.embedding_dim));counts=np.zeros(len(codes),int);maximum=0.
    enc.eval()
    with torch.no_grad():
        for start in range(0,len(train),batch):
            rows=train[start:start+batch];t=torch.as_tensor(data['tokens'][rows],device=device)
            ids=torch.as_tensor(data['ids'][rows],device=device);m=torch.as_tensor(data['mask'][rows],device=device)
            h,pool=encoded(enc,t,ids,m)
            expected=enc(t[:,None],ids[:,None],m[:,None])[:,0]
            torch.testing.assert_close(pool,expected,atol=1e-6,rtol=1e-6)
            maximum=max(maximum,float(abs(pool-expected).max()))
            hidden=h.cpu().numpy()
            for j,code in enumerate(codes):
                match=(raw['codes'][rows]==code)&data['mask'][rows]
                sums[j]+=hidden[match].sum(0);counts[j]+=int(match.sum())
    if (counts==0).any():raise ValueError('No native-context exposure for selected broker')
    return sums/counts[:,None],maximum


def audit_checkpoint(checkpoint,raw,prepared,ticker,fold,device):
    model,saved=restore(checkpoint,device);initial=initial_model(saved).to(device);state=saved['preprocess']
    train=prepared['train_indices'];test=prepared['test_indices'];codes,selection=select_brokers(raw,train)
    states,cuts,tokens,present=state_labels(raw,codes,train);ids=[state['vocabulary'].index(c)+2 for c in codes]
    learned_id=model.set_encoder.identity.weight.detach().cpu().numpy()[ids]
    initial_id=initial.set_encoder.identity.weight.detach().cpu().numpy()[ids]
    native,pool_error=native_centroids(model.set_encoder,prepared,raw,codes,train,device)
    native0,pool_error0=native_centroids(initial.set_encoder,prepared,raw,codes,train,device)
    signature=np.array([np.r_[np.bincount(states[train,j],minlength=5)/len(train),
        np.mean(abs(tokens[train,j,0])),np.std(tokens[train,j,0]),present[train,j].mean()] for j in range(len(codes))])
    signature=(signature-signature.mean(0))/np.maximum(signature.std(0),.01)
    matrices={name:cosine(values) for name,values in [('identity',learned_id),('initial_identity',initial_id),
        ('contextual_centroid',native),('initial_contextual_centroid',native0),('activity_signature',signature)]}
    # All train-selected pairs, evaluated descriptively; no test-pair selection.
    valid=test[raw['mask'][test].any(1)];test_flow=tokens[valid,:,0]
    correlation=np.corrcoef(test_flow.T);pair_rows=[];nearest=[]
    for i,j in combinations(range(len(codes)),2):
        pair_rows.append(dict(ticker=ticker,fold=fold,seed=saved['seed'],broker_a=codes[i],broker_b=codes[j],
            test_dates=len(valid),test_signed_log_net_correlation=float(correlation[i,j]),
            **{name:float(matrix[i,j]) for name,matrix in matrices.items()}))
    for name,matrix in matrices.items():
        for j,code in enumerate(codes):
            order=sorted([i for i in range(len(codes)) if i!=j],key=lambda i:(-matrix[j,i],codes[i]))[:3]
            nearest.extend([dict(ticker=ticker,fold=fold,seed=saved['seed'],geometry=name,broker=code,
                                 neighbor=codes[i],rank=rank+1,cosine=float(matrix[j,i])) for rank,i in enumerate(order)])
    record=dict(ticker=ticker,fold=fold,seed=saved['seed'],brokers=len(codes),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        identity_relative_l2_movement=float(np.linalg.norm(learned_id-initial_id)/max(np.linalg.norm(initial_id),1e-12)),
        identity_rank_vs_initial=rank_similarity(matrices['identity'],matrices['initial_identity']),
        contextual_rank_vs_initial=rank_similarity(matrices['contextual_centroid'],matrices['initial_contextual_centroid']),
        contextual_rank_vs_activity=rank_similarity(matrices['contextual_centroid'],matrices['activity_signature']),
        contextual_rank_vs_test_flow=rank_similarity(matrices['contextual_centroid'],correlation),
        initial_contextual_rank_vs_test_flow=rank_similarity(matrices['initial_contextual_centroid'],correlation),
        pooled_forward_replay_max_error=max(pool_error,pool_error0),selection=selection,brokers_selected=codes,
        thresholds_log_abs_net=cuts.tolist())
    return record,pair_rows,nearest


def stability(neighbors):
    rows=[]
    for (ticker,geometry),part in neighbors.groupby(['ticker','geometry']):
        groups={key:sub for key,sub in part.groupby(['fold','seed'])}
        for keya,keyb in combinations(groups,2):
            a,b=groups[keya],groups[keyb];shared=set(a.broker)&set(b.broker)
            common=shared
            scores=[]
            for broker in sorted(shared):
                aa=set(a[a.broker==broker].neighbor)&common;bb=set(b[b.broker==broker].neighbor)&common
                if aa and bb:scores.append(len(aa&bb)/len(aa|bb))
            if scores:rows.append(dict(ticker=ticker,geometry=geometry,fold_a=keya[0],seed_a=keya[1],fold_b=keyb[0],seed_b=keyb[1],
                comparison='across_seed_same_fold' if keya[0]==keyb[0] else 'across_fold',common_brokers=len(common),mean_top3_jaccard=float(np.mean(scores))))
    return pd.DataFrame(rows)


def common_set_stability(pairs):
    """Recompute top3 in the common eligible universe, never filter eligibility
    according to which brokers happened to be selected as neighbors."""
    rows=[];names=['identity','initial_identity','contextual_centroid','initial_contextual_centroid','activity_signature']
    for ticker,part in pairs.groupby('ticker'):
        groups={key:sub for key,sub in part.groupby(['fold','seed'])}
        for keya,keyb in combinations(groups,2):
            aa,bb=groups[keya],groups[keyb]
            common=(set(aa.broker_a)|set(aa.broker_b))&(set(bb.broker_a)|set(bb.broker_b))
            if len(common)<4:continue
            for name in names:
                lookups=[{frozenset([r.broker_a,r.broker_b]):getattr(r,name) for r in frame.itertuples()}
                         for frame in [aa,bb]]
                scores=[]
                for code in sorted(common):
                    neighbors=[]
                    for lookup in lookups:
                        neighbors.append(set(sorted(common-{code},key=lambda other:(-lookup[frozenset([code,other])],other))[:3]))
                    scores.append(len(neighbors[0]&neighbors[1])/len(neighbors[0]|neighbors[1]))
                rows.append(dict(ticker=ticker,geometry=name,fold_a=keya[0],seed_a=keya[1],fold_b=keyb[0],seed_b=keyb[1],
                    comparison='across_seed_same_fold' if keya[0]==keyb[0] else 'across_fold',common_brokers=len(common),
                    mean_top3_jaccard=float(np.mean(scores))))
    return pd.DataFrame(rows)


def run(source,out,device='auto'):
    source=Path(source);root=Path(out);root.mkdir(parents=True,exist_ok=False);device=runtime(device)
    (root/'sources').mkdir()
    for name in ['event_broker_geometry.py','EVENT-RELATION-AUDIT.md','EVENT-RELATION-PROTOCOL.md']:
        shutil.copy2(Path(__file__).parent/name,root/'sources'/name)
    config=json.loads((source/'config.json').read_text());records=[];pairs=[];neighbors=[]
    for ticker in config['tickers']:
        raw=load_raw(source,ticker)
        for fold in config['folds']:
            folder=source/ticker/f'fold-{fold}'/'local_only';prepared=np.load(folder/'prepared.npz',allow_pickle=False)
            for seed in config['seeds']:
                print('GEOMETRY',ticker,fold,seed,flush=True)
                record,part,near=audit_checkpoint(folder/f'none-seed{seed}/best.pt',raw,prepared,ticker,fold,device)
                records.append(record);pairs.extend(part);neighbors.extend(near)
                dump(root/'checkpoint_audit.json',records)
                pd.DataFrame(pairs).to_csv(root/'all_pairs.csv',index=False);pd.DataFrame(neighbors).to_csv(root/'neighbors.csv',index=False)
    scalar=[{key:value for key,value in row.items() if key not in ['selection','brokers_selected','thresholds_log_abs_net']} for row in records]
    pd.DataFrame(scalar).to_csv(root/'geometry_summary.csv',index=False)
    common_set_stability(pd.DataFrame(pairs)).to_csv(root/'stability.csv',index=False)
    dump(root/'config.json',dict(source=str(source.resolve()),device=device,checkpoints=len(records),
        policy='27 local_only checkpoints preselected; train-only broker selection and native-context centroids; all pairs retained',
        test_status='Exploratory historical reuse; geometry and flow correlations descriptive, not causal or confirmatory'))
    print('COMPLETE',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='runs/event-market-context-20261117')
    p.add_argument('--out',required=True);p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    args=p.parse_args();run(args.source,args.out,args.device)
