"""ADRO-only event-token embeddings, trained to predict next-session broker states."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA

from research import load_stock
from broker_tokens import fit_tokenizer,make_samples,STATES,CONTEXT_FEATURES


class EventEmbedding(nn.Module):
    def __init__(self,vocabulary,brokers,dimension=8,context_days=3):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary,dimension,padding_idx=0)
        self.transform=nn.Linear(dimension,dimension,bias=False)
        self.bias=nn.Parameter(torch.zeros(brokers,5))
        self.register_buffer('output_ids',torch.arange(1,1+brokers*5).reshape(brokers,5))
        w=torch.arange(1,context_days+1,dtype=torch.float32);self.register_buffer('recency',w/w.sum())
        nn.init.normal_(self.embedding.weight,mean=0,std=.1)
        with torch.no_grad():self.embedding.weight[0].zero_();self.transform.weight.copy_(torch.eye(dimension))

    def forward(self,tokens):
        vectors=self.embedding(tokens)
        nonpad=(tokens!=0).unsqueeze(-1)
        daily=(vectors*nonpad).sum(dim=2)/nonpad.sum(dim=2).clamp(min=1)
        context=(daily*self.recency[None,:,None]).sum(dim=1)
        hidden=self.transform(context)
        candidates=self.embedding(self.output_ids)
        return torch.einsum('nd,bkd->nbk',hidden,candidates)+self.bias


def probability(model,x):
    model.eval()
    with torch.no_grad():return model(torch.as_tensor(x,dtype=torch.long)).softmax(dim=-1).numpy()


def scores(target,p):
    chosen=np.take_along_axis(p,target[:,:,None],axis=2)[:,:,0]
    prediction=p.argmax(axis=2)
    macro=[]
    for b in range(target.shape[1]):
        recalls=[float((prediction[target[:,b]==k,b]==k).mean()) for k in range(5) if (target[:,b]==k).any()]
        macro.append(float(np.mean(recalls)))
    collapse=np.array([0,1,1,2,2])
    return {'nll':float(-np.log(chosen.clip(1e-12)).mean()),
            'state_accuracy':float((prediction==target).mean()),
            'direction_accuracy':float((collapse[prediction]==collapse[target]).mean()),
            'macro_state_recall':float(np.mean(macro)),
            'brier':float(np.mean(np.sum((p-np.eye(5)[target])**2,axis=2)))}


def baseline_probabilities(y,previous,masks):
    brokers=y.shape[1];train=y[masks['train']];past=previous[masks['train']]
    frequency=np.ones((brokers,5),dtype=float)
    transitions=np.ones((brokers,5,5),dtype=float)
    for b in range(brokers):
        frequency[b]+=np.bincount(train[:,b],minlength=5)
        for before,after in zip(past[:,b],train[:,b]):transitions[b,before,after]+=1
    frequency/=frequency.sum(axis=1,keepdims=True)
    transitions/=transitions.sum(axis=2,keepdims=True)
    constant=np.broadcast_to(frequency,(len(y),brokers,5)).copy()
    markov=np.stack([transitions[np.arange(brokers),row] for row in previous])
    return constant,markov,frequency,transitions


def fit_model(x,y,masks,vocabulary,brokers,args,seed,shuffle=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=EventEmbedding(vocabulary,brokers,args.dimension,args.context_days)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.01)
    tx=torch.as_tensor(x[masks['train']],dtype=torch.long);ty=torch.as_tensor(y[masks['train']],dtype=torch.long)
    if shuffle:
        order=np.random.default_rng(seed+10000).permutation(len(tx));tx=tx[order]
    vx=torch.as_tensor(x[masks['val']],dtype=torch.long);vy=torch.as_tensor(y[masks['val']],dtype=torch.long)
    best=float('inf');best_state=None;best_epoch=None;stale=0;history=[]
    initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    for epoch in range(args.epochs):
        model.train();optimizer.zero_grad()
        logits=model(tx);loss=nn.functional.cross_entropy(logits.reshape(-1,5),ty.reshape(-1))
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        model.eval()
        with torch.no_grad():vl=float(nn.functional.cross_entropy(model(vx).reshape(-1,5),vy.reshape(-1)))
        history.append({'epoch':epoch+1,'train_nll':float(loss.detach()),'validation_nll':vl})
        if vl<best-1e-7:
            best=vl;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
        else:stale+=1
        if stale>=args.patience:break
    model.load_state_dict(best_state)
    record={'seed':seed,'shuffled_training_context':shuffle,'best_epoch':best_epoch,'validation_nll':best,
            'initial_weights_sha256':initial_hash,'history':history,'parameters':sum(p.numel() for p in model.parameters())}
    return model,record


def export_embeddings(model,loaded,tokenizer,x,targets,masks,args,out):
    vectors=model.embedding.weight.detach().numpy()
    exposed=set()
    for target in targets[masks['train']]:exposed.update(range(target-args.context_days,target+1))
    from broker_tokens import transform_tokens
    daily,_,_=transform_tokens(loaded,tokenizer)
    frequency=np.zeros(len(vectors),dtype=int)
    for day in exposed:
        for token in set(daily[day]):frequency[token]+=1
    rows=[]
    for i,name in enumerate(tokenizer['tokens']):
        row={'token_id':i,'token':name,'training_days_seen':int(frequency[i]),'norm':float(np.linalg.norm(vectors[i]))}
        row.update({f'embedding_{j+1}':float(v) for j,v in enumerate(vectors[i])});rows.append(row)
    table=pd.DataFrame(rows)
    reliable=(frequency>=5)&(np.arange(len(vectors))>0)
    pca=PCA(n_components=3).fit(vectors[reliable]);xyz=pca.transform(vectors)
    for j in range(3):table[f'pc{j+1}']=xyz[:,j]
    table.to_csv(out/'token_vectors.csv',index=False)
    active=[]
    for i,name in enumerate(tokenizer['tokens']):
        if i==0 or i>len(tokenizer['brokers'])*5 or (i-1)%5==0 or frequency[i]<5:continue
        active.append(i)
    neighbor_rows=[]
    for query in active:
        code=tokenizer['tokens'][query].split('_')[0]
        candidates=[j for j in active if tokenizer['tokens'][j].split('_')[0]!=code]
        similarity=vectors[candidates]@vectors[query]/np.maximum(np.linalg.norm(vectors[candidates],axis=1)*np.linalg.norm(vectors[query]),1e-8)
        for rank,idx in enumerate(np.argsort(-similarity)[:5],1):
            j=candidates[int(idx)]
            neighbor_rows.append({'query':tokenizer['tokens'][query],'rank':rank,'neighbor':tokenizer['tokens'][j],
                                  'cosine_similarity':float(similarity[idx]),'query_train_days':int(frequency[query]),'neighbor_train_days':int(frequency[j])})
    neighbors=pd.DataFrame(neighbor_rows);neighbors.to_csv(out/'token_neighbors.csv',index=False)
    import plotly.graph_objects as go
    fig=go.Figure()
    colors={'BUY_NORMAL':'#60a5fa','BUY_LARGE':'#2563eb','SELL_NORMAL':'#f59e0b','SELL_LARGE':'#dc2626','NET_ZERO':'#94a3b8','CONTEXT':'#8b5cf6'}
    labels={'BUY_NORMAL':'Net buy biasa','BUY_LARGE':'Net buy besar','SELL_NORMAL':'Net sell biasa','SELL_LARGE':'Net sell besar','NET_ZERO':'Net nol / tidak tercatat','CONTEXT':'Konteks OHLCV / Retail'}
    for state in ['BUY_NORMAL','BUY_LARGE','SELL_NORMAL','SELL_LARGE','NET_ZERO','CONTEXT']:
        ids=[]
        for i,name in enumerate(tokenizer['tokens']):
            if not reliable[i]:continue
            is_broker=1<=i<=len(tokenizer['brokers'])*5
            if (state=='CONTEXT' and not is_broker) or (state!='CONTEXT' and is_broker and STATES[(i-1)%5]==state):ids.append(i)
        if not ids:continue
        text=[tokenizer['tokens'][i] for i in ids]
        fig.add_trace(go.Scatter3d(x=xyz[ids,0],y=xyz[ids,1],z=xyz[ids,2],mode='markers+text',
            marker=dict(size=5,color=colors[state],symbol='diamond' if state=='CONTEXT' else 'circle'),
            text=text,textposition='top center',textfont=dict(size=10),name=labels[state],
            customdata=[[int(frequency[i]),float(np.linalg.norm(vectors[i]))] for i in ids],
            hovertemplate='%{text}<br>Hari train memuat token: %{customdata[0]}<br>Norma vektor: %{customdata[1]:.3f}<extra></extra>',
            visible='legendonly' if state in ['NET_ZERO','CONTEXT'] else True))
    fig.update_layout(template='plotly_white',height=800,
        title=dict(text=f'ADRO — embedding kejadian broker<br><sup>Setiap titik = token perilaku broker · {args.dimension} dimensi asli · PCA 3D {100*pca.explained_variance_ratio_.sum():.1f}% variasi</sup>',x=.02),
        margin=dict(l=10,r=10,t=145,b=100),legend=dict(orientation='h',x=0,y=1.11),
        updatemenus=[dict(type='buttons',direction='right',x=0,y=1.04,buttons=[
            dict(label='Titik + label',method='restyle',args=[{'mode':'markers+text'}]),
            dict(label='Titik saja',method='restyle',args=[{'mode':'markers'}])])],
        scene=dict(xaxis_title='PC1',yaxis_title='PC2',zaxis_title='PC3',dragmode='orbit',aspectmode='data'),
        annotations=[dict(text='Drag untuk memutar · Scroll untuk zoom · Klik legenda untuk menampilkan kelompok',x=.5,y=-.08,xref='paper',yref='paper',showarrow=False),
                     dict(text='Kedekatan bukan bukti koordinasi broker. Token yang muncul <5 hari train disembunyikan.',x=.5,y=-.14,xref='paper',yref='paper',showarrow=False)])
    fig.write_html(out/'index.html',include_plotlyjs=True,full_html=True,config=dict(responsive=True,scrollZoom=True,displaylogo=False))
    (out/'embedding_summary.json').write_text(json.dumps({'pca_variance':float(pca.explained_variance_ratio_.sum()),
        'visible_broker_action_tokens':len(active),'reliable_tokens':int(reliable.sum()),'training_exposure_days':len(exposed),
        'nearest_neighbor_space':f'cosine in original {args.dimension} dimensions, not PCA coordinates'},indent=2))
    return table,neighbors


def run(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    loaded=load_stock('ADRO',args.summary,args.brokers)
    n=len(loaded[0]);train_stop=int(n*.6);val_stop=int(n*.8)
    tokenizer=fit_tokenizer(loaded,train_stop,args.brokers_count)
    x,y,previous,targets,masks,daily,states,valid=make_samples(loaded,tokenizer,args.context_days,val_stop)
    (out/'tokenizer.json').write_text(json.dumps(tokenizer,indent=2),encoding='utf-8')
    corpus=[]
    for i,day in enumerate(loaded[0].index):
        corpus.append({'date':str(day.date()),'valid':bool(valid[i]),'tokens':[tokenizer['tokens'][t] for t in daily[i]]})
    (out/'corpus.json').write_text(json.dumps(corpus,indent=2),encoding='utf-8')
    audit=[]
    for i,target in enumerate(targets):
        split=next(s for s,m in masks.items() if m[i])
        audit.append({'split':split,'context_start':str(loaded[0].index[target-args.context_days].date()),
                      'context_end':str(loaded[0].index[target-1].date()),'target_date':str(loaded[0].index[target].date()),'target_index':int(target)})
    pd.DataFrame(audit).to_csv(out/'samples.csv',index=False)
    np.savez_compressed(out/'token_dataset.npz',x=x,y=y,previous=previous,targets=targets)
    constant,markov,freq,transition=baseline_probabilities(y,previous,masks)
    np.savez_compressed(out/'baselines.npz',frequency=freq,transition=transition)
    records=[];metrics=[];models=[];all_predictions=[]
    for shuffle in [False,True]:
        method='shuffled_context' if shuffle else 'event_embedding'
        for seed in args.seeds:
            model,record=fit_model(x,y,masks,len(tokenizer['tokens']),len(tokenizer['brokers']),args,seed,shuffle)
            p=probability(model,x);records.append(record)
            metric={'method':method,'seed':seed,**scores(y[masks['test']],p[masks['test']])};metrics.append(metric)
            if not shuffle:models.append((record['validation_nll'],seed,model,record))
            for row,target in enumerate(targets[masks['test']]):
                for broker,code in enumerate(tokenizer['brokers']):
                    all_predictions.append({'method':method,'seed':seed,'date':str(loaded[0].index[target].date()),'broker':code,
                        'actual':STATES[y[masks['test']][row,broker]],'predicted':STATES[p[masks['test']][row,broker].argmax()],
                        **{state:float(p[masks['test']][row,broker,k]) for k,state in enumerate(STATES)}})
            print(method,seed,'best_epoch',record['best_epoch'],'test_nll',metric['nll'],flush=True)
    for name,p in [('training_frequency',constant),('markov_previous_state',markov)]:
        metrics.append({'method':name,'seed':-1,**scores(y[masks['test']],p[masks['test']])})
    # Embedding seed is chosen solely by validation NLL, never test score.
    best_val,best_seed,chosen,chosen_record=min(models,key=lambda row:row[0])
    torch.save({'state_dict':chosen.state_dict(),'tokenizer':tokenizer,'dimension':args.dimension,
                'context_days':args.context_days,'seed':best_seed,'validation_nll':best_val},out/'model.pt')
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(all_predictions).to_csv(out/'test_predictions.csv',index=False)
    (out/'training_runs.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    table,neighbors=export_embeddings(chosen,loaded,tokenizer,x,targets,masks,args,out)
    latest=None
    if valid[-args.context_days:].all():
        prob=probability(chosen,daily[-args.context_days:][None])[0]
        latest=[]
        for i,code in enumerate(tokenizer['brokers']):
            state=int(prob[i].argmax())
            latest.append({'broker':code,'most_likely_state':STATES[state],'probability':float(prob[i,state]),
                           'probabilities':{label:float(prob[i,k]) for k,label in enumerate(STATES)}})
    (out/'latest_next_session.json').write_text(json.dumps({'context_end':str(loaded[0].index[-1].date()),
        'target':'next trading session, date not assumed','seed':best_seed,'predictions':latest,
        'note':'Probabilities are per-broker states and are not calibrated or probabilities of stock-price increase.'},indent=2),encoding='utf-8')
    summary=pd.DataFrame(metrics).groupby('method',as_index=False)[['nll','state_accuracy','direction_accuracy','macro_state_recall','brier']].mean()
    summary.to_csv(out/'summary.csv',index=False)
    config=vars(args).copy();config.update(ticker='ADRO',source_brokers=args.brokers,source_summary=args.summary,
        source_start=str(loaded[0].index[0].date()),
        source_end=str(loaded[0].index[-1].date()),source_days=n,brokers=tokenizer['brokers'],vocabulary=len(tokenizer['tokens']),
        counts={s:int(m.sum()) for s,m in masks.items()},best_seed_by_validation=best_seed,parameters=chosen_record['parameters'],
        train_last_target=str(loaded[0].index[train_stop-1].date()),val_first_target=str(loaded[0].index[train_stop].date()),
        test_first_target=str(loaded[0].index[val_stop].date()),test_sessions=int(masks['test'].sum()),
        selected_seed_test=scores(y[masks['test']],probability(chosen,x[masks['test']])),
        device='cpu',source_quality=loaded[-1])
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    write_report(out,config,summary,neighbors,latest)
    print(summary.to_string(index=False),flush=True)


def write_report(out,config,summary,neighbors,latest):
    metrics=summary.set_index('method')
    real=float(metrics.loc['event_embedding','nll']);freq=float(metrics.loc['training_frequency','nll']);random_context=float(metrics.loc['shuffled_context','nll'])
    lines=['# ADRO: embedding token kejadian broker','',
        f'Ringkasan: NLL model {real:.4f}, baseline frekuensi {freq:.4f}, dan kontrol konteks acak {random_context:.4f}.',
        f'Perbaikan NLL terhadap frekuensi sekitar {100*(1-real/freq):.2f}%, tetapi selisih NLL terhadap konteks acak hanya {random_context-real:.6f}. Hubungan kontekstual yang berguna belum terbukti kuat.',
        'Akurasi lima state model juga belum mengalahkan baseline frekuensi. Grafik bukan bukti koordinasi broker atau kemampuan memprediksi harga.',
        '',
        f"Data {config['source_days']} sesi, {config['source_start']} sampai {config['source_end']}. Broker: {', '.join(config['brokers'])}.",
        f"{config['vocabulary']} token, embedding {config['dimension']} dimensi, {config['parameters']} parameter. Train/validasi/test: {config['counts']}.",
        'Setiap hari menjadi himpunan token broker-state dan delapan kondisi OHLCV/Retail. Tidak ada urutan broker buatan dalam satu hari.',
        'Embedding token dirata-rata per hari, dirangkum dengan bobot recency tiga hari, ditransformasi linear, lalu dibandingkan dengan embedding state broker yang sama (tied weights).',
        'Softmax dipisah per broker atas NET_ZERO/BUY_NORMAL/BUY_LARGE/SELL_NORMAL/SELL_LARGE, karena banyak broker dapat aktif bersamaan.',
        'NET_ZERO berarti net nol atau tidak tercatat pada tabel net; bukan bukti broker tidak melakukan transaksi gross.',
        'Objective: cross-entropy untuk state broker pada sesi berikutnya. Tidak ada target prediksi harga/return.',
        'Broker dan threshold besar/quantile kondisi hanya fit pada periode train. Normal/besar berarti relatif terhadap porsi net value historis broker itu, bukan gross turnover.',
        '',f"Seed embedding yang ditampilkan dipilih melalui validasi: {config['best_seed_by_validation']}. Angka model di tabel adalah rata-rata seluruh seed, bukan memilih test terbaik.",
        'Kontrol shuffled_context merusak pasangan konteks-target train; label, arsitektur dan inisialisasi seed tetap sama. Baseline frekuensi dan Markov memakai train saja.',
        '','| Metode | NLL (lebih kecil lebih baik) | Akurasi state | Akurasi arah net | Macro recall state | Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append(f"| {row['method']} | {row['nll']:.4f} | {100*row['state_accuracy']:.1f}% | {100*row['direction_accuracy']:.1f}% | {100*row['macro_state_recall']:.1f}% | {row['brier']:.4f} |")
    lines.extend(['','## Contoh tetangga lintas broker','',
        'Kedekatan memakai cosine pada embedding asli, bukan jarak PCA. Token harus muncul minimal 5 hari training.',
        '','| Token | Tetangga lintas broker terdekat | Cosine |','|---|---|---:|'])
    examples=neighbors[neighbors['rank']==1].head(10)
    for row in examples.to_dict('records'):lines.append(f"| {row['query']} | {row['neighbor']} | {row['cosine_similarity']:.3f} |")
    lines.extend(['','## Batas interpretasi','',
        'Embedding ini adaptasi konsep video, bukan replikasi source video: token berupa aksi broker, dan softmax per broker memprediksi kejadian sesi berikutnya.',
        'Titik adalah token perilaku, bukan potongan window atau satu broker tanpa kondisi. LOW/MID/HIGH adalah bin quantile training.',
        'Kedekatan dapat berasal dari pola atau frekuensi yang mirip; tidak membuktikan koordinasi, identitas investor, akumulasi tersembunyi, atau sebab-akibat harga.',
        '42 tanggal test dan broker pada hari sama saling terkait, bukan ratusan observasi independen. Lima seed tidak memberi bukti signifikansi statistik.',
        'Periode historis sudah dilihat dalam pengembangan sebelumnya. Belum ada konfirmasi pada data baru, uji beberapa tahun, kalibrasi probabilitas atau backtest profit.',
        'Data net bukan gross transaksi. Retail mengikuti definisi sumber. Token langka tidak ditampilkan dalam grafik agar tidak memberi kesan kuat pada representasi yang kurang terlatih.',
        'Buka index.html untuk memutar grafik; token_neighbors.csv dan token_vectors.csv memuat angka lengkap. latest_next_session.json adalah ranking state tiap broker untuk sesi berikutnya, bukan rekomendasi saham.'])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/adro-token-v1')
    parser.add_argument('--summary',default=r'data/Daily/Summary')
    parser.add_argument('--brokers',default=r'data/Daily/All')
    parser.add_argument('--brokers-count',type=int,default=12)
    parser.add_argument('--dimension',type=int,default=8)
    parser.add_argument('--context-days',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--patience',type=int,default=50)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,42,123,2026,3050])
    args=parser.parse_args()
    if min(args.brokers_count,args.context_days,args.epochs,args.patience)<1 or args.dimension<3:parser.error('Invalid positive model size')
    if len(set(args.seeds))!=len(args.seeds):parser.error('Use distinct seeds')
    run(args)
