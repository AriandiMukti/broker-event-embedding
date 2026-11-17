"""Per-stock feature selection, compact models and an expanded 25-stock universe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from main import Encoder,read_summary,HORIZONS
from research import load_stock,make_folds,normalize_fold,fit_variant
from compact_features import compact_matrix,feature_mask,dataset,PRICE,NUMERIC,GROUPS,align_loaded


def choose_universe(args):
    existing=json.loads((Path(args.previous)/'config.json').read_text())['tickers']
    reference,_=read_summary(Path(args.summary)/'ADRO_Summary.csv')
    directories=sorted(p.name for p in Path(args.brokers).iterdir() if p.is_dir() and len(p.name)==10 and p.name[4]=='-')
    calendar=reference.loc[:directories[-1]].index
    cutoff=calendar[make_folds(len(calendar))[0]['train_end']]
    candidates=[]
    for path in sorted(Path(args.summary).glob('*_Summary.csv')):
        ticker=path.name.removesuffix('_Summary.csv')
        if ticker in existing or len(ticker)!=4 or not ticker.isalpha():continue
        try:
            df,good=read_summary(path)
            train=df.loc[(df.index<=cutoff)&good]
            proxy=(train.Close*train.Volume);proxy=proxy[proxy>0]
            if len(proxy)>=90:
                candidates.append({'ticker':ticker,'median_value_proxy':float(proxy.median()),'training_rows':len(proxy)})
        except (ValueError,KeyError):continue
    candidates=sorted(candidates,key=lambda r:(-r['median_value_proxy'],r['ticker']))
    added=candidates[:args.add_stocks]
    if len(added)!=args.add_stocks:raise ValueError('Insufficient additional eligible stocks')
    return {'existing':existing,'added':added,'tickers':existing+[r['ticker'] for r in added],
            'selection_cutoff':str(cutoff.date()),'selection_rule':'Median Close*Volume on valid training-prefix rows only; >=90 positive-volume rows. Proxy, not verified exchange turnover.',
            'ranked_candidates':candidates}


def ridge_candidate(xn,yn,masks,selected,pairs):
    if not selected:
        score=float(np.mean(yn[masks['val']]**2))
        return None,score,None
    allowed=feature_mask(selected,pairs,xn.shape[2])
    flat=xn[:,:,allowed].reshape(len(xn),-1)
    best=float('inf');chosen=None;alpha_chosen=None
    for alpha in (1.,100.,10000.):
        model=Ridge(alpha=alpha).fit(flat[masks['train']],yn[masks['train']])
        score=float(np.mean((model.predict(flat[masks['val']])-yn[masks['val']])**2))
        if score<best:best=score;chosen=model;alpha_chosen=alpha
    return chosen,best,alpha_chosen


def select_ridge(xn,yn,masks,pairs,max_features=8,min_gain=.02):
    _,current,_=ridge_candidate(xn,yn,masks,[],pairs)
    selected=[];path=[];attempts=[]
    eligible=[name for name in NUMERIC if xn[masks['train'],:,pairs[name][0]].std()>1e-6]
    for step in range(max_features):
        best=None
        for name in eligible:
            if name in selected:continue
            _,score,alpha=ridge_candidate(xn,yn,masks,selected+[name],pairs)
            record={'step':step+1,'added_feature':name,'features':selected+[name],'validation_mse':score,'alpha':alpha}
            attempts.append(record)
            if best is None or score<best['validation_mse']:best=record
        if best is None or best['validation_mse']>=current*(1-min_gain):break
        selected=best['features'];current=best['validation_mse'];path.append(best)
    return selected,path,attempts


def infer(state,xn,allowed,device):
    model=Encoder(xn.shape[2],xn.shape[1],channels=16,compact_decoder=True).to(device)
    model.load_state_dict(state);model.eval()
    x=xn.copy();x[:,:,~allowed]=0
    gain=float(np.sqrt(x.shape[2]/int(allowed.sum())))
    with torch.no_grad():z,p,_=model(torch.from_numpy(x).to(device)*gain)
    return z.cpu().numpy(),p.cpu().numpy()


def prune_state(state,allowed):
    result={name:value.clone() for name,value in state.items()}
    result['conv.0.weight']=result['conv.0.weight'][:,allowed,:].clone()
    result['decode.2.weight']=result['decode.2.weight'][allowed,:].clone()
    result['decode.2.bias']=result['decode.2.bias'][allowed].clone()
    return result


def choose_models(x,y,masks,pairs,args,device):
    xn,yn,mean,std,ym,ys=normalize_fold(x,y,masks)
    selected,path,attempts=select_ridge(xn,yn,masks,pairs,args.max_features,args.min_gain)
    ridge,_,alpha=ridge_candidate(xn,yn,masks,selected,pairs)
    base_ridge,_,base_alpha=ridge_candidate(xn,yn,masks,PRICE,pairs)
    # Ridge proposes at most three compact subsets; CNN validates these independently.
    recipes=[PRICE]
    for entry in sorted(path,key=lambda row:row['validation_mse'])[:3]:
        if entry['features'] not in recipes:recipes.append(entry['features'])
    candidates=[]
    safe_x=xn.copy();safe_y=yn.copy()
    future=~(masks['train']|masks['val'])
    safe_x[future]=0;safe_y[future]=0
    fit_masks={'train':masks['train'],'val':masks['val'],'test':masks['val']}
    for recipe in recipes:
        allowed=feature_mask(recipe,pairs,x.shape[2]);records=[];states=[]
        for seed in args.seeds:
            _,record,state=fit_variant(safe_x,safe_y,safe_y,fit_masks,allowed,seed,args.epochs,device,return_state=True)
            records.append(record);states.append(state)
        candidates.append({'features':recipe,'score':float(np.mean([r['val_mse_standardized'] for r in records])),
                           'records':records,'states':states})
    for j,seed in enumerate(args.seeds):
        if len({r['records'][j]['initial_weights_sha256'] for r in candidates})!=1:raise ValueError('Initialization mismatch')
    winner=min(candidates,key=lambda row:row['score'])
    base=candidates[0]
    audit={'ridge_selected_features':selected,'ridge_alpha':alpha,'ohlcv_ridge_alpha':base_alpha,
           'ridge_path':path,'ridge_attempts':attempts,
           'cnn_selected_features':winner['features'],'cnn_validation_mse':winner['score'],
           'cnn_candidates':[{'features':r['features'],'validation_mse':r['score'],'records':r['records']} for r in candidates],
           'selection_policy':f'Ridge greedy forward selection (>={100*args.min_gain:.1f}% validation MSE reduction per addition), max {args.max_features} numeric features; CNN selects from top 3 Ridge-path subsets plus OHLCV baseline by mean validation MSE over seeds. No test features or labels reach fitting/selection.'}
    return xn,yn,mean,std,ym,ys,ridge,base_ridge,winner,base,audit


def ridge_predict(model,xn,selected,pairs):
    if not selected:return np.zeros((len(xn),3),dtype='float32')
    mask=feature_mask(selected,pairs,xn.shape[2])
    return model.predict(xn[:,:,mask].reshape(len(xn),-1))


def add_forecasts(rows,ticker,fold,model,seed,meta,actual,pred):
    for i,date in enumerate(meta.date):
        for j,h in enumerate(HORIZONS):
            rows.append({'ticker':ticker,'fold':fold,'model':model,'seed':seed,'date':date,'horizon':h,
                         'actual':float(actual[i,j]),'prediction':float(pred[i,j])})


def latest_model(root,ticker,loaded,matrix,names,pairs,args,device):
    n=len(matrix)
    fold={'val_start':n-30,'test_start':n,'test_stop':n}
    x,y,meta,masks=dataset(loaded,matrix,fold,latest=True)
    xn,yn,mean,std,ym,ys,ridge,base_ridge,winner,base,audit=choose_models(x,y,masks,pairs,args,device)
    selected=winner['features'];allowed=feature_mask(selected,pairs,len(names));state=winner['states'][0]
    z,p=infer(state,xn,allowed,device);p=p*ys+ym
    folder=root/'latest'/ticker;folder.mkdir(parents=True,exist_ok=True)
    physical_state=prune_state(state,allowed)
    checkpoint={'state_dict':physical_state,'features':int(allowed.sum()),
                'feature_names':[name for name,active in zip(names,allowed) if active],
                'candidate_feature_names':names,'source_feature_mask':allowed,
                'selected_features':selected,
                'mean':mean[allowed],'std':std[allowed],'target_mean':ym,'target_std':ys,
                'input_gain':float(np.sqrt(len(names)/int(allowed.sum()))),
                'channels':16,'window':20,'compact_decoder':True,'ticker':ticker,'seed':args.seeds[0]}
    torch.save(checkpoint,folder/'model.pt')
    reference=masks['train'];pca=PCA(n_components=3).fit(z[reference]);xyz=pca.transform(z[reference])
    latest_valid=bool(masks['latest'].any())
    q=pca.transform(z[masks['latest']])[0].tolist() if latest_valid else [None,None,None]
    rz=z[reference]
    if latest_valid:
        query=z[masks['latest']][0]
        similarity=rz@query/np.maximum(np.linalg.norm(rz,axis=1)*np.linalg.norm(query),1e-8)
    else:similarity=np.array([])
    refmeta=meta.loc[reference].reset_index(drop=True);chosen=[];dates=[]
    for i in np.argsort(-similarity):
        date=pd.Timestamp(refmeta.iloc[i].date)
        if any(abs((date-old).days)<28 for old in dates):continue
        dates.append(date);chosen.append(int(i))
        if len(chosen)==10:break
    neighbor=refmeta.iloc[chosen].copy();neighbor['cosine_similarity']=similarity[chosen]
    for j,h in enumerate(HORIZONS):neighbor[f'realized_return_{h}']=np.expm1(y[reference][chosen,j])
    neighbor.to_csv(folder/'neighbors.csv',index=False)
    predictions={str(h):float(np.expm1(p[masks['latest']][0,j])) if latest_valid else None for j,h in enumerate(HORIZONS)}
    projection={'ticker':ticker,'latest_date':str(loaded[0].index[-1].date()),'variance_explained':float(pca.explained_variance_ratio_.sum()),
                'train_dates':refmeta.date.tolist(),'train_xyz':xyz.tolist(),'query_xyz':q,
                'neighbors_dates':neighbor.date.tolist(),'predictions_arithmetic_return':predictions,
                'selected_features':selected,'latest_features':{},'latest_prediction_available':latest_valid}
    (folder/'projection.json').write_text(json.dumps(projection,indent=2),encoding='utf-8')
    (folder/'selection.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (folder/'data_quality.json').write_text(json.dumps(loaded[-1],indent=2),encoding='utf-8')
    return {'ticker':ticker,'selected_features':selected,'numeric_feature_count':len(selected),
            'candidate_fits':len(audit['cnn_candidates'])*len(args.seeds),
            'input_channels_including_masks':int(allowed.sum()),'seed':args.seeds[0],
            'train_samples':int(reference.sum()),'val_samples':int(masks['val'].sum()),
            'date':projection['latest_date'],'predictions':predictions,
            'latest_prediction_available':latest_valid,'data_quality':loaded[-1],
            'model_sha256':hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()}


def summarize(root,output,universe,args):
    metric=[]
    for (ticker,model,seed,h),g in output.groupby(['ticker','model','seed','horizon']):
        actual=g.actual.to_numpy();pred=g.prediction.to_numpy()
        up=actual>0;correct=(pred>0)==up
        metric.append({'ticker':ticker,'model':model,'seed':seed,'horizon':h,'n_dates':len(g),
                       'mae':float(np.mean(abs(actual-pred))),'direction_accuracy':float(correct.mean()),
                       'balanced_accuracy':float((correct[up].mean()+correct[~up].mean())/2) if up.any() and (~up).any() else np.nan})
    metrics=pd.DataFrame(metric);metrics.to_csv(root/'metrics_by_seed.csv',index=False)
    average=metrics.groupby(['ticker','model','horizon'],as_index=False)[['mae','direction_accuracy','balanced_accuracy']].mean()
    comparison=average.pivot(index=['ticker','horizon'],columns='model',values='mae').reset_index();comparison.columns.name=None
    comparison['adaptive_cnn_gain_vs_ohlcv']=1-comparison.adaptive_cnn/comparison.ohlcv_cnn
    comparison['adaptive_ridge_gain_vs_ohlcv']=1-comparison.adaptive_ridge/comparison.ohlcv_ridge
    comparison.to_csv(root/'comparison.csv',index=False)
    baseline=average[average.model=='train_mean'][['ticker','horizon','mae']].rename(columns={'mae':'baseline'})
    normalized=average.merge(baseline,on=['ticker','horizon'])
    normalized['relative_mae']=normalized.mae/normalized.baseline.where(normalized.baseline>1e-10)
    macro=normalized.groupby('model',as_index=False).agg(relative_mae=('relative_mae','mean'),defined_pairs=('relative_mae','count'),
                    direction_accuracy=('direction_accuracy','mean'),balanced_accuracy=('balanced_accuracy','mean'))
    macro.to_csv(root/'macro_summary.csv',index=False)
    wins_cnn=int((comparison.adaptive_cnn<comparison.ohlcv_cnn).sum());wins_ridge=int((comparison.adaptive_ridge<comparison.ohlcv_ridge).sum())
    lines=[f'# Pemilihan fitur per saham: {len(universe["tickers"])} saham mandiri','',
       f"Saham tambahan: {', '.join(r['ticker'] for r in universe['added'])}. Pemilihan likuiditas hanya sampai {universe['selection_cutoff']}.",
       'Setiap saham/fold memiliki susunan fitur Ridge dan CNN sendiri. Fitur tidak dipilih dari hasil test.',
       '38 kandidat numerik, masing-masing dipasangkan dengan missing flag. Tidak memakai kolom individual per broker.',
       'Ridge melakukan forward selection dari nol; tambahan harus memperbaiki MSE validasi minimal 2%, maksimum 8 fitur.',
       'CNN membandingkan maksimal 3 subset dari jalur Ridge dan baseline OHLCV menggunakan rata-rata validasi dua seed. Model boleh tetap OHLCV.',
       'Semua kandidat CNN memakai arsitektur dan bobot awal identik per seed, kanal nonaktif nol, serta koreksi input energy yang sama seperti riset sebelumnya.',
       '',f'CNN adaptif mengurangi MAE terhadap CNN OHLCV pada {wins_cnn}/{len(comparison)} pasangan saham-horizon.',
       f'Ridge adaptif mengurangi MAE terhadap Ridge OHLCV pada {wins_ridge}/{len(comparison)} pasangan saham-horizon.',
       '','| Model | MAE relatif ke baseline rata-rata train | Pasangan terdefinisi |','|---|---:|---:|']
    for row in macro.to_dict('records'):lines.append(f"| {row['model']} | {row['relative_mae']:.3f} | {row['defined_pairs']} |")
    lines.extend(['','## Batasan','',
       'Target desain: 37/51/65 train, 10 validasi, 14 test per fold. Baris OHLCV invalid atau tanggal hilang mengurangi sampel aktual; lihat sample_counts.csv. Label return tetap overlap.',
       'Validation set kecil dan banyak kandidat diuji, sehingga pilihan fitur rentan overfit. Dua seed tidak cukup membuktikan signifikansi.',
       'Periode historis sudah dipakai dalam pengembangan sebelumnya. Ini eksperimen pengembangan, bukan konfirmasi pada data baru.',
       'Hasil tidak membuktikan profit transaksi; belum ada biaya, slippage, atau backtest trading.',
       'Model terbaru di latest/ dipilih kembali menggunakan label yang tersedia sampai akhir data, memakai 99 train dan 10 validasi; itu bukan tambahan hasil test.',
       'Grafik 3D latest/index.html memakai model seed pertama. PCA dan bank pola selalu dari train saham terkait.',
       'Pemilihan fitur seluruh horizon dilakukan bersama, bukan memilih susunan berbeda setelah melihat test setiap horizon.'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return macro


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    universe=(json.loads((root/'universe.json').read_text()) if args.resume and (root/'universe.json').exists() else choose_universe(args))
    if args.tickers:
        if not set(args.tickers).issubset(set(universe['tickers'])):raise ValueError('Requested ticker outside selected universe')
        universe['tickers']=args.tickers
    (root/'universe.json').write_text(json.dumps(universe,indent=2),encoding='utf-8')
    print('Additional stocks:',[r['ticker'] for r in universe['added']],flush=True)
    forecasts=[];selections=[];latest=[];fits=0
    if args.resume:
        if (root/'predictions.csv').exists():forecasts=pd.read_csv(root/'predictions.csv').to_dict('records')
        if (root/'selection_audit.json').exists():selections=json.loads((root/'selection_audit.json').read_text())
        if (root/'latest_manifest.json').exists():latest=json.loads((root/'latest_manifest.json').read_text())
        fits=sum(len(r['cnn_candidates'])*len(args.seeds) for r in selections)
    if args.refresh_stocks:
        refresh=set(args.refresh_stocks)
        forecasts=[r for r in forecasts if r['ticker'] not in refresh]
        selections=[r for r in selections if r['ticker'] not in refresh]
        latest=[r for r in latest if r['ticker'] not in refresh]
        fits=sum(len(r['cnn_candidates'])*len(args.seeds) for r in selections)
    reference,_=read_summary(Path(args.summary)/'ADRO_Summary.csv')
    directories=sorted(p.name for p in Path(args.brokers).iterdir() if p.is_dir() and len(p.name)==10 and p.name[4]=='-')
    calendar=reference.loc[:directories[-1]].index
    for number,ticker in enumerate(universe['tickers'],1):
        if any(r['ticker']==ticker for r in latest):
            print(f'Resume: completed {ticker}',flush=True);continue
        loaded=align_loaded(load_stock(ticker,args.summary,args.brokers),calendar);matrix,names,pairs=compact_matrix(loaded)
        for fold in make_folds(len(matrix)):
            if any(r['ticker']==ticker and r['fold']==fold['fold'] for r in selections):
                print(f'Resume: completed {ticker} fold {fold["fold"]}',flush=True);continue
            x,y,meta,masks=dataset(loaded,matrix,fold)
            xn,yn,mean,std,ym,ys,ridge,base_ridge,winner,base,audit=choose_models(x,y,masks,pairs,args,device)
            fits+=len(audit['cnn_candidates'])*len(args.seeds)
            audit.update(ticker=ticker,fold=fold['fold'],val_start=str(loaded[0].index[fold['val_start']].date()),
                         test_start=str(loaded[0].index[fold['test_start']].date()),samples=meta.to_dict('records'))
            selections.append(audit)
            tm=meta.loc[masks['test']].reset_index(drop=True);actual=y[masks['test']]
            for key,selection,model in [('adaptive_ridge',audit['ridge_selected_features'],ridge),('ohlcv_ridge',PRICE,base_ridge)]:
                prediction=ridge_predict(model,xn[masks['test']],selection,pairs)*ys+ym
                add_forecasts(forecasts,ticker,fold['fold'],key,0,tm,actual,prediction)
            add_forecasts(forecasts,ticker,fold['fold'],'train_mean',0,tm,actual,np.tile(ym,(len(tm),1)))
            for key,candidate in [('adaptive_cnn',winner),('ohlcv_cnn',base)]:
                allowed=feature_mask(candidate['features'],pairs,len(names))
                for seed,state in zip(args.seeds,candidate['states']):
                    _,prediction=infer(state,xn[masks['test']],allowed,device)
                    add_forecasts(forecasts,ticker,fold['fold'],key,seed,tm,actual,prediction*ys+ym)
            print(f'[{number}/{len(universe["tickers"])}] {ticker} fold {fold["fold"]}: Ridge={audit["ridge_selected_features"]}; CNN={audit["cnn_selected_features"]}',flush=True)
            pd.DataFrame(forecasts).to_csv(root/'predictions.csv',index=False)
            (root/'selection_audit.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
        record=latest_model(root,ticker,loaded,matrix,names,pairs,args,device);latest.append(record)
        (root/'latest_manifest.json').write_text(json.dumps(latest,indent=2),encoding='utf-8')
        print(f'Latest model {ticker}: {record["selected_features"]}',flush=True)
    output=pd.DataFrame(forecasts);macro=summarize(root,output,universe,args)
    output.groupby(['ticker','model','seed','horizon'],as_index=False).date.nunique().rename(columns={'date':'test_query_dates'}).to_csv(root/'sample_counts.csv',index=False)
    config=vars(args).copy();config.update(device=str(device),stocks=universe['tickers'],candidate_features=NUMERIC,
       input_channels=2*len(NUMERIC),research_cnn_fits=fits,latest_models=len(latest),seeds=args.seeds,
       latest_cnn_fits=sum(r['candidate_fits'] for r in latest),
       note='Latest model selection also fits candidates; research_cnn_fits counts walk-forward candidates only.')
    (root/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    pd.DataFrame([{'ticker':r['ticker'],'selected_features':'; '.join(r['selected_features']),
                  'numeric_feature_count':r['numeric_feature_count'],'input_channels':r['input_channels_including_masks']} for r in latest]).to_csv(root/'latest_selected_features.csv',index=False)
    from dashboard import build_dashboard
    build_dashboard(root/'latest',universe['tickers'])
    print(macro.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',default='runs/research-v1')
    parser.add_argument('--out',default='runs/adaptive-v1')
    parser.add_argument('--summary',default=r'data/Daily/Summary')
    parser.add_argument('--brokers',default=r'data/Daily/All')
    parser.add_argument('--add-stocks',type=int,default=10)
    parser.add_argument('--tickers',nargs='+',help='Optional subset of the automatically selected universe')
    parser.add_argument('--resume',action='store_true',help='Continue completed ticker/fold checkpoints in the same output directory')
    parser.add_argument('--refresh-stocks',nargs='+',help='Recompute named stocks while retaining other completed stocks on resume')
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,123])
    parser.add_argument('--max-features',type=int,default=8)
    parser.add_argument('--min-gain',type=float,default=.02)
    args=parser.parse_args()
    if args.tickers:args.tickers=[t.upper() for t in args.tickers]
    if args.refresh_stocks:args.refresh_stocks=[t.upper() for t in args.refresh_stocks]
    if min(args.epochs,args.max_features)<1 or args.add_stocks<0:parser.error('Invalid experiment sizes')
    if not 0<=args.min_gain<1 or len(set(args.seeds))!=len(args.seeds):parser.error('Invalid selection settings')
    run(args)
