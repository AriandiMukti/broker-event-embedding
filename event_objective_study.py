"""Paired objective ablation, common return-MAE selection and frozen inputs.

Phase 1 fits/selects ALL checkpoints on train/validation before phase 2 exports
test forecasts. Historical tests have already been explored: no confirmation
claim, no pooled stock training, no feature/architecture search in this study.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score

from event_training import fit,infer,decoded,selection_loss
from event_predict import restore
from event_uncertainty import intervals

TASKS=['multitask','return_only']
MODES={'cnn':'CNN','none':'Set+TCN','static':'Static','dynamic':'Dynamic'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return str(x)
    Path(path).write_text(json.dumps(value,indent=2,default=convert,allow_nan=False),encoding='utf-8')


def runtime(device):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():device='cpu'
    if device=='cuda':
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return device


def return_metrics(actual,prediction):
    error=prediction-actual;positive=actual>0;guess=prediction>0
    return dict(n=len(actual),mae=float(abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(error.mean()),direction_accuracy=float(accuracy_score(positive,guess)),
        balanced_direction_accuracy=float(balanced_accuracy_score(positive,guess)) if len(np.unique(positive))==2 else None)


def get_data(folder,source_checkpoint,ticker):
    saved=torch.load(source_checkpoint,map_location='cpu',weights_only=False)
    arrays=np.load(folder/'prepared.npz',allow_pickle=False)
    data={k:arrays[k] for k in ['x','tokens','ids','mask','y']}
    data.update(ticker=ticker,names=saved['features'],groups=saved['groups'],source_hash=saved['source_hash'])
    split={s:arrays[s+'_indices'] for s in ['train','val','test']};state=saved['preprocess']
    for s,key in [('train','training_indices'),('val','validation_indices')]:
        np.testing.assert_array_equal(split[s],saved[key])
    np.testing.assert_allclose(state['target_mean'],data['y'][split['train']].mean(0),atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(state['target_std'],data['y'][split['train']].std(0).clip(.001),atol=1e-7,rtol=1e-6)
    return data,split,state,saved


def run(args):
    source=Path(args.source).resolve();args.source=str(source)
    source_config=json.loads((source/'config.json').read_text())
    args.device=runtime(args.device);root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    stocks=args.tickers or source_config['tickers'];seeds=args.seeds or source_config['seeds']
    folds=list(range(1,(args.folds or source_config['folds'])+1))
    if not set(stocks)<=set(source_config['tickers']):raise ValueError('Stock absent from immutable source cohort')
    if not set(seeds)<=set(source_config['seeds']):raise ValueError('Requested new seeds require explicit protocol')
    config=dict(vars(args),tickers=stocks,seeds=seeds,folds=folds,torch=torch.__version__,python=platform.python_version(),
        window=source_config['window'],width=source_config['width'],batch_size=source_config['batch_size'],
        lr=source_config['lr'],no_amp=source_config['no_amp'],
        embargo=source_config['embargo'],max_horizon=20,
        selection_metric='return_mae',selection_definition='mean absolute raw log-return error over validation dates and horizons 5/10/20',
        training_return='MSE of train-standardized returns, same in both arms',
        auxiliary_weights=dict(log_vol_mse=.25,direction_bce=.25),
        return_only_auxiliary_heads='untrained; output values must not be evaluated or used',
        checkpoint_min_delta=1e-6,planned_fits=len(stocks)*len(folds)*len(seeds)*len(args.modes)*2,
        data_policy='Reuse exact train-only preprocessing and immutable prepared arrays; initialize model weights afresh',
        test_status='Historical exploratory reuse; no untouched confirmation test',
        primary_comparison='return_only vs multitask, common validation selection; also zero/train_mean controls',
        statistic='seed losses averaged, synchronized date blocks20/40, simultaneous bands over all reported comparisons',
        source_config_sha256=sha(source/'config.json'))
    write_json(root/'config.json',config)
    (root/'sources').mkdir()
    code=['event_objective_study.py','event_objective_verify.py','event_training.py','event_models.py','event_data.py','event_predict.py',
          'event_uncertainty.py','event_evaluation.py','features.py','compact_features.py','dashboard.py',
          'main.py','event_sources.py','requirements.txt','EVENT-OBJECTIVE-PROTOCOL.md']
    for name in code:shutil.copy2(Path(__file__).parent/name,root/'sources'/name)
    write_json(root/'code_manifest.json',{name:sha(root/'sources'/name) for name in code})
    histories=[];gates=[];training=[];pairs=[];fits=[];data_manifest=[]
    for ticker in stocks:
        for name in [f'{ticker}-daily.npz',f'{ticker}-data-audit.json',f'{ticker}-ohlcv-snapshot.csv']:
            shutil.copy2(source/name,root/name)
        for fold in folds:
            origin=source/ticker/f'fold-{fold}';folder=root/ticker/f'fold-{fold}';folder.mkdir(parents=True)
            shutil.copy2(origin/'prepared.npz',folder/'prepared.npz');shutil.copy2(origin/'audit.json',folder/'audit.json')
            source_checkpoint=next(origin.glob('none-seed*/best.pt'))
            data,split,state,saved=get_data(folder,source_checkpoint,ticker)
            digest=sha(folder/'prepared.npz')
            if digest!=sha(origin/'prepared.npz'):raise ValueError('Prepared data changed while copying')
            data_manifest.append(dict(ticker=ticker,fold=fold,prepared_sha256=digest,original=str(origin/'prepared.npz'),
                preprocessing_checkpoint=str(source_checkpoint),preprocessing_checkpoint_sha256=sha(source_checkpoint),
                counts={key:len(rows) for key,rows in split.items()}))
            torch.save(dict(preprocess=state,features=saved['features'],groups=saved['groups'],source_hash=saved['source_hash']),folder/'preprocess.pt')
            for seed in seeds:
                for mode in args.modes:
                    initial={};parameter_counts={}
                    for task in TASKS:
                        print(f'FIT {ticker} fold={fold} seed={seed} mode={mode} task={task}',flush=True)
                        settings=SimpleNamespace(out=str(root),device=args.device,window=config['window'],width=config['width'],
                            batch_size=config['batch_size'],lr=config['lr'],no_amp=config['no_amp'],epochs=args.epochs,
                            patience=args.patience,training_objective=task,selection_metric='return_mae',log_grad_norm=True,
                            calendar_summary=source_config.get('calendar_summary'))
                        destination=folder/f'{mode}-{task}-seed{seed}'
                        model,record,history,gate_history=fit(data,split,state,settings,mode,seed,destination)
                        context=dict(ticker=ticker,fold=fold,seed=seed,backbone=mode,training_objective=task)
                        training.append(dict(context,**record))
                        histories.extend([{**context,**row} for row in history]);gates.extend([{**context,**row} for row in gate_history])
                        initial[task]=record['initial_weights_sha256'];parameter_counts[task]=record['parameters']
                        fits.append(dict(context,checkpoint=str(destination.relative_to(root)),prepared_sha256=digest,
                                         checkpoint_sha256=sha(destination/'best.pt'),
                                         best_epoch=record['best_epoch'],validation_return_mae=record['val_loss']))
                        del model
                    if initial['return_only']!=initial['multitask'] or len(set(parameter_counts.values()))!=1:
                        raise ValueError('Paired initialization or architecture mismatch')
                    pairs.append(dict(ticker=ticker,fold=fold,seed=seed,backbone=mode,prepared_sha256=digest,
                        initial_weights_sha256=initial['return_only'],equal_initialization=True,parameters=parameter_counts['return_only']))
                    pd.DataFrame(training).to_csv(root/'training.csv',index=False)
                    pd.DataFrame(histories).to_csv(root/'history.csv',index=False)
                    if gates:pd.DataFrame(gates).to_csv(root/'gates.csv',index=False)
                    pd.DataFrame(pairs).to_csv(root/'pair_proof.csv',index=False)
    if len(fits)!=config['planned_fits']:raise ValueError('Incomplete planned fit grid')
    write_json(root/'data_manifest.json',data_manifest)
    write_json(root/'selected_checkpoints.json',fits)
    # Freeze the complete train/validation selection before the first test call.
    write_json(root/'phase1_frozen.json',dict(status='selected',fits=len(fits),paired_initializations=len(pairs),
        selected_checkpoints_sha256=sha(root/'selected_checkpoints.json'),selection_metric=config['selection_metric'],
        data_manifest_sha256=sha(root/'data_manifest.json'),config_sha256=sha(root/'config.json'),
        code_manifest_sha256=sha(root/'code_manifest.json')))
    evaluate(root)


def evaluate(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());config['device']=runtime(config['device'])
    frozen=json.loads((root/'phase1_frozen.json').read_text())
    if frozen['selected_checkpoints_sha256']!=sha(root/'selected_checkpoints.json'):raise ValueError('Selections changed after freeze')
    fits=json.loads((root/'selected_checkpoints.json').read_text());predictions=[];scores=[];secondary=[];val_scores=[]
    from event_evaluation import metrics as auxiliary_metrics
    for item in fits:
        checkpoint=root/item['checkpoint']/'best.pt'
        if sha(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Checkpoint changed after phase1 freeze')
        model,saved=restore(checkpoint,config['device'])
        folder=checkpoint.parents[1];arrays=np.load(folder/'prepared.npz',allow_pickle=False)
        if sha(folder/'prepared.npz')!=item['prepared_sha256']:raise ValueError('Prepared inputs changed')
        data={key:arrays[key] for key in ['x','tokens','ids','mask','y']};data['groups']=saved['groups']
        state=saved['preprocess'];settings=SimpleNamespace(window=saved['window'],batch_size=config['batch_size'],device=config['device'])
        meta=json.loads((root/f'{item["ticker"]}-data-audit.json').read_text());dates=meta['dates']
        context={key:item[key] for key in ['ticker','fold','seed','backbone','training_objective']}
        for split in ['val','test']:
            rows=arrays[split+'_indices'];output,_=infer(model,data,rows,settings,state)
            returns,probability,volatility=decoded(output,state)
            for j,horizon in enumerate([5,10,20]):
                row=dict(context,horizon=horizon,**return_metrics(data['y'][rows,j],returns[:,j]))
                (val_scores if split=='val' else scores).append(row)
                if split=='test':
                    for k,i in enumerate(rows):
                        predictions.append(dict(context,horizon=horizon,date=dates[i],calendar_index=int(i),
                            actual=float(data['y'][i,j]),prediction=float(returns[k,j])))
            if split=='val':
                target=torch.from_numpy(((data['y'][rows]-state['target_mean'])/state['target_std']).astype('float32'))
                positive=torch.from_numpy((data['y'][rows,:3]>0).astype('float32'))
                loss=float(selection_loss(torch.from_numpy(output),target,positive,torch.from_numpy(state['target_std']),'return_mae'))
                if not np.isclose(loss,saved['record']['val_loss'],atol=2e-6,rtol=2e-5):raise ValueError('Validation selection does not replay')
            elif item['training_objective']=='multitask':
                metrics,_=auxiliary_metrics(data['y'][rows],returns,probability,volatility)
                secondary.extend([{**context,**row} for row in metrics])
        del model
    source=Path(config['source']);controls=pd.read_csv(source/'predictions.csv')
    controls=controls[controls.ticker.isin(config['tickers'])&controls.fold.isin(config['folds'])
                      &controls.seed.isin([-1]+config['seeds'])
                      &controls.model.isin(['zero','train_mean','lightgbm_manual'])]
    controls.to_csv(root/'baseline_source_predictions.csv',index=False)
    write_json(root/'baseline_provenance.json',dict(source=str(source/'predictions.csv'),
        source_sha256=sha(source/'predictions.csv'),copied_rows=len(controls),
        snapshot_sha256=sha(root/'baseline_source_predictions.csv'),refitted=False))
    for keys,part in controls.groupby(['ticker','fold','model','seed','horizon']):
        ticker,fold,name,seed,horizon=keys
        context=dict(ticker=ticker,fold=int(fold),seed=int(seed),backbone='baseline',training_objective=name,horizon=int(horizon))
        scores.append(dict(context,**return_metrics(part.actual.to_numpy(),part.prediction.to_numpy())))
        predictions.extend([dict(context,date=row.date,calendar_index=int(row.calendar_index),actual=row.actual,prediction=row.prediction)
                            for row in part.itertuples()])
    pd.DataFrame(predictions).to_csv(root/'predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(root/'metrics.csv',index=False);pd.DataFrame(val_scores).to_csv(root/'validation_metrics.csv',index=False)
    pd.DataFrame(secondary).to_csv(root/'secondary_multitask_metrics.csv',index=False)
    frame=pd.DataFrame(scores);columns=['mae','rmse','bias','direction_accuracy','balanced_direction_accuracy']
    frame.groupby(['ticker','horizon','backbone','training_objective'])[columns].mean().reset_index().to_csv(root/'by_stock_horizon.csv',index=False)
    frame.groupby(['backbone','training_objective'])[columns].mean().reset_index().to_csv(root/'macro_summary.csv',index=False)
    statistics(root)
    write_json(root/'progress.json',dict(status='complete',new_fits=len(fits),paired_initializations=frozen['paired_initializations'],
                                       selection_metric=config['selection_metric'],test_status=config['test_status']))
    from dashboard import build_event_dashboard
    build_event_dashboard(root)
    print(f'COMPLETE {root}',flush=True)


def statistics(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());frame=pd.read_csv(root/'predictions.csv')
    frame['loss']=abs(frame.actual-frame.prediction)
    loss=frame.groupby(['date','ticker','horizon','backbone','training_objective']).loss.mean().reset_index()
    matrices=[];keys=[]
    for mode in config['modes']:
        method=loss[loss.backbone==mode].pivot(index=['date','ticker','horizon'],columns='training_objective',values='loss')
        control=loss[loss.backbone=='baseline'].pivot(index=['date','ticker','horizon'],columns='training_objective',values='loss')
        method=pd.concat([method,control],axis=1)
        comparisons=[('return_only','multitask'),('return_only','zero'),('multitask','zero'),
                     ('return_only','train_mean'),('multitask','train_mean')]
        for treatment,comparator in comparisons:
            wide=(method[comparator]-method[treatment]).unstack(['ticker','horizon']).sort_index()
            if not np.isfinite(wide.to_numpy()).all():raise ValueError('Unequal date coverage')
            matrices.append(np.column_stack([wide.to_numpy(),wide.mean(1).to_numpy()]))
            keys.extend([(mode,treatment,comparator,*column) for column in wide.columns])
            keys.append((mode,treatment,comparator,'ALL','ALL'))
    values=np.column_stack(matrices);rows=[]
    for block in [20,40]:
        bounds=intervals(values,block=block,repeats=config['bootstrap_repeats'])
        for j,(mode,treatment,comparator,ticker,horizon) in enumerate(keys):
            rows.append(dict(backbone=mode,treatment=treatment,comparator=comparator,ticker=ticker,horizon=horizon,
                block_sessions=block,dates=len(values),gain_mae=bounds[0][j],simultaneous_low=bounds[1][j],
                simultaneous_high=bounds[2][j],nominal_low=bounds[3][j],nominal_high=bounds[4][j],family_columns=values.shape[1]))
    pd.DataFrame(rows).to_csv(root/'paired_bands.csv',index=False)
    metric=pd.read_csv(root/'metrics.csv');paired=[]
    for (ticker,horizon,mode),part in metric[metric.backbone!='baseline'].groupby(['ticker','horizon','backbone']):
        table=part.pivot(index=['fold','seed'],columns='training_objective',values='mae')
        gain=table.multitask-table.return_only;byfold=gain.groupby(level='fold').mean();byseed=gain.groupby(level='seed').mean()
        paired.append(dict(ticker=ticker,horizon=horizon,backbone=mode,return_only_mae=float(table.return_only.mean()),
            multitask_mae=float(table.multitask.mean()),gain_mae=float(gain.mean()),folds_won=int((byfold>0).sum()),
            folds=len(byfold),seeds_won=int((byseed>0).sum()),seeds=len(byseed),
            seed_gain_min=float(byseed.min()),seed_gain_max=float(byseed.max())))
    pd.DataFrame(paired).to_csv(root/'paired_summary.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='runs/event-adro-antm-bbri-20261116')
    parser.add_argument('--out',required=True);parser.add_argument('--tickers',nargs='+',default=None)
    parser.add_argument('--modes',nargs='+',choices=list(MODES),default=['none'])
    parser.add_argument('--seeds',nargs='+',type=int,default=None);parser.add_argument('--folds',type=int,default=None)
    parser.add_argument('--epochs',type=int,default=24);parser.add_argument('--patience',type=int,default=6)
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    args=parser.parse_args()
    if min(args.epochs,args.patience,args.bootstrap_repeats)<1:parser.error('Positive epoch/patience/bootstrap budget required')
    run(args)
