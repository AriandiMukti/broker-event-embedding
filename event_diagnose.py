"""Post-fit diagnostics, not feature/model selection or confirmation evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from event_uncertainty import intervals
from event_sources import resolve
from event_predict import restore
from event_training import infer, objective

KEYS=['date','ticker','horizon']
LOSSES={'return_mae':'return_loss','brier':'probability_loss','vol_mae':'volatility_loss'}


def seed_losses(predictions):
    """Average LOSSES, never substitute loss of the seed-ensemble prediction."""
    frame=predictions.copy()
    if frame.duplicated(KEYS+['model','seed']).any():raise ValueError('Duplicate prediction identities')
    for target in ['actual','actual_vol']:
        if (frame.groupby(KEYS)[target].nunique()!=1).any():raise ValueError('Targets differ between models/seeds')
    if not frame.probability.between(0,1).all():raise ValueError('Probability outside [0,1]')
    frame['return_loss']=abs(frame.prediction-frame.actual)
    frame['probability_loss']=(frame.probability-(frame.actual>0))**2
    frame['volatility_loss']=abs(frame.predicted_vol-frame.actual_vol)
    group=frame.groupby(KEYS+['model'],sort=True)
    result=group[list(LOSSES.values())+['prediction','probability','predicted_vol','actual','actual_vol']].mean()
    result=result.rename(columns={'prediction':'mean_prediction','probability':'mean_probability','predicted_vol':'mean_predicted_vol'})
    return result.reset_index()


def paired_bands(losses,repeats=2000):
    matrices=[];identities=[]
    for metric,column in LOSSES.items():
        wide=losses.pivot(index=KEYS,columns='model',values=column)
        for comparator in ['zero','train_mean']:
            methods=[m for m in wide.columns if m!=comparator]
            gains=wide[methods].rsub(wide[comparator],axis=0)
            matrix=gains.stack(future_stack=True).unstack(['ticker','horizon','model']).sort_index()
            if not np.isfinite(matrix.to_numpy()).all():raise ValueError('Unequal coverage; no silent date dropping')
            macro=pd.DataFrame({m:matrix.xs(m,level='model',axis=1).mean(1) for m in methods})
            matrices.append(np.column_stack([matrix.to_numpy(),macro.to_numpy()]))
            identities.extend([(metric,comparator,*key) for key in matrix.columns])
            identities.extend([(metric,comparator,'ALL','ALL',m) for m in methods])
    combined=np.column_stack(matrices);rows=[]
    for block in [20,40]:
        bounds=intervals(combined,block=block,repeats=repeats)
        for i,(metric,comparator,ticker,horizon,model) in enumerate(identities):
            rows.append(dict(metric=metric,comparator=comparator,ticker=ticker,horizon=horizon,model=model,
                block_sessions=block,dates=len(combined),gain=bounds[0][i],simultaneous_low=bounds[1][i],
                simultaneous_high=bounds[2][i],nominal_low=bounds[3][i],nominal_high=bounds[4][i],
                family_columns=combined.shape[1],repeats=repeats))
    return pd.DataFrame(rows)


def stability(metrics):
    rows=[]
    for (ticker,horizon,model),part in metrics.groupby(['ticker','horizon','model']):
        for comparator in ['zero','train_mean']:
            base=metrics[(metrics.ticker==ticker)&(metrics.horizon==horizon)&(metrics.model==comparator)]
            for metric in ['mae','brier','vol_mae']:
                baseline=base.groupby('fold')[metric].mean()
                byfold=part.groupby('fold')[metric].mean()
                paired=part[['seed','fold',metric]].merge(baseline.rename('baseline'),on='fold',validate='many_to_one')
                paired['gain']=paired.baseline-paired[metric]
                seedgain=paired.groupby('seed').gain.mean()
                rows.append(dict(ticker=ticker,horizon=horizon,model=model,comparator=comparator,metric=metric,
                    baseline_loss=baseline.mean(),model_loss=byfold.mean(),gain=baseline.mean()-byfold.mean(),
                    folds_won=int(((baseline-byfold)>0).sum()),folds=len(byfold),
                    seeds_won=int((seedgain>0).sum()),seeds=len(seedgain),
                    seed_gain_min=seedgain.min(),seed_gain_max=seedgain.max()))
    return pd.DataFrame(rows)


def contexts(root,losses):
    config=json.loads((root/'config.json').read_text());window=config['window'];rows=[]
    for ticker,part in losses.groupby('ticker'):
        frame=pd.read_csv(root/f'{ticker}-ohlcv-snapshot.csv',parse_dates=['Date']).set_index('Date')
        cached=np.load(root/f'{ticker}-daily.npz',allow_pickle=False)
        audit=json.loads((root/f'{ticker}-data-audit.json').read_text())
        dates=pd.DatetimeIndex(audit['dates']);logret=np.log(frame.Close).diff().to_numpy()
        paths=[resolve(root/'raw-source/Daily/All',ticker,d) for d in dates]
        formats=[p.suffix if p.exists() else 'missing' for p in paths]
        for row in part[KEYS].drop_duplicates().itertuples(index=False):
            i=dates.get_loc(pd.Timestamp(row.date));h=int(row.horizon);future=logret[i+1:i+h+1]
            coverage=cached['mask'][i-window+1:i+1].any(1)
            target=float(np.log(frame.Close.iloc[i+h]/frame.Close.iloc[i]))
            actual=float(part[(part.date==row.date)&(part.horizon==h)].actual.iloc[0])
            if not np.isclose(target,actual,atol=1e-6):raise ValueError('Cached target does not match frozen prices')
            rows.append(dict(date=row.date,ticker=ticker,horizon=h,quarter=str(pd.Timestamp(row.date).to_period('Q')),
                input_format=formats[i],input_format_changes=len(set(formats[i-window+1:i+1]))>1,
                broker_coverage_fraction=float(coverage.mean()),broker_missing_input_days=int((~coverage).sum()),
                label_end=str(dates[i+h].date()),query_close=float(frame.Close.iloc[i]),
                label_close=float(frame.Close.iloc[i+h]),largest_future_abs_daily_log_return=float(np.max(abs(future))),
                large_move_flag=bool((abs(future)>.1).any())))
    return pd.DataFrame(rows)


def dispersion(predictions):
    rows=[]
    for (ticker,horizon,model,seed),part in predictions.groupby(['ticker','horizon','model','seed']):
        a=part.actual.to_numpy();p=part.prediction.to_numpy()
        sd=float(np.std(a));psd=float(np.std(p))
        corr=float(part.actual.corr(part.prediction,method='spearman')) if sd>0 and psd>1e-12 else np.nan
        rows.append(dict(ticker=ticker,horizon=horizon,model=model,seed=int(seed),dates=part.date.nunique(),
            bias=float(np.mean(p-a)),actual_mean=float(np.mean(a)),prediction_mean=float(np.mean(p)),
            actual_std=sd,prediction_std=psd,dispersion_ratio=psd/sd if sd else np.nan,
            spearman=corr,positive_prevalence=float(np.mean(a>0)),mean_probability=float(part.probability.mean())))
    return pd.DataFrame(rows)


def validation_components(root):
    """Evaluate selected checkpoints on VALIDATION; never refit or tune."""
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    config=json.loads((root/'config.json').read_text());records=[]
    for checkpoint in sorted(root.glob('*/fold-*/*/best.pt')):
        model,saved=restore(checkpoint,device)
        if saved.get('drop') is not None:continue
        arrays=np.load(checkpoint.parents[1]/'prepared.npz',allow_pickle=False)
        data={k:arrays[k] for k in ['x','tokens','ids','mask','y']};data['groups']=saved['groups']
        rows=arrays['val_indices'];state=saved['preprocess']
        args=SimpleNamespace(window=saved['window'],batch_size=config['batch_size'],device=device)
        output,_=infer(model,data,rows,args,state)
        target=((data['y'][rows]-state['target_mean'])/state['target_std']).astype('float32')
        positive=(data['y'][rows,:3]>0).astype('float32')
        returns=float(np.mean((output[:,:3]-target[:,:3])**2))
        vol=float(.25*np.mean((output[:,3:6]-target[:,3:6])**2))
        bce=float(.25*torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(output[:,6:]),torch.from_numpy(positive)))
        total=returns+vol+bce
        if not np.isclose(total,saved['record']['val_loss'],atol=2e-5,rtol=2e-5):raise ValueError('Validation objective does not replay')
        records.append(dict(ticker=checkpoint.parents[2].name,fold=int(checkpoint.parents[1].name.split('-')[1]),
            mode=saved['mode'],seed=saved['seed'],epoch=saved['record']['best_epoch'],n=len(rows),
            return_mse=returns,weighted_logvol_mse=vol,weighted_bce=bce,total_loss=total,
            return_share=returns/total,volatility_share=vol/total,probability_share=bce/total))
    return pd.DataFrame(records)


def distribution_audit(root):
    """Descriptive train/val/test shifts; no feature removal or refitting."""
    root=Path(root);targets=[];inputs=[]
    for folder in sorted(root.glob('*/fold-*')):
        ticker=folder.parent.name;fold=int(folder.name.split('-')[1])
        model,saved=restore(next(folder.glob('none-seed*/best.pt')),'cpu');del model
        state=saved['preprocess'];arrays=np.load(folder/'prepared.npz',allow_pickle=False)
        raw=np.load(root/f'{ticker}-daily.npz',allow_pickle=False)
        for split in ['train','val','test']:
            rows=arrays[split+'_indices'];y=arrays['y'][rows]
            for j,h in enumerate([5,10,20]):
                z=(y[:,j]-state['target_mean'][j])/state['target_std'][j]
                targets.append(dict(ticker=ticker,fold=fold,split=split,horizon=h,n=len(rows),
                    return_mean=float(y[:,j].mean()),return_median=float(np.median(y[:,j])),return_std=float(y[:,j].std()),
                    return_q05=float(np.quantile(y[:,j],.05)),return_q95=float(np.quantile(y[:,j],.95)),
                    positive_prevalence=float((y[:,j]>0).mean()),annualized_vol_mean=float(np.exp(y[:,j+3]).mean()),
                    standardized_return_mean=float(z.mean()),standardized_return_std=float(z.std())))
            days=np.unique(np.concatenate([np.arange(i-saved['window']+1,i+1) for i in rows]))
            observed=arrays['mask'][days];ids=arrays['ids'][days]
            for group,columns in saved['groups'].items():
                columns=[c for c in columns if not state['constant_columns'][c]]
                z=(raw['x'][days][:,columns]-state['mean'][columns])/state['std'][columns] if columns else np.empty((len(days),0))
                inputs.append(dict(ticker=ticker,fold=fold,split=split,group=group,input_dates=len(days),
                    varying_train_columns=len(columns),input_clipping_fraction=float((abs(z)>10).mean()) if z.size else np.nan,
                    standardized_input_mean_abs=float(abs(z.mean(0)).mean()) if z.size else np.nan,
                    unknown_broker_fraction=float(((ids==1)&observed).sum()/max(1,observed.sum())),
                    broker_missing_day_fraction=float((~observed.any(1)).mean())))
    return pd.DataFrame(targets),pd.DataFrame(inputs)


def run(root,out=None,repeats=2000):
    root=Path(root);out=Path(out) if out else root/'diagnostics';out.mkdir(parents=True,exist_ok=False)
    predictions=pd.read_csv(root/'predictions.csv');metrics=pd.read_csv(root/'metrics.csv')
    losses=seed_losses(predictions);losses.to_csv(out/'daily_seed_mean_losses.csv',index=False)
    scorecols=['mae','rmse','accuracy','balanced_accuracy','roc_auc','brier','calibration_ece','vol_mae','vol_rmse']
    summary=metrics.groupby(['ticker','horizon','model'])[scorecols].mean().reset_index()
    summary.to_csv(out/'by_stock_horizon.csv',index=False)
    stable=stability(metrics);stable.to_csv(out/'fold_seed_stability.csv',index=False)
    bands=paired_bands(losses,repeats);bands.to_csv(out/'paired_bands.csv',index=False)
    context=contexts(root,losses)
    regime=predictions[KEYS+['regime']].drop_duplicates()
    daily=losses.merge(context,on=KEYS,validate='many_to_one').merge(regime,on=KEYS,validate='many_to_one')
    for grouping,name in [('quarter','error_by_quarter'),('regime','error_by_regime'),('input_format','error_by_format'),
                           ('input_format_changes','error_by_format_transition'),('broker_missing_input_days','error_by_missing_broker')]:
        table=daily.groupby(['ticker','horizon','model',grouping]).agg(
            dates=('date','nunique'),return_mae=('return_loss','mean'),brier=('probability_loss','mean'),vol_mae=('volatility_loss','mean'))
        table.reset_index().to_csv(out/(name+'.csv'),index=False)
    largest=daily.sort_values('return_loss',ascending=False).groupby(['ticker','horizon','model'],sort=False).head(10)
    largest.to_csv(out/'largest_errors.csv',index=False)
    tail=[]
    for (ticker,horizon),part in daily.groupby(['ticker','horizon']):
        targets=part.drop_duplicates('date').actual.abs();threshold=float(targets.quantile(.95))
        for model,group in part.groupby('model'):
            flag=group.actual.abs()>=threshold
            tail.append(dict(ticker=ticker,horizon=horizon,model=model,threshold=threshold,tail_dates=int(flag.sum()),
                tail_error_share=float(group.loc[flag,'return_loss'].sum()/group.return_loss.sum()),
                tail_mae=float(group.loc[flag,'return_loss'].mean()),non_tail_mae=float(group.loc[~flag,'return_loss'].mean()),
                large_move_labels=int(group.large_move_flag.sum())))
    pd.DataFrame(tail).to_csv(out/'tail_concentration.csv',index=False)
    dispersion(predictions).to_csv(out/'prediction_dispersion.csv',index=False)
    components=validation_components(root);components.to_csv(out/'validation_head_components.csv',index=False)
    target_shift,input_shift=distribution_audit(root)
    target_shift.to_csv(out/'target_distribution.csv',index=False);input_shift.to_csv(out/'input_distribution.csv',index=False)
    cells=[]
    for (ticker,horizon),part in summary.groupby(['ticker','horizon']):
        indexed=part.set_index('model');winner=part.loc[part.mae.idxmin(),'model']
        row=dict(ticker=ticker,horizon=int(horizon),point_minimum_model=winner,minimum_mae=float(indexed.loc[winner,'mae']),
            zero_mae=float(indexed.loc['zero','mae']),mean_mae=float(indexed.loc['train_mean','mae']))
        for comparator in ['zero','train_mean']:
            if winner==comparator:
                row[comparator+'_gain']=0.;row[comparator+'_sim_low']=0.;row[comparator+'_sim_high']=0.
            else:
                p=bands[(bands.ticker==ticker)&(bands.horizon==horizon)&(bands.model==winner)
                        &(bands.metric=='return_mae')&(bands.comparator==comparator)]
                row[comparator+'_gain']=float(p.gain.iloc[0]);row[comparator+'_sim_low']=float(p.simultaneous_low.min())
                row[comparator+'_sim_high']=float(p.simultaneous_high.max())
        row['conditional_positive_vs_both']=row['zero_sim_low']>0 and row['train_mean_sim_low']>0
        cells.append(row)
    cells=pd.DataFrame(cells);cells.to_csv(out/'nine_cells.csv',index=False)
    notes=dict(status='complete',purpose='historical exploratory diagnosis; no model/feature/hyperparameter selection',
        stocks=sorted(predictions.ticker.unique()),stock_horizon_cells=len(cells),family_columns=int(bands.family_columns.iloc[0]),
        bootstrap_blocks=[20,40],bootstrap_repeats=repeats,validation_components_checkpoints=len(components),
        source_format_caution='Format and time are confounded; association is not a format effect.',
        event_flag_caution='Large future price move is a residual audit flag, not a confirmed corporate action or an input feature.')
    (out/'audit.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    text=f'''# Diagnosis per saham dan horizon

Hasil ini eksploratif. BBCA diganti dengan ANTM atas pilihan pengguna; ADRO/BBRI tetap memakai checkpoint dan prediksi yang sama. Tabel minimum point MAE berikut bukan pemilihan model produksi.

{cells.round(6).to_markdown(index=False)}

## Stabilitas dan ketidakpastian

by_stock_horizon.csv berisi seluruh7model pada9kombinasi dan semua metrik. fold_seed_stability.csv menghitung kemenangan pada3fold dan3seed secara terpisah; seed bukan tambahan tanggal independen. paired_bands.csv memakai shared date-block20/40, {repeats}draws, family simultan{notes['family_columns']}kolom meliputi returnMAE/Brier/volMAE terhadap zero/train_mean, seluruh model/saham/horizon serta macro. Interval conditional fittedmodels, tidak mencakup seluruh training/tuning/stock-selection uncertainty, bukan konfirmatori.

## Audit residual

largest_errors.csv menampilkan10error terbesar setiapmodel/saham/horizon, mean seed LOSS dan mean prediction ditandai terpisah, harga query/label/enddate, cakupan inputbroker, perubahan besar di intervaltarget. tail_concentration.csv mengukur porsierror pada5%labelabsolute terbesar; jangan menghapus hari/fitur berdasarkan tabeltest ini. Formatfile bercampur dengan waktu sehingga error_by_format.csv bukan bukti format menyebabkan error. Tabel perquarter/regime/missingbroker dan prediction_dispersion.csv membantu membedakan bias, shrinkageprediksi, dan pergantian kondisi tanpa mengklaim kausalitas.

## Komponen loss validation

{components.groupby('mode')[['return_share','volatility_share','probability_share']].mean().round(4).to_markdown()}

Komponen diukur pada validation checkpoint terpilih, dengan bobot objective asli dan tanpa refit. Dominasi satu komponen tidak membuktikan interferensi multitask; perlu eksperimen train/validation terencana dan konfirmasi baru. Headprobabilitas belum dikalibrasi. Volatilitas adalah annualizedRMS future logreturns. Regime di sini volatilitas saham, bukan rezimpasar luas.

Tidak ada pooling data/bobot antarsaham. Tidak ada fitur dihapus, parameter diubah, atau training arsitektur baru dipicu oleh diagnosis test. Harga/corporateaction/vendorrevisions/timingEOD dan eksekusibacktest masih perlu verifikasi. Rekomendasi berikutnya ditulis setelah menilai seluruh tabel, bukan mengejar seltest yang menang.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8');print(cells.to_string(index=False));print(json.dumps(notes,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--out',default=None)
    p.add_argument('--repeats',type=int,default=2000);a=p.parse_args();run(a.root,a.out,a.repeats)
