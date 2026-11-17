"""Paired, multiplicity-aware evaluation of purged return-transfer tests."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from embedding_returns import HORIZONS,CONTROLS


def bootstrap(values,block,repeats=20000,seed=78941):
    """One synchronized date sequence, allowing blocks across fold boundaries.
    This retains nearby overlapping h=5/10 test-target dependence.
    """
    rng=np.random.default_rng(seed);n=len(values);means=np.nanmean(values,axis=0)
    draws=np.empty((repeats,values.shape[1]))
    for start in range(0,repeats,128):
        count=min(128,repeats-start)
        offsets=rng.integers(0,n,(count,int(np.ceil(n/block))))
        ids=((offsets[:,:,None]+np.arange(block))%n).reshape(count,-1)[:,:n]
        draws[start:start+count]=np.nanmean(values[ids],axis=1)
    std=draws.std(0,ddof=1);safe=np.where(std>1e-12,std,1.)
    critical=float(np.quantile(np.abs((draws-means)/safe).max(1),.95))
    return dict(mean=means,low=means-critical*std,high=means+critical*std,
                nominal_low=np.quantile(draws,.025,axis=0),nominal_high=np.quantile(draws,.975,axis=0),
                std=std,critical=critical)


def classify(rows,fold_gain):
    if set(zip(rows.comparator,rows.block_sessions))!={(c,b) for c in CONTROLS for b in [10,15,20]}:return 'belum'
    consistent=all((fold_gain[c]>0).sum()>=2 for c in CONTROLS)
    if consistent and (rows.sim_ci_low>0).all():return 'didukung'
    if consistent and (rows[rows.block_sessions==10].nominal_ci_low>0).all():return 'indikasi'
    return 'belum'


def illustrative_trades(folder,horizon,method,buy_fee=.0015,sell_fee=.0025,slippage=.0005):
    """Long/flat illustration only: signal after close t, entry Open[t+1],
    exit Close[t+h], one position at a time. No actual trading action.
    """
    folder=Path(folder);data=np.load(folder/'source_snapshot.npz',allow_pickle=False)
    table=pd.read_csv(folder/'test_predictions.csv')
    forecasts=table[(table.horizon==horizon)&(table.method==method)].groupby('feature_index').prediction.mean()
    last_exit=-1;trades=[];cost=buy_fee+sell_fee+2*slippage
    for t,p in forecasts.items():
        t=int(t)
        if t<last_exit or p<=np.log1p(cost):continue
        entry=t+1;exit_day=t+horizon
        if not data['good'][entry:exit_day+1].all():continue
        opening=float(data['open'][entry]);closing=float(data['close'][exit_day])
        gross=closing/opening-1
        # Assumed scenario: buy fee15bp, sell25bp, slippage5bp each side.
        net=(closing/opening)*(1-buy_fee-slippage)*(1-sell_fee-slippage)-1
        trades.append(dict(feature_index=t,entry_index=entry,exit_index=exit_day,
            signal_date=str(data['dates'][t]),entry_date=str(data['dates'][entry]),exit_date=str(data['dates'][exit_day]),
            predicted_log_return=float(p),entry_open=opening,exit_close=closing,gross_return=gross,net_return=net))
        last_exit=exit_day
    compound=float(np.prod([1+r['net_return'] for r in trades])-1) if trades else 0.
    return trades,dict(method=method,horizon=horizon,trades=len(trades),net_compound_return=compound,
                       win_rate=float(np.mean([t['net_return']>0 for t in trades])) if trades else None)


def analyze(root,repeats=20000):
    root=Path(root);plan=json.loads((root/'experiment_plan.json').read_text());stocks=plan['tickers']
    series=[];fold_gains={};daily_all=[];summaries=[]
    for ticker in stocks:
        daily=pd.read_csv(root/ticker/'daily_test_scores.csv');daily_all.append(daily)
        summaries.append(pd.read_csv(root/ticker/'summary.csv'))
        for horizon,group in daily.groupby('horizon'):
            pivot=group.pivot(index=['fold','date'],columns='method',values='absolute_error')
            differences=pd.DataFrame({c:pivot[c]-pivot['learned_embedding'] for c in CONTROLS})
            fold_gains[ticker,horizon]=differences.groupby(level='fold').mean()
            for c in CONTROLS:
                s=differences[c].reset_index(level='fold',drop=True)
                s.name=(ticker,int(horizon),c);series.append(s)
    frame=pd.concat(series,axis=1).sort_index();intervals=[];critical={}
    for block in [10,15,20]:
        result=bootstrap(frame.to_numpy(),block,repeats)
        critical[str(block)]=result['critical']
        for i,(ticker,horizon,c) in enumerate(frame.columns):
            intervals.append(dict(ticker=ticker,horizon=horizon,comparator=c,block_sessions=block,
                gain_mae=result['mean'][i],sim_ci_low=result['low'][i],sim_ci_high=result['high'][i],
                nominal_ci_low=result['nominal_low'][i],nominal_ci_high=result['nominal_high'][i],dates=int(frame.iloc[:,i].notna().sum())))
    intervals=pd.DataFrame(intervals);intervals.to_csv(root/'paired_intervals.csv',index=False)
    summary=pd.concat(summaries,ignore_index=True);summary.to_csv(root/'summary.csv',index=False)
    per_fold=pd.concat(daily_all).groupby(['ticker','fold','horizon','method'],as_index=False)[['absolute_error','squared_error','direction_correct']].mean()
    per_fold.to_csv(root/'summary_by_fold.csv',index=False)
    overview=[]
    for ticker in stocks:
        for h in HORIZONS:
            rows=intervals[(intervals.ticker==ticker)&(intervals.horizon==h)]
            scores=summary[(summary.ticker==ticker)&(summary.horizon==h)].set_index('method')
            status=classify(rows,fold_gains[ticker,h])
            row=dict(ticker=ticker,horizon=h,status=status,test_dates=int(rows.dates.min()),
                embedding_mae_bp=10000*scores.loc['learned_embedding','absolute_error'],
                ohlcv_mae_bp=10000*scores.loc['ohlcv','absolute_error'],
                direct_mae_bp=10000*scores.loc['broker_direct','absolute_error'],
                onehot_mae_bp=10000*scores.loc['broker_onehot','absolute_error'],
                random_mae_bp=10000*scores.loc['random_embedding','absolute_error'],
                zero_mae_bp=10000*scores.loc['zero','absolute_error'],
                median_mae_bp=10000*scores.loc['training_median','absolute_error'],
                embedding_direction_accuracy=scores.loc['learned_embedding','direction_correct'],
                ohlcv_direction_accuracy=scores.loc['ohlcv','direction_correct'])
            for c in CONTROLS:
                r=rows[(rows.comparator==c)&(rows.block_sessions==10)].iloc[0]
                row[f'gain_{c}_bp']=r.gain_mae*10000
                row[f'sim_low_{c}_bp']=r.sim_ci_low*10000
                row[f'sim_high_{c}_bp']=r.sim_ci_high*10000
                row[f'positive_folds_{c}']=int((fold_gains[ticker,h][c]>0).sum())
            overview.append(row)
    overview=pd.DataFrame(overview);overview.to_csv(root/'relevance_summary.csv',index=False)
    config=dict(bootstrap_repeats=repeats,blocks=[10,15,20],simultaneous_comparisons=len(frame.columns),
        critical_values=critical,confidence=.95,bootstrap='synchronized circular date blocks across all stocks/horizons/controls, not split at fold boundaries',
        interpretation='approximate conditional intervals for already fitted models; not all training/model-selection uncertainty',
        metric='MAE of log-return prediction; 1 bp = .0001 log-return error',
        score_aggregation='loss averaged across five seeds, then dates; not MAE of averaged predictions')
    (root/'analysis_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    qualified=overview[overview.status=='didukung'][['ticker','horizon']].to_dict('records')
    decisions=dict(qualified_cases=qualified,executed=bool(qualified),reason='Only strict simultaneous, all-control criterion qualifies.',
        cost_scenario='Assumed buy fee15bp, sell fee25bp, slippage5bp each side; approximately50bp roundtrip, not verified actual broker fees',
        rule='ensemble-mean signal after close t > log(1.005), long at next Open, exit Close t+h, no overlapping positions',
        warning='Any triggered simulation is post-selection on the same historical test and not independent confirmation; raw price only, dividends omitted')
    if qualified:
        trade_rows=[];trade_scores=[]
        for case in qualified:
            for method in ['learned_embedding','ohlcv','broker_direct']:
                trades,metrics=illustrative_trades(root/case['ticker'],case['horizon'],method)
                trade_rows += [dict(ticker=case['ticker'],horizon=case['horizon'],method=method,**t) for t in trades]
                trade_scores.append(dict(ticker=case['ticker'],**metrics))
        pd.DataFrame(trade_rows).to_csv(root/'illustrative_trades.csv',index=False)
        pd.DataFrame(trade_scores).to_csv(root/'illustrative_trade_summary.csv',index=False)
    else:
        for name in ['illustrative_trades.csv','illustrative_trade_summary.csv']:
            (root/name).unlink(missing_ok=True)
    (root/'backtest_decision.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    (root/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    for ticker in stocks:stock_page(root,ticker,overview[overview.ticker==ticker])
    report(root,overview,config,decisions)
    index_page(root,overview)
    chart(root,overview)
    print(overview[['ticker','horizon','status','embedding_mae_bp','ohlcv_mae_bp','direct_mae_bp','random_mae_bp','zero_mae_bp']].to_string(index=False),flush=True)


def stock_page(root,ticker,overview):
    folder=root/ticker;pred=pd.read_csv(folder/'test_predictions.csv');parts=[]
    for h in HORIZONS:
        figure=go.Figure();group=pred[pred.horizon==h]
        actual=group.drop_duplicates('date').sort_values('date')
        figure.add_trace(go.Scatter(x=actual.date,y=actual.actual*10000,name='Return aktual',line=dict(color='#17233b',width=2)))
        for method,label,color,seed in [('ohlcv','OHLCV','#94a3b8',-1),('broker_direct','Broker langsung','#e49a29',-1),
                                        ('learned_embedding','Embedding seed 11','#087f8c',11),('random_embedding','Acak seed 11','#9f7aea',11)]:
            rows=group[(group.method==method)&(group.seed==seed)].sort_values('date')
            figure.add_trace(go.Scatter(x=rows.date,y=rows.prediction*10000,name=label,line=dict(color=color,width=1.6),
                                       visible='legendonly' if method=='random_embedding' else True))
        figure.update_layout(template='plotly_white',height=410,title=f'{ticker}: return {h} sesi ke depan',
            yaxis_title='Log return (bp)',xaxis_title='Tanggal fitur / asal forecast',legend=dict(orientation='h',y=1.12),
            margin=dict(l=60,r=20,t=85,b=50))
        parts.append(figure.to_html(full_html=False,include_plotlyjs='../plotly.min.js' if not parts else False,
                                   config={'displaylogo':False,'responsive':True}))
    cols=['horizon','status','embedding_mae_bp','ohlcv_mae_bp','direct_mae_bp','random_mae_bp','zero_mae_bp']
    text=f'<!doctype html><html lang="id"><meta charset="utf-8"><title>{ticker} — uji return</title><style>body{{font:15px system-ui;color:#17233b;margin:24px}}p{{line-height:1.6}}table{{border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #ddd}}</style><h1>{ticker}: apakah embedding membantu prediksi return?</h1>'
    text+='<p>Grafik menampilkan seed 11 yang ditentukan sebelumnya. Tabel skor merata-ratakan loss seluruh lima seed. Grafik bukan rentang keyakinan atau prediksi live.</p>'
    text+=overview[cols].to_html(index=False,float_format=lambda v:f'{v:.2f}',border=0)+''.join(parts)
    text+='<p>Fitur tersedia setelah penutupan; return target adalah close-to-close tanpa penyesuaian dividen. Peningkatan probabilitas state broker dari eksperimen sebelumnya tidak otomatis menghasilkan prediksi harga.</p><a href="../index.html">Kembali ke perbandingan</a></html>'
    (folder/'index.html').write_text(text,encoding='utf-8')


def index_page(root,overview):
    rows=[]
    for r in overview.to_dict('records'):
        rows.append(f"<tr><td><a href='{r['ticker']}/index.html'>{r['ticker']}</a></td><td>{r['horizon']}</td><td>{r['status']}</td><td>{r['embedding_mae_bp']:.2f}</td><td>{r['ohlcv_mae_bp']:.2f}</td><td>{r['direct_mae_bp']:.2f}</td><td>{r['random_mae_bp']:.2f}</td><td>{r['zero_mae_bp']:.2f}</td></tr>")
    names=html.escape(', '.join(overview.ticker.unique()))
    text='<!doctype html><html lang="id"><meta charset="utf-8"><title>Uji embedding untuk return saham</title><style>body{font:15px system-ui;margin:30px;color:#17233b}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd}p{line-height:1.6;max-width:1000px}</style><h1>Embedding broker → prediksi return?</h1><p>Model terpisah untuk '+names+'. Klik saham untuk membuka grafik. MAE dalam basis poin log-return; lebih kecil lebih baik. Skor embedding/acak adalah rata-rata loss lima seed.</p><table><thead><tr><th>Saham</th><th>Horizon</th><th>Status</th><th>Embedding</th><th>OHLCV</th><th>Broker langsung</th><th>Embedding acak</th><th>Prediksi nol</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table><p>Status didukung memerlukan perbaikan MAE terhadap seluruh tujuh kontrol, interval simultan positif pada blok 10/15/20 sesi, dan minimal dua fold positif. Indikasi memakai interval nominal blok10. Belum berarti bukti belum cukup, bukan membuktikan tidak ada hubungan.</p><p>Pengujian historis eksploratif; seleksi saham dan sejarah telah dilihat pada pengembangan sebelumnya. Interval bersyarat pada model yang sudah dilatih. Tidak ada rekomendasi transaksi.</p><a href="RESULTS.md">Laporan lengkap</a> · <a href="relevance_summary.csv">Angka lengkap</a></html>'
    (root/'index.html').write_text(text,encoding='utf-8')


def chart(root,overview):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,6.8),sharey=True)
    colors={'didukung':'#087f8c','indikasi':'#dda126','belum':'#8391a5'}
    for ax,control,title in zip(axes,['ohlcv','broker_direct'],['Dibanding OHLCV saja','Dibanding broker langsung']):
        for i,r in enumerate(overview.to_dict('records')):
            value=r[f'gain_{control}_bp'];low=r[f'sim_low_{control}_bp'];high=r[f'sim_high_{control}_bp']
            ax.errorbar(value,i,xerr=[[value-low],[high-value]],fmt='o',color=colors[r['status']],capsize=4,linewidth=1.5)
        ax.set_yticks(range(len(overview)),[f'{r.ticker} · {r.horizon} sesi' for r in overview.itertuples()])
        ax.axvline(0,color='#b44757',linestyle='--',linewidth=1)
        ax.grid(axis='x',alpha=.2);ax.set_title(title,fontweight='bold',pad=16)
        ax.set_xlabel('Pengurangan MAE (bp log-return)\nPositif = embedding lebih baik')
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Apakah embedding broker membantu memprediksi harga?',fontsize=16,fontweight='bold',y=.97)
    date_text=str(int(overview.test_dates.iloc[0])) if overview.test_dates.nunique()==1 else f'{int(overview.test_dates.min())}–{int(overview.test_dates.max())}'
    fig.text(.5,.025,f'{overview.ticker.nunique()} saham × 3 horizon × 7 kontrol | Interval simultan 95%, blok 10 sesi\n{date_text} tanggal test per kasus; target 5/10 sesi saling tumpang tindih. Historis eksploratif, bukan konfirmasi baru.',ha='center',fontsize=9,color='#536170')
    fig.tight_layout(rect=[.01,.11,.99,.91],w_pad=3)
    fig.savefig(root/'return_transfer_evidence.png',dpi=160,facecolor='white');plt.close(fig)


def report(root,overview,config,decision):
    date_text=str(int(overview.test_dates.iloc[0])) if overview.test_dates.nunique()==1 else f'{int(overview.test_dates.min())}–{int(overview.test_dates.max())}'
    sources=[json.loads((root/t/'config.json').read_text()) for t in overview.ticker.unique()]
    source_text='; '.join(f"{s['ticker']}: {s['source_days']} sesi, {s['source_start']} s.d. {s['source_end']}" for s in sources)
    lines=['# Apakah embedding broker menambah kemampuan prediksi return?','',
        f"Hasil: {int((overview.status=='didukung').sum())} dari {len(overview)} kombinasi saham/horizon lolos kriteria ketat; {int((overview.status=='indikasi').sum())} indikasi nominal.",'',
        '| Saham | Horizon sesi | Status | MAE embedding (bp) | MAE OHLCV | MAE broker langsung | MAE acak | MAE nol |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in overview.to_dict('records'):
        lines.append(f"| {r['ticker']} | {r['horizon']} | {r['status']} | {r['embedding_mae_bp']:.2f} | {r['ohlcv_mae_bp']:.2f} | {r['direct_mae_bp']:.2f} | {r['random_mae_bp']:.2f} | {r['zero_mae_bp']:.2f} |")
    lines += ['', 'MAE adalah absolute error log-return, dikali 10.000 untuk satuan bp. Lebih kecil lebih baik. Target adalah raw close-to-close, bukan total return yang sudah memasukkan dividen. Target nol berarti memprediksi tidak ada perubahan harga, benchmark penting pada tugas return.', '',
        '## Desain yang dijaga', '',
        '- Saham terpisah: BRPT/MDKA dipilih dari uji state broker terdahulu, ADRO pembanding. Ini seleksi historis, bukan konfirmasi baru.',
        f'- Horizon 1/5/10 menggunakan tanggal fitur test yang sama: {date_text} tanggal dalam tiga expanding folds. Sembilan sesi fitur paling akhir tidak dipakai agar seluruh horizon memiliki label dan sampel yang sama.',
        '- Train t+10 harus sebelum awal validasi; validasi t+10 sebelum awal test. Sampel celah dibuang. Label test 5/10 sesi tetap dapat saling tumpang tindih dan tidak dianggap observasi independen.',
        '- Embedding lama tidak digunakan. Tiap fold melatih ulang masked broker-only encoder dari data di dalam prefix training harga. Tokenizer memakai awal 80% prefix itu; sisa prefix menjadi validasi state broker. Tidak ada target return untuk melatih encoder.',
        '- Encoder dibekukan. Head prediksi semua varian adalah Ridge, agar perbandingan mengisolasi representasi. Alpha dan lookback broker 1, 1+5, atau 1+5+20 dipilih melalui validasi return per saham/fold/horizon/method/seed, tanpa memilih berdasarkan test. Head tidak di-refit pada validasi.',
        '- OHLCV: return1/5/20/40, open gap, range, body, log volume, volume relatif20, volatilitas20. Empat agregat broker (Retail sesuai sumber, buyer fraction, log rasio jumlah buyer/seller, total abs net value) diberikan ke seluruh varian broker, termasuk kontrol agregat saja.',
        '- Broker langsung memuat net shares per broker DAN one-hot state identik dengan input encoder. One-hot saja memisahkan efek kuantisasi. Random embedding memakai encoder awal seed yang sama dan dimensi sama. Mean token encoder ditransformasi linear ke 8D, tanpa gradien return.',
        '- Preprocessor impute median, indikator missing dan standardisasi hanya di-fit pada train. Batas token, broker, input dan head per saham tidak digabung.',
        '- Skor learned/random adalah rata-rata loss lima seed sebelum merata-ratakan tanggal; bukan loss dari prediksi ensemble. Grafik hanya seed11 yang ditetapkan sebelumnya.', '',
        '## Ketidakpastian dan keputusan', '',
        f"{config['simultaneous_comparisons']} perbandingan serentak: saham × horizon × tujuh kontrol. Bootstrap {config['bootstrap_repeats']:,} blok tanggal tersinkronisasi, panjang10/15/20, menyeberangi batas fold untuk mempertahankan kedekatan target test yang overlap. Interval bersyarat pada model yang sudah di-fit; belum mencakup seluruh ketidakpastian training, tuning dan seleksi saham.",
        'Didukung memerlukan seluruh batas bawah interval simultan >0 di ketiga panjang blok dan >=2 fold positif untuk setiap kontrol. Indikasi memerlukan seluruh interval nominal blok10 >0 dan konsistensi fold yang sama. Belum berarti manfaat belum terbukti terhadap seluruh kontrol; bukan bukti tidak ada pola.',
        'Tabel paired_intervals.csv memuat gain MAE berpasangan dan interval. summary.csv menyertakan RMSE dan akurasi arah (sign tiga kelas termasuk nol); keduanya sekunder, bukan dasar pemilihan model.', '',
        '## Backtest transaksi', '',
        ('Tidak dijalankan: belum ada kasus yang lolos kriteria return ketat. Tidak ada klaim profit atau model siap trading.' if not decision['executed'] else
         'Ada ilustrasi transaksi untuk kasus yang lolos. Ini seleksi pada test historis yang sama, bukan konfirmasi independen. Lihat illustrative_trade_summary.csv dan illustrative_trades.csv. Sinyal ensemble setelah close t, entry Open t+1, exit Close t+h, satu posisi, threshold log(1.005). Asumsi fee beli15bp/jual25bp dan slippage5bp per sisi; biaya aktual belum diverifikasi. Strategi raw-price tanpa dividen.'), '',
        '## Batas interpretasi', '',
        'Data sumber: '+source_text+'. Return raw dapat terpengaruh ex-dividend/corporate action; tidak ada penyesuaian yang direka. Tidak ada data baru yang belum dilihat, beberapa tahun data, prediksi live, atau pengujian ekonomi independen.',
        'Kemenangan pada state broker tidak otomatis menghasilkan kemenangan pada return. Encoder mean-pooling 8D adalah satu cara transfer yang diuji; hasil ini tidak membuktikan semua arsitektur embedding tidak berguna.', '',
        '## Menjalankan', '', '```powershell','python embedding_returns.py','python return_results.py','python audit_return_transfer.py','python -m unittest test_embedding_returns -v','```','',
        'Buka index.html atau halaman saham untuk grafik. selected_heads.csv menyimpan pilihan fitur/lookback/alpha yang dapat berbeda antar saham dan horizon. sample_audit.csv dan embedding_training.json menyimpan batas waktu training; checkpoint encoder dan head per fold tersedia untuk reproduksi.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='runs/embedding-returns-v1')
    parser.add_argument('--bootstrap-repeats',type=int,default=20000)
    args=parser.parse_args()
    if args.bootstrap_repeats<1000:parser.error('Use >=1000 bootstrap replicates')
    analyze(args.out,args.bootstrap_repeats)
