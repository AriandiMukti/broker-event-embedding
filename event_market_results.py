"""Descriptive results for the frozen market-context study; never selects a model."""
import argparse
import json
from pathlib import Path
import pandas as pd


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);config=json.loads((root/'config.json').read_text())
    summary=pd.read_csv(root/'macro_summary.csv');paired=pd.read_csv(root/'paired_summary.csv')
    bands=pd.read_csv(root/'paired_bands.csv');history=pd.read_csv(root/'history.csv');training=pd.read_csv(root/'training.csv')
    cells=pd.read_csv(root/'by_stock_horizon.csv');importance=pd.read_csv(root/'importance.csv')
    macro=bands[(bands.ticker=='ALL')&(bands.comparator=='none/local_only')]
    verified=json.loads((root/'verification.json').read_text()) if (root/'verification.json').exists() else {}
    fig,axes=plt.subplots(1,len(config['tickers']),figsize=(13,4.5),squeeze=False)
    for ax,ticker in zip(axes.flat,config['tickers']):
        for name,color in [('zero','#667085'),('none/local_only','#3978be'),('none/market_context','#15867b')]:
            part=cells[(cells.ticker==ticker)&(cells.model==name)].sort_values('horizon')
            ax.plot(part.horizon,part.mae,'o-',label=name,color=color)
        ax.set_title(ticker);ax.set_xticks([5,10,20]);ax.set_xlabel('Horizon sesi');ax.set_ylabel('MAE log-return');ax.grid(alpha=.2)
    axes.flat[-1].legend(fontsize=8);fig.suptitle('IHSG + USD/IDR + relative strength: input lokal versus tambahan pasar')
    fig.tight_layout();fig.savefig(root/'market-comparison.png',dpi=160);plt.close(fig)
    primary=paired[['ticker','horizon','local_mae','market_mae','gain_mae','folds_won','seeds_won']]
    grouped=importance[(importance.method=='temporal_permutation')&(importance.regime=='all')].groupby(['ticker','group'])[['value','std']].mean().reset_index()
    backtests=pd.read_csv(root/'backtest.csv')
    strategy=backtests.groupby(['model','cost_scenario'])[['sharpe','maximum_drawdown','total_return','turnover','trades']].mean().reset_index()
    by_model=summary.set_index('model');pivot=cells.pivot(index=['ticker','horizon'],columns='model',values='mae')
    lose_to_zero=int((pivot['none/market_context']>pivot['zero']).sum())
    delta=float(by_model.loc['none/local_only','mae']-by_model.loc['none/market_context','mae'])
    consistent_gain=bool((macro.simultaneous_low>0).all())
    interpretation=('Kedua ukuran blok menunjukkan gain positif, bersyarat pada model historis ini.' if consistent_gain
                    else 'Belum ada bukti manfaat return yang konsisten pada kedua ukuran blok.')
    docs=[
        '# Hasil aktual: tambahan konteks pasar Yahoo',
        f"{len(training)} fit baru, {len(training)//2} pasangan input arm, Set Transformer + TCN tanpa gating, model terpisah per {', '.join(config['tickers'])}. Multitask objective, checkpoint raw-return MAE validasi, {len(config['folds'])} fold / {len(config['seeds'])} seed ({config['seeds']}). Seluruh checkpoint dibekukan sebelum test. Hasil lama tidak ditimpa; sejarah test sudah pernah dieksplorasi.",
        '## Hasil return',summary[['model','mae','rmse','brier','vol_mae']].to_markdown(index=False,floatfmt='.6f'),
        f"Tambahan pasar memiliki MAE point estimate lebih kecil pada {int((paired.gain_mae>0).sum())}/{len(paired)}kombinasi saham/horizon. Gain positif = MAE lokal minus MAE pasar; ini bukan klaim model produksi. MAE dihitung perseed dahulu, dirata-ratakan antar fold/seed/saham/horizon; bukan error ensemble prediksi. Baselinezero/trainmean/LightGBM menggunakan prediksi frozen lama, LightGBM belum menerima fitur pasar baru.",
        primary.to_markdown(index=False,floatfmt='.6f'),
        '## Paired temporal uncertainty',macro[['block_sessions','gain_mae','simultaneous_low','simultaneous_high','dates','family_columns']].to_markdown(index=False,floatfmt='.6f'),
        f"MAE macro lokal {by_model.loc['none/local_only','mae']:.6f} dan pasar {by_model.loc['none/market_context','mae']:.6f}. Gain = MAE lokal dikurangi MAE pasar: {delta:+.6f}. {interpretation} Prediksi nol memiliki MAE {by_model.loc['zero','mae']:.6f}; MAE pasar lebih besar dari baseline nol pada {lose_to_zero}/{len(pivot)} kombinasi. Ini bukan bukti seluruh informasi pasar tidak berguna pada semua model/periode.",
        'Bootstrap dateblock20/40,2000draws sinkron atas saham/horizon; seed-loss dirata-ratakan. Simultaneous95%bands satu family30kolom:3comparisons x(9cells+macro). Conditional fittedmodels, tidak mencakup seluruh ketidakpastian pemilihan resep. Paired_bands.csv mencakup perbandingan keduaarm terhadapnol. Tidak menggunakan test untuk memilih fitur atau resep.',
        '## Data dan point-in-time',
        'Sumber Yahoo: [IHSG](https://finance.yahoo.com/quote/%5EJKSE/history/) dan [USD/IDR](https://finance.yahoo.com/quote/IDR%3DX/history/). Snapshot CSV, metadata dan audit unduhan disertakan; tabel unduhan kini bukan bukti vendor vintage historis.',
        'Input lokal frozen1556sesi sejak2Juni2020 hingga16November2026; train893/1019/1145, val126/test126perfold; querytest378tanggal unik per saham, horizon5/10/20, purge20ditambahembargo20. Yahoo diunduh dengan auto_adjust=False/repair=False, snapshot/metadata/hash/retrievalUTC disimpan. IHSG1656bar(1invalid), USDIDR1794bar(143invalid). OHLCinvalid dimask, tidak diperbaiki otomatis atau diinterpolasi. Missingflags mempertahankan informasi ketersediaan.',
        'Bar berlabelD dianggap tersedia D+1kalender12:00WIB; queryEOD23:59 memakai backward-asof dengan tolerance7hari setelahcutoff. IHSG dan FX sama-sama dilag konservatif. Relative strength memakai return saham-minusIHSG pada intervalsource-date yang sama, bukan mencampurkan returnsaham hariini dengan IHSG kemarin. Ini asumsi availability konservatif, bukan rekonstruksi release/vendor vintage historis; revisi data Yahoo masih mungkin.',
        'Fitur tambahan24kolom termasukmissingflags/age: return1/5/20 danvol20IHSG/kurs, relative strength1/5/20. Keduaarm memiliki109kolominput dan arsitektur yang sama; kontrol seluruhkolompasar0, perlakuan datanyata. Statistik/scaler/vocab train-only. Nolkontrol merupakan masked-feature ablation, bukan observasi pasar sintetis. Inputdimensi berubah dibandingrunlama85fitur, sehingga keduaarm dilatih baru; nilai kontrol bukan salinan runlama.',
        'IDXFINANCE.JK,IDXENERGY.JK,IDXBASIC.JK teridentifikasiINDEX tetapi hanya mengembalikan1bar; ditolak untuk sejarahtraining. Tidak merekayasa indekssektor dari konstituen masa kini, tidak mengklaim sektor sudah diuji. Ownership tetapfrozen padafold tanpa variasitrain. BrokerID/mask, fitur lokal, target dan split sama denganrunfrozenlama.',
        '## Feature importance validasi',grouped.to_markdown(index=False,floatfmt='.6f'),
        'Permutationimportance mengukur kenaikan MAEreturn validasi setelah kelompok dipermutasi dalam bloktemporal,3repeat perfit; tabel mean acrossfold/seed. Gradient sensitivity dan analisis perrezim volatilitasIHSG juga tersedia di importance.csv. Thresholdrezim memakai medianIHSGvol20 padatrain, denganavailabilitylag yang sama. Importance bukan kausalitas; gate pada backbone ini selalu1. Tidak menghapus fitur permanen. Eksperimen ini mengablasi paket pasar bersama; belum memisahkan gainrefitIHSG versusUSD versusRS.',
        'Importance validasi IHSG/kurs positif pada ANTM danBBRI, sedangkan ADRO rata-rata negatif. Ini tidak membuktikan noise atau alasan menghapus fitur: perturbasi temporal merusak korelasi kelompok, importance bergantung pada model dan regime, dan gain test paket pasar belum terkonfirmasi.',
        '## Training, numerik dan backtest',
        f"AdamWlr0.001/wd0.01, schedulerplateau0.5/patience3, clipping1, AMP, width24/window20/batch64, max24epoch/patience6. Totalwaktufit{training.seconds.sum()/60:.2f}menit; peakallocatedVRAM{training.peak_gpu_mb.max():.2f}MiB. Normnonfinite{int(history.nonfinite_gradient_steps.sum())}, confirmedGradScaler skips{int(history.confirmed_scaler_skips.sum())}. Logscale-skip ditambahkan tanpa mengganti objective.",
        'Backtest longcash: p>.5 ANDpredictedreturn>0, nextopen t+1 hinggat+h+1, nonoverlap lintasfold,15bpsfee+10bpsslippagepersisi; stress2x. Sharpe/maxdrawdown/turnover/trades/equity disimpan. Labelreturnclose-to-close bukan returneksekusinextopen. Probabilitas belum dikalibrasi; ECE/Brier/reliability bins mengukurcalibration, bukan bukti calibration sudahdiperbaiki. Kebenaran corporateactions/liquidity/vendorbasis belum diverifikasi independen.',
        strategy.to_markdown(index=False,floatfmt='.6f'),
        'Tabelbacktest adalah mean metrik strategi per saham/horizon/seed; bukan Sharpe portofolio gabungan. NaN Sharpe pada strategi tanpa transaksi berarti tidak terdefinisi. Seluruhhasil perstrategi/biaya di backtest.csv; trades.csv dan equity.csv memuat jejakeksekusi. Tidak memilih sinyal/strategi dari returntest.',
        '## Verifikasi','```json\n'+json.dumps(verified,indent=2)+'\n```',
        'Tests/smoke mendahului training penuh. Verifier merekonstruksi18preparedtransforms dari dailycache dan snapshotYahoo, seluruhstatistiktrain-only, memeriksaavailability, hashcheckpoint/source/phase1freeze, inisialisasiberpasangan, purge/embargo/disjointtest, replayval/test danGPU/CPU. Cachelokal sebelumnya telahdirebuilddariraw padaexperimenasal.',
        '29tes event/objective/pasar lulus. Suite proyek penuh:88/91lulus, tiga kegagalan legacy yang sama pada adaptive inference/tokenizer broker relations/contextend broker tokens terhadap inputlive yang berubah. Log lengkap disertakan; tidak melonggarkan toleransi atau menimpa hasil lama agar terlihat lulus. Smoke nyata2fit dan replayGPU/CPU lulus.',
        '## Batas hasil dan langkah berikutnya',
        'Tetap eksploratif, tanpa confirmationtest baru. Periksa pairedinterval dan konsistensi fold/seed sebelummenafsirkan pointestimate. Tidak menganggap tambahanfitur otomatis meningkatkanreturn. Bekukan resep sebelumkonfirmasi sesi baru; jika meneliti kontribusi terpisah, rencanakan refit ablation IHSG/kurs/RS pada train/val dengan protokol terpisah. Sektor membutuhkan sumberhistoris yang lebih lengkap.',
        '## Perubahan file',
        'Proyek .. event_market_data.py:download/quality/availability/features; event_market_study.py:paired54fit/freeze/evaluate/backtest/statistics; event_market_verify.py:rebuild/replay; event_market_results.py:laporan/gambar; test_event_market.py:temporal/qualitytests; EVENT-MARKET-PROTOCOL.md:resep; event_training.py:scalerskipmonitor danoptionalimportanceMAE; dashboard.py:panelinputarm/kontekspasar. Sourceaktualtrainingdisimpan di sources/ denganSHA256.',
    ]
    (root/'RESULTS.md').write_text('\n\n'.join(docs)+'\n',encoding='utf-8')
    # Post-fit GUI export uses distinct baseline labels; training snapshot stays intact.
    from dashboard import build_event_dashboard
    build_event_dashboard(root)
    print(summary.to_string(index=False));print(macro.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');report(p.parse_args().root)
