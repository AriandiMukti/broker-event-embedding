"""Descriptive report of frozen same-session masked association results."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from event_association_study import build_dashboard
from event_relation_study import load_arrays
from event_market_study import load_raw
from event_broker_geometry import dense_selected


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);config=json.loads((root/'config.json').read_text())
    # Descriptive source mechanics on train/validation only, never pair selection.
    balance=[]
    for ticker in config['tickers']:
        raw=load_raw(root/'encoder-source',ticker)
        logs=np.where(raw['mask'],raw['tokens'][...,0],0).astype('float64')
        net=np.sign(logs)*np.expm1(abs(logs));total_abs=abs(net).sum(-1).clip(1)
        imbalance=abs(net.sum(-1))/total_abs
        for fold in config['folds']:
            data=load_arrays(root/ticker/f'fold-{fold}'/'dataset.npz')
            local,_=dense_selected(raw,data['codes']);v=local[...,0].astype('float64')
            coverage=np.expm1(abs(v)).sum(-1)/total_abs
            for split in ['train','val']:
                q=data['queries'][data['splits']==split]
                balance.append(dict(ticker=ticker,fold=fold,split=split,dates=len(q),
                    mean_selected_abs_net_coverage=float(coverage[q].mean()),
                    median_full_snapshot_abs_imbalance=float(np.median(imbalance[q])),
                    fraction_full_snapshot_balanced_1e4=float((imbalance[q]<1e-4).mean())))
    balance=pd.DataFrame(balance);balance.to_csv(root/'flow_balance_audit.csv',index=False)
    table=pd.read_csv(root/'comparison.csv');macro=pd.read_csv(root/'macro_summary.csv')
    bounds=pd.read_csv(root/'paired_intervals.csv');training=pd.read_csv(root/'training.csv')
    history=pd.read_csv(root/'history.csv');peer=pd.read_csv(root/'peer_removal_validation.csv')
    byfold=peer.groupby(['ticker','peer','broker','fold']).mean_delta_nll.mean().reset_index()
    candidates=byfold.groupby(['ticker','peer','broker']).mean_delta_nll.agg(
        mean_delta_nll='mean',folds='count',positive_folds=lambda v:int((v>0).sum())).reset_index()
    fraction=peer.groupby(['ticker','peer','broker']).mean_delta_nll.apply(lambda v:float((v>0).mean())).rename('positive_fit_fraction').reset_index()
    candidates=candidates.merge(fraction,on=['ticker','peer','broker'])
    candidates['stable_descriptive']=(candidates.positive_folds>=2)&(candidates.positive_fit_fraction>=.6)&(candidates.mean_delta_nll>0)
    candidates.to_csv(root/'peer_candidates.csv',index=False)
    wide=table.pivot(index='ticker',columns='method',values='nll');summary=[]
    for ticker in config['tickers']:
        primary=bounds[(bounds.ticker==ticker)&bounds.comparator.isin(['own_only','frozen_peers','shifted_peers'])]
        manual=bounds[(bounds.ticker==ticker)&bounds.comparator.isin(['raw_linear','aggregate_linear'])]
        summary.append(dict(ticker=ticker,own_nll=wide.loc[ticker,'own_only'],relational_nll=wide.loc[ticker,'relational'],
            gain_vs_own=wide.loc[ticker,'own_only']-wide.loc[ticker,'relational'],
            all_primary_intervals_positive=bool((primary.simultaneous_low>0).all()),
            all_manual_intervals_positive=bool((manual.simultaneous_low>0).all())))
    summary=pd.DataFrame(summary);summary.to_csv(root/'primary_summary.csv',index=False)
    fig,axes=plt.subplots(1,len(config['tickers']),figsize=(14,5),squeeze=False)
    names=['own_only','frozen_peers','shifted_peers','raw_linear','aggregate_linear']
    for ax,ticker in zip(axes.flat,config['tickers']):
        b=bounds[(bounds.ticker==ticker)&(bounds.block_sessions==5)].set_index('comparator').reindex(names)
        ax.errorbar(b.gain_nll,range(len(names)),xerr=[b.gain_nll-b.simultaneous_low,b.simultaneous_high-b.gain_nll],
            fmt='o',color='#177d78',capsize=4)
        ax.axvline(0,color='#9b3545',linestyle='--');ax.set_yticks(range(len(names)),names)
        ax.set_title(ticker);ax.set_xlabel('Gain NLL: control minus relational');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same-session masked association: simultaneous intervals, block 5 sessions')
    fig.tight_layout();fig.savefig(root/'association-comparison.png',dpi=160);plt.close(fig)
    new=training[~training.reused];verification=json.loads((root/'verification.json').read_text()) if (root/'verification.json').exists() else {}
    fence=chr(96)*3
    primary_winners=summary.loc[summary.all_primary_intervals_positive,'ticker'].tolist()
    manual_winners=summary.loc[summary.all_manual_intervals_positive,'ticker'].tolist()
    text=[
        '# Hasil masked association antarbroker pada sesi yang sama',
        f"{len(new)} fit neural baru dan {len(training)-len(new)} kontrol own_only dipakai ulang. Saham: {', '.join(config['tickers'])}; {len(config['folds'])} fold x {len(config['seeds'])} seed. Semua hasil lama dipertahankan.",
        'Pertanyaan: apakah aktivitas broker lain pada sesi t membantu merekonstruksi net-state broker yang disembunyikan pada sesi t, setelah mempertimbangkan riwayat broker itu sendiri? Ini asosiasi setelah snapshot EOD tersedia. Ini bukan prediksi yang tersedia sebelum penutupan, bukan relasi investor, dan bukan prediksi return.',
        '## Hasil utama',summary.to_markdown(index=False,floatfmt='.6f'),
        f"Saham yang mengalahkan seluruh tiga kontrol neural dengan batas bawah interval simultan positif pada semua ukuran blok: {', '.join(primary_winners) or 'tidak ada'}. Saham yang mengalahkan kedua kontrol manual linear dengan syarat yang sama: {', '.join(manual_winners) or 'tidak ada'}. Kedua syarat perlu dibaca bersama; kemenangan atas own_only saja belum membuktikan nilai tambah pembelajaran relasi.",
        macro.to_markdown(index=False,floatfmt='.6f'),table.to_markdown(index=False,floatfmt='.6f'),
        'NLL dan Brier lebih kecil lebih baik; accuracy dan recall lebih besar lebih baik. Rata-rata broker dan seed dihitung sebelum bootstrap tanggal. Nilai rata-rata antarsaham tidak berarti satu model gabungan: model tiap saham dilatih terpisah.',
        '## Metodologi dan pencegahan kebocoran',
        'Target t adalah query studi sebelumnya +1 sehingga label dan tanggal test identik. Own input t-5..t-1 dan OHLCV t-1 identik bytewise dengan kontrol lama. Peer input t-4..t; target broker dimask pada seluruh lima sesi, termasuk nilai, mask, dan identitas pada encoder. Input neural tidak memakai OHLCV hari t atau agregat broker pasar hari t.',
        'Token share dan share acceleration lama tidak dipakai pada peer karena denominatornya mengandung aktivitas target. Delapan token baru hanya berasal dari broker masing-masing: signed log net value/lot, lot-missing, consistency-5, perubahan pertama/kedua signed log net value, indikator net positif/negatif. Mean/std dihitung hanya dari slot present broker terpilih di train, clip [-10,10]; slot hilang menjadi nol. Pemilihan 12 broker, vocabulary dan threshold kelas besar tetap train-only dari snapshot frozen. Test mutation invariance diuji.',
        'SetEncoder dua SAB dengan pooling width24, tiga causal TCN kernel3/dilasi1/2/4, target ID, own projection dan head lima kelas dipakai ulang tanpa membuat backbone baru. Relational dan shifted memperbarui encoder; frozen_peers memakai random encoder beku. Keempat varian mempunyai nominal parameter dan bobot awal identik per fold/seed; jumlah parameter trainable berbeda pada kontrol.',
        training.groupby(['method','reused'])[['parameters','trainable_parameters','encoder_changed']].mean().reset_index().to_markdown(index=False,floatfmt='.6f'),
        'Own_only tidak diulang fit karena seluruh input yang dipakai model, label dan split tetap identik. Checkpoint dan probabilitas direplay dan dibandingkan dengan arsip. Own linear, frequency, Markov, dan persistence juga dipakai ulang pada tanggal/label yang sama. Indeks query lama dipetakan ke target t secara eksplisit.',
        'Raw_linear baru memakai own88 + seluruh history peer yang dimask, tanpa attention. Aggregate_linear memakai own88 + ringkasan tanpa identitas: mean/std token, present fraction, signed log jumlah net dan log jumlah absolut net atas peer terpilih pada lima sesi. Kontrol agregat penting karena keseimbangan net flow dapat menimbulkan asosiasi mekanis. Keduanya memakai StandardScaler train-only, logistic C 0.01/0.1/1, max_iter300, pilihan validation NLL. Satu head deterministik per broker/fold, tidak diulang untuk seed.',
        '## Ketidakpastian dan kontrol',
        bounds.to_markdown(index=False,floatfmt='.6f'),
        'Paired fold-stratified date-block 3/5/10 sesi, 2000 draw pada full run, family simultan 3 saham x 9 pembanding. Gain positif berarti relational lebih baik. Interval conditional pada fitted models dan historis sudah dieksplorasi; ini bukan konfirmasi baru. Purge20+embargo20 dan test yang tidak saling tumpang tindih dipertahankan. Semua checkpoint/head dan hash dibekukan sebelum evaluasi test baru.',
        'Shifted memakai circular shift peer di dalam split, termasuk kemungkinan sesi lebih akhir. Ini hanya kontrol alignment offline, bukan prediktor live dan bukan uji kausalitas. Input peer sampai t pada model utama juga belum tersedia sebelum snapshot EOD t.',
        pd.read_csv(root/'fold_seed_stability.csv').to_markdown(index=False,floatfmt='.6f'),
        '## Kontribusi peer pada validation',
        balance.to_markdown(index=False,floatfmt='.6f'),
        'Audit mekanisme hanya memakai train/validation. Coverage adalah porsi absolut net value pada 12 broker terpilih terhadap seluruh slot snapshot; imbalance adalah absolut jumlah net dibagi jumlah absolut net. Snapshot yang hampir seimbang dapat menghasilkan dependensi mekanis. Ini tidak membuktikan snapshot mencakup seluruh pasar, dan tidak membuktikan koordinasi investor.',
        candidates.sort_values(['ticker','mean_delta_nll'],ascending=[True,False]).groupby('ticker').head(10).to_markdown(index=False,floatfmt='.6f'),
        'Semua pasangan berarah diekspor. Positif berarti NLL model memburuk saat peer dihapus tanpa refit. Stable_descriptive mensyaratkan sedikitnya dua fold positif dan >=60% fit positif; ini bukan p-value. Mask perturbation dapat berada di luar distribusi, dan pasangan teratas bukan bukti koordinasi investor atau efek kausal. Tidak ada pasangan dipilih dari test.',
        '## Training dan verifikasi',
        f"AdamW lr0.001/wd0.01, batch32, AMP CUDA/CPU fallback, clip1, scheduler plateau, early stopping validation NLL. Fit baru berjalan {new.epochs.min()}–{new.epochs.max()} epoch, total waktu fit {new.seconds.sum()/60:.2f} menit, peak allocated CUDA {new.peak_gpu_mb.max():.2f} MiB. Angka memori ini bukan total VRAM proses; waktu tidak mencakup audit, evaluasi dan pengemasan.",
        history.groupby('method')[['nonfinite_gradient_norms','confirmed_scaler_skips']].sum().reset_index().to_markdown(index=False),
        fence+'json\n'+json.dumps(verification,indent=2)+'\n'+fence,
        'Enam unit test baru dan lima tes model sebelumnya lulus. Suite lengkap: 107 tes, 104 lulus dan tiga kegagalan legacy yang sama terkait data/tokenizer/tanggal live. Smoke data ANTM satu fold/seed dan satu epoch mendahului full run; empat checkpoint dan 24 head berhasil direplay, selisih GPU/CPU maksimum sekitar 1.34e-7. Replay full membangun ulang dataset/scaler, label kalender, hash, bobot awal, best epoch, pembaruan/freeze encoder, checkpoint neural, head linear, baseline reuse dan NLL/probabilitas. Log lengkap disertakan; tidak mengklaim seluruh tes lulus.',
        '## Batasan dan langkah berikutnya',
        'NET_ZERO berarti net nol/tidak tercatat pada snapshot valid, bukan tidak ada transaksi gross. Data net tidak diubah menjadi gross buy/sell buatan. Ekspor memeriksa bahwa top-k token source tidak mengalami truncation; kelengkapan vendor dan corporate action historis belum diaudit independen. Macro state recall menghitung kelas yang muncul dalam evaluasi. Asosiasi yang muncul dapat berasal dari mekanisme pasar, aktivitas umum, atau struktur data; embedding cosine dan attention bukan penjelasan ekonomi dengan sendirinya.',
        'Jika relational belum mengungguli kontrol neural dan manual secara konsisten, jangan menyebut pasangan kandidat sebagai relasi prediktif yang terbukti. Jika ada manfaat asosiasi, bekukan resep dan periksa stabilitas pada sesi baru sebelum menghubungkannya ke target harga. Task ini tidak menyediakan backtest trading atau bukti keuntungan investasi.',
        '## Lokasi perubahan',
        'Proyek: .. File baru event_association_study.py (data, kontrol, training dan evaluasi), event_association_verify.py (replay), event_association_results.py (laporan), test_event_association.py, EVENT-ASSOCIATION-AUDIT.md, EVENT-ASSOCIATION-PROTOCOL.md. README.md diperbarui. Model/trainer dan exporter dashboard proyek yang sudah berfungsi dipakai ulang. Snapshot source sebelum training tersimpan di sources/.',
    ]
    (root/'RESULTS.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    build_dashboard(root)
    print(macro.to_string(index=False));print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');report(p.parse_args().root)
