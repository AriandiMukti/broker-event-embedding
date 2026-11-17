"""Post-fit audit and descriptive report; never changes training or selections."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from event_data import preprocess
from event_objective_study import sha,write_json


def audit_prepared(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());receipts=[]
    for ticker in config['tickers']:
        meta=json.loads((root/f'{ticker}-data-audit.json').read_text())
        cache=np.load(root/f'{ticker}-daily.npz',allow_pickle=False)
        raw={key:cache[key] for key in cache.files};raw.update(names=meta['features'],groups=meta['groups'])
        for fold in config['folds']:
            folder=root/ticker/f'fold-{fold}';arrays=np.load(folder/'prepared.npz',allow_pickle=False)
            saved=torch.load(folder/'preprocess.pt',map_location='cpu',weights_only=False)['preprocess']
            prepared,state=preprocess(raw,arrays['train_indices'],config['window'])
            for key in ['x','tokens','ids','mask','y']:np.testing.assert_array_equal(prepared[key],arrays[key])
            for key in ['mean','std','token_mean','token_std','target_mean','target_std','constant_columns','fit_days']:
                np.testing.assert_array_equal(state[key],saved[key])
            assert state['vocabulary']==saved['vocabulary']
            receipts.append(dict(ticker=ticker,fold=fold,prepared_arrays_exact=True,train_only_transform_exact=True,
                raw_cache_sha256=sha(root/f'{ticker}-daily.npz'),prepared_sha256=sha(folder/'prepared.npz')))
    write_json(root/'prepared_rebuild_verification.json',receipts);return receipts


def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);table=pd.read_csv(root/'by_stock_horizon.csv')
    names=['zero','lightgbm_manual','multitask','return_only']
    labels=['Nol','LightGBM','Multitask','Return-only'];colors=['#667085','#c08a20','#3978be','#15867b']
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=False)
    for ax,ticker in zip(axes,sorted(table.ticker.unique())):
        part=table[table.ticker==ticker];horizons=sorted(part.horizon.unique());xs=np.arange(len(horizons));width=.19
        for j,(name,label,color) in enumerate(zip(names,labels,colors)):
            values=part[part.training_objective==name].set_index('horizon').reindex(horizons).mae
            ax.bar(xs+(j-1.5)*width,values,width,label=label,color=color)
        ax.set_title(ticker,fontweight='bold');ax.set_xticks(xs,[str(h) for h in horizons])
        ax.set_xlabel('Horizon (sesi)');ax.set_ylabel('MAE log-return');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].legend(frameon=False,fontsize=8)
    fig.suptitle('Objective ablation: Set Transformer + TCN tanpa gating',fontsize=15,fontweight='bold')
    fig.text(.5,.015,'3 fold × 3 seed per saham · rata-rata loss seed · test historis eksploratif',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.92));fig.savefig(root/'objective-comparison.png',dpi=160);plt.close(fig)


def report(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());audit_prepared(root);plots(root)
    summary=pd.read_csv(root/'macro_summary.csv');paired=pd.read_csv(root/'paired_summary.csv')
    bands=pd.read_csv(root/'paired_bands.csv');training=pd.read_csv(root/'training.csv');history=pd.read_csv(root/'history.csv')
    cells=pd.read_csv(root/'by_stock_horizon.csv')
    predictions=pd.read_csv(root/'predictions.csv')
    pivot=cells.pivot(index=['ticker','horizon'],columns='training_objective',values='mae')
    primary=bands[(bands.treatment=='return_only')&(bands.comparator=='multitask')]
    macro=primary[primary.ticker=='ALL']
    wins=int((paired.gain_mae>0).sum());beats_zero=int((pivot.return_only<pivot.zero).sum())
    table=pivot[['zero','train_mean','lightgbm_manual','multitask','return_only']].reset_index()
    table.columns=['Saham','Horizon','Nol','Mean train','LightGBM','Multitask baru','Return-only']
    table.to_csv(root/'comparison_nine_cells.csv',index=False)
    macro_table=summary[['training_objective','mae','rmse']].rename(columns={'training_objective':'Objective / baseline'})
    verified=json.loads((root/'verification.json').read_text()) if (root/'verification.json').exists() else {}
    counts={(row['ticker'],row['fold']):row['counts']['train']
            for row in json.loads((root/'data_manifest.json').read_text())}
    batches=sum(int(row.epochs)*int(np.ceil(counts[row.ticker,row.fold]/config['batch_size']))
                for row in training.itertuples())
    checkpoints=list(root.glob('*/fold-*/*/best.pt'))
    finite=all(all(bool(torch.isfinite(value).all()) for value in
        torch.load(path,map_location='cpu',weights_only=False)['state_dict'].values()) for path in checkpoints)
    assert finite and len(checkpoints)==config['planned_fits']
    numeric=dict(checkpoints_checked=len(checkpoints),all_checkpoint_tensors_finite=finite,
        training_batches=batches,nonfinite_gradient_norm_batches=int(history.nonfinite_gradient_steps.sum()),
        nonfinite_norm_fraction=float(history.nonfinite_gradient_steps.sum()/batches),
        optimizer_uses_amp_gradscaler=True,per_batch_scaler_skip_receipts_recorded=False,
        explanation='Logs count nonfinite global gradient norms; do not independently establish exact scaler skip counts')
    write_json(root/'numeric_audit.json',numeric)
    gradient=history.groupby('training_objective').agg(gradient_norm_mean=('gradient_norm_mean','mean'),
        clipped_fraction=('clipped_fraction','mean'),nonfinite_norm_batches=('nonfinite_gradient_steps','sum')).reset_index()
    parts=[
        '# Return-only versus multitask — hasil aktual',
        f"Selesai {len(training)} fit baru, {len(training)//2} pasangan inisialisasi identik. ADRO/ANTM/BBRI, Set Transformer + TCN tanpa gating; model terpisah per saham, 3 fold × 3 seed11/42/123 × 2objective. Baseline menggunakan kembali prediksi frozen; tidak dilatih ulang. Hasil lama tidak ditimpa.",
        '## Kesimpulan yang didukung data',
        f'Return-only memiliki MAE point estimate lebih rendah daripada multitask baru pada {wins}/{len(paired)} kombinasi saham/horizon; lebih rendah daripada prediksi nol pada {beats_zero}/{len(pivot)} kombinasi. Ini hasil eksploratif, bukan konfirmasi kemampuan memprediksi return.',
        macro_table.to_markdown(index=False,floatfmt='.6f'),
        'MAE/RMSE macro adalah rata-rata metrik tiap saham/horizon/fold/seed. MAE = mean(abs(prediksi − log(Close[t+h]/Close[t]))). Loss tiap seed dihitung dahulu; ini bukan error ensemble prediksi. Setiap saham mempunyai378tanggal test unik,126per fold; label overlap, saham dan seed bukan observasi tanggal independen.',
        '## Per saham dan horizon',table.to_markdown(index=False,floatfmt='.6f'),
        'Multitask baru memakai seleksi MAE return validasi. Hasilnya dapat berbeda dari multitask lama yang dipilih berdasarkan loss gabungan; keduanya tidak boleh dicampur dalam primary paired comparison.',
        '## Ketidakpastian paired comparison',
        macro[['backbone','block_sessions','dates','gain_mae','simultaneous_low','simultaneous_high','family_columns']].to_markdown(index=False,floatfmt='.6f'),
        'Gain positif berarti MAE return-only lebih kecil. Circular date-block bootstrap20/40 dengan2000draws sinkron atas saham/horizon, seed-loss dirata-ratakan. Simultaneous95%bands mencakup seluruh50kolom perbandingan termasuk macro. Interval conditional fitted models, tidak mencakup seluruh ketidakpastian pemilihan resep. Lihat paired_bands.csv untuk seluruh kombinasi; paired_summary.csv memisahkan kemenangan fold dan seed.',
        ('Tidak ada bukti perbaikan macro yang konsisten terhadap multitask pada kedua ukuran blok: interval mencakup nol.'
         if ((macro.simultaneous_low<=0)&(macro.simultaneous_high>=0)).all()
         else 'Periksa arah dan batas kedua ukuran blok pada tabel. Kesimpulan berlaku hanya untuk cohort historis eksploratif ini.'),
        bands[(bands.ticker=='ALL')&(bands.comparator=='zero')][['treatment','block_sessions','gain_mae','simultaneous_low','simultaneous_high']].to_markdown(index=False,floatfmt='.6f'),
        'Kedua objective memiliki MAE macro lebih besar daripada prediksi nol. Pada cohort ini, batas simultaneous gain terhadap nol juga negatif untuk kedua panjang blok; kesimpulan tidak otomatis berlaku pada sesi pasar yang belum diamati.',
        '## Prosedur dan training aktual',
        'Snapshot input mencakup 1.556 sesi, 2 Juni 2020–16 November 2026. Train per fold: 893/1.019/1.145 sampel, validasi dan test masing-masing126 per fold. Label purge20 sesi ditambah embargo20 sesi sebelum blok berikutnya. Tanggal sumber yang tidak valid atau broker yang tidak tersedia tetap dimask sesuai audit sebelumnya.',
        'Broker token8fitur + broker-ID embedding → dua self-attention set block dan learned-query pooling harian → gabung proyeksi fitur harian → causal residual TCN kernel3/dilasi1,2,4 → head9output. Broker padding dimask; jumlah broker bisa berbeda per hari. Width24, window20, max96broker; receptive field TCN15sesi, sementara fitur rolling juga merangkum riwayat sebelumnya. Gate pada arm tanpa gating selalu1 dan bukan feature importance. CNN/static/dynamic lama tetap tersimpan, tidak dilatih ulang pada eksperimen objective ini.',
        'Arsitektur, scaler train-only, broker identity/masking, window20, width24, batch64 dan input snapshot sama. Return loss standardizedMSE pada kedua objective. Multitask menambahkan0.25log-volMSE+0.25directionBCE. Seleksi checkpoint/scheduler/early stopping memakai raw-returnMAE validasi yang sama. Seluruh54checkpoint dipilih dan hash dibekukan sebelum evaluasi test pertama.',
        f"Tanggal query test {predictions.date.min()} sampai {predictions.date.max()}, {predictions.date.nunique()} tanggal unik bersama. Horizon5/10/20 sama; label dihitung sesudah query dan tidak dimasukkan ke fitur. Parameter model {training.parameters.min()}–{training.parameters.max()} karena kosakata broker train berbeda antar saham/fold.",
        f"AdamW lr0.001/wd0.01, ReduceLROnPlateau, clipping1, max24epoch/patience6, AMP CUDA. Total waktu fit {training.seconds.sum()/60:.2f} menit; epoch aktual {training.epochs.min()}–{training.epochs.max()}; best epoch {training.best_epoch.min()}–{training.best_epoch.max()}; maksimum allocated VRAM {training.peak_gpu_mb.max():.2f} MiB. Waktu ini jumlah fit, tidak termasuk audit/evaluasi/package; allocated VRAM bukan seluruh VRAM proses.",
        gradient.to_markdown(index=False,floatfmt='.6f'),
        'Norm dicatat sebelum clipping; norm/frekuensi clipping di atas rata-rata baris epoch, bukan seluruh batch berbobot. Kolom nonfinite adalah jumlah minibatch. Clipping terjadi pada kedua arm dan tidak membuktikan sumber kegagalan prediksi.',
        f"Audit numerik: {numeric['nonfinite_gradient_norm_batches']} dari {batches} minibatch mencatat norm gradient nonfinite ({100*numeric['nonfinite_norm_fraction']:.3f}%). Loss train/val tetap finite dan seluruh tensor54checkpoint finite. Optimizer memakai AMP GradScaler; log tidak menyimpan receipt skip/scale per batch, sehingga angka norm nonfinite tidak diklaim sebagai hitungan skip yang diverifikasi independen. Replay GPU/CPU menguji inference, bukan membuktikan training AMP identik dengan FP32. Lihat numeric_audit.json.",
        '## Verifikasi','```json\n'+json.dumps(verified,indent=2)+'\n```',
        'Prepared input dan seluruh statistik scaler/vocab/constant-column direkonstruksi ulang dari daily cache menggunakan hanya train, cocok persis untuk9stock-fold. Cache berasal dari snapshot raw yang sudah diverifikasi pada cohort sebelumnya. Ini tidak memverifikasi independen kebenaran vendor/corporate action. Automated test results dan smoke GPU/CPU disertakan dalam paket.',
        'Tes:23tes event/objective awal lulus, ditambah2tes statistik seed-loss/cakupan tanggal lulus (25kasus unik). Suite proyek penuh diperiksa sebelum dua tes tambahan:82/85lulus. Tiga kegagalan legacy yang sama: adaptive checkpoint berbeda terhadap input live yang direvisi; tokenizer broker-relations berubah terhadap cache lama; context_end broker-token live16November berbeda dari ekspor13November. Log lengkap disertakan. Ketiganya bukan tes objective baru dan tidak disembunyikan dengan melonggarkan toleransi.',
        '## Batas interpretasi dan langkah berikutnya',
        'Eksperimen ini adalah ablation objective, bukan ablation fitur. Tidak menghapus fitur atau mengklaim gate/importance baru; feature importance dan refit ablation sebelumnya tetap berada di hasil cohort terdahulu. Probabilitas/volatilitas return-only tidak dilatih dan tidak dievaluasi. Secondary multitask metrics tersedia terpisah, tidak menjadi dasar memilih objective.',
        'Feature importance/ablation sebelumnya: runs/event-adro-antm-bbri-20261116/importance.csv dan ablation.csv. Nilainya bukan hasil objective study ini. Tidak menafsirkan gate kecil sebagai bukti noise dan tidak menghapus fitur dari test historis.',
        'Tidak menjalankan backtest baru: rule lama memakai probabilitas positif yang tidak tersedia pada arm return-only. Perbandingan strategi memerlukan protokol sinyal return-only yang ditentukan sebelum menguji data baru; memakai auxiliary head yang tidak dilatih akan tidak valid.',
        'IHSG/USDIDR/sektor belum tersedia. Ownership konstan di train tetap frozen; rezim historis adalah volatilitas saham. Harga belum diverifikasi corporate action secara independen. Seluruh periode test telah dipakai untuk eksplorasi; tidak mengklaim test baru/untouched atau kesiapan trading.',
        'Bekukan resep dan kumpulkan sesi baru untuk konfirmasi. Untuk pemeriksaan numerik berikutnya, bandingkan training AMP/FP32 pada train/val dengan logging scale dan skipped update yang eksplisit; jangan memilihnya berdasarkan test. Jika tetap meneliti return-MAE, rencanakan eksperimen terpisah yang menyelaraskan training loss dengan MAE (misalnya L1), dengan kontrol dan seleksi validasi sama. Hasil saat ini tidak otomatis mengizinkan penggantian objective/fitur produksi atau perluasan kompleksitas model.',
        '## File',
        'Proyek asal: .. Hasil: runs/event-objectives-20261116. Source training snapshot dengan SHA256 ada di sources/.',
        'event_training.py dan event_predict.py: objective terpisah, common validation selection dan pemblokiran output auxiliary return-only. event_objective_study.py: paired runner/freeze/test/statistics. event_objective_verify.py: checkpoint replay. event_objective_results.py: audit transformasi, laporan dan gambar. dashboard.py: panel objective pada exporter GUI yang sudah ada. test_event_objectives.py dan EVENT-OBJECTIVE-PROTOCOL.md: tes dan resep. README.md: perintah dan lokasi hasil.',
    ]
    (root/'RESULTS.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(macro_table.to_string(index=False));print(macro.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root');report(parser.parse_args().root)
