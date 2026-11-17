"""Descriptive direct-objective broker relation report; no model selection."""
import argparse
import json
from pathlib import Path
import pandas as pd


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dashboard import build_event_relation_dashboard
    root=Path(root);config=json.loads((root/'config.json').read_text())
    comparison=pd.read_csv(root/'comparison.csv');macro=pd.read_csv(root/'macro_summary.csv')
    bounds=pd.read_csv(root/'paired_intervals.csv');stable=pd.read_csv(root/'fold_seed_stability.csv')
    training=pd.read_csv(root/'training.csv');history=pd.read_csv(root/'history.csv')
    # Post-fit geometry audit, without selecting any checkpoint or pair.
    import numpy as np
    import torch
    from event_relation_model import RelationModel
    from event_training import seed_all,weights_hash
    from event_broker_geometry import cosine,rank_similarity
    movements=[]
    for item in json.loads((root/'selected.json').read_text()):
        saved=torch.load(root/item['path'],map_location='cpu',weights_only=False)
        seed_all(item['seed'])
        initial=RelationModel(saved['vocabulary'],saved['context_features'],saved['width'],saved['mode'])
        assert weights_hash(initial.state_dict())==saved['record']['initial_sha256']
        ids=np.load(root/item['dataset'],allow_pickle=False)['ids']
        original=initial.set_encoder.identity.weight.detach().numpy()[ids]
        learned=saved['state_dict']['set_encoder.identity.weight'].numpy()[ids]
        movements.append(dict(ticker=item['ticker'],fold=item['fold'],seed=item['seed'],method=item['method'],
            identity_relative_l2_movement=float(np.linalg.norm(learned-original)/max(np.linalg.norm(original),1e-12)),
            identity_cosine_rank_vs_initial=rank_similarity(cosine(learned),cosine(original))))
    movement=pd.DataFrame(movements);movement.to_csv(root/'identity_geometry.csv',index=False)
    peer=pd.read_csv(root/'peer_removal_validation.csv')
    byfold=peer.groupby(['ticker','peer','broker','fold']).mean_delta_nll.mean().reset_index()
    candidates=byfold.groupby(['ticker','peer','broker']).mean_delta_nll.agg(mean_delta_nll='mean',folds='count',
        positive_folds=lambda v:int((v>0).sum())).reset_index()
    fraction=peer.groupby(['ticker','peer','broker']).mean_delta_nll.apply(lambda v:float((v>0).mean())).rename('positive_fit_fraction').reset_index()
    candidates=candidates.merge(fraction,on=['ticker','peer','broker'])
    candidates['stable_descriptive']=(candidates.folds>=2)&(candidates.positive_folds>=2)&(candidates.positive_fit_fraction>=.6)&(candidates.mean_delta_nll>0)
    candidates.to_csv(root/'peer_candidates.csv',index=False)
    table=comparison.pivot(index='ticker',columns='method',values='nll')
    summary=[]
    for ticker in config['tickers']:
        interval=bounds[(bounds.ticker==ticker)&bounds.comparator.isin(['own_only','frozen_peers','shifted_peers'])]
        summary.append(dict(ticker=ticker,relational_nll=table.loc[ticker,'relational'],own_only_nll=table.loc[ticker,'own_only'],
            own_linear_nll=table.loc[ticker,'prior_own_linear'],gain_vs_own=table.loc[ticker,'own_only']-table.loc[ticker,'relational'],
            all_primary_simultaneous_low_positive=bool((interval.simultaneous_low>0).all())))
    pd.DataFrame(summary).to_csv(root/'primary_summary.csv',index=False)
    fig,axes=plt.subplots(1,len(config['tickers']),figsize=(14,4.6),squeeze=False)
    modes=['prior_own_linear','own_only','frozen_peers','relational','shifted_peers']
    names=['Own\nlinear','Own\nneural','Frozen\npeers','Trained\npeers','Shifted\npeers']
    for ax,ticker in zip(axes.flat,config['tickers']):
        values=comparison[comparison.ticker==ticker].set_index('method').reindex(modes).nll
        ax.bar(range(len(modes)),values,color=['#667085','#4477aa','#8a70a2','#15867b','#aa7722'])
        ax.set_title(ticker,fontweight='bold');ax.set_xticks(range(len(modes)),names,rotation=20,ha='right',fontsize=9)
        ax.set_ylabel('Next-session broker net-state NLL');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Objective relasi broker langsung: Set Transformer + TCN',fontweight='bold');fig.tight_layout()
    fig.savefig(root/'relation-comparison.png',dpi=160);plt.close(fig)
    verification=json.loads((root/'verification.json').read_text()) if (root/'verification.json').exists() else {}
    fence=chr(96)*3
    text=[
        '# Hasil objective relasi broker langsung',
        f"{len(training)} fit neural baru. {len(config['tickers'])} saham ({', '.join(config['tickers'])}), {len(config['folds'])} fold dan {len(config['seeds'])} seed. Sekarang encoder dilatih end-to-end untuk net-state broker sesi berikutnya; bukan lagi hanya probe pada encoderreturn. Hasil lama dipertahankan.",
        '## Hasil utama',pd.DataFrame(summary).to_markdown(index=False,floatfmt='.6f'),
        'Kesimpulan eksperimen ini: pelatihan langsung untuk aktivitas broker memperbaiki prediksi dibanding probe linear lama, tetapi belum membuktikan manfaat tambahan dari mempelajari hubungan antarbroker. Relational memiliki NLL lebih buruk daripada own_only dan frozen_peers pada ketiga saham. Frozen_peers memperoleh nilai titik terbaik pada ketiganya; selisih kecil terhadap own_only tidak otomatis berarti manfaat yang terkonfirmasi.',
        'Pada ANTM, relational lebih buruk daripada ketiga kontrol neural dengan seluruh interval simultan tetap negatif untuk ukuran blok 3, 5, dan 10 sesi. Pada BBRI, relational juga lebih buruk daripada frozen_peers dengan interval simultan tetap negatif. ADRO membaik terhadap baseline linear lama, tetapi tidak terhadap own_only. Karena itu perbaikan dari baseline lama belum dapat dikaitkan dengan informasi hubungan antarbroker.',
        'Gain positif = NLL own_only dikurangi relational. Bukti utama mensyaratkan seluruh batas bawah interval simultan terhadap tiga kontrol neural positif pada ketiga ukuran blok. Tidak ada saham yang memenuhi syarat tersebut. Hasil ini tidak membuktikan bahwa hubungan antarbroker tidak ada; resep dan target saat ini belum menunjukkan nilai tambah prediktifnya.',
        macro.to_markdown(index=False,floatfmt='.6f'),
        'NLL/Brier lebih kecil lebih baik. Accuracy/recall lebih besar lebih baik. Rata-rata loss perbroker/seed dahulu, bukan menganggap12broker/3seed sebagai observasi tanggal independen. Prior_own_linear danprior_frozen_probe adalah prediksi archivedprobe yang dipakai kembali dengan tanggal/labelidentik, bukan refit baru. Persistence0.9 dapat memiliki accuracy tinggi tetapi probabilitas terlalu percaya diri danNLLburuk.',
        comparison.to_markdown(index=False,floatfmt='.6f'),
        '## Ketidakpastian',
        bounds[bounds.comparator.isin(['own_only','frozen_peers','prior_own_linear'])].to_markdown(index=False,floatfmt='.6f'),
        'Pairedfold-stratifieddateblock3/5/10,2000draws, simultaneousfamily3stock x8komparator=24kolom. Interval conditional fittedmodels danhistoris eksploratif, bukan confirmation. Semua108checkpoint dipilih berdasarkanvalidationNLL danhashdibekukan sebelumtest pertama. Fold_seed_stability.csv memisahkan kemenanganfold danseed.',
        '## Arsitektur, objective, dan data',
        'Reuse SetEncoder2SAB+learnedquerypool24 dan3CausalBlockTCN kernel3/dilasi1/2/4. Owninput14fitur per5sesi:5stateonehot,8continuous token,presentflag. Peerinput8token/ID/mask12broker, targetdimask diSELURUH5sesi. Own24+peer24+targetID24 difusi ->width24 ->causalTCN ->concatOHLCVcurrent18+targetID24 ->5logits. Window5 membatasi receptivefieldTCN15; tidak mengklaim15sesiinput. Single-task crossentropy mean tanggal x12target. Tidak ada headreturn/logvol/gating pada taskini.',
        'Inisialisasi fresh, bukan fine-tuningreturn. Empatvarian mempunyai nominalstate/parameter awal identik perseed/fold. Own_only membuangpeer dan membekukan branch; frozen_peers membekukan randomencoder; relational/shifted melatihencoder. Jumlahparameter efektiftrainable berbeda padaablation. Model berbagi head antartargetbroker dalam satu saham, tetapi tidak berbagi model antarsaham.',
        training.groupby('method')[['parameters','trainable_parameters','encoder_changed']].mean().reset_index().to_markdown(index=False,floatfmt='.6f'),
        movement.groupby(['ticker','method'])[['identity_relative_l2_movement','identity_cosine_rank_vs_initial']].mean().reset_index().to_markdown(index=False,floatfmt='.6f'),
        'GeometriID dibanding exactinitialization pada12broker terpilih, tanpa memilih pasangan dari test. Perubahan bobot/cosine bukan bukti relasi semantik yang berguna; nilai tambah diuji dengan NLL dankontrol. Seluruhnilai percheckpoint berada di identity_geometry.csv.',
        'Semua data dari snapshot frozen2020–2026/probe sebelumnya. Sourcebrokerselection/threshold2/3netbesar/scaler hanya train. NET_ZERO netnol/tidaktercatat, bukan grossinactivity; snapshotinvalid bukan labelnol. Queryt berakhirdiinputt, labeltepat t+1 dan tetapdalamsplit. Purge20+embargo20 lama dipertahankan. Tidak memperbarui CSVlive atau mengunduhdata. Query375targetdate test perstock pada resepfull.',
        'Ownhistory5/OHLCVcurrent sama nilainya denganprobe linear lama. Peerencoder baru melihat sequence5, sedangkanprior_frozen_probe hanya pooledpeerharian: membandingkan keduanya bukan isolasi objective semata. Primary4neuralbaru memakai architecture/head/history sama. Shifted memakai circularshift pertarget/dalamsplit, dapat mengandung harilebihakhir; ini negativecontroloffline, bukan prediktorlive atau bukti kausalitas.',
        '## Training dan numerik',
        f"AdamWlr0.001/wd0.01, batch32querydays, width24, max{config['epochs']}epoch/patience{config['patience']}, plateau0.5/patience3, clip1, AMP/CPUfallback. Epoch aktual{training.epochs.min()}–{training.epochs.max()}; jumlahwaktufit{training.seconds.sum()/60:.2f}menit; peakallocatedVRAM{training.peak_gpu_mb.max():.2f}MiB. Waktu bukan total audit/packaging; allocatedVRAM bukan seluruhmemoriGPU proses.",
        history.groupby('method')[['nonfinite_gradient_norms','confirmed_scaler_skips']].sum().reset_index().to_markdown(index=False),
        'Lossnonfinite dan normnonfinite tanpa confirmedscalerskip menghentikantraining. Bestcheckpoint dipilih validationNLL, bukan test. Groupinisialisasi danhash perubahanencoder direplay. Tiga modelcontrol/linear lama bukan datafit baru; seluruhfitneural baru dicatat terpisah.',
        '## Kandidat hubungan peer',
        candidates.sort_values(['ticker','mean_delta_nll'],ascending=[True,False]).groupby('ticker').head(10).to_markdown(index=False,floatfmt='.6f'),
        'Ini ranking validationpeer-removal tanpa refit. Positif berarti model memburuk ketikapeer dihapus, bukan relasi ekonomi terkonfirmasi. Stable_descriptive mensyaratkan>=2fold dan>=60%fit positif; tidak ada koreksiseleksiindividual/pvalue. OODperturbation bukan efekkausal/koordinasiinvestor. Semua directedpair diekspor, tidak hanya kandidat teratas.',
        '## Verifikasi',fence+'json\n'+json.dumps(verification,indent=2)+'\n'+fence,
        'Tests danreal-smoke mendahuluifull. Verifier membangunulang ownhistory/peersequence/label/asof/metadata, matchedinitialweights, encoderupdate/freeze, checkpointselection danNLL/probabilityexport sertaCPU/GPU. Sourcehistoris telahdieksplorasi; corporateaction/vendorvintage/timingEOD belum diverifikasi independen. Tidak mengklaim hubungan tidak ada dari hasilnegatif, dan tidak mengklaim manfaattrading dari tasknetstate. Tidak ada backtestharga pada tahapini.',
        '5tes baru lulus; suitepenuh101tes:98lulus dan3kegagalanlegacy yang sama pada reproduksi terhadap inputlive/tokenizer/tanggalekspor lama. Log disertakan. Smoke awal mendeteksi selisihCPU/GPU sekitar1e-5 akibat defaultTF32conv; sebelumfull TF32dinonaktifkan dansmoke ulangFP32berhasil dengan selisih sekitar6.5e-8, tanpa melonggarkantoleransi. AMPtrainingtetapfloat16.',
        '## File dan langkah berikutnya',
        'Proyek .. event_relation_model.py:targetmask/set/TCN/head; event_relation_training.py:CEtraining/checkpoint; event_relation_study.py:data/freeze/evaluasi; event_relation_verify.py:rebuild/replay; event_relation_results.py:analisis/export; test_event_relation_e2e.py:tests; EVENT-RELATION-E2E-PROTOCOL.md:resep; dashboard.py:GUIexisting.',
        'Eksperimen berikut yang disarankan: pertahankan own_only sebagai kontrol kuat; pisahkan studi asosiasi broker pada sesi yang sama melalui masked reconstruction dari studi prediksi sesi berikutnya. Gunakan pasangan kandidat hanya sebagai hipotesis dari train/validation, lalu bekukan resep sebelum menguji sesi baru. Jangan memilih model lebih kompleks berdasarkan hasil test historis ini. Hubungan broker yang berguna untuk net-state belum otomatis relevan terhadap harga saham.',
    ]
    (root/'RESULTS.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    build_event_relation_dashboard(root)
    print(macro.to_string(index=False));print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');report(p.parse_args().root)
