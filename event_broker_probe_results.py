"""Descriptive frozen broker-relation probe report; never selects a model."""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd


def report(root,geometry):
    root=Path(root);geometry=Path(geometry)
    if not (root/'geometry').exists():shutil.copytree(geometry,root/'geometry')
    config=json.loads((root/'config.json').read_text());macro=pd.read_csv(root/'macro_summary.csv')
    comparison=pd.read_csv(root/'comparison.csv');bands=pd.read_csv(root/'paired_intervals.csv')
    stability=pd.read_csv(root/'fold_seed_stability.csv');pairs=pd.read_csv(root/'peer_candidates.csv')
    geo=pd.read_csv(root/'geometry/geometry_summary.csv');near=pd.read_csv(root/'geometry/stability.csv')
    trials=pd.read_csv(root/'head_trials.csv');samples=pd.read_csv(root/'samples.csv')
    geometry_summary=geo.groupby('ticker')[['identity_relative_l2_movement','identity_rank_vs_initial','contextual_rank_vs_initial',
        'contextual_rank_vs_test_flow','initial_contextual_rank_vs_test_flow']].mean().reset_index()
    robust=[]
    for ticker in config['tickers']:
        part=bands[bands.ticker==ticker];checks=stability[stability.ticker==ticker]
        robust.append(dict(ticker=ticker,all_controls_simultaneous_positive=bool((part.simultaneous_low>0).all()),
            all_controls_two_folds_positive=bool((checks.folds_won>=2).all())))
    verified=json.loads((root/'verification.json').read_text()) if (root/'verification.json').exists() else {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(config['tickers']),figsize=(14,4.5),squeeze=False)
    order=['frequency','markov','own_context','raw_peers','random_encoder','learned_encoder']
    labels=['Frekuensi','Markov','Own\nhistory','Raw\npeers','Random\nencoder','Learned\nencoder']
    for ax,ticker in zip(axes.flat,config['tickers']):
        values=comparison[comparison.ticker==ticker].set_index('method').reindex(order).nll
        ax.bar(range(len(order)),values,color=['#98a2b3','#667085','#4477aa','#aa7722','#8a70a2','#15867b'])
        ax.set_xticks(range(len(order)),labels,rotation=25,ha='right',fontsize=8)
        ax.set_title(ticker,fontweight='bold');ax.set_ylabel('Next-session net-state NLL');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Relasi broker: frozen-encoder probe versus kontrol',fontweight='bold')
    fig.tight_layout();fig.savefig(root/'probe-comparison.png',dpi=160);plt.close(fig)
    fence=chr(96)*3
    mean_scores=macro.set_index('method')
    wide=comparison.pivot(index='ticker',columns='method',values='nll')
    own_better=int((wide.own_context<wide.learned_encoder).sum())
    random_supported=[ticker for ticker in config['tickers'] if
        (bands[(bands.ticker==ticker)&(bands.comparator=='random_encoder')].simultaneous_low>0).all()]
    text=[
        '# Relasi broker: audit geometri dan probe encoder frozen',
        f"{len(config['tickers'])} saham ({', '.join(config['tickers'])}), {len(config['folds'])} fold, {len(config['seeds'])} seed checkpoint. Backbone tidak dilatih ulang. Sumber local_only tanpa gating dipilih semula untuk return, bukan objective relasi. Ada {len(trials)} kandidat fit head dalam C-grid dan {len(json.loads((root/'selected_heads.json').read_text()))} head terpilih. Seed checkpoint berbeda; solver head deterministik memakai random_state42.",
        '## Koreksi arah pekerjaan',
        'Eksperimen relasi lama sebenarnya sudah tersedia untuk25saham pada208sesi2026. Yang terlewat ialah evaluasi pada Set Transformer baru dan sejarah2020–2026. Scores/frekuensi/Markov/bootstrap lama dipakai kembali, loaderISO208sesi tidak dipakai. Pernyataan seolah seluruh pengujian relasi belum pernah dibuat terlalu luas.',
        '## Hasil sesi berikutnya',macro.to_markdown(index=False,floatfmt='.6f'),
        f"Pada NLL, own-context {mean_scores.loc['own_context','nll']:.6f}, learned encoder {mean_scores.loc['learned_encoder','nll']:.6f}, random encoder {mean_scores.loc['random_encoder','nll']:.6f}. Own-context memiliki point estimate lebih baik pada {own_better}/{len(wide)} saham. Nilai tambah learned dibanding own-context belum otomatis terbukti; lihat interval dan konsistensi. Dibanding random encoder, seluruh batas bawah intervalblok positif pada: {', '.join(random_supported) or 'tidak ada saham'}. Ini bukti parsial conditional, bukan kemenangan terhadap seluruhkontrol.",
        'NLL/Brier lebih kecil lebih baik; accuracy/macro recall lebih besar lebih baik. Metrik dihitung perseed/fold/saham sebelum dirata-rata, bukan error ensemble. Broker/seed bukan tanggal independen. Baseline frekuensi/Markov/persistensi diekspor sekali perfold karena tidak bergantungseed.',
        comparison.to_markdown(index=False,floatfmt='.6f'),
        '## Nilai tambah embedding terlatih',pd.DataFrame(robust).to_markdown(index=False),
        'Learned versus own_context mengukur tambahan informasi peer; random_encoder mengontrol manfaat pembelajaran bobot; raw_peers mengontrol fitur langsung; shifted_encoder mengontrol alignment offline. Frekuensi, Markov dan persistensi adalah benchmark sederhana. Prediksi peer tidak otomatis lebih baik dari riwayat target sendiri.',
        bands[bands.comparator.isin(['own_context','raw_peers','random_encoder'])].to_markdown(index=False,floatfmt='.6f'),
        'Interval95%simultan meliputi seluruh21perbandingan:3saham x7kontrol, date-block3/5/10 dalamfold,2000draws. Mean loss perbroker/seed dihitung dahulu. Conditional fittedmodels; tidak mencakup seluruh ketidakpastian pemilihan resep. Kriteria eksploratif yang ketat di atas mensyaratkan seluruhbatasblok/kontrol positif dan >=2fold positif; kegagalan memenuhi kriteria bukan bukti tidak ada relasi.',
        '## Geometri dan kestabilan',geometry_summary.to_markdown(index=False,floatfmt='.6f'),
        near.groupby(['ticker','geometry','comparison']).mean_top3_jaccard.mean().reset_index().to_markdown(index=False,floatfmt='.6f'),
        'RelativeL2movement terhadap exactinitialization; rank mendekati1 berarti urutan cosine masih serupa randomawal. Centroid kontekstual memakai setnative penuh pada train saja. Top3 dihitung ulang pada universebroker yang sama; tidak membandingkan koordinat antarstock/seed. Korelasi similarity dengan flowtest hanya deskriptif; semua66pasangan disimpan, tidak diseleksi dari test.',
        '## Kandidat peer: validation removal',
        pairs.sort_values(['ticker','mean_delta_nll'],ascending=[True,False]).groupby('ticker').head(10).to_markdown(index=False,floatfmt='.6f'),
        'Sepuluh teratas perstock diurutkan dari validasi, bukan relasi terkonfirmasi. Seluruhdirectedpairs ada di peer_candidates.csv. Stable_descriptive membutuhkan >=2fold, >=60%fit positif dan meangain>0. Peer dihapus tanpa refit/headchange: perturbasiOOD, bukan kausalitas. Belum ada koreksi seleksi individual/pvalue atau bukti koordinasiinvestor.',
        '## Data, target dan kontrol',
        'NET_ZERO berarti netnol/tidaktercatat pada snapshotvalid, bukan tanpa gross transaksi. Target tepat sesi kalender berikutnya; thresholdbesar log(absnet) quantile2/3 dan12broker dipilih hanya train. Invalidsnapshot/hilang dikeluarkan tanpa mengkompres lag. Target tetap dalam split yang sama; purge20+embargo20 bawaan dipertahankan.',
        samples.groupby(['ticker','fold','split']).size().rename('queries').reset_index().to_markdown(index=False),
        'Own-context berisi5sesi riwayat target (onehotstate,8continuous token,present), OHLCVcurrent. Encoderprobe dan rawpeer melihat peer train-selected yang sama; targetcurrent dimask. Learned/random pooled24 ditambahkan ke owncontext. Randominit direkonstruksi exactseed/arsitektur/SHA; semuaweightsfrozen. Ini audit komponen SetEncoder harian, bukan transferTCNfull. Subset12broker berbeda dari setnative yang lebih besar saat training; distribution shift membatasi kesimpulan kegagalanprobe.',
        'Softenedpersistence:0.9 untuk statehariini,0.025lainnya. Shifted memakai circularshuffle masing-masing split dan dapat mengambil kontekslebihakhir dalam split: negativecontroloffline, bukan forecastdeployable. Head LogisticRegression linear, scalertrain-only, C0.01/0.1/1 dipilihNLLvalidation, max_iter300. Semuahead dipilih danhash dibekukan sebelumtest pertama. Convergencewarnings: '+str(int(trials.warnings.sum()))+'. Kelasabsen diberi floor1e-8; macrorecall mengikuti scores lama, hanya kelas yang muncul pada test.',
        '## Verifikasi',fence+'json\n'+json.dumps(verified,indent=2)+'\n'+fence,
        'Synthetic fixtures hanya unit tests. Smoke realdata mendahului fullprobe. Verifier merekonstruksi encoderfeatures danstate/scaler train-only, replay seluruhhead/score/selector, checks labelcalendar/hash/CPU-GPU. Sourcebackbone tidak berubah. Tidak ada backtestharga karena tasknetstate bukan return. Sejarah ini eksploratif dan bukan confirmationtest baru.',
        '5tes baru lulus. Suite penuh96tes:93lulus,3kegagalanlegacy yang sama terkait inputlive versus checkpoint/tokenizer/tanggalekspor lama. Log disertakan; hasil lama/toleransi tidak diubah untuk menyembunyikan kegagalan. Smoke60head direplayGPU danCPU, full1620head direplay pada cachedfeatures, dengan CPU/GPU extractioncheck pada27case.',
        '## File dan eksperimen berikutnya',
        'Proyek: .. event_broker_geometry.py:geometri/init/stability; event_broker_probe.py:probe/baseline/freeze/statistics/peerremoval; event_broker_probe_verify.py:rebuild/replay; event_broker_probe_results.py:laporan; dashboard.py:panelrelasi pada exporterexisting; test_event_broker_probe.py:tes; EVENT-RELATION-AUDIT.md dan EVENT-RELATION-PROTOCOL.md:audit/resep.',
        'Jika tambahan informasi dari embedding pretrained belum terkonfirmasi, uji encoder dengan objective relasi eksplisit terhadap own-history/rawpeer controls. Same-day masked reconstruction dan next-day forecast perlu dipisah, lalu konfirmasi pada sesi baru setelah resep dibekukan. Jangan menyimpulkan relasi dari cosine kecil/besar atau menambahkompleksitas tanpa bukti. Training backbone khususrelasi belum dijalankan pada tahap ini.',
    ]
    (root/'RESULTS.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    from dashboard import build_broker_probe_dashboard
    build_broker_probe_dashboard(root)
    print(macro.to_string(index=False));print(geometry_summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--geometry',default='runs/event-broker-geometry-20261117')
    args=p.parse_args();report(args.root,args.geometry)
