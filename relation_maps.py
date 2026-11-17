"""Independent per-stock maps and simultaneous cross-stock evidence intervals."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from broker_tokens import STATES
from relation_io import read_predictions

TASKS = ['same_day_masked', 'next_day_peers']
CONTROLS = ['frequency', 'ohlcv_only', 'shifted_peers', 'fixed_embedding']
LABELS = {'didukung':'Didukung pada sejarah ini', 'indikasi':'Indikasi eksploratif', 'belum':'Belum didukung'}


def simultaneous_intervals(values, fold_ids, seed=93017, repeats=20000, block=5):
    """Synchronized date-block bootstrap, max standardized deviation over ALL
    stocks x tasks x controls. NaNs retain missing-date/stock coverage.

    Intervals are conditional on fitted models, approximate, not exact tests.
    """
    rng = np.random.default_rng(seed)
    means = np.nanmean(values, axis=0)
    groups = [np.flatnonzero(fold_ids == f) for f in np.unique(fold_ids)]
    samples = np.empty((repeats, values.shape[1]))
    for start in range(0, repeats, 128):
        count = min(128, repeats-start); index = []
        for ids in groups:
            starts = rng.integers(0, len(ids), (count, int(np.ceil(len(ids)/block))))
            local = ((starts[:,:,None]+np.arange(block)) % len(ids)).reshape(count,-1)[:, :len(ids)]
            index.append(ids[local])
        sample_ids = np.concatenate(index, axis=1)
        samples[start:start+count] = np.nanmean(values[sample_ids], axis=1)
    std = samples.std(axis=0, ddof=1)
    safe = np.where(std > 1e-12, std, 1)
    maximum = np.max(np.abs((samples-means)/safe), axis=1)
    critical = float(np.quantile(maximum, .95))
    half = critical*std
    return means, means-half, means+half, std, critical


def classify(rows, fold_rows, test_dates):
    """Rules fixed before viewing the expanded stock results."""
    valid = set(zip(rows.comparator,rows.block_sessions)) == {(c,b) for c in CONTROLS for b in [3,5,10]}
    if not valid or test_dates < 60: return 'belum'
    robust = bool((rows.sim_ci_low > 0).all())
    # At least two chronological folds must improve against every control.
    consistent = all((fold_rows[c] > 0).sum() >= 2 for c in CONTROLS)
    if robust and consistent: return 'didukung'
    block5 = rows[rows.block_sessions == 5]
    if (block5.nominal_ci_low > 0).all() and consistent: return 'indikasi'
    return 'belum'


def stable_pairs(neighbors):
    records = []
    for task in TASKS:
        part = neighbors[neighbors.task == task]
        eligibility = {}
        chosen = {}
        for (method, fold, seed), group in part.groupby(['method','fold','seed']):
            key = (method, int(fold), int(seed))
            eligibility[key] = set(group['query'])
            chosen[key] = set(zip(group['query'], group.neighbor))
        candidates = set(zip(part[part.method=='full']['query'], part[part.method=='full'].neighbor))
        for query, neighbor in sorted(candidates):
            counts = {}
            for method in ['full','shifted_peers']:
                byfold = {}
                for key, tokens in eligibility.items():
                    if key[0] != method or query not in tokens or neighbor not in tokens: continue
                    k = key[1]; hits, eligible = byfold.get(k, (0,0))
                    byfold[k] = (hits+int((query,neighbor) in chosen[key]), eligible+1)
                counts[method] = byfold
            full, null = counts['full'], counts['shifted_peers']
            eligible = sum(n for _,n in full.values()); hits = sum(h for h,_ in full.values())
            if not eligible: continue
            nulln = sum(n for _,n in null.values()); nullhits = sum(h for h,_ in null.values())
            support = hits/eligible; control = nullhits/nulln if nulln else np.nan
            minimum = min(h/n for h,n in full.values())
            selected_folds = sum(h > 0 for h,n in full.values())
            stable = (len(full)>=2 and selected_folds>=2 and support>=.6 and minimum>=.4 and
                      np.isfinite(control) and support-control>=.2)
            records.append(dict(task=task, query=query, neighbor=neighbor, seed_support=support,
                shifted_support=control, support_gap=support-control, minimum_fold_support=minimum,
                eligible_folds=len(full), selected_folds=selected_folds, eligible_fits=eligible,
                stable_geometry=stable))
    return pd.DataFrame(records)


def peer_summary(raw):
    fold = raw.groupby(['task','fold','peer','target'],as_index=False).gain_nll.mean()
    summary = fold.groupby(['task','peer','target']).gain_nll.agg(
        mean_gain_nll='mean', folds='count', positive_folds=lambda s:int((s>0).sum()),
        minimum_fold_gain='min', maximum_fold_gain='max').reset_index()
    positive=raw.groupby(['task','peer','target']).gain_nll.apply(lambda s:float((s>0).mean())).rename('seed_positive_fraction').reset_index()
    summary=summary.merge(positive,on=['task','peer','target'],validate='one_to_one')
    summary['stable_contribution']=(summary.folds>=2)&(summary.positive_folds>=2)&(summary.mean_gain_nll>0)&(summary.seed_positive_fraction>=.6)
    return summary


def same_day_cooccurrence(folder,pairs):
    """Descriptive held-out joint counts; not a selected-pair significance test."""
    exported=read_predictions(folder)
    actual=exported[(exported.task=='same_day_masked')&(exported.method=='full')].drop_duplicates(['fold','date','broker'])
    pivots={fold:part.pivot(index='date',columns='broker',values='actual') for fold,part in actual.groupby('fold')}
    records=[]
    for row in pairs.to_dict('records'):
        n=joint=0;expected=0.
        if row['task']=='same_day_masked':
            a,sa=row['query'].split('_',1); b,sb=row['neighbor'].split('_',1)
            for pivot in pivots.values():
                if a not in pivot or b not in pivot:continue
                pair=pivot[[a,b]].dropna(); size=len(pair)
                qa=pair[a].to_numpy()==STATES.index(sa); qb=pair[b].to_numpy()==STATES.index(sb)
                n+=size;joint+=int((qa&qb).sum());expected+=float(qa.sum()*qb.sum()/max(size,1))
        row.update(eligible_test_days=n if n else np.nan,joint_test_days=joint if n else np.nan,
                   expected_joint_days=expected if n else np.nan,cooccurrence_lift=joint/expected if expected>0 else np.nan)
        records.append(row)
    return pd.DataFrame(records)


def map_for_stock(folder, ticker, pairs, peers, stock_evidence):
    checkpoint = torch.load(folder/'fold-3/same_day_masked.pt',map_location='cpu',weights_only=False)
    tokenizer = checkpoint['tokenizer']; codes = tokenizer['brokers']
    vectors = checkpoint['state_dict']['embedding.weight'].numpy()
    # Query presence is the original >=5 training-day exposure filter.
    neighbors = pd.read_csv(folder/'neighbors_by_seed.csv')
    eligible = set(neighbors[(neighbors.task=='same_day_masked')&(neighbors.fold==3)&
                             (neighbors.method=='full')]['query'])
    ids = [i for i,t in enumerate(tokenizer['tokens']) if t in eligible]
    if len(ids)<3: raise ValueError('Too few reliable tokens for a 3D map')
    pca = PCA(n_components=3).fit(vectors[ids]); xyz = pca.transform(vectors)
    figure = go.Figure()
    colors = {'BUY_NORMAL':'#60a5fa','BUY_LARGE':'#1d4ed8','SELL_NORMAL':'#f59e0b','SELL_LARGE':'#dc2626'}
    name_to_id = {t:i for i,t in enumerate(tokenizer['tokens'])}
    for state, color in colors.items():
        chosen = [i for i in ids if tokenizer['tokens'][i].endswith('_'+state)]
        figure.add_trace(go.Scatter3d(x=xyz[chosen,0],y=xyz[chosen,1],z=xyz[chosen,2],mode='markers+text',
            text=[tokenizer['tokens'][i] for i in chosen],textposition='top center',
            textfont=dict(size=10),marker=dict(size=5,color=color),name=state,
            hovertemplate='%{text}<extra></extra>'))
    shown = pairs[(pairs.task=='same_day_masked') & pairs.stable_geometry].sort_values('seed_support',ascending=False)
    edge_x=[]; edge_y=[]; edge_z=[]; seen=set()
    for row in shown.to_dict('records'):
        key=tuple(sorted([row['query'],row['neighbor']]))
        if key in seen or not all(t in eligible for t in key): continue
        seen.add(key); a,b = [name_to_id[t] for t in key]
        for storage,col in [(edge_x,0),(edge_y,1),(edge_z,2)]: storage.extend([xyz[a,col],xyz[b,col],None])
    figure.add_trace(go.Scatter3d(x=edge_x,y=edge_y,z=edge_z,mode='lines',name='Tetangga stabil (eksploratif)',
                                  line=dict(width=3,color='#6b7280'),hoverinfo='skip'))
    figure.update_layout(template='plotly_white',height=660,margin=dict(l=0,r=0,t=85,b=10),
        title=dict(text=f'{ticker} — embedding kejadian broker<br><sup>Fold 3, seed {checkpoint["seed"]} dipilih lewat validasi · PCA 3D {pca.explained_variance_ratio_.sum():.1%} variasi</sup>',x=.02),
        legend=dict(orientation='h',y=1.03),scene=dict(dragmode='orbit',aspectmode='data',
        xaxis_title='PC1',yaxis_title='PC2',zaxis_title='PC3'))
    table = pd.DataFrame({'token':[tokenizer['tokens'][i] for i in ids]})
    for d in range(vectors.shape[1]): table[f'dim_{d+1}'] = vectors[ids,d]
    table.to_csv(folder/'map_vectors.csv',index=False)
    for task in TASKS:
        own = peers[peers.task==task]
        pivot = own.pivot(index='peer',columns='target',values='mean_gain_nll')
        if task=='same_day_masked':
            heat = go.Figure(go.Heatmap(z=pivot.to_numpy(),x=pivot.columns,y=pivot.index,colorscale='RdBu',
                zmid=0,reversescale=True,colorbar=dict(title='Gain NLL'),
                hovertemplate='Peer %{y} → target %{x}<br>Gain saat peer tersedia: %{z:.5f}<extra></extra>'))
            heat.update_layout(template='plotly_white',height=550,title='Kontribusi peer: menghapus broker mana memperburuk model?',
                               xaxis_title='Broker target yang disembunyikan',yaxis_title='Peer yang dihapus')
    limited = shown.head(20)[['query','neighbor','seed_support','minimum_fold_support','eligible_folds','joint_test_days','cooccurrence_lift']]
    limited = limited.rename(columns={'query':'Token','neighbor':'Tetangga','seed_support':'Dukungan seed',
                            'minimum_fold_support':'Minimum per fold','eligible_folds':'Fold tersedia',
                            'joint_test_days':'Hari muncul bersama (test)','cooccurrence_lift':'Lift test'})
    css = 'body{font:15px system-ui;color:#17233b;margin:22px;background:#f6f8fc}section{background:white;padding:18px;border-radius:12px;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;text-align:left;border-bottom:1px solid #e3e8f0}p{line-height:1.6}.muted{color:#536170}h1{margin-bottom:6px}'
    status = ' · '.join(f'{"Hari sama" if t==TASKS[0] else "Sesi berikutnya"}: {LABELS[stock_evidence[t]]}' for t in TASKS)
    content = f'<!doctype html><html lang="id"><meta charset="utf-8"><title>{ticker} — hubungan broker</title><style>{css}</style><h1>{ticker}: peta hubungan kejadian broker</h1><p>{html.escape(status)}</p>'
    content += '<p class="muted">Drag untuk memutar; scroll untuk zoom; klik legenda untuk menyaring. Titik adalah kejadian broker, bukan investor. Garis adalah kedekatan embedding yang stabil menurut aturan eksploratif, bukan bukti koordinasi.</p>'
    content += '<section>'+figure.to_html(full_html=False,include_plotlyjs='../plotly.min.js',config={'scrollZoom':True,'displaylogo':False})+'</section>'
    content += '<section><h2>Tetangga yang stabil lintas seed dan periode</h2><p>Minimal dua fold; dukungan total ≥60%, setiap fold ≥40%, selisih dukungan terhadap kontrol geser ≥20 poin persentase. Dukungan seed bukan probabilitas ekonomi atau p-value.</p>'
    content += limited.to_html(index=False,float_format=lambda v:f'{v:.2f}',border=0) if len(limited) else '<p>Belum ada pasangan yang memenuhi seluruh aturan kestabilan.</p>'
    content += '<p class="muted">Lift membandingkan jumlah muncul bersama dengan harapan dari frekuensi masing-masing dalam setiap fold. Di bawah 1 berarti lebih jarang bersama; tetangga geometris tidak otomatis sering bersama. Angka ini deskriptif, belum dikoreksi seleksi banyak pasangan.</p></section><section>'+heat.to_html(full_html=False,include_plotlyjs=False,config={'displaylogo':False})+'</section>'
    beneficial=peers[(peers.task=='same_day_masked')&peers.stable_contribution].sort_values('mean_gain_nll',ascending=False).head(15)
    columns=['peer','target','mean_gain_nll','positive_folds','folds','seed_positive_fraction']
    content+='<section><h2>Kontribusi peer yang konsisten (eksploratif)</h2><p>Minimal dua fold positif, gain rata-rata positif, dan minimal 60% fit seed/fold membantu. Bukan bukti hubungan kausal; belum ada koreksi seleksi pada pasangan individual.</p>'+beneficial[columns].to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)+'</section>'
    content += '<p class="muted">Heatmap merata-ratakan fold dan seed. Nilai positif berarti penghapusan peer memperburuk NLL. Penghapusan input di luar distribusi training; bukan efek kausal. Kandidat pasangan dipetakan dari sejarah yang sama dan perlu konfirmasi data baru. Grafik 3D adalah proyeksi; tetangga dihitung di 8 dimensi asli.</p>'
    content += '<p><a href="RESULTS.md">Laporan per saham</a> · <a href="stable_pairs.csv">Seluruh pasangan</a> · <a href="peer_stability.csv">Kontribusi peer</a></p></html>'
    (folder/'map.html').write_text(content,encoding='utf-8')


def analyze(root, repeats=20000):
    root = Path(root)
    batch = json.loads((root/'batch_config.json').read_text())
    stocks = [t for t in batch['stocks'] if (root/t/'summary.csv').exists()]
    all_gains=[]; all_nominal=[]; fold_gains={}; date_counts={}; quality=[]
    for ticker in stocks:
        folder=root/ticker
        daily = pd.read_csv(folder/'daily_test_scores.csv')
        nominal = pd.read_csv(folder/'paired_block_bootstrap.csv'); nominal['ticker']=ticker; all_nominal.append(nominal)
        config=json.loads((folder/'config.json').read_text())
        quality.append(dict(ticker=ticker,source_days=config['source_days'],source_quality=config['source_quality']))
        for task, group in daily.groupby('task'):
            pivot = group.pivot(index=['fold','date'],columns='method',values='nll')
            gain = pd.DataFrame({c:pivot[c]-pivot['full'] for c in CONTROLS})
            fold_gains[ticker,task]=gain.groupby(level='fold').mean()
            date_counts[ticker,task]=len(pivot)
            for c in CONTROLS:
                s=gain[c].rename((ticker,task,c)); all_gains.append(s)
    frame=pd.concat(all_gains,axis=1).sort_index()
    results=[]; criticals={}
    for block in [3,5,10]:
        mean, lo, hi, std, critical=simultaneous_intervals(frame.to_numpy(),frame.index.get_level_values('fold').to_numpy(),
                                                          repeats=repeats,block=block)
        criticals[str(block)]=critical
        for i,(ticker,task,control) in enumerate(frame.columns):
            results.append(dict(ticker=ticker,task=task,comparator=control,block_sessions=block,
                                gain_nll=mean[i],sim_ci_low=lo[i],sim_ci_high=hi[i],bootstrap_std=std[i],dates=date_counts[ticker,task]))
    simultaneous=pd.DataFrame(results)
    nominal=pd.concat(all_nominal,ignore_index=True).rename(columns={'ci_low':'nominal_ci_low','ci_high':'nominal_ci_high'})
    simultaneous=simultaneous.merge(nominal[['ticker','task','comparator','block_sessions','nominal_ci_low','nominal_ci_high']],
                                    on=['ticker','task','comparator','block_sessions'],validate='one_to_one')
    simultaneous.to_csv(root/'simultaneous_intervals.csv',index=False)
    summaries=[]; all_pairs=[]
    for ticker in stocks:
        folder=root/ticker; summary=pd.read_csv(folder/'summary.csv')
        stability=pd.read_csv(folder/'neighbor_stability.csv')
        neighbors=pd.read_csv(folder/'neighbors_by_seed.csv')
        pairs=same_day_cooccurrence(folder,stable_pairs(neighbors)); peers=peer_summary(pd.read_csv(folder/'peer_ablation_by_seed.csv'))
        pairs.to_csv(folder/'stable_pairs.csv',index=False); peers.to_csv(folder/'peer_stability.csv',index=False)
        all_pairs.append(pairs.assign(ticker=ticker))
        status={}
        for task in TASKS:
            rows=simultaneous[(simultaneous.ticker==ticker)&(simultaneous.task==task)]
            state=classify(rows,fold_gains[ticker,task],date_counts[ticker,task]); status[task]=state
            score=summary[summary.task==task].set_index('method')
            st=stability[stability.task==task].groupby('method').top3_jaccard.mean()
            row=dict(ticker=ticker,task=task,status=state,test_dates=date_counts[ticker,task],
                     full_nll=score.loc['full','nll'],full_accuracy=score.loc['full','state_accuracy'],
                     frequency_accuracy=score.loc['frequency','state_accuracy'],
                     markov_nll=score.loc['own_previous_markov','nll'],markov_accuracy=score.loc['own_previous_markov','state_accuracy'],
                     gain_vs_markov=score.loc['own_previous_markov','nll']-score.loc['full','nll'],
                     neighbor_jaccard=st['full'],shifted_jaccard=st['shifted_peers'],
                     stable_pairs=int(((pairs.task==task)&pairs.stable_geometry).sum()),
                     stable_peer_links=int(((peers.task==task)&peers.stable_contribution).sum()))
            row['relative_gain_ohlcv_percent']=100*(score.loc['ohlcv_only','nll']-score.loc['full','nll'])/score.loc['ohlcv_only','nll']
            for c in CONTROLS:
                r=rows[(rows.comparator==c)&(rows.block_sessions==5)].iloc[0]
                row[f'gain_{c}']=r.gain_nll
                row[f'sim_low_{c}']=r.sim_ci_low; row[f'sim_high_{c}']=r.sim_ci_high
                row[f'positive_folds_{c}']=int((fold_gains[ticker,task][c]>0).sum())
            summaries.append(row)
        map_for_stock(folder,ticker,pairs,peers,status)
        print('Mapped',ticker,status,flush=True)
    overview=pd.DataFrame(summaries)
    overview.to_csv(root/'relevance_summary.csv',index=False)
    pd.concat(all_pairs,ignore_index=True).to_csv(root/'all_token_pairs.csv',index=False)
    (root/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    config=dict(stocks=stocks,bootstrap_repeats=repeats,simultaneous_family=frame.shape[1],
                critical_values=criticals,block_lengths=[3,5,10],confidence=.95,
                rule='didukung: all 4 controls simultaneous lower CI >0 for all 3 block lengths, >=2 positive folds/control, >=60 test dates; indikasi: all 4 nominal block5 lower CI >0 and same fold criterion',
                family='all completed stocks x both tasks x 4 controls jointly; synchronized date-block resampling',
                stable_geometry='>=2 eligible folds, total seed support>=.6, minimum fold support>=.4, support minus shifted>=.2',
                stable_peer_contribution='>=2 positive folds, >=2 eligible folds, positive mean gain, >=.6 seed/fold positive fraction; descriptive, no pair-selection correction',
                note='Exploratory historical reuse. Intervals conditional on fitted models, not all training uncertainty. No price prediction or profit evidence.',
                source_quality=quality)
    (root/'analysis_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    build_index(root,overview)
    chart(root,overview)
    report(root,overview,config)


def build_index(root,overview):
    first=overview.ticker.iloc[0]; rows=[]
    for ticker in overview.ticker.unique():
        a=overview[(overview.ticker==ticker)&(overview.task==TASKS[0])].iloc[0]
        b=overview[(overview.ticker==ticker)&(overview.task==TASKS[1])].iloc[0]
        rows.append(f'<tr data-ticker="{ticker}"><td><button onclick="selectStock(\'{ticker}\')">{ticker}</button></td><td>{LABELS[a.status]}</td><td>{LABELS[b.status]}</td><td>{a.gain_ohlcv_only:.4f}</td><td>{a.gain_shifted_peers:.4f}</td><td>{a.neighbor_jaccard:.2f}</td><td>{int(a.stable_pairs)}</td></tr>')
    text='''<!doctype html><html lang="id"><meta charset="utf-8"><title>Hubungan broker per saham</title><style>
    body{font:15px system-ui;background:#eef3fa;color:#152039;margin:24px}h1{margin-bottom:6px}.card{padding:20px;background:white;border-radius:14px;margin:18px 0}p{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e5eaf2}tr:hover{background:#f0f7ff}button{cursor:pointer;color:#1557a6;background:none;border:none;font-weight:bold;font-size:14px}iframe{border:0;width:100%;height:1600px;background:white;border-radius:14px}input{padding:10px;border:1px solid #ccd6e5;border-radius:8px} .muted{color:#52647d}
    </style><h1>Apakah embedding broker relevan pada setiap saham?</h1><p>Model dan tokenizer terpisah per saham. Klik saham untuk membuka peta 3D dan kontribusi peer.</p>
    <div class="card"><p><b>Didukung</b>: unggul terhadap empat kontrol dengan interval simultan lintas saham/tugas, tiga panjang blok, dan minimal dua fold positif. <b>Indikasi</b>: lolos interval perbandingan biasa, belum lolos kriteria simultan. <b>Belum didukung</b>: bukti belum memenuhi kriteria; bukan bukti tidak ada hubungan.</p><p class="muted">Hari sama = rekonstruksi setelah sesi selesai. Sesi berikutnya = prediksi state broker, bukan harga saham. Kedekatan dan net buy/sell dapat mencerminkan keseimbangan pasar, bukan koordinasi.</p>
    <input id="search" placeholder="Cari saham..." oninput="filterRows()"><table><thead><tr><th>Saham</th><th>Hari sama</th><th>Sesi berikutnya</th><th>Gain vs OHLCV</th><th>Gain vs geser</th><th>Stabilitas top-3</th><th>Pasangan token stabil</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
    <h2 id="selected">'''+first+'''</h2><p><a id="open" href="'''+first+'''/map.html" target="_blank">Buka peta di tab terpisah</a> · <a href="RESULTS.md">Laporan lengkap</a> · <a href="relevance_summary.csv">Angka per saham</a></p><iframe id="map" src="'''+first+'''/map.html"></iframe>
    <script>function selectStock(t){document.getElementById('map').src=t+'/map.html';document.getElementById('selected').textContent=t;document.getElementById('open').href=t+'/map.html';document.getElementById('selected').scrollIntoView({behavior:'smooth'})}function filterRows(){const q=document.getElementById('search').value.toUpperCase();document.querySelectorAll('tbody tr').forEach(r=>r.style.display=r.dataset.ticker.includes(q)?'':'none')}</script></html>'''
    (root/'index.html').write_text(text,encoding='utf-8')


def chart(root,overview):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=overview[overview.task==TASKS[0]].sort_values('gain_ohlcv_only',ascending=False).ticker.tolist()
    colors={'didukung':'#087f8c','indikasi':'#df9e28','belum':'#8c98aa'}
    fig,axes=plt.subplots(1,2,figsize=(12,11),sharey=True,sharex=True)
    for ax,task,title in zip(axes,TASKS,['Rekonstruksi hari yang sama','Prediksi state sesi berikutnya']):
        part=overview[overview.task==task].set_index('ticker').loc[order]
        for i,(ticker,r) in enumerate(part.iterrows()):
            ax.errorbar(r.gain_ohlcv_only,i,xerr=[[r.gain_ohlcv_only-r.sim_low_ohlcv_only],[r.sim_high_ohlcv_only-r.gain_ohlcv_only]],
                        fmt='o',color=colors[r.status],capsize=2,markersize=4,linewidth=1.2)
        ax.set_yticks(range(len(order)),order); ax.axvline(0,color='#aa4455',linestyle='--',linewidth=1)
        ax.set_title(title,fontsize=12,fontweight='bold'); ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Gain NLL terhadap OHLCV saja\nPositif = embedding full lebih baik')
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(f'{len(order)} saham | Relevansi embedding broker',fontsize=17,fontweight='bold',y=.985)
    fig.text(.5,.018,'Interval simultan 95% seluruh saham × 2 tugas × 4 kontrol; blok 5 sesi\nWarna status juga mempertimbangkan kontrol lain, blok 3/10 sesi dan konsistensi fold.\nHijau: didukung historis · Kuning: indikasi · Abu-abu: belum didukung. Bukan prediksi harga.',ha='center',fontsize=9,color='#536170')
    fig.tight_layout(rect=[.02,.085,.99,.95],w_pad=3)
    fig.savefig(root/'relevance_comparison.png',dpi=160,facecolor='white');plt.close(fig)


def report(root,overview,config):
    lines=['# Relevansi embedding kejadian broker per saham','',
           f"{len(config['stocks'])} saham, model terpisah; tiga fold expanding, lima seed, lima varian embedding dan dua benchmark per tugas. Maksimum 84 tanggal test; cakupan aktual berbeda jika data tidak valid.",'',
           '## Hasil lintas saham','',
           '| Saham | Hari sama | Sesi berikutnya | Gain vs OHLCV hari sama | Gain vs geser hari sama | Pasangan stabil |',
           '|---|---|---|---:|---:|---:|']
    for ticker in overview.ticker.unique():
        a=overview[(overview.ticker==ticker)&(overview.task==TASKS[0])].iloc[0]
        b=overview[(overview.ticker==ticker)&(overview.task==TASKS[1])].iloc[0]
        lines.append(f'| {ticker} | {LABELS[a.status]} | {LABELS[b.status]} | {a.gain_ohlcv_only:.5f} | {a.gain_shifted_peers:.5f} | {int(a.stable_pairs)} |')
    lines += ['', '## Apa arti relevan?', '',
        '- Didukung pada sejarah ini: full mengalahkan frekuensi, OHLCV-only, konteks geser dan embedding acak tetap; semua batas bawah interval simultan positif pada blok 3/5/10 sesi, minimal dua dari tiga fold positif per kontrol, dan >=60 tanggal test.',
        '- Indikasi eksploratif: semua interval nominal blok 5 positif dan memenuhi konsistensi fold, tetapi kriteria simultan belum terpenuhi.',
        '- Belum didukung: kriteria di atas belum terpenuhi. Ini bukan bukti bahwa hubungan tidak ada.', '',
        f"Interval simultan diperoleh dengan bootstrap blok tanggal tersinkronisasi lintas saham dan tugas, {config['bootstrap_repeats']:,} pengambilan ulang; maksimum deviasi terstandar meliputi {config['simultaneous_family']} perbandingan sekaligus. Broker dan seed bukan observasi independen. Ini koreksi perkiraan bersyarat pada model yang telah dilatih, bukan uji exact atau mencakup seluruh ketidakpastian training.", '',
        '## Model per saham', '',
        'Setiap fold memilih hingga 12 broker berdasarkan aktivitas/nilai pada train saham itu sendiri. Token dan batas besar/kecil serta bin konteks di-fit pada train per saham, bukan gabungan lintas saham. Objective dan skema kontrol sama agar hasil bisa dibandingkan. Ruang embedding antar-saham tidak disejajarkan dan koordinat tidak boleh dibandingkan langsung.',
        'Fitur konteks: return 1/20/40 sesi, volume relatif median20, range dan body candle. Retail share dan buyer fraction dikeluarkan agar tidak membocorkan kontribusi target. OHLCV tidak valid dimask di memori sebelum menghitung konteks; sumber asli tidak diubah. NET_ZERO berarti net nol/tidak tercatat, bukan gross tidak aktif.',
        'Markov state broker sendiri dari sesi sebelumnya dilaporkan sebagai benchmark tambahan, tetapi bukan bagian dari empat kontrol utama karena memakai informasi target yang sengaja tidak diberikan pada model peer. Kategori didukung berarti lolos empat kontrol tersebut, bukan mengalahkan semua model alternatif atau membuktikan nilai tambah peer di atas riwayat target sendiri.',
        'State besar memakai nilai net absolut per broker dibanding quantile 2/3 hari aktif train, tanpa denominator pasar. Token broker target selalu dihilangkan, termasuk riwayatnya pada tugas sesi berikutnya. Tugas hari sama memakai informasi sesudah penutupan dan tidak dapat disebut sinyal sebelum penutupan.',
        'Kontrol pergeseran tanggal adalah prosedur offline yang dapat memakai tanggal lebih akhir dalam split; tidak untuk dipakai live. Kontrol vektor acak tetap memiliki lebih sedikit parameter trainable dan dapat dipengaruhi perbedaan optimisasi.', '',
        '## Peta dan kestabilan pasangan', '',
        'Buka index.html, pilih saham. Peta 3D memakai checkpoint full fold terakhir yang dipilih lewat validasi, tidak berdasarkan test. Tetangga dihitung melalui cosine di 8 dimensi asli; PCA hanya visualisasi. Token harus terlihat minimal lima hari training.',
        'Aturan kandidat tetangga stabil ditentukan sebelum analisis hasil 25 saham: tersedia >=2 fold, dipilih pada >=60% seed/fold yang memenuhi syarat, minimum dukungan setiap fold >=40%, selisih dukungan terhadap kontrol geser >=20 poin persentase. Dukungan seed bersifat deskriptif, bukan p-value. Batas token besar berubah sesuai train tiap fold.',
        'Heatmap memperlihatkan kenaikan NLL saat satu peer dihapus. Ini diagnostik input di luar distribusi training dan bukan efek kausal. Daftar pasangan dipilih secara eksploratif dari sejarah yang sama; belum dikonfirmasi pada data baru. Pasangan geometris tidak otomatis berarti sering muncul bersamaan, saling mendahului, atau membantu memprediksi harga.', '',
        'Tabel kontribusi peer menandai kandidat yang membantu pada >=2 fold, rata-rata gain positif, dan >=60% fit seed/fold positif. Ini deskriptif dan belum dikoreksi seleksi banyak pasangan. Jumlah kemunculan bersama dan lift test dihitung untuk pasangan state hari sama; lift memakai frekuensi per fold sebagai pembanding, bukan bukti signifikansi pasangan.', '',
        '## Batas interpretasi', '',
        'Keseimbangan net buy/sell dapat menghasilkan hubungan mekanis. Model tidak mengidentifikasi investor/koordinasi/niat. Hasil hari sama dan sesi berikutnya dinilai terpisah. Tidak ada eksperimen target return harga atau backtest profit pada tahap ini.',
        'Data pendek dan telah digunakan dalam pengembangan terdahulu; ini eksplorasi historis diperluas. Perlu data baru dan beberapa tahun data untuk konfirmasi. ASII/BBNI atau saham lain dapat memiliki cakupan test berbeda karena masalah data; periksa sample_audit.csv dan config.json per saham.', '',
        '## Menjalankan ulang', '', '```powershell','python multi_broker_relations.py','python relation_maps.py','python -m unittest test_broker_relations test_relation_maps -v','```','',
        'Output per saham: tokenizer/checkpoint per fold, metrics_by_seed.csv, test_predictions.csv, stable_pairs.csv, peer_stability.csv, map_vectors.csv dan map.html. Output gabungan: relevance_summary.csv, simultaneous_intervals.csv, all_token_pairs.csv dan relevance_comparison.png.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/broker-relations-25')
    parser.add_argument('--bootstrap-repeats',type=int,default=20000)
    args=parser.parse_args()
    if args.bootstrap_repeats<1000:parser.error('Use at least 1000 bootstrap replicates')
    analyze(Path(args.out),args.bootstrap_repeats)
