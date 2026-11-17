"""Export self-contained model selector and static overview; never pools embeddings."""
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def build_dashboard(root, tickers):
    root=Path(root)
    figure=go.Figure();buttons=[]
    projections=[]
    for i,ticker in enumerate(tickers):
        data=json.loads((root/ticker/'projection.json').read_text())
        projections.append(data)
        p=np.array(data['train_xyz']);q=data['query_xyz']
        selected=np.array([j for j,day in enumerate(data['train_dates']) if day in data['neighbors_dates']],dtype=int)
        visible=i==0
        figure.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',
            marker=dict(size=4,color='#94a3b8',opacity=.6),name=f'Riwayat {ticker}',
            text=[f'{ticker} | {day}' for day in data['train_dates']],hovertemplate='%{text}<extra>Training</extra>',visible=visible))
        figure.add_trace(go.Scatter3d(x=p[selected,0],y=p[selected,1],z=p[selected,2],mode='markers',
            marker=dict(size=7,color='#d97706'),name=f'Pola terdekat {ticker}',
            text=[f'{ticker} | {data["train_dates"][j]}' for j in selected],hovertemplate='%{text}<extra>Training</extra>',visible=visible))
        figure.add_trace(go.Scatter3d(x=[q[0]],y=[q[1]],z=[q[2]],mode='markers',
            marker=dict(size=10,color='#2563eb',symbol='diamond'),name=f'{ticker} terbaru',
            hovertemplate=f'{ticker} | {data["latest_date"]}<extra>Terbaru</extra>',visible=visible))
        title=f'{ticker} — model khusus {ticker}<br><sup>OHLCV + broker {ticker} · {len(p)} pola train · PCA 3D {100*data["variance_explained"]:.1f}% variasi</sup>'
        if data.get('selected_features'):
            title=f'{ticker} — model dengan fitur terpilih<br><sup>{len(data["selected_features"])} fitur numerik · {len(p)} pola train · PCA 3D {100*data["variance_explained"]:.1f}% variasi</sup>'
        available=all(data['predictions_arithmetic_return'].get(str(h)) is not None for h in [5,10,20])
        pred=(' · '.join(f'{h} sesi {100*data["predictions_arithmetic_return"][str(h)]:+.2f}%' for h in [5,10,20])
              if available else 'tidak tersedia: input OHLCV terbaru tidak valid')
        annotations=[dict(text='Drag: putar 3D · Scroll: zoom · Pilih saham di menu atas',x=.5,y=-.07,xref='paper',yref='paper',showarrow=False),
                     dict(text='Prediksi eksperimental '+ticker+': '+pred,x=.5,y=-.13,xref='paper',yref='paper',showarrow=False),
                     dict(text='Setiap saham memakai model dan PCA sendiri. Sumbu antarsaham tidak sebanding.',x=.5,y=-.19,xref='paper',yref='paper',showarrow=False)]
        features=data.get('latest_features',{})
        if features.get('price_return_40') is not None:
            def fmt(value,digits=2):
                return 'tidak tersedia' if value is None else f'{value:,.{digits}f}'
            label=(f"Retail {fmt(features.get('summary_retail_raw'),0)} | Broker net buy {fmt(features.get('broker_buy_count'),0)} / net sell {fmt(features.get('broker_sell_count'),0)}"
                   f" | Ratio {fmt(features.get('buyer_seller_ratio'),3)}"
                   f"<br>Perubahan harga: 20 sesi {100*features['price_return_20']:+.2f}% | 40 sesi {100*features['price_return_40']:+.2f}%")
            annotations.append(dict(text=label,x=.5,y=-.28,xref='paper',yref='paper',showarrow=False,font=dict(size=12)))
        layout=dict(title=dict(text=title,x=.025),annotations=annotations,
                    scene=dict(xaxis=dict(autorange=True,title='PC1'),yaxis=dict(autorange=True,title='PC2'),zaxis=dict(autorange=True,title='PC3'),
                               camera=dict(eye=dict(x=1.4,y=1.4,z=1)),aspectmode='data',dragmode='orbit'))
        buttons.append(dict(label=ticker,method='update',args=[{'visible':[j//3==i for j in range(len(tickers)*3)]},layout]))
    figure.update_layout(template='plotly_white',height=860,margin=dict(l=10,r=10,t=140,b=210),
                          legend=dict(orientation='h',x=.2,y=1.08),
                          updatemenus=[dict(buttons=buttons,direction='down',x=0,y=1.12,xanchor='left',yanchor='top')],
                          **buttons[0]['args'][1])
    figure.write_html(root/'index.html',include_plotlyjs=True,full_html=True,
                      config=dict(responsive=True,scrollZoom=True,displaylogo=False))
    (root/'dashboard-config.json').write_text(figure.to_json(),encoding='utf-8')
    make_preview(root,projections)


def make_preview(root, projections):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    panel_count=len(projections)+1
    nrows=(panel_count+2)//3
    fig=plt.figure(figsize=(14,4*nrows+1),facecolor='#f8fafc')
    fig.suptitle(f'{len(projections)} saham · {len(projections)} model terpisah',x=.05,y=.985,ha='left',fontsize=22,fontweight='bold',color='#182333')
    fig.text(.05,.96,'OHLCV + broker harian masing-masing saham | Data s/d '+projections[0]['latest_date'],fontsize=12,color='#475569')
    for i,d in enumerate(projections):
        ax=fig.add_subplot(nrows,3,i+1,projection='3d',facecolor='#f8fafc')
        p=np.array(d['train_xyz']);q=d['query_xyz']
        select=np.array([j for j,date in enumerate(d['train_dates']) if date in d['neighbors_dates']],dtype=int)
        ax.scatter(*p.T,s=14,color='#94a3b8',alpha=.6,depthshade=False)
        ax.scatter(*p[select].T,s=42,color='#d97706',depthshade=False)
        if all(v is not None for v in q):
            ax.scatter(*q,s=85,color='#2563eb',marker='D',depthshade=False)
        ax.set_title(f'{d["ticker"]} | PCA {100*d["variance_explained"]:.1f}%',loc='left',pad=7,fontsize=13,fontweight='bold')
        ax.set(xlabel='PC1',ylabel='PC2',zlabel='PC3');ax.tick_params(labelsize=8)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.set_major_locator(MaxNLocator(3))
        ax.view_init(elev=22,azim=35)
    ax=fig.add_subplot(nrows,3,len(projections)+1);ax.axis('off')
    ax.text(.03,.85,'Setiap panel berdiri sendiri',fontsize=14,fontweight='bold',color='#182333')
    ax.text(.03,.57,'Satu saham, satu model\nFitur dan scaler mandiri\nPCA fit hanya pada train\nEmbedding asli: 32 dimensi',linespacing=1.8,color='#475569')
    handles=[Line2D([],[],marker='o',linestyle='',color='#94a3b8',label='Riwayat saham yang sama'),
             Line2D([],[],marker='o',linestyle='',color='#d97706',label='Pola historis terdekat'),
             Line2D([],[],marker='D',linestyle='',color='#2563eb',label='Kondisi terbaru')]
    ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(.01,.13),frameon=False)
    fig.text(.05,.012,'PCA dan sumbu tiap panel berbeda. Jarak antarpanel tidak dapat dibandingkan. Hasil prediksi masih eksperimental.',fontsize=11,color='#9a3412')
    fig.subplots_adjust(left=.04,right=.95,top=.925,bottom=.035,hspace=.30,wspace=.18)
    fig.savefig(root/'overview.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)


def build_event_dashboard(root):
    """Extend the existing Plotly exporter with event-research panels.

    Reads only exported results. Training writes progress.json independently;
    serve this run directory over localhost for live polling, or open offline
    for the last exported snapshot. No independent GUI application/service.
    """
    import html
    import pandas as pd
    from plotly.offline import get_plotlyjs
    root = Path(root)
    def read(name):
        path = root/f'{name}.csv'
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if 'input_arm' in frame and 'backbone' in frame:
            labels=frame['backbone']+' / '+frame['input_arm']
            frame['model'] = labels.where(frame['backbone']!='baseline',frame.get('model',labels))
        elif 'training_objective' in frame and 'backbone' in frame:
            frame['model'] = frame['backbone']+' / '+frame['training_objective']
        return frame
    panels = []
    def panel(title, figure):
        figure.update_layout(template='plotly_white', height=440, margin=dict(t=75, b=90), title=title)
        panels.append(figure.to_html(full_html=False, include_plotlyjs=False))
    def grouped_plot(frame, x, y, title, group='model', kind='line', extra=None):
        if frame.empty or y not in frame:
            return
        fig = go.Figure(); visibility = []; stocks = sorted(frame.ticker.unique())
        for ticker in stocks:
            part = frame[frame.ticker == ticker]
            by = [group]+([extra] if extra else [])
            for key, chunk in part.groupby(by, dropna=False):
                chunk = chunk.groupby(x, as_index=False)[y].mean().sort_values(x)
                label = ' / '.join(map(str, key)) if isinstance(key, tuple) else str(key)
                kwargs = dict(x=chunk[x], y=chunk[y], name=f'{ticker} {label}', visible=ticker == stocks[0])
                fig.add_trace(go.Bar(**kwargs) if kind == 'bar' else go.Scatter(mode='lines+markers', **kwargs))
                visibility.append(ticker)
        fig.update_layout(updatemenus=[dict(buttons=[dict(label=t, method='update',
            args=[{'visible': [v == t for v in visibility]}, {'title': f'{title} — {t}'}]) for t in stocks], x=1, y=1.2)],
            xaxis_title=x, yaxis_title=y)
        panel(title, fig)
    grouped_plot(read('history'), 'epoch', 'train_loss', 'Training loss (rata-rata fit)')
    grouped_plot(read('history'), 'epoch', 'val_loss', 'Validation loss / checkpoint objective')
    history=read('history')
    is_market='input_arm' in history
    if 'training_objective' in history:
        grouped_plot(history, 'epoch', 'gradient_norm_mean', 'Norm gradient sebelum clipping (mean batch)')
        grouped_plot(history, 'epoch', 'clipped_fraction', 'Fraksi minibatch yang terkena gradient clipping')
        paired=read('paired_summary')
        grouped_plot(paired.assign(model='local MAE − market MAE' if is_market else 'multitask MAE − return-only MAE') if not paired.empty else paired,
                     'horizon','gain_mae',('Konteks pasar: gain vs input lokal (positif = lebih baik)' if is_market else
                         'Objective ablation: gain return-only vs multitask (positif = lebih baik)'),kind='bar')
    gates = read('gates')
    if not gates.empty:
        grouped_plot(gates, 'epoch', 'mean', 'Feature gate evolution', extra='group')
    metric = read('metrics')
    grouped_plot(metric, 'horizon', 'mae', 'Model comparison: log-return MAE', kind='bar')
    grouped_plot(metric, 'horizon', 'brier', 'Probability Brier score (lebih kecil lebih baik)', kind='bar')
    grouped_plot(metric, 'horizon', 'vol_mae', 'Future annualized realized volatility MAE', kind='bar')
    importance = read('importance')
    if not importance.empty:
        grouped_plot(importance[(importance.method == 'temporal_permutation') & (importance.regime == 'all')],
            'group', 'value', 'Grouped temporal permutation: delta validation loss', kind='bar')
    ablation = read('ablation')
    grouped_plot(ablation, 'group', 'delta_loss', 'Refit ablation: delta validation loss', kind='bar')
    curve = read('equity')
    if not curve.empty:
        grouped_plot(curve[curve.cost_scenario == 'base'], 'date', 'equity', 'Walk-forward next-open equity (mean seeds)', extra='horizon')
    grouped_plot(read('calibration'), 'probability', 'positive_rate', 'Reliability bins (small samples; uncalibrated outputs)', extra='horizon')
    detailed=read('diagnostics/by_stock_horizon')
    if not detailed.empty:
        matrix=detailed.pivot(index=['ticker','horizon'],columns='model',values='mae')
        gain=matrix.rsub(matrix['zero'],axis=0).div(matrix['zero'].replace(0,np.nan),axis=0)*100
        values=gain.to_numpy();finite=values[np.isfinite(values)]
        limit=max(1.,float(np.max(abs(finite)))) if len(finite) else 1.
        figure=go.Figure(go.Heatmap(z=values,x=matrix.columns.tolist(),
            y=[f'{ticker} / {horizon} sesi' for ticker,horizon in matrix.index],
            customdata=matrix.to_numpy(),colorscale='RdBu',zmid=0,zmin=-limit,zmax=limit,
            colorbar=dict(title='Gain MAE %'),
            hovertemplate='%{y}<br>%{x}<br>Gain MAE: %{z:.2f}%<br>MAE: %{customdata:.5f}<extra></extra>'))
        panel('Diagnosis 9 kombinasi saham/horizon: gain MAE vs nol (positif = error lebih kecil)',figure)
    tables = []
    for name, title in [('comparison', 'Perbandingan model/horizon'), ('training', 'Training progress dan memori'),
                        ('macro_summary','Objective study: macro seluruh saham/horizon'),
                        ('by_stock_horizon','Objective study: metrik per saham/horizon'),
                        ('paired_summary','Return-only vs multitask: stabilitas fold/seed'),
                        ('paired_bands','Paired temporal bootstrap: simultaneous bands'),
                        ('pair_proof','Bukti bobot awal dan input identik setiap pasangan'),
                        ('ablation', 'Ablation validasi'), ('importance', 'Kontribusi per saham dan rezim volatilitas saham'),
                        ('backtest', 'Backtest biaya dasar dan stress'), ('regime_metrics', 'Metrik per saham/rezim'),
                        ('diagnostics/nine_cells','Diagnosis per saham dan horizon (minimum point, bukan seleksi produksi)'),
                        ('diagnostics/by_stock_horizon','Seluruh metrik per saham dan horizon'),
                        ('diagnostics/fold_seed_stability','Stabilitas fold/seed terhadap baseline'),
                        ('diagnostics/validation_head_components','Komponen objective di validation checkpoint'),
                        ('diagnostics/largest_errors','Audit error terbesar (flag harga, bukan corporate action terverifikasi)'),
                        ('diagnostics/tail_concentration','Konsentrasi error pada tail return')]:
        frame = read(name)
        if not frame.empty:
            tables.append(f'<details><summary>{html.escape(title)} ({len(frame)} baris)</summary>'
                          +frame.round(6).to_html(index=False, classes='data')+'</details>')
    progress = root/'progress.json'
    status = progress.read_text(encoding='utf-8') if progress.exists() else '{}'
    page = '''<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Broker Event Embedding — Research</title><style>
body{font-family:system-ui;margin:24px;background:#f5f7fb;color:#18243b}main{max-width:1440px;margin:auto}
section,details{background:white;padding:18px;border-radius:12px;margin:16px 0;overflow:auto}h1{margin-bottom:8px}
.warning{border-left:5px solid #d28d14;background:#fff6dd;padding:16px}td,th{padding:7px;white-space:nowrap}
pre{white-space:pre-wrap}summary{cursor:pointer;font-weight:bold}a{color:#2459ad}</style><main>
<h1>Broker Event Embedding</h1><a href="../adaptive-v1/latest/index.html">Buka peta embedding CNN adaptif yang sudah ada</a>
<p>Model dilatih terpisah per saham. Tabel gabungan merangkum metrik; bobot model tidak digabung.</p>
<p class="warning">Eksploratif: sejarah sudah pernah dipakai untuk pengembangan. Label overlap dan seed bukan tambahan tanggal independen.
Gate kecil bukan bukti noise. Importance dan ablation memakai validasi; tidak ada fitur dihapus permanen.
IHSG, USD/IDR dan sektor tidak tersedia. Rezim di sini adalah volatilitas saham, bukan rezim pasar.
Probabilitas belum dikalibrasi; backtest memakai asumsi biaya, slippage dan ketersediaan EOD, tanpa verifikasi corporate action.</p>
<section><h2>Training progress</h2><pre id="progress">PROGRESS</pre><p>Offline: snapshot terakhir.
Untuk polling langsung: jalankan <code>python -m http.server 8765 --bind 127.0.0.1</code> di direktori run, lalu buka localhost:8765.</p></section>
<script>PLOTLY</script>PANELS TABLES
<script>if(location.protocol.startsWith('http'))setInterval(async()=>{try{let r=await fetch('progress.json',{cache:'no-store'});
document.getElementById('progress').textContent=JSON.stringify(await r.json(),null,2)}catch(e){}},2000)</script></main></html>'''
    page = page.replace('PROGRESS', html.escape(status)).replace('PLOTLY', get_plotlyjs()).replace('PANELS', ''.join(panels)).replace('TABLES', ''.join(tables))
    if 'training_objective' in history and not is_market:
        page=page.replace('<h1>Broker Event Embedding</h1>',
            '<h1>Broker Event Embedding — Objective Ablation</h1><p class="warning">'
            'Return-only dan multitask dipilih memakai MAE return validasi yang sama. '
            'Training loss kedua objective berbeda skala/komponen dan tidak sebanding langsung. '
            'Head probabilitas dan volatilitas return-only tidak dilatih dan tidak digunakan. '
            'Run ini tidak menghasilkan backtest strategi; tabel ablation adalah eksperimen objective, bukan penghapusan fitur.</p>')
    if is_market:
        page=page.replace('<h1>Broker Event Embedding</h1>','<h1>Broker Event Embedding — Konteks Pasar Yahoo</h1>')
        page=page.replace('IHSG, USD/IDR dan sektor tidak tersedia. Rezim di sini adalah volatilitas saham, bukan rezim pasar.',
            'IHSG dan USD/IDR memakai lag konservatif D+1 pukul12:00WIB. Sektor tidak lolos audit sejarah. '
            'Importance memakai MAE return validasi; rezim importance berdasarkan volatilitas IHSG yang tersedia. '
            'Gate tetap1 karena backbone tanpa gating. Baseline LightGBM lama tidak memakai tambahan fitur pasar.')
    (root/'index.html').write_text(page, encoding='utf-8')


def build_event_relation_dashboard(root):
    """Direct relation-objective experiment in the existing Plotly exporter."""
    import html
    import pandas as pd
    from plotly.offline import get_plotlyjs
    root=Path(root);panels=[];tables=[]
    def read(name):
        path=root/name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    def stock_plot(frame,x,y,title,kind='line'):
        if frame.empty or y not in frame:return
        figure=go.Figure();visible=[];stocks=sorted(frame.ticker.unique())
        for ticker in stocks:
            for method,part in frame[frame.ticker==ticker].groupby('method'):
                part=part.groupby(x,as_index=False)[y].mean().sort_values(x)
                opts=dict(x=part[x],y=part[y],name=method,visible=ticker==stocks[0])
                figure.add_trace(go.Bar(**opts) if kind=='bar' else go.Scatter(mode='lines+markers',**opts))
                visible.append(ticker)
        figure.update_layout(template='plotly_white',height=460,title=title,margin=dict(t=80,b=120),
            updatemenus=[dict(buttons=[dict(label=t,method='update',args=[{'visible':[s==t for s in visible]},
                {'title':title+' — '+t}]) for t in stocks],x=1,y=1.2)])
        panels.append(figure.to_html(full_html=False,include_plotlyjs=False))
    history=read('history.csv')
    stock_plot(history,'epoch','train_nll','Training NLL (mean fit yang mencapai epoch)')
    stock_plot(history,'epoch','validation_nll','Validation NLL / checkpoint selection')
    stock_plot(history,'epoch','gradient_norm_mean','Gradient norm sebelum clipping')
    comparison=read('comparison.csv')
    stock_plot(comparison,'method','nll','Test next-session net-state NLL (lebih kecil lebih baik)',kind='bar')
    stock_plot(comparison,'method','brier','Brier 5-class',kind='bar')
    stock_plot(comparison,'method','macro_state_recall','Macro state recall',kind='bar')
    peer=read('peer_removal_validation.csv')
    if not peer.empty:
        figure=go.Figure();stocks=sorted(peer.ticker.unique())
        for ticker in stocks:
            matrix=peer[peer.ticker==ticker].pivot_table(index='peer',columns='broker',values='mean_delta_nll',aggfunc='mean')
            figure.add_trace(go.Heatmap(z=matrix.to_numpy(),x=matrix.columns,y=matrix.index,colorscale='RdBu',zmid=0,
                visible=ticker==stocks[0],hovertemplate='%{y} → %{x}<br>Delta NLL: %{z:.6f}<extra></extra>'))
        figure.update_layout(template='plotly_white',height=560,title='Validation peer removal: positif = membantu prediksi model',
            updatemenus=[dict(buttons=[dict(label=t,method='update',args=[{'visible':[s==t for s in stocks]}]) for t in stocks],x=1,y=1.15)])
        panels.append(figure.to_html(full_html=False,include_plotlyjs=False))
    for name,title in [('comparison.csv','Metrik per saham'),('training.csv','Training, memori dan perubahan encoder'),
        ('paired_intervals.csv','Interval paired date-block simultan'),('fold_seed_stability.csv','Stabilitas fold/seed'),
        ('initial_pairs.csv','Inisialisasi berpasangan'),('peer_removal_validation.csv','Semua peer-removal validasi')]:
        table=read(name)
        if not table.empty:tables.append('<details><summary>'+html.escape(title)+f' ({len(table)} baris)</summary>'+
            table.round(6).to_html(index=False)+'</details>')
    status=(root/'progress.json').read_text(encoding='utf-8') if (root/'progress.json').exists() else '{}'
    page='''<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Relasi Broker — Objective Langsung</title><style>
body{font-family:system-ui;margin:24px;background:#f5f7fb;color:#18243b}main{max-width:1440px;margin:auto}
section,details{background:white;padding:18px;border-radius:12px;margin:16px 0;overflow:auto}td,th{padding:6px;white-space:nowrap}
summary{cursor:pointer;font-weight:bold}.note{padding:16px;background:#fff6dd;border-left:5px solid #d28d14}pre{white-space:pre-wrap}</style>
<main><h1>Relasi Broker — Training Objective Langsung</h1>
<p>Target: net-state broker tepat sesi berikutnya. Model terpisah per saham; Set Transformer dan TCN dilatih untuk target broker.</p>
<p class="note">Historis eksploratif. Bandingkan relational dengan own_only dan frozen_peers, bukan hanya baseline frekuensi.
NET_ZERO bukan tanpa transaksi gross. Shifted peers adalah kontrol offline, bukan prediktor live.
Peer-removal validasi bukan efek kausal/koordinasi atau relasi terkonfirmasi. Ini bukan prediksi return/backtest saham.</p>
<section><h2>Progress</h2><pre id="progress">STATUS</pre><p>File offline menampilkan snapshot; HTTP lokal dapat polling progress.</p></section>
<script>LIBRARY</script>CHARTS TABLES
<script>if(location.protocol.startsWith('http'))setInterval(async()=>{try{let r=await fetch('progress.json',{cache:'no-store'});
document.getElementById('progress').textContent=JSON.stringify(await r.json(),null,2)}catch(e){}},2000)</script></main></html>'''
    page=page.replace('STATUS',html.escape(status)).replace('LIBRARY',get_plotlyjs()).replace('CHARTS',''.join(panels)).replace('TABLES',''.join(tables))
    (root/'index.html').write_text(page,encoding='utf-8')


def build_broker_probe_dashboard(root):
    """Broker relation panels in the existing offline Plotly exporter."""
    import html
    import pandas as pd
    from plotly.offline import get_plotlyjs
    root=Path(root);panels=[];tables=[]
    def read(name):return pd.read_csv(root/name)
    def panel(title,fig):
        fig.update_layout(template='plotly_white',title=title,height=480,margin=dict(t=90,b=120))
        panels.append(fig.to_html(full_html=False,include_plotlyjs=False))
    def bars(frame,value,title,group='method'):
        fig=go.Figure();visibility=[];stocks=sorted(frame.ticker.unique())
        for ticker in stocks:
            part=frame[frame.ticker==ticker].groupby(group)[value].mean()
            fig.add_trace(go.Bar(x=part.index,y=part.values,name=ticker,visible=ticker==stocks[0]))
            visibility.append(ticker)
        fig.update_layout(updatemenus=[dict(buttons=[dict(label=t,method='update',
            args=[{'visible':[v==t for v in visibility]},{'title':title+' — '+t}]) for t in stocks],x=1,y=1.18)])
        panel(title,fig)
    metric=read('comparison.csv')
    bars(metric,'nll','Prediksi net-state sesi berikutnya: NLL (lebih kecil lebih baik)')
    bars(metric,'macro_state_recall','Macro state recall (kelas yang muncul pada test)')
    geo=read('geometry/geometry_summary.csv')
    bars(geo,'identity_relative_l2_movement','Perubahan broker-ID terhadap inisialisasi acak',group='seed')
    stable=read('geometry/stability.csv')
    stable=stable[stable.comparison=='across_seed_same_fold']
    bars(stable,'mean_top3_jaccard','Top3-neighbor Jaccard antar-seed pada fold sama',group='geometry')
    for filename,x,y,value,title in [
        ('geometry/all_pairs.csv','broker_b','broker_a','contextual_centroid','Cosine centroid native-context (train only)'),
        ('peer_candidates.csv','broker','peer','mean_delta_nll','Penghapusan peer pada validasi: delta NLL (positif = membantu)')
    ]:
        frame=read(filename)
        if 'broker_a' in frame:
            mirror=frame.rename(columns={'broker_a':'broker_b','broker_b':'broker_a'})
            frame=pd.concat([frame,mirror],ignore_index=True)
        fig=go.Figure();stocks=sorted(frame.ticker.unique())
        for ticker in stocks:
            matrix=frame[frame.ticker==ticker].pivot_table(index=y,columns=x,values=value,aggfunc='mean')
            fig.add_trace(go.Heatmap(z=matrix.to_numpy(),x=matrix.columns,y=matrix.index,zmid=0,
                colorscale='RdBu',visible=ticker==stocks[0],hovertemplate='%{y} → %{x}<br>%{z:.6f}<extra></extra>'))
        fig.update_layout(updatemenus=[dict(buttons=[dict(label=t,method='update',
            args=[{'visible':[s==t for s in stocks]},{'title':title+' — '+t}]) for t in stocks],x=1,y=1.18)])
        panel(title,fig)
    for filename,title in [('comparison.csv','Seluruh metrik per saham'),
        ('paired_intervals.csv','Paired date-block simultaneous intervals'),
        ('fold_seed_stability.csv','Konsistensi fold dan seed'),
        ('peer_candidates.csv','Semua kandidat peer, validasi saja'),
        ('geometry/geometry_summary.csv','Audit geometri versus exact initialization'),
        ('geometry/stability.csv','Kestabilan geometri pada common broker universe')]:
        data=read(filename)
        tables.append('<details><summary>'+html.escape(title)+f' ({len(data)} baris)</summary>'+
                      data.round(6).to_html(index=False)+'</details>')
    progress=(root/'progress.json').read_text(encoding='utf-8')
    page='''<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Relasi Broker — Frozen Encoder Probe</title><style>
body{font-family:system-ui;margin:24px;background:#f5f7fb;color:#18243b}main{max-width:1440px;margin:auto}
section,details{background:white;padding:18px;border-radius:12px;margin:16px 0;overflow:auto}
td,th{padding:6px;white-space:nowrap}summary{cursor:pointer;font-weight:bold}
.note{background:#fff6dd;padding:16px;border-left:5px solid #d28d14}pre{white-space:pre-wrap}</style><main>
<h1>Relasi Broker — Frozen Encoder Probe</h1>
<p>Model terpisah per saham. Target: net-state tepat sesi berikutnya, bukan return harga.
Backbone dibekukan; head linear dipilih hanya dengan validasi.</p>
<p class="note">Eksploratif. Cosine, coactivity, dan kontribusi prediktif berbeda.
NET_ZERO berarti net nol/tidak tercatat pada snapshot valid, bukan tidak ada transaksi gross.
Peer-removal adalah perturbasi validasi, bukan efek kausal atau bukti koordinasi investor.
Kontrol shifted bersifat offline. Probe memakai subset broker dan komponen set harian; TCN tidak diuji oleh probe ini.</p>
<section><h2>Status</h2><pre>STATUS</pre></section>
<script>LIBRARY</script>CHARTS TABLES</main></html>'''
    page=page.replace('STATUS',html.escape(progress)).replace('LIBRARY',get_plotlyjs()).replace('CHARTS',''.join(panels)).replace('TABLES',''.join(tables))
    (root/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    build_dashboard('runs/individual',['ADRO','BBCA','BBRI','BMRI','TLKM'])
