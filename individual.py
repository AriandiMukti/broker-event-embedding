"""Train independent stock models with expanded OHLCV, broker and Retail features."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from main import HORIZONS, prepare, train, neighbors
from dashboard import build_dashboard
from feature_audit import audit_run, read_latest_features


def export_view(folder, ticker):
    dataset = np.load(folder/'dataset.npz')
    vectors = np.load(folder/'embeddings.npz')
    meta = pd.read_csv(folder/'samples.csv')
    similar = pd.read_csv(folder/f'{ticker}_neighbors.csv')
    latest = pd.read_csv(folder/'latest_predictions.csv')
    if set(meta.ticker) != {ticker} or set(similar.ticker) != {ticker} or set(latest.ticker) != {ticker}:
        raise ValueError(f'Cross-stock data detected in {folder}')
    reference = (meta.split == 'train').to_numpy()
    # Each PCA is fitted to this stock's own TRAIN embeddings only.
    pca = PCA(n_components=3).fit(vectors['z'][reference])
    xyz = pca.transform(vectors['z'][reference])
    query = pca.transform(vectors['latest_z'])[0]
    refmeta = meta.loc[reference].reset_index(drop=True)
    refy = dataset['y'][reference]
    selected = np.flatnonzero(refmeta.date.isin(similar.date).to_numpy())
    variance = float(pca.explained_variance_ratio_.sum())
    fig = go.Figure()
    actual = np.expm1(refy)*100
    hover = ('%{text}<br>Return historis 5 sesi: %{customdata[0]:+.2f}%'
             '<br>10 sesi: %{customdata[1]:+.2f}%<br>20 sesi: %{customdata[2]:+.2f}%<extra></extra>')
    fig.add_trace(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode='markers',
        marker=dict(size=4, color='#94a3b8', opacity=.55), name=f'Riwayat {ticker}',
        text=[f'{ticker} | {d}' for d in refmeta.date], customdata=actual, hovertemplate=hover))
    fig.add_trace(go.Scatter3d(x=xyz[selected,0], y=xyz[selected,1], z=xyz[selected,2], mode='markers',
        marker=dict(size=7, color='#d97706'), name='Pola terdekat saham ini',
        text=[f'{ticker} | {d}' for d in refmeta.iloc[selected].date], customdata=actual[selected], hovertemplate=hover))
    fig.add_trace(go.Scatter3d(x=[query[0]], y=[query[1]], z=[query[2]], mode='markers',
        marker=dict(size=10, color='#2563eb', symbol='diamond'), name=f'{ticker} terbaru',
        hovertemplate=f'{ticker} | {latest.iloc[0].date}<extra>Embedding terbaru</extra>'))
    predictions = {str(h): float(np.expm1(latest.iloc[0][f'predicted_log_return_{h}'])) for h in HORIZONS}
    predtext = ' · '.join(f'{h} sesi {100*predictions[str(h)]:+.2f}%' for h in HORIZONS)
    fig.update_layout(template='plotly_white', height=690,
        title=dict(text=f'{ticker} — model mandiri<br><sup>Data s/d {latest.iloc[0].date} · PCA 3D: {100*variance:.1f}% variasi · {len(xyz)} pola training {ticker}</sup>', x=.025),
        margin=dict(l=10,r=10,t=100,b=95), legend=dict(orientation='h',x=0,y=1.02),
        scene=dict(xaxis_title='PC1',yaxis_title='PC2',zaxis_title='PC3',dragmode='orbit',aspectmode='data'),
        annotations=[dict(text='Drag untuk memutar · Scroll untuk zoom · Arahkan kursor ke titik untuk detail',x=.5,y=-.07,xref='paper',yref='paper',showarrow=False),
                     dict(text='Prediksi eksperimental: '+predtext,x=.5,y=-.15,xref='paper',yref='paper',showarrow=False)])
    # Store a local Plotly copy beside each chart; no CDN/network.
    fig.write_html(folder/'embedding-3d.html', include_plotlyjs='directory', full_html=True,
                   config=dict(responsive=True, scrollZoom=True, displaylogo=False))
    projection = {'ticker':ticker, 'variance_explained':variance, 'latest_date':str(latest.iloc[0].date),
                  'train_dates':refmeta.date.tolist(), 'train_xyz':xyz.tolist(),
                  'query_xyz':query.tolist(), 'neighbors_dates':similar.date.tolist(),
                  'predictions_arithmetic_return':predictions}
    projection['latest_features']=read_latest_features(folder)
    (folder/'projection.json').write_text(json.dumps(projection,indent=2),encoding='utf-8')
    return projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', nargs='+', default=['ADRO','BBCA','BBRI','BMRI','TLKM','ANTM','PTBA','BREN','TPIA','BRPT','PTRO','INKP','BUMI','BRMS','ISAT'])
    parser.add_argument('--brokers',default=r'data/Daily/All')
    parser.add_argument('--summary',default=r'data/Daily/Summary')
    parser.add_argument('--out',default='runs/expanded-v2')
    parser.add_argument('--feature-set',choices=['basic','expanded_v2'],default='expanded_v2')
    parser.add_argument('--epochs',type=int,default=24)
    options = parser.parse_args()
    tickers=[t.upper() for t in options.tickers]
    if len(set(tickers)) != len(tickers) or any(len(t)!=4 or not t.isalpha() for t in tickers):
        parser.error('Use unique four-letter tickers')
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    root=Path(options.out);root.mkdir(parents=True,exist_ok=True)
    rows=[];manifest=[]
    for ticker in tickers:
        folder=root/ticker
        args=SimpleNamespace(out=str(folder),brokers=options.brokers,summary=options.summary,
                             only_ticker=ticker,stocks=1,window=20,epochs=options.epochs,
                             batch=32,seed=42,channels=16,ticker=ticker,k=10,feature_set=options.feature_set)
        print(f'=== Independent model {ticker} ===',flush=True)
        prepare(args)
        train(args)  # Creates a fresh model, optimizer and scalers for this stock.
        neighbors(args)
        projection=export_view(folder,ticker)
        config=json.loads((folder/'data_report.json').read_text())
        report=json.loads((folder/'training_report.json').read_text())
        digest=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()
        manifest.append({'ticker':ticker,'scope':config['scope'],'source_summary_files':config['universe_files'],
                         'dataset_tickers':config['tickers'],'model_sha256':digest,
                         'relative_model':f'{ticker}/model.pt','samples':config['counts'],
                         'feature_count':len(config['features']),'best_epoch':report['best_epoch'],
                         'feature_set':config['feature_set'],
                         'projection_variance':projection['variance_explained']})
        for h in HORIZONS:
            row={'ticker':ticker,'horizon':h,'date':projection['latest_date'],
                 'prediction_return':projection['predictions_arithmetic_return'][str(h)],
                 'test_dates':report['test']['dates'],'train_samples':config['counts']['train']}
            for method in ['cnn','ridge','train_mean']:
                for metric in ['mae_log_return','direction_accuracy','balanced_direction_accuracy']:
                    row[f'{method}_{metric}']=report['test'][method][str(h)][metric]
            rows.append(row)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    result=pd.DataFrame(rows);result.to_csv(root/'comparison.csv',index=False)
    report_lines=[f'# {len(tickers)} model saham mandiri - {options.feature_set}','',
        'Setiap saham memiliki satu dataset, kosakata broker, scaler, bobot model, PCA, dan bank pola sendiri.',
        'Tidak ada transfer bobot atau penggabungan sampel antarsaham. Seed yang sama digunakan untuk kontrol eksperimen.',
        'Input: OHLCV, net value/lot dan harga broker, persistensi transaksi, buyer/seller ratio, Top3/Top5, Retail, serta return harga 20/40 sesi.',
        'Titik 3D adalah window 20 sesi dari saham itu sendiri. Harga rata-rata broker hanya dibandingkan dalam skala harga mentah sumber broker.',
        '','## Hasil test','',
        '| Saham | Horizon | MAE CNN (log return) | MAE Ridge | MAE rata-rata train | Akurasi arah CNN |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        report_lines.append(f"| {r['ticker']} | {r['horizon']} | {r['cnn_mae_log_return']:.4f} | {r['ridge_mae_log_return']:.4f} | {r['train_mean_mae_log_return']:.4f} | {100*r['cnn_direction_accuracy']:.1f}% |")
    report_lines.extend(['','## Batas interpretasi','',
        'Jumlah sampel per saham ada di manifest.json. Fitur 40 sesi memerlukan warmup; dataset 208 tanggal menghasilkan sekitar 45 train, 22 validasi, dan 22 test. Sampel saling overlap.',
        'Model lebih kecil (16 channel, embedding 32 dimensi) dipakai karena data per saham terbatas. Epoch terbaik dipilih pada validasi.',
        'Prediksi bersifat eksperimental. Perbandingan ini bukan backtest profit, belum walk-forward atau beberapa seed.',
        'Titik terbaru berasal dari 13 November; test query berhenti lebih awal agar label 20 sesi sudah tersedia.',
        'Cosine similarity bukan probabilitas. PCA terpisah tidak memiliki sumbu yang sebanding antarsaham.',
        'Bank pencarian pola hanya periode train. Kandidat berdekatan kurang dari 28 hari kalender dikeluarkan sehingga jumlah tetangga bisa kurang dari 10.',
        '','## Membuka grafik','',
        'Buka index.html di Brave/Chrome setelah mengekstrak semua file. Pilih saham di bagian atas. Setiap halaman hanya menampilkan saham terpilih.',
        'index.html mandiri dan bisa dibuka sendiri tanpa internet. Halaman per saham embedding-3d.html memakai plotly.min.js di folder yang sama.'])
    wins=int(((result.cnn_mae_log_return<result.ridge_mae_log_return)&(result.cnn_mae_log_return<result.train_mean_mae_log_return)).sum())
    report_lines.extend(['',f'CNN mengungguli MAE kedua baseline pada {wins}/{len(result)} kombinasi saham-horizon. Ini satu holdout dan satu seed; bukan bukti keunggulan yang konsisten.'])
    (root/'RESULTS.md').write_text('\n'.join(report_lines)+'\n',encoding='utf-8')
    build_dashboard(root,tickers)
    if options.feature_set=='expanded_v2':
        audit_run(root)
    print(result[['ticker','horizon','cnn_mae_log_return','ridge_mae_log_return','train_mean_mae_log_return']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
