"""Plot adaptive feature choices and matched forecasting comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compact_features import NUMERIC

ALIASES={
 'return':'Return 1 sesi','open_gap':'Gap pembukaan','range':'Rentang High-Low','body':'Perubahan Open-Close',
 'log_volume_change':'Perubahan volume','volume_relative20':'Volume relatif 20 sesi','volatility20':'Volatilitas 20 sesi',
 'price_return_20':'Return 20 sesi','price_return_40':'Return 40 sesi','summary_retail_raw':'Retail',
 'summary_retail_signed_log1p':'Retail (log bertanda)','summary_retail_share_abs_net_lot':'Porsi Retail',
 'summary_retail_cumulative_share_5':'Retail kumulatif 5 sesi','summary_retail_cumulative_share_20':'Retail kumulatif 20 sesi',
 'broker_buy_count':'Jumlah broker net buy','broker_sell_count':'Jumlah broker net sell',
 'buyer_seller_ratio':'Rasio buyer-seller','broker_buyer_fraction':'Porsi broker pembeli',
 'buyer_seller_ratio_delta_5':'Perubahan rasio buyer-seller 5 sesi','buyer_seller_ratio_delta_20':'Perubahan rasio buyer-seller 20 sesi',
 'buy_hhi':'Konsentrasi nilai pembeli','sell_hhi':'Konsentrasi nilai penjual','lot_buy_hhi':'Konsentrasi lot pembeli','lot_sell_hhi':'Konsentrasi lot penjual',
 'summary_net_top3_all_lot_share':'Porsi net Top3','summary_net_top5_all_lot_share':'Porsi net Top5',
 'summary_net_top5_all_lot_cumulative_20':'Net Top5 kumulatif 20 sesi','broker_abs_net_value_log1p':'Intensitas net value (log)',
 'broker_abs_net_lot_log1p':'Intensitas net lot (log)','broker_abs_net_value_relative20':'Intensitas nilai relatif 20 sesi',
 'broker_abs_net_lot_relative20':'Intensitas lot relatif 20 sesi','broker_price_spread':'Selisih harga buyer-seller',
 'persistent_buyer_share_5':'Persistensi pembeli 5 sesi','persistent_seller_share_5':'Persistensi penjual 5 sesi',
 'persistent_buyer_share_20':'Persistensi pembeli 20 sesi','persistent_seller_share_20':'Persistensi penjual 20 sesi',
 'flow_turnover_1':'Perubahan komposisi broker','flow_turnover_mean20':'Rata-rata perubahan komposisi 20 sesi'}


def render(root='runs/adaptive-v1'):
    root=Path(root)
    latest=json.loads((root/'latest_manifest.json').read_text())
    universe=json.loads((root/'universe.json').read_text())
    comparison=pd.read_csv(root/'comparison.csv')
    macro=pd.read_csv(root/'macro_summary.csv').set_index('model')
    latest=sorted(latest,key=lambda r:r['ticker'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    matrix=np.array([[name in row['selected_features'] for name in NUMERIC] for row in latest],dtype=int)
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(19,12),facecolor='#f8fafc')
    ax.imshow(matrix,cmap=ListedColormap(['#e2e8f0','#2563eb']),vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(len(latest)),[f"{r['ticker']} ({r['numeric_feature_count']})" for r in latest])
    ax.set_xticks(range(len(NUMERIC)),[ALIASES[n] for n in NUMERIC],rotation=65,ha='right',fontsize=9)
    ax.tick_params(length=0)
    ax.set_title('Susunan fitur berbeda untuk tiap saham',loc='left',pad=40,fontsize=21,fontweight='bold')
    fig.text(.12,.915,'Biru = fitur numerik terpilih pada model terbaru | angka dalam kurung = jumlah fitur | missing flag disimpan terpisah',color='#475569')
    fig.text(.12,.02,'Pilihan berdasarkan validasi saham terkait, bukan hasil test. Susunan dapat berubah saat retraining.',color='#9a3412')
    fig.subplots_adjust(left=.12,right=.98,top=.90,bottom=.30)
    fig.savefig(root/'feature-map.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)

    fig=plt.figure(figsize=(14,13),facecolor='#f8fafc')
    grid=fig.add_gridspec(1,2,width_ratios=[1,1.3],wspace=.35)
    fig.suptitle('25 saham: fitur dipilih masing-masing saham',x=.06,y=.97,ha='left',fontsize=21,fontweight='bold')
    fig.text(.06,.936,'3 fold walk-forward | 2 seed CNN | hingga 42 query per saham | ASII 41, BBNI 39 | pemilihan memakai validasi saja',color='#475569')
    ax=fig.add_subplot(grid[0,0],facecolor='#f8fafc')
    order=['ohlcv_cnn','adaptive_cnn','ohlcv_ridge','adaptive_ridge','train_mean']
    labels=['CNN OHLCV','CNN adaptif','Ridge OHLCV','Ridge adaptif','Rata-rata train']
    values=[macro.loc[n,'relative_mae'] for n in order]
    ax.barh(range(5),values,color=['#94a3b8','#2563eb','#94a3b8','#2563eb','#64748b'],height=.5)
    ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.axvline(1,color='#9a3412',lw=1,linestyle='--')
    ax.set_xlim(0,max(1.1,max(values)*1.17));ax.set_title('Error relatif: lebih kecil lebih baik',loc='left',fontsize=12,pad=20)
    for i,value in enumerate(values):ax.text(value+.02,i,f'{value:.3f}',va='center')
    ax.set_xlabel('MAE model / MAE baseline rata-rata train')
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    gains=comparison.groupby('ticker')[['adaptive_cnn_gain_vs_ohlcv','adaptive_ridge_gain_vs_ohlcv']].mean()*100
    gains=gains.sort_index();values=gains.to_numpy();limit=max(5,float(np.nanmax(abs(values))))
    bx=fig.add_subplot(grid[0,1]);im=bx.imshow(values,cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
    bx.set_yticks(range(len(gains)),gains.index)
    bx.set_xticks([0,1],['CNN adaptif','Ridge adaptif'])
    bx.set_title('Penurunan error vs OHLCV\nrata-rata 3 horizon (%)',loc='left',fontsize=12,pad=15)
    for i in range(len(gains)):
        for j in range(2):
            value=values[i,j]
            bx.text(j,i,f'{value:+.1f}%',ha='center',va='center',fontsize=9,color='#ffffff' if abs(value)>.65*limit else '#111827')
    fig.colorbar(im,ax=bx,fraction=.035,pad=.025,label='Positif = error berkurang')
    fig.text(.06,.04,'Kandidat banyak, validasi hanya 10 tanggal per fold. Hasil pengembangan historis; bukan bukti profit atau signifikansi.',color='#9a3412',fontsize=10)
    fig.subplots_adjust(left=.14,right=.94,top=.88,bottom=.10)
    fig.savefig(root/'comparison.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)

    marker='\n## Susunan model terbaru per saham\n'
    report=(root/'RESULTS.md').read_text(encoding='utf-8').split(marker)[0]
    lines=[marker,'','| Saham | Jumlah fitur numerik | Fitur terpilih |','|---|---:|---|']
    for row in latest:
        lines.append(f"| {row['ticker']} | {row['numeric_feature_count']} | {', '.join(ALIASES[n] for n in row['selected_features'])} |")
    blocked=[r['ticker'] for r in latest if any(v is None for v in r['predictions'].values())]
    lines.extend(['','## Kelengkapan prediksi terbaru','',
                  f"Model terlatih: {len(latest)}. Prediksi terbaru tidak tersedia untuk: {', '.join(blocked) if blocked else 'tidak ada'}.",
                  'Review missing sessions and invalid OHLCV in the input audit. Source prices are not repaired or forward-filled.',
                  'Jumlah query test aktual: ASII 41, BBNI 39, saham lain 42. Label hanya memerlukan endpoint t+5/t+10/t+20 yang valid; window input memakai 20 sesi valid.'])
    (root/'RESULTS.md').write_text(report+'\n'.join(lines)+'\n',encoding='utf-8')
    # Cohorts are reported separately so the ten new stocks do not silently
    # change the interpretation of results for the original fifteen stocks.
    metric=pd.read_csv(root/'metrics_by_seed.csv')
    avg=metric.groupby(['ticker','model','horizon'],as_index=False).mae.mean()
    baseline=avg[avg.model=='train_mean'][['ticker','horizon','mae']].rename(columns={'mae':'baseline'})
    avg=avg.merge(baseline,on=['ticker','horizon'])
    avg['relative_mae']=avg.mae/avg.baseline.where(avg.baseline>1e-10)
    avg['cohort']=np.where(avg.ticker.isin(universe['existing']),'original_15','additional_10')
    avg.groupby(['cohort','model'],as_index=False).relative_mae.mean().to_csv(root/'cohort_summary.csv',index=False)
    print(root.resolve())


if __name__=='__main__':render()
