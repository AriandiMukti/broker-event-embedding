"""Render controlled research comparisons from saved metrics, without refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root='runs/research-v1'):
    root=Path(root)
    macro=pd.read_csv(root/'macro_summary.csv').set_index('variant')
    comparison=pd.read_csv(root/'comparison.csv')
    heat=comparison.pivot(index='ticker',columns='horizon',values='combined_improvement_vs_ohlcv')*100
    heat=heat.sort_index().reindex(columns=[5,10,20])
    fig=plt.figure(figsize=(13,10),facecolor='#f8fafc')
    grid=fig.add_gridspec(1,2,width_ratios=[1,1.5],wspace=.4)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig.suptitle('Apakah data broker + Retail membantu?',x=.06,y=.97,ha='left',fontsize=21,fontweight='bold',color='#182333')
    fig.text(.06,.923,'15 saham mandiri | 3 fold walk-forward | 2 seed | 42 tanggal prediksi per saham',color='#475569',fontsize=12)
    left=fig.add_subplot(grid[0,0],facecolor='#f8fafc')
    order=['ohlcv','broker_retail','combined'];labels=['OHLCV','Broker + Retail','Gabungan']
    values=[macro.loc[name,'mean_relative_mae'] for name in order]
    left.barh(range(3),values,color=['#64748b','#94a3b8','#2563eb'],height=.45)
    left.axvline(1,lw=1,color='#9a3412',linestyle='--')
    left.set_yticks(range(3),labels);left.invert_yaxis()
    left.set_xlim(0,max(1.2,max(values)*1.2))
    left.set_title('Error relatif: lebih kecil lebih baik',loc='left',fontsize=12,pad=16)
    left.set_xlabel('MAE model / MAE baseline rata-rata train')
    left.spines[['top','right','left']].set_visible(False);left.tick_params(axis='y',length=0)
    for i,v in enumerate(values):left.text(v+.02,i,f'{v:.3f}',va='center')
    ax=fig.add_subplot(grid[0,1])
    limit=max(5,float(np.nanmax(abs(heat.to_numpy()))))
    im=ax.imshow(heat.to_numpy(),cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(heat)),heat.index);ax.set_xticks(range(3),['5 sesi','10 sesi','20 sesi'])
    ax.set_title('Perubahan error gabungan vs OHLCV',loc='left',fontsize=12,pad=16)
    for i in range(len(heat)):
        for j in range(3):
            value=heat.iloc[i,j]
            ax.text(j,i,f'{value:+.1f}%',ha='center',va='center',color='#111827' if abs(value)<limit*.65 else '#ffffff',fontsize=10)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.035,label='Penurunan MAE (%) | positif = membaik')
    fig.text(.06,.055,'Gabungan hanya dapat dianggap membantu pada pasangan saham-horizon yang error-nya turun.',color='#182333',fontsize=11)
    fig.text(.06,.028,'Eksploratif: label overlap, periode pendek, data historis ini pernah dilihat. Belum ada backtest profit atau biaya transaksi.',color='#9a3412',fontsize=10)
    fig.subplots_adjust(left=.13,right=.94,top=.86,bottom=.13)
    fig.savefig(root/'comparison.png',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    # ADRO plot is descriptive forecasting evidence, not a broker attribution.
    forecasts=pd.read_csv(root/'predictions.csv')
    sub=forecasts[(forecasts.ticker=='ADRO')&(forecasts.horizon==5)]
    mean=sub.groupby(['date','variant'],as_index=False)[['actual','prediction']].mean()
    actual=mean.groupby('date').actual.first()
    fig,ax=plt.subplots(figsize=(11,5),facecolor='#f8fafc');ax.set_facecolor('#f8fafc')
    x=pd.to_datetime(actual.index)
    ax.plot(x,100*np.expm1(actual.to_numpy()),label='Return ADRO aktual 5 sesi',color='#182333',lw=2)
    for key,color,label in [('ohlcv','#64748b','Prediksi OHLCV'),('combined','#2563eb','Prediksi gabungan')]:
        values=mean[mean.variant==key].set_index('date').reindex(actual.index).prediction
        ax.plot(x,100*np.expm1(values.to_numpy()),label=label,color=color,lw=1.5)
    ax.axhline(0,color='#cbd5e1',lw=.8)
    ax.set_title('ADRO: prediksi walk-forward 5 sesi vs hasil aktual',loc='left',pad=15)
    ax.set_ylabel('Return (%)');ax.set_xlabel('Tanggal query | prediksi dirata-rata dari 2 seed')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(root/'adro-walk-forward.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
    print(root.resolve())


if __name__=='__main__':render()
