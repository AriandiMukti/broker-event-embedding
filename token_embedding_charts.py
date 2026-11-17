"""Exact static preview of ADRO event embeddings and control-model results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def render(root='runs/adro-token-v1'):
    root=Path(root);table=pd.read_csv(root/'token_vectors.csv')
    neighbors=pd.read_csv(root/'token_neighbors.csv');summary=pd.read_csv(root/'summary.csv').set_index('method')
    config=json.loads((root/'config.json').read_text());embedding=json.loads((root/'embedding_summary.json').read_text())
    query='YP_BUY_LARGE' if 'YP_BUY_LARGE' in set(neighbors['query']) else neighbors.iloc[0]['query']
    chosen=neighbors[neighbors['query']==query].sort_values('rank').head(5)
    points=table[table.token.isin(set(neighbors['query']))].set_index('token')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig=plt.figure(figsize=(14,9),facecolor='#f8fafc')
    gs=fig.add_gridspec(2,2,width_ratios=[1.2,1],height_ratios=[1,1],hspace=.45,wspace=.25)
    fig.suptitle('ADRO | Embedding kejadian broker',x=.05,y=.97,ha='left',fontsize=22,fontweight='bold',color='#182333')
    fig.text(.05,.919,f"Titik = broker + aksi net | {config['dimension']} dimensi asli | PCA 3D {100*embedding['pca_variance']:.1f}% variasi",color='#475569',fontsize=12)
    ax=fig.add_subplot(gs[:,0],projection='3d',facecolor='#f8fafc')
    colors={'BUY_NORMAL':'#60a5fa','BUY_LARGE':'#2563eb','SELL_NORMAL':'#f59e0b','SELL_LARGE':'#dc2626'}
    for state,color in colors.items():
        sub=points[points.index.str.endswith('_'+state)]
        ax.scatter(sub.pc1,sub.pc2,sub.pc3,s=30,color=color,alpha=.7,label=state,depthshade=False)
    q=points.loc[query]
    ax.scatter(q.pc1,q.pc2,q.pc3,s=140,color='#182333',marker='D',depthshade=False)
    ax.text(q.pc1+.015,q.pc2+.015,q.pc3+.015,query,fontsize=10,color='#182333')
    for row in chosen.to_dict('records'):
        p=points.loc[row['neighbor']]
        ax.plot([q.pc1,p.pc1],[q.pc2,p.pc2],[q.pc3,p.pc3],color='#94a3b8',lw=.8)
    ax.set(xlabel='PC1',ylabel='PC2',zlabel='PC3');ax.view_init(elev=22,azim=35)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=9);ax.legend(loc='lower left',bbox_to_anchor=(-.05,-.04),frameon=False,fontsize=9)
    bx=fig.add_subplot(gs[0,1]);bx.axis('off')
    bx.set_title('Tetangga lintas broker: '+query,loc='left',fontsize=12,pad=12)
    values=[[r['neighbor'],f"{r['cosine_similarity']:.3f}",str(r['neighbor_train_days'])] for r in chosen.to_dict('records')]
    tab=bx.table(cellText=values,colLabels=['Token','Cosine 8D','Hari train'],cellLoc='left',colLoc='left',bbox=[0,.1,1,.85],colWidths=[.60,.23,.17])
    tab.auto_set_font_size(False);tab.set_fontsize(10)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#e2e8f0');cell.set_linewidth(.5);cell.set_facecolor('#e2e8f0' if r==0 else '#f8fafc')
    cx=fig.add_subplot(gs[1,1],facecolor='#f8fafc')
    methods=['event_embedding','shuffled_context','training_frequency','markov_previous_state']
    labels=['Embedding konteks','Konteks diacak','Frekuensi train','Markov state terakhir']
    vals=[summary.loc[m,'nll'] for m in methods]
    cx.barh(range(4),vals,color=['#2563eb','#94a3b8','#64748b','#64748b'],height=.5)
    cx.set_yticks(range(4),labels);cx.invert_yaxis();cx.set_xlim(0,max(vals)*1.16)
    cx.set_title('Error probabilitas (NLL): lebih kecil lebih baik',loc='left',fontsize=12,pad=15)
    for i,v in enumerate(vals):cx.text(v+.025,i,f'{v:.3f}',va='center')
    cx.spines[['top','right','left']].set_visible(False);cx.tick_params(axis='y',length=0)
    cx.set_xlabel('Model: rata-rata 5 seed | Uji: 42 tanggal')
    fig.text(.05,.06,'Konteks asli dan konteks acak menghasilkan skor hampir sama: hubungan kontekstual yang berguna belum terbukti.',color='#9a3412',fontsize=11)
    fig.text(.05,.032,'Cosine bukan probabilitas dan bukan bukti koordinasi broker. Prediksi model adalah state net broker, bukan harga saham.',color='#475569',fontsize=10)
    fig.subplots_adjust(left=.04,right=.96,top=.85,bottom=.16)
    fig.savefig(root/'embedding-preview.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
    print(root.resolve())


if __name__=='__main__':render()
