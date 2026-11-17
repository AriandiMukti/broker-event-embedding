"""Run an already trained, physically pruned per-stock adaptive model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from main import Encoder
from research import load_stock
from compact_features import compact_matrix,align_loaded
from main import read_summary


def predict(ticker,root='runs/adaptive-v1/latest',summary=r'data/Daily/Summary',brokers=r'data/Daily/All'):
    ticker=ticker.upper()
    checkpoint=torch.load(Path(root)/ticker/'model.pt',map_location='cpu',weights_only=False)
    if checkpoint['ticker']!=ticker:raise ValueError('Model ticker mismatch')
    loaded=load_stock(ticker,summary,brokers)
    reference,_=read_summary(Path(summary)/'ADRO_Summary.csv')
    latest_broker_date=max(p.name for p in Path(brokers).iterdir() if p.is_dir() and len(p.name)==10 and p.name[4]=='-')
    loaded=align_loaded(loaded,reference.loc[:latest_broker_date].index)
    matrix,names,_=compact_matrix(loaded)
    if names!=checkpoint['candidate_feature_names']:raise ValueError('Feature schema changed; retrain model')
    mask=checkpoint['source_feature_mask'];window=checkpoint['window']
    if len(matrix)<40+window or not loaded[1].iloc[-window:].all():
        return {'ticker':ticker,'date':str(loaded[0].index[-1].date()),'available':False,
                'selected_features':checkpoint['selected_features'],'input_channels':checkpoint['features'],
                'predicted_returns':{str(h):None for h in [5,10,20]},
                'reason':'Input 20 sesi terbaru mengandung OHLCV invalid atau baris Summary hilang.',
                'data_quality':loaded[-1]}
    x=np.clip((matrix[-window:,mask]-checkpoint['mean'])/checkpoint['std'],-10,10).astype('float32')
    model=Encoder(checkpoint['features'],window,checkpoint['channels'],compact_decoder=True).eval()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():_,p,_=model(torch.from_numpy(x[None])*checkpoint['input_gain'])
    prediction=p.numpy()[0]*checkpoint['target_std']+checkpoint['target_mean']
    return {'ticker':ticker,'date':str(loaded[0].index[-1].date()),'available':True,'selected_features':checkpoint['selected_features'],
            'numeric_feature_count':len(checkpoint['selected_features']),'input_channels':checkpoint['features'],
            'predicted_returns':{str(h):float(np.expm1(prediction[j])) for j,h in enumerate([5,10,20])}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticker',default='ADRO')
    parser.add_argument('--root',default='runs/adaptive-v1/latest')
    args=parser.parse_args();torch.set_num_threads(4)
    print(json.dumps(predict(args.ticker,args.root),indent=2))
