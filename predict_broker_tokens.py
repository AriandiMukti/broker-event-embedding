"""Predict next-session ADRO net broker states using the saved token model."""
import argparse
import json
from pathlib import Path
import torch
from research import load_stock
from broker_tokens import transform_tokens,STATES
from adro_token_embedding import EventEmbedding,probability


def predict(root='runs/adro-token-v1',summary=r'data/Daily/Summary',brokers=r'data/Daily/All'):
    checkpoint=torch.load(Path(root)/'model.pt',map_location='cpu',weights_only=False)
    tokenizer=checkpoint['tokenizer'];codes=tokenizer['brokers'];days=checkpoint['context_days']
    expected=[f'{code}_{state}' for code in codes for state in STATES]
    if tokenizer['tokens'][1:1+len(expected)]!=expected:raise ValueError('Token schema changed; retrain model')
    loaded=load_stock('ADRO',summary,brokers)
    daily,_,valid=transform_tokens(loaded,tokenizer)
    if len(daily)<40+days or not valid[-days:].all():
        return {'ticker':'ADRO','available':False,'reason':'Latest context contains missing broker data or invalid OHLCV.'}
    model=EventEmbedding(len(tokenizer['tokens']),len(codes),checkpoint['dimension'],days)
    model.load_state_dict(checkpoint['state_dict']);p=probability(model,daily[-days:][None])[0]
    return {'ticker':'ADRO','available':True,'context_end':str(loaded[0].index[-1].date()),
        'target':'next trading session; date not assumed','predictions':[
            {'broker':code,'state':STATES[p[i].argmax()],
             'probabilities':{state:float(p[i,k]) for k,state in enumerate(STATES)}} for i,code in enumerate(codes)]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='runs/adro-token-v1')
    args=parser.parse_args();torch.set_num_threads(1)
    print(json.dumps(predict(args.root),indent=2))
