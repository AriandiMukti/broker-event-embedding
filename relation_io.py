"""Compact, loss-bounded storage of exported state probabilities."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from broker_tokens import STATES


def source_fingerprint(ticker,summary_path,brokers_path):
    digest=hashlib.sha256()
    path=Path(summary_path)/f'{ticker}_Summary.csv'
    digest.update(path.name.encode());digest.update(path.read_bytes())
    for folder in sorted(Path(brokers_path).iterdir()):
        if not folder.is_dir() or len(folder.name)!=10 or folder.name[4]!='-':continue
        digest.update(folder.name.encode())
        path=folder/f'{ticker}_ALL_{folder.name}.json'
        if not path.exists():path=folder/f'{ticker}.xls'
        if path.exists():digest.update(path.name.encode());digest.update(path.read_bytes())
        else:digest.update(b'MISSING')
    return digest.hexdigest()


def compact_predictions(folder):
    folder=Path(folder); frame=pd.read_csv(folder/'test_predictions.csv')
    arrays={c:frame[c].to_numpy().astype('U') for c in ['task','method','date','broker']}
    arrays.update(fold=frame.fold.to_numpy(dtype=np.int8),seed=frame.seed.to_numpy(dtype=np.int32),
                  actual=frame.actual.to_numpy(dtype=np.int8),probabilities=frame[STATES].to_numpy(dtype=np.float32))
    np.savez_compressed(folder/'test_predictions.npz',**arrays)


def read_predictions(folder):
    folder=Path(folder)
    if (folder/'test_predictions.csv').exists(): return pd.read_csv(folder/'test_predictions.csv')
    data=np.load(folder/'test_predictions.npz',allow_pickle=False)
    frame=pd.DataFrame({c:data[c] for c in ['task','fold','method','seed','date','broker','actual']})
    for i,c in enumerate(STATES):frame[c]=data['probabilities'][:,i]
    return frame
