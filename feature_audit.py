"""Read saved model inputs and export a human-readable latest feature audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=['price_return_20','price_return_40','summary_retail_raw','summary_retail_cumulative_5',
      'summary_retail_share_abs_net_lot','broker_buy_count','broker_sell_count',
      'buyer_seller_ratio','broker_buyer_fraction','broker_count_balance',
      'buyer_seller_ratio_delta_5','buyer_seller_ratio_delta_20',
      'summary_total_buyer','summary_total_seller','summary_buyer_seller_ratio',
      'broker_price_spread','lot_buy_hhi','lot_sell_hhi',
      'broker_abs_net_value_relative20','broker_abs_net_lot_relative20']


def read_latest_features(folder):
    folder=Path(folder)
    report=json.loads((folder/'data_report.json').read_text())
    vector=np.load(folder/'dataset.npz')['latest_x'][0,-1]
    features=dict(zip(report['features'],vector.tolist()))
    row={'ticker':report['tickers'][0],'date':report['end'],'feature_count':len(vector)}
    for key in KEYS:
        row[key]=None if features.get(key+'_missing',0)>0 else features.get(key)
    return row


def audit_run(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());rows=[]
    for item in manifest:
        ticker=item['ticker'];folder=root/ticker
        row=read_latest_features(folder)
        (folder/'latest_features.json').write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf-8')
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'latest_feature_audit.csv',index=False)
    return rows


if __name__=='__main__':
    print(pd.DataFrame(audit_run('runs/expanded-v2')).to_string(index=False))
