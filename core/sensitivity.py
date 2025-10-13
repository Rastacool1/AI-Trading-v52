import pandas as pd
import numpy as np
from .backtest import backtest, metrics

def sensitivity_costs(close, score, buy_thr, sell_thr, costs=[0,5,10,15,20]):
    rows = []
    for c in costs:
        bt = backtest(close, score, buy_thr, sell_thr, c/2, c/2)
        m = metrics(bt['eq'], bt['ret'])
        m['cost_bps'] = c
        rows.append(m)
    return pd.DataFrame(rows)

def sensitivity_thresholds(close, score, p, deltas=[-0.1,-0.05,0,0.05,0.1]):
    # tylko dla statycznych progów
    from copy import deepcopy
    rows = []
    for d in deltas:
        buy = p.score_buy + d
        sell = p.score_sell - d
        bt = backtest(close, score, buy, sell, 5, 5)
        m = metrics(bt['eq'], bt['ret'])
        m['delta'] = d
        rows.append(m)
    return pd.DataFrame(rows)
