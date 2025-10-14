from itertools import product
import pandas as pd
from .signals import SignalParams, compute_features, partial_signals, ensemble_score, dynamic_thresholds
from .backtest import backtest, metrics

def grid_space():
    return {
        "rsi_window": [10,14,20],
        "rsi_buy": [25,30,35],
        "rsi_sell": [65,70,75],
        "ma_fast": [10,20,30],
        "ma_mid": [40,50,60],
        "ma_slow": [80,100,150],
        "bb_window": [20],
        "bb_std": [1.5,2.0,2.5],
        "w_rsi": [0.2,0.3,0.4],
        "w_ma": [0.2,0.3,0.4],
        "w_bb": [0.1,0.2,0.3],
        "w_breakout": [0.05,0.1,0.2],
        "w_sent": [0.1,0.2,0.3],
        "score_buy": [0.5,0.6,0.7],
        "score_sell": [-0.7,-0.6,-0.5],
        "percentile_mode": [True],
        "percentile_window": [60,90,120],
    }

def walk_forward(close: pd.Series, sentiment: pd.Series | None, space: dict, folds:int=4, cost_bps:int=10):
    n = len(close)
    fold_size = n // (folds+1)
    results = []
    for f in range(folds):
        is_start = f*fold_size
        is_end = is_start + fold_size
        os_start = is_end
        os_end = os_start + fold_size
        close_is = close.iloc[is_start:is_end]
        close_os = close.iloc[os_start:os_end]
        sent_is = None if sentiment is None else sentiment.reindex(close_is.index).fillna(method="ffill")
        sent_os = None if sentiment is None else sentiment.reindex(close_os.index).fillna(method="ffill")
        best = None
        for rsi_w, rsi_b, rsi_s, ma_f, ma_m, ma_s, bb_w, bb_std, wr, wm, wbb, wbrk, ws, t_buy, t_sell, pm, pw in product(
            space["rsi_window"], space["rsi_buy"], space["rsi_sell"],
            space["ma_fast"], space["ma_mid"], space["ma_slow"],
            space["bb_window"], space["bb_std"],
            space["w_rsi"], space["w_ma"], space["w_bb"], space["w_breakout"], space["w_sent"],
            space["score_buy"], space["score_sell"], space["percentile_mode"], space["percentile_window"]
        ):
            p = SignalParams(
                rsi_window=rsi_w, rsi_buy=rsi_b, rsi_sell=rsi_s,
                ma_fast=ma_f, ma_mid=ma_m, ma_slow=ma_s,
                bb_window=bb_w, bb_std=bb_std,
                w_rsi=wr, w_ma=wm, w_bb=wbb, w_breakout=wbrk, w_sent=ws,
                score_buy=t_buy, score_sell=t_sell,
                percentile_mode=pm, percentile_window=pw
            )
            feat_is = compute_features(close_is, p)
            sig_is = partial_signals(feat_is, p)
            sc_is = ensemble_score(sig_is, sent_is, p)
            buy_thr_is, sell_thr_is = dynamic_thresholds(sc_is, p)
            bt_is = backtest(close_is, sc_is, buy_thr_is, sell_thr_is, cost_bps/2, cost_bps/2)
            m_is = metrics(bt_is["eq"], bt_is["ret"])
            key = (m_is["Sharpe"], m_is["CAGR"])
            if (best is None) or (key > best[0]):
                best = (key, p)
        p_star = best[1]
        feat_os = compute_features(close_os, p_star)
        sig_os = partial_signals(feat_os, p_star)
        sc_os = ensemble_score(sig_os, sent_os, p_star)
        buy_thr_os, sell_thr_os = dynamic_thresholds(sc_os, p_star)
        bt_os = backtest(close_os, sc_os, buy_thr_os, sell_thr_os, cost_bps/2, cost_bps/2)
        m_os = metrics(bt_os["eq"], bt_os["ret"])
        results.append({"fold": f+1, "params": p_star, "metrics_os": m_os})
    stability = {}
    for r in results:
        p = r["params"]
        for k,v in p.__dict__.items():
            key = f"{k}:{v}"
            stability[key] = stability.get(key, 0) + 1
    return results, stability
