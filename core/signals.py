from dataclasses import dataclass
import pandas as pd
from .indicators import rsi, sma, ema, bollinger_bands, swings
from .regime import market_regime

@dataclass
class SignalParams:
    rsi_window: int = 14
    rsi_buy: int = 30
    rsi_sell: int = 70
    ma_fast: int = 20
    ma_mid: int = 50
    ma_slow: int = 100
    ma_type: str = "ema"
    bb_window: int = 20
    bb_std: float = 2.0
    w_rsi: float = 0.2
    w_ma: float = 0.25
    w_bb: float = 0.15
    w_atr: float = 0.2
    w_breakout: float = 0.1
    w_sent: float = 0.1
    score_buy: float = 0.6
    score_sell: float = -0.6
    percentile_mode: bool = True
    percentile_window: int = 90

def _ma(close: pd.Series, win:int, typ:str):
    return ema(close, win) if typ=="ema" else sma(close, win)

def compute_features(close: pd.Series, p: SignalParams) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    out["Close"] = close
    out["RSI"] = rsi(close, p.rsi_window)
    out["MA_fast"] = _ma(close, p.ma_fast, p.ma_type)
    out["MA_mid"]  = _ma(close, p.ma_mid,  p.ma_type)
    out["MA_slow"] = _ma(close, p.ma_slow, p.ma_type)
    out["BB_mid"], out["BB_up"], out["BB_lo"] = bollinger_bands(close, p.bb_window, p.bb_std)
    out["Regime"] = market_regime(close, p.ma_mid)
    return out

def partial_signals(feat: pd.DataFrame, p: SignalParams) -> pd.DataFrame:
    s = pd.DataFrame(index=feat.index)
    s["sig_rsi"] = 0.0
    s.loc[feat["RSI"] <= p.rsi_buy, "sig_rsi"] = 1.0
    s.loc[feat["RSI"] >= p.rsi_sell, "sig_rsi"] = -1.0
    s["sig_ma"] = 0.0
    s.loc[feat["MA_fast"] > feat["MA_slow"], "sig_ma"] = 1.0
    s.loc[feat["MA_fast"] < feat["MA_slow"], "sig_ma"] = -1.0
    s["sig_bb"] = 0.0
    s.loc[feat["Close"] < feat["BB_lo"], "sig_bb"] = 1.0
    s.loc[feat["Close"] > feat["BB_up"], "sig_bb"] = -1.0
    # simple breakout
    n = 5
    rolling_max = feat["Close"].rolling(n).max()
    rolling_min = feat["Close"].rolling(n).min()
    s["sig_breakout"] = 0.0
    s.loc[feat["Close"] >= rolling_max, "sig_breakout"] = 1.0
    s.loc[feat["Close"] <= rolling_min, "sig_breakout"] = -1.0
    return s

def ensemble_score(sig: pd.DataFrame, sentiment: pd.Series | None, p: SignalParams) -> pd.Series:
    sc = (p.w_rsi*sig["sig_rsi"] + p.w_ma*sig["sig_ma"] + p.w_bb*sig["sig_bb"] + p.w_breakout*sig["sig_breakout"])
    if sentiment is not None:
        sc = sc + p.w_sent*sentiment.reindex(sig.index).fillna(method="ffill").fillna(0)
    return sc.clip(-1,1)

def dynamic_thresholds(score: pd.Series, p: SignalParams):
    if not p.percentile_mode:
        return p.score_buy, p.score_sell
    w = p.percentile_window
    roll = score.rolling(w)
    buy_thr = roll.quantile(0.80).fillna(p.score_buy)
    sell_thr = roll.quantile(0.20).fillna(p.score_sell)
    return buy_thr, sell_thr

def confidence_and_explain(sig, score, buy_thr, sell_thr, p):
    import numpy as np
    idx = score.index
    if isinstance(buy_thr, pd.Series):
        bt = buy_thr
        st = sell_thr
    else:
        bt = pd.Series(p.score_buy, index=idx)
        st = pd.Series(p.score_sell, index=idx)
    dist = np.minimum((score - bt).abs(), (score - st).abs())
    maxd = (bt - st).abs()
    conf = (1 - (dist / maxd.clip(lower=1e-9))).clip(0,1)
    parts = pd.DataFrame(index=idx)
    parts["RSI"] = p.w_rsi*sig["sig_rsi"]
    parts["MA"] = p.w_ma*sig["sig_ma"]
    parts["BB"] = p.w_bb*sig["sig_bb"]
    parts["BRK"] = p.w_breakout*sig["sig_breakout"]
    return conf, parts
