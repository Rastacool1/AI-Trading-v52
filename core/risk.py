import numpy as np
import pandas as pd

def volatility_target_position(returns: pd.Series, target_vol_annual: float = 0.12, lookback:int=20):
    vol = returns.rolling(lookback).std() * np.sqrt(252)
    pos = (target_vol_annual / vol).clip(upper=1.0)  # nie używamy lewara w MVP
    return pos.fillna(0)

def clipped_kelly(hit_rate: float, payoff: float, clip: float = 0.5):
    # Kelly = p - (1-p)/b, gdzie b=payoff
    if payoff <= 0:
        return 0.0
    p = hit_rate
    k = p - (1-p)/payoff
    return float(np.clip(k, 0, clip))

def de_risk_overlay(equity: pd.Series, hard_dd: float = -0.08):
    dd = equity / equity.cummax() - 1.0
    mask = (dd <= hard_dd)
    # gdy DD przekroczy próg — pozycja 0 do czasu powrotu powyżej max drawdown + 1%
    overlay = pd.Series(1.0, index=equity.index)
    if mask.any():
        last = None
        for t, bad in mask.items():
            if bad and last is None:
                last = t
                overlay.loc[t:] = 0.0
                break
    return overlay
