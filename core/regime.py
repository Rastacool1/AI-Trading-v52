import pandas as pd
from .indicators import ema

def market_regime(close: pd.Series, ma_mid:int=50, vol_window:int=20) -> pd.Series:
    mid = ema(close, ma_mid)
    regime = pd.Series("side", index=close.index)
    regime[(close > mid)] = "bull"
    regime[(close < mid)] = "bear"
    return regime
