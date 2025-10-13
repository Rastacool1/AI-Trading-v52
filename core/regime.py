import pandas as pd
from .indicators import ema

def market_regime(close: pd.Series, ma_mid:int=50, vol_window:int=20) -> pd.Series:
    mid = ema(close, ma_mid)
    ret = close.pct_change().rolling(vol_window).std()
    # bull: close>mid, bear: close<mid, side: w pasie ±1*std*close
    regime = pd.Series("side", index=close.index)
    regime[(close > mid)] = "bull"
    regime[(close < mid)] = "bear"
    return regime
