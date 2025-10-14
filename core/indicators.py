import pandas as pd
import numpy as np

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()

def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()

def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs() if high is not None and low is not None else (close - prev_close).abs()
    tr2 = (high - prev_close).abs() if high is not None else tr1
    tr3 = (low - prev_close).abs() if low is not None else tr1
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def swings(close: pd.Series, lookback:int=5):
    rolling_max = close.rolling(lookback).max()
    rolling_min = close.rolling(lookback).min()
    return rolling_max, rolling_min
