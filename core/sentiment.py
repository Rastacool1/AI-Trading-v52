import pandas as pd

def from_csv(file) -> pd.Series:
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    s = df["Sentiment"].clip(-1, 1)
    return s

def ewma(series: pd.Series, span:int=10):
    return series.ewm(span=span, adjust=False).mean()

def heuristic_from_vix(vix_close: pd.Series, span:int=10, cap:float=0.85) -> pd.Series:
    v = vix_close.dropna()
    p5, p95 = v.quantile(0.05), v.quantile(0.95)
    if p95 == p5:
        s = pd.Series(0, index=v.index)
    else:
        z = (v.clip(p5, p95) - p5) / (p95 - p5)
        s = 1 - 2*z  # 1 -> -1 gdy VIX rośnie
    s = ewma(s, span=span).clip(-cap, cap)
    return s.reindex(vix_close.index).fillna(method="ffill").fillna(0)
