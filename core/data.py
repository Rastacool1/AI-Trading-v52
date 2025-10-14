from __future__ import annotations
import pandas as pd

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

SYMBOL_MAP = {
    "^spx": "^spx", "spx": "^spx",
    "^ndx": "^ndx", "ndx": "^ndx",
    "eurusd": "eurusd", "btcusd": "btcusd",
    "^vix": "^vix",
}

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("data")
    close_col = cols.get("close") or cols.get("zamkniecie")
    if not date_col or not close_col:
        raise ValueError("CSV must contain Date/Data and Close/Zamkniecie columns")
    out = df.rename(columns={date_col: "Date", close_col: "Close"})[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out = out.sort_values("Date").dropna()
    out = out.set_index("Date")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna()
    return out

def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _normalize_df(df)

def from_stooq(symbol: str) -> pd.DataFrame:
    sym = SYMBOL_MAP.get(symbol.lower(), symbol.lower())
    url = STOOQ_URL.format(symbol=sym)
    df = pd.read_csv(url)
    return _normalize_df(df)

try:
    import yfinance as yf
except Exception:
    yf = None

def from_yf(symbol: str, period: str = "max") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    data = data.rename(columns={"Close": "Close"})
    data = data[["Close"]].dropna()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data
