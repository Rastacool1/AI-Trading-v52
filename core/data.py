# core/data.py — stabilny reader Stooq (jak w v4.4.2) + Yahoo
from __future__ import annotations
import io
import requests
import pandas as pd

def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Stabilny reader Stooq — działa dla symboli typu btcpln, eurusd, spx itd.
    Zawsze normalizuje symbol: usuwa ^ / = i wymusza lower-case.
    """
    sym = symbol.strip().lower().replace("^", "").replace("/", "").replace("=", "")
    url = f"https://stooq.pl/q/d/l/?s={sym}&i=d"

    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = r.text.strip()

    # wykrycie HTML lub komunikatu
    low = text.lower()
    if low.startswith("<") or "brak danych" in low or "error" in low:
        raise ValueError(f"Brak danych dla symbolu '{sym}' w Stooq ({url})")

    # Stooq używa ';' i PL nagłówków (Data, Zamkniecie)
    df = pd.read_csv(io.StringIO(text), sep=";")

    if "Data" not in df.columns or ("Zamkniecie" not in df.columns and "Zamknięcie" not in df.columns):
        raise ValueError("Nie znaleziono kolumn Data/Zamkniecie w pliku ze Stooq.")

    # tolerancja na diakrytyk w 'Zamknięcie'
    if "Zamkniecie" in df.columns:
        close_col = "Zamkniecie"
    else:
        close_col = "Zamknięcie"

    df = df.rename(columns={"Data": "Date", close_col: "Close"})
    # zamiana przecinka na kropkę gdyby się zdarzył
    df["Close"] = df["Close"].astype(str).str.replace(",", ".", regex=False)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")
    return df[["Close"]]

def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})
    if "Close" not in df.columns and "Zamkniecie" in df.columns:
        df = df.rename(columns={"Zamkniecie": "Close"})
    if "Close" not in df.columns and "Zamknięcie" in df.columns:
        df = df.rename(columns={"Zamknięcie": "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")
    return df[["Close"]]

# Yahoo (alternatywa / fallback)
try:
    import yfinance as yf
except Exception:
    yf = None

def from_yf(symbol: str, period: str = "max") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance nie jest zainstalowany.")
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"Yahoo: brak danych dla symbolu '{symbol}'.")
    data = data[["Close"]].dropna()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data
