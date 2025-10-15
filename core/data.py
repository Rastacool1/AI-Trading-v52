# core/data.py — robust Stooq (.pl/.com, http/https) + smart BTCPLN Yahoo fallback
from __future__ import annotations
import io
import requests
import pandas as pd

# ---------- helpers ----------
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    # mapuj kolumny PL/EN
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})
    close_col = None
    for cand in ("Close", "Zamkniecie", "Zamknięcie", "Kurs", "Price"):
        if cand in df.columns:
            close_col = cand; break
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})
    # normalizacja
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")
    if df.empty:
        raise ValueError("Po normalizacji brak danych.")
    return df[["Close"]]

def _parse_semicolon_csv(text: str) -> pd.DataFrame | None:
    if not text or "<html" in text.lower(): return None
    text = text.lstrip("\ufeff")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or ";" not in lines[0]: return None
    cols = [c.strip() for c in lines[0].split(";")]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(";")]
        if len(parts) == len(cols): rows.append(parts)
    if not rows: return None
    return pd.DataFrame(rows, columns=cols)

# ---------- Stooq ----------
def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Stabilne pobieranie ze Stooq – próby:
    https://stooq.pl, http://stooq.pl, https://stooq.com, http://stooq.com
    """
    sym = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
    base_urls = [
        "https://stooq.pl", "http://stooq.pl",
        "https://stooq.com", "http://stooq.com",
    ]
    last_err = None
    for base in base_urls:
        url = f"{base}/q/d/l/?s={sym}&i=d"
        try:
            r = requests.get(
                url, timeout=10,
                headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv,*/*;q=0.9"},
            )
            r.raise_for_status()
            text = (r.text or "").strip()
            low = text.lower()
            if not text or low.startswith("<") or "brak danych" in low or "error" in low:
                last_err = f"empty/HTML at {url}"; continue
            # 1) parser ręczny (;)
            df = _parse_semicolon_csv(text)
            if df is not None:
                return _normalize_df(df)
            # 2) fallback pandas (różne sepy)
            for sep in (";", ",", "\t"):
                try:
                    tmp = pd.read_csv(io.StringIO(text), sep=sep)
                    if tmp is not None and not tmp.empty and tmp.shape[1] >= 2:
                        return _normalize_df(tmp)
                except Exception:
                    pass
            last_err = f"parse-failed at {url}"
        except Exception as e:
            last_err = f"{type(e).__name__} at {url}: {e}"
    raise ValueError(f"Stooq: nie udało się pobrać '{sym}'. Ostatni błąd: {last_err}")

# ---------- CSV ----------
def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _normalize_df(df)

# ---------- Yahoo (z inteligentnym BTCPLN) ----------
try:
    import yfinance as yf
except Exception:
    yf = None

def from_yf(symbol: str, period: str = "max") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance nie jest zainstalowany.")

    # specjalny przypadek: BTC-PLN nie zawsze istnieje w Yahoo → policz syntetycznie
    sym_up = symbol.upper().replace("^","")
    if sym_up in ("BTC-PLN", "BTCPLN"):
        # BTCPLN ~= (BTC-USD) * (USDPLN=X)
        btc = yf.download("BTC-USD", period=period, interval="1d", auto_adjust=True, progress=False)
        usdpln = yf.download("USDPLN=X", period=period, interval="1d", auto_adjust=True, progress=False)
        if btc is None or btc.empty or usdpln is None or usdpln.empty:
            raise ValueError("Yahoo: brak danych składowych do wyliczenia BTC-PLN.")
        df = pd.DataFrame(index=pd.to_datetime(btc.index))
        df["Close"] = btc["Close"].reindex(df.index).fillna(method="ffill") * \
                      usdpln["Close"].reindex(df.index).fillna(method="ffill")
        df = df.dropna()
        df.index.name = "Date"
        if df.empty:
            raise ValueError("Yahoo: pusta seria syntetyczna BTC-PLN.")
        return df[["Close"]]

    # standardowo
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"Yahoo: brak danych dla symbolu '{symbol}'.")
    data = data[["Close"]].dropna()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data
