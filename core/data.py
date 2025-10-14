# core/data.py — robust Stooq reader (btcpln etc.) + CSV/Yahoo helpers
from __future__ import annotations
import pandas as pd

STOOQ_URL = "https://stooq.pl/q/d/l/?s={symbol}&i=d"

SYMBOL_MAP = {
    "^spx": "^spx", "spx": "^spx",
    "^ndx": "^ndx", "ndx": "^ndx",
    "eurusd": "eurusd",
    "btcusd": "btcusd",
    "btcpln": "btcpln",
    "^vix": "^vix",
}

def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    # tolerancja na diakrytyki/spacje
    norm = {str(c).strip().lower()
            .replace("ę","e").replace("ń","n").replace("ś","s").replace("ł","l").replace("ó","o").replace("ą","a").replace("ć","c").replace("ź","z").replace("ż","z")
            : c for c in cols}
    for cand in candidates:
        key = cand.strip().lower()
        key = (key.replace("ę","e").replace("ń","n").replace("ś","s").replace("ł","l")
                  .replace("ó","o").replace("ą","a").replace("ć","c").replace("ź","z").replace("ż","z"))
        if key in norm:
            return norm[key]
    return None

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Pusty zestaw danych – sprawdź symbol lub źródło.")
    df.columns = [str(c).strip() for c in df.columns]

    date_col  = _find_col(df.columns, ["date", "data"])
    close_col = _find_col(df.columns, ["close", "zamkniecie", "zamknięcie", "kurs", "price"])

    if not date_col or not close_col:
        raise ValueError("CSV musi mieć kolumny Date/Data i Close/Zamkniecie (lub Kurs/Price).")

    out = df.rename(columns={date_col: "Date", close_col: "Close"})[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None)
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna().sort_values("Date").set_index("Date")
    if out.empty:
        raise ValueError("Po normalizacji brak danych (nieparsowalne wartości).")
    return out

def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _normalize_df(df)

def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Czyta CSV ze Stooq (np. https://stooq.pl/q/d/l/?s=btcpln&i=d)
    z obsługą separatorów ; , \t i kodowań UTF-8 / CP1250 / ISO-8859-2.
    """
    sym = SYMBOL_MAP.get(symbol.lower().strip(), symbol.lower().strip())
    url = STOOQ_URL.format(symbol=sym)

    # próby: różne kodowania x separatory
    trials = [("utf-8", ";"), ("utf-8", ","), ("cp1250", ";"), ("iso-8859-2", ";"), ("utf-8", "\t")]
    last_err = None
    for enc, sep in trials:
        try:
            df = pd.read_csv(url, sep=sep, encoding=enc)
            # jeżeli Stooq zwrócił HTML albo 1 kolumnę, spróbuj dalej
            if df is None or df.empty or df.shape[1] < 2:
                continue
            # w btcpln zazwyczaj: Data;Otwarcie;Najwyzszy;Najnizszy;Zamkniecie;Wolumen
            return _normalize_df(df)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Stooq: nie udało się odczytać CSV dla symbolu '{sym}'. Ostatni błąd: {last_err}")

# --- Yahoo fallback (na wypadek, gdybyś chciał korzystać równolegle) ---
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
