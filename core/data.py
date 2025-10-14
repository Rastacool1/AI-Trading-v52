# core/data.py — robust Stooq reader + Yahoo fallback
from __future__ import annotations
import io
import requests
import pandas as pd

# Zostajemy przy .pl
STOOQ_URL = "https://stooq.pl/q/d/l/?s={symbol}&i=d"

# Mapy aliasów (dla wygodnego fallbacku do Yahoo)
YF_MAP = {
    "^spx": "^GSPC",
    "^ndx": "^NDX",
    "^vix": "^VIX",
    "btcusd": "BTC-USD",
    "btcpln": "BTC-PLN",
    "eurusd": "EURUSD=X",
}

def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    # tolerancja na diakrytyki i spacje
    def norm(s: str) -> str:
        return (
            str(s).strip().lower()
            .replace("ą","a").replace("ć","c").replace("ę","e").replace("ł","l")
            .replace("ń","n").replace("ó","o").replace("ś","s").replace("ź","z").replace("ż","z")
        )
    lookup = {norm(c): c for c in cols}
    for cand in candidates:
        k = norm(cand)
        if k in lookup:
            return lookup[k]
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

def _parse_stooq_text(text: str) -> pd.DataFrame | None:
    """Próbuje sparsować treść CSV Stooq różnymi separatorami i kodowaniami."""
    encodings = ("utf-8", "cp1250", "iso-8859-2")
    seps = (";", ",", "\t")
    for enc in encodings:
        try:
            decoded = text.encode("utf-8", errors="ignore").decode(enc, errors="ignore")
        except Exception:
            decoded = text  # fallback
        for sep in seps:
            try:
                df = pd.read_csv(io.StringIO(decoded), sep=sep)
                if df is not None and not df.empty and df.shape[1] >= 2:
                    return df
            except Exception:
                continue
    return None

def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _normalize_df(df)

def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Czyta dane dzienne ze Stooq. Akceptuje wpisy typu:
    BTCPLN, btcpln, ^BTCPLN (znaki ^/= usuwane; wymuszamy lower-case).
    """
    # Normalizacja symbolu: usuń znaki specjalne i wymuś małe litery
    sym = symbol.strip().lower().replace("^", "").replace("/", "").replace("=", "")
    url = STOOQ_URL.format(symbol=sym)

    # Pobierz surowy tekst CSV
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    text = r.text.strip()

    # Wykryj HTML/„brak danych”
    low = text.lower()
    if low.startswith("<") or "brak danych" in low or "error" in low:
        raise ValueError(f"Stooq nie zwrócił poprawnego CSV dla '{sym}'.")

    # Spróbuj sparsować różnymi metodami
    df = _parse_stooq_text(text)
    if df is None:
        # jeszcze prosty fallback na sepy bez zmiany kodowania
        for sep in (";", ","):
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if not df.empty:
                    break
            except Exception:
                df = None
        if df is None or df.empty:
            raise ValueError(f"Nie udało się sparsować CSV ze Stooq ({url}).")

    return _normalize_df(df)

# --- Yahoo (fallback/alternatywa) ---
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
