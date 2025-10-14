from __future__ import annotations
import io
import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

SYMBOL_MAP = {
    "^spx": "^spx", "spx": "^spx",
    "^ndx": "^ndx", "ndx": "^ndx",
    "eurusd": "eurusd", "btcusd": "btcusd", "btcpln": "btcpln",
    "^vix": "^vix",
}

# --- helper: znajdź kolumnę po wielu możliwych nazwach (również z diakrytykami/odstępami)
def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    norm = {str(c).strip().lower().replace("ę","e"): c for c in cols}
    for cand in candidates:
        key = cand.strip().lower().replace("ę","e")
        if key in norm:
            return norm[key]
    return None

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Pusty zestaw danych z serwera – sprawdź symbol lub źródło.")

    # ujednolić nagłówki (czasem stooq dodaje spacje)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = _find_col(df.columns, ["date", "data"])
    close_col = _find_col(df.columns, ["close", "zamkniecie", "zamknięcie", "kurs", "price"])

    if not date_col or not close_col:
        raise ValueError(
            "Nie rozpoznano kolumn daty/zamknięcia. W CSV powinny być Date/Data i Close/Zamkniecie."
        )

    out = df.rename(columns={date_col: "Date", close_col: "Close"})[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None)
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna().sort_values("Date").set_index("Date")
    if out.empty:
        raise ValueError("Po normalizacji nie ma danych (same braki lub nieparsowalne wartości).")
    return out

def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _normalize_df(df)

def from_stooq(symbol: str) -> pd.DataFrame:
    sym = SYMBOL_MAP.get(symbol.lower().strip(), symbol.lower().strip())
    url = STOOQ_URL.format(symbol=sym)

    # użyj requests, żeby wychwycić HTML/404 i inne niespodzianki
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    text = r.text.strip()

    # Stooq przy błędnym symbolu potrafi zwrócić HTML – wykryj to szybko
    if text.startswith("<") and "</html>" in text.lower():
        raise ValueError(f"Stooq zwrócił HTML (symbol '{sym}' prawdopodobnie nie istnieje).")

    # niektóre odpowiedzi mają BOM/niestandardowe separatory – spróbuj standardowo, potem fallback
    for sep in (",", ";"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if not df.empty:
                return _normalize_df(df)
        except Exception:
            continue

    raise ValueError("Nie udało się sparsować CSV ze Stooq. Spróbuj inne źródło (Yahoo) lub inny symbol.")

# --- Yahoo (bez zmian) ---
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
