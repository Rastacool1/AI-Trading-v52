# core/data.py — Stooq (parser ręczny) + Yahoo fallback
from __future__ import annotations
import io
import requests
import pandas as pd

STOOQ_URL = "https://stooq.pl/q/d/l/?s={symbol}&i=d"

YF_MAP = {
    "^spx": "^GSPC",
    "^ndx": "^NDX",
    "^vix": "^VIX",
    "btcusd": "BTC-USD",
    "btcpln": "BTC-PLN",
    "eurusd": "EURUSD=X",
}

def _find_col(cols: list[str], candidates: list[str]) -> str | None:
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

    # Tolerancja na przecinek jako separator dziesiętny
    out["Close"] = (
        out["Close"].astype(str).str.replace(",", ".", regex=False)
    )

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None)
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna().sort_values("Date").set_index("Date")
    if out.empty:
        raise ValueError("Po normalizacji brak danych (nieparsowalne wartości).")
    return out

def _parse_stooq_semicolon(text: str) -> pd.DataFrame | None:
    """
    Ręczny parser CSV Stooq (separator ';', nagłówki PL np. Data;Otwarcie;...;Zamknięcie;Wolumen).
    Odrzuca puste/niepełne wiersze, usuwa BOM i białe znaki.
    """
    if not text or "<html" in text.lower():
        return None

    # Usuń ewentualny BOM
    if text[:1] == "\ufeff":
        text = text.lstrip("\ufeff")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    header = lines[0]
    if ";" not in header:
        return None

    cols = [c.strip() for c in header.split(";")]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(";")]
        if len(parts) != len(cols):
            # sporadyczne puste/ucięte linie — pomijamy
            continue
        rows.append(parts)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=cols)
    return df

def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _normalize_df(df)

def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Czyta dane dzienne ze Stooq.
    Normalizacja: usuwa ^ / = i wymusza lower-case (np. '^BTCPLN' -> 'btcpln').
    """
    sym = symbol.strip().lower().replace("^", "").replace("/", "").replace("=", "")
    url = STOOQ_URL.format(symbol=sym)

    # Pobierz surowy tekst CSV (z nagłówkiem UA dla pewności)
    r = requests.get(
        url, timeout=10,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*;q=0.9",
        },
    )
    r.raise_for_status()
    text = r.text.strip()

    # Wykryj HTML/komunikaty
    low = text.lower()
    if low.startswith("<") or "brak danych" in low or "error" in low:
        raise ValueError(f"Stooq nie zwrócił poprawnego CSV dla '{sym}'.")

    # 1) Parser ręczny dla typowego formatu Stooq (;)
    df = _parse_stooq_semicolon(text)

    # 2) Jeśli ręczny parser nie zadziałał, spróbuj pandas z kilkoma wariantami
    if df is None or df.empty:
        for enc in ("utf-8", "cp1250", "iso-8859-2"):
            for sep in (";", ",", "\t"):
                try:
                    decoded = text.encode("utf-8", errors="ignore").decode(enc, errors="ignore")
                    tmp = pd.read_csv(io.StringIO(decoded), sep=sep)
                    if tmp is not None and not tmp.empty and tmp.shape[1] >= 2:
                        df = tmp
                        break
                except Exception:
                    continue
            if df is not None:
                break

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
