# core/data.py — proste Stooq + CSV (wersja "działa jak w przykładzie")
from __future__ import annotations
import io, time
import pandas as pd
import requests

__all__ = ["from_stooq", "from_csv"]

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*;q=0.9",
    "Referer": "https://stooq.pl/",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def _norm_symbol(s: str) -> str:
    return s.strip().lower().replace("^","").replace("/","").replace("=","")

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Pusty DataFrame.")
    df.columns = [str(c).strip() for c in df.columns]

    # znajdź Date/Data i Close/Zamkniecie/Zamknięcie/Kurs/Price
    date_col = next((c for c in ("Date","Data") if c in df.columns), None)
    close_col = next((c for c in ("Close","Zamkniecie","Zamknięcie","Kurs","Price","Adj Close") if c in df.columns), None)

    # fallback pozycyjny (typowy układ stooq: 0=date, 4=close)
    if (date_col is None or close_col is None) and df.shape[1] >= 5:
        date_col = date_col or df.columns[0]
        close_col = close_col or df.columns[4]
    if (date_col is None or close_col is None) and df.shape[1] >= 2:
        date_col = date_col or df.columns[0]
        close_col = close_col or df.columns[-1]

    if date_col is None or close_col is None:
        raise ValueError(f"Brak Date/Close w danych. Kolumny: {list(df.columns)}")

    out = df[[date_col, close_col]].rename(columns={date_col:"Date", close_col:"Close"}).copy()
    out["Date"]  = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None)
    # zamiana przecinka dziesiętnego na kropkę (stooq bywa z przecinkami)
    out["Close"] = pd.to_numeric(out["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out = out.dropna().sort_values("Date").set_index("Date")
    if out.empty:
        raise ValueError("Po normalizacji brak danych.")
    return out[["Close"]]

def _read_text_to_df(text: str) -> pd.DataFrame | None:
    if not text or text.lstrip().startswith("<"):
        return None
    text = text.replace("\ufeff","").replace("\r\n","\n").replace("\r","\n")
    # najpierw auto-sep
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    # proste sepy
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None

def from_stooq(symbol: str, forced_sep: str | None = None) -> pd.DataFrame:
    """Proste pobranie CSV ze Stooq, z nagłówkami i jednym proxy-fallbackiem."""
    sym = _norm_symbol(symbol)
    url = f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"

    # 1) próba bezpośrednia
    r = requests.get(url, timeout=12, headers=UA)
    r.raise_for_status()
    text = (r.text or "").strip()

    # jeśli pusto/HTML → fallback przez proxy (często pomaga na Streamlit Cloud)
    if not text or text.lstrip().startswith("<"):
        proxy_url = f"https://r.jina.ai/http://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"
        r2 = requests.get(proxy_url, timeout=12, headers=UA)
        r2.raise_for_status()
        text = (r2.text or "").strip()

    if not text or text.lstrip().startswith("<"):
        raise ValueError(f"Stooq: pusty/HTML-owy response ({url}).")

    if forced_sep:
        df = pd.read_csv(io.StringIO(text), sep=forced_sep, engine="python")
    else:
        df = _read_text_to_df(text)
        if df is None:
            raise ValueError("Nie udało się rozpoznać separatora.")

    return _normalize_df(df)

def from_csv(file) -> pd.DataFrame:
    """Upload CSV → Date/Data + Close/Zamkniecie (lub Kurs/Price)."""
    # najpierw spróbuj 'po prostu'
    try:
        df = pd.read_csv(file, sep=None, engine="python")
        if df is not None and not df.empty:
            return _normalize_df(df)
    except Exception:
        pass

    # jeśli się nie da, czytaj jako tekst (obsługa dziwnych obiektów)
    try:
        file.seek(0)
    except Exception:
        pass
    data = file.read()
    text = data.decode("utf-8-sig") if isinstance(data, (bytes,bytearray)) else str(data)
    df = _read_text_to_df(text)
    if df is None:
        raise ValueError("Nie udało się odczytać CSV (sprawdź nagłówki/sep).")
    return _normalize_df(df)
