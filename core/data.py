# core/data.py — ONLY Stooq (robust), + CSV
from __future__ import annotations
import io
import requests
import pandas as pd

# --- helpers ---
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})
    close_col = None
    for cand in ("Close", "Zamkniecie", "Zamknięcie", "Kurs", "Price"):
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        raise ValueError("Brak kolumny Close/Zamkniecie.")
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # liczby z przecinkiem i daty
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna().sort_values("Date").set_index("Date")
    if df.empty:
        raise ValueError("Po normalizacji brak danych.")
    return df[["Close"]]

def _parse_semicolon_csv(text: str) -> pd.DataFrame | None:
    if not text or "<html" in text.lower():
        return None
    text = text.lstrip("\ufeff")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or ";" not in lines[0]:
        return None
    cols = [c.strip() for c in lines[0].split(";")]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(";")]
        if len(parts) == len(cols):
            rows.append(parts)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=cols)

# --- ONLY STOOQ ---
def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Stabilne pobieranie danych dziennych ze Stooq.
    Próbuje kilka bazowych adresów (https/http, .pl/.com).
    Nie ma żadnych fallbacków do innych źródeł.
    """
    sym = symbol.strip().lower().replace("^", "").replace("/", "").replace("=", "")
    bases = [
        "https://stooq.pl", "http://stooq.pl",
        "https://stooq.com", "http://stooq.com",
    ]
    last_err = None
    for base in bases:
        url = f"{base}/q/d/l/?s={sym}&i=d"
        try:
            r = requests.get(
                url, timeout=10,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*;q=0.9"},
            )
            r.raise_for_status()
            text = (r.text or "").strip()
            low = text.lower()
            if not text or low.startswith("<") or "brak danych" in low or "error" in low:
                last_err = f"empty/HTML at {url}"
                continue

            # 1) ręczny parser na ';'
            df = _parse_semicolon_csv(text)
            if df is not None:
                return _normalize_df(df)

            # 2) na wszelki wypadek: pandas na różnych separatorach
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

# --- CSV (upload w UI) ---
def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _normalize_df(df)
