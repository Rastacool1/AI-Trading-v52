# core/data.py — ONLY Stooq + CSV (ultra-robust)
from __future__ import annotations

import io
import time
import csv
from typing import Optional, Iterable

import pandas as pd
import requests

__all__ = ["from_stooq", "from_csv"]

# =========================
# Robust fetch (mirrory/proxy + nagłówki)
# =========================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv, text/plain; q=0.9, */*; q=0.8",
    "Accept-Language": "pl,en;q=0.8",
    "Referer": "https://stooq.pl/",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}

MIRRORS = [
    "https://stooq.pl", "https://stooq.com",
    "http://stooq.pl",  "http://stooq.com",
]

# proxy fetch (przez inną domenę; bywa pomocne przy limitach/CDN)
PROXIES = [
    "https://r.jina.ai/http://stooq.pl",
    "https://r.jina.ai/http://stooq.com",
]


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().lower().replace("^", "").replace("/", "").replace("=", "")


def _fetch_stooq_text(sym: str) -> tuple[str, str]:
    """Próbuje kilka baz (mirrory + proxy). Zwraca (tekst_csv, użyty_url)."""
    import random
    q = _normalize_symbol(sym)
    bases = MIRRORS + PROXIES
    last_err = None
    for base in bases:
        url = f"{base}/q/d/l/?s={q}&i=d&_={int(time.time())}{random.randint(0,9999)}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True)
            r.raise_for_status()
            text = r.text or ""
            if not text.strip() or text.lstrip().startswith("<"):
                last_err = f"html/empty from {url}"
                continue
            return text, url
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    raise ValueError(f"Stooq: nie udało się pobrać '{sym}'. Ostatni błąd: {last_err}")


def _clean_text_for_csv(txt: str) -> str:
    """Usuwa BOM/CRLF/puste linie."""
    txt = txt.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return "\n".join(lines)


# =========================
# Normalizacja i parsery
# =========================

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ujednolica do indeksu 'Date' i kolumny 'Close'.
    Obsługuje PL/EN nagłówki, skróty (np. 'Zamk.'), oraz fallback pozycyjny.
    """
    if df is None or df.empty:
        raise ValueError("Pusty DataFrame.")

    # ujednolicenie
    df.columns = [str(c).strip() for c in df.columns]

    # mapowanie daty
    date_col = None
    for cand in ("Date", "Data"):
        if cand in df.columns:
            date_col = cand
            break

    # możliwe warianty 'Close'
    close_candidates = (
        "Close", "Zamkniecie", "Zamknięcie", "Zamk.", "Zamk", "Kurs", "Price"
    )
    close_col = None
    for cand in close_candidates:
        if cand in df.columns:
            close_col = cand
            break

    # fallback pozycyjny: 0=Date, 4=Close (typowy układ Stooq)
    if (date_col is None or close_col is None) and df.shape[1] >= 5:
        if date_col is None:
            date_col = df.columns[0]
        if close_col is None:
            close_col = df.columns[4]

    # ostatnia deska: 0=Date, -1=Close
    if (date_col is None or close_col is None) and df.shape[1] >= 2:
        date_col = date_col or df.columns[0]
        close_col = close_col or df.columns[-1]

    if date_col is None or close_col is None:
        raise ValueError(f"Brak kolumn Date/Close w danych. Kolumny: {list(df.columns)}")

    # standaryzuj nazwy
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # numery i daty
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    if df.empty:
        raise ValueError("Po normalizacji brak danych.")
    return df[["Close"]]


def _sniff_sep(sample: str) -> Optional[str]:
    """Wykrywa separator CSV z próbki tekstu."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t ")
        return dialect.delimiter
    except Exception:
        # heurystyka z nagłówka (pierwsza linia)
        header = sample.splitlines()[0] if sample else ""
        if ";" in header:
            return ";"
        if "," in header:
            return ","
        if "\t" in header:
            return "\t"
    return None


def _try_read_text(text: str) -> Optional[pd.DataFrame]:
    """Próby parsowania z tekstu: autodetekcja + sniffer + kilka sep."""
    if not text:
        return None

    # 1) Autodetekcja separatora Pandas (engine='python')
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if df is not None and not df.empty and df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # 2) Sniffer + próby na najczęstszych separatorach
    sep = _sniff_sep(text[:5000])
    seps: Iterable[str] = (sep,) if sep else ("; ", ";", ",", "\t")
    for s in seps:
        try:
            df = pd.read_csv(io.StringIO(text), sep=s)
            if df is not None and not df.empty and df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return None


def _manual_parse(text: str) -> Optional[pd.DataFrame]:
    """Ręczny parser: autodetekcja ;/,/\\t, tolerancja BOM i CRLF, wyrównanie długości wierszy."""
    if not text:
        return None
    t = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if not lines:
        return None
    header = lines[0]
    sep = ";" if ";" in header else ("," if "," in header else ("\t" if "\t" in header else None))
    if not sep:
        return None
    cols = [c.strip() for c in header.split(sep)]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(sep)]
        if len(parts) < len(cols):
            parts += [""] * (len(cols) - len(parts))
        elif len(parts) > len(cols):
            parts = parts[:len(cols)]
        rows.append(parts)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=cols)


# =========================
# Public API
# =========================

def from_stooq(symbol: str, forced_sep: str | None = None) -> pd.DataFrame:
    """
    Odporny czytnik CSV ze Stooq (mirrory/proxy + autodetekcja separatora).
    Zwraca DataFrame z ['Close'] i indeksem Date.
    """
    # pobierz tekst z kilku źródeł
    text, used_url = _fetch_stooq_text(symbol)
    text = _clean_text_for_csv(text)

    # jeśli wymuszony separator – użyj od razu
    if forced_sep:
        try:
            df = pd.read_csv(io.StringIO(text), sep=forced_sep, engine="python")
            if df is not None and not df.empty and df.shape[1] >= 2:
                return _normalize_df(df)
        except Exception:
            pass  # spróbuj dalej

    # 1) parser tolerancyjny: autodetekcja
    df = _try_read_text(text)
    if df is not None:
        return _normalize_df(df)

    # 2) alternatywne kodowania + manualny parser
    for enc in ("utf-8-sig", "cp1250", "iso-8859-2"):
        try:
            decoded = text.encode("utf-8", errors="ignore").decode(enc, errors="ignore")
        except Exception:
            decoded = text
        # autodetekcja
        df2 = _try_read_text(decoded)
        if df2 is not None:
            return _normalize_df(df2)
        # manual
        manual = _manual_parse(decoded)
        if manual is not None and not manual.empty:
            return _normalize_df(manual)

    # 3) ostatnia próba – manual na oryginalnym tekście
    manual = _manual_parse(text)
    if manual is not None and not manual.empty:
        return _normalize_df(manual)

    raise ValueError(f"Nie udało się sparsować CSV ze Stooq ({used_url}).")


def from_csv(file) -> pd.DataFrame:
    """
    Wczytuje CSV z uploadu. Autodetekcja separatora (sep=None, engine='python'),
    a w razie niepowodzenia próby z ; , \\t oraz fallback ręczny.
    """
    # 1) spróbuj „po prostu”
    try:
        df = pd.read_csv(file, sep=None, engine="python")
        if df is not None and not df.empty:
            return _normalize_df(df)
    except Exception:
        pass

    # 2) spróbuj po odczycie do tekstu (gdy obiekt nie wspiera seek)
    try:
        file.seek(0)
    except Exception:
        pass
    try:
        data = file.read()
        if isinstance(data, bytes):
            # spróbuj najpierw utf-8-sig, potem cp1250
            try:
                text = data.decode("utf-8-sig")
            except Exception:
                text = data.decode("cp1250", errors="ignore")
        else:
            text = data
    except Exception:
        text = None

    if text:
        # próby parsowania
        df2 = _try_read_text(text)
        if df2 is not None:
            return _normalize_df(df2)
        manual = _manual_parse(text)
        if manual is not None and not manual.empty:
            return _normalize_df(manual)

    raise ValueError("Nie udało się odczytać CSV (sprawdź separator i nagłówki).")
