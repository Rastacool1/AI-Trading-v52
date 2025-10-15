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
# helpers
# =========================

def _build_stooq_url(symbol: str) -> str:
    """Normalizuje ticker i buduje URL do dziennych danych Stooq, z cache-busterem."""
    sym = (
        symbol.strip()
        .lower()
        .replace("^", "")
        .replace("/", "")
        .replace("=", "")
    )
    return f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ujednolica do indeksu 'Date' i kolumny 'Close'.
    Obsługuje PL/EN nagłówki oraz fallback pozycyjny (kol.0=Date, kol.4=Close; w ostateczności ostatnia kolumna=Close).
    """
    if df is None or df.empty:
        raise ValueError("Pusty DataFrame.")

    df.columns = [str(c).strip() for c in df.columns]

    # Mapuj datę
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})

    # Znajdź 'Close' (różne warianty)
    close_col = None
    for cand in ("Close", "Zamkniecie", "Zamknięcie", "Kurs", "Price"):
        if cand in df.columns:
            close_col = cand
            break

    # Fallback pozycyjny — typowy CSV Stooq:
    # Data;Otwarcie;Najwyzszy;Najnizszy;Zamkniecie;Wolumen
    if close_col is None and df.shape[1] >= 5:
        df = df.rename(columns={df.columns[0]: "Date", df.columns[4]: "Close"})
        close_col = "Close"

    # Ostatnia deska ratunku: jeśli >=2 kolumn, weź 0=Date, -1=Close
    if (close_col is None or "Date" not in df.columns) and df.shape[1] >= 2:
        df = df.rename(columns={df.columns[0]: "Date", df.columns[-1]: "Close"})
        close_col = "Close"

    if close_col is None or "Date" not in df.columns:
        raise ValueError("Brak kolumn Date/Close w danych.")

    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # Liczby: kropka jako separator dziesiętny (jeśli przyjdzie przecinek – zamień)
    # (Uwaga: w datach '-' i w liczbach '.' NIE są separatorami CSV)
    df["Close"] = pd.to_numeric(
        df["Close"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    # Daty
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    if df.empty:
        raise ValueError("Po normalizacji brak danych.")
    return df[["Close"]]


def _sniff_sep(sample: str) -> Optional[str]:
    """Wykrywa separator CSV z próbki tekstu."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
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
# public api
# =========================

def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Pobierz dzienne notowania ze Stooq i zwróć DataFrame [Date index, Close].
    Zero fallbacków do innych źródeł.

    Kolejność prób:
      (A) pd.read_csv(URL, sep=None, engine='python')  # autodetekcja
      (B) requests + parsowanie tekstu (sniffer + sepy) i alternatywne kodowania
      (C) ręczny parser + fallback pozycyjny kolumn
    """
    url = _build_stooq_url(symbol)

    # (A) Najprościej — pozwól pandasowi samemu wykryć separator
    try:
        df_direct = pd.read_csv(url, sep=None, engine="python")
        if df_direct is not None and not df_direct.empty and df_direct.shape[1] >= 2:
            return _normalize_df(df_direct)
    except Exception:
        pass

    # (B) Pobierz zawartość i probuj różne dekodowania + sepy
    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,text/plain,*/*;q=0.9",
                "Referer": "https://stooq.pl/",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
            },
        )
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Stooq: błąd HTTP przy pobieraniu ({url}): {e}")

    raw = resp.content or b""
    text = (resp.text or "").strip()

    # Jeżeli HTML/pustka — nic nie zrobimy (bez innych źródeł).
    if not raw or not text or text.lstrip().startswith("<"):
        raise ValueError(f"Stooq: pusty/HTML-owy response ({url}).")

    # Próby na oryginalnym tekście
    df = _try_read_text(text)
    if df is not None:
        return _normalize_df(df)

    # Alternatywne dekodowania
    for enc in ("utf-8-sig", "cp1250", "iso-8859-2"):
        try:
            decoded = raw.decode(enc, errors="ignore")
        except Exception:
            continue
        df = _try_read_text(decoded)
        if df is not None:
            return _normalize_df(df)
        manual = _manual_parse(decoded)
        if manual is not None and not manual.empty:
            return _normalize_df(manual)

    # Ostatnia próba na oryginalnym tekście
    manual = _manual_parse(text)
    if manual is not None and not manual.empty:
        return _normalize_df(manual)

    raise ValueError(f"Nie udało się sparsować CSV ze Stooq ({url}).")


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
