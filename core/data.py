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


# --- w core/data.py ---

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
        "Close", "Zamkniecie", "Zamknięcie", "Zamkn.", "Zamk", "Kurs", "Price"
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

def from_stooq(symbol: str, forced_sep: str | None = None) -> pd.DataFrame:
    """
    Pobierz dzienne notowania ze Stooq i zwróć DataFrame [Date index, Close].
    Opcjonalnie można wymusić separator: ',', ';' lub '\\t'.
    """
    url = _build_stooq_url(symbol)

    # (A) jeśli wymuszono separator — spróbuj od razu tym trybem
    if forced_sep:
        try:
            df_direct = pd.read_csv(url, sep=forced_sep, engine="python")
            if df_direct is not None and not df_direct.empty and df_direct.shape[1] >= 2:
                return _normalize_df(df_direct)
        except Exception:
            pass  # przejdź do ogólnych prób

    # (B) autodetekcja Pandas
    try:
        df_direct = pd.read_csv(url, sep=None, engine="python")
        if df_direct is not None and not df_direct.empty and df_direct.shape[1] >= 2:
            return _normalize_df(df_direct)
    except Exception:
        pass

    # (C) pobierz treść i próbuj ręcznie
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
    if not raw or not text or text.lstrip().startswith("<"):
        raise ValueError(f"Stooq: pusty/HTML-owy response ({url}).")

    # (C1) prosto z tekstu
    if forced_sep:
        try:
            df = pd.read_csv(io.StringIO(text), sep=forced_sep, engine="python")
            if df is not None and not df.empty and df.shape[1] >= 2:
                return _normalize_df(df)
        except Exception:
            pass
    else:
        df = _try_read_text(text)
        if df is not None:
            return _normalize_df(df)

    # (C2) alternatywne kodowania
    for enc in ("utf-8-sig", "cp1250", "iso-8859-2"):
        try:
            decoded = raw.decode(enc, errors="ignore")
        except Exception:
            continue

        if forced_sep:
            try:
                df = pd.read_csv(io.StringIO(decoded), sep=forced_sep, engine="python")
                if df is not None and not df.empty and df.shape[1] >= 2:
                    return _normalize_df(df)
            except Exception:
                pass
        else:
            df = _try_read_text(decoded)
            if df is not None:
                return _normalize_df(df)

        manual = _manual_parse(decoded)
        if manual is not None and not manual.empty:
            return _normalize_df(manual)

    # (C3) ostatnia próba: manual na oryginalnym tekście
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
