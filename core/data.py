# core/data.py — ONLY Stooq + CSV (robust)
from __future__ import annotations

import io
import time
from typing import Optional, Iterable

import pandas as pd
import requests


__all__ = ["from_stooq", "from_csv"]


# =========================
# Helpers
# =========================

def _build_stooq_url(symbol: str) -> str:
    """Znormalizuj ticker i zbuduj URL do dziennych notowań na Stooq."""
    sym = (
        symbol.strip()
        .lower()
        .replace("^", "")
        .replace("/", "")
        .replace("=", "")
    )
    # cache-buster (unikamy zwrotki z cache CDN)
    return f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"


def _try_read_csv_text(text: str, seps: Iterable[str]) -> Optional[pd.DataFrame]:
    """Spróbuj sparsować CSV z gotowego tekstu różnymi separatorami."""
    for sep in seps:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df is not None and not df.empty and df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return None


def _parse_semicolon_manual(text: str) -> Optional[pd.DataFrame]:
    """Ręczny parser CSV oparty na separatorze (toleruje ; i ,), BOM, CRLF."""
    if not text:
        return None
    low = text.lower()
    if "<html" in low:
        return None

    text = text.lstrip("\ufeff")  # BOM
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return None

    header = lines[0]
    # wykryj separator z nagłówka
    sep = ";" if ";" in header else ("," if "," in header else None)
    if sep is None:
        return None

    cols = [c.strip() for c in header.split(sep)]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(sep)]
        # dopasuj długość wiersza do liczby kolumn nagłówka
        if len(parts) < len(cols):
            parts += [""] * (len(cols) - len(parts))
        elif len(parts) > len(cols):
            parts = parts[:len(cols)]
        rows.append(parts)

    if not rows:
        return None

    return pd.DataFrame(rows, columns=cols)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ujednolica DataFrame do formatu:
    - index = Date
    - kolumna 'Close'
    Obsługuje nagłówki PL/EN + fallback pozycyjny.
    """
    df.columns = [str(c).strip() for c in df.columns]

    # mapowanie nazwy daty
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})

    # znajdź kolumnę Close (różne warianty)
    close_col = None
    for cand in ("Close", "Zamkniecie", "Zamknięcie", "Kurs", "Price"):
        if cand in df.columns:
            close_col = cand
            break

    # fallback pozycyjny dla klasycznego CSV Stooq:
    # Data;Otwarcie;Najwyzszy;Najnizszy;Zamkniecie;Wolumen
    if close_col is None and df.shape[1] >= 5:
        df = df.rename(columns={df.columns[0]: "Date", df.columns[4]: "Close"})
        close_col = "Close"

    if close_col is None or "Date" not in df.columns:
        raise ValueError("Brak kolumn Date/Close w danych Stooq/CSV.")

    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # liczby mogą mieć przecinek jako separator dziesiętny → zamień na kropkę
    df["Close"] = pd.to_numeric(
        df["Close"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    if df.empty:
        raise ValueError("Po normalizacji brak danych.")
    return df[["Close"]]


# =========================
# Public API
# =========================

def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Pobierz dzienne notowania ze Stooq i zwróć DataFrame z kolumną 'Close' i indeksem 'Date'.
    Tylko Stooq (bez Yahoo).
    Strategia:
      1) pd.read_csv(URL, sep=';')  — najczęstszy przypadek Stooq
      2) requests z UA/Referer + próby separatorów ('; ', ';', ',', '\\t') na tekście i alternatywnych dekodowaniach
      3) ręczny parser z autodetekcją ;/, i fallback pozycyjny kolumn
    """
    url = _build_stooq_url(symbol)

    # (1) Najprościej – często wystarcza (dla plików .; )
    try:
        df_direct = pd.read_csv(url, sep=";")
        if df_direct is not None and not df_direct.empty and df_direct.shape[1] >= 2:
            return _normalize_df(df_direct)
    except Exception:
        pass

    # (2) Pobierz treść z nagłówkami, bez agresywnej heurystyki 'error'
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
    low = text.lower() if text else ""

    # UZNAJEMY HTML tylko gdy zaczyna się od '<' albo zawiera tag HTML — samo słowo 'error' nas nie obchodzi
    if not raw or not text or low.startswith("<") or "<html" in low:
        raise ValueError(f"Stooq: pusty/HTML-owy response ({url}).")

    # próby bezpośrednio na oryginalnym tekście: ; , \t
    df = _try_read_csv_text(text, seps=("; ", ";", ",", "\t"))
    if df is not None:
        return _normalize_df(df)

    # alternatywne dekodowania (utf-8-sig / cp1250 / iso-8859-2)
    for enc in ("utf-8-sig", "cp1250", "iso-8859-2"):
        try:
            decoded = raw.decode(enc, errors="ignore")
        except Exception:
            continue
        df = _try_read_csv_text(decoded, seps=("; ", ";", ",", "\t"))
        if df is not None:
            return _normalize_df(df)

        manual = _parse_semicolon_manual(decoded)
        if manual is not None and not manual.empty:
            return _normalize_df(manual)

    # ostatnia próba: ręczny parser na oryginalnym tekście
    manual = _parse_semicolon_manual(text)
    if manual is not None and not manual.empty:
        return _normalize_df(manual)

    raise ValueError(f"Nie udało się sparsować CSV ze Stooq ({url}).")


def from_csv(file) -> pd.DataFrame:
    """
    Wczytaj lokalny CSV (upload z UI) i znormalizuj do [Date, Close].
    Autodetekcja separatora (',' lub ';').
    """
    # próbuj standardowo, jak nie — spróbuj alternatywnego separatora
    try:
        df = pd.read_csv(file)  # pandas zwykle autodetekuje ','
        if df is not None and not df.empty:
            return _normalize_df(df)
    except Exception:
        pass

    # fallback: spróbuj średnika
    file.seek(0)
    try:
        text = file.read()
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8-sig")
            except Exception:
                text = text.decode("cp1250", errors="ignore")
        df2 = _try_read_csv_text(text, seps=("; ", ";", ",", "\t"))
        if df2 is not None:
            return _normalize_df(df2)
    except Exception:
        pass

    raise ValueError("Nie udało się odczytać CSV (spróbuj innym separatorem lub poprawić nagłówki).")
