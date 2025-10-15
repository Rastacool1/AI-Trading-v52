# core/data.py — ONLY Stooq + CSV (robust, minimal dependencies)
from __future__ import annotations

import io
from typing import Optional, Iterable

import pandas as pd
import requests


__all__ = ["from_stooq", "from_csv"]


# =========================
# Helpers
# =========================

def _build_stooq_url(symbol: str) -> str:
    """
    Znormalizuj ticker i zbuduj URL do dziennych notowań na Stooq.
    Przykłady:
      'BTCPLN'  -> https://stooq.pl/q/d/l/?s=btcpln&i=d
      '^spx'    -> https://stooq.pl/q/d/l/?s=spx&i=d
      ' eurusd' -> https://stooq.pl/q/d/l/?s=eurusd&i=d
    """
    sym = (
        symbol.strip()
        .lower()
        .replace("^", "")
        .replace("/", "")
        .replace("=", "")
    )
    return f"https://stooq.pl/q/d/l/?s={sym}&i=d"


def _try_read_csv_text(text: str, seps: Iterable[str]) -> Optional[pd.DataFrame]:
    """
    Spróbuj sparsować CSV z gotowego tekstu różnymi separatorami.
    Zwraca DataFrame albo None.
    """
    for sep in seps:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df is not None and not df.empty and df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return None


def _parse_semicolon_manual(text: str) -> Optional[pd.DataFrame]:
    """
    „Ręczny” parser CSV oparty na średnikach — toleruje:
    - BOM (UTF-8-SIG),
    - dodatkowe średniki na końcu,
    - mieszane końce linii (\r\n / \n / \r),
    - puste wartości.
    Zwraca DataFrame albo None.
    """
    if not text or "<html" in text.lower():
        return None

    text = text.lstrip("\ufeff")  # BOM
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return None

    header = lines[0]
    if ";" not in header:
        return None

    cols = [c.strip() for c in header.split(";")]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(";")]
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
    Ujednolica DataFrame do formatu z kolumnami:
    - Date (index po normalizacji),
    - Close (kolumna numeryczna).
    Obsługuje nagłówki PL/EN oraz fallback pozycyjny, gdy nagłówków brak/spięte.
    """
    # ujednolicenie nagłówków
    df.columns = [str(c).strip() for c in df.columns]

    # mapowanie kolumn daty
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})

    # wyszukanie kolumny close (z tolerancją PL diakrytyk)
    close_col = None
    for cand in ("Close", "Zamkniecie", "Zamknięcie", "Kurs", "Price"):
        if cand in df.columns:
            close_col = cand
            break

    # fallback pozycyjny dla klasycznego CSV Stooq:
    # Data;Otwarcie;Najwyzszy;Najnizszy;Zamkniecie;Wolumen
    # kol. 0 -> Data, kol. 4 -> Zamkniecie
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
    # daty
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

    # sprzątanie i ustawienie indeksu
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
    Zero fallbacków do innych źródeł — tylko Stooq.

    Strategia:
      1) Bezpośrednio: pd.read_csv(URL, sep=';')
      2) requests -> dekodowanie (utf-8-sig / cp1250 / iso-8859-2) -> próby sep ('; ', ';', ',', '\\t')
      3) ręczny parser po ';' z dopasowaniem kolumn
    """
    url = _build_stooq_url(symbol)

    # 1) najpierw spróbuj najprostszą ścieżką (często wystarcza)
    try:
        df = pd.read_csv(url, sep=";")
        if df is not None and not df.empty and df.shape[1] >= 2:
            return _normalize_df(df)
    except Exception:
        pass

    # 2) pobierz treść i próbuj różne kodowania + sepy
    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,*/*;q=0.9",
                "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
            },
        )
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Stooq: błąd HTTP przy pobieraniu ({url}): {e}")

    raw = resp.content or b""
    text = (resp.text or "").strip()
    low = text.lower() if text else ""

    # puste ciało/HTML/komunikat serwera
    if not raw or not text or low.startswith("<") or "brak danych" in low or "error" in low:
        raise ValueError(f"Stooq: pusty/HTML-owy response ({url}).")

    # 2a) próby bezpośrednio na oryginalnym tekście
    df = _try_read_csv_text(text, seps=("; ", ";", ",", "\t"))
    if df is not None:
        return _normalize_df(df)

    # 2b) próby po innych dekodowaniach
    for enc in ("utf-8-sig", "cp1250", "iso-8859-2"):
        try:
            decoded = raw.decode(enc, errors="ignore")
        except Exception:
            continue
        df = _try_read_csv_text(decoded, seps=("; ", ";", ",", "\t"))
        if df is not None:
            return _normalize_df(df)

        # 3) ręczny parser średnikowy
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
    Akceptuje polskie nagłówki (Data/Zamknięcie/Zamkniecie).
    """
    df = pd.read_csv(file)
    return _normalize_df(df)
