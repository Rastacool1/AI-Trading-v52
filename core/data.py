# core/data.py — Stooq/CSV loader (proxy fallback + ręczny import)
from __future__ import annotations

import io
import time
import csv
from typing import Optional, Iterable

import pandas as pd
import requests

__all__ = ["from_stooq", "from_csv", "direct_stooq_url", "proxy_stooq_url"]


def _norm_symbol(symbol: str) -> str:
    return (
        symbol.strip().lower()
        .replace("^", "")
        .replace("/", "")
        .replace("=", "")
    )


def direct_stooq_url(symbol: str) -> str:
    return f"https://stooq.pl/q/d/l/?s={_norm_symbol(symbol)}&i=d"


def proxy_stooq_url(symbol: str) -> str:
    # proste proxy typu „read-only fetch”, często omija anty-boty
    return f"https://r.jina.ai/http://stooq.pl/q/d/l/?s={_norm_symbol(symbol)}&i=d"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Pusty DataFrame.")

    df.columns = [str(c).strip() for c in df.columns]

    # Kandydaci na Date / Close (PL/EN + skróty Stooq)
    date_col = next((c for c in ("Date", "Data") if c in df.columns), df.columns[0])
    close_col = next(
        (
            c
            for c in (
                "Close", "Zamkniecie", "Zamknięcie", "Zamk.", "Zamk", "Kurs", "Price", "Adj Close"
            )
            if c in df.columns
        ),
        (df.columns[4] if df.shape[1] >= 5 else df.columns[-1]),
    )

    out = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"}).copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None)
    out["Close"] = pd.to_numeric(out["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out = out.dropna().sort_values("Date").set_index("Date")
    if out.empty:
        raise ValueError("Po normalizacji brak danych.")
    return out[["Close"]]


def _sniff_sep(sample: str) -> Optional[str]:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except Exception:
        head = sample.splitlines()[0] if sample else ""
        if ";" in head: return ";"
        if "," in head: return ","
        if "\t" in head: return "\t"
    return None


def _try_text(text: str) -> Optional[pd.DataFrame]:
    if not text:
        return None
    # 1) autodetekcja
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if df is not None and not df.empty and df.shape[1] >= 2:
            return df
    except Exception:
        pass
    # 2) sniffer + popularne sepy
    seps: Iterable[str] = tuple(filter(None, (_sniff_sep(text[:5000]),))) or (";", ",", "\t")
    for s in seps:
        try:
            df = pd.read_csv(io.StringIO(text), sep=s)
            if df is not None and not df.empty and df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return None


def from_stooq(symbol: str, forced_sep: str | None = None) -> pd.DataFrame:
    """
    Pobierz dzienne notowania ze Stooq → DataFrame z indexem Date i kolumną Close.
    - automatyczna detekcja separatora
    - fallback: proxy r.jina.ai
    - opcjonalne forced_sep: ',', ';', '\\t'
    """
    url = f"{direct_stooq_url(symbol)}&_={int(time.time())}"

    # 1) bezpośrednio — najpierw spróbuj od razu parsować
    try:
        if forced_sep:
            df = pd.read_csv(url, sep=forced_sep, engine="python")
        else:
            df = pd.read_csv(url, sep=None, engine="python")
        if df is not None and not df.empty:
            return _normalize_df(df)
    except Exception:
        pass

    # 2) pobierz jako tekst i próbuj ręcznie
    try:
        r = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,text/plain,*/*;q=0.9",
                "Referer": "https://stooq.pl/",
            },
        )
        r.raise_for_status()
        txt = (r.text or "").strip()
    except Exception as e:
        raise ValueError(f"Stooq: błąd HTTP ({url}): {e}")

    if not txt or txt.lstrip().startswith("<"):
        # 3) proxy fallback (często działa na hostingach)
        purl = f"{proxy_stooq_url(symbol)}&_={int(time.time())}"
        pr = requests.get(purl, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        pr.raise_for_status()
        txt = (pr.text or "").strip()

    if not txt or txt.lstrip().startswith("<"):
        raise ValueError(f"Stooq: pusty/HTML-owy response ({url}).")

    if forced_sep:
        df2 = pd.read_csv(io.StringIO(txt), sep=forced_sep, engine="python")
        return _normalize_df(df2)

    df2 = _try_text(txt)
    if df2 is not None:
        return _normalize_df(df2)

    # 4) inne kodowania + manual
    for enc in ("utf-8-sig", "cp1250", "iso-8859-2"):
        try:
            dec = txt.encode("utf-8", errors="ignore").decode(enc, errors="ignore")
            df3 = _try_text(dec)
            if df3 is not None:
                return _normalize_df(df3)
        except Exception:
            continue

    raise ValueError(f"Nie udało się sparsować CSV ze Stooq ({url}).")


def from_csv(file) -> pd.DataFrame:
    """Wczytaj CSV z uploadu (autodetekcja sep + fallbacki)."""
    try:
        df = pd.read_csv(file, sep=None, engine="python")
        if df is not None and not df.empty:
            return _normalize_df(df)
    except Exception:
        pass

    try:
        file.seek(0)
    except Exception:
        pass
    try:
        data = file.read()
        text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    except Exception:
        text = None

    if text:
        df2 = _try_text(text)
        if df2 is not None:
            return _normalize_df(df2)

    raise ValueError("Nie udało się odczytać CSV (sprawdź separator i nagłówki).")
