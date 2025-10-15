# core/data.py — stabilny Stooq + bezpieczny fallback do Yahoo (fix EmptyDataError)
from __future__ import annotations
import io
import requests
import pandas as pd

# Mapy aliasów do Yahoo (fallback)
YF_MAP = {
    "^spx": "^GSPC",
    "^ndx": "^NDX",
    "^vix": "^VIX",
    "btcusd": "BTC-USD",
    "btcpln": "BTC-PLN",
    "eurusd": "EURUSD=X",
}

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    # tolerancja PL/EN nagłówków
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data": "Date"})
    close_col = None
    for cand in ("Close", "Zamkniecie", "Zamknięcie", "Kurs", "Price"):
        if cand in df.columns:
            close_col = cand
            break
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})
    # normalizacja wartości
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")
    if df.empty:
        raise ValueError("Po normalizacji brak danych (pusta odpowiedź lub nieparsowalne wartości).")
    return df[["Close"]]

def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Stabilny reader Stooq:
    - normalizuje symbol (usuwa ^ / =, wymusza lower-case),
    - łapie pustą/HTML-ową odpowiedź i NIE rzuca EmptyDataError,
    - próbuje parsować najpierw ręcznie (;), potem pandas,
    - na błąd robi fallback do Yahoo (jeśli dostępne).
    """
    sym = symbol.strip().lower().replace("^", "").replace("/", "").replace("=", "")
    url = f"https://stooq.pl/q/d/l/?s={sym}&i=d"

    try:
        r = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*;q=0.9"},
        )
        r.raise_for_status()
        text = (r.text or "").strip()

        # 1) pusta/HTML-owa odpowiedź -> spróbuj fallback do Yahoo
        low = text.lower()
        if not text or low.startswith("<") or "brak danych" in low or "error" in low:
            return from_yf(YF_MAP.get(sym, symbol))

        # 2) parser ręczny dla ';' (format Stooq)
        if ";" in text:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                cols = [c.strip() for c in lines[0].lstrip("\ufeff").split(";")]
                rows = []
                for ln in lines[1:]:
                    parts = [p.strip() for p in ln.split(";")]
                    if len(parts) == len(cols):
                        rows.append(parts)
                if rows:
                    df = pd.DataFrame(rows, columns=cols)
                    return _normalize_df(df)

        # 3) fallback: spróbuj pandas różnymi separatorami
        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if df is not None and not df.empty and df.shape[1] >= 2:
                    return _normalize_df(df)
            except Exception:
                continue

        # 4) jeśli nadal nic – spróbuj Yahoo
        return from_yf(YF_MAP.get(sym, symbol))

    except Exception as e:
        # Ostatecznie: spróbuj Yahoo, a jak nie zadziała – podnieś czytelny błąd
        try:
            return from_yf(YF_MAP.get(sym, symbol))
        except Exception as e2:
            raise ValueError(
                f"Nie udało się pobrać danych dla '{symbol}'.\n"
                f"Stooq URL: {url}\n"
                f"Stooq error: {e}\nYahoo error: {e2}"
            )

def from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
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
