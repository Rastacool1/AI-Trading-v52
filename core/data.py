# --- helpers (zostaw w pliku razem z resztą) ---
import io, requests, pandas as pd

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    # mapuj PL/EN
    if "Date" not in df.columns and "Data" in df.columns:
        df = df.rename(columns={"Data":"Date"})
    close_col = None
    for cand in ("Close","Zamkniecie","Zamknięcie","Kurs","Price"):
        if cand in df.columns:
            close_col = cand; break
    # fallback pozycyjny (typowy format Stooq: Data;Otwarcie;Najwyzszy;Najnizszy;Zamkniecie;Wolumen)
    if close_col is None:
        if df.shape[1] >= 5:
            df = df.rename(columns={df.columns[0]:"Date", df.columns[4]:"Close"})
            close_col = "Close"
        else:
            raise ValueError("Brak kolumn Date/Close w CSV (i nie można zmapować pozycyjnie).")
    if close_col != "Close":
        df = df.rename(columns={close_col:"Close"})

    # liczby i daty
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",",".", regex=False), errors="coerce")
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna().sort_values("Date").set_index("Date")
    if df.empty:
        raise ValueError("Po normalizacji brak danych.")
    return df[["Close"]]

def _parse_semicolon_csv(text: str) -> pd.DataFrame | None:
    if not text or "<html" in text.lower(): return None
    text = text.lstrip("\ufeff")
    lines = [ln for ln in (text.replace("\r\n","\n").replace("\r","\n")).split("\n") if ln.strip()]
    if not lines: return None
    # toleruj dodatkowe ; na końcu
    cols = [c.strip() for c in lines[0].split(";") if c is not None]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(";")]
        # docięcie/padding do liczby kolumn nagłówka
        if len(parts) < len(cols):
            parts = parts + [""]*(len(cols)-len(parts))
        elif len(parts) > len(cols):
            parts = parts[:len(cols)]
        rows.append(parts)
    if not rows: return None
    return pd.DataFrame(rows, columns=cols)

# --- ONLY STOOQ (utwardzone) ---
def from_stooq(symbol: str) -> pd.DataFrame:
    """
    Pobiera dzienne CSV ze Stooq (TYLKO Stooq).
    Kolejność prób:
      1) pandas.read_csv(url, sep=';')
      2) requests -> decode (utf-8-sig / cp1250 / iso-8859-2) -> pandas
      3) ręczny parser po ';' + mapowanie pozycyjne kolumn
    """
    sym = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
    url = f"https://stooq.pl/q/d/l/?s={sym}&i=d"

    # 1) Najprościej — pozwól pandasowi czytać z URL
    try:
        df = pd.read_csv(url, sep=";")
        if df is not None and not df.empty and df.shape[1] >= 2:
            return _normalize_df(df)
    except Exception:
        pass

    # 2) Pobierz bajty i próbuj różne kodowania + sepy
    try:
        r = requests.get(url, timeout=12, headers={
            "User-Agent":"Mozilla/5.0",
            "Accept":"text/csv,*/*;q=0.9",
            "Accept-Language":"pl-PL,pl;q=0.9,en;q=0.8"
        })
        r.raise_for_status()
        raw = r.content or b""
        if not raw:
            raise ValueError("pusty response body")

        for enc in ("utf-8-sig","cp1250","iso-8859-2"):
            try:
                text = raw.decode(enc, errors="ignore").strip()
            except Exception:
                continue

            # 2a) pandas z różnymi separatorami
            for sep in ("; ", ";", ",", "\t"):
                try:
                    tmp = pd.read_csv(io.StringIO(text), sep=sep)
                    if tmp is not None and not tmp.empty and tmp.shape[1] >= 2:
                        return _normalize_df(tmp)
                except Exception:
                    continue

            # 2b) ręczny parser po ';'
            df_manual = _parse_semicolon_csv(text)
            if df_manual is not None and not df_manual.empty:
                return _normalize_df(df_manual)

        raise ValueError("parse-failed (po wszystkich kodowaniach/sepach)")

    except Exception as e:
        raise ValueError(f"Stooq: nie udało się pobrać '{sym}'. Ostatni błąd: {e}")
