# =====================  DATA & SIGNALS  =====================
# Bezpieczne ładowanie danych + fallback + komunikaty + podgląd URL Stooq

YF_MAP = {
    "^spx": "^GSPC",
    "^ndx": "^NDX",
    "^vix": "^VIX",
    "btcusd": "BTC-USD",
    "btcpln": "BTC-PLN",
    "eurusd": "EURUSD=X",
}

def stooq_url_preview(sym: str) -> str:
    # dokładnie tak samo normalizujemy jak w core/data.py
    sym_norm = sym.strip().lower().replace("^", "").replace("/", "").replace("=", "")
    return f"https://stooq.pl/q/d/l/?s={sym_norm}&i=d"

# mały tester źródła w panelu
with st.expander("🔎 Diagnostyka źródła danych", expanded=False):
    st.write("Symbol wpisany:", f"`{symbol}`")
    st.write("Podgląd URL (Stooq):", stooq_url_preview(symbol))
    st.caption("Jeśli Stooq zwróci HTML/„Brak danych”, nastąpi automat. fallback do Yahoo (mapa aliasów wyżej).")

def safe_load_data(src_choice: str, symbol_in: str, file_obj):
    symbol_norm = symbol_in.strip()
    # 1) CSV (lokalny plik)
    if src_choice == "CSV":
        if file_obj is None:
            st.warning("Wybierz plik CSV (kolumny: Date/Data, Close/Zamkniecie).")
            return None, "CSV"
        try:
            df = from_csv(file_obj)
            return df, "CSV"
        except Exception as e:
            st.error(f"Nie udało się wczytać CSV: {e}")
            return None, "CSV"

    # 2) Stooq (pierwszy wybór) z miękkim fallbackiem do Yahoo
    if src_choice == "Stooq":
        try:
            df = from_stooq(symbol_norm)
            return df, "Stooq"
        except Exception as e_stq:
            yf_sym = YF_MAP.get(symbol_norm.lower(), symbol_norm)
            st.info(f"Stooq nie zadziałał: {e_stq}\nPróbuję Yahoo: `{yf_sym}`")
            try:
                df = from_yf(yf_sym)
                return df, f"Yahoo ({yf_sym})"
            except Exception as e_yf:
                st.error(
                    "❌ Nie udało się pobrać danych ani ze Stooq, ani z Yahoo.\n\n"
                    f"Stooq: {e_stq}\nYahoo: {e_yf}\n\n"
                    "➡️ Zmień symbol/źródło albo wgraj własny CSV."
                )
                return None, "Error"

    # 3) Yahoo (bezpośrednio)
    if src_choice == "Yahoo":
        yf_sym = YF_MAP.get(symbol_norm.lower(), symbol_norm)
        try:
            df = from_yf(yf_sym)
            return df, f"Yahoo ({yf_sym})"
        except Exception as e_yf:
            st.error(f"Yahoo zwróciło błąd dla `{yf_sym}`: {e_yf}. Spróbuj Stooq lub wgraj CSV.")
            return None, "Yahoo"

    st.error("Nieznane źródło danych.")
    return None, "Unknown"

df, used_source = safe_load_data(src, symbol, csv_file)
if df is None or df.empty:
    st.stop()

# krótka metryka diagnostyczna
with st.expander("ℹ️ Informacja o danych", expanded=False):
    st.write("Źródło użyte:", used_source)
    st.write("Liczba rekordów:", len(df))
    st.write("Zakres dat:", f"{df.index.min().date()} → {df.index.max().date()}")

close = df["Close"].dropna()
