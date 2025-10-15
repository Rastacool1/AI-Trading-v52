# --- lewy panel: źródło i pobieranie danych (tylko Stooq/CSV) ---
with left:
    src = st.selectbox("Źródło", ["Stooq", "CSV"])
    symbol = st.text_input("Symbol", value="btcpln", help="np. btcpln / eurusd / ^spx", placeholder="ticker")
    csv_file = st.file_uploader("CSV (Date/Data, Close/Zamknięcie)", type=["csv"])

    # stan sesji na dane
    if "data_ok" not in st.session_state:
        st.session_state.data_ok = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "used_source" not in st.session_state:
        st.session_state.used_source = None

    # przycisk: Pobierz dane
    if st.button("⬇️ Pobierz dane", use_container_width=True):
        try:
            if src == "CSV":
                if csv_file is None:
                    st.warning("Wgraj plik CSV z kolumnami Date/Data i Close/Zamknięcie.")
                    st.session_state.data_ok = False
                else:
                    _df = from_csv(csv_file)
                    st.session_state.df = _df
                    st.session_state.used_source = "CSV"
                    st.session_state.data_ok = True
                    st.success(f"✅ Wczytano dane z CSV: {len(_df)} wierszy.")
            else:  # Stooq
                _df = from_stooq(symbol)
                st.session_state.df = _df
                st.session_state.used_source = "Stooq"
                st.session_state.data_ok = True
                st.success(f"✅ Pobranie OK ze Stooq: {len(_df)} wierszy.")
        except Exception as e:
            st.session_state.data_ok = False
            st.session_state.df = None
            st.session_state.used_source = None
            st.error(f"❌ Błąd wczytywania: {e}")

    # blok informacji diagnostycznej (po udanym wczytaniu)
    if st.session_state.data_ok and st.session_state.df is not None:
        _df = st.session_state.df
        st.caption("ℹ️ Podsumowanie danych")
        cA, cB = st.columns(2)
        with cA:
            st.write(f"Źródło: **{st.session_state.used_source}**")
            st.write(f"Wiersze: **{len(_df)}**")
        with cB:
            st.write(f"Zakres: **{_df.index.min().date()} → {_df.index.max().date()}**")
            st.write(f"Ostatnie Close: **{float(_df['Close'].iloc[-1]):,.4f}**")
