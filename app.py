# app.py — AI Trader by SO • v4.9.1 UI + silnik v52 (Stooq/CSV + proxy + Auto-Tune)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests, time, io

from core.data import from_csv, from_stooq
from core.signals import SignalParams, compute_features, partial_signals, ensemble_score, dynamic_thresholds
from core.sentiment import heuristic_from_vix
from core.backtest import backtest, metrics
from core.autotune import grid_space, walk_forward
from core.risk import volatility_target_position
# --- Left: input data (CSV / Stooq) + preview + ręczne linki + load ---
with left:
    import io, time, requests

    st.markdown("### Input data ↩️")
    src = st.radio("Source", ["Upload CSV", "Stooq"], horizontal=True, index=1)
    interval = st.selectbox("Interval", ["D1"], index=0, disabled=True)  # placeholder na przyszłość
    start_date = st.date_input("Backtest start (YYYY-MM-DD)", pd.to_datetime("2022-01-01"))
    symbol = st.text_input("Ticker", value="btcpln")

    # Upload CSV (pokazujemy nazwę/rozmiar po wgraniu)
    if src == "Upload CSV":
        csv_file = st.file_uploader("CSV (Date/Data, Close/Zamknięcie)", type=["csv"], label_visibility="collapsed")
        if csv_file is not None:
            size_kb = f"{(getattr(csv_file, 'size', 0)/1024):.1f} KB"
            st.caption(f"📎 Wczytano: **{csv_file.name}** {size_kb}")
    else:
        csv_file = None

    sep_choice = st.selectbox("Separator", ["Auto", ",", ";", "\\t"], index=0,
                              help="Wymuś separator jeśli parser się myli")

    # Trzy równe przyciski w jednym rzędzie
    a1, a2, a3 = st.columns(3, gap="small")
    with a1:
        primary_click = st.button("📥 Stooq" if src=="Stooq" else "📎 Wgraj CSV", use_container_width=True)
    with a2:
        preview_click = st.button("🔎 Preview first lines (Stooq)", use_container_width=True, disabled=(src!="Stooq"))
    with a3:
        load_click = st.button("⚡ Load data", type="primary", use_container_width=True)

    # Podgląd pierwszych linii dla Stooq
    if preview_click and src=="Stooq":
        try:
            sym = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
            url = f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"
            r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv"})
            r.raise_for_status()
            st.code("\n".join((r.text or "").splitlines()[:6]) or "(pusto)", language="text")
        except Exception as e:
            st.error(f"Preview failed: {e}")

    # Flagi stanu (gdyby nie istniały)
    st.session_state.setdefault("data_ok", False)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("used_source", None)

    # Właściwe ładowanie danych
    if load_click:
        try:
            if src == "Upload CSV":
                if not csv_file:
                    st.warning("Wgraj plik CSV (Date/Data i Close/Zamknięcie).")
                    st.session_state.update(data_ok=False, df=None, used_source=None)
                else:
                    _df = from_csv(csv_file)
                    st.session_state.update(df=_df, used_source="CSV", data_ok=True)
                    st.success(f"✅ CSV OK: {len(_df)} wierszy.")
            else:
                # Stooq
                sym = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
                forced = None
                if sep_choice != "Auto":
                    forced = "\t" if sep_choice == "\\t" else sep_choice
                else:
                    sniff_url = f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"
                    r = requests.get(sniff_url, timeout=12, headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv"})
                    r.raise_for_status()
                    header = (r.text or "").splitlines()[0] if (r.text or "") else ""
                    if ";" in header: forced = ";"
                    elif "," in header: forced = ","
                    elif "\t" in header: forced = "\t"
                    if not header or header.lstrip().startswith("<"):
                        raise ValueError("Stooq zwrócił pusty/HTML – spróbuj ponownie lub użyj CSV.")

                _df = from_stooq(symbol, forced_sep=forced)
                st.session_state.update(df=_df, used_source="Stooq", data_ok=True)
                st.success(f"✅ Stooq OK: {len(_df)} wierszy. (sep={forced or 'auto'})")

        except Exception as e:
            # Reset i komunikaty + linki ręczne + jednorazowy proxy-fallback
            st.session_state.update(data_ok=False, df=None, used_source=None)
            st.error(f"❌ Błąd wczytywania: {e}")

            sym_norm = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
            direct_url = f"https://stooq.pl/q/d/l/?s={sym_norm}&i=d"
            proxy_url  = f"https://r.jina.ai/http://stooq.pl/q/d/l/?s={sym_norm}&i=d"

            st.markdown("**🔗 Pobierz ręcznie:** "
                        f"[CSV (Stooq)]({direct_url}) • "
                        f"[Proxy]({proxy_url})  \n"
                        "Zapisz plik i wgraj go opcją **Upload CSV**.")

            # Próbujemy raz pobrać proxy po stronie serwera i załadować automatycznie
            try:
                rproxy = requests.get(proxy_url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
                rproxy.raise_for_status()
                txt = (rproxy.text or "").strip()
                if txt and not txt.lstrip().startswith("<"):
                    # Autodetekcja separatora -> normalizacja -> Date index, Close col
                    try:
                        df_try = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
                    except Exception:
                        df_try = None
                    if df_try is None or df_try.empty:
                        for s in (";", ",", "\t"):
                            try:
                                df_try = pd.read_csv(io.StringIO(txt), sep=s)
                                if not df_try.empty:
                                    break
                            except Exception:
                                pass
                    if df_try is not None and not df_try.empty:
                        df_try.columns = [str(c).strip() for c in df_try.columns]
                        date_col = next((c for c in ("Date","Data") if c in df_try.columns), df_try.columns[0])
                        close_col = next(
                            (c for c in ("Close","Zamkniecie","Zamknięcie","Zamk.","Kurs","Price","Adj Close") if c in df_try.columns),
                            (df_try.columns[4] if df_try.shape[1] >= 5 else df_try.columns[-1])
                        )
                        out = df_try[[date_col, close_col]].rename(columns={date_col:"Date", close_col:"Close"}).copy()
                        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None)
                        out["Close"] = pd.to_numeric(out["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                        out = out.dropna().sort_values("Date").set_index("Date")[["Close"]]
                        if not out.empty:
                            st.success("✅ Proxy działa — dane załadowano automatycznie.")
                            st.session_state.update(df=out, used_source="Stooq (proxy)", data_ok=True)
                            csv_bytes = out.reset_index().to_csv(index=False).encode("utf-8")
                            st.download_button("⬇️ Pobierz CSV (z serwera)", csv_bytes,
                                               file_name=f"{sym_norm}_stooq.csv", mime="text/csv", use_container_width=True)
                            st.stop()
            except Exception:
                pass  # zostawiamy instrukcję ręczną

    # Krótka diagnostyka po sukcesie
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
    else:
        st.warning("Najpierw wybierz **Source** i kliknij **Load data**.")


# ---------------------------------------------------------------------
# PAGE STYLE — motyw w stylu v4.9.1
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Trader by SO — v4.9.1", layout="wide")
st.markdown("""
<style>
:root{
  --bg:#0e1117; --panel:#151a21; --panel-2:#10151c; --text:#e6e9f0; --muted:#9aa3af;
  --accent:#3BAFDA; --good:#00C389; --bad:#FF5C7A; --amber:#CBA85B; --border:rgba(255,255,255,.08);
}
html, body, .block-container{background:var(--bg) !important; color:var(--text) !important;}
.block-container{max-width:1200px; padding-top:14px;}
.title-xl{font-weight:900; font-size:2.1rem; letter-spacing:.2px; margin:6px 0 18px;}
.kicker{color:var(--muted); font-weight:700; font-size:1rem; letter-spacing:.12rem; text-transform:uppercase;}
.card{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px 16px; box-shadow:0 6px 18px rgba(0,0,0,.30);}
.section-title{font-size:1.2rem; font-weight:900; margin-bottom:12px;}
label, .stSlider label, .stSelectbox label, .stNumberInput label, .stTextInput label { color: var(--text) !important; font-weight:600; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{
  background:#0E1115 !important; color:var(--text) !important; border-radius:10px; border:1px solid var(--border) !important;
}
[data-testid="stFileUploaderDropzone"]{
  background:transparent !important; border:1px dashed rgba(203,168,91,.35) !important;
  border-radius:10px !important; min-height:40px !important; padding:6px 10px !important;
}
[data-testid="stFileUploaderInstructions"], [data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] svg{ display:none !important; }
.btn-accent > button{ background:var(--accent); color:#081019; border:none; font-weight:900; width:100%; border-radius:10px; padding:.5rem .8rem;}
.btn-ghost > button{ background:#1c222b; color:var(--text); border:1px solid var(--border); width:100%; border-radius:10px; padding:.5rem .8rem;}
.stepper{display:flex; gap:8px; align-items:center;}
.stepper .val{min-width:56px; text-align:center; padding:6px 8px; background:#0e1218; border:1px solid var(--border); border-radius:8px;}
.stepper .btn{background:#1f272f; border:1px solid var(--border); padding:6px 10px; border-radius:8px; font-weight:900;}
.stepper .lab{min-width:110px; color:var(--muted); font-weight:700; letter-spacing:.02rem;}
.reco.good{border:1px solid rgba(0,195,137,.4); background:linear-gradient(180deg, rgba(0,195,137,.18), rgba(0,195,137,.05));}
.reco.bad{border:1px solid rgba(255,92,122,.4); background:linear-gradient(180deg, rgba(255,92,122,.18), rgba(255,92,122,.05));}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='kicker'>AI Trader by SO — v4.9.1</div>", unsafe_allow_html=True)
st.markdown("<div class='title-xl'>AUTO-TUNE AI • %cut-loss • %trailing • OOS</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# INPUT DATA SECTION
# ---------------------------------------------------------------------
st.markdown("<div class='card'><div class='section-title'>Input data</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.4, 1.2], gap="medium")

with c1:
    src = st.radio("Source", ["Upload CSV", "Stooq"], horizontal=True, index=1)
with c2:
    interval = st.selectbox("Interval", ["D1"], index=0, disabled=True)
with c3:
    start_bt = st.text_input("Backtest start (YYYY-MM-DD)", "2022-01-01")
with c4:
    symbol = st.text_input("Ticker", "btcpln")

st.write("")
cA, cB = st.columns([1.4, 2.6], gap="large")

with cA:
    if src == "Upload CSV":
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file is not None:
            size_kb = f"{(getattr(csv_file,'size',0)/1024):.1f} KB"
            st.caption(f"📎 {csv_file.name} • {size_kb}")
    else:
        st.markdown("<div class='btn-accent'>", unsafe_allow_html=True)
        get_stooq = st.button("📥 Stooq", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with cB:
    sep_choice = st.selectbox("Separator", ["Auto", ",", ";", "\\t"], index=0)
    if st.button("🔎 Preview first lines (Stooq)", key="pv", use_container_width=True):
        try:
            sym = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
            url = f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"
            r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv"})
            r.raise_for_status()
            st.code("\n".join((r.text or "").splitlines()[:5]) or "(empty)")
        except Exception as e:
            st.error(f"Preview error: {e}")

# stan sesji
st.session_state.setdefault("data_ok", False)
st.session_state.setdefault("df", None)
st.session_state.setdefault("used_source", None)

st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
load_click = st.button("Load data", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if load_click or (src == "Stooq" and 'get_stooq' in locals() and get_stooq):
    try:
        if src == "Upload CSV":
            if not csv_file:
                st.warning("Wgraj CSV z kolumnami Date/Data i Close/Zamknięcie.")
                st.session_state.update(data_ok=False, df=None, used_source=None)
            else:
                df = from_csv(csv_file)
                st.session_state.update(df=df, data_ok=True, used_source="CSV")
                st.success(f"✅ CSV loaded: {len(df)} rows.")
        else:
            forced = None
            if sep_choice != "Auto":
                forced = "\t" if sep_choice == "\\t" else sep_choice
            df = from_stooq(symbol, forced_sep=forced)
            st.session_state.update(df=df, data_ok=True, used_source="Stooq")
            st.success(f"✅ Stooq OK: {len(df)} rows.")
    except Exception as e:
        st.session_state.update(data_ok=False, df=None, used_source=None)
        st.error(f"❌ Load error: {e}")
        sym = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
        direct = f"https://stooq.pl/q/d/l/?s={sym}&i=d"
        proxy  = f"https://r.jina.ai/http://stooq.pl/q/d/l/?s={sym}&i=d"
        st.markdown(f"[Pobierz CSV (Stooq)]({direct})  •  [Proxy]({proxy})")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# DATA GATE
# ---------------------------------------------------------------------
if not st.session_state.get("data_ok") or st.session_state.get("df") is None:
    st.info("Najpierw wybierz **Source** i kliknij **Load data**.")
    st.stop()

df = st.session_state.df.copy()
close = df["Close"].dropna()


# ---------------------------------------------------------------------
# PARAMETERS & RISK (v4.9.1 style)
# ---------------------------------------------------------------------
def stepper(label:str, key:str, value:int, minv:int, maxv:int, step:int=1):
    if key not in st.session_state: st.session_state[key] = value
    c1, c2, c3 = st.columns([1,1,1], gap="small")
    with c1:
        if st.button("–", key=f"minus_{key}", use_container_width=True):
            st.session_state[key] = max(minv, st.session_state[key]-step)
    with c2: st.markdown(f"<div class='val stepper val'>{st.session_state[key]}</div>", unsafe_allow_html=True)
    with c3:
        if st.button("+", key=f"plus_{key}", use_container_width=True):
            st.session_state[key] = min(maxv, st.session_state[key]+step)
    return st.session_state[key]

st.markdown("<div class='card' style='margin-top:14px'><div class='section-title'>Parameters & risk</div>", unsafe_allow_html=True)
pc1, pc2, pc3 = st.columns(3, gap="large")
with pc1:
    ema_fast = stepper("EMA fast", "ema_fast", 15, 5, 60)
    rsi_win  = stepper("RSI (period)", "rsi_win", 14, 5, 40)
with pc2:
    ema_mid  = stepper("EMA mid", "ema_mid", 40, 10, 120, 2)
    atr_win  = stepper("ATR (period)", "atr_win", 14, 5, 40)
with pc3:
    ema_slow = stepper("EMA slow", "ema_slow", 100, 20, 300, 5)

rsi_buy  = st.slider("RSI entry floor", 10, 60, 45)
rsi_sell = st.slider("RSI exit ceiling", 40, 90, 60)
st.markdown("</div>", unsafe_allow_html=True)

p = SignalParams()
p.ma_fast, p.ma_mid, p.ma_slow = ema_fast, ema_mid, ema_slow
p.rsi_window, p.rsi_buy, p.rsi_sell = rsi_win, rsi_buy, rsi_sell


# ---------------------------------------------------------------------
# SENTIMENT
# ---------------------------------------------------------------------
try:
    vix = from_stooq("^vix")["Close"]
    sent = heuristic_from_vix(vix).reindex(close.index).ffill()
except Exception:
    sent = pd.Series(0, index=close.index)


# ---------------------------------------------------------------------
# SIGNALS + RECOMMENDATION
# ---------------------------------------------------------------------
feat = compute_features(close, p)
sig  = partial_signals(feat, p)
score = ensemble_score(sig, sent, p)
buy_thr, sell_thr = dynamic_thresholds(score, p)

last_score = float(score.iloc[-1])
buy_now = float(buy_thr.iloc[-1] if isinstance(buy_thr, pd.Series) else buy_thr)
sell_now = float(sell_thr.iloc[-1] if isinstance(sell_thr, pd.Series) else sell_thr)

if last_score >= buy_now: action, rec_cl = "KUP / AKUMULUJ", "good"
elif last_score <= sell_now: action, rec_cl = "SPRZEDAJ / REDUKUJ", "bad"
else: action, rec_cl = "TRZYMAJ", ""

st.markdown(
    f"<div class='card reco {rec_cl}'><b>🧭 Rekomendacja:</b> {action}<br>"
    f"<span style='color:var(--muted)'>Score={last_score:.2f}, BUY_thr={buy_now:.2f}, SELL_thr={sell_now:.2f}</span></div>",
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------
# CHART + BACKTEST + AUTO-TUNE (Light / Full)
# ---------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=feat.index, y=feat["Close"], name="Close", mode="lines"))
st.plotly_chart(fig, use_container_width=True, theme=None)

def _quick_space():
    return {"rsi_window":[10,14,20],"rsi_buy":[25,30,35],"rsi_sell":[65,70,75],
            "ma_fast":[10,20],"ma_mid":[50,100],"ma_slow":[150],"bb_window":[20,30],"bb_std":[1.5,2.0],
            "w_rsi":[0.3,0.5,0.7],"w_ma":[0.3,0.5,0.7],"w_bb":[0.0,0.2,0.4],
            "w_breakout":[0.0,0.2,0.4],"w_sent":[0.0,0.2],
            "percentile_window":[60,120],"percentile_mode":[True]}

def _run_walk_forward_safely(space, folds, cost_bps):
    dummy_sent = pd.Series(0, index=close.index)
    try: return walk_forward(close, space=space, folds=folds, cost_bps=cost_bps)
    except: return walk_forward(close, dummy_sent, space=space, folds=folds, cost_bps=cost_bps)

def _apply_best_params(best_params):
    updates = {}
    keymap = {"rsi_window":"rsi_win","rsi_buy":"rsi_buy","rsi_sell":"rsi_sell"}
    for k,v in (best_params or {}).items():
        if k in keymap: updates[keymap[k]]=v
    if updates:
        import time
        st.session_state.update(updates)
        st.success("✅ Zastosowano najlepsze parametry — odświeżam widok…")
        time.sleep(0.3)
        st.rerun()

def _autotune(profile:str):
    st.markdown(f"### 🔁 {profile} Auto-Tune")
    try:
        if profile=="Light": space=_quick_space(); folds=2; cost=10
        else: space=grid_space(); folds=4; cost=10
        results, stab=_run_walk_forward_safely(space, folds, cost)
        best=max(results, key=lambda r:r.get('metrics_os',{}).get('sharpe',0))
        _apply_best_params(best.get("best") or best.get("params"))
    except Exception as e:
        st.error(f"Auto-Tune błąd: {e}")

b1,b2,b3=st.columns([1,1,1])
with b1:
    st.markdown("<div class='btn-accent'>", unsafe_allow_html=True)
    if st.button("⚡ Light Auto-Tune", use_container_width=True): _autotune("Light")
    st.markdown("</div>", unsafe_allow_html=True)
with b2:
    st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
    if st.button("🔁 Full Auto-Tune", use_container_width=True): _autotune("Full")
    st.markdown("</div>", unsafe_allow_html=True)
with b3:
    st.button("⚡ Recompute", use_container_width=True)
