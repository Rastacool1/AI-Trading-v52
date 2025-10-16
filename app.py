# app.py — AI Trading Edge • Excel-style Dark Dashboard (ONLY Stooq/CSV)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.data import from_csv, from_stooq
from core.signals import (
    SignalParams, compute_features, partial_signals,
    ensemble_score, dynamic_thresholds, confidence_and_explain
)
from core.sentiment import heuristic_from_vix
from core.backtest import backtest, metrics
from core.autotune import grid_space, walk_forward
from core.risk import volatility_target_position


# -----------------------------------------------------------------------------
# PAGE & THEME
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Trading Edge — Dashboard", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#111317; --panel:#1A1F26; --panel-2:#171B21; --text:#E8ECF1; --muted:#AAB2BE;
  --accent:#3BAFDA; --good:#00C389; --bad:#FF5C7A; --amber:#CBA85B; --border:rgba(255,255,255,.06);
}
html, body, .block-container{background:var(--bg) !important; color:var(--text) !important;}
.block-container{padding-top:66px; max-width:1500px;}

/* Top bar */
.topbar{position:fixed; top:0; left:0; right:0; z-index:1000;
  background:linear-gradient(180deg, rgba(26,31,38,.97) 0%, rgba(26,31,38,.9) 100%);
  border-bottom:1px solid var(--border); backdrop-filter:blur(6px); padding:10px 14px;}
.logo{display:flex; gap:10px; align-items:center;}
.logo .mark{width:36px; height:36px; border-radius:8px; background:#1F2530; display:flex; align-items:center; justify-content:center; font-weight:800; color:var(--amber);}
.logo .title{font-weight:800; font-size:1.0rem; letter-spacing:.3px;}
/* Topbar buttons – równe wymiary */
.topbar .stButton > button { width:100% !important; height:40px !important; }

/* CSV uploader – kompakt, bez dużego białego pola/instrukcji */
[data-testid="stFileUploaderDropzone"]{
  background: transparent !important;
  border: 1px dashed rgba(203,168,91,.35) !important;
  border-radius: 10px !important;
  min-height: 40px !important;
  padding: 6px 10px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--amber) !important; }
[data-testid="stFileUploaderDropzone"] svg { display: none !important; }        /* ikona chmury off */
[data-testid="stFileUploaderInstructions"] { display:none !important; }         /* instrukcje off */
[data-testid="stFileUploaderDropzone"] label { display:none !important; }       /* label w polu off */
/* Cards */
.card{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:10px 12px; box-shadow:0 6px 16px rgba(0,0,0,.25);}
.card-2{background:var(--panel-2); border:1px solid var(--border); border-radius:14px; padding:10px 12px;}
.h1{font-weight:800; font-size:1.06rem; letter-spacing:.2px; margin-bottom:6px;}
.h2{font-weight:700; font-size:.96rem; letter-spacing:.2px; margin-bottom:4px;}
.sub{color:var(--muted); font-size:.86rem;}

/* Inputs */
label, .stSlider label, .stSelectbox label, .stNumberInput label, .stTextInput label { color: var(--text) !important; font-weight:600; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{
  background:#0E1115 !important; color:var(--text) !important; border-radius:10px; border:1px solid var(--border) !important;
}
.stFileUploader, .stFileUploader div[data-testid="stFileUploaderDropzone"]{
  background:linear-gradient(180deg, rgba(203,168,91,.18), rgba(203,168,91,.10)) !important;
  border:1px dashed rgba(203,168,91,.65) !important; border-radius:12px !important; color:var(--amber) !important;
}

/* Sliders (compact) */
.stSlider > div[data-baseweb="slider"]{ padding:2px 4px; }
.stSlider [data-baseweb="slider"] div{ background-color: transparent; }
.stSlider [role="slider"]{ background:var(--accent) !important; box-shadow:0 0 0 2px rgba(59,175,218,.25); }
.stSlider .css-1dp5vir, .stSlider .e1yxm3t61{ color:var(--text) !important; }

/* Buttons */
.stButton button{background:var(--accent); color:#0B0E12; border:none; border-radius:10px; padding:.48rem .8rem; font-weight:800; font-size:.9rem;}
.stButton button:hover{filter:brightness(1.05); transform:translateY(-1px);}
.btn-ghost button{background:#232A34; color:var(--text);}
.btn-amber button{background:var(--amber); color:#101419;}
.btn-green button{background:var(--good); color:#07140F;}

/* Recommendation */
.reco{border:1px solid rgba(203,168,91,.35); background:linear-gradient(180deg, rgba(203,168,91,.14), rgba(203,168,91,.06));}
.reco.good{border-color:rgba(0,195,137,.35); background:linear-gradient(180deg, rgba(0,195,137,.14), rgba(0,195,137,.06));}
.reco.bad{border-color:rgba(255,92,122,.35); background:linear-gradient(180deg, rgba(255,92,122,.14), rgba(255,92,122,.06));}

/* Parameter grid: left label + columns with sliders */
.row{display:grid; grid-template-columns: 90px 1fr; gap:10px; align-items:center; margin:2px 0;}
.row .name{font-weight:800; color:var(--text); text-transform:uppercase; font-size:.85rem; letter-spacing:.4px;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TOP BAR (logo + main actions)
# -----------------------------------------------------------------------------
# równe kolumny na top-bar
tb1, tb2, tb3, tb4, tb5 = st.columns(5, gap="small")
# wymuś jednakową szerokość i wysokość guzików

with tb1:
    st.markdown(
        "<div class='topbar card-2'>"
        "<div class='logo'><div class='mark'>AI</div>"
        "<div class='title'>Trading Edge — Dashboard</div></div>"
        "</div>", unsafe_allow_html=True
    )
with tb2:
    st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
    bL, bF = st.columns(2, gap="small")
    with bL:
        auto_tune_light_click = st.button("⚡ Light Auto-Tune", use_container_width=True)
    with bF:
        auto_tune_full_click  = st.button("🔁 Pełny Auto-Tune", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tb3:
    st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
    recalc_click = st.button("⚡ Przelicz", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with tb4:
    st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
    autoscale_click = st.button("🖼️ Autoskaluj", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with tb5:
    st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
    export_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# STARTER PANEL (left: source & download; right: compact parameter grid)
# -----------------------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h1'>⚙️ Ustawienia / Filtry</div>", unsafe_allow_html=True)

left, right = st.columns([1.0, 3.0], gap="large")

# --- Left: data source & download gate (ONLY Stooq/CSV) ---
with left:
    src = st.selectbox("Źródło", ["Stooq", "CSV"])
    symbol = st.text_input("Symbol", value="btcpln", help="np. btcpln / eurusd / ^spx", placeholder="ticker")
    csv_file = st.file_uploader("CSV (Date/Data, Close/Zamknięcie)", type=["csv"])

    # 👇 ten blok MUSI być wcięty (4 spacje) – pokazuje nazwę pliku po wgraniu
    if csv_file is not None:
        size_kb = f"{(csv_file.size/1024):.1f} KB" if hasattr(csv_file, "size") else ""
        st.caption(f"📎 Wczytano: **{csv_file.name}** {size_kb}")

    # separator (musi być zdefiniowany zanim użyjemy go w kliknięciu)
    sep_choice = st.selectbox("Separator (opcjonalnie)", ["Auto", ",", ";", "\\t"], index=0,
                              help="Wymuś separator jeśli parser się myli")

    # quick tester
    if st.button("🔎 Test Stooq (podgląd pierwszych linii)", use_container_width=True):
        try:
            import requests, time
            sym = symbol.strip().lower().replace("^","").replace("/","").replace("=","")
            test_url = f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"
            r = requests.get(test_url, timeout=12, headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv"})
            r.raise_for_status()
            preview = "\n".join((r.text or "").splitlines()[:5])
            st.code(preview or "(pusto)", language="text")
        except Exception as e:
            st.error(f"Test nie powiódł się: {e}")

    # session state for data gate
    if "data_ok" not in st.session_state:
        st.session_state.data_ok = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "used_source" not in st.session_state:
        st.session_state.used_source = None

        # download button
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
                forced = None
                if sep_choice != "Auto":
                    forced = "\t" if sep_choice == "\\t" else sep_choice
                else:
                    # Auto: zrób szybki podgląd i zgadnij separator
                    import requests, time
                    sym = symbol.strip().lower().replace("^", "").replace("/", "").replace("=", "")
                    test_url = f"https://stooq.pl/q/d/l/?s={sym}&i=d&_={int(time.time())}"
                    r = requests.get(test_url, timeout=12, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv"})
                    r.raise_for_status()
                    head = (r.text or "").splitlines()[:1]
                    header = head[0] if head else ""
                    if ";" in header:
                        forced = ";"
                    elif "," in header:
                        forced = ","
                    elif "\t" in header:
                        forced = "\t"
                    # jeśli header pusty/HTML – oddaj czytelny błąd
                    if not header or header.lstrip().startswith("<"):
                        raise ValueError("Stooq zwrócił pusty/HTML – spróbuj ponownie za chwilę lub użyj CSV.")

                _df = from_stooq(symbol, forced_sep=forced)
                st.session_state.df = _df
                st.session_state.used_source = "Stooq"
                st.session_state.data_ok = True
                st.success(f"✅ Pobranie OK ze Stooq: {len(_df)} wierszy. (sep = {forced or 'auto'})")

        except Exception as e:
            st.session_state.data_ok = False
            st.session_state.df = None
            st.session_state.used_source = None
            st.error(f"❌ Błąd wczytywania: {e}")

    # diagnostics after successful load
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

# --- Right: compact parameter grid (names left, sliders inline) ---
with right:
    p = SignalParams()

    # RSI
    st.markdown("<div class='row'><div class='name'>RSI</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1,1,1], gap="small")
    with r1: p.rsi_window = st.slider("RSI window", 5, 30, p.rsi_window, key="rsi_w")
    with r2: p.rsi_buy    = st.slider("RSI BUY",    10, 50, p.rsi_buy, key="rsi_b")
    with r3: p.rsi_sell   = st.slider("RSI SELL",   50, 90, p.rsi_sell, key="rsi_s")
    st.markdown("</div>", unsafe_allow_html=True)

    # MA
    st.markdown("<div class='row'><div class='name'>MA</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns([1,1,1], gap="small")
    with m1: p.ma_fast = st.slider("MA fast", 5, 50, p.ma_fast, key="ma_f")
    with m2: p.ma_mid  = st.slider("MA mid", 20, 100, p.ma_mid, key="ma_m")
    with m3: p.ma_slow = st.slider("MA slow", 20, 250, p.ma_slow, key="ma_s")
    st.markdown("</div>", unsafe_allow_html=True)

    # BB
    st.markdown("<div class='row'><div class='name'>BB</div>", unsafe_allow_html=True)
    b1, b2 = st.columns([1,1], gap="small")
    with b1: p.bb_window = st.slider("BB window", 10, 40, p.bb_window, key="bb_w")
    with b2: p.bb_std    = st.slider("BB std",    1.0, 3.0, p.bb_std, key="bb_s")
    st.markdown("</div>", unsafe_allow_html=True)

    # Wagi
    st.markdown("<div class='row'><div class='name'>Wagi</div>", unsafe_allow_html=True)
    w1, w2, w3, w4, w5 = st.columns([1,1,1,1,1], gap="small")
    with w1: p.w_rsi      = st.slider("w_rsi",      0.0, 1.0, p.w_rsi, key="wg_rsi")
    with w2: p.w_ma       = st.slider("w_ma",       0.0, 1.0, p.w_ma,  key="wg_ma")
    with w3: p.w_bb       = st.slider("w_bb",       0.0, 1.0, p.w_bb,  key="wg_bb")
    with w4: p.w_breakout = st.slider("w_breakout", 0.0, 1.0, p.w_breakout, key="wg_br")
    with w5: p.w_sent     = st.slider("w_sent",     0.0, 1.0, p.w_sent, key="wg_se")
    st.markdown("</div>", unsafe_allow_html=True)

    # Progi (percentyle)
    st.markdown("<div class='row'><div class='name'>Progi</div>", unsafe_allow_html=True)
    pr1, pr2 = st.columns([1,1], gap="small")
    with pr1: p.percentile_mode = st.checkbox("Progi dynamiczne", value=True, key="perc_on")
    with pr2: p.percentile_window = st.slider("Okno percentyli", 30, 180, p.percentile_window, key="perc_win")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # /card


# -----------------------------------------------------------------------------
# DATA GATE — require explicit download first
# -----------------------------------------------------------------------------
if not st.session_state.get("data_ok") or st.session_state.get("df") is None:
    st.info("Najpierw wybierz **Źródło** i **Symbol/CSV**, a następnie kliknij **„Pobierz dane”**.")
    st.stop()

df = st.session_state.df.copy()
used_source = st.session_state.used_source or "Stooq"
close = df["Close"].dropna()


# -----------------------------------------------------------------------------
# SENTIMENT (optional, silent fail)
# -----------------------------------------------------------------------------
try:
    vix = from_stooq("^vix")["Close"]
    sent = heuristic_from_vix(vix).reindex(close.index).fillna(method="ffill")
except Exception:
    sent = pd.Series(0, index=close.index)


# -----------------------------------------------------------------------------
# SIGNALS & DECISION
# -----------------------------------------------------------------------------
feat = compute_features(close, p)
sig  = partial_signals(feat, p)
score = ensemble_score(sig, sent, p)
buy_thr, sell_thr = dynamic_thresholds(score, p)

last_score = float(score.iloc[-1])
buy_now = float(buy_thr.iloc[-1] if isinstance(buy_thr, pd.Series) else buy_thr)
sell_now = float(sell_thr.iloc[-1] if isinstance(sell_thr, pd.Series) else sell_thr)

if last_score >= buy_now:
    action, rec_cl = "KUP / AKUMULUJ", "good"
elif last_score <= sell_now:
    action, rec_cl = "SPRZEDAJ / REDUKUJ", "bad"
else:
    action, rec_cl = "TRZYMAJ", ""

# export signals CSV
export_csv = pd.DataFrame({
    "Date": feat.index, "Close": feat["Close"], "RSI": feat.get("RSI", pd.NA), "Score": score
}).to_csv(index=False).encode("utf-8")
with export_placeholder:
    st.download_button("⬇️ Eksport sygnałów (CSV)", export_csv,
                       file_name=f"signals_{(symbol or 'asset').replace('^','')}.csv",
                       use_container_width=True)


# -----------------------------------------------------------------------------
# RECOMMENDATION BLOCK
# -----------------------------------------------------------------------------
st.markdown(
    f"<div class='card reco {rec_cl}'><div class='h1'>🧭 Rekomendacja na dziś</div>"
    f"<div class='sub'>Źródło: {used_source} • Score: {last_score:.2f} • BUY_thr: {buy_now:.2f} • SELL_thr: {sell_now:.2f}</div>"
    f"<h2 style='margin:8px 0 4px 0; font-size:1.8rem;'>{action}</h2>"
    f"</div>", unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# MAIN CHART
# -----------------------------------------------------------------------------
st.markdown("<div class='card-2'>", unsafe_allow_html=True)
st.markdown("<div class='h1'>📈 Wykres ceny (markery sygnałów)</div>", unsafe_allow_html=True)

range_choice = st.radio("Zakres", ["1M","3M","6M","YTD","1Y","3Y","MAX"], horizontal=True)

def pick_range(idx, choice):
    if len(idx)==0: return idx
    end = idx[-1]
    if choice=="1M": start = end - pd.DateOffset(months=1)
    elif choice=="3M": start = end - pd.DateOffset(months=3)
    elif choice=="6M": start = end - pd.DateOffset(months=6)
    elif choice=="YTD": start = pd.Timestamp(end.year,1,1)
    elif choice=="1Y": start = end - pd.DateOffset(years=1)
    elif choice=="3Y": start = end - pd.DateOffset(years=3)
    else: start = idx[0]
    return idx[(idx>=start)&(idx<=end)]

idx = pick_range(feat.index, range_choice)
fsel = feat.loc[idx]; scsel = score.loc[idx]
bsel = buy_thr.loc[idx] if isinstance(buy_thr, pd.Series) else pd.Series(buy_thr, index=idx)
ssel = sell_thr.loc[idx] if isinstance(sell_thr, pd.Series) else pd.Series(sell_thr, index=idx)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fsel.index, y=fsel["Close"], name="Close", mode="lines"))
buy_mask = scsel >= bsel; sell_mask = scsel <= ssel
fig.add_trace(go.Scatter(x=fsel.index[buy_mask], y=fsel["Close"][buy_mask], mode="markers", name="BUY",
                         marker_symbol="triangle-up", marker_size=11))
fig.add_trace(go.Scatter(x=fsel.index[sell_mask], y=fsel["Close"][sell_mask], mode="markers", name="SELL",
                         marker_symbol="triangle-down", marker_size=11))
fig.update_layout(height=540, xaxis=dict(rangeslider=dict(visible=True)),
                  margin=dict(l=40,r=20,t=10,b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True, theme=None)
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# EXTRA PANELS (RSI & SCORE)
# -----------------------------------------------------------------------------
cA, cB = st.columns(2, gap="large")
with cA:
    st.markdown("<div class='card-2'><div class='h2'>RSI</div>", unsafe_allow_html=True)
    frsi = go.Figure()
    if "RSI" in fsel.columns:
        frsi.add_trace(go.Scatter(x=fsel.index, y=fsel["RSI"], name="RSI", mode="lines"))
    frsi.update_layout(height=260, margin=dict(l=40,r=20,t=10,b=10))
    st.plotly_chart(frsi, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

with cB:
    st.markdown("<div class='card-2'><div class='h2'>Score + progi</div>", unsafe_allow_html=True)
    fsc = go.Figure()
    fsc.add_trace(go.Scatter(x=fsel.index, y=scsel, name="Score", mode="lines"))
    fsc.add_trace(go.Scatter(x=bsel.index, y=bsel, name="Buy_thr", line=dict(dash="dot")))
    fsc.add_trace(go.Scatter(x=ssel.index, y=ssel, name="Sell_thr", line=dict(dash="dot")))
    fsc.update_layout(height=260, margin=dict(l=40,r=20,t=10,b=10))
    st.plotly_chart(fsc, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# BACKTEST (vol targeting) vs Buy&Hold
# -----------------------------------------------------------------------------
st.markdown("<div class='card-2'><div class='h2'>Backtest (vol targeting) vs Buy&Hold</div>", unsafe_allow_html=True)
ret = close.pct_change().fillna(0.0)
size = volatility_target_position(ret, target_vol_annual=0.12, lookback=20)
bt = backtest(close, score, buy_thr, sell_thr, 5, 5, size_series=size)
m = metrics(bt["eq"], bt["ret"])
st.write(m)

feq = go.Figure()
feq.add_trace(go.Scatter(x=bt.index, y=bt["eq"], name="Strategy"))
feq.add_trace(go.Scatter(x=bt.index, y=bt["bh"], name="Buy&Hold"))
feq.update_layout(height=300, margin=dict(l=40,r=20,t=10,b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(feq, use_container_width=True, theme=None)


# -----------------------------------------------------------------------------
# AUTO-TUNE (on demand) — Light & Full + safe rerun
# -----------------------------------------------------------------------------
def _quick_space():
    """Szybki, mały grid (ok. kilkadziesiąt kombinacji) — wyniki w ~15–40s."""
    return {
        "rsi_window": [10, 14, 20],
        "rsi_buy":    [25, 30, 35],
        "rsi_sell":   [65, 70, 75],
        "ma_fast":    [10, 20],
        "ma_mid":     [50, 100],
        "ma_slow":    [150],
        "bb_window":  [20, 30],
        "bb_std":     [1.5, 2.0],
        "w_rsi":      [0.3, 0.5, 0.7],
        "w_ma":       [0.3, 0.5, 0.7],
        "w_bb":       [0.0, 0.2, 0.4],
        "w_breakout": [0.0, 0.2, 0.4],
        "w_sent":     [0.0, 0.2],
        "percentile_window": [60, 120],
        "percentile_mode":   [True],
    }

def _run_walk_forward_safely(space, folds, cost_bps):
    """Uruchamia walk_forward z różnymi sygnaturami bez modyfikacji silnika."""
    dummy_sent = pd.Series(0, index=close.index)

    # (A) nazwy argumentów bez sent
    try:
        return walk_forward(close, space=space, folds=folds, cost_bps=cost_bps)
    except TypeError:
        pass
    except AttributeError as e:
        if "reindex" not in str(e):
            raise

    # (B) pozycyjnie bez sent
    try:
        return walk_forward(close, space, folds, cost_bps)
    except Exception:
        pass

    # (C) pozycyjnie z sent (częsty wariant: (close, sent, space, folds, cost_bps))
    try:
        return walk_forward(close, dummy_sent, space, folds, cost_bps)
    except Exception:
        pass

    # (D) nazwy argumentów z sent (gdy implementacja go wymaga)
    return walk_forward(close, dummy_sent, space=space, folds=folds, cost_bps=cost_bps)

def _score_run_os(r):
    m = r.get("metrics_os", {}) or {}
    return m.get("sharpe", 0.0) or m.get("cagr", 0.0) or m.get("ret_total", 0.0)

def _apply_best_params(best_params):
    """Wstrzykuje parametry do sliderów + bezpieczny rerun z krótką pauzą."""
    keymap = {
        "rsi_window": "rsi_w",
        "rsi_buy": "rsi_b",
        "rsi_sell": "rsi_s",
        "ma_fast": "ma_f",
        "ma_mid": "ma_m",
        "ma_slow": "ma_s",
        "bb_window": "bb_w",
        "bb_std": "bb_s",
        "w_rsi": "wg_rsi",
        "w_ma": "wg_ma",
        "w_bb": "wg_bb",
        "w_breakout": "wg_br",
        "w_sent": "wg_se",
        "percentile_window": "perc_win",
        "percentile_mode": "perc_on",
    }
    updates = {}
    for k, v in (best_params or {}).items():
        if k in keymap:
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            updates[keymap[k]] = v

    if updates:
        import time
        st.session_state.update(updates)
        st.success("✅ Zastosowano najlepsze parametry — odświeżam widok…")
        time.sleep(0.3)  # bezpieczne opóźnienie, by uniknąć "SessionInfo not initialized"
        try:
            st.rerun()
        except Exception as e:
            st.warning(f"Rerun opóźniony (sesja się inicjuje): {e}")
    else:
        st.warning("Auto-Tune zakończony, ale nie zwrócił rozpoznawalnych parametrów.")

def _autotune(profile: str):
    st.markdown(f"<div class='card'><div class='h2'>🔁 {profile} Auto-Tune</div>", unsafe_allow_html=True)
    try:
        if profile == "Light":
            space = _quick_space()
            folds = 2
            cost  = 10
        else:  # Full
            space = grid_space()   # korzystamy z Twojej pełnej przestrzeni
            folds = 4
            cost  = 10

        results, stability = _run_walk_forward_safely(space, folds, cost)
        st.write("Wyniki OS per fold:", [{"fold": r.get("fold"), "metrics": r.get("metrics_os")} for r in results])
        st.write("Stability (liczność wybieranych parametrów):", stability)

        best_run    = max(results, key=_score_run_os)
        best_params = best_run.get("best") or best_run.get("params") or {}
        st.write("Wybrane parametry:", best_params)

        _apply_best_params(best_params)
    except Exception as e:
        st.error(f"Auto-Tune błąd: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# — przyciski
if auto_tune_light_click:
    _autotune("Light")
elif auto_tune_full_click:
    _autotune("Full")
