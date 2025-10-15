# app.py — AI Trading Edge • Excel-style Dark Dashboard (UX-only)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.data import from_csv, from_stooq, from_yf
from core.signals import (
    SignalParams, compute_features, partial_signals,
    ensemble_score, dynamic_thresholds, confidence_and_explain
)
from core.sentiment import heuristic_from_vix
from core.backtest import backtest, metrics
from core.autotune import grid_space, walk_forward
from core.risk import volatility_target_position

st.set_page_config(page_title="AI Trading Edge — Dashboard", layout="wide")

# =========================  THEME (Excel-like dark, compact & responsive)  =========================
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

/* Uploader: złote tło */
.stFileUploader, .stFileUploader div[data-testid="stFileUploaderDropzone"]{
  background:linear-gradient(180deg, rgba(203,168,91,.18), rgba(203,168,91,.10)) !important;
  border:1px dashed rgba(203,168,91,.65) !important; border-radius:12px !important; color:var(--amber) !important;
}

/* Slidery – kompakt i bez ciemnego tła */
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

/* Rekomendacja */
.reco{border:1px solid rgba(203,168,91,.35); background:linear-gradient(180deg, rgba(203,168,91,.14), rgba(203,168,91,.06));}
.reco.good{border-color:rgba(0,195,137,.35); background:linear-gradient(180deg, rgba(0,195,137,.14), rgba(0,195,137,.06));}
.reco.bad{border-color:rgba(255,92,122,.35); background:linear-gradient(180deg, rgba(255,92,122,.14), rgba(255,92,122,.06));}

/* Grid wskaźników: lewa kolumna = nazwa, po prawej parametry w jednej linii */
.row{display:grid; grid-template-columns: 84px 1fr; gap:10px; align-items:center; margin:2px 0;}
.row .name{font-weight:800; color:var(--text); text-transform:uppercase; font-size:.85rem; letter-spacing:.4px;}
.row .cells{display:grid; grid-template-columns: repeat(5, 1fr); gap:8px;}
/* responsywność: gdy wężej, redukuj liczbę kolumn parametrów */
@media (max-width: 1400px){ .row .cells{grid-template-columns: repeat(4, 1fr);} }
@media (max-width: 1120px){ .row .cells{grid-template-columns: repeat(3, 1fr);} }
@media (max-width: 880px) { .row .cells{grid-template-columns: repeat(2, 1fr);} }
@media (max-width: 640px) { .row .cells{grid-template-columns: 1fr;} }

</style>
""", unsafe_allow_html=True)

# =========================  TOP BAR (logo + actions)  =========================
with st.container():
    cols = st.columns([2.6, 1.2, 1.0, 1.0, 1.0], gap="small")
    with cols[0]:
        st.markdown(
            "<div class='topbar card-2'>"
            "<div class='logo'>"
            "<div class='mark'>AI</div>"
            "<div class='title'>Trading Edge — Dashboard</div>"
            "</div>"
            "</div>", unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
        auto_tune_click = st.button("🔁 Auto-Tune", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
        recalc_click = st.button("⚡ Przelicz", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[3]:
        st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
        autoscale_click = st.button("🖼️ Autoskaluj", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[4]:
        st.markdown("<div class='topbar card-2'>", unsafe_allow_html=True)
        export_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

# =========================  STARTER PANEL — kompakt + nazwy po lewej =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h1'>⚙️ Ustawienia / Filtry</div>", unsafe_allow_html=True)

left, right = st.columns([1.0, 3.0], gap="large")

with left:
    src = st.selectbox("Źródło", ["Stooq", "Yahoo", "CSV"])
    symbol = st.text_input("Symbol", value="btcpln", help="np. btcpln / eurusd / ^spx", placeholder="ticker")
    csv_file = st.file_uploader("CSV (Date/Data, Close/Zamknięcie)", type=["csv"])
    st.markdown("<div class='h2' style='margin-top:8px;'>Ryzyko & koszty</div>", unsafe_allow_html=True)
    tc = st.number_input("Prowizja (bps)", 0, 100, 5)
    sl = st.number_input("Poślizg (bps)", 0, 100, 5)
    target_vol = st.number_input("Target vol (roczna)", 0.01, 1.0, 0.12, step=0.01)

with right:
    p = SignalParams()

    # RSI — trzy suwaki w jednej linii (etykiety suwaków schowane)
    st.markdown("<div class='row'><div class='name'>RSI</div><div class='cells'>", unsafe_allow_html=True)
    p.rsi_window = st.slider("RSI window", 5, 30, p.rsi_window, label_visibility="collapsed", key="rsi_w")
    p.rsi_buy    = st.slider("RSI BUY",    10, 50, p.rsi_buy,    label_visibility="collapsed", key="rsi_b")
    p.rsi_sell   = st.slider("RSI SELL",   50, 90, p.rsi_sell,   label_visibility="collapsed", key="rsi_s")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # MA — fast/mid/slow w jednej linii
    st.markdown("<div class='row'><div class='name'>MA</div><div class='cells'>", unsafe_allow_html=True), p.ma_fast = st.slider("MA fast", 5, 50, p.ma_fast, label_visibility="collapsed", key="ma_f")
    p.ma_mid  = st.slider("MA mid",  20, 100, p.ma_mid, label_visibility="collapsed", key="ma_m")
    p.ma_slow = st.slider("MA slow", 20, 250, p.ma_slow, label_visibility="collapsed", key="ma_s")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # BB — window/std w jednej linii
    st.markdown("<div class='row'><div class='name'>BB</div><div class='cells'>", unsafe_allow_html=True)
    p.bb_window = st.slider("BB window", 10, 40, p.bb_window, label_visibility="collapsed", key="bb_w")
    p.bb_std    = st.slider("BB std",    1.0, 3.0, p.bb_std,    label_visibility="collapsed", key="bb_s")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Wagi — pięć suwaków w jednej linii
    st.markdown("<div class='row'><div class='name'>Wagi</div><div class='cells'>", unsafe_allow_html=True)
    p.w_rsi      = st.slider("w_rsi",      0.0, 1.0, p.w_rsi,      label_visibility="collapsed", key="wg_rsi")
    p.w_ma       = st.slider("w_ma",       0.0, 1.0, p.w_ma,       label_visibility="collapsed", key="wg_ma")
    p.w_bb       = st.slider("w_bb",       0.0, 1.0, p.w_bb,       label_visibility="collapsed", key="wg_bb")
    p.w_breakout = st.slider("w_breakout", 0.0, 1.0, p.w_breakout, label_visibility="collapsed", key="wg_br")
    p.w_sent     = st.slider("w_sent",     0.0, 1.0, p.w_sent,     label_visibility="collapsed", key="wg_se")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Progi (percentyle) — w jednej linii
    st.markdown("<div class='row'><div class='name'>Progi</div><div class='cells'>", unsafe_allow_html=True)
    p.percentile_mode   = st.checkbox("Progi dynamiczne", value=True, key="perc_on")
    p.percentile_window = st.slider("Okno percentyli", 30, 180, p.percentile_window, label_visibility="collapsed", key="perc_win")
    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # /card

# =========================  SAFE DATA LOAD (bez crashy)  =========================
YF_MAP = {"^spx":"^GSPC","^ndx":"^NDX","^vix":"^VIX","btcusd":"BTC-USD","btcpln":"BTC-PLN","eurusd":"EURUSD=X"}

def safe_load(src_choice, sym, fileobj):
    sym = sym.strip()
    if src_choice=="CSV":
        if not fileobj:
            st.warning("Wgraj CSV z kolumnami Date/Data i Close/Zamkniecie.")
            return None, "CSV"
        try:
            return from_csv(fileobj), "CSV"
        except Exception as e:
            st.error(f"Błąd CSV: {e}"); return None, "CSV"
    if src_choice=="Stooq":
        try:
            return from_stooq(sym), "Stooq"
        except Exception as e:
            y = YF_MAP.get(sym.lower(), sym)
            st.info(f"Stooq nie działa ({e}). Próbuję Yahoo: {y}")
            try:
                return from_yf(y), f"Yahoo ({y})"
            except Exception as e2:
                st.error(f"Brak danych: Stooq={e} | Yahoo={e2}")
                return None, "Error"
    if src_choice=="Yahoo":
        y = YF_MAP.get(sym.lower(), sym)
        try:
            return from_yf(y), f"Yahoo ({y})"
        except Exception as e:
            st.error(f"Yahoo błąd: {e}"); return None, "Yahoo"

df, used_source = safe_load(src, symbol, csv_file)
if df is None or df.empty:
    st.stop()
close = df["Close"].dropna()

# Optional sentiment
try:
    vix = from_stooq("^vix")["Close"]
    sent = heuristic_from_vix(vix).reindex(close.index).fillna(method="ffill")
except Exception:
    sent = pd.Series(0, index=close.index)

# =========================  SIGNALS & DECISION  =========================
feat = compute_features(close, p)
sig = partial_signals(feat, p)
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

# eksport sygnałów
export_csv = pd.DataFrame({"Date": feat.index, "Close": feat["Close"], "RSI": feat["RSI"], "Score": score}).to_csv(index=False).encode("utf-8")
with export_placeholder:
    st.download_button("⬇️ Eksport sygnałów (CSV)", export_csv, file_name=f"signals_{symbol.replace('^','')}.csv", use_container_width=True)

# =========================  RECOMMENDATION  =========================
st.markdown(
    f"<div class='card reco {rec_cl}'><div class='h1'>🧭 Rekomendacja na dziś</div>"
    f"<div class='sub'>Źródło: {used_source} • Score: {last_score:.2f} • BUY_thr: {buy_now:.2f} • SELL_thr: {sell_now:.2f}</div>"
    f"<h2 style='margin:8px 0 4px 0; font-size:1.8rem;'>{action}</h2>"
    f"</div>", unsafe_allow_html=True
)

# =========================  MAIN CHART (środek)  =========================
st.markdown("<div class='card-2'>", unsafe_allow_html=True)
st.markdown("<div class='h1'>📈 Wykres ceny (markery sygnałów)</div>", unsafe_allow_html=True)
range_choice = st.radio("Zakres", ["1M","3M","6M","YTD","1Y","3Y","MAX"], horizontal=True)

def pick_range(idx, choice):
    if len(idx)==0: return idx
    end = idx[-1]
    import pandas as pd
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

# =========================  EXTRA PANELS (czytelnie, osobno)  =========================
with st.container():
    cA, cB = st.columns(2, gap="large")
    with cA:
        st.markdown("<div class='card-2'><div class='h2'>RSI</div>", unsafe_allow_html=True)
        frsi = go.Figure()
        frsi.add_trace(go.Scatter(x=fsel.index, y=fsel["RSI"], name="RSI", mode="lines"))
        frsi.add_hline(y=p.rsi_buy, line_dash="dot")
        frsi.add_hline(y=p.rsi_sell, line_dash="dot")
        frsi.update_layout(height=260, margin=dict(l=40,r=20,t=10,b=10))
        st.plotly_chart(frsi, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)
    with cB:
        st.markdown("<div class='card-2'><div class='h2'>Score + progi</div>", unsafe_allow_html=True)
        fsc = go.Figure()
        fsc.add_trace(go.Scatter(x=fsel.index, y=scsel, name="Score", mode="lines"))
        fsc.add_trace(go.Scatter(x=bsel.index, y=bsel, name="Buy_thr", mode="lines", line=dict(dash="dot")))
        fsc.add_trace(go.Scatter(x=ssel.index, y=ssel, name="Sell_thr", mode="lines", line=dict(dash="dot")))
        fsc.update_layout(height=260, margin=dict(l=40,r=20,t=10,b=10))
        st.plotly_chart(fsc, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================  BACKTEST  =========================
st.markdown("<div class='card-2'><div class='h2'>Backtest (vol targeting) vs Buy&Hold</div>", unsafe_allow_html=True)
ret = close.pct_change().fillna(0)
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

# =========================  Auto-Tune (na żądanie)  =========================
if auto_tune_click:
    st.markdown("<div class='card'><div class='h2'>🔁 Auto-Tune (walk-forward)</div>", unsafe_allow_html=True)
    space = grid_space()
    results, stability = walk_forward(close, sent=None, space=space, folds=4, cost_bps=10)
    st.write("Wyniki OS per fold:", [{"fold": r['fold'], "metrics": r['metrics_os']} for r in results])
    st.write("Stability (liczność wybieranych parametrów):", stability)
    st.markdown("</div>", unsafe_allow_html=True)
