import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data import from_csv, from_stooq, from_yf
from core.signals import SignalParams, compute_features, partial_signals, ensemble_score, dynamic_thresholds, confidence_and_explain
from core.sentiment import heuristic_from_vix
from core.backtest import backtest, metrics
from core.autotune import grid_space, walk_forward
from core.risk import volatility_target_position

st.set_page_config(page_title="AI Trading Edge v5.3 NeoUI", layout="wide")

# ===== Custom CSS (dark modern) =====
st.markdown(
    """
    <style>
    :root{
      --bg:#0E1117; --panel:#161B22; --text:#E6EDF3; --muted:#9DA7B1; --accent:#3BAFDA;
      --success:#104A2F; --danger:#4A1010; --warn:#4A3A10;
    }
    html, body, .block-container{background:var(--bg) !important; color:var(--text) !important;}
    .block-container{padding-top:1.2rem; padding-bottom:2rem; max-width: 1500px;}
    .neopanel{background:var(--panel); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:14px 16px; box-shadow: 0 6px 18px rgba(0,0,0,.25);}
    .neotitle{font-weight:700; font-size:1.05rem; letter-spacing:.2px; margin-bottom:.35rem;}
    .subtitle{color:var(--muted); font-size:.9rem; margin-top:-6px;}
    .stSlider > div[data-baseweb="slider"] { padding: 6px 10px; }
    .stSlider [data-baseweb="slider"] div{ background-color: rgba(59,175,218,0.18); }
    .stSlider [role="slider"]{ background:var(--accent) !important; box-shadow: 0 0 0 3px rgba(59,175,218,.25); }
    .stButton button{ background:var(--accent); color:#0A0D12; border-radius:12px; padding:.55rem .9rem; font-weight:700; border:none; }
    .stButton button:hover{ filter:brightness(1.05); transform: translateY(-1px); }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{ background:#0C0F14; color:var(--text); border-radius:10px; }
    .metric-card{border-radius:14px; padding:10px 14px; background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); border:1px solid rgba(255,255,255,0.08);}
    .rec-green{background:rgba(16,74,47,.6); border:1px solid rgba(0,255,150,.25);}
    .rec-red{background:rgba(74,16,16,.6); border:1px solid rgba(255,0,80,.25);}
    .rec-yellow{background:rgba(74,58,16,.6); border:1px solid rgba(255,196,0,.25);}
    .sticky{ position: sticky; top: 0; z-index: 999; padding-top: 8px; backdrop-filter: blur(6px);}
    .section{margin-top: 12px;}
    .range-radio .stRadio > div{flex-wrap: wrap;}
    .range-radio label{padding-right:10px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='sticky neopanel'><div class='neotitle'>⚙️ Panel parametrów</div>", unsafe_allow_html=True)

# ===== TOP PANEL (no engine changes) =====
c0, c1, c2, c3 = st.columns([1.2, 2.2, 2.2, 2.2])
with c0:
    src = st.selectbox("Źródło", ["Stooq", "Yahoo", "CSV"])
    symbol = st.text_input("Symbol", value="^spx")
    csv_file = st.file_uploader("CSV (Date, Close)", type=["csv"])
with c1:
    st.markdown("**Wskaźniki**")
    p = SignalParams()
    p.rsi_window = st.slider("RSI window", 5, 30, p.rsi_window)
    p.ma_fast    = st.slider("MA fast", 5, 50, p.ma_fast)
    p.ma_mid     = st.slider("MA mid", 20, 100, p.ma_mid)
    p.ma_slow    = st.slider("MA slow", 20, 250, p.ma_slow)
with c2:
    st.markdown("**Bollinger / RSI progi**")
    p.bb_window  = st.slider("BB window", 10, 40, p.bb_window)
    p.bb_std     = st.slider("BB std", 1.0, 3.0, p.bb_std)
    p.rsi_buy    = st.slider("RSI BUY", 10, 50, p.rsi_buy)
    p.rsi_sell   = st.slider("RSI SELL", 50, 90, p.rsi_sell)
with c3:
    st.markdown("**Wagi & Progi**")
    p.w_rsi      = st.slider("w_rsi", 0.0, 1.0, p.w_rsi)
    p.w_ma       = st.slider("w_ma", 0.0, 1.0, p.w_ma)
    p.w_bb       = st.slider("w_bb", 0.0, 1.0, p.w_bb)
    p.w_atr      = st.slider("w_atr", 0.0, 1.0, p.w_atr)
    p.w_breakout = st.slider("w_breakout", 0.0, 1.0, p.w_breakout)
    p.w_sent     = st.slider("w_sent", 0.0, 1.0, p.w_sent)
    p.percentile_mode = st.checkbox("Progi dynamiczne (percentyle)", value=True)
    p.percentile_window = st.slider("Okno percentyli", 30, 180, p.percentile_window)

st.markdown("</div>", unsafe_allow_html=True)

# SECOND ROW OF CONTROLS
c4, c5, c6 = st.columns([2.2, 1.2, 1.2])
with c4:
    st.markdown("<div class='neopanel'><div class='neotitle'>Ryzyko & koszty</div>", unsafe_allow_html=True)
    tc = st.number_input("Prowizja (bps)", 0, 100, 5)
    sl = st.number_input("Poślizg (bps)", 0, 100, 5)
    target_vol = st.number_input("Target vol (roczna)", 0.01, 1.0, 0.12, step=0.01)
    has_pos = st.checkbox("Mam już pozycję?", value=False)
    st.markdown("</div>", unsafe_allow_html=True)
with c5:
    st.markdown("<div class='neopanel'><div class='neotitle'>Akcje</div>", unsafe_allow_html=True)
    run_autotune = st.button("🔁 Auto‑Tune (walk‑forward)")
    recalc = st.button("⚡ Przelicz wykres")
    st.markdown("</div>", unsafe_allow_html=True)
with c6:
    st.markdown("<div class='neopanel'><div class='neotitle'>Zakres</div>", unsafe_allow_html=True)
    range_choice = st.radio("Szybki zakres", ["1M","3M","6M","YTD","1Y","3Y","MAX"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== DATA LOAD =====
if src == "CSV" and csv_file is not None:
    df = from_csv(csv_file)
elif src == "Stooq":
    df = from_stooq(symbol)
else:
    df = from_yf(symbol)

close = df["Close"].dropna()

# Sentiment (VIX proxy)
try:
    vix = from_stooq("^vix")["Close"]
    sent = heuristic_from_vix(vix).reindex(close.index).fillna(method="ffill")
except Exception:
    sent = pd.Series(0, index=close.index)

# Features & signals
feat = compute_features(close, p)
sig = partial_signals(feat, p)
score = ensemble_score(sig, sent, p)
buy_thr, sell_thr = dynamic_thresholds(score, p)

# Decision today
last_score = float(score.iloc[-1])
curr_buy_thr = float(buy_thr.iloc[-1] if isinstance(buy_thr, pd.Series) else buy_thr)
curr_sell_thr = float(sell_thr.iloc[-1] if isinstance(sell_thr, pd.Series) else sell_thr)

if last_score >= curr_buy_thr:
    action = "AKUMULUJ" if has_pos else "KUP"
    rec_class = "rec-green"
elif last_score <= curr_sell_thr:
    action = "REDUKUJ" if has_pos else "SPRZEDAJ"
    rec_class = "rec-red"
else:
    action = "TRZYMAJ"
    rec_class = "rec-yellow"

# ===== RECOMMENDATION BLOCK =====
st.markdown(f"<div class='neopanel {rec_class}'><div class='neotitle'>🧭 Rekomendacja na dziś</div><h2 style='margin:4px 0'>{action}</h2><div class='subtitle'>Score: {last_score:.2f} • BUY_thr: {curr_buy_thr:.2f} • SELL_thr: {curr_sell_thr:.2f}</div></div>", unsafe_allow_html=True)

# ===== TOP INDICATORS =====
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Score", f"{last_score:.2f}")
m2.metric("RSI", f"{feat['RSI'].iloc[-1]:.2f}")
m3.metric("MA cross", "FAST > SLOW" if feat["MA_fast"].iloc[-1] > feat["MA_slow"].iloc[-1] else "FAST < SLOW")
pos_bb = "Poniżej BB_lo" if feat["Close"].iloc[-1] < feat["BB_lo"].iloc[-1] else ("Powyżej BB_up" if feat["Close"].iloc[-1] > feat["BB_up"].iloc[-1] else "W paśmie")
m4.metric("Pozycja vs BB", pos_bb)
m5.metric("Regime", str(feat["Regime"].iloc[-1]))
m6.metric("Sygnał", action)

# ===== INTERACTIVE CHART =====
def get_range_index(idx, choice):
    if len(idx) == 0:
        return idx
    end = idx[-1]
    if choice == "1M":
        start = end - pd.DateOffset(months=1)
    elif choice == "3M":
        start = end - pd.DateOffset(months=3)
    elif choice == "6M":
        start = end - pd.DateOffset(months=6)
    elif choice == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif choice == "1Y":
        start = end - pd.DateOffset(years=1)
    elif choice == "3Y":
        start = end - pd.DateOffset(years=3)
    else:
        start = idx[0]
    return idx[(idx >= start) & (idx <= end)]

plot_idx = get_range_index(feat.index, range_choice)

st.markdown("<div class='section neopanel'>", unsafe_allow_html=True)
st.markdown("<div class='neotitle'>🖱️ Interaktywny wykres</div>", unsafe_allow_html=True)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38])

fsel = feat.loc[plot_idx]
scsel = score.loc[plot_idx]
bsel = buy_thr.loc[plot_idx] if isinstance(buy_thr, pd.Series) else pd.Series(buy_thr, index=plot_idx)
ssel = sell_thr.loc[plot_idx] if isinstance(sell_thr, pd.Series) else pd.Series(sell_thr, index=plot_idx)

fig.add_trace(go.Scatter(x=fsel.index, y=fsel["Close"], name="Close", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=fsel.index, y=fsel["MA_fast"], name="MA_fast", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=fsel.index, y=fsel["MA_mid"], name="MA_mid", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=fsel.index, y=fsel["MA_slow"], name="MA_slow", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=fsel.index, y=fsel["BB_up"], name="BB_up", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=fsel.index, y=fsel["BB_lo"], name="BB_lo", mode="lines"), row=1, col=1)

buy_mask = scsel >= bsel
sell_mask = scsel <= ssel
fig.add_trace(go.Scatter(x=fsel.index[buy_mask], y=fsel["Close"][buy_mask], mode="markers", name="BUY", marker_symbol="triangle-up", marker_size=10), row=1, col=1)
fig.add_trace(go.Scatter(x=fsel.index[sell_mask], y=fsel["Close"][sell_mask], mode="markers", name="SELL", marker_symbol="triangle-down", marker_size=10), row=1, col=1)

fig.add_trace(go.Scatter(x=fsel.index, y=fsel["RSI"], name="RSI", mode="lines"), row=2, col=1)
fig.add_hline(y=p.rsi_buy, line_dash="dot", row=2, col=1)
fig.add_hline(y=p.rsi_sell, line_dash="dot", row=2, col=1)
fig.add_trace(go.Scatter(x=fsel.index, y=scsel, name="Score", mode="lines"), row=2, col=1)
fig.add_trace(go.Scatter(x=bsel.index, y=bsel, name="Buy_thr", mode="lines", line=dict(dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=ssel.index, y=ssel, name="Sell_thr", mode="lines", line=dict(dash="dot")), row=2, col=1)

fig.update_layout(
    height=720,
    xaxis=dict(rangeslider=dict(visible=True)),
    margin=dict(l=40, r=20, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True, theme=None)
st.markdown("</div>", unsafe_allow_html=True)

# ===== Backtest & Explainability =====
st.markdown("<div class='section neopanel'><div class='neotitle'>Backtest (vol targeting) • porównanie z Buy&Hold</div>", unsafe_allow_html=True)
ret = close.pct_change().fillna(0)
size = volatility_target_position(ret, target_vol_annual=target_vol, lookback=20)
bt = backtest(close, score, buy_thr, sell_thr, tc, sl, size_series=size)
m = metrics(bt["eq"], bt["ret"])
st.write(m)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["eq"], name="Strategy"))
fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["bh"], name="Buy&Hold"))
fig_eq.update_layout(height=350, margin=dict(l=40, r=20, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_eq, use_container_width=True, theme=None)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section neopanel'><div class='neotitle'>Explainability</div>", unsafe_allow_html=True)
conf, parts = confidence_and_explain(sig, score, buy_thr, sell_thr, p)
st.write("Confidence (ostatnie 10):")
st.dataframe(conf.tail(10))
st.write("Wkład komponentów (ostatnie 10):")
st.dataframe(parts.tail(10))
st.markdown("</div>", unsafe_allow_html=True)

# ===== Auto-Tune action =====
if run_autotune:
    st.markdown("<div class='section neopanel'><div class='neotitle'>🔁 Auto‑Tune (walk‑forward)</div>", unsafe_allow_html=True)
    space = grid_space()
    results, stability = walk_forward(close, sent, space, folds=4, cost_bps=tc+sl)
    st.write("Wyniki OS per fold:", [{"fold": r['fold'], "metrics": r['metrics_os']} for r in results])
    st.write("Stability (liczność wybieranych parametrów):", stability)
    st.markdown("</div>", unsafe_allow_html=True)
