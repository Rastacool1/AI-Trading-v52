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

st.set_page_config(page_title="AI Trading Edge v5.2", layout="wide")
st.title("📈 AI Trading Edge v5.2 — sygnały, walk-forward & risk (interaktywny wykres)")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("Dane")
    src = st.selectbox("Źródło", ["Stooq", "Yahoo", "CSV"])
    symbol = st.text_input("Symbol (np. ^spx, eurusd, btcusd)", value="^spx")
    csv_file = st.file_uploader("Wgraj CSV (Date, Close)", type=["csv"])

    st.header("Parametry (start)")
    p = SignalParams()
    p.rsi_window = st.slider("RSI window", 5, 30, p.rsi_window)
    p.rsi_buy = st.slider("RSI BUY", 10, 50, p.rsi_buy)
    p.rsi_sell = st.slider("RSI SELL", 50, 90, p.rsi_sell)
    p.ma_fast = st.slider("MA fast", 5, 50, p.ma_fast)
    p.ma_mid  = st.slider("MA mid", 20, 100, p.ma_mid)
    p.ma_slow = st.slider("MA slow", 20, 250, p.ma_slow)
    p.bb_window = st.slider("BB window", 10, 40, p.bb_window)
    p.bb_std = st.slider("BB std", 1.0, 3.0, p.bb_std)
    p.percentile_mode = st.checkbox("Dynamiczne progi (percentyle)", value=True)
    p.percentile_window = st.slider("Okno percentyli (dni)", 30, 180, p.percentile_window)

    st.header("Wagi sygnałów")
    p.w_rsi = st.slider("w_rsi", 0.0, 1.0, p.w_rsi)
    p.w_ma = st.slider("w_ma", 0.0, 1.0, p.w_ma)
    p.w_bb = st.slider("w_bb", 0.0, 1.0, p.w_bb)
    p.w_atr = st.slider("w_atr", 0.0, 1.0, p.w_atr)
    p.w_breakout = st.slider("w_breakout", 0.0, 1.0, p.w_breakout)
    p.w_sent = st.slider("w_sent", 0.0, 1.0, p.w_sent)

    st.header("Progi (jeśli statyczne)")
    p.score_buy = st.slider("Próg BUY", 0.0, 1.0, p.score_buy)
    p.score_sell = st.slider("Próg SELL", -1.0, 0.0, p.score_sell)

    st.header("Ryzyko & koszty")
    tc = st.number_input("Prowizja (bps)", 0, 100, 5)
    sl = st.number_input("Poślizg (bps)", 0, 100, 5)
    target_vol = st.number_input("Target vol (roczna)", 0.01, 1.0, 0.12, step=0.01)
    has_pos = st.checkbox("Mam już pozycję?", value=False)

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
elif last_score <= curr_sell_thr:
    action = "REDUKUJ" if has_pos else "SPRZEDAJ"
else:
    action = "TRZYMAJ"

# ===== TOP INDICATORS (cards) =====
st.subheader("📌 Wskaźniki — widok skrócony (top)")
c1, c2, c3, c4, c5, c6 = st.columns(6)

# Helper to compute last values safely
def last(series, fmt="{:.2f}"):
    try:
        v = float(series.dropna().iloc[-1])
        return fmt.format(v)
    except Exception:
        return "—"

with c1:
    st.metric("Score", f"{last_score:.2f}")
with c2:
    st.metric("RSI", last(feat["RSI"]))
with c3:
    cross = "FAST> SLOW" if feat["MA_fast"].iloc[-1] > feat["MA_slow"].iloc[-1] else "FAST< SLOW"
    st.metric("MA cross", cross)
with c4:
    pos_bb = "Poniżej BB_lo" if feat["Close"].iloc[-1] < feat["BB_lo"].iloc[-1] else ("Powyżej BB_up" if feat["Close"].iloc[-1] > feat["BB_up"].iloc[-1] else "W paśmie")
    st.metric("Pozycja vs BB", pos_bb)
with c5:
    st.metric("Regime", str(feat["Regime"].iloc[-1]))
with c6:
    st.metric("Rekomendacja", action)

# ===== INTERACTIVE CHART (Plotly) =====
st.subheader("🖱️ Interaktywny wykres (zoom, range slider, sygnały)")

# Choose quick ranges
range_choice = st.radio("Szybki zakres", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "MAX"], horizontal=True)

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

# Build figure with two rows: price+overlays and RSI/Score
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35])

# Row 1: Price with MA/BB and buy/sell markers
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

# Buy/Sell markers based on thresholds
buy_mask = scsel >= bsel
sell_mask = scsel <= ssel
fig.add_trace(go.Scatter(x=fsel.index[buy_mask], y=fsel["Close"][buy_mask], mode="markers", name="BUY", marker_symbol="triangle-up", marker_size=10), row=1, col=1)
fig.add_trace(go.Scatter(x=fsel.index[sell_mask], y=fsel["Close"][sell_mask], mode="markers", name="SELL", marker_symbol="triangle-down", marker_size=10), row=1, col=1)

# Row 2: RSI and Score with thresholds
fig.add_trace(go.Scatter(x=fsel.index, y=fsel["RSI"], name="RSI", mode="lines"), row=2, col=1)
fig.add_hline(y=p.rsi_buy, line_dash="dot", row=2, col=1)
fig.add_hline(y=p.rsi_sell, line_dash="dot", row=2, col=1)
fig.add_trace(go.Scatter(x=fsel.index, y=scsel, name="Score", mode="lines"), row=2, col=1)
fig.add_trace(go.Scatter(x=bsel.index, y=bsel, name="Buy_thr", mode="lines", line=dict(dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=ssel.index, y=ssel, name="Sell_thr", mode="lines", line=dict(dash="dot")), row=2, col=1)

fig.update_layout(
    height=700,
    xaxis=dict(rangeslider=dict(visible=True)),
    margin=dict(l=40, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ===== Backtest & Explainability =====
st.subheader("Backtest (vol targeting)")
ret = close.pct_change().fillna(0)
size = volatility_target_position(ret, target_vol_annual=target_vol, lookback=20)
bt = backtest(close, score, buy_thr, sell_thr, tc, sl, size_series=size)
m = metrics(bt["eq"], bt["ret"])
st.write(m)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["eq"], name="Strategy"))
fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["bh"], name="Buy&Hold"))
fig_eq.update_layout(height=350, margin=dict(l=40, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_eq, use_container_width=True)

st.subheader("Explainability")
from core.signals import confidence_and_explain
conf, parts = confidence_and_explain(sig, score, buy_thr, sell_thr, p)
st.write("Confidence (ostatnie 10):")
st.dataframe(conf.tail(10))
st.write("Wkład komponentów (ostatnie 10):")
st.dataframe(parts.tail(10))

st.subheader("Auto-Tune (walk-forward)")
if st.button("Uruchom walk-forward auto-tune"):
    space = grid_space()
    results, stability = walk_forward(close, sent, space, folds=4, cost_bps=tc+sl)
    st.write("Wyniki OS per fold:", [{"fold": r['fold'], "metrics": r['metrics_os']} for r in results])
    st.write("Stability (liczność wybieranych parametrów):", stability)
