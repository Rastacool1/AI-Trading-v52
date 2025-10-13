import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.data import from_csv, from_stooq, from_yf
from core.signals import SignalParams, compute_features, partial_signals, ensemble_score, dynamic_thresholds, confidence_and_explain
from core.sentiment import heuristic_from_vix
from core.backtest import backtest, metrics
from core.autotune import grid_space, walk_forward
from core.risk import volatility_target_position, de_risk_overlay

st.set_page_config(page_title="AI Trading Edge v5.2", layout="wide")
st.title("📈 AI Trading Edge v5.2 — sygnały, walk‑forward & risk")

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

# --- Dane ---
if src == "CSV" and csv_file is not None:
    df = from_csv(csv_file)
elif src == "Stooq":
    df = from_stooq(symbol)
else:
    df = from_yf(symbol)

close = df["Close"].dropna()

# --- Sentyment (VIX proxy) ---
try:
    vix = from_stooq("^vix")["Close"]
    sent = heuristic_from_vix(vix).reindex(close.index).fillna(method="ffill")
except Exception:
    sent = pd.Series(0, index=close.index)

# --- Cechy i sygnały ---
feat = compute_features(close, p)
sig = partial_signals(feat, p)
score = ensemble_score(sig, sent, p)
buy_thr, sell_thr = dynamic_thresholds(score, p)

# --- Decyzja na dziś ---
last_score = float(score.iloc[-1])
curr_buy_thr = float(buy_thr.iloc[-1] if isinstance(buy_thr, pd.Series) else buy_thr)
curr_sell_thr = float(sell_thr.iloc[-1] if isinstance(sell_thr, pd.Series) else sell_thr)

if last_score >= curr_buy_thr:
    action = "AKUMULUJ" if has_pos else "KUP"
elif last_score <= curr_sell_thr:
    action = "REDUKUJ" if has_pos else "SPRZEDAJ"
else:
    action = "TRZYMAJ"

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Status na dziś")
    st.metric(label="Sygnał (score)", value=f"{last_score:.2f}")
    st.success(f"Rekomendacja: **{action}**")

    st.subheader("Wykres ceny + MA/BB")
    fig, ax = plt.subplots()
    ax.plot(feat.index, feat["Close"], label="Close")
    ax.plot(feat.index, feat["MA_fast"], label="MA_fast")
    ax.plot(feat.index, feat["MA_mid"], label="MA_mid")
    ax.plot(feat.index, feat["MA_slow"], label="MA_slow")
    ax.plot(feat.index, feat["BB_up"], label="BB_up")
    ax.plot(feat.index, feat["BB_lo"], label="BB_lo")
    ax.legend()
    st.pyplot(fig)

    st.subheader("RSI")
    fig2, ax2 = plt.subplots()
    ax2.plot(feat.index, feat["RSI"], label="RSI")
    ax2.axhline(p.rsi_buy, linestyle="--")
    ax2.axhline(p.rsi_sell, linestyle="--")
    ax2.legend()
    st.pyplot(fig2)

with col2:
    st.subheader("Backtest (vol targeting + DD overlay)")
    # volatility targeting sizing
    ret = close.pct_change().fillna(0)
    size = volatility_target_position(ret, target_vol_annual=target_vol, lookback=20)
    bt = backtest(close, score, buy_thr, sell_thr, tc, sl, size_series=size)
    m = metrics(bt["eq"], bt["ret"])
    st.write(m)
    fig3, ax3 = plt.subplots()
    ax3.plot(bt.index, bt["eq"], label="Strategy")
    ax3.plot(bt.index, bt["bh"], label="Buy&Hold")
    ax3.legend()
    st.pyplot(fig3)

st.subheader("Explainability")
conf, parts = confidence_and_explain(sig, score, buy_thr, sell_thr, p)
st.write("Confidence (ostatnie 10):", conf.tail(10))
st.dataframe(parts.tail(10))

st.subheader("Auto‑Tune (walk‑forward)")
if st.button("Uruchom walk‑forward auto‑tune"):
    space = grid_space()
    results, stability = walk_forward(close, sent, space, folds=4, cost_bps=tc+sl)
    st.write("Wyniki OS per fold:", [{"fold": r['fold'], "metrics": r['metrics_os']} for r in results])
    st.write("Stability (liczność wybieranych parametrów):", stability)
