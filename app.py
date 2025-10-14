# app.py — AI Trading Edge v5.3 NeoUI • Starter Panel (33vh) + clean charts
import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data import from_csv, from_stooq, from_yf
from core.signals import (
    SignalParams, compute_features, partial_signals,
    ensemble_score, dynamic_thresholds, confidence_and_explain
)
from core.sentiment import heuristic_from_vix
from core.backtest import backtest, metrics
from core.autotune import grid_space, walk_forward
from core.risk import volatility_target_position

st.set_page_config(page_title="AI Trading Edge v5.3 NeoUI", layout="wide")

# =====================  CSS  =====================
st.markdown("""
<style>
:root{
  --bg:#0E1117; --panel:#161B22; --text:#E6EDF3; --muted:#9DA7B1; --accent:#3BAFDA;
  --good:#0d5e40; --bad:#6a1b1b; --warn:#6a5d1b;
}
html, body, .block-container { background: var(--bg) !important; color: var(--text) !important; }
.block-container { padding-top: 0.8rem; max-width: 1500px; }

/* STARTER PANEL (Top) */
.starter-wrap { position: sticky; top: 0; z-index: 1000; backdrop-filter: blur(6px); }
.starter { height: 33vh; min-height: 320px; background: var(--panel);
  border:1px solid rgba(255,255,255,.06); border-radius: 14px; box-shadow: 0 6px 18px rgba(0,0,0,.25);
  padding: 12px 14px; overflow: auto; }
.starter::-webkit-scrollbar { height: 10px; width: 10px; }
.starter::-webkit-scrollbar-thumb { background: rgba(255,255,255,.18); border-radius: 10px; }

/* Titles */
.hdr { font-weight: 700; font-size: 1.02rem; letter-spacing: .2px; margin: 2px 0 8px 2px; }
.subhdr { color: var(--muted); font-size: .88rem; margin: 0 0 8px 2px; }

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{
  background: #0C0F14; color: var(--text); border-radius: 10px;
}
.stSlider > div[data-baseweb="slider"] { padding: 6px 10px; }
.stSlider [data-baseweb="slider"] div{ background-color: rgba(59,175,218,0.18); }
.stSlider [role="slider"]{ background:var(--accent) !important; box-shadow: 0 0 0 3px rgba(59,175,218,.25); }

/* Buttons */
.stButton button{ background:var(--accent); color:#0A0D12; border-radius:12px; padding:.55rem .9rem; font-weight:700; border:none; }
.stButton button:hover{ filter:brightness(1.05); transform: translateY(-1px); }

/* Cards */
.panel { background: var(--panel); border:1px solid rgba(255,255,255,.06); border-radius: 14px; padding: 12px 14px; }
.rec-green{background:rgba(13,94,64,.6); border:1px solid rgba(0,255,150,.25);}
.rec-red{background:rgba(106,27,27,.6); border:1px solid rgba(255,0,80,.25);}
.rec-yellow{background:rgba(106,93,27,.6); border:1px solid rgba(255,196,0,.25);}

.legend-top { margin: 2px 0 8px 0; font-size: .88rem; color: var(--muted); }
</style>
""", unsafe_allow_html=True)

# =====================  STARTER PANEL  =====================
st.markdown("<div class='starter-wrap'>", unsafe_allow_html=True)
st.markdown("<div class='starter'>", unsafe_allow_html=True)
st.markdown("<div class='hdr'>⚙️ Panel startowy</div>", unsafe_allow_html=True)

# ROW 1: data source + core indicator sliders (compact 4 cols)
c0, c1, c2, c3 = st.columns([1.1, 2.2, 2.2, 2.2], vertical_alignment="top")
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
    p.w_breakout = st.slider("w_breakout", 0.0, 1.0, 0.10)
    p.w_sent     = st.slider("w_sent", 0.0, 1.0, 0.10)
    p.percentile_mode = st.checkbox("Progi dynamiczne (percentyle)", value=True)
    p.percentile_window = st.slider("Okno percentyli", 30, 180, 90)

# ROW 2: actions + risk
a1, a2, a3, a4 = st.columns([1.2, 1.0, 1.0, 3.8], vertical_alignment="top")
with a1:
    st.markdown("**Akcje**")
    do_autotune = st.button("🔁 Auto-Tune (walk-forward)")
    recalc = st.button("⚡ Przelicz")
    autoscale = st.button("🖼️ Autoskaluj wykres")
with a2:
    st.markdown("**Ryzyko & koszty**")
    tc = st.number_input("Prowizja (bps)", 0, 100, 5)
    sl = st.number_input("Poślizg (bps)", 0, 100, 5)
with a3:
    st.markdown("**Ekspozycja**")
    target_vol = st.number_input("Target vol (roczna)", 0.01, 1.0, 0.12, step=0.01)
    has_pos = st.checkbox("Mam już pozycję?", value=False)
with a4:
    st.markdown("**Eksport**")
    # placeholder na późniejszy download (wypełnimy po obliczeniach)
    dl_placeholder = st.empty()

st.markdown("</div>", unsafe_allow_html=True)  # end .starter
st.markdown("</div>", unsafe_allow_html=True)  # end .starter-wrap

# =====================  DATA & SIGNALS  =====================
if src == "CSV" and csv_file is not None:
    df = from_csv(csv_file)
elif src == "Stooq":
    df = from_stooq(symbol)
else:
    df = from_yf(symbol)

close = df["Close"].dropna()

try:
    vix = from_stooq("^vix")["Close"]
    sent = heuristic_from_vix(vix).reindex(close.index).fillna(method="ffill")
except Exception:
    sent = pd.Series(0, index=close.index)

feat = compute_features(close, p)
sig = partial_signals(feat, p)
score = ensemble_score(sig, sent, p)
buy_thr, sell_thr = dynamic_thresholds(score, p)

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

# Data for export (po obliczeniach)
export_df = pd.DataFrame({
    "Date": feat.index,
    "Close": feat["Close"],
    "RSI": feat["RSI"],
    "Score": score,
})
export_csv = export_df.to_csv(index=False).encode("utf-8")
with dl_placeholder:
    st.download_button("⬇️ Pobierz sygnały (CSV)", export_csv, file_name=f"signals_{symbol.replace('^','')}.csv")

# =====================  RECOMMENDATION CARD  =====================
st.markdown(
    f"<div class='panel {rec_class}'><div class='hdr'>🧭 Rekomendacja na dziś</div>"
    f"<h2 style='margin:4px 0'>{action}</h2>"
    f"<div class='legend-top'>Score: {last_score:.2f} • BUY_thr: {curr_buy_thr:.2f} • SELL_thr: {curr_sell_thr:.2f}</div>"
    f"</div>",
    unsafe_allow_html=True
)

# =====================  MAIN CHART — CLEAN (PRICE + BUY/SELL ONLY)  =====================
st.markdown("<div class='panel'><div class='hdr'>📈 Wykres (tylko cena + markery sygnałów)</div>", unsafe_allow_html=True)

# Quick ranges
range_choice = st.radio("Szybki zakres", ["1M","3M","6M","YTD","1Y","3Y","MAX"], horizontal=True)

def get_range_index(idx, choice):
    if len(idx) == 0: return idx
    end = idx[-1]
    if choice == "1M": start = end - pd.DateOffset(months=1)
    elif choice == "3M": start = end - pd.DateOffset(months=3)
    elif choice == "6M": start = end - pd.DateOffset(months=6)
    elif choice == "YTD": start = pd.Timestamp(year=end.year, month=1, day=1)
    elif choice == "1Y": start = end - pd.DateOffset(years=1)
    elif choice == "3Y": start = end - pd.DateOffset(years=3)
    else: start = idx[0]
    return idx[(idx >= start) & (idx <= end)]

plot_idx = get_range_index(feat.index, range_choice)
fsel = feat.loc[plot_idx]
scsel = score.loc[plot_idx]
bsel = buy_thr.loc[plot_idx] if isinstance(buy_thr, pd.Series) else pd.Series(buy_thr, index=plot_idx)
ssel = sell_thr.loc[plot_idx] if isinstance(sell_thr, pd.Series) else pd.Series(sell_thr, index=plot_idx)

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=fsel.index, y=fsel["Close"], name="Close", mode="lines"))
buy_mask = scsel >= bsel
sell_mask = scsel <= ssel
fig_price.add_trace(go.Scatter(x=fsel.index[buy_mask], y=fsel["Close"][buy_mask], mode="markers",
                               name="BUY", marker_symbol="triangle-up", marker_size=11))
fig_price.add_trace(go.Scatter(x=fsel.index[sell_mask], y=fsel["Close"][sell_mask], mode="markers",
                               name="SELL", marker_symbol="triangle-down", marker_size=11))
fig_price.update_layout(
    height=520, xaxis=dict(rangeslider=dict(visible=True)),
    margin=dict(l=40, r=20, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_price, use_container_width=True, theme=None)
st.markdown("</div>", unsafe_allow_html=True)

# =====================  SEPARATE INDICATOR CHARTS (no clutter)  =====================
# 1) RSI alone
st.markdown("<div class='panel'><div class='hdr'>RSI</div>", unsafe_allow_html=True)
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=fsel.index, y=fsel["RSI"], name="RSI", mode="lines"))
fig_rsi.add_hline(y=p.rsi_buy, line_dash="dot")
fig_rsi.add_hline(y=p.rsi_sell, line_dash="dot")
fig_rsi.update_layout(height=260, margin=dict(l=40,r=20,t=10,b=10))
st.plotly_chart(fig_rsi, use_container_width=True, theme=None)
st.markdown("</div>", unsafe_allow_html=True)

# 2) Score with thresholds (helpful to see signals formation)
st.markdown("<div class='panel'><div class='hdr'>Score + progi</div>", unsafe_allow_html=True)
fig_sc = go.Figure()
fig_sc.add_trace(go.Scatter(x=fsel.index, y=scsel, name="Score", mode="lines"))
fig_sc.add_trace(go.Scatter(x=bsel.index, y=bsel, name="Buy_thr", mode="lines", line=dict(dash="dot")))
fig_sc.add_trace(go.Scatter(x=ssel.index, y=ssel, name="Sell_thr", mode="lines", line=dict(dash="dot")))
fig_sc.update_layout(height=260, margin=dict(l=40,r=20,t=10,b=10))
st.plotly_chart(fig_sc, use_container_width=True, theme=None)
st.markdown("</div>", unsafe_allow_html=True)

# =====================  BACKTEST (opcjonalnie na dole)  =====================
st.markdown("<div class='panel'><div class='hdr'>Backtest (vol targeting) • Buy&Hold</div>", unsafe_allow_html=True)
ret = close.pct_change().fillna(0)
size = volatility_target_position(ret, target_vol_annual=target_vol, lookback=20)
bt = backtest(close, score, buy_thr, sell_thr, tc, sl, size_series=size)
m = metrics(bt["eq"], bt["ret"])
st.write(m)
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["eq"], name="Strategy"))
fig_eq.add_trace(go.Scatter(x=bt.index, y=bt["bh"], name="Buy&Hold"))
fig_eq.update_layout(height=300, margin=dict(l=40,r=20,t=10,b=10),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_eq, use_container_width=True, theme=None)

# =====================  AUTO-TUNE on click (wyniki pod spodem)  =====================
if do_autotune:
    st.markdown("<div class='panel'><div class='hdr'>🔁 Auto-Tune (walk-forward)</div>", unsafe_allow_html=True)
    space = grid_space()
    results, stability = walk_forward(close, sent, space, folds=4, cost_bps=tc+sl)
    st.write("Wyniki OS per fold:", [{"fold": r['fold'], "metrics": r['metrics_os']} for r in results])
    st.write("Stability (liczność wybieranych parametrów):", stability)
    st.markdown("</div>", unsafe_allow_html=True)
