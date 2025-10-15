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

# =========================  THEME (Excel-like dark)  =========================
st.markdown("""
<style>
:root{
  --bg:#111317; --panel:#1A1F26; --panel-2:#171B21; --text:#E8ECF1; --muted:#9BA3AE;
  --accent:#3BAFDA; --good:#00C389; --bad:#FF5C7A; --amber:#CBA85B; --border:rgba(255,255,255,.06);
}
html, body, .block-container{background:var(--bg) !important; color:var(--text) !important;}
.block-container{padding-top:0; max-width:1500px;}

.card{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:12px 14px; box-shadow:0 6px 16px rgba(0,0,0,.25);}
.card-2{background:var(--panel-2); border:1px solid var(--border); border-radius:14px; padding:12px 14px;}
.h1{font-weight:800; font-size:1.25rem; letter-spacing:.2px;}
.h2{font-weight:700; font-size:1.05rem; letter-spacing:.2px;}
.sub{color:var(--muted); font-size:.9rem;}
.flat input, .flat textarea, .flat [data-baseweb="select"]{background:#0E1115 !important; color:var(--text) !important; border-radius:10px;}
.stSlider > div[data-baseweb="slider"]{padding:6px 10px;}
.stSlider [data-baseweb="slider"] div{background-color:rgba(59,175,218,.18);}
.stSlider [role="slider"]{background:var(--accent) !important; box-shadow:0 0 0 3px rgba(59,175,218,.25);}

.stButton button{background:var(--accent); color:#0B0E12; border:none; border-radius:10px; padding:.55rem .9rem; font-weight:700;}
.stButton button:hover{filter:brightness(1.05); transform:translateY(-1px);}
.btn-ghost button{background:#232A34; color:var(--text);}
.btn-amber button{background:var(--amber); color:#101419;}
.btn-green button{background:var(--good); color:#07140F;}

.topbar{position:sticky; top:0; z-index:1000; background:linear-gradient(180deg, rgba(26,31,38,.95) 0%, rgba(26,31,38,.85) 100%);
  border-bottom:1px solid var(--border); backdrop-filter:blur(6px); padding:10px 14px; margin-bottom:10px;}
.logo{display:flex; gap:10px; align-items:center;}
.logo .mark{width:38px; height:38px; border-radius:8px; background:#1F2530; display:flex; align-items:center; justify-content:center; font-weight:800; color:var(--amber);}
.logo .title{font-weight:800; font-size:1.05rem; letter-spacing:.3px;}

.reco{border:1px solid rgba(203,168,91,.35); background:linear-gradient(180deg, rgba(203,168,91,.14), rgba(203,168,91,.06));}
.reco.good{border-color:rgba(0,195,137,.35); background:linear-gradient(180deg, rgba(0,195,137,.14), rgba(0,195,137,.06));}
.reco.bad{border-color:rgba(255,92,122,.35); background:linear-gradient(180deg, rgba(255,92,122,.14), rgba(255,92,122,.06));}

# tighten default spacing
section.main > div{padding-top:0;}
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

# =========================  STARTER PANEL (poziomy jak w Excelu)  =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h1'>⚙️ Ustawienia / Filtry</div>", unsafe_allow_html=True)
c0, c1, c2, c3 = st.columns([1.1, 2.2, 2.2, 2.2], gap="large")

with c0:
    st.markdown("<div class='h2'>Źródło</div>", unsafe_allow_html=True)
    src = st.selectbox("Źródło", ["Stooq", "Yahoo", "CSV"], label_visibility="collapsed")
    symbol = st.text_input("Symbol", value="btcpln", help="np. btcpln / eurusd / ^spx", key="sym", placeholder="ticker")
    csv_file = st.file_uploader("CSV (Date/Data, Close/Zamkniecie)", type=["csv"])
with c1:
    st.markdown("<div class='h2'>Wskaźniki</div>", unsafe_allow_html=True)
    p = SignalParams()
    p.rsi_window = st.slider("RSI window", 5, 30, p.rsi_window)
    p.ma_fast    = st.slider("MA fast", 5, 50, p.ma_fast)
    p.ma_mid     = st.slider("MA mid", 20, 100, p.ma_mid)
    p.ma_slow    = st.slider("MA slow", 20, 250, p.ma_slow)
with c2:
    st.markdown("<div class='h2'>Bollinger / RSI progi</div>", unsafe_allow_html=True)
    p.bb_window  = st.slider("BB window", 10, 40, p.bb_window)
    p.bb_std     = st.slider("BB std", 1.0, 3.0, p.bb_std)
    p.rsi_buy    = st.slider("RSI BUY", 10, 50, p.rsi_buy)
    p.rsi_sell   = st.slider("RSI SELL", 50, 90, p.rsi_sell)
with c3:
    st.markdown("<div class='h2'>Wagi & progi</div>", unsafe_allow_html=True)
    p.w_rsi      = st.slider("w_rsi", 0.0, 1.0, p.w_rsi)
    p.w_ma       = st.slider("w_ma", 0.0, 1.0, p.w_ma)
    p.w_bb       = st.slider("w_bb", 0.0, 1.0, p.w_bb)
    p.w_breakout = st.slider("w_breakout", 0.0, 1.0, p.w_breakout)
    p.w_sent     = st.slider("w_sent", 0.0, 1.0, p.w_sent)
    p.percentile_mode = st.checkbox("Progi dynamiczne (percentyle)", value=True)
    p.percentile_window = st.slider("Okno percentyli", 30, 180, p.percentile_window)
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
