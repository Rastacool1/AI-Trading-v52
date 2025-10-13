import numpy as np
import pandas as pd

def backtest(close: pd.Series, score: pd.Series, buy_thr, sell_thr,
             tc_bps: float = 5, slip_bps: float = 5,
             size_series: pd.Series | None = None) -> pd.DataFrame:
    # sygnał pozycji: 1 long, 0 cash (z dynamicznymi progami)
    if isinstance(buy_thr, pd.Series):
        buy_th = buy_thr.reindex(score.index).fillna(method="ffill").fillna(0.6)
        sell_th = sell_thr.reindex(score.index).fillna(method="ffill").fillna(-0.6)
        sig = (score >= buy_th).astype(float)
        sig[score <= sell_th] = 0.0
    else:
        sig = (score >= buy_thr).astype(float)
        sig[score <= sell_thr] = 0.0

    sig = sig.shift(1).fillna(0)  # wejście następnego dnia
    ret = close.pct_change().fillna(0)

    pos = sig if size_series is None else (sig * size_series.reindex(sig.index).fillna(method="ffill").fillna(0))

    churn = (pos.diff().abs()).fillna(pos.abs())
    cost = churn * (tc_bps + slip_bps) / 10000.0

    strat_ret = pos*ret - cost
    eq = (1 + strat_ret).cumprod()
    bh = (1 + ret).cumprod()

    df = pd.DataFrame({
        "ret": strat_ret,
        "eq": eq,
        "bh": bh,
        "pos": pos,
        "sig": sig
    }, index=close.index)
    return df

def metrics(equity: pd.Series, ret: pd.Series) -> dict:
    daily = ret
    n = len(equity)
    cagr = equity.iloc[-1]**(252/max(n,1)) - 1 if n > 0 else 0
    vol = daily.std()*np.sqrt(252) if n > 1 else 0
    sharpe = (daily.mean()/daily.std())*np.sqrt(252) if daily.std() > 0 else 0
    downside = daily[daily<0]
    sortino = (daily.mean()/downside.std())*np.sqrt(252) if downside.std()>0 else 0
    dd = (equity / equity.cummax() - 1).min() if n>0 else 0
    wins = (daily>0).sum(); losses = (daily<0).sum()
    hit = wins / max(wins+losses,1)
    pf = daily[daily>0].sum() / abs(daily[daily<0].sum()) if (daily[daily<0].sum())<0 else np.inf
    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe), "Sortino": float(sortino), "MaxDD": float(dd), "HitRate": float(hit), "ProfitFactor": float(pf)}
