# AI Trading Edge v5.2

Gotowa paczka łącząca to, co już działało u Ciebie (reżimy, whipsaw re-entry, silence-guard, „2 z 3” exit, ATR trailing),
z rzeczami, które dodałem: walk-forward auto‑tune, volatility targeting / pół‑Kelly, dynamiczne progi percentylowe,
confidence/explainability oraz panel wrażliwości kosztów i progów.

## Najważniejsze funkcje
- **Źródła danych**: Stooq (CSV HTTP), Yahoo (yfinance), pliki CSV (Date/Data, Close/Zamkniecie).
- **Wskaźniki**: RSI, SMA/EMA, Bollinger Bands, ATR, HH/HL/LL/LH (swingi).
- **Reżimy**: bull/bear/side (MA_mid + filtr zmienności), wpływ na trailing i progi.
- **Sygnały**: ensemble (RSI/MA/BB/ATR/Breakout/Sentiment), whipsaw re-entry, silence-guard, „2 z 3” exit.
- **Sentyment**: heurystyka z VIX lub CSV ze skorem (-1..+1) z EWMA i cap/floor.
- **Auto‑Tune**: walk‑forward (purged) z raportem stabilności parametrów, priorytet Sharpe → CAGR.
- **Ryzyko i kapitał**: volatility targeting do celu (np. 12% rocznie), pół‑Kelly „clipped”, hard-DD de‑risk overlay.
- **Backtest**: metryki CAGR/Sharpe/Sortino/MaxDD/Vol/WinRate/ProfitFactor; benchmark Buy&Hold.
- **UI**: Streamlit (ciemny motyw), status „KUP/SPRZEDAJ/AKUMULUJ/REDUKUJ”, wykresy, tabele, panel auto‑tune, panel wrażliwości.
- **Explainability**: wkład RSI/MA/BB/ATR/Sentiment do score + confidence.

## Szybki start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Domyślny symbol: `^spx` (Stooq). Możesz też wybrać Yahoo lub wgrać CSV.

## Struktura
```
AI-Trading-Edge-v5.2/
├─ app.py
├─ config.toml
├─ requirements.txt
└─ core/
   ├─ __init__.py
   ├─ data.py
   ├─ indicators.py
   ├─ regime.py
   ├─ sentiment.py
   ├─ signals.py
   ├─ risk.py
   ├─ backtest.py
   ├─ autotune.py
   └─ sensitivity.py
```

## Uwaga dot. zależności
Do działania źródeł Yahoo wymagane jest `yfinance`. Jeśli chcesz tylko CSV i Stooq, możesz pominąć tę zależność.
