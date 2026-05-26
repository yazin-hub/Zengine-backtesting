# ZEngine — Open Source Backlog & Future Improvements

> Things found, fixed, and built during the BTC/ETH pairs trading research session.
> Add these to the public ZEngine repo when ready — in priority order.

---

## 🐛 Bugs Fixed (already fixed in private repo — port to public)

### 1. `close_all_at()` exit_reason hardcoded
**File:** `engine/broker.py`, `engine/portfolio.py`  
**Problem:** `close_all_at()` was hardcoding `exit_reason = "end_of_data"` for every exit,
meaning 99% of trades were being filtered out of results (only `"end_of_data"` trades were excluded,
so real exits like `"zscore_exit"` were disappearing).  
**Fix:** Add `exit_reason: str = "end_of_data"` parameter to `close_all_at()` signature
and pass it through to `t.exit_reason = exit_reason`.  
**Impact:** Critical — without this fix the engine reports 0 trades for any pairs strategy.

### 2. `PortfolioBroker.close_all_at()` signature mismatch
**File:** `engine/portfolio.py`  
**Problem:** `PortfolioBroker` override of `close_all_at()` didn't include the `exit_reason`
parameter, causing mypy CI to fail.  
**Fix:** Match the base `Broker` signature exactly.

---

## 🚀 Features to Add

### 3. Pairs Trading Engine — `engine/pairs.py`
Full pairs trading backtest runner. Already built in private repo.  
**What it does:**
- Accepts two OHLCV DataFrames (asset A and asset B)
- Inner joins on timestamp — only trades bars where both assets have data
- Runs two `Broker` instances simultaneously (one per leg)
- Strategy sets `self._exit_requested = True` → runner calls `close_all_at()` on both brokers
- Returns `PairsResult` with: `closed_trades_a`, `closed_trades_b`, `equity_curve`, `spread_history`, `zscore_history`
- Filters out `exit_reason == "end_of_data"` from closed trades automatically

**Key classes:**
- `PairsStrategy(ABC)` — base class, exposes `feed_a`, `feed_b`, `broker_a`, `broker_b`
- `PairsResult` — result container with `.summary()` method
- `run_backtest_pairs(df_a, df_b, strategy, broker_config_a, broker_config_b, ...)` — main runner

---

### 4. Pairs Trading Example Strategy — Strategy Library
**File:** `strategies/btc_eth_spread.py` (outline only — no tuned params)  
**Description:** BTC/ETH statistical spread mean reversion.

**Concept:**
- Spread = BTC_close − (hedge_ratio × ETH_close)
- Z-score = (Spread − rolling_mean) / rolling_std
- Entry when |Z| > entry_z, exit when |Z| < exit_z
- Emergency stop when |Z| > sl_z (spread diverging further)

**Key PARAMS to expose (let users find values themselves):**
```python
PARAMS = {
    "hedge_ratio":   {"default": 15,   "min": 1,    "max": 50,   "step": 1},
    "lookback":      {"default": 100,  "min": 20,   "max": 500,  "step": 10},
    "entry_z":       {"default": 2.0,  "min": 0.5,  "max": 4.0,  "step": 0.1},
    "exit_z":        {"default": 0.5,  "min": 0.0,  "max": 2.0,  "step": 0.05},
    "sl_z":          {"default": 3.0,  "min": 0.0,  "max": 6.0,  "step": 0.25},
    "volume_btc":    {"default": 0.01, "min": 0.001,"max": 1.0,  "step": 0.001},
    "corr_min":      {"default": 0.6,  "min": 0.0,  "max": 0.99, "step": 0.05},
    "max_hold_bars": {"default": 0,    "min": 0,    "max": 200,  "step": 5},
    "adx_period":    {"default": 14,   "min": 5,    "max": 50,   "step": 1},
    "adx_max":       {"default": 0,    "min": 0,    "max": 60,   "step": 5},
    "dynamic_hedge": {"default": False},
}
```

**Filters implemented (outline):**
1. **Rolling correlation filter** — skip entry if BTC/ETH log-return correlation < `corr_min`
   - Gatev et al. (2006): pairs trading requires stable correlation
2. **Max holding period (time stop)** — force exit after `max_hold_bars` bars
   - Avellaneda & Lee (2010): OU half-life guides max holding period
3. **Spread ADX filter** — skip entry when spread ADX > `adx_max`
   - Wilder (1978): ADX > 25 = trending regime, mean reversion unlikely
   - Applied to spread as single-price series (simplified DM: diff of spread)
4. **Dynamic hedge ratio** — optional rolling OLS regression (toggle with `dynamic_hedge`)
   - Alexander (2001): cointegration-based hedge ratio

---

### 5. Forward (FWD) Data Split Support
**File:** `engine/backtest.py` → `run_splits()`  
**Problem:** Current `run_splits()` only supports IS and OOS splits from a single DataFrame.
Pairs trading needs a **separate** FWD DataFrame (different date range, different file).  
**Feature request:** Accept optional `df_fwd` parameter.  
Also: the Pairs Trading Streamlit tab uses a workaround of two separate file inputs.
Cleaner to support this natively in the engine.

---

### 6. Pairs Trading Streamlit Tab — `app.py`
**Tab:** `🔗 Pairs Trading` (Tab 11)  
Already built. Shows:
- Locked params display
- IS / OOS / FWD data file pickers
- Run button (runs all three splits)
- Metrics comparison table (IS vs OOS vs FWD)
- Equity curves overlaid (blue=IS, orange=OOS, green=FWD)
- Drawdown chart
- Trade PnL histogram
- Exit reason breakdown by period

Port this tab to the public repo once `engine/pairs.py` is added.

---

### 7. Mixed Timezone CSV Handling
**File:** `engine/data.py` or any CSV loader  
**Problem:** cTrader exports naive timestamps up to a certain date, then switches to
`+00:00` offset format (e.g. `2026-01-01 00:00:00+00:00`). pandas `parse_dates` crashes.  
**Fix:**
```python
df["time"] = df["time"].str.replace(r"[+-]\d{2}:\d{2}$|Z$", "", regex=True)
df["time"] = pd.to_datetime(df["time"])
```
Strip tz offset before parsing — makes all timestamps naive UTC. Add this to the CSV loader.

---

### 8. cTrader Data Exporter cBot (C#)
**Useful tool for users:** A ready-made cTrader cBot that exports M15 OHLCV data to CSV.
Add to `tools/` or `docs/data_sources/`.

**Key points:**
- Use Mac/Linux paths in `OutputDirectory` parameter (not Windows `C:\...`)
- `LoadMoreHistory()` loop with `maxAttempts` guard
- Writes `DateTime,Open,High,Low,Close,Volume` header
- Works for any symbol/timeframe — just change `TimeFrame.Minute15`

---

## 📋 Strategy Library Additions

| Strategy | Type | Status |
|---|---|---|
| IFVG M1 Scalping | Single asset | ✅ In public repo |
| EMA Crossover | Single asset | ✅ In public repo |
| BTC/ETH Spread | Pairs (mean reversion) | 🔲 Add outline |
| Gold/Silver Spread | Pairs (mean reversion) | 🔲 Future idea |
| Stat Arb (3+ assets) | Multi-leg | 🔲 Future idea |

---

## 📝 Notes

- All pairs strategy params in this file are **outlines only** — defaults are neutral,
  not the research-optimised values. Users should run their own IS optimisation.
- The `adx_max=0` and `corr_min=0.0` defaults disable the filters — clean starting point.
- The `dynamic_hedge=False` default uses fixed ratio — simpler, more stable on M15.
- Reference papers to cite in docs:
  - Gatev, Goetzmann & Rouwenhorst (2006) — pairs trading performance
  - Vidyamurthy (2004) — pairs trading quantitative methods
  - Avellaneda & Lee (2010) — statistical arbitrage in US equities
  - Wilder (1978) — ADX directional movement
  - Alexander (2001) — market models, cointegration hedge ratio
