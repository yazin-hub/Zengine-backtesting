# ZEngine — Look-Ahead-Free Backtesting Engine

[![Tests](https://github.com/yazin-hub/zengine-backtesting/actions/workflows/test.yml/badge.svg)](https://github.com/yazin-hub/zengine-backtesting/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Structurally** impossible to introduce look-ahead bias. Not by convention — by design.

ZEngine is an open-source, strategy-agnostic backtesting framework built around a single guarantee: the strategy can never see future price data, even accidentally. A `DataFeed` guard raises a `LookAheadError` at runtime if any code attempts to access a bar beyond the current cursor position.

Built for serious algo traders who have been burned by inflated backtest results.

---

## Why ZEngine?

Most backtesting bugs don't come from careless code — they come from subtle look-ahead in signal generation: passing the full dataset to a function that internally scans forward, or computing "when does price fill this zone?" with hindsight. ZEngine makes this class of bug impossible:

- **DataFeed guard** — `feed.close[i+1]` raises `LookAheadError` when cursor is at `i`
- **Strict fill timing** — orders placed at bar `i` can only fill at bar `i+1` or later
- **Same-bar SL/TP protection** — a trade filled at bar `i` is never checked for SL/TP on bar `i`
- **Fresh state per split** — IS/OOS/FWD splits each get independent strategy instances

---

## Install

```bash
git clone https://github.com/yazin-hub/zengine-backtesting.git
cd zengine
pip install -e .
```

**Requirements:** Python 3.10+, see `pyproject.toml` for dependencies.

---

## Quick Start — Python

```python
import pandas as pd
from engine.backtest import run_splits
from engine.broker import BrokerConfig
from strategies import REGISTRY

# Load your OHLCV data
df = pd.read_csv("your_data.csv", parse_dates=[0], index_col=0)
df.index = pd.to_datetime(df.index, utc=True)

# Pick a strategy
IFVGStrategy = REGISTRY["IFVG M1 Scalping"]

# Configure the broker
broker_cfg = BrokerConfig(
    commission_flat=6.0,   # $6 per 100-unit lot round-trip
    lot_size=100.0,
    max_concurrent=2,
    risk_usd=20.0,         # $20 risk per trade
    size_mode="fixed_risk",
)

# Run IS / OOS / FWD splits
results = run_splits(
    df=df,
    strategy_class=IFVGStrategy,
    params={},               # use strategy defaults
    broker_config=broker_cfg,
    splits={
        "IS  (2020–2024)": ("2020-01-01", "2024-12-31"),
        "OOS (2025)":      ("2025-01-01", "2025-12-31"),
        "FWD (2026–now)":  ("2026-01-01", None),
    },
)

for label, res in results.items():
    print(f"{label}: {res.n_trades} trades, final equity ${res.final_equity:,.2f}")
```

---

## Quick Start — Streamlit UI

```bash
cd zengine
streamlit run app.py
```

The UI has ten tabs:

| Tab | What it shows |
|-----|--------------|
| **Overview** | Headline metrics (PnL, WR, Sharpe, max DD) for each split |
| **Equity** | Equity curves + drawdown chart, all splits overlaid |
| **Trades** | Filterable trade browser + PnL distribution histogram |
| **Breakdown** | By direction, exit reason, monthly PnL heatmap |
| **Compare** | Run two parameter sets side by side |
| **Walk-Forward** | Rolling IS/OOS optimisation — score timeline, combined OOS equity, best params per window |
| **Monte Carlo** | Bootstrap equity fan chart, prob ruin, drawdown & Sharpe distributions |
| **Portfolio** | Multi-symbol portfolio backtest with shared equity and position limits |
| **Chart** | MT5-style candlestick chart with trade overlays, Win/Loss toggle, and trade filters |
| **Funded** | Simulate funded account rules (FTMO, Topstep, The5ers) against your backtest equity curve |

Parameters are auto-generated from each strategy's `PARAMS` dict — no UI code changes needed when you add a new strategy.

---

## Data Sources

ZEngine supports four data sources. All sources produce the same DataFrame format so the engine and strategies work identically regardless of where the data came from.

| Source | Platform | M1 history | Real spread | Setup |
|--------|----------|------------|-------------|-------|
| **CSV** | All | Unlimited (your file) | No | None |
| **MetaTrader 5** | Windows only | 7–10 years | ✅ Per bar | MT5 terminal running |
| **cTrader** | All (Mac ✅) | ~1 year M1 | ✅ Per bar | Register app once |
| **Yahoo Finance** | All (Mac ✅) | 7 days M1, unlimited daily | No | `pip install yfinance` |

### Optional installs

```bash
# MetaTrader 5 (Windows only)
pip install "zengine[mt5]"

# cTrader (all platforms — Mac, Linux, Windows)
pip install "zengine[ctrader]"

# Yahoo Finance (all platforms, free, no account)
pip install "zengine[yahoo]"

# Everything at once
pip install "zengine[all-sources]"
```

---

### MetaTrader 5 Setup

No credentials needed — ZEngine talks to the MT5 terminal process already running on your machine (same as Excel talking to another Office app). Just:

1. Open MetaTrader 5 and log in to your broker account as normal
2. Install the package: `pip install "zengine[mt5]"`
3. In the ZEngine UI: select **MetaTrader 5**, pick your symbol and dates, click **📥 Fetch**

Spread, commission, and lot size are auto-filled from your broker's symbol info.

> **Mac / Linux:** MT5 terminal is Windows-only. Options: use a Windows VPS to export a CSV, use cTrader instead, or use Yahoo Finance for development.

---

### cTrader Setup

cTrader uses OAuth2 — you register your own application once (free, takes 5 minutes), then authenticate with a single browser click. Works on **Mac, Linux, and Windows**.

#### Step 1 — Register your application

1. Go to **[openapi.ctrader.com](https://openapi.ctrader.com)** and sign in with your cTrader account
2. Click **New Application**
3. Fill in any name (e.g. `ZEngine Backtest`)
4. In the **Redirect URI** field, enter exactly:

```
http://localhost:8182/callback
```

> ⚠️ This must match exactly — same port (8182), same path (/callback). ZEngine starts a local server on this port during authentication.

5. Copy your **Client ID** and **Client Secret** — you'll paste these into the ZEngine UI

#### Step 2 — Install and authenticate

```bash
pip install "zengine[ctrader]"
```

In the ZEngine UI:
1. Select **cTrader** as the data source
2. Choose **Demo** or **Live** (demo spreads differ from live — use live for production accuracy)
3. Enter your **Client ID** and **Client Secret**
4. Click **🔑 Authenticate** — your browser opens to cTrader's login page
5. Log in and approve — you're redirected back automatically
6. Tokens are saved to `~/.zengine/ctrader_tokens.json` — you won't need to authenticate again unless tokens expire (~60 days)

#### Why demo vs live matters

Demo accounts use synthetic spreads that are typically 20–40% tighter than live, and don't widen during news events. If you backtest on demo spread data and trade live, your results will be slightly optimistic. **Connect your actual live account for production-accurate backtests** — ZEngine reads historical spread data, it never places orders.

#### What you get

- OHLC bars with **real bid/ask spread per bar** (ask\_open − bid\_open at every minute)
- Tick volume per bar
- Symbol info: contract size, commission structure, digits → auto-fills BrokerConfig
- Works cross-platform: no Windows required

---

### Yahoo Finance Setup

Free, no account, cross-platform. Best for daily-bar strategies or quick validation on Mac/Linux.

```bash
pip install "zengine[yahoo]"
```

**Limitations:** M1 data is limited to the last 7 days. For XAUUSD use `GC=F` (Gold Futures, real volume) or `XAUUSD=X` (spot, sometimes unavailable). The UI shows a warning automatically if your requested range exceeds the interval's limit.

---

## Data Format

Any CSV with a DateTime index and OHLCV columns:

```
DateTime,Open,High,Low,Close,Volume
2020-01-02 00:01:00,1517.23,1517.45,1517.10,1517.38,42
2020-01-02 00:02:00,1517.38,1517.60,1517.25,1517.51,38
...
```

- Column names are case-insensitive (`Open`, `open`, `OPEN` all work)
- Timezone: UTC preferred; naive datetimes are assumed UTC
- Volume column is optional
- Any instrument, any timeframe

---

## Adding a Strategy

1. **Create `strategies/my_strategy.py`:**

```python
from engine.strategy import BaseStrategy
from engine.data import DataFeed
from engine.broker import Broker, OrderSide, OrderType
from engine import indicators as ind

class MyStrategy(BaseStrategy):

    NAME = "My Strategy"

    PARAMS = {
        "period": {"default": 20, "min": 5,   "max": 100, "step": 1,   "label": "Period"},
        "atr_sl": {"default": 1.5,"min": 0.5, "max": 5.0, "step": 0.1, "label": "SL (ATR×)"},
        "session":{"default": "both", "options": ["london","ny","both","all"], "label": "Session"},
    }

    def prepare(self, feed: DataFeed) -> None:
        # Called ONCE before the bar loop.
        # Access raw arrays here — feed._cursor is -1, no look-ahead risk.
        close = feed.close._data
        high  = feed.high._data
        low   = feed.low._data

        # Compute causal indicators and attach to feed
        feed._attach("sma",  ind.sma(close, self.params["period"]))
        feed._attach("atr",  ind.atr(high, low, close, 14))
        feed._attach("sess", ind.session_mask(feed.index, self.params["session"]).astype(float))

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        i = feed.i
        if i < 2 or not feed["sess"][i] or not broker.has_capacity:
            return

        close_i = feed.close[i]   # guarded — feed.close[i+1] raises LookAheadError
        sma_i   = feed["sma"][i]
        atr_i   = feed["atr"][i]

        if close_i > sma_i:
            sl = close_i - 1.5 * atr_i
            tp = close_i + 3.0 * atr_i
            broker.place_order(
                side=OrderSide.LONG, order_type=OrderType.MARKET,
                limit_price=close_i, sl=sl, tp=tp, placed_bar=i,
            )
```

2. **Register it in `strategies/__init__.py`:**

```python
from .my_strategy import MyStrategy

REGISTRY = {
    ...,
    MyStrategy.NAME: MyStrategy,
}
```

That's it. The Streamlit UI will automatically discover your strategy and generate sliders for all its parameters.

---

## Built-in Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| `IFVG M1 Scalping` | `IFVGStrategy` | ICT Implied Fair Value Gap — liquidity sweep + FVG inversion zones. Defaults tuned for XAUUSD M1. |
| `EMA Crossover` | `MACrossStrategy` | Simple fast/slow EMA crossover with ATR-based SL/TP. Works on any instrument/timeframe. |

---

## Available Indicators (`engine/indicators.py`)

| Category | Functions |
|----------|-----------|
| **Trend** | `ema`, `sma`, `wma`, `dema`, `tema` |
| **Volatility** | `atr`, `bollinger` |
| **Momentum** | `rsi`, `macd`, `stochastic`, `cci`, `mfi`, `adx` |
| **Structure** | `swing_high`, `swing_low`, `fair_value_gaps`, `liquidity_sweeps` |
| **Filters** | `session_mask` (London / NY / both / all) |

All indicators are computed causally (no look-ahead) using rolling/vectorised numpy operations.

---

## Advanced Features

### Walk-Forward Optimisation

Prevents overfitting by never letting your optimiser see OOS data. The engine rolls a window across your dataset, optimises parameters on the IS portion, then immediately validates on the OOS portion that follows.

```python
from engine.walkforward import run_walk_forward

wf = run_walk_forward(
    df            = df,
    strategy_class = IFVGStrategy,
    param_grid    = {
        "atr_sl": [1.0, 1.5, 2.0],
        "period": [14, 21, 30],
    },
    broker_config   = broker_cfg,
    window_bars     = 50_000,    # IS window size
    step_bars       = 10_000,    # OOS window size (roll-forward step)
    score_fn        = "sharpe",  # or "pf", "calmar", "pnl", "sortino", "win_rate"
    starting_equity = 5_000.0,
    anchored        = False,     # True = anchored (growing IS), False = rolling
)

print(wf.summary())
# Combined OOS equity:  wf.combined_oos_equity
# Best params per window: [(w.best_params, w.oos_score) for w in wf.windows]
# Avg score ratio (OOS/IS): wf.avg_score_ratio  — close to 1.0 = no overfitting
```

### Monte Carlo Simulation

Bootstrap-resamples the trade PnL sequence N times to build a distribution of possible outcomes. The equity fan chart shows where your strategy ends up across thousands of orderings.

```python
from engine.monte_carlo import run_monte_carlo

mc = run_monte_carlo(
    result          = results["IS (2020–2024)"],
    n_sims          = 1_000,
    starting_equity = 5_000.0,
    ruin_floor      = 2_500.0,   # 50% of starting equity
    seed            = 42,
)

print(f"Prob ruin:      {mc.prob_ruin:.1%}")
print(f"Median Sharpe:  {mc.sharpes.mean():.2f}")

# Fan chart percentile bands
bands = mc.equity_bands(percentiles=[5, 25, 50, 75, 95])
```

### Multi-Timeframe Strategies

Access higher-timeframe data from a primary (fast) timeframe with full look-ahead protection. The engine uses close-time semantics — an H1 bar is only visible after its close, never while it's still forming.

```python
from engine.mtf import MTFStrategy, build_mtf_feeds
from engine.backtest import run_backtest_mtf
from engine import indicators as ind

class MyCrossStrategy(MTFStrategy):
    NAME       = "MA Cross MTF"
    PARAMS     = {}
    TIMEFRAMES = ["M1", "H1"]   # first = primary (drives bar loop)

    def prepare_mtf(self, feeds):
        feeds["H1"]._attach("trend_ema",
            ind.ema(feeds["H1"].close._data, 50))
        feeds["M1"]._attach("atr",
            ind.atr(feeds["M1"].high._data,
                    feeds["M1"].low._data,
                    feeds["M1"].close._data, 14))

    def on_bar_mtf(self, feeds, broker):
        m1, h1 = feeds["M1"], feeds["H1"]
        i, h1_i = m1.i, h1.i
        if h1_i < 0 or not broker.has_capacity:
            return
        # h1["trend_ema"][h1_i] is the last COMPLETED H1 bar — no look-ahead
        bias = m1.close[i] > h1["trend_ema"][h1_i]
        ...

mtf = build_mtf_feeds({"M1": df_m1, "H1": df_h1}, primary_key="M1")
result = run_backtest_mtf(mtf, MyCrossStrategy(), broker_cfg)
```

### Portfolio Backtesting

Run multiple strategies on multiple symbols with shared capital. Position sizing respects a portfolio-level risk cap, not just per-symbol limits.

```python
from engine.portfolio import PortfolioConfig, run_backtest_portfolio

pcfg = PortfolioConfig(
    symbols           = ["XAUUSD", "EURUSD"],
    broker_configs    = {
        "XAUUSD": broker_cfg_gold,
        "EURUSD": broker_cfg_fx,
    },
    starting_equity   = 20_000.0,
    max_total_positions = 4,      # across ALL symbols
    max_total_risk_usd  = 200.0,  # total open risk cap
)

portfolio_result = run_backtest_portfolio(
    dfs        = {"XAUUSD": df_gold, "EURUSD": df_fx},
    strategies = {"XAUUSD": IFVGStrategy(), "EURUSD": MACrossStrategy()},
    portfolio_cfg = pcfg,
)

# Shared equity curve across all symbols
print(portfolio_result.equity_curve)
# Per-symbol trade results
for sym, res in portfolio_result.symbol_results.items():
    print(f"{sym}: {res.n_trades} trades")
```

---

## Project Structure

```
zengine/
├── engine/
│   ├── data_sources/
│   │   ├── __init__.py       # available_sources() factory
│   │   ├── csv_source.py     # CSV / file-upload (always available)
│   │   ├── mt5_source.py     # MetaTrader 5 terminal (Windows only)
│   │   ├── ctrader_source.py # cTrader Open API OAuth2 (all platforms)
│   │   └── yahoo_source.py   # Yahoo Finance via yfinance (all platforms)
│   ├── data.py          # DataFeed + LookAheadError + CausalityError
│   ├── broker.py        # Virtual broker (orders, fills, SL/TP, sizing, spread, trail)
│   ├── strategy.py      # BaseStrategy ABC
│   ├── backtest.py      # Event loop + IS/OOS/FWD splits + causality audit
│   ├── metrics.py       # Sharpe, Sortino, Calmar, drawdown, trade log
│   ├── indicators.py    # Causal indicator library
│   ├── walkforward.py   # Walk-forward optimisation
│   ├── monte_carlo.py   # Monte Carlo bootstrap simulation
│   ├── mtf.py           # Multi-timeframe feeds + MTFStrategy
│   └── portfolio.py     # Portfolio-level backtesting + shared equity
├── strategies/
│   ├── __init__.py      # REGISTRY — add your strategy here
│   ├── ifvg.py          # IFVG M1 Scalping strategy
│   └── ma_cross.py      # EMA Crossover strategy
├── tests/
│   ├── test_engine.py        # Core engine: look-ahead, fills, spread, trailing, causality
│   ├── test_features.py      # Walk-forward, Monte Carlo, MTF, Portfolio
│   └── test_data_sources.py  # CSV, MT5, cTrader, Yahoo source tests (mocked)
├── app.py               # Streamlit Strategy Lab UI (10 tabs)
├── pyproject.toml
└── README.md
```

---

## Broker Configuration

```python
BrokerConfig(
    commission_pct   = 0.0,     # % of trade value (e.g. 0.001 = 0.1%)
    commission_flat  = 6.0,     # flat $ per lot round-trip
    lot_size         = 100.0,   # units per lot (100 oz for XAUUSD)
    max_concurrent   = 1,       # max simultaneous open positions
    risk_usd         = 20.0,    # $ risk per trade (for size_mode="fixed_risk")
    size_mode        = "fixed_risk",  # fixed_risk | fixed_size | pct_equity
    fixed_size       = 1.0,     # units (for size_mode="fixed_size")
    pct_equity_risk  = 0.01,    # fraction of equity (for size_mode="pct_equity")
    slippage_pct     = 0.0,     # slippage as fraction of price
)
```

---

## Key Design Decisions

**Why `LookAheadError` instead of just being careful?**
Convention fails. A strategy author passes the full dataset to a helper function, which scans forward for zone inversions — perfectly readable code, catastrophic look-ahead. A runtime error is the only reliable guard.

**Why `placed_bar < bar_idx` (strict)?**
Same-bar fill means the strategy could place an order and observe its fill in the same iteration. In live trading, order routing latency means the fill always arrives on a subsequent bar. The strict comparison models this correctly.

**Why daily-bucketed Sharpe?**
Treating each M1 bar as an independent return observation inflates annualised Sharpe by sqrt(390) (~20×). Daily bucketing (grouping by exit day) gives a number comparable to what you'd see from a daily strategy and from live trading reports.

**Why the causality audit in `prepare()`?**
The bar-loop `LookAheadError` guard only protects `on_bar`. `prepare()` receives the full dataset array — a strategy author could accidentally pass it to `scipy.signal.savgol_filter` (bidirectional), or normalise using the global max. The causality audit runs `prepare()` twice (half dataset vs full dataset) and compares indicator values near the midpoint. If any value changes when future data is added, the indicator is non-causal and `CausalityError` is raised before a single trade is evaluated.

**Why close-time semantics for MTF alignment?**
Most open-source MTF implementations show a higher-timeframe bar as soon as its opening time passes — meaning the M1 strategy at 08:30 sees the H1 08:00 bar, which is still forming. ZEngine uses close-time alignment: an H1 bar is visible only once the next H1 bar has started (at 09:00). This is the only semantics that matches live trading.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers 210+ tests across four files:

- `tests/test_engine.py` — core engine: look-ahead guard, same-bar fill protection, causality audit, spread/slippage asymmetry, trailing stops manual + auto (85 tests)
- `tests/test_features.py` — walk-forward (11 tests), Monte Carlo (11 tests), multi-timeframe cursor alignment (6 tests), portfolio backtesting (7 tests)
- `tests/test_data_sources.py` — CSV loading, MT5 source (mocked), cTrader source + delta-decoding (mocked), Yahoo Finance source (mocked) (75 tests)
- `tests/test_funded.py` — funded account simulation: config validation, absolute/trailing drawdown, daily loss limit, profit target, payout cycles, equity scaling, edge cases (50 tests)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide — dev setup, code style, how to add a strategy or data source, and the PR checklist.

Short version: fork → branch → `pip install -e ".[dev]"` → implement → `pytest` passes → open a PR.

**Please do not submit strategies with look-ahead bias** — the engine will catch it at test time via `LookAheadError`.

---

## License

MIT — see `LICENSE`.
