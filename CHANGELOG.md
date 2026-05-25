# Changelog

All notable changes to ZEngine are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2026-05-25 — Initial Public Release

### Added
- **Look-ahead-free engine core** (`engine/data.py`, `engine/broker.py`)
  - `DataFeed` with bar-cursor guard — raises `LookAheadError` if any code reads beyond the current bar
  - `Broker` with market/limit orders, SL/TP, trailing stops, gap-fill, slippage model, spread schedule
  - `BrokerConfig` for commission (flat or pct), risk-based position sizing, max concurrent positions

- **Backtest runner** (`engine/backtest.py`)
  - `run_backtest()` — single date-range event loop
  - `run_splits()` — IS / OOS / FWD splits in one call
  - `audit_indicator_causality()` — split-half check that strategy `prepare()` uses no future data
  - Multi-timeframe variant: `run_backtest_mtf()`

- **Strategy base** (`engine/strategy.py`)
  - `BaseStrategy` with `PARAMS` template, `prepare()`, `on_bar()`, `on_fill()`, `on_close()` hooks
  - `MTFStrategy` mixin for multi-timeframe strategies

- **Indicators** (`engine/indicators.py`)
  - Guaranteed-causal: EMA, SMA, ATR, Bollinger Bands, RSI, MACD, Stochastic, ADX, VWAP, Supertrend, Donchian

- **Metrics** (`engine/metrics.py`)
  - `compute_metrics()` — Sharpe, Sortino, Calmar, Profit Factor, Win Rate, Max Drawdown, Expectancy, R-multiple

- **Walk-Forward Optimisation** (`engine/walkforward.py`)
  - Rolling and anchored modes; pluggable score functions; `WalkForwardResult` with OOS equity stitching

- **Monte Carlo simulation** (`engine/monte_carlo.py`)
  - Bootstrap resampling of trade PnLs; probability of ruin; equity fan-chart bands

- **Portfolio backtesting** (`engine/portfolio.py`)
  - Shared equity pool across multiple instruments; portfolio-level risk cap; strict chronological bar merging

- **Funded account simulator** (`engine/funded.py`)
  - Prop-firm rules: daily/total drawdown limits, profit target, trailing drawdown; `FundedResult` pass/fail report

- **Data sources**
  - CSV (`engine/data_sources/csv_source.py`)
  - Yahoo Finance (`engine/data_sources/yahoo_source.py`)
  - MetaTrader 5 (`engine/data_sources/mt5_source.py`) — Windows + Mac fallback
  - cTrader Open API (`engine/data_sources/ctrader_source.py`) — OAuth2, real bid/ask spread per bar

- **Example strategies** (`strategies/`)
  - `MACrossStrategy` — dual EMA crossover
  - `IFVGStrategy` — Implied Fair Value Gap with session filter and ATR-based SL/TP

- **Streamlit UI** (`app.py`) — ten interactive tabs:
  | Tab | Feature |
  |-----|---------|
  | 📊 Overview | Key metrics, equity curve, drawdown chart |
  | 📋 Trades | Full trade log with filters |
  | ⚡ Compare | Side-by-side IS / OOS / FWD split results |
  | 🔄 Walk-Forward | Grid search + per-window OOS/IS score ratio |
  | 🎲 Monte Carlo | Bootstrap simulation fan chart + ruin probability |
  | 💼 Portfolio | Multi-symbol shared-equity backtest |
  | 📈 Chart | MT5-style candlestick chart with trade overlays |
  | 🏆 Funded | Prop-firm rules simulator |
  | 🔧 Strategy | Live parameter tuning with instant re-run |
  | ℹ️ About | Engine architecture explainer |

- **CI** (`.github/workflows/test.yml`) — ruff lint + mypy type-check + pytest coverage on Python 3.10/3.11/3.12

- **Tests** (`tests/`) — 85+ tests covering engine core, features, data sources, and funded rules

### Library quality
- All engine progress output uses Python `logging` (callers can silence or redirect)
- `BaseStrategy.PARAMS` template is normalised on init — nested spec-dicts are safely unwrapped
- Zero `print()` calls in library code; bare prints remain only in example scripts
