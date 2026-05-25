"""
engine/backtest.py — Main Event Loop
======================================

The engine is the single source of truth for time. It advances the DataFeed
cursor bar by bar, calls the strategy, then calls the broker. This order
guarantees the strategy CANNOT react to fills that happen on the current bar.

Timeline per bar i:
  1. feed._advance(i)          — expose bars 0..i to strategy
  2. strategy.on_bar(feed, broker) — strategy may place orders
  3. broker.on_bar(i, ...)     — broker fills pending, manages positions
  4. equity_curve[i] = broker.equity

Look-ahead is impossible because:
  - feed blocks index > i via LookAheadError
  - broker fills happen AFTER strategy runs (step 3 after step 2)
  - Strategy cannot observe fills until the NEXT bar

Supports:
  - Any strategy implementing BaseStrategy
  - IS / OOS / FWD splits or custom date ranges
  - Warmup period (strategy skips signal generation)
  - Equity curve output for each split
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .data import DataFeed, CausalityError
from .broker import Broker, BrokerConfig, OrderStatus
from .strategy import BaseStrategy

if TYPE_CHECKING:
    from .mtf import MultiTimeframeFeed, MTFStrategy  # noqa: F401

# Library-style logging: callers control visibility via
#   logging.getLogger("engine.backtest").setLevel(logging.INFO)
# or by configuring the root "engine" logger.  Default level is WARNING
# so installed library users see nothing unless they opt in.
_log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    label:        str
    trades:       list          # list of Order (closed)
    equity_curve: np.ndarray
    timestamps:   pd.DatetimeIndex
    params:       dict
    strategy_name: str

    @property
    def closed_trades(self):
        return [t for t in self.trades
                if t.status == OrderStatus.CLOSED
                and t.exit_reason != "end_of_data"]

    @property
    def n_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def final_equity(self) -> float:
        valid = self.equity_curve[~np.isnan(self.equity_curve)]
        return float(valid[-1]) if len(valid) else 0.0


def audit_indicator_causality(
    df: pd.DataFrame,
    strategy: "BaseStrategy",
    symbol: str = "UNKNOWN",
    timeframe: str = "M1",
    n_check_bars: int = 5,
    rtol: float = 1e-5,
) -> None:
    """
    Verify that all indicators attached in strategy.prepare() are causal
    (i.e. no indicator value at bar i depends on bars i+1, i+2, ...).

    Method — split-half comparison:
      1. Run prepare() on the first half of the dataset → record indicator
         values at `check_bar` (a few bars before the midpoint).
      2. Run prepare() on the full dataset → record the same indicator values
         at the same `check_bar`.
      3. If any value differs beyond floating-point tolerance, the indicator
         used future data to compute past values → raise CausalityError.

    This catches entire classes of look-ahead that the bar-by-bar LookAheadError
    guard cannot catch, including:
      - scipy/statsmodels bidirectional filters (Savitzky-Golay, Butterworth, etc.)
      - Global normalisation (min-max scaling over the full series)
      - Incorrectly bounded rolling operations
      - Any NumPy/pandas operation that touches future rows in prepare()

    Args:
        df            : Full OHLCV DataFrame (same one used for backtesting)
        strategy      : Strategy instance whose prepare() will be tested
        n_check_bars  : Number of bars before midpoint to compare (default 5)
        rtol          : Relative tolerance for float comparison (default 1e-5)

    Raises:
        CausalityError : If any attached indicator fails the causality check.
        ValueError     : If the dataset is too short to run the audit.
    """
    N = len(df)
    if N < 100:
        raise ValueError(
            f"Dataset too short ({N} bars) to run causality audit. Need ≥ 100 bars."
        )

    mid     = N // 2
    df_half = df.iloc[:mid].copy()

    # --- Run prepare() on half dataset ---
    feed_half = DataFeed(df_half, symbol=symbol, timeframe=timeframe)
    strategy.prepare(feed_half)
    names_half = list(feed_half._custom.keys())

    if not names_half:
        return   # no custom indicators attached — nothing to check

    # --- Run prepare() on full dataset ---
    # Fresh strategy instance so prepare() state doesn't carry over.
    fresh_strategy = strategy.__class__(strategy.params)
    feed_full = DataFeed(df, symbol=symbol, timeframe=timeframe)
    fresh_strategy.prepare(feed_full)

    # --- Find best check bars: scan backwards from mid-1 ---
    # We want bars as close to the midpoint as possible — that's where the
    # influence of future data is strongest. We scan n_check_bars bars starting
    # from mid-1 and compare all of them, using any bar with valid values in
    # both half and full datasets.
    violations: list[str] = []

    for offset in range(1, min(n_check_bars + 1, mid)):
        check_bar = mid - offset

        for name in names_half:
            if name not in feed_full._custom:
                continue
            raw_half = feed_half._custom[name]._data
            raw_full = feed_full._custom[name]._data
            if check_bar >= len(raw_half) or check_bar >= len(raw_full):
                continue

            val_half = float(raw_half[check_bar])
            val_full = float(raw_full[check_bar])

            # NaN == NaN is fine; NaN vs number is a violation
            if np.isnan(val_half) and np.isnan(val_full):
                continue
            if np.isnan(val_half) or np.isnan(val_full):
                violation_key = f"{name}@{check_bar}"
                if violation_key not in {v.split("'")[1] + "@" + v.split("bar ")[1].split(":")[0]
                                         for v in violations}:
                    violations.append(
                        f"  '{name}' at bar {check_bar}: "
                        f"half-data={val_half}, full-data={val_full} — one is NaN"
                    )
                continue

            if not np.isclose(val_half, val_full, rtol=rtol, atol=0):
                pct_diff = abs(val_full - val_half) / (abs(val_half) + 1e-12) * 100
                # Only report the first (closest-to-boundary) violation per indicator
                already_reported = any(f"'{name}'" in v for v in violations)
                if not already_reported:
                    violations.append(
                        f"  '{name}' at bar {check_bar}: "
                        f"half-data={val_half:.8g}, full-data={val_full:.8g} "
                        f"({pct_diff:.4f}% diff) — indicator uses future data in prepare()"
                    )

    if violations:
        raise CausalityError(
            f"Causality audit FAILED for strategy '{strategy.NAME}'.\n"
            f"The following indicators change value at bar {check_bar} "
            f"when future bars are added to the dataset.\n"
            f"This means they were computed using future data in prepare() "
            f"— a form of look-ahead bias that the bar-loop guard cannot catch.\n\n"
            + "\n".join(violations) + "\n\n"
            "Fix: ensure all indicator computations in prepare() only use causal "
            "operations (causal EMA, rolling windows, etc.). See engine/indicators.py "
            "for a library of guaranteed-causal indicators."
        )


def run_backtest(
    df:              pd.DataFrame,
    strategy:        BaseStrategy,
    broker_config:   BrokerConfig,
    starting_equity: float = 5_000.0,
    warmup_bars:     int   = 200,
    label:           str   = "backtest",
    symbol:          str   = "UNKNOWN",
    timeframe:       str   = "M1",
    audit_causality: bool  = True,
    verbose:         bool  = True,
) -> BacktestResult:
    """
    Run a full backtest for one date range.

    Args:
        df              : OHLCV DataFrame with DatetimeIndex (UTC)
        strategy        : BaseStrategy instance (already constructed with params)
        broker_config   : BrokerConfig (commission, sizing, max_concurrent, etc.)
        starting_equity : initial portfolio value ($)
        warmup_bars     : bars to skip before strategy starts trading
        label           : display name for this run (e.g. 'IS 2020-2024')
        symbol / timeframe : metadata only
        verbose         : emit INFO-level log messages via the "engine.backtest"
                          logger. Configure with logging.getLogger("engine.backtest").

    Returns:
        BacktestResult with all trades and equity curve
    """
    if verbose:
        _log.info("\n[%s] Preparing strategy ...", label)

    # Validate data
    df = _validate_df(df)
    N  = len(df)

    if N < warmup_bars + 10:
        raise ValueError(
            f"Not enough bars ({N}) for warmup ({warmup_bars}). "
            "Provide more data or reduce warmup_bars."
        )

    # Build feed and broker
    feed   = DataFeed(df, symbol=symbol, timeframe=timeframe)
    broker = Broker(broker_config, starting_equity)

    # Strategy prepare: compute causal indicators over full array
    # (feed._cursor is -1 here — no bar access yet, only _data access)
    strategy.prepare(feed)

    # Causality audit: verify no indicator in prepare() used future data.
    # Skip with audit_causality=False in walk-forward inner loops (already
    # audited once on the full dataset before the WFA run starts).
    if audit_causality:
        if verbose:
            _log.info("[%s] Auditing indicator causality ...", label)
        audit_indicator_causality(df, strategy, symbol=symbol, timeframe=timeframe)
        if verbose:
            _log.info("[%s] Causality audit passed ✓", label)

    if verbose:
        _log.info("[%s] Running %s bars | %s → %s",
                  label, f"{N:,}", df.index[0].date(), df.index[-1].date())

    high   = df["high"].to_numpy()
    low    = df["low"].to_numpy()
    open_  = df["open"].to_numpy()
    close  = df["close"].to_numpy()
    times  = df.index   # DatetimeIndex — passed to broker for spread_schedule

    # Pre-compute a rolling ATR (EMA of true range, period=14) for the engine.
    # Used by broker when slippage_atr_mult > 0. Computed here so the broker
    # stays strategy-agnostic and doesn't need indicator knowledge.
    # Based on Wilder's smoothing: ATR(i) = alpha*TR(i) + (1-alpha)*ATR(i-1)
    _atr_period = 14
    _alpha      = 1.0 / _atr_period
    _tr         = np.maximum(high - low,
                  np.maximum(np.abs(high - np.roll(close, 1)),
                             np.abs(low  - np.roll(close, 1))))
    _tr[0] = high[0] - low[0]
    _rolling_atr = np.zeros(N)
    _rolling_atr[0] = _tr[0]
    for _j in range(1, N):
        _rolling_atr[_j] = _alpha * _tr[_j] + (1 - _alpha) * _rolling_atr[_j - 1]

    equity_curve = np.full(N, np.nan)
    equity_curve[warmup_bars] = starting_equity

    prev_n_open = 0

    for i in range(N):
        # Step 1: advance feed cursor
        feed._advance(i)

        # Step 2: strategy acts (warmup period: skip signal generation)
        if i >= warmup_bars:
            try:
                strategy.on_bar(feed, broker)
            except Exception as e:
                _log.warning("  strategy.on_bar error at bar %d: %s", i, e)

        # Step 3: broker processes this bar (fills, SL/TP, gap fill, trailing)
        # bar_time enables spread_schedule lookup; atr enables slippage_atr_mult
        broker.on_bar(i, open_[i], high[i], low[i], close[i],
                      bar_time=times[i], atr=float(_rolling_atr[i]))

        # Fire on_fill hook if a new position was filled
        if len(broker.open_positions) > prev_n_open:
            for pos in broker.open_positions[-1:]:
                try:
                    strategy.on_fill(pos, feed)
                except Exception:
                    pass

        # Fire on_close hook for any trades closed this bar
        for t in broker.closed_trades:
            if t.exit_bar == i:
                try:
                    strategy.on_close(t, feed)
                except Exception:
                    pass

        prev_n_open = len(broker.open_positions)

        # Step 4: record equity
        equity_curve[i] = broker.equity

    # Close any open positions at last bar (mark-to-market)
    if broker.open_positions:
        broker.close_all_at(N - 1, close[N - 1], bar_time=times[N - 1])
        equity_curve[N - 1] = broker.equity

    # Forward-fill equity curve gaps
    for i in range(1, N):
        if np.isnan(equity_curve[i]):
            equity_curve[i] = equity_curve[i - 1]

    all_trades = broker._history
    n_closed   = sum(1 for t in all_trades if t.status == OrderStatus.CLOSED
                     and t.exit_reason != "end_of_data")
    n_expired  = sum(1 for t in all_trades if t.status == OrderStatus.EXPIRED)

    if verbose:
        _log.info("[%s] Done — %d closed trades, %d expired orders, final equity: $%s",
                  label, n_closed, n_expired, f"{broker.equity:,.2f}")

    return BacktestResult(
        label=label,
        trades=all_trades,
        equity_curve=equity_curve,
        timestamps=pd.DatetimeIndex(df.index),
        params=strategy.params,
        strategy_name=strategy.NAME,
    )


def run_splits(
    df:              pd.DataFrame,
    strategy_class:  type,
    params:          dict,
    broker_config:   BrokerConfig,
    starting_equity: float = 5_000.0,
    warmup_bars:     int   = 200,
    splits:          Optional[dict] = None,
    symbol:          str   = "UNKNOWN",
    timeframe:       str   = "M1",
    audit_causality: bool  = True,
    verbose:         bool  = True,
) -> dict[str, BacktestResult]:
    """
    Run backtest across multiple date splits (IS, OOS, FWD, etc.).

    Args:
        splits : dict of label → (start_date, end_date) strings.
                 e.g. {"IS": ("2020-01-01", "2024-12-31"),
                        "OOS": ("2025-01-01", "2025-12-31")}
                 Use None as end_date for "up to last bar".

    Each split gets a fresh strategy instance (independent state).
    Equity resets to starting_equity for each split.
    """
    if splits is None:
        splits = {
            "IS  (2020–2024)": ("2020-01-01", "2024-12-31"),
            "OOS (2025)":      ("2025-01-01", "2025-12-31"),
            "FWD (2026–now)":  ("2026-01-01", None),
        }

    results: dict[str, BacktestResult] = {}

    for label, (start, end) in splits.items():
        df_slice = df.loc[start:end].copy() if end else df.loc[start:].copy()
        if len(df_slice) < warmup_bars + 10:
            if verbose:
                _log.info("\n[%s] Not enough bars (%d) — skipping.", label, len(df_slice))
            continue
        # Fresh strategy per split (no state leakage between splits)
        strat = strategy_class(params)
        results[label] = run_backtest(
            df_slice, strat, broker_config,
            starting_equity=starting_equity,
            warmup_bars=warmup_bars,
            label=label, symbol=symbol, timeframe=timeframe,
            audit_causality=audit_causality,
            verbose=verbose,
        )

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()

    required = {"open", "high", "low", "close"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.sort_index(inplace=True)
    df.dropna(subset=list(required), inplace=True)

    # Drop malformed bars
    mask = (
        (df["high"] >= df["low"]) &
        (df["open"] >= df["low"])  & (df["open"]  <= df["high"]) &
        (df["close"] >= df["low"]) & (df["close"] <= df["high"])
    )
    n_bad = (~mask).sum()
    if n_bad:
        df = df[mask]

    return df


# ── Multi-timeframe backtest ───────────────────────────────────────────────────

def run_backtest_mtf(
    mtf_feeds:       "MultiTimeframeFeed",   # noqa: F821
    strategy:        "MTFStrategy",           # noqa: F821
    broker_config:   BrokerConfig,
    starting_equity: float = 5_000.0,
    warmup_bars:     int   = 200,
    label:           str   = "backtest_mtf",
    verbose:         bool  = True,
) -> BacktestResult:
    """
    Run a multi-timeframe backtest.

    The primary feed drives the bar loop. Secondary feeds (H1, M15, etc.) have
    their cursors advanced to the last COMPLETED bar at each primary bar's
    timestamp — the current secondary bar (still in progress) is never visible.

    Args:
        mtf_feeds       : MultiTimeframeFeed built via engine.mtf.build_mtf_feeds()
        strategy        : MTFStrategy instance (must implement prepare_mtf / on_bar_mtf)
        broker_config   : BrokerConfig
        starting_equity : Starting equity
        warmup_bars     : Primary bars to skip before strategy produces signals
        label           : Display label
        verbose         : emit INFO-level log messages via the "engine.backtest" logger.

    Returns:
        BacktestResult on the primary timeframe's bar structure.

    Look-ahead guarantee (in addition to standard guards):
        Secondary feed cursors use strictly-before-T alignment, so no secondary
        bar that is currently in progress is ever readable.
    """

    primary     = mtf_feeds.primary
    primary_df  = pd.DataFrame({
        "open":   primary.open._data,
        "high":   primary.high._data,
        "low":    primary.low._data,
        "close":  primary.close._data,
    }, index=primary.index)

    N      = len(primary_df)
    broker = Broker(broker_config, starting_equity)

    if verbose:
        _log.info("\n[%s] Preparing MTF strategy ...", label)

    # prepare_mtf receives a plain dict of DataFeeds
    strategy.prepare_mtf(dict(mtf_feeds._feeds))

    if verbose:
        tfs = list(mtf_feeds.keys())
        _log.info("[%s] Running %s primary bars | TFs: %s", label, f"{N:,}", tfs)

    open_   = primary_df["open"].to_numpy()
    high    = primary_df["high"].to_numpy()
    low     = primary_df["low"].to_numpy()
    close   = primary_df["close"].to_numpy()

    equity_curve = np.full(N, np.nan)
    equity_curve[warmup_bars] = starting_equity
    prev_n_open  = 0

    for i in range(N):
        # Advance ALL feeds (primary + secondaries)
        mtf_feeds.advance(i)

        if i >= warmup_bars:
            try:
                strategy.on_bar_mtf(dict(mtf_feeds._feeds), broker)
            except Exception as e:
                _log.warning("  on_bar_mtf error at bar %d: %s", i, e)

        broker.on_bar(i, open_[i], high[i], low[i], close[i])

        if len(broker.open_positions) > prev_n_open:
            for pos in broker.open_positions[-1:]:
                try:
                    strategy.on_fill(pos, primary)
                except Exception:
                    pass

        for t in broker.closed_trades:
            if t.exit_bar == i:
                try:
                    strategy.on_close(t, primary)
                except Exception:
                    pass

        prev_n_open      = len(broker.open_positions)
        equity_curve[i]  = broker.equity

    if broker.open_positions:
        broker.close_all_at(N - 1, close[N - 1])
        equity_curve[N - 1] = broker.equity

    for i in range(1, N):
        if np.isnan(equity_curve[i]):
            equity_curve[i] = equity_curve[i - 1]

    all_trades = broker._history
    n_closed   = sum(1 for t in all_trades if t.status == OrderStatus.CLOSED
                     and t.exit_reason != "end_of_data")

    if verbose:
        _log.info("[%s] Done — %d closed trades, final equity: $%s",
                  label, n_closed, f"{broker.equity:,.2f}")

    return BacktestResult(
        label         = label,
        trades        = all_trades,
        equity_curve  = equity_curve,
        timestamps    = pd.DatetimeIndex(primary.index),
        params        = strategy.params,
        strategy_name = strategy.NAME,
    )
