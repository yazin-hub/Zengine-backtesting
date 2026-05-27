"""
engine/pairs.py — Pairs Trading Backtest Engine
================================================

Runs a pairs trading backtest on two correlated assets simultaneously.
Both assets share a single equity pool, and the strategy receives both
DataFeeds and both Brokers at every aligned bar.

Architecture:
  - Two DataFeed instances (asset A and asset B)
  - Two Broker instances — each runs independently, combined equity tracked
  - Bars are aligned by timestamp (inner join) — only matching bars are traded
  - Strategy implements PairsStrategy.on_bar(feed_a, broker_a, feed_b, broker_b)
  - Exit requests are handled by the runner (clean close at current bar price)

Look-ahead guarantee:
  - Same DataFeed cursor guard as single-asset engine (LookAheadError)
  - Both feeds advance in lockstep at each aligned bar
  - Strategy cannot read ahead on either feed

Usage:
    from engine.pairs import run_backtest_pairs, PairsStrategy
    from strategies.btc_eth_spread import BTCETHSpreadStrategy

    result = run_backtest_pairs(
        df_a=df_btc,
        df_b=df_eth,
        strategy=BTCETHSpreadStrategy(),
        broker_config_a=BrokerConfig(commission_pct=0.0005),
        broker_config_b=BrokerConfig(commission_pct=0.0005),
        starting_equity=10_000.0,
        symbol_a="BTCUSD",
        symbol_b="ETHUSD",
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _validate_df
from .broker import Broker, BrokerConfig, OrderSide, OrderStatus
from .data import DataFeed
from .risk import DailyCircuitBreaker

_log = logging.getLogger(__name__)


# ── PairsStrategy base class ──────────────────────────────────────────────────

class PairsStrategy(ABC):
    """
    Abstract base class for pairs trading strategies.

    Unlike BaseStrategy (single asset), PairsStrategy receives BOTH DataFeeds
    and BOTH Brokers at every bar, enabling:
      - Spread and z-score calculation from both close prices
      - Simultaneous order placement on both instruments
      - Paired position state management

    The strategy communicates exit intent by setting self._exit_requested = True
    inside on_bar(). The pairs runner handles the actual close_all_at() calls
    cleanly at the current bar price before broker.on_bar() runs.

    Subclass and implement:
      - prepare(feed_a, feed_b)                              : pre-compute indicators
      - on_bar(feed_a, broker_a, feed_b, broker_b)           : signal + order logic
    """

    NAME: str = "PairsStrategy"
    PARAMS: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        defaults: dict[str, Any] = {}
        for k, v in self.PARAMS.items():
            defaults[k] = v["default"] if isinstance(v, dict) else v
        resolved: dict[str, Any] = {}
        for k, v in (params or {}).items():
            resolved[k] = v["default"] if isinstance(v, dict) else v
        self.params = {**defaults, **resolved}

        # Set by strategy in on_bar() when it wants both legs closed this bar.
        # The runner checks this flag after on_bar() and calls close_all_at().
        self._exit_requested: bool = False
        # Exit reason label passed to close_all_at(exit_reason=...).
        # Subclasses set this before setting _exit_requested = True.
        self._exit_reason: str = "zscore_exit"

    @abstractmethod
    def prepare(self, feed_a: DataFeed, feed_b: DataFeed) -> None:
        """Pre-compute indicators on both feeds before the bar loop."""

    @abstractmethod
    def on_bar(
        self,
        feed_a: DataFeed, broker_a: Broker,
        feed_b: DataFeed, broker_b: Broker,
    ) -> None:
        """
        Called at each aligned bar with both feeds and brokers.

        To exit: set self._exit_requested = True.
        The runner will close all positions at the current bar's close price.
        """

    def __repr__(self) -> str:
        return f"{self.NAME}({self.params})"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PairsResult:
    """
    Results from a pairs trading backtest.

    Attributes:
        trades_a        : All orders placed on asset A (BTC)
        trades_b        : All orders placed on asset B (ETH)
        equity_curve    : Combined equity at each aligned bar
        spread_history  : Raw spread value at each bar (BTC - ratio*ETH)
        zscore_history  : Z-score at each bar (NaN during warmup)
        timestamps      : DatetimeIndex of aligned bars
    """
    label:           str
    trades_a:        list
    trades_b:        list
    equity_curve:    np.ndarray
    timestamps:      pd.DatetimeIndex
    spread_history:  np.ndarray
    zscore_history:  np.ndarray
    params:          dict
    strategy_name:   str
    symbol_a:        str
    symbol_b:        str
    starting_equity: float

    @property
    def closed_trades_a(self) -> list:
        """Completed pairs trades — excludes only the final end_of_data cleanup."""
        return [t for t in self.trades_a
                if t.status == OrderStatus.CLOSED and t.exit_reason != "end_of_data"]

    @property
    def closed_trades_b(self) -> list:
        """Completed pairs trades — excludes only the final end_of_data cleanup."""
        return [t for t in self.trades_b
                if t.status == OrderStatus.CLOSED and t.exit_reason != "end_of_data"]

    @property
    def n_pairs_trades(self) -> int:
        """Number of completed pairs trades (leg A count = pairs count)."""
        return len(self.closed_trades_a)

    @property
    def final_equity(self) -> float:
        valid = self.equity_curve[~np.isnan(self.equity_curve)]
        return float(valid[-1]) if len(valid) else self.starting_equity

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.starting_equity - 1) * 100

    @property
    def total_pnl(self) -> float:
        pnl_a = sum(t.pnl_net for t in self.closed_trades_a)
        pnl_b = sum(t.pnl_net for t in self.closed_trades_b)
        return pnl_a + pnl_b

    @property
    def win_rate_pct(self) -> float:
        """Win rate based on combined P&L per pairs trade."""
        if not self.closed_trades_a:
            return 0.0
        # Pair trade is a win if combined PnL (leg A + matching leg B) > 0
        wins = 0
        for i, ta in enumerate(self.closed_trades_a):
            tb = self.closed_trades_b[i] if i < len(self.closed_trades_b) else None
            combined = ta.pnl_net + (tb.pnl_net if tb else 0.0)
            if combined > 0:
                wins += 1
        return wins / len(self.closed_trades_a) * 100

    @property
    def max_drawdown_pct(self) -> float:
        eq = self.equity_curve[~np.isnan(self.equity_curve)]
        if len(eq) < 2:
            return 0.0
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / running_max * 100
        return float(np.min(dd))

    def summary(self) -> str:
        pnl_a = sum(t.pnl_net for t in self.closed_trades_a)
        pnl_b = sum(t.pnl_net for t in self.closed_trades_b)
        lines = [
            f"\n{'=' * 58}",
            f"Pairs Backtest — {self.symbol_a} / {self.symbol_b}",
            f"Strategy : {self.strategy_name}",
            f"{'=' * 58}",
            f"  Pairs trades    : {self.n_pairs_trades}",
            f"  Win rate        : {self.win_rate_pct:.1f}%",
            f"  PnL {self.symbol_a:<10}: ${pnl_a:>10,.2f}",
            f"  PnL {self.symbol_b:<10}: ${pnl_b:>10,.2f}",
            f"  Total PnL       : ${self.total_pnl:>10,.2f}",
            f"  Total return    : {self.total_return_pct:>8.2f}%",
            f"  Max drawdown    : {self.max_drawdown_pct:>8.2f}%",
            f"  Final equity    : ${self.final_equity:>10,.2f}",
            f"{'=' * 58}\n",
        ]
        return "\n".join(lines)


# ── Main pairs backtest function ──────────────────────────────────────────────

def run_backtest_pairs(
    df_a:            pd.DataFrame,
    df_b:            pd.DataFrame,
    strategy:        PairsStrategy,
    broker_config_a: BrokerConfig,
    broker_config_b: BrokerConfig,
    starting_equity: float                        = 10_000.0,
    warmup_bars:     int                          = 100,
    label:           str                          = "pairs",
    symbol_a:        str                          = "BTCUSD",
    symbol_b:        str                          = "ETHUSD",
    verbose:         bool                         = True,
    circuit_breaker: DailyCircuitBreaker | None   = None,
) -> PairsResult:
    """
    Run a pairs trading backtest on two correlated assets.

    Bars are aligned by timestamp (inner join) — only timestamps present in
    BOTH DataFrames are traded. This prevents any synthetic look-ahead from
    misaligned series.

    Exit mechanism:
        Strategy sets self._exit_requested = True in on_bar().
        Runner calls close_all_at(i, close_price) on both brokers BEFORE
        broker.on_bar() — so the close executes at the current bar's price.

    Args:
        df_a / df_b      : OHLCV DataFrames (DateTime index, UTC or naive)
        strategy         : PairsStrategy instance
        broker_config_a/b: BrokerConfig for each leg (commission, slippage)
        starting_equity  : Total equity shared across both legs
        warmup_bars      : Bars to skip before strategy signals (z-score warmup)
        label            : Display label
        symbol_a/b       : Symbol names (metadata only — not used for data fetching)
        verbose          : Emit INFO-level log messages
        circuit_breaker  : Optional DailyCircuitBreaker — evaluated before
                           strategy.on_bar() each bar. Trips when realized +
                           floating daily loss hits the configured limit.
                           None (default) = no daily loss circuit breaker.

    Returns:
        PairsResult — use .summary(), .equity_curve, .spread_history, .zscore_history
    """
    # Validate both DataFrames
    df_a = _validate_df(df_a)
    df_b = _validate_df(df_b)

    # Inner join on timestamp — only trade bars where BOTH assets have data
    df_a, df_b = df_a.align(df_b, join="inner", axis=0)
    N = len(df_a)

    if N < warmup_bars + 10:
        raise ValueError(
            f"Only {N} aligned bars after inner join on timestamps. "
            f"Need at least warmup_bars ({warmup_bars}) + 10. "
            "Check that both DataFrames cover overlapping date ranges."
        )

    if verbose:
        _log.info(
            "\n[%s] %s / %s  |  %s aligned 15m bars  |  %s → %s",
            label, symbol_a, symbol_b, f"{N:,}",
            df_a.index[0].date(), df_a.index[-1].date(),
        )

    # Build DataFeeds
    feed_a = DataFeed(df_a, symbol=symbol_a, timeframe="M15")
    feed_b = DataFeed(df_b, symbol=symbol_b, timeframe="M15")

    # Two brokers — each starts with half the equity (one per leg).
    # Combined equity = broker_a.equity + broker_b.equity at every bar.
    broker_a = Broker(broker_config_a, starting_equity / 2)
    broker_b = Broker(broker_config_b, starting_equity / 2)

    # Strategy prepare: pre-compute spread, z-score, indicators causally
    if verbose:
        _log.info("[%s] Preparing strategy ...", label)
    strategy.prepare(feed_a, feed_b)

    # Pre-allocate output arrays
    equity_curve   = np.full(N, np.nan)
    spread_history = np.full(N, np.nan)
    zscore_history = np.full(N, np.nan)

    equity_curve[warmup_bars] = starting_equity

    # Pre-extract numpy arrays for speed
    close_a = df_a["close"].to_numpy()
    close_b = df_b["close"].to_numpy()
    open_a  = df_a["open"].to_numpy()
    high_a  = df_a["high"].to_numpy()
    low_a   = df_a["low"].to_numpy()
    open_b  = df_b["open"].to_numpy()
    high_b  = df_b["high"].to_numpy()
    low_b   = df_b["low"].to_numpy()
    times   = df_a.index

    if verbose:
        _log.info("[%s] Running %s bars ...", label, f"{N:,}")

    for i in range(N):
        # Step 1: advance both feed cursors
        feed_a._advance(i)
        feed_b._advance(i)

        # Step 2: record spread and z-score (attached by strategy.prepare())
        if "spread" in feed_a._custom:
            spread_history[i] = float(feed_a._custom["spread"]._data[i])
        if "zscore" in feed_a._custom:
            zscore_history[i] = float(feed_a._custom["zscore"]._data[i])

        # Step 3: circuit breaker + strategy on_bar (after warmup)
        if i >= warmup_bars:
            strategy._exit_requested = False

            # ── Engine-level daily loss circuit breaker ───────────────────────
            # Checked BEFORE strategy.on_bar() so any strategy benefits.
            # Computes realized (broker equity) + floating (open positions) loss
            # since the start of the current UTC calendar day.
            cb_block_entry = False
            if circuit_breaker is not None and circuit_breaker.enabled:
                today_str       = str(times[i].date())
                combined_equity = broker_a.equity + broker_b.equity

                # Day transition — reset baseline equity
                if today_str != circuit_breaker._current_day:
                    circuit_breaker.reset_day(today_str, combined_equity)

                # Floating P&L from any open positions on both legs.
                # broker.equity tracks only CLOSED trade P&L; open positions
                # need manual calculation against current bar close price.
                floating = 0.0
                for pos in broker_a.open_positions:
                    if pos.side == OrderSide.LONG:
                        floating += (close_a[i] - pos.fill_price) * pos.size
                    else:
                        floating += (pos.fill_price - close_a[i]) * pos.size
                for pos in broker_b.open_positions:
                    if pos.side == OrderSide.LONG:
                        floating += (close_b[i] - pos.fill_price) * pos.size
                    else:
                        floating += (pos.fill_price - close_b[i]) * pos.size

                tripped = circuit_breaker.check(i, today_str, combined_equity, floating)
                if tripped:
                    # Force-close any open position via the standard exit path
                    strategy._exit_reason    = "daily_circuit_breaker"
                    strategy._exit_requested = True
                    # Reset position state so strategy doesn't try to exit again
                    if hasattr(strategy, "_position"):
                        strategy._position  = 0
                    if hasattr(strategy, "_entry_bar"):
                        strategy._entry_bar = -1

                cb_block_entry = circuit_breaker.is_blocked()

            # Skip strategy.on_bar() entirely if circuit is tripped today.
            # Exits already handled above; entries must not be placed.
            if not cb_block_entry:
                try:
                    strategy.on_bar(feed_a, broker_a, feed_b, broker_b)
                except Exception as e:
                    _log.warning("  strategy.on_bar error at bar %d: %s", i, e)

            # Step 4: honour exit request — close both legs at this bar's price
            # Done BEFORE broker.on_bar() so fills are clean and immediate.
            # exit_reason is passed through to Order so closed_trades filtering works.
            if strategy._exit_requested:
                exit_rsn = getattr(strategy, "_exit_reason", "zscore_exit")
                if broker_a.open_positions:
                    broker_a.close_all_at(i, close_a[i], bar_time=times[i],
                                          exit_reason=exit_rsn)
                if broker_b.open_positions:
                    broker_b.close_all_at(i, close_b[i], bar_time=times[i],
                                          exit_reason=exit_rsn)
                if broker_a._pending is not None:
                    broker_a._pending = None
                if broker_b._pending is not None:
                    broker_b._pending = None
                strategy._exit_requested = False
                if hasattr(strategy, "_exit_reason"):
                    strategy._exit_reason = "zscore_exit"  # reset to default

        # Step 5: broker processes bar (fills pending, manages SL/TP)
        broker_a.on_bar(i, open_a[i], high_a[i], low_a[i], close_a[i], bar_time=times[i])
        broker_b.on_bar(i, open_b[i], high_b[i], low_b[i], close_b[i], bar_time=times[i])

        # Step 6: record combined equity
        equity_curve[i] = broker_a.equity + broker_b.equity

    # Close any remaining open positions at last bar
    if broker_a.open_positions:
        broker_a.close_all_at(N - 1, close_a[N - 1], bar_time=times[N - 1])
    if broker_b.open_positions:
        broker_b.close_all_at(N - 1, close_b[N - 1], bar_time=times[N - 1])
    equity_curve[N - 1] = broker_a.equity + broker_b.equity

    # Forward-fill equity gaps (bars with no change)
    for i in range(1, N):
        if np.isnan(equity_curve[i]):
            equity_curve[i] = equity_curve[i - 1]

    n_pairs = sum(1 for t in broker_a._history
                  if t.status == OrderStatus.CLOSED and t.exit_reason != "end_of_data")

    if verbose:
        _log.info(
            "[%s] Done — %d pairs trades | final equity: $%s",
            label, n_pairs,
            f"{broker_a.equity + broker_b.equity:,.2f}",
        )

    result = PairsResult(
        label           = label,
        trades_a        = broker_a._history,
        trades_b        = broker_b._history,
        equity_curve    = equity_curve,
        timestamps      = pd.DatetimeIndex(df_a.index),
        spread_history  = spread_history,
        zscore_history  = zscore_history,
        params          = strategy.params,
        strategy_name   = strategy.NAME,
        symbol_a        = symbol_a,
        symbol_b        = symbol_b,
        starting_equity = starting_equity,
    )

    if verbose:
        _log.info("%s", result.summary())

    return result
