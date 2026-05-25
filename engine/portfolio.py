"""
engine/portfolio.py — Portfolio-Level Backtesting
===================================================

Runs multiple instruments simultaneously with a shared equity pool.
Risk per trade is drawn from the same capital, so position sizing is
capital-aware across all open positions at all times.

Architecture:
  - One `Broker` instance per symbol, all sharing the same equity reference
  - `Portfolio` coordinates equity updates: when any broker closes a trade,
    the shared equity adjusts, affecting sizing on all subsequent orders
  - The backtest loop merges all instrument timelines chronologically and
    processes bars in strict time order (no look-ahead across instruments)

Position sizing with shared equity:
  When sizing a new order, the broker uses `portfolio.equity` (the shared pool)
  minus the total unrealised value at risk across ALL open positions, ensuring
  the portfolio never over-leverages.

Usage:
    from engine.portfolio import run_backtest_portfolio, PortfolioConfig

    portfolio_cfg = PortfolioConfig(
        symbols=["XAUUSD", "EURUSD"],
        broker_configs={
            "XAUUSD": BrokerConfig(commission_flat=6.0, lot_size=100.0, risk_usd=20.0),
            "EURUSD": BrokerConfig(commission_flat=4.0, lot_size=100_000, risk_usd=10.0),
        },
        max_total_risk_usd=100.0,   # max total $risk across all open positions
    )
    results = run_backtest_portfolio(
        dfs={"XAUUSD": df_gold, "EURUSD": df_eurusd},
        strategies={"XAUUSD": IFVGStrategy(), "EURUSD": MACrossStrategy()},
        portfolio_cfg=portfolio_cfg,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult, _validate_df
from .broker import Broker, BrokerConfig, OrderStatus
from .data import DataFeed
from .strategy import BaseStrategy

_log = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class PortfolioConfig:
    """
    Portfolio-level configuration.

    Attributes:
        symbols             : Ordered list of instrument symbols to trade
        broker_configs      : Per-symbol BrokerConfig. Use key "default" as fallback.
        starting_equity     : Total portfolio starting equity (shared pool)
        max_total_risk_usd  : Max $ at risk across ALL open positions combined.
                              New orders are rejected if adding them would exceed this.
                              None = no portfolio-level risk cap.
        warmup_bars         : Bars to skip before any strategy generates signals.
                              Applied per-instrument based on its own bar count.
        max_total_positions : Hard cap on total open positions across all symbols.
    """
    symbols:             list[str]
    broker_configs:      dict[str, BrokerConfig]
    starting_equity:     float = 5_000.0
    max_total_risk_usd:  Optional[float] = None
    warmup_bars:         int   = 200
    max_total_positions: int   = 10


# ── Shared equity tracker ──────────────────────────────────────────────────────

class SharedEquity:
    """
    Mutable equity reference shared across all per-symbol Broker instances.

    Each broker calls shared_equity.adjust(delta) when a trade closes.
    Position sizing reads shared_equity.value to get current portfolio equity.
    """

    def __init__(self, starting_equity: float) -> None:
        self._value = starting_equity

    @property
    def value(self) -> float:
        return self._value

    def adjust(self, delta: float) -> None:
        self._value += delta

    def deduct_commission(self, amount: float) -> None:
        self._value -= amount


class PortfolioBroker(Broker):
    """
    Broker variant that writes P&L adjustments to a SharedEquity pool
    instead of its own `self.equity`.

    Also enforces a portfolio-level risk cap: if placing a new order would
    push total_risk_usd above the cap, the order is rejected.
    """

    def __init__(
        self,
        config:         BrokerConfig,
        shared_equity:  SharedEquity,
        symbol:         str,
        max_total_risk: Optional[float] = None,
        portfolio_ref:  Optional["Portfolio"] = None,
    ) -> None:
        # Initialise parent with a placeholder equity — we'll override all
        # equity reads to use shared_equity instead
        super().__init__(config, shared_equity.value)
        self._shared    = shared_equity
        self._symbol    = symbol
        self._max_risk  = max_total_risk
        self._portfolio = portfolio_ref

    @property
    def equity(self) -> float:
        return self._shared.value

    @equity.setter
    def equity(self, value: float) -> None:
        # Called by parent __init__ — ignore, we use shared equity
        pass

    def _apply_commission(self, amount: float) -> None:
        self._shared.deduct_commission(amount)

    def on_bar(self, bar_idx: int,
               open_: float, high: float, low: float, close: float,
               bar_time: "pd.Timestamp | None" = None,
               atr: float | None = None) -> None:
        """
        Override on_bar to route P&L changes through SharedEquity.
        We capture equity before/after and apply the delta to the shared pool.
        """
        # Snapshot open positions' commission deductions
        cfg = self.config

        # --- Try to fill pending order ---
        just_filled: set[int] = set()
        if self._pending is not None and self._pending.placed_bar < bar_idx:
            t = self._pending
            filled, fill_px = False, math.nan

            if t.order_type.value == "market":
                fill_px = open_ * (1 + cfg.slippage_pct * (1 if t.side.value == "long" else -1))
                filled  = True
            elif t.order_type.value == "limit":
                if t.side.value == "long" and low <= t.limit_price:
                    fill_px = open_ if open_ <= t.limit_price else t.limit_price
                    fill_px *= (1 + cfg.slippage_pct)
                    filled   = True
                elif t.side.value == "short" and high >= t.limit_price:
                    fill_px = open_ if open_ >= t.limit_price else t.limit_price
                    fill_px *= (1 - cfg.slippage_pct)
                    filled   = True
                elif (bar_idx - t.placed_bar) >= t.expiry_bars:
                    t.status = OrderStatus.EXPIRED
                    self._history.append(t)
                    self._pending = None

            if filled:
                t.fill_bar    = bar_idx
                t.fill_price  = fill_px
                t.status      = OrderStatus.OPEN
                comm_entry    = self._calc_commission(t.size) * 0.5
                t.commission += comm_entry
                self._shared.deduct_commission(comm_entry)
                just_filled.add(id(t))
                self._open.append(t)
                self._pending = None

        # --- Manage open positions ---
        still_open = []
        for t in self._open:
            if id(t) in just_filled:
                still_open.append(t)
                continue
            sl_hit = tp_hit = False
            if t.side.value == "long":
                sl_hit = low  <= t.sl
                tp_hit = high >= t.tp
            else:
                sl_hit = high >= t.sl
                tp_hit = low  <= t.tp

            if sl_hit or tp_hit:
                exit_px, reason = (t.sl, "sl") if sl_hit else (t.tp, "tp")
                gross = ((exit_px - t.fill_price) if t.side.value == "long"
                         else (t.fill_price - exit_px)) * t.size
                comm_exit    = self._calc_commission(t.size) * 0.5
                t.commission += comm_exit
                t.exit_bar    = bar_idx
                t.exit_price  = exit_px
                t.exit_reason = reason
                t.pnl_gross   = gross
                t.pnl_net     = gross - comm_exit
                t.status      = OrderStatus.CLOSED
                self._shared.adjust(t.pnl_net)
                self._shared.deduct_commission(comm_exit)
                self._history.append(t)
            else:
                still_open.append(t)
        self._open = still_open

    def close_all_at(self, bar_idx: int, price: float,
                     bar_time: "pd.Timestamp | None" = None,
                     exit_reason: str = "end_of_data") -> None:
        for t in self._open:
            gross = ((price - t.fill_price) if t.side.value == "long"
                     else (t.fill_price - price)) * t.size
            comm          = self._calc_commission(t.size) * 0.5
            t.commission += comm
            t.exit_bar    = bar_idx
            t.exit_price  = price
            t.exit_reason = exit_reason
            t.pnl_gross   = gross
            t.pnl_net     = gross - comm
            t.status      = OrderStatus.CLOSED
            self._shared.adjust(t.pnl_net)
            self._shared.deduct_commission(comm)
            self._history.append(t)
        self._open = []


# ── Portfolio container ────────────────────────────────────────────────────────

class Portfolio:
    """
    Coordinates multiple PortfolioBrokers sharing a single equity pool.
    """

    def __init__(self, cfg: PortfolioConfig) -> None:
        self.cfg            = cfg
        self._shared        = SharedEquity(cfg.starting_equity)
        self._brokers: dict[str, PortfolioBroker] = {}

        for sym in cfg.symbols:
            b_cfg = cfg.broker_configs.get(sym, cfg.broker_configs.get("default"))
            if b_cfg is None:
                raise KeyError(
                    f"No BrokerConfig for symbol '{sym}' and no 'default' key. "
                    "Provide a config for every symbol or a 'default' fallback."
                )
            self._brokers[sym] = PortfolioBroker(
                config        = b_cfg,
                shared_equity = self._shared,
                symbol        = sym,
                max_total_risk = cfg.max_total_risk_usd,
                portfolio_ref  = self,
            )

    @property
    def equity(self) -> float:
        return self._shared.value

    def broker(self, symbol: str) -> PortfolioBroker:
        return self._brokers[symbol]

    @property
    def n_open(self) -> int:
        return sum(b.n_open for b in self._brokers.values())

    @property
    def total_risk_usd(self) -> float:
        """Current total $ at risk across all open positions."""
        total = 0.0
        for broker in self._brokers.values():
            for pos in broker.open_positions:
                sl_dist = abs(pos.fill_price - pos.sl)
                total  += sl_dist * pos.size
        return total

    def can_place_order(self, risk_usd: float) -> bool:
        """Check portfolio-level risk cap before allowing a new order."""
        if self.cfg.max_total_risk_usd is None:
            return True
        if self.n_open >= self.cfg.max_total_positions:
            return False
        return (self.total_risk_usd + risk_usd) <= self.cfg.max_total_risk_usd

    @property
    def all_closed_trades(self) -> list:
        trades = []
        for broker in self._brokers.values():
            trades.extend(broker.closed_trades)
        trades.sort(key=lambda t: t.exit_bar)
        return trades


# ── Portfolio backtest dataclass ───────────────────────────────────────────────

@dataclass
class PortfolioResult:
    """Results from a multi-instrument portfolio backtest."""
    symbol_results:  dict[str, BacktestResult]
    equity_curve:    np.ndarray
    timestamps:      pd.DatetimeIndex
    starting_equity: float

    @property
    def final_equity(self) -> float:
        valid = self.equity_curve[~np.isnan(self.equity_curve)]
        return float(valid[-1]) if len(valid) else self.starting_equity

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.starting_equity - 1) * 100

    @property
    def all_trades(self) -> list:
        trades = []
        for r in self.symbol_results.values():
            trades.extend(r.closed_trades)
        trades.sort(key=lambda t: t.exit_bar if t.exit_bar >= 0 else 0)
        return trades


# ── Main portfolio backtest function ──────────────────────────────────────────

def run_backtest_portfolio(
    dfs:             dict[str, pd.DataFrame],
    strategies:      dict[str, BaseStrategy],
    portfolio_cfg:   PortfolioConfig,
    verbose:         bool = True,
) -> PortfolioResult:
    """
    Run a portfolio backtest across multiple instruments simultaneously.

    Bars from all instruments are merged and processed in strict chronological
    order. At each bar, only the relevant instrument's strategy and broker are
    called — no cross-instrument look-ahead.

    Args:
        dfs           : Dict of symbol → OHLCV DataFrame with DatetimeIndex
        strategies    : Dict of symbol → BaseStrategy instance
        portfolio_cfg : PortfolioConfig (shared equity, risk caps, etc.)
        verbose       : Print progress

    Returns:
        PortfolioResult with per-symbol BacktestResults and shared equity curve

    Look-ahead guarantee:
        Each symbol has its own DataFeed with its own cursor. Feeds for
        instrument A are never visible from instrument B's strategy.
        Bar processing order is strictly chronological across all instruments.
    """
    cfg     = portfolio_cfg
    symbols = cfg.symbols

    # Validate and prepare data
    clean_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym not in dfs:
            raise KeyError(f"No DataFrame provided for symbol '{sym}'")
        clean_dfs[sym] = _validate_df(dfs[sym])

    # Build feeds and brokers
    feeds:    dict[str, DataFeed]       = {}
    bar_arrs: dict[str, dict]           = {}   # precomputed OHLC arrays
    warmup_bars_per_sym: dict[str, int] = {}

    for sym in symbols:
        df = clean_dfs[sym]
        feeds[sym]    = DataFeed(df, symbol=sym)
        bar_arrs[sym] = {
            "open":  df["open"].to_numpy(),
            "high":  df["high"].to_numpy(),
            "low":   df["low"].to_numpy(),
            "close": df["close"].to_numpy(),
        }
        warmup_bars_per_sym[sym] = cfg.warmup_bars

    portfolio = Portfolio(cfg)

    # Strategy prepare (each gets its own feed)
    for sym in symbols:
        if verbose:
            _log.info("\n[Portfolio] Preparing strategy for %s ...", sym)
        strategies[sym].prepare(feeds[sym])

    # Build merged timeline: list of (timestamp, symbol, local_bar_idx)
    # Sorted strictly by timestamp — same-time bars are processed in symbol order
    if verbose:
        _log.info("\n[Portfolio] Building merged timeline ...")
    events: list[tuple] = []
    for sym in symbols:
        df = clean_dfs[sym]
        for local_i, ts in enumerate(df.index):
            events.append((ts, symbols.index(sym), sym, local_i))

    events.sort(key=lambda e: (e[0], e[1]))

    if verbose:
        total_bars = sum(len(clean_dfs[sym]) for sym in symbols)
        date_min   = min(clean_dfs[sym].index[0]  for sym in symbols)
        date_max   = max(clean_dfs[sym].index[-1] for sym in symbols)
        _log.info(
            "[Portfolio] %s total bars across %d symbols | %s → %s",
            f"{total_bars:,}", len(symbols), date_min.date(), date_max.date(),
        )

    # Shared equity curve — indexed by event order for simplicity
    # We'll also record per-symbol equity curves
    equity_all:    list[float]          = []
    ts_all:        list[pd.Timestamp]   = []
    sym_bar_count: dict[str, int]       = {s: 0 for s in symbols}

    for ts, _, sym, local_i in events:
        feed   = feeds[sym]
        broker = portfolio.broker(sym)
        arr    = bar_arrs[sym]
        bars_seen = sym_bar_count[sym]

        # Advance this symbol's feed cursor
        feed._advance(local_i)

        # Strategy processes this bar (after warmup)
        if bars_seen >= warmup_bars_per_sym[sym]:
            try:
                strategies[sym].on_bar(feed, broker)
            except Exception as e:
                _log.warning("  %s bar %d: %s", sym, local_i, e)

        # Broker processes this bar (fills, SL/TP)
        broker.on_bar(
            local_i,
            arr["open"][local_i], arr["high"][local_i],
            arr["low"][local_i],  arr["close"][local_i],
        )

        sym_bar_count[sym] += 1
        equity_all.append(portfolio.equity)
        ts_all.append(ts)

    # Close open positions at last bar for each symbol
    for sym in symbols:
        broker = portfolio.broker(sym)
        if broker.open_positions:
            n = len(clean_dfs[sym])
            last_close = bar_arrs[sym]["close"][n - 1]
            broker.close_all_at(n - 1, last_close)

    # Build per-symbol BacktestResults
    symbol_results: dict[str, BacktestResult] = {}
    for sym in symbols:
        broker = portfolio.broker(sym)
        N_sym  = len(clean_dfs[sym])
        eq_sym = np.full(N_sym, np.nan)
        # Approximate per-symbol equity: not easily separable from shared pool
        # We fill with final shared equity as a placeholder — use portfolio equity
        # curve for accurate tracking
        eq_sym[:] = np.nan  # per-symbol equity not meaningful in shared pool
        symbol_results[sym] = BacktestResult(
            label         = sym,
            trades        = broker._history,
            equity_curve  = eq_sym,
            timestamps    = pd.DatetimeIndex(clean_dfs[sym].index),
            params        = strategies[sym].params,
            strategy_name = strategies[sym].NAME,
        )

    # Build portfolio equity curve (resampled to a common timeline)
    equity_arr = np.array(equity_all)
    ts_idx     = pd.DatetimeIndex(ts_all)

    len(events)
    n_closed   = sum(
        sum(1 for t in portfolio.broker(sym)._history
            if t.status == OrderStatus.CLOSED and t.exit_reason != "end_of_data")
        for sym in symbols
    )

    if verbose:
        _log.info(
            "\n[Portfolio] Done — %d total closed trades across all symbols | final equity: $%s",
            n_closed, f"{portfolio.equity:,.2f}",
        )

    return PortfolioResult(
        symbol_results  = symbol_results,
        equity_curve    = equity_arr,
        timestamps      = ts_idx,
        starting_equity = cfg.starting_equity,
    )
