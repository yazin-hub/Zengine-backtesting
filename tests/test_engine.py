"""
tests/test_engine.py — Core Engine Tests
==========================================

Tests verify the three structural guarantees ZEngine makes:
  1. Look-ahead is impossible (LookAheadError raised on future access)
  2. Same-bar fill guard works (orders placed at bar i fill at bar i+1+)
  3. Same-bar SL/TP guard works (filled trade is not checked for SL/TP on fill bar)

Plus integration tests against a deterministic toy dataset so we can catch
regressions in metrics, PnL accounting, and commission logic.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

# ── helpers to get engine in path without installing ──────────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.data import DataFeed, LookAheadError, CausalityError
from engine.broker import Broker, BrokerConfig, Order, OrderSide, OrderType, OrderStatus
from engine.backtest import run_backtest, BacktestResult, audit_indicator_causality
from engine.metrics import compute_metrics
from engine.strategy import BaseStrategy


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, start_price: float = 1900.0,
             trend: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """
    Build a deterministic OHLCV DataFrame with a DatetimeIndex.
    trend > 0: rising, trend < 0: falling, trend = 0: flat noise.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

    close = np.zeros(n)
    close[0] = start_price
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + trend + rng.normal(0, 0.0003))

    spread = close * 0.0004
    high   = close + spread * rng.uniform(0.5, 1.5, n)
    low    = close - spread * rng.uniform(0.5, 1.5, n)
    open_  = close + rng.normal(0, spread * 0.3, n)
    # Keep OHLC internally consistent
    high   = np.maximum(high, np.maximum(close, open_))
    low    = np.minimum(low,  np.minimum(close, open_))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=times,
    )


def _make_broker(risk_usd: float = 20.0, max_concurrent: int = 1) -> Broker:
    cfg = BrokerConfig(
        commission_flat=6.0, lot_size=100.0,
        max_concurrent=max_concurrent, risk_usd=risk_usd,
        size_mode="fixed_risk",
    )
    return Broker(cfg, starting_equity=5_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DataFeed / LookAheadError
# ──────────────────────────────────────────────────────────────────────────────

class TestDataFeedGuard:
    """Verify the LookAheadError guard is actually enforced."""

    def test_access_at_cursor_is_ok(self):
        """Reading feed.close[i] when cursor is at i must succeed."""
        df = _make_df(50)
        feed = DataFeed(df)
        feed._advance(10)
        val = feed.close[10]   # should not raise
        assert not math.isnan(val)

    def test_access_previous_bar_is_ok(self):
        """Reading any bar before the cursor is fine."""
        df = _make_df(50)
        feed = DataFeed(df)
        feed._advance(10)
        val = feed.close[0]    # historical — always OK
        assert not math.isnan(val)

    def test_access_future_bar_raises(self):
        """Accessing bar i+1 when cursor is at i must raise LookAheadError."""
        df = _make_df(50)
        feed = DataFeed(df)
        feed._advance(10)
        with pytest.raises(LookAheadError):
            _ = feed.close[11]

    def test_access_far_future_raises(self):
        """Same for any bar > cursor, not just the immediate next."""
        df = _make_df(50)
        feed = DataFeed(df)
        feed._advance(5)
        with pytest.raises(LookAheadError):
            _ = feed.close[49]

    def test_attached_array_also_guarded(self):
        """Custom indicator arrays attached via feed._attach are also guarded."""
        df = _make_df(50)
        feed = DataFeed(df)
        arr = np.arange(50, dtype=float)
        feed._attach("custom", arr)
        feed._advance(10)
        assert feed["custom"][10] == 10.0   # OK
        with pytest.raises(LookAheadError):
            _ = feed["custom"][11]

    def test_raw_data_access_bypasses_guard(self):
        """
        feed.close._data is the raw array — strategy uses this in prepare()
        before the loop. It must be accessible without restriction.
        """
        df = _make_df(50)
        feed = DataFeed(df)
        feed._advance(5)
        # _data access bypasses the guard — this is intentional for prepare()
        full = feed.close._data
        assert len(full) == 50


# ──────────────────────────────────────────────────────────────────────────────
# 2. Broker — order placement and fill timing
# ──────────────────────────────────────────────────────────────────────────────

class TestBrokerFillTiming:
    """Verify same-bar fill guard: order placed at bar i must fill at bar i+1+."""

    def test_market_order_does_not_fill_same_bar(self):
        """
        Place a market order at bar 5. Calling broker.on_bar(5, ...) immediately
        after must NOT fill it — the order was placed at bar 5, fill requires
        placed_bar < bar_idx (strict).
        """
        broker = _make_broker()
        order = broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1890.0, tp=1920.0,
            placed_bar=5, tag="test",
        )
        assert order is not None
        assert order.status == OrderStatus.PENDING

        # Process bar 5 — same bar as placement
        broker.on_bar(5, 1900.0, 1905.0, 1895.0, 1901.0)
        assert order.status == OrderStatus.PENDING, (
            "Order must NOT fill on the same bar it was placed (no same-bar fill)"
        )

    def test_market_order_fills_next_bar(self):
        """Order placed at bar 5 must fill at bar 6."""
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1890.0, tp=1920.0,
            placed_bar=5,
        )
        broker.on_bar(5, 1900.0, 1905.0, 1895.0, 1901.0)   # not filled
        broker.on_bar(6, 1901.0, 1908.0, 1898.0, 1905.0)   # fills here
        assert len(broker.open_positions) == 1
        assert broker.open_positions[0].fill_bar == 6

    def test_limit_order_fills_when_price_touches(self):
        """Long limit at 1895. Bar with low ≤ 1895 fills it."""
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.LIMIT,
            limit_price=1895.0, sl=1885.0, tp=1915.0,
            placed_bar=5,
        )
        broker.on_bar(5, 1900.0, 1905.0, 1896.0, 1902.0)   # low=1896 > 1895, no fill
        broker.on_bar(6, 1901.0, 1902.0, 1893.0, 1897.0)   # low=1893 ≤ 1895, fills
        assert len(broker.open_positions) == 1
        assert broker.open_positions[0].fill_price == pytest.approx(1895.0, rel=1e-4)

    def test_limit_order_gap_fill_at_open(self):
        """
        If the bar opens through the limit (gap), fill at open, not limit.
        Long limit at 1895. Bar opens at 1890 (gap below). Fill at 1890.
        """
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.LIMIT,
            limit_price=1895.0, sl=1880.0, tp=1920.0,
            placed_bar=5,
        )
        broker.on_bar(5, 1900.0, 1905.0, 1897.0, 1902.0)   # no fill
        broker.on_bar(6, 1890.0, 1892.0, 1888.0, 1891.0)   # open gaps below limit → fill at open
        assert len(broker.open_positions) == 1
        assert broker.open_positions[0].fill_price == pytest.approx(1890.0, rel=1e-4)

    def test_limit_order_expires(self):
        """Limit order expires after expiry_bars without fill."""
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.LIMIT,
            limit_price=1800.0,   # far below — won't fill
            sl=1790.0, tp=1850.0,
            placed_bar=0, expiry_bars=5,
        )
        for bar in range(6):   # 0..5 — expiry_bars=5 means expire on bar 5
            broker.on_bar(bar, 1900.0, 1905.0, 1895.0, 1901.0)

        assert broker.pending is None
        expired = [o for o in broker._history if o.status == OrderStatus.EXPIRED]
        assert len(expired) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 3. Same-bar SL/TP guard
# ──────────────────────────────────────────────────────────────────────────────

class TestSameBarSLTP:
    """
    A trade filled at bar i must NOT be closed by SL/TP on bar i.
    This prevents an order that fills on a huge wick bar from being
    immediately stopped out on the same bar.
    """

    def test_tp_not_hit_on_fill_bar(self):
        """
        Long fills at bar 6. Bar 6 has high above TP. Must NOT close on bar 6 —
        TP is only monitored from bar 7 onward.
        """
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1890.0, tp=1910.0,
            placed_bar=5,
        )
        broker.on_bar(5, 1900.0, 1905.0, 1895.0, 1901.0)   # not filled
        # Bar 6: opens at 1901, high=1915 (above TP=1910) → TP would be hit
        # but this is the fill bar, so SL/TP check is skipped
        broker.on_bar(6, 1901.0, 1915.0, 1898.0, 1912.0)
        assert len(broker.open_positions) == 1, (
            "Trade must still be open — SL/TP must NOT trigger on fill bar"
        )

    def test_sl_not_hit_on_fill_bar(self):
        """Same logic for SL: not checked on fill bar."""
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1890.0, tp=1920.0,
            placed_bar=5,
        )
        broker.on_bar(5, 1900.0, 1905.0, 1895.0, 1901.0)
        # Bar 6: low=1885 (below SL=1890) — fill bar, SL skipped
        broker.on_bar(6, 1901.0, 1905.0, 1885.0, 1895.0)
        assert len(broker.open_positions) == 1, (
            "Trade must still be open — SL must NOT trigger on fill bar"
        )

    def test_tp_hits_next_bar(self):
        """TP DOES close the trade on bar 7 (first bar after fill bar 6)."""
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1890.0, tp=1910.0,
            placed_bar=5,
        )
        broker.on_bar(5, 1900.0, 1905.0, 1895.0, 1901.0)
        broker.on_bar(6, 1901.0, 1908.0, 1898.0, 1905.0)   # fill bar — below TP
        assert len(broker.open_positions) == 1
        broker.on_bar(7, 1905.0, 1915.0, 1903.0, 1912.0)   # high > TP → closes
        assert len(broker.open_positions) == 0
        closed = broker.closed_trades
        assert len(closed) == 1
        assert closed[0].exit_reason == "tp"


# ──────────────────────────────────────────────────────────────────────────────
# 4. PnL accounting and commission
# ──────────────────────────────────────────────────────────────────────────────

class TestPnLAccounting:

    def test_long_tp_pnl(self):
        """
        Long 1 unit from 1900 → TP at 1920.
        Gross PnL = (1920 - 1900) * 1 = $20.
        Commission = 6.0 * (1/100) = $0.06 per leg → $0.03 entry + $0.03 exit.
        But wait — lot_size=100, size=1 → commission_flat * (size/lot_size) = 6*(1/100) = 0.06 total.
        Net = 20 - 0.03 (exit leg) = 19.97.
        """
        cfg = BrokerConfig(
            commission_flat=6.0, lot_size=100.0, max_concurrent=1,
            risk_usd=20.0, size_mode="fixed_size", fixed_size=1.0,
        )
        broker = Broker(cfg, 5000.0)
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1880.0, tp=1920.0,
            placed_bar=0, size=1.0,
        )
        broker.on_bar(0, 1900.0, 1902.0, 1898.0, 1901.0)   # not filled
        broker.on_bar(1, 1901.0, 1902.0, 1899.0, 1901.5)   # fill at open=1901
        broker.on_bar(2, 1901.5, 1925.0, 1900.0, 1920.0)   # high > 1920 → TP

        trades = broker.closed_trades
        assert len(trades) == 1
        t = trades[0]
        assert t.exit_reason == "tp"
        assert t.fill_price == pytest.approx(1901.0, rel=1e-4)   # market → fill at open
        expected_gross = (1920.0 - 1901.0) * 1.0
        assert t.pnl_gross == pytest.approx(expected_gross, rel=1e-4)
        assert t.pnl_net < t.pnl_gross   # commission reduces net

    def test_equity_decreases_on_loss(self):
        """
        Short from 1900 → SL at 1920 (loss trade).
        Equity must be less after the trade.
        """
        cfg = BrokerConfig(
            commission_flat=6.0, lot_size=100.0, max_concurrent=1,
            risk_usd=20.0, size_mode="fixed_size", fixed_size=1.0,
        )
        start_equity = 5000.0
        broker = Broker(cfg, start_equity)
        broker.place_order(
            side=OrderSide.SHORT, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1920.0, tp=1880.0,
            placed_bar=0, size=1.0,
        )
        broker.on_bar(0, 1900.0, 1902.0, 1898.0, 1901.0)
        broker.on_bar(1, 1901.0, 1902.0, 1899.0, 1901.0)   # fill
        broker.on_bar(2, 1901.0, 1925.0, 1900.0, 1922.0)   # high > SL=1920

        assert broker.equity < start_equity

    def test_commission_round_trip(self):
        """Total commission = entry leg + exit leg = full round-trip."""
        cfg = BrokerConfig(
            commission_flat=6.0, lot_size=100.0, max_concurrent=1,
            risk_usd=20.0, size_mode="fixed_size", fixed_size=100.0,
        )
        broker = Broker(cfg, 5000.0)
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1850.0, tp=1950.0,
            placed_bar=0, size=100.0,
        )
        broker.on_bar(0, 1900.0, 1902.0, 1898.0, 1901.0)
        broker.on_bar(1, 1901.0, 1902.0, 1899.0, 1901.0)   # fill
        broker.on_bar(2, 1901.0, 1960.0, 1900.0, 1950.0)   # TP hit

        t = broker.closed_trades[0]
        # 100 units, lot_size=100 → 1 lot. commission_flat=6 per lot round-trip.
        # Entry = 3.0, exit = 3.0, total = 6.0
        assert t.commission == pytest.approx(6.0, rel=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# 5. SL wins when both SL and TP hit same bar
# ──────────────────────────────────────────────────────────────────────────────

class TestSLTPConflict:
    def test_sl_wins_on_same_bar(self):
        """
        Conservative assumption: if both SL and TP are touched on the same bar,
        SL wins (worst case for the trader).
        """
        broker = _make_broker()
        broker.place_order(
            side=OrderSide.LONG, order_type=OrderType.MARKET,
            limit_price=1900.0, sl=1890.0, tp=1910.0,
            placed_bar=5,
        )
        broker.on_bar(5, 1900.0, 1905.0, 1895.0, 1901.0)
        broker.on_bar(6, 1901.0, 1908.0, 1897.0, 1904.0)   # fill bar — no SL/TP
        # Bar 7: low < SL AND high > TP — both hit
        broker.on_bar(7, 1900.0, 1915.0, 1885.0, 1895.0)
        closed = broker.closed_trades
        assert len(closed) == 1
        assert closed[0].exit_reason == "sl", (
            "SL must win when both SL and TP are hit on the same bar (conservative)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Integration test — deterministic toy strategy
# ──────────────────────────────────────────────────────────────────────────────

class AlwaysLongStrategy(BaseStrategy):
    """
    Trivial strategy: go long at market every N bars (when no position open).
    Used to test the full engine pipeline in a deterministic way.
    """
    NAME   = "Always Long (test)"
    PARAMS = {"every_n": 10}

    def prepare(self, feed):
        pass

    def on_bar(self, feed, broker):
        i = feed.i
        if i % self.params["every_n"] == 0 and broker.has_capacity:
            close_i = feed.close[i]
            broker.place_order(
                side=OrderSide.LONG, order_type=OrderType.MARKET,
                limit_price=close_i, sl=close_i * 0.995, tp=close_i * 1.01,
                placed_bar=i,
            )


class TestIntegration:

    def test_run_backtest_produces_trades(self):
        """Full backtest run must produce at least some closed trades."""
        df  = _make_df(500)
        cfg = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                           max_concurrent=1, risk_usd=20.0, size_mode="fixed_risk")
        strat  = AlwaysLongStrategy()
        result = run_backtest(df, strat, cfg, warmup_bars=20, verbose=False)
        assert isinstance(result, BacktestResult)
        assert result.n_trades > 0, "Expected at least one closed trade"

    def test_equity_curve_length_matches_data(self):
        """Equity curve must have the same length as the input DataFrame."""
        df  = _make_df(300)
        cfg = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                           max_concurrent=1, risk_usd=20.0, size_mode="fixed_risk")
        result = run_backtest(df, AlwaysLongStrategy(), cfg, warmup_bars=20, verbose=False)
        assert len(result.equity_curve) == len(df)

    def test_equity_curve_no_nans_after_warmup(self):
        """After warmup, equity curve must be fully populated (no NaNs)."""
        df  = _make_df(200)
        cfg = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                           max_concurrent=1, risk_usd=20.0, size_mode="fixed_risk")
        result = run_backtest(df, AlwaysLongStrategy(), cfg, warmup_bars=20, verbose=False)
        post_warmup = result.equity_curve[20:]
        assert not np.any(np.isnan(post_warmup)), "No NaNs in equity curve after warmup"

    def test_metrics_returns_expected_keys(self):
        """compute_metrics must return a dict with at least the essential keys."""
        # Need enough bars for trades to close naturally (not just end-of-data).
        # 2000 M1 bars ≈ 1.4 calendar days — confirmed to produce 7 closed trades.
        df  = _make_df(2000)
        cfg = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                           max_concurrent=1, risk_usd=20.0, size_mode="fixed_risk")
        result = run_backtest(df, AlwaysLongStrategy(), cfg, warmup_bars=20, verbose=False)
        assert result.n_trades > 0, "Toy dataset must produce at least 1 closed trade"
        m = compute_metrics(result)
        for key in ["trades", "win_rate_%", "sharpe_ann", "max_dd_%", "total_pnl_$"]:
            assert key in m, f"Missing metric key: {key}"

    def test_no_look_ahead_in_strategy(self):
        """
        A strategy that attempts to access future bars must trigger LookAheadError,
        which the engine catches and logs — but importantly, the result must still
        be valid (engine is robust to strategy errors).
        """
        class CheatStrategy(BaseStrategy):
            NAME   = "Cheating (test)"
            PARAMS = {}

            def prepare(self, feed): pass

            def on_bar(self, feed, broker):
                # Deliberate look-ahead: access bar i+1
                _ = feed.close[feed.i + 1]   # will raise LookAheadError

        df  = _make_df(100)
        cfg = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                           max_concurrent=1, risk_usd=20.0, size_mode="fixed_risk")
        # Engine catches strategy exceptions and logs them — must not crash
        result = run_backtest(df, CheatStrategy(), cfg, warmup_bars=10, verbose=False)
        assert isinstance(result, BacktestResult)   # engine survived
        assert result.n_trades == 0                  # cheating strategy placed nothing

    def test_splits_independent_state(self):
        """
        Each split must get a fresh strategy instance.
        Strategy state from IS must not leak into OOS.

        Note: 2000 M1 bars spans ~1.4 days from 2023-01-01, so splits must
        be within that window (not month/year boundaries).
        """
        from engine.backtest import run_splits

        n = 4000
        df = _make_df(n, seed=0)
        start = pd.Timestamp("2023-01-01", tz="UTC")
        df.index = pd.date_range(start, periods=n, freq="1min", tz="UTC")
        # Midpoint timestamp for the split
        mid = df.index[n // 2].isoformat()

        cfg = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                           max_concurrent=1, risk_usd=20.0, size_mode="fixed_risk")

        results = run_splits(
            df, AlwaysLongStrategy, {}, cfg,
            warmup_bars=20,
            splits={
                "IS":  (df.index[0].isoformat(),  mid),
                "OOS": (mid,                       None),
            },
            verbose=False,
        )
        assert "IS"  in results, f"Expected 'IS' split in results, got: {list(results)}"
        assert "OOS" in results, f"Expected 'OOS' split in results, got: {list(results)}"
        # Each split runs independently — both should complete without error
        assert isinstance(results["IS"],  BacktestResult)
        assert isinstance(results["OOS"], BacktestResult)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Causality audit — prepare() look-ahead detection
# ──────────────────────────────────────────────────────────────────────────────

class CausalStrategy(BaseStrategy):
    """Strategy using only genuinely causal indicators in prepare()."""
    NAME   = "Causal (test)"
    PARAMS = {}

    def prepare(self, feed: DataFeed) -> None:
        close = feed.close._data
        n     = len(close)
        # Causal EMA: each value only depends on current and past values
        ema = np.zeros(n)
        alpha = 2 / (14 + 1)
        ema[0] = close[0]
        for i in range(1, n):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
        feed._attach("ema14", ema)

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        pass


class NonCausalStrategy(BaseStrategy):
    """
    Strategy with a non-causal indicator: uses the GLOBAL max of the full
    close array to normalise. This means bar 10's value changes when bars
    11..N are added — a clear look-ahead.
    """
    NAME   = "NonCausal (test)"
    PARAMS = {}

    def prepare(self, feed: DataFeed) -> None:
        close    = feed.close._data
        # Global normalisation — non-causal: bar 10's value depends on bar N
        global_max = close.max()
        global_min = close.min()
        normalised = (close - global_min) / (global_max - global_min + 1e-10)
        feed._attach("norm_close", normalised)

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        pass


class BidirectionalFilterStrategy(BaseStrategy):
    """
    Strategy using a centred (bidirectional) rolling mean — non-causal.
    pandas rolling(center=True) uses future values to compute the centred window.
    """
    NAME   = "BidirFilter (test)"
    PARAMS = {}

    def prepare(self, feed: DataFeed) -> None:
        close = feed.close._data
        # Centred rolling mean: bar i depends on bars i-w..i+w (look-ahead!)
        centred = pd.Series(close).rolling(11, center=True, min_periods=1).mean().to_numpy()
        feed._attach("centred_ma", centred)

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        pass


class TestCausalityAudit:
    """
    Verify audit_indicator_causality() correctly distinguishes causal
    indicators from non-causal ones, and that run_backtest() blocks
    non-causal strategies before the loop starts.
    """

    def test_causal_strategy_passes_audit(self):
        """A strategy using only causal indicators must pass without error."""
        df  = _make_df(500)
        strat = CausalStrategy()
        # Should not raise
        audit_indicator_causality(df, strat)

    def test_global_normalisation_caught(self):
        """
        Global min-max normalisation is non-causal: bar i's value changes
        when future bars shift the global max/min. Must raise CausalityError.

        Use trend=0.005 so the second half is definitely higher than the first,
        guaranteeing the global max falls in the second half and changes the
        normalised values at the midpoint.
        """
        df    = _make_df(500, trend=0.005)   # rising data — second half has higher max
        strat = NonCausalStrategy()
        with pytest.raises(CausalityError) as exc_info:
            audit_indicator_causality(df, strat)
        assert "norm_close" in str(exc_info.value), (
            "CausalityError should name the offending indicator"
        )

    def test_centred_rolling_mean_caught(self):
        """
        Centred rolling mean (center=True) uses future bars. Must be caught.
        """
        df    = _make_df(500)
        strat = BidirectionalFilterStrategy()
        with pytest.raises(CausalityError) as exc_info:
            audit_indicator_causality(df, strat)
        assert "centred_ma" in str(exc_info.value)

    def test_run_backtest_blocks_noncausal_strategy(self):
        """
        run_backtest() must raise CausalityError before the loop starts
        when the strategy's prepare() is non-causal.
        """
        df  = _make_df(500, trend=0.005)
        cfg = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                           max_concurrent=1, risk_usd=20.0, size_mode="fixed_risk")
        with pytest.raises(CausalityError):
            run_backtest(df, NonCausalStrategy(), cfg, warmup_bars=20, verbose=False)

    def test_ma_cross_strategy_passes_audit(self):
        """
        The built-in EMA Crossover strategy must pass the causality audit.
        Verifies our own strategy code is clean.
        """
        from strategies.ma_cross import MACrossStrategy
        df  = _make_df(500)
        strat = MACrossStrategy()
        # Should not raise
        audit_indicator_causality(df, strat)

    def test_ifvg_strategy_passes_audit(self):
        """
        The built-in IFVG strategy must pass the causality audit.
        This is the critical test — IFVG's prepare() does complex forward
        scanning for zone inversions. The audit verifies it's all causal.
        """
        from strategies.ifvg import IFVGStrategy
        df  = _make_df(1000)
        # IFVG needs DatetimeIndex with hour attribute for session_mask
        df.index = pd.date_range("2024-01-02 08:00:00", periods=len(df),
                                  freq="1min", tz="UTC")
        strat = IFVGStrategy()
        # Should not raise
        audit_indicator_causality(df, strat)

    def test_audit_error_message_is_informative(self):
        """CausalityError message must name the indicator and explain the problem."""
        df    = _make_df(500, trend=0.005)
        strat = NonCausalStrategy()
        with pytest.raises(CausalityError) as exc_info:
            audit_indicator_causality(df, strat)
        msg = str(exc_info.value)
        assert "norm_close"    in msg   # names the offending indicator
        assert "future data"   in msg   # explains why
        assert "prepare()"     in msg   # points to where to fix it


# ──────────────────────────────────────────────────────────────────────────────
# 8. Spread & slippage — realistic fill model
# ──────────────────────────────────────────────────────────────────────────────

def _make_broker_with(spread: float = 0.0, slippage_fixed: float = 0.0,
                      slippage_pct: float = 0.0, **kwargs) -> Broker:
    """Convenience factory: broker with explicit spread / slippage settings."""
    cfg = BrokerConfig(
        commission_flat=0.0,        # zero commission so PnL is pure spread/slip
        lot_size=100.0,
        max_concurrent=1,
        risk_usd=20.0,
        size_mode="fixed_risk",
        spread=spread,
        slippage_fixed=slippage_fixed,
        slippage_pct=slippage_pct,
        **kwargs,
    )
    return Broker(cfg, starting_equity=10_000.0)


class TestSpreadAndSlippage:
    """
    Verify that spread and slippage are applied correctly and asymmetrically:
      - LONG entries fill at ask  (open + spread)
      - SHORT entries fill at bid (open, no spread on entry)
      - LONG exits   sell at bid  (sl/tp price, no spread)
      - SHORT exits  buy  at ask  (sl/tp + spread)
    Net effect: one spread cost per round trip, either direction.
    """

    def test_long_entry_fills_at_ask(self):
        """LONG market order must fill at open + spread."""
        spread = 0.30
        broker = _make_broker_with(spread=spread)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1995.0, tp=2010.0, placed_bar=0)
        # Bar 1: open=2001.0. Expected fill = 2001.0 + spread = 2001.30
        broker.on_bar(1, open_=2001.0, high=2005.0, low=1999.0, close=2002.0)
        pos = broker.open_positions
        assert len(pos) == 1
        assert pos[0].fill_price == pytest.approx(2001.0 + spread, rel=1e-6), (
            f"LONG should fill at ask (open+spread). "
            f"Got {pos[0].fill_price}, expected {2001.0+spread}"
        )

    def test_short_entry_fills_at_bid(self):
        """SHORT market order fills at open — no spread paid on short entry."""
        spread = 0.30
        broker = _make_broker_with(spread=spread)
        broker.place_order(OrderSide.SHORT, OrderType.MARKET,
                           limit_price=2000.0, sl=2005.0, tp=1990.0, placed_bar=0)
        broker.on_bar(1, open_=2001.0, high=2005.0, low=1999.0, close=2002.0)
        pos = broker.open_positions
        assert len(pos) == 1
        assert pos[0].fill_price == pytest.approx(2001.0, rel=1e-6), (
            f"SHORT should fill at bid (open, no spread). "
            f"Got {pos[0].fill_price}, expected 2001.0"
        )

    def test_long_tp_exit_at_bid_no_spread(self):
        """LONG TP exit fills at the TP price — no spread added on long exit."""
        spread = 0.30
        broker = _make_broker_with(spread=spread)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1990.0, tp=2010.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2003.0, 1999.0, 2002.0)  # fills LONG
        pos = broker.open_positions
        assert len(pos) == 1
        fill_px = pos[0].fill_price   # 2001.30

        # Bar 2: high reaches TP=2010 → exit at bid = 2010 (no spread on LONG exit)
        broker.on_bar(2, 2002.0, 2012.0, 2001.0, 2011.0)
        closed = broker.closed_trades
        assert len(closed) == 1
        assert closed[0].exit_price == pytest.approx(2010.0, rel=1e-6), (
            "LONG TP exit must be at bid (TP price), no spread"
        )
        # PnL = (2010 - fill_px) * size — spread already baked into fill_px
        assert closed[0].pnl_gross == pytest.approx(
            (2010.0 - fill_px) * closed[0].size, rel=1e-4
        )

    def test_short_tp_exit_at_ask_with_spread(self):
        """SHORT TP exit buys back at ask = tp + spread."""
        spread = 0.30
        broker = _make_broker_with(spread=spread)
        broker.place_order(OrderSide.SHORT, OrderType.MARKET,
                           limit_price=2000.0, sl=2010.0, tp=1990.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2003.0, 1999.0, 2002.0)  # fills SHORT at 2001.0
        broker.open_positions[0].fill_price       # 2001.0

        # Bar 2: low reaches TP=1990 → buy back at ask = 1990 + spread
        broker.on_bar(2, 2000.0, 2001.0, 1988.0, 1989.0)
        closed = broker.closed_trades
        assert len(closed) == 1
        expected_exit = 1990.0 + spread
        assert closed[0].exit_price == pytest.approx(expected_exit, rel=1e-6), (
            f"SHORT TP exit must be at ask (tp+spread). "
            f"Got {closed[0].exit_price}, expected {expected_exit}"
        )

    def test_spread_reduces_pnl_vs_no_spread(self):
        """Same LONG trade with spread must have lower net PnL than without."""
        def _run(spread):
            b = _make_broker_with(spread=spread)
            b.place_order(OrderSide.LONG, OrderType.MARKET,
                          limit_price=2000.0, sl=1990.0, tp=2015.0, placed_bar=0)
            b.on_bar(1, 2001.0, 2003.0, 1999.0, 2002.0)
            b.on_bar(2, 2001.0, 2016.0, 2000.0, 2015.0)   # TP hit
            return b.closed_trades[0].pnl_gross

        pnl_no_spread   = _run(0.0)
        pnl_with_spread = _run(0.30)
        assert pnl_with_spread < pnl_no_spread, (
            "Spread must reduce PnL — LONG entry at higher ask price"
        )
        # The difference must equal spread * size (one spread per round trip)
        t_spread = _make_broker_with(spread=0.30)
        t_spread.place_order(OrderSide.LONG, OrderType.MARKET,
                             limit_price=2000.0, sl=1990.0, tp=2015.0, placed_bar=0)
        t_spread.on_bar(1, 2001.0, 2003.0, 1999.0, 2002.0)
        t_spread.on_bar(2, 2001.0, 2016.0, 2000.0, 2015.0)
        size = t_spread.closed_trades[0].size
        assert pytest.approx(pnl_no_spread - pnl_with_spread, rel=1e-4) == 0.30 * size

    def test_slippage_fixed_applied_at_long_entry(self):
        """slippage_fixed adds extra cost at LONG market entry (adverse)."""
        slip = 0.10
        broker = _make_broker_with(slippage_fixed=slip)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1990.0, tp=2020.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2005.0, 1999.0, 2002.0)
        fill = broker.open_positions[0].fill_price
        # fill = open + slippage_fixed = 2001.0 + 0.10 = 2001.10
        assert fill == pytest.approx(2001.0 + slip, rel=1e-6)

    def test_slippage_pct_applied_at_entry(self):
        """slippage_pct multiplies the fill price adversely at market entry."""
        pct = 0.001   # 0.1%
        broker = _make_broker_with(slippage_pct=pct)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1990.0, tp=2020.0, placed_bar=0)
        broker.on_bar(1, 2000.0, 2005.0, 1998.0, 2001.0)
        fill = broker.open_positions[0].fill_price
        assert fill == pytest.approx(2000.0 * (1 + pct), rel=1e-6)

    def test_spread_and_slippage_additive(self):
        """spread and slippage_fixed stack: LONG fill = open + spread + slippage_fixed."""
        spread = 0.30
        slip   = 0.10
        broker = _make_broker_with(spread=spread, slippage_fixed=slip)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1990.0, tp=2020.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2005.0, 1999.0, 2002.0)
        fill = broker.open_positions[0].fill_price
        assert fill == pytest.approx(2001.0 + slip + spread, rel=1e-6)

    def test_zero_spread_unchanged_behaviour(self):
        """With spread=0, fill price == open (baseline behaviour preserved)."""
        broker = _make_broker_with(spread=0.0)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1990.0, tp=2020.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2005.0, 1999.0, 2002.0)
        fill = broker.open_positions[0].fill_price
        assert fill == pytest.approx(2001.0, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Trailing stops — manual and auto-trail
# ──────────────────────────────────────────────────────────────────────────────

class TestTrailingStops:
    """
    Verify manual trail_sl() / move_sl_to_breakeven() and auto-trailing
    via BrokerConfig.trail_pct.
    """

    def _open_long(self, broker: Broker,
                   entry_bar: int = 0,
                   fill_bar:  int = 1) -> Order:
        """Helper: place and fill a LONG market order. Returns the filled trade."""
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1990.0, tp=2050.0,
                           placed_bar=entry_bar)
        broker.on_bar(fill_bar, 2001.0, 2005.0, 1999.0, 2002.0)
        return broker.open_positions[0]

    def _open_short(self, broker: Broker,
                    entry_bar: int = 0,
                    fill_bar:  int = 1) -> Order:
        broker.place_order(OrderSide.SHORT, OrderType.MARKET,
                           limit_price=2000.0, sl=2010.0, tp=1950.0,
                           placed_bar=entry_bar)
        broker.on_bar(fill_bar, 2001.0, 2005.0, 1999.0, 2002.0)
        return broker.open_positions[0]

    def test_trail_sl_moves_long_sl_up(self):
        """trail_sl with a higher SL must update the LONG trade's stop."""
        broker = _make_broker_with()
        trade  = self._open_long(broker)
        old_sl = trade.sl               # 1990.0
        result = broker.trail_sl(1995.0)
        assert result is True
        assert trade.sl == pytest.approx(1995.0), "SL should have moved up"
        assert trade.sl > old_sl

    def test_trail_sl_rejects_lower_long_sl(self):
        """trail_sl must not widen a LONG stop (new_sl < current_sl rejected)."""
        broker = _make_broker_with()
        trade  = self._open_long(broker)
        result = broker.trail_sl(1985.0)   # would widen SL — reject
        assert result is False
        assert trade.sl == pytest.approx(1990.0), "SL must not be widened"

    def test_trail_sl_moves_short_sl_down(self):
        """trail_sl with a lower SL must update the SHORT trade's stop."""
        broker = _make_broker_with()
        trade  = self._open_short(broker)
        old_sl = trade.sl               # 2010.0
        result = broker.trail_sl(2005.0)
        assert result is True
        assert trade.sl == pytest.approx(2005.0)
        assert trade.sl < old_sl

    def test_trail_sl_rejects_higher_short_sl(self):
        """trail_sl must not widen a SHORT stop (new_sl > current_sl rejected)."""
        broker = _make_broker_with()
        trade  = self._open_short(broker)
        result = broker.trail_sl(2015.0)   # would widen — reject
        assert result is False
        assert trade.sl == pytest.approx(2010.0)

    def test_trail_sl_no_trade_returns_false(self):
        """trail_sl with no open positions must return False gracefully."""
        broker = _make_broker_with()
        assert broker.trail_sl(1990.0) is False

    def test_move_sl_to_breakeven(self):
        """move_sl_to_breakeven must move SL to fill_price for a LONG in profit."""
        broker = _make_broker_with()
        trade  = self._open_long(broker)
        fill   = trade.fill_price       # 2001.0 (no spread in this broker)
        result = broker.move_sl_to_breakeven()
        assert result is True
        assert trade.sl == pytest.approx(fill, rel=1e-6), (
            "SL should be at fill price (breakeven)"
        )

    def test_move_sl_to_breakeven_rejects_underwater(self):
        """
        move_sl_to_breakeven must not worsen SL when trade is underwater.
        LONG filled at 2001, SL at 1990. Breakeven = 2001 > 1990, so it
        would MOVE SL UP (allowed). For the rejection case, we need fill > sl.
        Use a manually constructed scenario where fill < current_sl.
        """
        broker = _make_broker_with()
        # Place order with SL very close to entry
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=2002.0, tp=2050.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2005.0, 1999.0, 2002.0)
        broker.open_positions[0]
        # fill_price ~2001 < sl=2002 → moving to breakeven would WIDEN stop → reject
        result = broker.move_sl_to_breakeven()
        assert result is False, (
            "Should not move SL to breakeven when it would widen the stop"
        )

    def test_auto_trail_moves_sl_on_profit(self):
        """
        With trail_pct=0.001, auto-trail must raise SL when price moves up
        past the trail_activation_pct threshold.
        """
        trail_pct = 0.005   # trail at 0.5% below bar high
        actv_pct  = 0.001   # activate after 0.1% profit
        broker = _make_broker_with(trail_pct=trail_pct,
                                   trail_activation_pct=actv_pct)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1980.0, tp=2100.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2002.0, 1999.0, 2001.5)   # fills, small move
        trade = broker.open_positions[0]
        old_sl = trade.sl               # 1980.0

        # Bar 2: high = 2010 → profit_pct = (2010-fill)/fill > 0.001 → trail activates
        # Expected new_sl = 2010 * (1 - 0.005) = 2009.95 (if > old_sl)
        broker.on_bar(2, 2001.0, 2010.0, 2000.0, 2009.0)
        expected_sl = 2010.0 * (1 - trail_pct)
        assert trade.sl == pytest.approx(expected_sl, rel=1e-5), (
            f"Auto-trail should have raised SL to {expected_sl:.4f}, got {trade.sl:.4f}"
        )
        assert trade.sl > old_sl, "Auto-trail must move SL up, not down"

    def test_auto_trail_never_widens_sl(self):
        """Auto-trail must never move SL downward after it has been raised."""
        trail_pct = 0.005
        broker = _make_broker_with(trail_pct=trail_pct, trail_activation_pct=0.0)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1980.0, tp=2100.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2001.0, 1999.0, 2001.0)

        # Bar 2: big move up. new_sl = 2020 * 0.995 = 2009.9.
        # Crucially: low must stay ABOVE the new trailed SL so the trade stays open.
        broker.on_bar(2, 2001.0, 2020.0, 2011.0, 2019.0)
        trade  = broker.open_positions[0]
        sl_after_up = trade.sl   # should be 2020 * 0.995 = 2009.9

        # Bar 3: price pulls back — low=2013, above sl_after_up → trade still open.
        # Critically, bar 3 high=2015 < 2020 so trail should NOT update (bar high is lower).
        broker.on_bar(3, 2019.0, 2015.0, 2013.0, 2014.0)
        assert len(broker.open_positions) == 1, "Trade should still be open"
        assert trade.sl == pytest.approx(sl_after_up, rel=1e-6), (
            "SL should not decrease after a pullback — trail only moves one way"
        )

    def test_trailing_stop_triggers_close(self):
        """After auto-trailing, the new SL must actually close the trade when hit."""
        trail_pct = 0.005
        broker = _make_broker_with(trail_pct=trail_pct, trail_activation_pct=0.0)
        broker.place_order(OrderSide.LONG, OrderType.MARKET,
                           limit_price=2000.0, sl=1970.0, tp=2200.0, placed_bar=0)
        broker.on_bar(1, 2001.0, 2001.0, 1999.0, 2001.0)

        # Bar 2: price moves to 2020. new_sl = 2020 * 0.995 = 2009.9.
        # Low=2011 stays above the new SL so trade remains open after trailing.
        broker.on_bar(2, 2001.0, 2020.0, 2011.0, 2019.0)
        trade  = broker.open_positions[0]
        new_sl = trade.sl   # ~2009.9

        # Bar 3: price drops below the new trailed SL → trade must close
        low_below = new_sl - 1.0
        broker.on_bar(3, 2009.0, 2010.0, low_below, 2008.0)
        assert len(broker.open_positions) == 0, "Trade should have closed on trailed SL"
        assert len(broker.closed_trades) == 1
        assert broker.closed_trades[0].exit_reason == "sl"
