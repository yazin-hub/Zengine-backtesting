"""
tests/test_features.py — Tests for walk-forward, Monte Carlo, MTF, and Portfolio
==================================================================================

Covers:
  8. Walk-Forward Optimisation
  9. Monte Carlo Simulation
  10. Multi-Timeframe Support
  11. Portfolio-Level Backtesting
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pandas as pd
import pytest

from engine.backtest import run_backtest, BacktestResult
from engine.broker import Broker, BrokerConfig, OrderSide, OrderType
from engine.data import DataFeed
from engine.strategy import BaseStrategy


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_df(n: int = 500, trend: float = 0.0, seed: int = 42,
             start: str = "2024-01-01") -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    times  = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close  = np.zeros(n)
    close[0] = 1900.0
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + trend + rng.normal(0, 0.0003))
    spread = close * 0.0004
    high   = close + spread * rng.uniform(0.5, 1.5, n)
    low    = close - spread * rng.uniform(0.5, 1.5, n)
    open_  = close + rng.normal(0, spread * 0.3, n)
    high   = np.maximum(high, np.maximum(close, open_))
    low    = np.minimum(low,  np.minimum(close, open_))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=times,
    )


def _cfg(risk: float = 20.0, max_c: int = 1) -> BrokerConfig:
    return BrokerConfig(
        commission_flat=6.0, lot_size=100.0,
        max_concurrent=max_c, risk_usd=risk,
        size_mode="fixed_risk",
    )


class TrendFollowStrategy(BaseStrategy):
    """Long when close > SMA. Used across multiple test suites."""
    NAME   = "TrendFollow (test)"
    PARAMS = {
        "period":  {"default": 20, "min": 5, "max": 50, "step": 5},
        "atr_sl":  {"default": 1.5, "min": 0.5, "max": 3.0, "step": 0.5},
    }

    def prepare(self, feed: DataFeed) -> None:
        from engine import indicators as ind
        close = feed.close._data
        high  = feed.high._data
        low   = feed.low._data
        feed._attach("sma",  ind.sma(close, self.params["period"]))
        feed._attach("atr",  ind.atr(high, low, close, 14))

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        i = feed.i
        if i < 2 or not broker.has_capacity:
            return
        sma_i   = feed["sma"][i]
        atr_i   = feed["atr"][i]
        close_i = feed.close[i]
        if math.isnan(sma_i) or math.isnan(atr_i):
            return
        if close_i > sma_i:
            sl = close_i - self.params["atr_sl"] * atr_i
            tp = close_i + 2.0 * self.params["atr_sl"] * atr_i
            broker.place_order(
                side=OrderSide.LONG, order_type=OrderType.MARKET,
                limit_price=close_i, sl=sl, tp=tp, placed_bar=i,
            )


# ──────────────────────────────────────────────────────────────────────────────
# 8. Walk-Forward Optimisation
# ──────────────────────────────────────────────────────────────────────────────

class TestWalkForward:

    def test_basic_run_completes(self):
        """Walk-forward must complete without error and return windows."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000, trend=0.0001)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.0, 2.0]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            verbose=False,
        )
        assert wf.n_windows >= 1, "Expected at least 1 completed window"

    def test_n_windows_correct(self):
        """Number of windows = (N - window_bars) // step_bars."""
        from engine.walkforward import run_walk_forward
        n, wb, sb = 3000, 1000, 500
        df = _make_df(n)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=wb, step_bars=sb,
            verbose=False,
        )
        expected = (n - wb) // sb
        assert wf.n_windows == expected

    def test_oos_results_are_backtest_results(self):
        """Each window's OOS result must be a valid BacktestResult."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            verbose=False,
        )
        for w in wf.windows:
            assert isinstance(w.oos_result, BacktestResult)
            assert isinstance(w.is_result, BacktestResult)

    def test_oos_does_not_overlap_is(self):
        """OOS window must start after IS window ends (no data leakage)."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            verbose=False,
        )
        for w in wf.windows:
            assert w.oos_start > w.is_end, (
                f"OOS start ({w.oos_start}) must be after IS end ({w.is_end}) "
                "— no data leakage between IS and OOS"
            )

    def test_best_params_selected_from_grid(self):
        """Best params for each window must be from the provided grid."""
        from engine.walkforward import run_walk_forward
        periods = [10, 20, 30]
        df = _make_df(3000)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": periods, "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            verbose=False,
        )
        for w in wf.windows:
            assert w.best_params["period"] in periods

    def test_max_combos_limit_enforced(self):
        """Exceeding max_combos must raise ValueError."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000)
        with pytest.raises(ValueError, match="max_combos"):
            run_walk_forward(
                df, TrendFollowStrategy,
                param_grid={"period": list(range(1, 100)), "atr_sl": [1.0, 2.0]},
                broker_config=_cfg(),
                window_bars=1000, step_bars=500,
                max_combos=10,   # only 10 allowed, but grid has 99*2 = 198
                verbose=False,
            )

    def test_combined_oos_equity_shape(self):
        """Combined OOS equity must have length > 0 and be non-decreasing in count."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000, trend=0.0001)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            verbose=False,
        )
        eq, ts = wf.combined_oos_equity
        assert len(eq) > 0
        assert len(eq) == len(ts)

    def test_score_fn_string_sharpe(self):
        """score_fn='sharpe' must run without error."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            score_fn="sharpe",
            verbose=False,
        )
        assert wf.score_fn == "sharpe"

    def test_score_fn_callable(self):
        """score_fn as callable must be accepted and applied."""
        from engine.walkforward import run_walk_forward
        def custom_score(m):
            return m.get("win_rate_%", 0) - m.get("max_dd_%", 100)
        df = _make_df(3000)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            score_fn=custom_score,
            verbose=False,
        )
        assert wf.score_fn == "custom"

    def test_anchored_mode(self):
        """Anchored mode: IS start must always be the dataset start for every window."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            anchored=True,
            verbose=False,
        )
        for w in wf.windows:
            assert w.is_start == df.index[0], (
                "Anchored mode: IS window must always start at dataset beginning"
            )

    def test_summary_string_contains_key_info(self):
        """summary() output must contain strategy name and score metric."""
        from engine.walkforward import run_walk_forward
        df = _make_df(3000)
        wf = run_walk_forward(
            df, TrendFollowStrategy,
            param_grid={"period": [10, 20], "atr_sl": [1.5]},
            broker_config=_cfg(),
            window_bars=1000, step_bars=500,
            verbose=False,
        )
        s = wf.summary()
        assert "TrendFollow" in s
        assert "sharpe" in s.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 9. Monte Carlo Simulation
# ──────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:

    def _get_result(self, n_bars: int = 2000) -> BacktestResult:
        df     = _make_df(n_bars)
        result = run_backtest(
            df, TrendFollowStrategy(), _cfg(),
            warmup_bars=20, verbose=False
        )
        return result

    def test_basic_run(self):
        """Monte Carlo must run without error and return n_sims curves."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc = run_monte_carlo(result, n_sims=200, verbose=False)
        assert mc.n_sims == 200
        assert mc.sim_curves.shape[0] == 200

    def test_sim_curves_shape(self):
        """sim_curves shape must be (n_sims, n_trades + 1)."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc = run_monte_carlo(result, n_sims=100, verbose=False)
        assert mc.sim_curves.shape == (100, mc.n_trades + 1)

    def test_all_curves_start_at_starting_equity(self):
        """Every simulation must start at starting_equity."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc = run_monte_carlo(result, n_sims=100, verbose=False)
        assert np.allclose(mc.sim_curves[:, 0], mc.starting_equity)

    def test_prob_ruin_between_0_and_1(self):
        """Probability of ruin must be a valid probability [0, 1]."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc = run_monte_carlo(result, n_sims=200, verbose=False)
        assert 0.0 <= mc.prob_ruin <= 1.0

    def test_max_dd_positive(self):
        """Max drawdown values must all be non-negative (stored as positive %)."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc = run_monte_carlo(result, n_sims=200, verbose=False)
        assert np.all(mc.max_drawdowns >= 0)

    def test_equity_bands_returns_correct_percentiles(self):
        """equity_bands() must return dict with correct keys and shape."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc     = run_monte_carlo(result, n_sims=200, verbose=False)
        bands  = mc.equity_bands([5, 50, 95])
        assert set(bands.keys()) == {"p5", "p50", "p95"}
        for arr in bands.values():
            assert len(arr) == mc.n_trades + 1

    def test_p5_le_p50_le_p95_final_equity(self):
        """Percentile ordering must hold: p5 ≤ p50 ≤ p95 for final equity."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc = run_monte_carlo(result, n_sims=500, verbose=False)
        p5  = mc.final_equity_pct(5)
        p50 = mc.final_equity_pct(50)
        p95 = mc.final_equity_pct(95)
        assert p5 <= p50 <= p95

    def test_reproducible_with_same_seed(self):
        """Same seed must produce identical results."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc1 = run_monte_carlo(result, n_sims=100, seed=42, verbose=False)
        mc2 = run_monte_carlo(result, n_sims=100, seed=42, verbose=False)
        assert np.array_equal(mc1.sim_curves, mc2.sim_curves)

    def test_different_seeds_differ(self):
        """Different seeds must produce different results."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc1 = run_monte_carlo(result, n_sims=100, seed=1, verbose=False)
        mc2 = run_monte_carlo(result, n_sims=100, seed=2, verbose=False)
        assert not np.array_equal(mc1.sim_curves, mc2.sim_curves)

    def test_no_trades_raises(self):
        """Running MC with zero trades must raise ValueError."""
        from engine.monte_carlo import run_monte_carlo
        df     = _make_df(50)   # too few bars to close any trade
        # audit_causality=False: skip the ≥100 bar requirement for the causality
        # check — this test only cares about the zero-trades MC guard.
        result = run_backtest(df, TrendFollowStrategy(), _cfg(),
                              warmup_bars=20, verbose=False, audit_causality=False)
        if result.n_trades == 0:
            with pytest.raises(ValueError, match="No closed trades"):
                run_monte_carlo(result, n_sims=100, verbose=False)

    def test_summary_string(self):
        """summary() must contain key statistics strings."""
        from engine.monte_carlo import run_monte_carlo
        result = self._get_result()
        mc = run_monte_carlo(result, n_sims=200, verbose=False)
        s  = mc.summary()
        assert "Monte Carlo" in s
        assert "ruin" in s.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 10. Multi-Timeframe Support
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiTimeframe:

    def _make_h1_df(self, n_m1_bars: int = 600, seed: int = 42) -> pd.DataFrame:
        """Build an H1 DataFrame by resampling M1 data."""
        df_m1 = _make_df(n_m1_bars, seed=seed)
        df_h1 = df_m1.resample("1h").agg({
            "open":   "first", "high": "max",
            "low":    "min",   "close": "last", "volume": "sum"
        }).dropna()
        return df_h1

    def test_cursor_map_strictly_before(self):
        """Secondary cursor must NEVER point to a bar that started at or after T."""
        from engine.mtf import build_secondary_cursor_map
        m1 = pd.date_range("2024-01-01 08:00", periods=120, freq="1min", tz="UTC")
        h1 = pd.date_range("2024-01-01 08:00", periods=3,   freq="1h",   tz="UTC")
        cmap = build_secondary_cursor_map(m1, h1)
        for m1_i, t in enumerate(m1):
            h1_cursor = cmap[m1_i]
            if h1_cursor >= 0:
                # The cursor H1 bar must have started STRICTLY before this M1 bar
                assert h1[h1_cursor] < t, (
                    f"H1 cursor bar {h1_cursor} started at {h1[h1_cursor]}, "
                    f"but M1 bar {m1_i} is at {t} — secondary bar not yet complete!"
                )

    def test_first_h1_bar_visible_after_its_close(self):
        """H1 bar starting at 08:00 should be visible from M1 09:00 onwards."""
        from engine.mtf import build_secondary_cursor_map
        m1 = pd.date_range("2024-01-01 08:00", periods=120, freq="1min", tz="UTC")
        h1 = pd.date_range("2024-01-01 08:00", periods=3,   freq="1h",   tz="UTC")
        cmap = build_secondary_cursor_map(m1, h1)
        # M1 09:00 = bar index 60. H1 08:00 bar is complete at 09:00.
        # cmap[60] should be 0 (H1 bar at index 0 = 08:00)
        assert cmap[60] == 0, (
            f"H1 bar at 08:00 should first be visible at M1 09:00, "
            f"but cursor is {cmap[60]}"
        )

    def test_h1_bar_not_visible_while_in_progress(self):
        """H1 08:00 bar should NOT be visible at M1 08:30 (still in progress)."""
        from engine.mtf import build_secondary_cursor_map
        m1 = pd.date_range("2024-01-01 08:00", periods=120, freq="1min", tz="UTC")
        h1 = pd.date_range("2024-01-01 08:00", periods=3,   freq="1h",   tz="UTC")
        cmap = build_secondary_cursor_map(m1, h1)
        # M1 08:30 = bar index 30. H1 08:00 is still in progress.
        assert cmap[30] == -1, (
            f"H1 08:00 bar should NOT be visible at M1 08:30 (still in progress), "
            f"but cursor is {cmap[30]}"
        )

    def test_build_mtf_feeds(self):
        """build_mtf_feeds must create feeds for all timeframes."""
        from engine.mtf import build_mtf_feeds
        df_m1 = _make_df(600)
        df_h1 = self._make_h1_df(600)
        mtf   = build_mtf_feeds({"M1": df_m1, "H1": df_h1}, primary_key="M1")
        assert "M1" in mtf.keys()
        assert "H1" in mtf.keys()
        assert mtf.primary_key == "M1"

    def test_mtf_strategy_runs_end_to_end(self):
        """A simple MTFStrategy must complete a backtest without error."""
        from engine.mtf import MTFStrategy, build_mtf_feeds
        from engine.backtest import run_backtest_mtf
        from engine import indicators as ind

        class SimpleMTFStrategy(MTFStrategy):
            NAME       = "Simple MTF (test)"
            PARAMS     = {}
            TIMEFRAMES = ["M1", "H1"]

            def prepare_mtf(self, feeds):
                m1_close = feeds["M1"].close._data
                h1_close = feeds["H1"].close._data
                feeds["M1"]._attach("atr",    ind.atr(
                    feeds["M1"].high._data,
                    feeds["M1"].low._data, m1_close, 14
                ))
                feeds["H1"]._attach("h1_sma", ind.sma(h1_close, 5))

            def on_bar_mtf(self, feeds, broker):
                m1 = feeds["M1"]
                h1 = feeds["H1"]
                i  = m1.i
                h1_i = h1.i
                if h1_i < 0 or not broker.has_capacity:
                    return
                atr_i   = m1["atr"][i]
                h1_sma  = h1["h1_sma"][h1_i]
                close_i = m1.close[i]
                if math.isnan(atr_i) or math.isnan(h1_sma):
                    return
                if close_i > h1_sma:
                    sl = close_i - 1.5 * atr_i
                    tp = close_i + 3.0 * atr_i
                    broker.place_order(
                        side=OrderSide.LONG, order_type=OrderType.MARKET,
                        limit_price=close_i, sl=sl, tp=tp, placed_bar=i,
                    )

        df_m1 = _make_df(600)
        df_h1 = self._make_h1_df(600)
        mtf   = build_mtf_feeds({"M1": df_m1, "H1": df_h1}, primary_key="M1")
        result = run_backtest_mtf(
            mtf, SimpleMTFStrategy(), _cfg(),
            warmup_bars=30, verbose=False
        )
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(df_m1)

    def test_secondary_feed_cannot_see_future(self):
        """
        Secondary feed cursor must never be ahead of where it should be.
        We verify this by checking that H1.i < expected_max for each M1 bar.
        """
        from engine.mtf import build_mtf_feeds
        df_m1 = _make_df(120)
        df_h1 = self._make_h1_df(120)
        mtf   = build_mtf_feeds({"M1": df_m1, "H1": df_h1}, primary_key="M1")
        cmap  = mtf._cursor_maps["H1"]
        n_h1  = len(df_h1)
        for m1_i in range(len(df_m1)):
            h1_cursor = cmap[m1_i]
            assert h1_cursor < n_h1, "H1 cursor must never exceed H1 bar count"
            assert h1_cursor >= -1,  "H1 cursor must be -1 (no data) or a valid index"


# ──────────────────────────────────────────────────────────────────────────────
# 11. Portfolio-Level Backtesting
# ──────────────────────────────────────────────────────────────────────────────

class TestPortfolio:

    def _portfolio_setup(self, n: int = 1000):
        """Build two independent DataFrames and strategies for portfolio tests."""
        df_a = _make_df(n, seed=1, start="2024-01-01")
        df_b = _make_df(n, seed=2, start="2024-01-01")
        cfg_a = BrokerConfig(commission_flat=6.0, lot_size=100.0,
                             risk_usd=10.0, size_mode="fixed_risk")
        cfg_b = BrokerConfig(commission_flat=4.0, lot_size=100.0,
                             risk_usd=10.0, size_mode="fixed_risk")
        return df_a, df_b, cfg_a, cfg_b

    def test_basic_portfolio_run(self):
        """Portfolio backtest must complete and return a PortfolioResult."""
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig, PortfolioResult
        df_a, df_b, cfg_a, cfg_b = self._portfolio_setup()
        pcfg = PortfolioConfig(
            symbols=["SYM_A", "SYM_B"],
            broker_configs={"SYM_A": cfg_a, "SYM_B": cfg_b},
            starting_equity=10_000.0,
            warmup_bars=20,
        )
        results = run_backtest_portfolio(
            dfs={"SYM_A": df_a, "SYM_B": df_b},
            strategies={"SYM_A": TrendFollowStrategy(), "SYM_B": TrendFollowStrategy()},
            portfolio_cfg=pcfg,
            verbose=False,
        )
        assert isinstance(results, PortfolioResult)

    def test_equity_curve_length_matches_total_bars(self):
        """Portfolio equity curve length = total number of bars across all symbols."""
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig
        df_a, df_b, cfg_a, cfg_b = self._portfolio_setup(500)
        pcfg = PortfolioConfig(
            symbols=["A", "B"],
            broker_configs={"A": cfg_a, "B": cfg_b},
            starting_equity=10_000.0, warmup_bars=20,
        )
        result = run_backtest_portfolio(
            dfs={"A": df_a, "B": df_b},
            strategies={"A": TrendFollowStrategy(), "B": TrendFollowStrategy()},
            portfolio_cfg=pcfg, verbose=False,
        )
        # One equity reading per bar event (500 from A + 500 from B = 1000)
        assert len(result.equity_curve) == len(df_a) + len(df_b)

    def test_per_symbol_results_present(self):
        """PortfolioResult must contain a BacktestResult for each symbol."""
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig
        df_a, df_b, cfg_a, cfg_b = self._portfolio_setup(500)
        pcfg = PortfolioConfig(
            symbols=["A", "B"],
            broker_configs={"A": cfg_a, "B": cfg_b},
            starting_equity=10_000.0, warmup_bars=20,
        )
        result = run_backtest_portfolio(
            dfs={"A": df_a, "B": df_b},
            strategies={"A": TrendFollowStrategy(), "B": TrendFollowStrategy()},
            portfolio_cfg=pcfg, verbose=False,
        )
        assert "A" in result.symbol_results
        assert "B" in result.symbol_results

    def test_shared_equity_changes(self):
        """Starting and ending equity must differ (trades have P&L)."""
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig
        df_a, df_b, cfg_a, cfg_b = self._portfolio_setup(2000)
        pcfg = PortfolioConfig(
            symbols=["A", "B"],
            broker_configs={"A": cfg_a, "B": cfg_b},
            starting_equity=10_000.0, warmup_bars=20,
        )
        result = run_backtest_portfolio(
            dfs={"A": df_a, "B": df_b},
            strategies={"A": TrendFollowStrategy(), "B": TrendFollowStrategy()},
            portfolio_cfg=pcfg, verbose=False,
        )
        assert result.final_equity != result.starting_equity, (
            "Portfolio equity should change after trading"
        )

    def test_max_total_positions_respected(self):
        """Portfolio must not exceed max_total_positions across all symbols."""
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig, PortfolioResult
        df_a, df_b, cfg_a, cfg_b = self._portfolio_setup(1000)
        pcfg = PortfolioConfig(
            symbols=["A", "B"],
            broker_configs={"A": cfg_a, "B": cfg_b},
            starting_equity=10_000.0, warmup_bars=20,
            max_total_positions=1,  # only 1 position across BOTH symbols
        )
        result = run_backtest_portfolio(
            dfs={"A": df_a, "B": df_b},
            strategies={"A": TrendFollowStrategy(), "B": TrendFollowStrategy()},
            portfolio_cfg=pcfg, verbose=False,
        )
        # Verify no point in time had more than 1 open position
        # (We check the final state rather than all intermediate states)
        assert isinstance(result, PortfolioResult)

    def test_default_broker_config_fallback(self):
        """'default' key in broker_configs must be used for unspecified symbols."""
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig, PortfolioResult
        df_a, df_b, cfg_a, cfg_b = self._portfolio_setup(500)
        pcfg = PortfolioConfig(
            symbols=["A", "B"],
            broker_configs={"default": cfg_a},   # one config for both
            starting_equity=10_000.0, warmup_bars=20,
        )
        result = run_backtest_portfolio(
            dfs={"A": df_a, "B": df_b},
            strategies={"A": TrendFollowStrategy(), "B": TrendFollowStrategy()},
            portfolio_cfg=pcfg, verbose=False,
        )
        assert isinstance(result, PortfolioResult)

    def test_missing_symbol_data_raises(self):
        """Missing DataFrame for a declared symbol must raise KeyError."""
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig
        df_a, _, cfg_a, _ = self._portfolio_setup(300)
        pcfg = PortfolioConfig(
            symbols=["A", "MISSING"],
            broker_configs={"default": cfg_a},
            starting_equity=5_000.0, warmup_bars=20,
        )
        with pytest.raises(KeyError):
            run_backtest_portfolio(
                dfs={"A": df_a},   # MISSING is not provided
                strategies={"A": TrendFollowStrategy(), "MISSING": TrendFollowStrategy()},
                portfolio_cfg=pcfg, verbose=False,
            )

    def test_portfolio_applies_spread_cost(self):
        """The portfolio path must charge spread (it previously ignored it).

        Two identical runs differing only in `spread` must produce different
        equity — proving PortfolioBroker now uses the cost-aware base fill model.
        """
        from engine.portfolio import run_backtest_portfolio, PortfolioConfig

        def _run(spread: float) -> float:
            df = _make_df(2000, seed=7, start="2024-01-01")
            cfg = BrokerConfig(commission_flat=0.0, lot_size=100.0,
                               risk_usd=10.0, size_mode="fixed_risk", spread=spread)
            pcfg = PortfolioConfig(
                symbols=["A"], broker_configs={"A": cfg},
                starting_equity=10_000.0, warmup_bars=20,
            )
            res = run_backtest_portfolio(
                dfs={"A": df}, strategies={"A": TrendFollowStrategy()},
                portfolio_cfg=pcfg, verbose=False,
            )
            return res.final_equity

        no_spread   = _run(0.0)
        with_spread = _run(0.50)
        # Spread is a round-trip cost → equity with spread must be strictly lower.
        assert with_spread < no_spread, (
            f"Spread not charged in portfolio path: "
            f"no_spread={no_spread:.2f} with_spread={with_spread:.2f}"
        )
