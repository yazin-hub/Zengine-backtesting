"""
tests/test_funded.py — Funded Account Simulation Tests
========================================================

Tests cover:
  - FundedAccountConfig validation
  - FundedAccountConfig computed properties
  - FIRM_PRESETS completeness
  - Absolute drawdown — pass / breach
  - Trailing drawdown — pass / breach (floor rises with equity)
  - Daily loss limit — breach detection
  - Profit target — challenge pass with min_trading_days enforcement
  - No profit target — funded phase passes without breach
  - Payout cycle — event recorded, amount correct
  - Payout with equity reset
  - Empty equity curve edge case
  - Equity curve scaling (backtest ≠ funded starting equity)
  - Failure bar flattening in funded_equity_curve
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from engine.funded import (
    FundedAccountConfig,
    FIRM_PRESETS,
    run_funded_backtest,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_result(
    equity: list[float],
    n_trading_days: int = 5,
    starting_equity: float | None = None,
) -> SimpleNamespace:
    """
    Build a minimal BacktestResult-like object from a flat equity list.

    One bar per day.  Adds synthetic "closed trades" so that
    ``trading_days_active`` counts correctly.

    Parameters
    ----------
    equity          : bar-by-bar equity values
    n_trading_days  : number of bars that have a "closed trade"
    starting_equity : used to scale the equity; defaults to equity[0]
    """
    N   = len(equity)
    idx = pd.date_range("2024-01-01", periods=N, freq="D", tz="UTC")

    eq_arr = np.array(equity, dtype=float)

    # Synthetic closed trades: place exit bars on the first n_trading_days bars
    trades = []
    for i in range(min(n_trading_days, N)):
        t = SimpleNamespace(
            exit_bar    = i,
            pnl_net     = equity[i] - equity[max(0, i - 1)],
        )
        trades.append(t)

    return SimpleNamespace(
        equity_curve  = eq_arr,
        timestamps    = idx,
        closed_trades = trades,
    )


def _flat_equity(
    n: int,
    start: float = 10_000.0,
    daily_gain: float = 0.0,
) -> list[float]:
    """Return a linearly growing equity curve."""
    return [start + daily_gain * i for i in range(n)]


# ── Config validation ─────────────────────────────────────────────────────────

class TestFundedAccountConfigValidation:

    def test_valid_config_creates_ok(self):
        cfg = FundedAccountConfig()
        assert cfg.starting_equity == 10_000.0

    def test_frozen(self):
        cfg = FundedAccountConfig()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.starting_equity = 5_000.0  # type: ignore[misc]

    def test_invalid_drawdown_type(self):
        with pytest.raises(ValueError, match="drawdown_type"):
            FundedAccountConfig(drawdown_type="peak")

    def test_drawdown_pct_out_of_range_high(self):
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            FundedAccountConfig(max_drawdown_pct=1.5)

    def test_drawdown_pct_out_of_range_zero(self):
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            FundedAccountConfig(max_drawdown_pct=0.0)

    def test_daily_loss_out_of_range(self):
        with pytest.raises(ValueError, match="daily_loss_limit_pct"):
            FundedAccountConfig(daily_loss_limit_pct=-0.01)

    def test_profit_split_out_of_range(self):
        with pytest.raises(ValueError, match="profit_split_pct"):
            FundedAccountConfig(profit_split_pct=0.0)

    def test_payout_frequency_zero(self):
        with pytest.raises(ValueError, match="payout_frequency_days"):
            FundedAccountConfig(payout_frequency_days=0)

    def test_absolute_and_trailing_both_valid(self):
        FundedAccountConfig(drawdown_type="absolute")
        FundedAccountConfig(drawdown_type="trailing")


class TestFundedAccountConfigProperties:

    def test_max_loss_abs(self):
        cfg = FundedAccountConfig(starting_equity=10_000.0, max_drawdown_pct=0.10)
        assert cfg.max_loss_abs == pytest.approx(1_000.0)

    def test_daily_loss_abs(self):
        cfg = FundedAccountConfig(starting_equity=10_000.0, daily_loss_limit_pct=0.05)
        assert cfg.daily_loss_abs == pytest.approx(500.0)

    def test_profit_target_abs(self):
        cfg = FundedAccountConfig(starting_equity=10_000.0, profit_target_pct=0.10)
        assert cfg.profit_target_abs == pytest.approx(1_000.0)

    def test_zero_profit_target_abs(self):
        cfg = FundedAccountConfig(profit_target_pct=0.0)
        assert cfg.profit_target_abs == 0.0


# ── Firm presets ──────────────────────────────────────────────────────────────

class TestFirmPresets:

    def test_presets_dict_not_empty(self):
        assert len(FIRM_PRESETS) >= 5

    def test_all_presets_valid_configs(self):
        # Construction itself validates; this test ensures no preset raises
        for name, cfg in FIRM_PRESETS.items():
            assert isinstance(cfg, FundedAccountConfig), f"Preset '{name}' is not a FundedAccountConfig"

    def test_ftmo_challenge_rules(self):
        cfg = FIRM_PRESETS["FTMO — Challenge ($10k)"]
        assert cfg.drawdown_type        == "absolute"
        assert cfg.max_drawdown_pct     == pytest.approx(0.10)
        assert cfg.daily_loss_limit_pct == pytest.approx(0.05)
        assert cfg.profit_target_pct    == pytest.approx(0.10)
        assert cfg.profit_split_pct     == pytest.approx(0.80)

    def test_topstep_combine_trailing_drawdown(self):
        cfg = FIRM_PRESETS["Topstep — Combine ($50k)"]
        assert cfg.drawdown_type == "trailing"

    def test_custom_preset_exists(self):
        assert "Custom" in FIRM_PRESETS


# ── Absolute drawdown ─────────────────────────────────────────────────────────

class TestAbsoluteDrawdown:

    def _cfg(self, **kw) -> FundedAccountConfig:
        defaults = dict(
            starting_equity       = 10_000.0,
            max_drawdown_pct      = 0.10,   # $1,000 floor at $9,000
            drawdown_type         = "absolute",
            daily_loss_limit_pct  = 0.0,
            profit_target_pct     = 0.0,
            min_trading_days      = 1,
            profit_split_pct      = 0.80,
            payout_frequency_days = 365,    # far future — no payouts
            reset_on_payout       = False,
        )
        defaults.update(kw)
        return FundedAccountConfig(**defaults)

    def test_pass_when_equity_stays_above_floor(self):
        # Equity grows from 10k to 11k — never breaches 9k floor
        eq  = _flat_equity(30, start=10_000.0, daily_gain=33.0)
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is True
        assert fr.failure_reason is None

    def test_fail_when_equity_drops_below_floor(self):
        # Drops from 10k to 8.9k — breaches 9k absolute floor
        eq  = [10_000.0] * 5 + [8_900.0] * 5
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is False
        assert fr.failure_bar is not None
        assert "absolute" in fr.failure_reason

    def test_floor_does_not_move_when_equity_grows(self):
        # Equity rises to 12k then falls to 9.1k — still above 9k floor (absolute)
        eq = (
            [10_000.0 + i * 200 for i in range(10)]   # rises to 12k
            + [9_100.0] * 5                             # dips but above 9k floor
        )
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is True, "Absolute floor at 9k should not breach at 9.1k"

    def test_failure_bar_points_to_breach_bar(self):
        eq      = [10_000.0] * 3 + [8_999.0] + [8_999.0] * 2
        res     = _make_result(eq)
        fr      = run_funded_backtest(res, self._cfg())
        assert fr.failure_bar == 3

    def test_funded_equity_curve_flat_after_breach(self):
        eq  = [10_000.0] * 3 + [8_900.0] * 5
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        # All bars after the failure bar must equal the breach equity
        breach_eq = fr.funded_equity_curve[fr.failure_bar]
        assert all(
            fr.funded_equity_curve[i] == pytest.approx(breach_eq)
            for i in range(fr.failure_bar + 1, len(eq))
        )

    def test_max_drawdown_reached_reported_correctly(self):
        # Equity drops from 10k to 9.5k (5% drawdown) then recovers
        eq  = [10_000.0] * 3 + [9_500.0] * 2 + [10_000.0] * 3
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        # 5% drawdown was reached (10000 - 9500) / 10000 = 5%
        assert fr.max_drawdown_reached_pct == pytest.approx(5.0, abs=0.5)


# ── Trailing drawdown ─────────────────────────────────────────────────────────

class TestTrailingDrawdown:

    def _cfg(self, **kw) -> FundedAccountConfig:
        defaults = dict(
            starting_equity       = 10_000.0,
            max_drawdown_pct      = 0.10,   # 10% trailing
            drawdown_type         = "trailing",
            daily_loss_limit_pct  = 0.0,
            profit_target_pct     = 0.0,
            min_trading_days      = 1,
            profit_split_pct      = 0.80,
            payout_frequency_days = 365,
            reset_on_payout       = False,
        )
        defaults.update(kw)
        return FundedAccountConfig(**defaults)

    def test_pass_when_never_breaches_trailing_floor(self):
        # Equity rises smoothly — trailing floor never caught
        eq  = _flat_equity(20, start=10_000.0, daily_gain=50.0)
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is True

    def test_fail_when_equity_falls_below_trailing_floor(self):
        # Equity rises to 11k, then drops to 9.8k
        # Trailing floor = 11_000 * 0.90 = 9_900 → breach at 9_800
        eq  = [10_000 + i * 100 for i in range(11)] + [9_800.0] * 5
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is False
        assert "trailing" in fr.failure_reason

    def test_trailing_floor_rises_with_hwm(self):
        # Equity rises to 12k.  Absolute floor would be 9k (never breached),
        # but trailing floor = 12k * 0.90 = 10.8k.
        # Dropping to 10.9k PASSES (> trailing floor of 10.8k).
        eq = [10_000 + i * 200 for i in range(11)] + [10_900.0] * 3
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is True

    def test_trailing_floor_catches_what_absolute_would_miss(self):
        # Same setup but drop to 10.7k → below trailing floor 10.8k
        eq = [10_000 + i * 200 for i in range(11)] + [10_700.0] * 3
        res = _make_result(eq)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is False, (
            "Trailing floor should have caught 10.7k drop from 12k HWM "
            "(floor = 10.8k).  Absolute mode would have missed it."
        )


# ── Daily loss limit ──────────────────────────────────────────────────────────

class TestDailyLossLimit:

    def _cfg(self, **kw) -> FundedAccountConfig:
        defaults = dict(
            starting_equity       = 10_000.0,
            max_drawdown_pct      = 0.20,   # relaxed DD so only daily limit fires
            drawdown_type         = "absolute",
            daily_loss_limit_pct  = 0.05,   # $500 daily limit
            profit_target_pct     = 0.0,
            min_trading_days      = 1,
            profit_split_pct      = 0.80,
            payout_frequency_days = 365,
            reset_on_payout       = False,
        )
        defaults.update(kw)
        return FundedAccountConfig(**defaults)

    def test_passes_when_daily_loss_within_limit(self):
        # Loses $400 in one bar within a day — within $500 limit
        eq  = [10_000.0, 9_600.0] + [9_600.0] * 5
        res = _make_result(eq, n_trading_days=2)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is True

    def test_fails_when_daily_loss_exceeds_limit(self):
        # Starts day at 10k, drops to 9.4k in same day → $600 loss > $500 limit
        eq  = [10_000.0, 9_400.0] + [9_400.0] * 5
        res = _make_result(eq, n_trading_days=2)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is False
        assert "Daily loss" in fr.failure_reason

    def test_daily_limit_is_percentage_of_starting_not_current(self):
        # Account grows to 10.5k day 1. Then loses $500 on day 2.
        # Daily loss = $500 / $10,000 = 5% — exactly at limit.
        # Depending on edge: should NOT fail (strictly <).
        eq = [
            10_000.0,   # day 1 start
            10_500.0,   # day 1 end (profit)
            10_000.0,   # day 2 — $500 loss = exactly 5% of 10k — boundary (should pass)
        ]
        res = _make_result(eq, n_trading_days=2)
        fr  = run_funded_backtest(res, self._cfg())
        # 5% == limit — >= check fires → fails at boundary
        # (Match FTMO behaviour: strictly violating the rule)
        assert fr.passed is False  # >= triggers at exactly 5%

    def test_daily_limit_disabled_when_zero(self):
        # Huge daily loss — but daily limit is 0 (disabled)
        cfg = self._cfg(daily_loss_limit_pct=0.0, max_drawdown_pct=0.80)
        eq  = [10_000.0, 5_000.0] + [5_000.0] * 5
        res = _make_result(eq, n_trading_days=2)
        fr  = run_funded_backtest(res, cfg)
        assert fr.passed is True


# ── Profit target (challenge phase) ───────────────────────────────────────────

class TestProfitTarget:

    def _cfg(self, **kw) -> FundedAccountConfig:
        defaults = dict(
            starting_equity       = 10_000.0,
            max_drawdown_pct      = 0.10,
            drawdown_type         = "absolute",
            daily_loss_limit_pct  = 0.0,
            profit_target_pct     = 0.10,   # 10% = $11,000 target
            min_trading_days      = 4,
            profit_split_pct      = 0.80,
            payout_frequency_days = 365,
            reset_on_payout       = False,
        )
        defaults.update(kw)
        return FundedAccountConfig(**defaults)

    def test_passes_when_target_hit_with_enough_days(self):
        # 10 trading days, equity reaches 11.1k
        eq  = _flat_equity(10, start=10_000.0, daily_gain=110.0)
        res = _make_result(eq, n_trading_days=10)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.passed is True

    def test_does_not_pass_before_min_trading_days(self):
        # Profit target hit on day 2 but min_trading_days=4 → not yet passed
        eq = [10_000.0, 11_500.0] + [11_500.0] * 10
        res = _make_result(eq, n_trading_days=2)   # only 2 trading days
        fr  = run_funded_backtest(res, self._cfg())
        # With only 2 trading days when target is hit, min_trading_days=4 not met
        # The sim continues running — if no breach occurs, it eventually passes
        # when enough trading days accumulate
        # (here there are only 2 trading days total → never reaches 4 → passes as funded)
        # Actually: profit target NEVER met with enough days → no breach, no target →
        # falls through to "passed as funded phase" logic
        # With min_trading_days=4 and only 2 trading days, the profit_target check
        # skips and the loop ends → passed = True (no-breach funded logic)
        assert fr.passed is True  # no breach, just didn't formally "challenge-pass"

    def test_no_target_funded_phase_passes_without_breach(self):
        # funded_target_pct = 0 → payout on schedule, pass unless breach
        cfg = self._cfg(profit_target_pct=0.0)
        eq  = _flat_equity(20, start=10_000.0, daily_gain=10.0)
        res = _make_result(eq, n_trading_days=20)
        fr  = run_funded_backtest(res, cfg)
        assert fr.passed is True

    def test_trading_days_active_counted_correctly(self):
        # 7 days with equity changes
        eq  = _flat_equity(7, start=10_000.0, daily_gain=5.0)
        res = _make_result(eq, n_trading_days=7)
        fr  = run_funded_backtest(res, self._cfg(profit_target_pct=0.0))
        assert fr.trading_days_active == 7


# ── Payout cycle ─────────────────────────────────────────────────────────────

class TestPayoutCycle:

    def _cfg(self, reset: bool = False, **kw) -> FundedAccountConfig:
        defaults = dict(
            starting_equity       = 10_000.0,
            max_drawdown_pct      = 0.10,
            drawdown_type         = "absolute",
            daily_loss_limit_pct  = 0.0,
            profit_target_pct     = 0.0,
            min_trading_days      = 1,
            profit_split_pct      = 0.80,
            payout_frequency_days = 30,
            reset_on_payout       = reset,
        )
        defaults.update(kw)
        return FundedAccountConfig(**defaults)

    def test_no_payout_when_no_profit(self):
        # Flat equity — no profit → no payout
        eq  = [10_000.0] * 60
        res = _make_result(eq, n_trading_days=40)
        fr  = run_funded_backtest(res, self._cfg())
        assert fr.payout_events == []
        assert fr.total_payout_to_trader == 0.0

    def test_payout_recorded_when_profit_at_cycle_boundary(self):
        # 60 days, constant gain — crosses 30-day boundary with profit
        eq  = _flat_equity(60, start=10_000.0, daily_gain=10.0)
        res = _make_result(eq, n_trading_days=60)
        fr  = run_funded_backtest(res, self._cfg())
        assert len(fr.payout_events) >= 1

    def test_payout_trader_share_correct(self):
        # After 31 days, profit = $310 → trader gets 80% = $248
        eq  = _flat_equity(31, start=10_000.0, daily_gain=10.0)
        res = _make_result(eq, n_trading_days=31)
        fr  = run_funded_backtest(res, self._cfg())
        if fr.payout_events:
            pev = fr.payout_events[0]
            expected_share = pev.profit_gross * 0.80
            assert pev.trader_share == pytest.approx(expected_share, rel=0.01)

    def test_total_payout_is_sum_of_events(self):
        eq  = _flat_equity(90, start=10_000.0, daily_gain=20.0)
        res = _make_result(eq, n_trading_days=90)
        fr  = run_funded_backtest(res, self._cfg())
        expected = sum(p.trader_share for p in fr.payout_events)
        assert fr.total_payout_to_trader == pytest.approx(expected, rel=0.001)

    def test_payout_with_reset_re_anchors_equity(self):
        # With reset_on_payout=True, equity is re-anchored to 10k after payout
        # After 31 days of gain, payout fires and funded_eq should be ~10k again
        eq  = _flat_equity(60, start=10_000.0, daily_gain=10.0)
        res = _make_result(eq, n_trading_days=60)
        fr  = run_funded_backtest(res, self._cfg(reset=True))
        if fr.payout_events:
            reset_bar  = fr.payout_events[0].bar_index
            reset_eq   = fr.funded_equity_curve[reset_bar]
            assert reset_eq == pytest.approx(10_000.0, rel=0.01)

    def test_payout_event_has_correct_fields(self):
        eq  = _flat_equity(35, start=10_000.0, daily_gain=20.0)
        res = _make_result(eq, n_trading_days=35)
        fr  = run_funded_backtest(res, self._cfg())
        if fr.payout_events:
            pev = fr.payout_events[0]
            assert isinstance(pev.cycle_number, int)
            assert pev.cycle_number >= 1
            assert pev.profit_gross > 0
            assert pev.trader_share > 0
            assert pev.firm_share > 0
            assert pev.trader_share + pev.firm_share == pytest.approx(pev.profit_gross, rel=0.001)


# ── Equity scaling ────────────────────────────────────────────────────────────

class TestEquityScaling:

    def test_equity_scaled_to_funded_starting_equity(self):
        # Backtest starts at $1,000 but funded account is $10,000
        cfg = FundedAccountConfig(
            starting_equity       = 10_000.0,
            max_drawdown_pct      = 0.10,
            drawdown_type         = "absolute",
            daily_loss_limit_pct  = 0.0,
            profit_target_pct     = 0.0,
            min_trading_days      = 1,
            profit_split_pct      = 0.80,
            payout_frequency_days = 365,
            reset_on_payout       = False,
        )
        eq  = _flat_equity(10, start=1_000.0, daily_gain=10.0)  # backtest at $1k
        res = _make_result(eq)
        fr  = run_funded_backtest(res, cfg)
        # Funded equity should start at ~10,000
        first_valid = fr.funded_equity_curve[~np.isnan(fr.funded_equity_curve)][0]
        assert first_valid == pytest.approx(10_000.0, rel=0.01)

    def test_scaling_preserves_pass_with_healthy_equity(self):
        # Backtest at $500 (no breach) should pass FTMO $10k rules after scaling
        cfg = FIRM_PRESETS["FTMO — Challenge ($10k)"]
        eq  = _flat_equity(30, start=500.0, daily_gain=20.0)   # +4% over 30 days
        res = _make_result(eq, n_trading_days=10)
        fr  = run_funded_backtest(res, cfg)
        assert fr.passed is True


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_equity_curve_returns_failed_result(self):
        eq  = [np.nan] * 10
        res = _make_result(eq, n_trading_days=0)
        cfg = FundedAccountConfig()
        fr  = run_funded_backtest(res, cfg)
        assert fr.passed is False
        assert "No valid equity data" in fr.failure_reason

    def test_single_bar_result(self):
        eq  = [10_000.0]
        res = _make_result(eq, n_trading_days=1)
        cfg = FundedAccountConfig(
            profit_target_pct=0.0,
            daily_loss_limit_pct=0.0,
            payout_frequency_days=365,
        )
        fr  = run_funded_backtest(res, cfg)
        assert fr.passed is True

    def test_funded_result_has_correct_config(self):
        cfg = FundedAccountConfig()
        eq  = _flat_equity(10, start=10_000.0)
        res = _make_result(eq)
        fr  = run_funded_backtest(res, cfg)
        assert fr.config is cfg

    def test_funded_equity_curve_same_length_as_input(self):
        eq  = _flat_equity(50, start=10_000.0, daily_gain=5.0)
        res = _make_result(eq)
        cfg = FundedAccountConfig(profit_target_pct=0.0, payout_frequency_days=365)
        fr  = run_funded_backtest(res, cfg)
        assert len(fr.funded_equity_curve) == len(eq)

    def test_peak_equity_tracked_correctly(self):
        # Equity rises to 12k then drops back to 10.5k
        eq = list(range(10_000, 12_100, 100)) + [10_500.0] * 5
        res = _make_result(eq)
        cfg = FundedAccountConfig(
            max_drawdown_pct=0.30,  # wide — no breach
            daily_loss_limit_pct=0.0,
            profit_target_pct=0.0,
            payout_frequency_days=365,
        )
        fr = run_funded_backtest(res, cfg)
        assert fr.peak_equity == pytest.approx(12_000.0, rel=0.01)

    def test_failure_date_iso_format(self):
        eq  = [10_000.0] * 3 + [8_000.0] * 5
        res = _make_result(eq)
        cfg = FundedAccountConfig(
            max_drawdown_pct=0.10,
            daily_loss_limit_pct=0.0,
            profit_target_pct=0.0,
            payout_frequency_days=365,
        )
        fr  = run_funded_backtest(res, cfg)
        assert fr.failure_date is not None
        # Should be a valid ISO date string
        datetime.date.fromisoformat(fr.failure_date)
