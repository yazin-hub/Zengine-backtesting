"""
engine/funded.py — Funded Account Simulation
=============================================

Wraps a standard BacktestResult to apply proprietary trading firm rules
bar-by-bar on the equity curve.  The underlying strategy trades freely;
this module only evaluates *whether* the equity curve would have survived.

Rules supported
---------------
1. **Daily loss limit**
   No single trading day may lose more than ``daily_loss_limit_pct`` of the
   initial account balance (FTMO/Topstep convention).
   Formula: ``(equity_at_day_start - equity_now) / starting_equity``

2. **Max drawdown — two modes**

   ``"absolute"``  (FTMO-style)
     The floor is fixed at ``starting_equity × (1 − max_drawdown_pct)``.
     It never moves up, even if the account grows.  Breaching it fails the
     account permanently.

   ``"trailing"``  (Topstep-style)
     The floor tracks the all-time high-water mark (HWM):
     ``floor = hwm × (1 − max_drawdown_pct)``.
     As equity rises the floor rises with it; it never comes back down.

3. **Profit target** (challenge phase)
   When equity reaches ``starting_equity × (1 + profit_target_pct)`` AND
   at least ``min_trading_days`` distinct calendar days had closed trades,
   the challenge is *passed*.  Set ``profit_target_pct = 0.0`` for funded
   phases that have no target (payout on schedule instead).

4. **Payout cycle** (funded phase)
   Every ``payout_frequency_days`` calendar days the accrued profit is paid
   out at ``profit_split_pct``.
   If ``reset_on_payout=True`` the equity curve is re-anchored to
   ``starting_equity`` (rare — some scale-up programmes do this).

References
----------
  - FTMO Trading Objectives  : https://ftmo.com/en/trading-objectives/
  - Topstep Rules            : https://www.topstep.com/how-it-works/
  - The5ers Programmes       : https://the5ers.com/programs/
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import numpy as np

# TYPE_CHECKING-only import avoids circular dependency at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .backtest import BacktestResult


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FundedAccountConfig:
    """
    Complete specification for one phase of a funded-account programme.

    Parameters
    ----------
    starting_equity : float
        Account balance at the start of the simulation ($).
        All percentage-based limits are calculated relative to this value.
    daily_loss_limit_pct : float
        Maximum permitted loss in a single calendar day as a fraction of
        *starting_equity* (e.g. 0.05 = 5 %).  Set 0.0 to disable.
    max_drawdown_pct : float
        Absolute peak-to-trough drawdown limit (e.g. 0.10 = 10 %).
        See ``drawdown_type`` for how the reference peak is defined.
    drawdown_type : str
        ``"absolute"``  — reference peak is always ``starting_equity``
                          (floor is fixed regardless of growth).
        ``"trailing"``  — reference peak is the running equity HWM
                          (floor rises with equity, never falls back).
    profit_target_pct : float
        Challenge is passed when equity exceeds
        ``starting_equity * (1 + profit_target_pct)``.
        Set 0.0 for funded phases that use payout scheduling instead.
    min_trading_days : int
        Minimum calendar days with at least one closed trade required
        before the profit target counts as achieved.
    profit_split_pct : float
        Trader's share of profit on each payout (e.g. 0.80 = 80 %).
    payout_frequency_days : int
        Calendar days between consecutive payout evaluations.
    reset_on_payout : bool
        If True, equity is re-anchored to ``starting_equity`` after each
        payout and the HWM resets accordingly.  Rare in practice.
    """
    starting_equity:       float = 10_000.0
    daily_loss_limit_pct:  float = 0.05
    max_drawdown_pct:      float = 0.10
    drawdown_type:         str   = "absolute"   # "absolute" | "trailing"
    profit_target_pct:     float = 0.10
    min_trading_days:      int   = 4
    profit_split_pct:      float = 0.80
    payout_frequency_days: int   = 30
    reset_on_payout:       bool  = False

    def __post_init__(self) -> None:
        if self.drawdown_type not in ("absolute", "trailing"):
            raise ValueError(
                f"drawdown_type must be 'absolute' or 'trailing', "
                f"got {self.drawdown_type!r}"
            )
        if not 0.0 < self.max_drawdown_pct < 1.0:
            raise ValueError("max_drawdown_pct must be in the open interval (0, 1)")
        if not 0.0 <= self.daily_loss_limit_pct < 1.0:
            raise ValueError("daily_loss_limit_pct must be in [0, 1)")
        if not 0.0 < self.profit_split_pct <= 1.0:
            raise ValueError("profit_split_pct must be in (0, 1]")
        if self.payout_frequency_days < 1:
            raise ValueError("payout_frequency_days must be ≥ 1")

    @property
    def max_loss_abs(self) -> float:
        """Absolute dollar value of the max drawdown limit."""
        return self.starting_equity * self.max_drawdown_pct

    @property
    def daily_loss_abs(self) -> float:
        """Absolute dollar value of the daily loss limit."""
        return self.starting_equity * self.daily_loss_limit_pct

    @property
    def profit_target_abs(self) -> float:
        """Absolute dollar value of the profit target."""
        return self.starting_equity * self.profit_target_pct


# ── Payout record ─────────────────────────────────────────────────────────────

@dataclass
class PayoutEvent:
    """Records one payout disbursement."""
    cycle_number:      int
    date:              str     # ISO-8601 calendar date
    bar_index:         int     # bar index in the equity curve
    equity_before:     float
    profit_gross:      float   # equity_before − cycle_start_equity
    trader_share:      float   # profit_gross × profit_split_pct
    firm_share:        float   # profit_gross × (1 − profit_split_pct)
    equity_after:      float   # equity after optional reset


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class FundedResult:
    """
    Funded-account overlay on a standard BacktestResult.

    The underlying ``result`` is untouched — it represents the strategy
    running freely.  This object adds the rule-replay results on top.
    """
    result:                   "BacktestResult"
    config:                   FundedAccountConfig

    # ── Pass / fail
    passed:                   bool
    failure_reason:           Optional[str]   # None if passed/funded
    failure_bar:              Optional[int]   # bar index of breach
    failure_date:             Optional[str]   # ISO date of breach

    # ── Progress metrics
    trading_days_active:      int    # distinct days with ≥1 closed trade
    max_drawdown_reached_pct: float  # worst drawdown % seen during sim
    peak_equity:              float  # highest equity reached

    # ── Payout history (funded phase)
    payout_events:            List[PayoutEvent]
    total_payout_to_trader:   float

    # ── Adjusted equity curve
    # Same length as result.equity_curve; scaled to config.starting_equity.
    # Bars after a failure are flattened to the breach equity value so the
    # chart shows a clear "fell off a cliff" moment.
    funded_equity_curve:      np.ndarray


# ── Firm presets ──────────────────────────────────────────────────────────────

# Approximate rules as of 2024–2025.  Verify current rules on each firm's
# website before treating these as authoritative benchmarks.
FIRM_PRESETS: dict[str, FundedAccountConfig] = {
    "FTMO — Challenge ($10k)": FundedAccountConfig(
        starting_equity       = 10_000.0,
        daily_loss_limit_pct  = 0.05,
        max_drawdown_pct      = 0.10,
        drawdown_type         = "absolute",
        profit_target_pct     = 0.10,
        min_trading_days      = 4,
        profit_split_pct      = 0.80,
        payout_frequency_days = 30,
        reset_on_payout       = False,
    ),
    "FTMO — Funded ($10k)": FundedAccountConfig(
        starting_equity       = 10_000.0,
        daily_loss_limit_pct  = 0.05,
        max_drawdown_pct      = 0.10,
        drawdown_type         = "absolute",
        profit_target_pct     = 0.0,          # no target — payout on schedule
        min_trading_days      = 1,
        profit_split_pct      = 0.80,
        payout_frequency_days = 30,
        reset_on_payout       = False,
    ),
    "Topstep — Combine ($50k)": FundedAccountConfig(
        starting_equity       = 50_000.0,
        daily_loss_limit_pct  = 0.03,         # $1,500 daily limit
        max_drawdown_pct      = 0.05,         # $2,500 trailing drawdown
        drawdown_type         = "trailing",
        profit_target_pct     = 0.06,         # $3,000 profit target
        min_trading_days      = 1,
        profit_split_pct      = 0.90,
        payout_frequency_days = 7,
        reset_on_payout       = False,
    ),
    "Topstep — Funded ($50k)": FundedAccountConfig(
        starting_equity       = 50_000.0,
        daily_loss_limit_pct  = 0.03,
        max_drawdown_pct      = 0.04,         # tighter trailing DD on funded
        drawdown_type         = "trailing",
        profit_target_pct     = 0.0,
        min_trading_days      = 1,
        profit_split_pct      = 0.90,
        payout_frequency_days = 7,
        reset_on_payout       = False,
    ),
    "The5ers — Challenge ($10k)": FundedAccountConfig(
        starting_equity       = 10_000.0,
        daily_loss_limit_pct  = 0.04,
        max_drawdown_pct      = 0.08,
        drawdown_type         = "absolute",
        profit_target_pct     = 0.08,
        min_trading_days      = 1,
        profit_split_pct      = 0.50,         # scales up over time on real account
        payout_frequency_days = 30,
        reset_on_payout       = False,
    ),
    "Custom": FundedAccountConfig(),          # editable defaults in the UI
}


# ── Simulation ────────────────────────────────────────────────────────────────

def run_funded_backtest(
    result: "BacktestResult",
    config: FundedAccountConfig,
) -> FundedResult:
    """
    Replay a BacktestResult through funded-account rules bar by bar.

    The strategy's trades are **not** modified — this function evaluates
    whether the resulting equity curve would have breached the firm's rules,
    and when.

    The equity curve is re-scaled from the backtest's starting equity to
    ``config.starting_equity``, so you can run a $10,000 backtest and apply
    FTMO $100k rules without re-running the strategy.

    Parameters
    ----------
    result : BacktestResult
        Output of ``engine.backtest.run_backtest()`` or ``run_splits()``.
    config : FundedAccountConfig
        The funded account rules to simulate.

    Returns
    -------
    FundedResult
    """
    eq_raw     = result.equity_curve.copy()
    timestamps = result.timestamps
    N          = len(eq_raw)
    cfg        = config

    # ── Scale backtest equity to the funded account's starting balance ─────────
    # Finds the first non-NaN value and uses it as the backtest's origin.
    valid_mask = ~np.isnan(eq_raw)
    if not valid_mask.any():
        # Degenerate result — no valid equity points
        return FundedResult(
            result=result, config=config,
            passed=False,
            failure_reason="No valid equity data in backtest result.",
            failure_bar=None, failure_date=None,
            trading_days_active=0,
            max_drawdown_reached_pct=0.0, peak_equity=cfg.starting_equity,
            payout_events=[], total_payout_to_trader=0.0,
            funded_equity_curve=eq_raw,
        )
    bt_origin = float(eq_raw[valid_mask][0])
    scale     = cfg.starting_equity / bt_origin if bt_origin != 0.0 else 1.0
    funded_eq = eq_raw * scale

    # ── Build a set of calendar dates with at least one closed trade ───────────
    closed_trade_dates: set = set()
    for t in result.closed_trades:
        if 0 <= t.exit_bar < N:
            closed_trade_dates.add(timestamps[t.exit_bar].date())

    # ── State ──────────────────────────────────────────────────────────────────
    hwm                = cfg.starting_equity    # running high-water mark
    anchor             = cfg.starting_equity    # fixed reference (absolute mode)
    # day_start_equity tracks equity at the END of the previous calendar day.
    # Initialised to starting_equity so the very first bar of day 1 is correct.
    # When a new day is detected we assign `prev_eq` (previous bar's close),
    # not the current bar's value — this ensures the drop from yesterday's
    # close to today's first bar is measured correctly on daily-resolution data.
    day_start_equity   = cfg.starting_equity
    prev_eq            = cfg.starting_equity    # equity at previous bar
    last_day           = None
    trading_days_set: set = set()

    # Payout / cycle tracking
    cycle_start_equity = cfg.starting_equity
    if N > 0:
        _d0 = timestamps[0].date()
    else:
        _d0 = None
    next_payout_date   = (
        _d0 + timedelta(days=cfg.payout_frequency_days) if _d0 else None
    )
    cycle_number       = 0
    payout_events: list[PayoutEvent] = []

    # Output tracking
    peak_eq       = cfg.starting_equity
    max_dd_pct    = 0.0
    failed        = False
    passed        = False
    failure_bar   = None
    failure_reason: Optional[str] = None

    for i in range(N):
        eq = funded_eq[i]
        if np.isnan(eq):
            continue

        day = timestamps[i].date()

        # ── Start of a new calendar day ────────────────────────────────────────
        if day != last_day:
            # Payout check: have we crossed the next payout boundary?
            if next_payout_date is not None and day >= next_payout_date:
                _do_payout(
                    i, day, prev_eq, cycle_start_equity,
                    cfg, cycle_number, payout_events,
                )
                if payout_events:
                    cycle_number += 1
                    if cfg.reset_on_payout:
                        # Re-anchor equity to starting_equity by shifting all
                        # future bars down by the accumulated profit
                        offset = eq - cfg.starting_equity
                        funded_eq[i:] -= offset
                        eq             = cfg.starting_equity
                        hwm            = cfg.starting_equity
                    cycle_start_equity = cfg.starting_equity if cfg.reset_on_payout else eq
                next_payout_date = day + timedelta(days=cfg.payout_frequency_days)

            if day in closed_trade_dates:
                trading_days_set.add(day)

            # Day starts at the CLOSE of the previous bar (prev_eq), not at
            # the current bar's open — so intraday drops from prev bar close
            # to this bar's value are correctly attributed to the new day.
            day_start_equity = prev_eq
            last_day = day

        # ── Update peak ────────────────────────────────────────────────────────
        peak_eq = max(peak_eq, eq)

        # ── Drawdown floor ─────────────────────────────────────────────────────
        if cfg.drawdown_type == "trailing":
            hwm   = max(hwm, eq)
            floor = hwm * (1.0 - cfg.max_drawdown_pct)
        else:  # absolute
            floor = anchor * (1.0 - cfg.max_drawdown_pct)

        # Track worst drawdown seen so far (as % of relevant reference)
        ref_for_dd = hwm if cfg.drawdown_type == "trailing" else anchor
        dd_now     = (ref_for_dd - eq) / ref_for_dd * 100.0 if ref_for_dd > 0 else 0.0
        if dd_now > max_dd_pct:
            max_dd_pct = dd_now

        # ── Rule 1 — Max drawdown breach ───────────────────────────────────────
        if eq <= floor:
            failed         = True
            failure_bar    = i
            floor_type     = "trailing HWM" if cfg.drawdown_type == "trailing" else "absolute anchor"
            failure_reason = (
                f"Max drawdown breached ({floor_type}) — "
                f"equity ${eq:,.2f} ≤ floor ${floor:,.2f} "
                f"[{cfg.max_drawdown_pct*100:.1f}% limit | date: {day}]"
            )
            break

        # ── Rule 2 — Daily loss limit ──────────────────────────────────────────
        if cfg.daily_loss_limit_pct > 0.0:
            daily_loss_pct = (day_start_equity - eq) / cfg.starting_equity
            if daily_loss_pct >= cfg.daily_loss_limit_pct:
                failed         = True
                failure_bar    = i
                failure_reason = (
                    f"Daily loss limit breached — "
                    f"lost ${day_start_equity - eq:,.2f} today "
                    f"({daily_loss_pct*100:.2f}% of ${cfg.starting_equity:,.0f}; "
                    f"limit: {cfg.daily_loss_limit_pct*100:.1f}%) "
                    f"[date: {day}]"
                )
                break

        # ── Rule 3 — Profit target (challenge phase) ───────────────────────────
        if cfg.profit_target_pct > 0.0:
            target = cfg.starting_equity * (1.0 + cfg.profit_target_pct)
            if eq >= target:
                n_days = len(trading_days_set)
                if n_days >= cfg.min_trading_days:
                    passed = True
                    break
                # Not enough trading days yet — keep running

        # ── Advance prev_eq for next bar's day-start detection ────────────────
        prev_eq = eq

    # ── Flatten equity curve after breach ─────────────────────────────────────
    if failed and failure_bar is not None:
        funded_eq[failure_bar + 1:] = funded_eq[failure_bar]

    # ── Determine overall pass/fail ────────────────────────────────────────────
    # For funded phases with no profit target: pass if no rule was breached
    if not failed and not passed:
        passed = True

    return FundedResult(
        result                   = result,
        config                   = config,
        passed                   = passed,
        failure_reason           = failure_reason,
        failure_bar              = failure_bar,
        failure_date             = (
            timestamps[failure_bar].date().isoformat()
            if failure_bar is not None else None
        ),
        trading_days_active      = len(trading_days_set),
        max_drawdown_reached_pct = round(max_dd_pct, 2),
        peak_equity              = round(peak_eq, 2),
        payout_events            = payout_events,
        total_payout_to_trader   = round(
            sum(p.trader_share for p in payout_events), 2
        ),
        funded_equity_curve      = funded_eq,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _do_payout(
    bar_index:          int,
    day,
    equity:             float,
    cycle_start_equity: float,
    cfg:                FundedAccountConfig,
    cycle_number:       int,
    payout_events:      list[PayoutEvent],
) -> None:
    """Record a payout event.  No-op if there is no profit to distribute."""
    gross_profit = equity - cycle_start_equity
    if gross_profit <= 0.0:
        return
    trader_share = gross_profit * cfg.profit_split_pct
    firm_share   = gross_profit * (1.0 - cfg.profit_split_pct)
    equity_after = cfg.starting_equity if cfg.reset_on_payout else equity
    payout_events.append(PayoutEvent(
        cycle_number  = cycle_number + 1,
        date          = day.isoformat(),
        bar_index     = bar_index,
        equity_before = round(equity, 2),
        profit_gross  = round(gross_profit, 2),
        trader_share  = round(trader_share, 2),
        firm_share    = round(firm_share, 2),
        equity_after  = round(equity_after, 2),
    ))
