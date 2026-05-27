"""
engine/risk.py — Risk Management Components
============================================

Engine-level risk controls applied by the pairs runner (and optionally the
single-asset runner) at every bar, independent of any strategy logic.

Components
----------
DailyCircuitBreaker
    Monitors realized + floating daily P&L on a combined equity pool.
    Trips when the total daily loss exceeds ``daily_loss_pct × initial_equity``.
    Resets at midnight UTC.  When triggered the runner force-closes any open
    position and blocks new entries for the rest of that calendar day.

Design notes
------------
* The limit is fixed against ``initial_equity``, NOT the current balance.
  This matches FTMO / Topstep convention where the daily-loss rule is stated
  as "X% of the funded account size", not "X% of whatever your balance is today".

* Combined equity formula: ``broker_a.equity + broker_b.equity`` — do NOT
  subtract ``initial_equity``.  The pairs runner splits starting equity across
  two brokers (each starts at ``starting_equity / 2``), so their sum equals
  the full account value from the very first bar.  Subtracting initial_equity
  would give 0 at bar 0 and immediately trip the circuit.

References
----------
- FTMO Trading Objectives (2024): daily_loss_limit = 5% of funded account size
- Topstep Rules (2024): daily loss limit = 3% of combine account size
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

_log = logging.getLogger(__name__)


@dataclass
class DailyCircuitBreaker:
    """
    Engine-level daily loss circuit breaker for the pairs runner.

    When ``daily_loss_pct > 0``, the runner evaluates this check at the
    start of every bar (before ``strategy.on_bar()``).  If the combined
    realized + floating daily loss has reached the limit the runner:

      1. Sets ``strategy._exit_reason = "daily_circuit_breaker"``
      2. Sets ``strategy._exit_requested = True`` (engine closes both legs)
      3. Skips ``strategy.on_bar()`` for the rest of the UTC day

    The circuit resets automatically at midnight UTC each day.

    Parameters
    ----------
    daily_loss_pct : float
        Maximum permitted daily loss as a fraction of ``initial_equity``
        (e.g. 0.04 = 4%).  Set 0.0 to disable — a disabled breaker adds
        zero overhead to the bar loop (early exit on the pct check).
    initial_equity : float
        Reference capital for the daily limit.  Should match the
        ``starting_equity`` passed to ``run_backtest_pairs()``.

    Examples
    --------
    >>> cb = DailyCircuitBreaker(daily_loss_pct=0.04, initial_equity=10_000.0)
    >>> cb.limit_abs  # $400 daily loss limit
    400.0
    >>> cb.enabled
    True
    >>> DailyCircuitBreaker().enabled   # default 0.0 = disabled
    False
    """
    daily_loss_pct: float = 0.0
    initial_equity: float = 10_000.0

    # ── Runtime state (reset each day) ───────────────────────────────────────
    # These fields are managed by the engine bar loop — not set by the user.
    _current_day:      str   = field(default="", repr=False)
    _day_start_equity: float = field(default=0.0, repr=False)
    _tripped:          bool  = field(default=False, repr=False)

    @property
    def enabled(self) -> bool:
        """True when the circuit breaker is active (daily_loss_pct > 0)."""
        return self.daily_loss_pct > 0.0

    @property
    def limit_abs(self) -> float:
        """Absolute dollar value of the daily loss limit."""
        return self.daily_loss_pct * self.initial_equity

    def reset_day(self, today_str: str, combined_equity: float) -> None:
        """
        Called by the engine at the first bar of a new UTC calendar day.

        Resets the circuit and records today's opening equity as the baseline.
        """
        self._current_day      = today_str
        self._day_start_equity = combined_equity
        self._tripped          = False

    def check(
        self,
        bar_index:       int,
        today_str:       str,
        combined_equity: float,
        floating_pnl:    float,
    ) -> bool:
        """
        Evaluate whether the daily loss limit has been breached.

        Called by the engine each bar after any day-transition reset.

        Parameters
        ----------
        bar_index       : current bar index (for logging)
        today_str       : current UTC calendar date (YYYY-MM-DD)
        combined_equity : broker_a.equity + broker_b.equity
        floating_pnl    : unrealized P&L on all open positions (both legs)

        Returns
        -------
        bool  True = circuit just tripped (close + block), False = OK / already tripped.
        """
        if not self.enabled or self._tripped:
            return False

        total_daily_loss = (combined_equity - self._day_start_equity) + floating_pnl
        limit            = -self.limit_abs

        if total_daily_loss <= limit:
            self._tripped = True
            _log.warning(
                "  [Bar %d | %s] DAILY CIRCUIT BREAKER tripped — "
                "daily loss $%,.2f (%.1f%% of $%,.0f) ≥ limit $%,.2f (%.1f%%) "
                "— closing position and halting entries today",
                bar_index, today_str,
                -total_daily_loss,
                (-total_daily_loss / self.initial_equity) * 100,
                self.initial_equity,
                self.limit_abs,
                self.daily_loss_pct * 100,
            )
            return True

        return False

    def is_blocked(self) -> bool:
        """
        True when today's circuit has tripped — entries should be blocked.
        The caller should still allow exits (strategy may request them independently).
        """
        return self._tripped
