"""
engine/monte_carlo.py — Monte Carlo Simulation
================================================

Takes a completed backtest's trade log and runs N simulations by resampling
the trade PnLs with replacement (bootstrap). Each simulation produces a
different equity curve — the spread of outcomes reveals the true risk profile
of the strategy.

Why this matters:
  A single equity curve is one path through randomness. The same strategy,
  same market, would have produced thousands of slightly different curves
  depending on trade sequence. Monte Carlo shows the distribution of those
  possible outcomes, answering:
    - What is the 5th-percentile max drawdown? (worst realistic case)
    - What is the probability of ruin (equity dropping below floor)?
    - What is the realistic Sharpe range, not just the point estimate?
    - Is the observed equity curve luck, or typical of the strategy?

Usage:
    from engine.monte_carlo import run_monte_carlo

    mc = run_monte_carlo(backtest_result, n_sims=2000)
    print(f"5th pct max DD: {mc.max_dd_pct(5):.1f}%")
    print(f"Prob of ruin:   {mc.prob_ruin:.1%}")
    print(f"Median Sharpe:  {mc.sharpe_median:.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    """
    Results from N Monte Carlo simulations.

    All simulation equity curves are stored in `sim_curves` (shape: N_sims × N_trades+1).
    Use the percentile properties and methods for summary statistics.
    """
    sim_curves:      np.ndarray   # shape: (n_sims, n_trades + 1)
    final_equities:  np.ndarray   # shape: (n_sims,)
    max_drawdowns:   np.ndarray   # shape: (n_sims,)  as positive % values
    sharpes:         np.ndarray   # shape: (n_sims,)
    starting_equity: float
    n_sims:          int
    n_trades:        int
    ruin_floor:      float        # equity level considered "ruin"

    # ── Drawdown percentiles ───────────────────────────────────────────────────

    def max_dd_pct(self, percentile: float) -> float:
        """Max drawdown at given percentile (e.g. 5 = worst 5% of scenarios)."""
        return float(np.percentile(self.max_drawdowns, percentile))

    @property
    def max_dd_median(self) -> float:
        return float(np.median(self.max_drawdowns))

    @property
    def max_dd_p5(self) -> float:
        """5th percentile max drawdown — worst realistic case."""
        return self.max_dd_pct(95)   # higher = worse DD (we stored as positive %)

    # ── Final equity percentiles ───────────────────────────────────────────────

    def final_equity_pct(self, percentile: float) -> float:
        return float(np.percentile(self.final_equities, percentile))

    @property
    def final_equity_median(self) -> float:
        return float(np.median(self.final_equities))

    # ── Probability of ruin ────────────────────────────────────────────────────

    @property
    def prob_ruin(self) -> float:
        """
        Fraction of simulations where equity touched or fell below ruin_floor.
        ruin_floor defaults to 50% of starting_equity.
        """
        ruined = np.sum(
            np.min(self.sim_curves, axis=1) <= self.ruin_floor
        )
        return float(ruined) / self.n_sims

    # ── Sharpe ────────────────────────────────────────────────────────────────

    @property
    def sharpe_median(self) -> float:
        valid = self.sharpes[~np.isnan(self.sharpes)]
        return float(np.median(valid)) if len(valid) else float("nan")

    def sharpe_pct(self, percentile: float) -> float:
        valid = self.sharpes[~np.isnan(self.sharpes)]
        return float(np.percentile(valid, percentile)) if len(valid) else float("nan")

    # ── Equity curve percentile bands (for fan chart) ────────────────────────

    def equity_bands(self, percentiles: list[float] = None) -> dict[str, np.ndarray]:
        """
        Compute equity curve percentile bands for visualisation.

        Returns a dict mapping percentile label → array of equity values at each
        trade step (0..n_trades). Useful for drawing confidence fan charts.

        Args:
            percentiles: list of percentiles to compute (default: [5, 25, 50, 75, 95])
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        return {
            f"p{int(p)}": np.percentile(self.sim_curves, p, axis=0)
            for p in percentiles
        }

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 55}",
            f"Monte Carlo Summary  ({self.n_sims:,} simulations, {self.n_trades} trades each)",
            f"{'=' * 55}",
            f"  Starting equity : ${self.starting_equity:,.2f}",
            f"  Probability of ruin (<50% equity) : {self.prob_ruin:.1%}",
            "",
            "  Final equity percentiles:",
            f"    5th  : ${self.final_equity_pct(5):>10,.2f}",
            f"    25th : ${self.final_equity_pct(25):>10,.2f}",
            f"    50th : ${self.final_equity_pct(50):>10,.2f}  (median)",
            f"    75th : ${self.final_equity_pct(75):>10,.2f}",
            f"    95th : ${self.final_equity_pct(95):>10,.2f}",
            "",
            "  Max drawdown percentiles (% from peak):",
            f"    5th  : {self.max_dd_pct(5):>8.1f}%  (best case)",
            f"    50th : {self.max_dd_pct(50):>8.1f}%  (median)",
            f"    95th : {self.max_dd_pct(95):>8.1f}%  (worst realistic case)",
            "",
            "  Sharpe (annualised, daily-bucketed):",
            f"    5th  : {self.sharpe_pct(5):>8.2f}",
            f"    50th : {self.sharpe_median:>8.2f}  (median)",
            f"    95th : {self.sharpe_pct(95):>8.2f}",
            f"{'=' * 55}\n",
        ]
        return "\n".join(lines)


# ── Main Monte Carlo function ──────────────────────────────────────────────────

def run_monte_carlo(
    result,                          # BacktestResult
    n_sims:          int   = 1_000,
    starting_equity: Optional[float] = None,
    ruin_floor:      Optional[float] = None,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> MonteCarloResult:
    """
    Run a bootstrap Monte Carlo simulation on a completed backtest.

    Each simulation:
      1. Resamples the closed trade PnLs with replacement (same number of trades)
      2. Accumulates them into an equity curve starting from `starting_equity`
      3. Records final equity, max drawdown, and Sharpe

    Args:
        result          : BacktestResult from run_backtest() or a WFA OOS window
        n_sims          : Number of simulations (default 1,000; use 5,000 for publishing)
        starting_equity : Starting equity for simulations. Defaults to the
                          backtest's own starting equity (first equity curve value).
        ruin_floor      : Equity level considered "ruin". Defaults to 50% of
                          starting_equity (i.e., -50% drawdown = ruined).
        seed            : Random seed for reproducibility
        verbose         : Print summary stats

    Returns:
        MonteCarloResult — use .equity_bands(), .prob_ruin, .summary()

    Raises:
        ValueError : If no closed trades in result (nothing to simulate)
    """
    trades = result.closed_trades if hasattr(result, "closed_trades") else result
    if hasattr(result, "closed_trades"):
        trades = result.closed_trades
    else:
        trades = list(result)

    if not trades:
        raise ValueError(
            "No closed trades in result — cannot run Monte Carlo simulation. "
            "Run a backtest that produces at least a few closed trades first."
        )

    # Extract net PnLs from closed trades
    pnls = np.array([t.pnl_net for t in trades], dtype=float)
    n_trades = len(pnls)

    # Determine starting equity from the backtest result
    if starting_equity is None:
        eq_curve = getattr(result, "equity_curve", None)
        if eq_curve is not None:
            valid = eq_curve[~np.isnan(eq_curve)]
            starting_equity = float(valid[0]) if len(valid) else 5_000.0
        else:
            starting_equity = 5_000.0

    if ruin_floor is None:
        ruin_floor = starting_equity * 0.5

    if verbose:
        _log.info(
            "\n[MC] Running %s simulations | %d trades | starting equity $%s",
            f"{n_sims:,}", n_trades, f"{starting_equity:,.2f}",
        )

    rng = np.random.default_rng(seed)

    # ── Vectorised simulation ─────────────────────────────────────────────────
    # Resample PnLs: shape (n_sims, n_trades)
    sampled = rng.choice(pnls, size=(n_sims, n_trades), replace=True)

    # Cumulative sum per simulation → equity curve (n_sims, n_trades + 1)
    # First column is starting_equity, then add each trade's PnL
    cum_pnl = np.hstack([
        np.zeros((n_sims, 1)),
        np.cumsum(sampled, axis=1)
    ])
    sim_curves = starting_equity + cum_pnl

    # Final equities
    final_equities = sim_curves[:, -1]

    # Max drawdowns (as positive percentages from peak)
    running_max = np.maximum.accumulate(sim_curves, axis=1)
    drawdowns   = (sim_curves - running_max) / running_max * 100  # negative values
    max_drawdowns = np.abs(np.min(drawdowns, axis=1))              # positive %

    # Sharpe: treat each trade PnL as a daily return proxy
    # (simple annualisation — not calendar-accurate, but consistent across sims)
    means = np.mean(sampled, axis=1)
    stds  = np.std(sampled,  axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(stds > 0, means / stds * np.sqrt(252), np.nan)

    mc_result = MonteCarloResult(
        sim_curves      = sim_curves,
        final_equities  = final_equities,
        max_drawdowns   = max_drawdowns,
        sharpes         = sharpes,
        starting_equity = starting_equity,
        n_sims          = n_sims,
        n_trades        = n_trades,
        ruin_floor      = ruin_floor,
    )

    if verbose:
        _log.info("%s", mc_result.summary())

    return mc_result
