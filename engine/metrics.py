"""
engine/metrics.py — Performance Metrics
==========================================

Standard backtesting metrics. All functions take a BacktestResult or a list
of closed trades + equity curve.

References:
  - Bailey et al. (2014) "The Deflation of the Sharpe Ratio"
  - De Prado (2018) "Advances in Financial Machine Learning" ch. 14
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest import BacktestResult


def compute_metrics(result: "BacktestResult",
                    trading_days_per_year: int = 252) -> dict:
    """
    Compute a comprehensive set of performance metrics from a BacktestResult.

    Returns a flat dict suitable for display in a table or DataFrame.
    """
    trades = result.closed_trades
    if not trades:
        return {"label": result.label, "trades": 0}

    pnls    = np.array([t.pnl_net for t in trades])
    wins    = pnls[pnls > 0]
    losses  = pnls[pnls <= 0]

    # ── Basic stats ───────────────────────────────────────────────────────────
    n_trades      = len(pnls)
    n_wins        = len(wins)
    n_losses      = len(losses)
    win_rate      = n_wins / n_trades * 100
    total_pnl     = float(pnls.sum())
    avg_win       = float(wins.mean())  if n_wins   else 0.0
    avg_loss      = float(losses.mean()) if n_losses else 0.0
    expectancy    = float(pnls.mean())
    profit_factor = (float(wins.sum()) / abs(float(losses.sum()))
                     if losses.sum() != 0 else float("inf"))
    payoff_ratio  = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # ── Sharpe (daily-bucketed to avoid M1 autocorrelation inflation) ─────────
    # Bucketing by exit_bar // ~390 (approx trading bars per session day on M1)
    daily: dict = defaultdict(float)
    for t in trades:
        day = t.exit_bar // 390
        daily[day] += t.pnl_net
    dv = np.array(list(daily.values()))
    if len(dv) > 1 and dv.std() > 0:
        sharpe = (dv.mean() / dv.std()) * np.sqrt(trading_days_per_year)
    else:
        sharpe = 0.0

    # ── Sortino (downside deviation only) ────────────────────────────────────
    neg_dv = dv[dv < 0]
    if len(neg_dv) > 0 and neg_dv.std() > 0:
        sortino = (dv.mean() / neg_dv.std()) * np.sqrt(trading_days_per_year)
    else:
        sortino = 0.0

    # ── Drawdown on equity curve ──────────────────────────────────────────────
    eq       = result.equity_curve[~np.isnan(result.equity_curve)]
    roll_max = np.maximum.accumulate(eq)
    dd_abs   = eq - roll_max
    dd_pct   = dd_abs / roll_max * 100
    max_dd_pct = float(dd_pct.min())
    max_dd_abs = float(dd_abs.min())

    # Longest drawdown duration (in bars)
    in_dd = dd_pct < 0
    max_dd_dur = _max_consecutive(in_dd)

    # ── Returns ───────────────────────────────────────────────────────────────
    total_return_pct = (eq[-1] - eq[0]) / eq[0] * 100 if eq[0] != 0 else 0.0
    calmar = (total_return_pct / abs(max_dd_pct)) if max_dd_pct != 0 else 0.0

    # ── Streaks ───────────────────────────────────────────────────────────────
    win_streak  = _max_streak(pnls,  positive=True)
    loss_streak = _max_streak(pnls, positive=False)

    # ── Per-side breakdown ───────────────────────────────────────────────────
    from .broker import OrderSide
    long_trades  = [t for t in trades if t.side == OrderSide.LONG]
    short_trades = [t for t in trades if t.side == OrderSide.SHORT]
    long_pnl  = sum(t.pnl_net for t in long_trades)
    short_pnl = sum(t.pnl_net for t in short_trades)

    return {
        "label":           result.label,
        "strategy":        result.strategy_name,
        # Trade counts
        "trades":          n_trades,
        "wins":            n_wins,
        "losses":          n_losses,
        "long_trades":     len(long_trades),
        "short_trades":    len(short_trades),
        # Win/loss stats
        "win_rate_%":      round(win_rate, 1),
        "avg_win_$":       round(avg_win, 2),
        "avg_loss_$":      round(avg_loss, 2),
        "payoff_ratio":    round(payoff_ratio, 2),
        "profit_factor":   round(profit_factor, 2),
        "expectancy_$":    round(expectancy, 2),
        # P&L
        "total_pnl_$":     round(total_pnl, 2),
        "long_pnl_$":      round(long_pnl, 2),
        "short_pnl_$":     round(short_pnl, 2),
        # Risk-adjusted
        "sharpe_ann":      round(sharpe, 2),
        "sortino_ann":     round(sortino, 2),
        "calmar":          round(calmar, 2),
        # Drawdown
        "max_dd_%":        round(max_dd_pct, 2),
        "max_dd_$":        round(max_dd_abs, 2),
        "max_dd_dur_bars": max_dd_dur,
        # Returns
        "total_return_%":  round(total_return_pct, 2),
        "final_equity_$":  round(float(eq[-1]), 2),
        # Streaks
        "max_win_streak":  win_streak,
        "max_loss_streak": loss_streak,
    }


def metrics_to_df(results: dict) -> pd.DataFrame:
    """Convert a dict of {label: BacktestResult} to a metrics comparison DataFrame."""
    rows = []
    for label, res in results.items():
        m = compute_metrics(res)
        rows.append(m)
    return pd.DataFrame(rows).set_index("label")


def trade_log_df(result: "BacktestResult") -> pd.DataFrame:
    """Convert closed trades to a tidy DataFrame for display/export."""
    from .broker import OrderSide
    trades = result.closed_trades
    if not trades:
        return pd.DataFrame()

    ts = result.timestamps
    N  = len(ts)

    rows = []
    for t in trades:
        rows.append({
            "entry_time":  ts[t.placed_bar].isoformat() if t.placed_bar < N else "",
            "fill_time":   ts[t.fill_bar].isoformat()   if 0 <= t.fill_bar  < N else "",
            "exit_time":   ts[t.exit_bar].isoformat()   if 0 <= t.exit_bar  < N else "",
            "direction":   "LONG" if t.side == OrderSide.LONG else "SHORT",
            "limit":       round(t.limit_price, 4),
            "fill_price":  round(t.fill_price, 4),
            "sl":          round(t.sl, 4),
            "tp":          round(t.tp, 4),
            "size":        t.size,
            "exit_price":  round(t.exit_price, 4),
            "exit_reason": t.exit_reason,
            "pnl_gross_$": round(t.pnl_gross, 2),
            "commission_$": round(t.commission, 2),
            "pnl_net_$":   round(t.pnl_net, 2),
            "tag":         t.tag,
        })
    return pd.DataFrame(rows)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _max_streak(pnls: np.ndarray, positive: bool) -> int:
    streak = max_s = 0
    for p in pnls:
        if (p > 0) == positive:
            streak += 1
            max_s = max(max_s, streak)
        else:
            streak = 0
    return max_s


def _max_consecutive(bool_arr: np.ndarray) -> int:
    streak = max_s = 0
    for v in bool_arr:
        if v:
            streak += 1
            max_s = max(max_s, streak)
        else:
            streak = 0
    return max_s
