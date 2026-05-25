"""
engine/walkforward.py — Walk-Forward Optimisation
====================================================

Walk-forward analysis systematically tests whether a strategy's parameters
generalise out-of-sample, rather than just fitting the in-sample period.

How it works:
  1. Slide a window across the dataset
  2. On each IS (in-sample) window: try all param combinations, pick the best
  3. On the OOS (out-of-sample) window immediately after: validate best params
  4. Record OOS result — this is the unbiased estimate of live performance
  5. Repeat for all windows

If OOS results consistently track IS results (score_ratio ≈ 1.0), the params
are robust. If OOS collapses, the strategy is curve-fit to each period.

Rolling vs Anchored:
  - Rolling (default): IS window slides forward with fixed size. More realistic
    for strategies where older data becomes irrelevant.
  - Anchored: IS start is fixed, window grows. More data per later window.

Usage:
    from engine.walkforward import run_walk_forward

    wf = run_walk_forward(
        df=df,
        strategy_class=IFVGStrategy,
        param_grid={
            "atr_sl":  [1.0, 1.5, 2.0],
            "tp_atr":  [1.5, 2.0, 3.0],
            "session": ["london", "ny", "both"],
        },
        broker_config=broker_cfg,
        window_bars=100_000,   # ~70 calendar days on M1
        step_bars=20_000,      # ~14 calendar days on M1
        score_fn="sharpe",
    )
    print(wf.summary())
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest, audit_indicator_causality
from .broker import BrokerConfig
from .data import CausalityError
from .metrics import compute_metrics

_log = logging.getLogger(__name__)


# ── Score functions ────────────────────────────────────────────────────────────

SCORE_FUNCTIONS: dict[str, Callable[[dict], float]] = {
    "sharpe":   lambda m: m.get("sharpe_ann",     -999.0),
    "pf":       lambda m: m.get("profit_factor",     0.0),
    "calmar":   lambda m: m.get("calmar",          -999.0),
    "pnl":      lambda m: m.get("total_pnl_$",    -999.0),
    "sortino":  lambda m: m.get("sortino_ann",     -999.0),
    "win_rate": lambda m: m.get("win_rate_%",         0.0),
}


def _score(result, score_fn: Union[str, Callable]) -> float:
    """Compute a scalar score from a BacktestResult. Returns -999 if no trades."""
    if result.n_trades == 0:
        return -999.0
    m = compute_metrics(result)
    if callable(score_fn):
        return float(score_fn(m))
    fn = SCORE_FUNCTIONS.get(score_fn)
    if fn is None:
        raise ValueError(
            f"Unknown score_fn '{score_fn}'. "
            f"Choose from: {list(SCORE_FUNCTIONS)} or pass a callable."
        )
    val = fn(m)
    return float(val) if not math.isnan(val) else -999.0


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """
    Results for a single IS+OOS window pair.

    Attributes:
        window_idx  : 0-based window index
        is_start    : IS window start timestamp
        is_end      : IS window end timestamp
        oos_start   : OOS window start timestamp
        oos_end     : OOS window end timestamp
        best_params : Parameter dict that scored highest on IS
        is_score    : Best IS score achieved
        oos_score   : OOS score using best_params
        is_result   : Full BacktestResult for IS with best_params
        oos_result  : Full BacktestResult for OOS with best_params
    """
    window_idx:  int
    is_start:    pd.Timestamp
    is_end:      pd.Timestamp
    oos_start:   pd.Timestamp
    oos_end:     pd.Timestamp
    best_params: dict
    is_score:    float
    oos_score:   float
    is_result:   "BacktestResult"
    oos_result:  "BacktestResult"

    @property
    def score_ratio(self) -> float:
        """
        OOS / IS score ratio.
        1.0 = OOS exactly matches IS (ideal).
        < 0 = OOS is negative while IS was positive (overfitting).
        Typical healthy range: 0.4–0.8.
        """
        if abs(self.is_score) < 1e-9:
            return 0.0
        return self.oos_score / self.is_score


@dataclass
class WalkForwardResult:
    """
    Full walk-forward optimisation result.

    Key properties:
        combined_oos_equity : OOS equity curves stitched in chronological order
        avg_score_ratio     : Mean OOS/IS ratio across all windows (stability metric)
        oos_trades          : All OOS closed trades in order (for Monte Carlo input)
    """
    windows:        list[WalkForwardWindow]
    strategy_name:  str
    param_grid:     dict
    score_fn:       str
    n_combos:       int
    window_bars:    int
    step_bars:      int
    anchored:       bool

    @property
    def oos_trades(self) -> list:
        """All OOS closed trades concatenated in chronological order."""
        trades = []
        for w in self.windows:
            trades.extend(w.oos_result.closed_trades)
        return trades

    @property
    def combined_oos_equity(self) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Stitch all OOS equity curves into a single continuous equity curve.
        Each window's curve is shifted to continue from where the previous ended.
        Returns (equity_array, timestamps).
        """
        all_eq:  list[np.ndarray]        = []
        all_ts:  list[pd.DatetimeIndex]  = []
        last_val: float = np.nan

        for w in self.windows:
            eq = w.oos_result.equity_curve
            ts = w.oos_result.timestamps
            valid = ~np.isnan(eq)
            if not valid.any():
                continue
            eq_v = eq[valid].copy()
            ts_v = ts[valid]

            # Shift so this window starts where previous ended
            if not np.isnan(last_val):
                shift = last_val - eq_v[0]
                eq_v  = eq_v + shift

            last_val = eq_v[-1]
            all_eq.append(eq_v)
            all_ts.append(ts_v)

        if not all_eq:
            return np.array([]), pd.DatetimeIndex([])

        return (
            np.concatenate(all_eq),
            pd.DatetimeIndex(np.concatenate([t.values for t in all_ts])),
        )

    @property
    def avg_score_ratio(self) -> float:
        """Mean OOS/IS score ratio across windows. Closer to 1.0 = more robust."""
        ratios = [w.score_ratio for w in self.windows if abs(w.is_score) > 1e-9]
        return float(np.mean(ratios)) if ratios else 0.0

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def best_params_stability(self) -> dict[str, list]:
        """
        For each parameter, list its chosen value across all windows.
        Stable strategies pick the same (or similar) values each window.
        """
        if not self.windows:
            return {}
        keys = list(self.windows[0].best_params.keys())
        return {k: [w.best_params[k] for w in self.windows] for k in keys}

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 65}",
            f"Walk-Forward Summary — {self.strategy_name}",
            f"{'=' * 65}",
            f"  Score metric : {self.score_fn}",
            f"  Mode         : {'anchored' if self.anchored else 'rolling'}",
            f"  Windows      : {len(self.windows)}",
            f"  Param combos : {self.n_combos}",
            f"  Avg OOS/IS   : {self.avg_score_ratio:.3f}  "
            f"(1.0=perfect, 0.5=good, <0=overfit)",
            "",
            f"  {'Win':<4} {'IS score':>10} {'OOS score':>10} {'Ratio':>7}  "
            f"{'IS trades':>9}  {'OOS trades':>10}  Best params",
            f"  {'-' * 62}",
        ]
        for w in self.windows:
            param_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
            lines.append(
                f"  {w.window_idx:<4} {w.is_score:>10.3f} {w.oos_score:>10.3f} "
                f"{w.score_ratio:>7.3f}  "
                f"{w.is_result.n_trades:>9}  {w.oos_result.n_trades:>10}  "
                f"{param_str}"
            )
        lines.append(f"{'=' * 65}\n")
        return "\n".join(lines)


# ── Main walk-forward function ─────────────────────────────────────────────────

def run_walk_forward(
    df:              pd.DataFrame,
    strategy_class:  type,
    param_grid:      dict[str, list],
    broker_config:   BrokerConfig,
    window_bars:     int                    = 50_000,
    step_bars:       int                    = 10_000,
    score_fn:        Union[str, Callable]   = "sharpe",
    starting_equity: float                  = 5_000.0,
    warmup_bars:     int                    = 200,
    anchored:        bool                   = False,
    max_combos:      int                    = 500,
    verbose:         bool                   = True,
) -> WalkForwardResult:
    """
    Run a walk-forward optimisation.

    Args:
        df              : Full OHLCV DataFrame with DatetimeIndex (UTC)
        strategy_class  : Strategy class (not instance) — e.g. IFVGStrategy
        param_grid      : Dict of param_name → list of values to search.
                          e.g. {"atr_sl": [1.0, 1.5, 2.0], "tp_atr": [2.0, 3.0]}
                          Strategy defaults are used for params NOT in the grid.
        broker_config   : BrokerConfig (identical across all windows)
        window_bars     : Number of bars in the IS optimisation window
        step_bars       : Number of bars in the OOS validation window; also the
                          step size between windows (rolling mode)
        score_fn        : Metric to maximise. String: "sharpe" | "pf" | "calmar" |
                          "pnl" | "sortino" | "win_rate". Or a callable:
                          fn(metrics_dict) → float.
        starting_equity : Equity reset to this at the start of each window
        warmup_bars     : Bars skipped for indicator warmup per window
        anchored        : If True, IS start is always bar 0 (growing window).
                          If False (default), rolling window of fixed size.
        max_combos      : Safety cap on grid size. Raise if exceeded.
                          Prevents accidental data-mining on huge grids.
        verbose         : Print per-window progress

    Returns:
        WalkForwardResult — use .summary(), .combined_oos_equity, .oos_trades

    Raises:
        ValueError      : If grid too large, dataset too short, etc.
        CausalityError  : If strategy prepare() uses future data (audited once
                          on full dataset before the WFA loop starts).
    """
    N = len(df)
    score_fn_name = score_fn if isinstance(score_fn, str) else "custom"
    strategy_name = getattr(strategy_class, "NAME", strategy_class.__name__)

    # ── Validate grid ─────────────────────────────────────────────────────────
    if not param_grid:
        raise ValueError(
            "param_grid is empty. Provide at least one parameter to search."
        )
    keys       = list(param_grid.keys())
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
    all_params = [dict(zip(keys, c)) for c in all_combos]
    n_combos   = len(all_params)

    if n_combos > max_combos:
        raise ValueError(
            f"param_grid produces {n_combos:,} combinations (limit={max_combos:,}). "
            "Large grids increase data-mining bias. "
            "Reduce the grid or set max_combos= to a higher value if intentional."
        )

    # ── Validate dataset size ─────────────────────────────────────────────────
    min_bars = window_bars + step_bars
    if N < min_bars:
        raise ValueError(
            f"Dataset too short: {N:,} bars. Need at least "
            f"window_bars ({window_bars:,}) + step_bars ({step_bars:,}) = "
            f"{min_bars:,} bars."
        )

    n_windows = (N - window_bars) // step_bars
    if n_windows < 1:
        raise ValueError("Not enough bars for any complete IS+OOS window pair.")

    # ── Causality audit — once on full dataset, not per window ───────────────
    if verbose:
        _log.info("\n[WFA] Auditing indicator causality on full dataset ...")
    _sample_strat = strategy_class()
    try:
        audit_indicator_causality(df, _sample_strat)
    except CausalityError:
        raise
    if verbose:
        _log.info("[WFA] Causality audit passed ✓")
        _log.info(
            "\n[WFA] %s  |  %d combos  |  %d windows  |  score=%s",
            strategy_name, n_combos, n_windows, score_fn_name
        )
        _log.info(
            "[WFA] IS=%s bars  OOS=%s bars  (%s)  total runs: %s IS + %s OOS",
            f"{window_bars:,}", f"{step_bars:,}",
            "anchored" if anchored else "rolling",
            f"{n_combos * n_windows:,}", f"{n_windows:,}",
        )

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    windows: list[WalkForwardWindow] = []

    for w_idx in range(n_windows):
        oos_start_i = window_bars + w_idx * step_bars
        oos_end_i   = oos_start_i + step_bars
        if oos_end_i > N:
            break

        is_start_i = 0 if anchored else w_idx * step_bars
        is_end_i   = oos_start_i

        df_is  = df.iloc[is_start_i:is_end_i].copy()
        df_oos = df.iloc[oos_start_i:oos_end_i].copy()

        if verbose:
            _log.info(
                "[WFA] Window %d/%d  IS: %s → %s  OOS: %s → %s",
                w_idx + 1, n_windows,
                df_is.index[0].date(), df_is.index[-1].date(),
                df_oos.index[0].date(), df_oos.index[-1].date(),
            )

        # ── Optimise on IS ────────────────────────────────────────────────────
        best_score:     float  = -np.inf
        best_params:    dict   = all_params[0]
        best_is_result         = None

        for combo_i, params in enumerate(all_params):
            strat = strategy_class(params)
            try:
                result = run_backtest(
                    df_is, strat, broker_config,
                    starting_equity=starting_equity,
                    warmup_bars=warmup_bars,
                    label=f"W{w_idx}_IS_c{combo_i}",
                    audit_causality=False,   # already audited on full dataset
                    verbose=False,
                )
            except Exception as e:
                _log.warning("  IS combo %d error: %s", combo_i, e)
                continue

            s = _score(result, score_fn)
            if s > best_score:
                best_score     = s
                best_params    = params
                best_is_result = result

            if verbose and n_combos > 5 and (combo_i + 1) % max(1, n_combos // 4) == 0:
                _log.info("  IS: %d/%d combos, best=%.3f", combo_i + 1, n_combos, best_score)

        if best_is_result is None:
            if verbose:
                _log.info("  [skip] No valid IS result for window %d", w_idx + 1)
            continue

        if verbose:
            pstr = ", ".join(f"{k}={v}" for k, v in best_params.items())
            _log.info("  IS best: %s=%.3f  params: %s", score_fn_name, best_score, pstr)

        # ── Validate on OOS ───────────────────────────────────────────────────
        strat_oos = strategy_class(best_params)
        try:
            oos_result = run_backtest(
                df_oos, strat_oos, broker_config,
                starting_equity=starting_equity,
                warmup_bars=warmup_bars,
                label=f"W{w_idx}_OOS",
                audit_causality=False,
                verbose=False,
            )
        except Exception as e:
            _log.warning("  OOS run error: %s", e)
            continue

        oos_score = _score(oos_result, score_fn)
        ratio     = oos_score / best_score if abs(best_score) > 1e-9 else 0.0

        if verbose:
            _log.info(
                "  OOS: %s=%.3f  ratio=%.2f  trades=%d",
                score_fn_name, oos_score, ratio, oos_result.n_trades,
            )

        windows.append(WalkForwardWindow(
            window_idx  = w_idx,
            is_start    = df_is.index[0],
            is_end      = df_is.index[-1],
            oos_start   = df_oos.index[0],
            oos_end     = df_oos.index[-1],
            best_params = best_params,
            is_score    = best_score,
            oos_score   = oos_score,
            is_result   = best_is_result,
            oos_result  = oos_result,
        ))

    wf_result = WalkForwardResult(
        windows       = windows,
        strategy_name = strategy_name,
        param_grid    = param_grid,
        score_fn      = score_fn_name,
        n_combos      = n_combos,
        window_bars   = window_bars,
        step_bars     = step_bars,
        anchored      = anchored,
    )

    if verbose:
        _log.info("%s", wf_result.summary())

    return wf_result
