"""
strategies/btc_eth_spread.py — BTC/ETH Statistical Spread Strategy
===================================================================

Mean reversion pairs trading on the BTC/ETH price spread.

Core concept (Gatev et al., 2006):
    BTC and ETH share a long-run equilibrium. When the spread deviates
    beyond 2 standard deviations, it tends to revert — trade that reversion.

Spread formula:
    Spread = BTC_close - (hedge_ratio × ETH_close)

Z-score (Vidyamurthy, 2004):
    Z = (Spread - rolling_mean(lookback)) / rolling_std(lookback)

Entry signals:
    Z > +entry_z  → SHORT spread: Sell BTC, Buy  ETH  (BTC expensive)
    Z < -entry_z  → LONG  spread: Buy  BTC, Sell ETH  (ETH expensive)

Exit signals:
    abs(Z) < exit_z        → Mean reversion complete, close both legs
    abs(Z) > sl_z (>0)     → Emergency stop, spread diverging further

Design for extensibility (v2 features — see PARAMS stubs):
    - Dynamic hedge ratio (rolling OLS regression)
    - Volatility filter (ATR spike detection)
    - Correlation stability filter
    - Session filter (avoid rollovers, low liquidity)
    - Cointegration testing (Engle-Granger)

References:
    - Gatev, E., Goetzmann, W., Rouwenhorst, K.G. (2006).
      "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
      Review of Financial Studies, 19(3), 797-827.
    - Vidyamurthy, G. (2004).
      "Pairs Trading: Quantitative Methods and Analysis."
      Wiley Finance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from engine.broker import Broker, OrderSide, OrderType
from engine.data import DataFeed
from engine.pairs import PairsStrategy

_log = logging.getLogger(__name__)


class BTCETHSpreadStrategy(PairsStrategy):
    """
    BTC/ETH z-score mean reversion pairs strategy — v1 prototype.

    All rolling computations happen in prepare() (causal, no look-ahead).
    on_bar() only reads pre-computed values — no rolling math in the loop.

    Position state:
        _position =  0 : flat (no trade)
        _position = +1 : long spread  (long BTC, short ETH)
        _position = -1 : short spread (short BTC, long ETH)
    """

    NAME = "BTC/ETH Spread — Z-Score Mean Reversion v1"

    PARAMS: dict[str, Any] = {
        # ── Core spread parameters ────────────────────────────────────────────
        "hedge_ratio": {
            "default": 18.0, "min": 5.0, "max": 50.0, "step": 0.5,
            "label": "Hedge Ratio (Spread = BTC - ratio×ETH)",
        },
        "lookback": {
            "default": 100, "min": 20, "max": 500, "step": 10,
            "label": "Rolling Window (bars)",
        },
        # ── Signal thresholds ─────────────────────────────────────────────────
        "entry_z": {
            "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.25,
            "label": "Entry Z-Score Threshold",
        },
        "exit_z": {
            "default": 0.3, "min": 0.0, "max": 1.5, "step": 0.05,
            "label": "Exit Z-Score (mean reversion complete)",
        },
        "sl_z": {
            "default": 3.5, "min": 0.0, "max": 6.0, "step": 0.25,
            "label": "Emergency Stop Z-Score (0 = disabled)",
        },
        # ── Position sizing ───────────────────────────────────────────────────
        "volume_btc": {
            "default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001,
            "label": "BTC Volume (lots) — ETH = volume × hedge_ratio",
        },
        # ── Future extension stubs (v2) ───────────────────────────────────────
        # "dynamic_hedge"    : use rolling OLS regression for hedge_ratio
        # "vol_filter_atr"   : skip entry if ATR spike detected
        # "corr_min"         : minimum rolling correlation required
        # "session_filter"   : avoid rollover / late Friday periods
        # "cointegration"    : run Engle-Granger test before entering
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        # Current position direction: 0=flat, +1=long spread, -1=short spread
        self._position: int = 0
        # Trade count for logging
        self._trade_count: int = 0
        # Passed to broker.close_all_at(exit_reason=...) — set before _exit_requested
        self._exit_reason: str = "zscore_exit"

    def prepare(self, feed_a: DataFeed, feed_b: DataFeed) -> None:
        """
        Pre-compute spread, z-score, and rolling stats over full aligned arrays.

        All computations are causal (rolling backward-looking windows only).
        Based on Vidyamurthy (2004) spread construction with fixed hedge ratio.

        Future v2 extension: replace fixed hedge_ratio with rolling OLS
        regression beta (dynamic hedge ratio) using np.polyfit on rolling window.
        """
        hedge_ratio = self.params["hedge_ratio"]
        lookback    = self.params["lookback"]

        close_btc = feed_a.close._data   # BTC close — full causal array
        close_eth = feed_b.close._data   # ETH close — full causal array

        # Spread = BTC - hedge_ratio × ETH
        # Stationary when BTC/ETH are cointegrated — Gatev et al. (2006)
        # hedge_ratio ~18 keeps the spread roughly dollar-neutral at typical prices
        spread = close_btc - hedge_ratio * close_eth

        # Causal rolling mean and std (min_periods=lookback → NaN during warmup)
        spread_s     = pd.Series(spread)
        rolling_mean = spread_s.rolling(lookback, min_periods=lookback).mean().values
        rolling_std  = spread_s.rolling(lookback, min_periods=lookback).std().values

        # Z-score: how many std deviations from the rolling mean?
        # NaN where rolling_std is ~0 (flat spread) or during warmup
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = np.where(
                rolling_std > 1e-8,
                (spread - rolling_mean) / rolling_std,
                np.nan,
            )

        # Attach all arrays to feed_a for bar-by-bar access in on_bar()
        # The pairs runner also reads 'spread' and 'zscore' for PairsResult
        feed_a._attach("spread",      spread)
        feed_a._attach("zscore",      zscore)
        feed_a._attach("roll_mean",   rolling_mean)
        feed_a._attach("roll_std",    rolling_std)

        # Log preparation summary
        valid_z = zscore[~np.isnan(zscore)]
        if len(valid_z):
            _log.info(
                "  [Prepare] Spread: %.2f → %.2f | "
                "Z-score: %.2f → %.2f | Valid from bar %d",
                float(spread.min()), float(spread.max()),
                float(valid_z.min()), float(valid_z.max()),
                lookback,
            )

    def on_bar(
        self,
        feed_a: DataFeed, broker_a: Broker,
        feed_b: DataFeed, broker_b: Broker,
    ) -> None:
        """
        Signal generation — reads pre-computed z-score, places paired orders.

        Exit:  sets self._exit_requested = True (runner handles close_all_at)
        Entry: places market orders on both brokers simultaneously
        """
        i       = feed_a.i
        zscore  = feed_a["zscore"][i]

        # Skip during warmup (z-score NaN) or if z-score is invalid
        if np.isnan(zscore):
            return

        entry_z = self.params["entry_z"]
        exit_z  = self.params["exit_z"]
        sl_z    = self.params["sl_z"]

        # ── Emergency stop (z-score diverging beyond sl_z) ───────────────────
        # Future v2: replace with ATR-based monetary stop or time stop
        if sl_z > 0 and self._position != 0:
            triggered = (
                (self._position == -1 and zscore >  sl_z) or
                (self._position ==  1 and zscore < -sl_z)
            )
            if triggered:
                _log.warning(
                    "  [Bar %d] EMERGENCY STOP — Z=%.3f exceeded ±%.1f | "
                    "trade #%d closed at loss",
                    i, zscore, sl_z, self._trade_count,
                )
                self._exit_reason = "emergency_stop"
                self._exit_requested = True
                self._position = 0
                return

        # ── Exit: mean reversion complete ─────────────────────────────────────
        if self._position != 0 and abs(zscore) < exit_z:
            _log.info(
                "  [Bar %d] EXIT — Z=%.3f reverted inside ±%.2f | trade #%d",
                i, zscore, exit_z, self._trade_count,
            )
            self._exit_reason = "zscore_exit"
            self._exit_requested = True
            self._position = 0
            return

        # ── Entry: only when flat ─────────────────────────────────────────────
        if self._position != 0:
            return   # already in a trade

        # Both legs must have capacity (no pending order, not at max_concurrent)
        if not broker_a.has_capacity or not broker_b.has_capacity:
            return
        # Also require no open positions on either leg (one trade at a time)
        if broker_a.n_open > 0 or broker_b.n_open > 0:
            return

        vol_btc = self.params["volume_btc"]
        # ETH volume: hedge_ratio × BTC volume
        # Keeps the spread dollar-exposure approximately neutral
        # v2 improvement: vol_eth = vol_btc * hedge_ratio * (btc_price/eth_price)
        vol_eth = vol_btc * self.params["hedge_ratio"]

        btc_price = feed_a.close[i]
        eth_price = feed_b.close[i]

        if zscore > entry_z:
            # SHORT SPREAD: BTC overpriced vs ETH → sell BTC, buy ETH
            # Profit when spread reverts down (Z decreases toward 0)
            self._trade_count += 1
            _log.info(
                "  [Bar %d] SHORT SPREAD #%d — Z=%.3f > +%.2f | "
                "Sell %.4f BTC @ %.2f | Buy %.4f ETH @ %.2f",
                i, self._trade_count, zscore, entry_z,
                vol_btc, btc_price, vol_eth, eth_price,
            )
            self._enter_short_spread(broker_a, broker_b, vol_btc, vol_eth,
                                     btc_price, eth_price, i)
            self._position = -1

        elif zscore < -entry_z:
            # LONG SPREAD: ETH overpriced vs BTC → buy BTC, sell ETH
            # Profit when spread reverts up (Z increases toward 0)
            self._trade_count += 1
            _log.info(
                "  [Bar %d] LONG SPREAD  #%d — Z=%.3f < -%.2f | "
                "Buy %.4f BTC @ %.2f | Sell %.4f ETH @ %.2f",
                i, self._trade_count, zscore, entry_z,
                vol_btc, btc_price, vol_eth, eth_price,
            )
            self._enter_long_spread(broker_a, broker_b, vol_btc, vol_eth,
                                    btc_price, eth_price, i)
            self._position = 1

    # ── Private helpers ───────────────────────────────────────────────────────

    def _enter_short_spread(
        self,
        broker_a: Broker, broker_b: Broker,
        vol_btc: float, vol_eth: float,
        btc_price: float, eth_price: float,
        bar_idx: int,
    ) -> None:
        """
        Short spread: Sell BTC (broker_a SHORT), Buy ETH (broker_b LONG).

        SL/TP are set wide — primary exit is z-score based (handled in on_bar).
        The 8% SL acts as a last-resort monetary stop for extreme moves.
        v2: replace with dynamic z-score SL → price conversion.
        """
        broker_a.place_order(
            side        = OrderSide.SHORT,
            order_type  = OrderType.MARKET,
            limit_price = btc_price,
            placed_bar  = bar_idx,
            size        = vol_btc,
            sl          = btc_price * 1.08,    # 8% adverse move stop
            tp          = btc_price * 0.85,    # wide TP (z-score exit handles it)
        )
        broker_b.place_order(
            side        = OrderSide.LONG,
            order_type  = OrderType.MARKET,
            limit_price = eth_price,
            placed_bar  = bar_idx,
            size        = vol_eth,
            sl          = eth_price * 0.92,
            tp          = eth_price * 1.15,
        )

    def _enter_long_spread(
        self,
        broker_a: Broker, broker_b: Broker,
        vol_btc: float, vol_eth: float,
        btc_price: float, eth_price: float,
        bar_idx: int,
    ) -> None:
        """
        Long spread: Buy BTC (broker_a LONG), Sell ETH (broker_b SHORT).
        """
        broker_a.place_order(
            side        = OrderSide.LONG,
            order_type  = OrderType.MARKET,
            limit_price = btc_price,
            placed_bar  = bar_idx,
            size        = vol_btc,
            sl          = btc_price * 0.92,
            tp          = btc_price * 1.15,
        )
        broker_b.place_order(
            side        = OrderSide.SHORT,
            order_type  = OrderType.MARKET,
            limit_price = eth_price,
            placed_bar  = bar_idx,
            size        = vol_eth,
            sl          = eth_price * 1.08,
            tp          = eth_price * 0.85,
        )
