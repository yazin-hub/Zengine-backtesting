"""
strategies/ma_cross.py — Simple EMA Crossover Strategy
=========================================================

A minimal example strategy to demonstrate the engine is truly
strategy-agnostic. Works on any OHLCV instrument/timeframe.

Long when fast EMA crosses above slow EMA.
Short when fast EMA crosses below slow EMA.
SL at ATR-based distance. TP at 2× SL.
"""

from __future__ import annotations

import numpy as np
from engine.strategy import BaseStrategy
from engine.data import DataFeed
from engine.broker import Broker, OrderSide, OrderType
from engine import indicators as ind


class MACrossStrategy(BaseStrategy):

    NAME = "EMA Crossover"

    PARAMS = {
        "fast_ema":  {"default": 10,  "min": 3,   "max": 50,  "step": 1,   "label": "Fast EMA Period"},
        "slow_ema":  {"default": 30,  "min": 10,  "max": 200, "step": 5,   "label": "Slow EMA Period"},
        "atr_period":{"default": 14,  "min": 5,   "max": 30,  "step": 1,   "label": "ATR Period"},
        "atr_sl":    {"default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1, "label": "SL (ATR×)"},
        "rr":        {"default": 2.0, "min": 0.5, "max": 5.0, "step": 0.5, "label": "Risk:Reward"},
        "session":   {"default": "all", "options": ["london","ny","both","all"], "label": "Session Filter"},
    }

    def prepare(self, feed: DataFeed) -> None:
        p     = self.params
        high  = feed.high._data
        low   = feed.low._data
        close = feed.close._data

        feed._attach("ema_fast", ind.ema(close, p["fast_ema"]))
        feed._attach("ema_slow", ind.ema(close, p["slow_ema"]))
        feed._attach("atr",      ind.atr(high, low, close, p["atr_period"]))
        sess = ind.session_mask(feed.index, p["session"])
        feed._attach("in_session", sess.astype(float))

        self._last_signal = 0   # track last crossover direction

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        i = feed.i
        if i < 2:
            return
        if not feed["in_session"][i]:
            return
        if not broker.has_capacity:
            return

        fast_now  = feed["ema_fast"][i]
        fast_prev = feed["ema_fast"][i - 1]
        slow_now  = feed["ema_slow"][i]
        slow_prev = feed["ema_slow"][i - 1]
        atr_i     = feed["atr"][i]
        close_i   = feed.close[i]

        if np.isnan(fast_now) or np.isnan(slow_now) or np.isnan(atr_i):
            return

        sl_dist = self.params["atr_sl"] * atr_i
        rr      = self.params["rr"]

        # Bullish crossover
        if fast_prev <= slow_prev and fast_now > slow_now and self._last_signal != 1:
            sl = close_i - sl_dist
            tp = close_i + sl_dist * rr
            broker.place_order(
                side=OrderSide.LONG, order_type=OrderType.MARKET,
                limit_price=close_i, sl=sl, tp=tp,
                placed_bar=i, tag="ma_cross_long",
            )
            self._last_signal = 1

        # Bearish crossover
        elif fast_prev >= slow_prev and fast_now < slow_now and self._last_signal != -1:
            sl = close_i + sl_dist
            tp = close_i - sl_dist * rr
            broker.place_order(
                side=OrderSide.SHORT, order_type=OrderType.MARKET,
                limit_price=close_i, sl=sl, tp=tp,
                placed_bar=i, tag="ma_cross_short",
            )
            self._last_signal = -1
