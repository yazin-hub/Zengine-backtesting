"""
strategies/ifvg.py — IFVG-Only M1 Scalping Strategy
======================================================

ICT Implied Fair Value Gap strategy. Detects liquidity sweeps followed by
Fair Value Gaps (FVGs), waits for the FVG to be "inverted" (filled by price),
then trades the inverted zone as support/resistance.

Compatible with any instrument. Tuned defaults for XAUUSD M1.

Look-ahead guarantee:
  - All zone pre-computation in prepare() is causal
  - IFVG zones are only activated on bar AFTER inversion (strict inv_bar < i)
  - Feed cursor guards prevent any bar > i access in on_bar()
"""

from __future__ import annotations

import numpy as np
from collections import deque

from engine.strategy import BaseStrategy
from engine.data import DataFeed
from engine.broker import Broker, OrderSide, OrderType
from engine import indicators as ind


class IFVGStrategy(BaseStrategy):

    NAME = "IFVG M1 Scalping"

    PARAMS = {
        "sweep_lookback":  {"default": 10, "min": 5,   "max": 30,  "step": 1,   "label": "Sweep Lookback (bars)"},
        "sweep_min_atr":   {"default": 0.3,"min": 0.1, "max": 2.0, "step": 0.1, "label": "Min Sweep Size (ATR×)"},
        "fvg_window":      {"default": 20, "min": 5,   "max": 50,  "step": 1,   "label": "FVG-Sweep Association Window"},
        "max_fvg_age":     {"default": 45, "min": 10,  "max": 100, "step": 5,   "label": "Max Zone Age (bars)"},
        "atr_period":      {"default": 14, "min": 5,   "max": 30,  "step": 1,   "label": "ATR Period"},
        "atr_sl":          {"default": 1.5,"min": 0.5, "max": 5.0, "step": 0.1, "label": "SL (ATR×)"},
        "tp_atr":          {"default": 2.0,"min": 0.5, "max": 8.0, "step": 0.1, "label": "TP (ATR×)"},
        "entry_pct":       {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.1, "label": "Entry % in Zone (0=outer, 1=inner)"},
        "max_rr":          {"default": 5.0,"min": 1.0, "max": 20., "step": 0.5, "label": "Max R:R Cap"},
        "session":         {"default": "both", "options": ["london","ny","both","all"], "label": "Session Filter"},
        "min_bars_warmup": {"default": 250, "min": 50, "max": 500, "step": 50,  "label": "Warmup Bars"},
    }

    def prepare(self, feed: DataFeed) -> None:
        p     = self.params
        high  = feed.high._data
        low   = feed.low._data
        close = feed.close._data
        N     = len(close)

        # Causal indicators
        atr_arr  = ind.atr(high, low, close, p["atr_period"])
        bear_sw, bull_sw = ind.liquidity_sweeps(
            high, low, close, atr_arr,
            lookback=p["sweep_lookback"], min_atr=p["sweep_min_atr"]
        )
        (bear_fvg, bull_fvg,
         bfvg_top, bfvg_bot,
         ufvg_top, ufvg_bot) = ind.fair_value_gaps(high, low)

        # Attach arrays so on_bar can read them through the guard
        feed._attach("atr", atr_arr)

        # Session mask
        if hasattr(feed.index, 'hour'):
            sess = ind.session_mask(feed.index, p["session"])
        else:
            sess = np.ones(N, bool)
        feed._attach("in_session", sess.astype(float))

        # Build IFVG zone lists (causal — zones activated only AFTER inv_bar)
        MA = p["max_fvg_age"]
        FW = p["fvg_window"]

        ifvg_bull, ifvg_bear = [], []

        for f in np.where(bear_fvg)[0]:
            start = max(0, f - FW)
            if not bear_sw[start:f].any():
                continue
            zbot, ztop = bfvg_bot[f], bfvg_top[f]
            if np.isnan(zbot) or np.isnan(ztop) or ztop <= zbot:
                continue
            end  = min(f + MA + 1, N)
            hits = np.where(close[f + 1:end] > ztop)[0]
            if len(hits):
                ifvg_bull.append((f + 1 + hits[0], zbot, ztop, f))

        for f in np.where(bull_fvg)[0]:
            start = max(0, f - FW)
            if not bull_sw[start:f].any():
                continue
            zbot, ztop = ufvg_bot[f], ufvg_top[f]
            if np.isnan(zbot) or np.isnan(ztop) or ztop <= zbot:
                continue
            end  = min(f + MA + 1, N)
            hits = np.where(close[f + 1:end] < zbot)[0]
            if len(hits):
                ifvg_bear.append((f + 1 + hits[0], zbot, ztop, f))

        ifvg_bull.sort(key=lambda x: x[0])
        ifvg_bear.sort(key=lambda x: x[0])

        # Store on strategy (not feed — these are strategy state, not indicators)
        self._ifvg_bull  = ifvg_bull
        self._ifvg_bear  = ifvg_bear
        self._bull_ptr   = 0
        self._bear_ptr   = 0
        self._act_bull: deque = deque()
        self._act_bear: deque = deque()
        self._used_zones: set  = set()
        self._ts         = feed.index   # for zone_id construction

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        i  = feed.i
        p  = self.params
        MA = p["max_fvg_age"]

        # Session filter
        if not feed["in_session"][i]:
            return

        atr_i = feed["atr"][i]
        if np.isnan(atr_i) or atr_i <= 0:
            return

        if not broker.has_capacity:
            return

        # Advance zone pointers (strict: inv_bar < i, not <=)
        while (self._bull_ptr < len(self._ifvg_bull) and
               self._ifvg_bull[self._bull_ptr][0] < i):
            self._act_bull.append(self._ifvg_bull[self._bull_ptr])
            self._bull_ptr += 1
        while (self._bear_ptr < len(self._ifvg_bear) and
               self._ifvg_bear[self._bear_ptr][0] < i):
            self._act_bear.append(self._ifvg_bear[self._bear_ptr])
            self._bear_ptr += 1

        # Expire old zones
        while self._act_bull and (i - self._act_bull[0][0]) > MA:
            self._act_bull.popleft()
        while self._act_bear and (i - self._act_bear[0][0]) > MA:
            self._act_bear.popleft()

        close_i    = feed.close[i]
        entry_pct  = p["entry_pct"]
        atr_sl     = p["atr_sl"]
        tp_atr     = p["tp_atr"]
        max_rr     = p["max_rr"]
        expiry     = MA

        # Bear IFVG → short
        for idx in range(len(self._act_bear) - 1, -1, -1):
            inv_bar, zbot, ztop, orig_f = self._act_bear[idx]
            lim     = zbot + entry_pct * (ztop - zbot)
            zone_id = f"bear_{self._ts[orig_f].isoformat()}"
            if zone_id in self._used_zones:
                continue
            if close_i > ztop:
                continue
            sl      = ztop + atr_sl * atr_i
            tp      = lim  - tp_atr * atr_i
            sl_dist = abs(lim - sl)
            if sl_dist <= 0 or abs(tp - lim) / sl_dist > max_rr:
                continue
            order = broker.place_order(
                side=OrderSide.SHORT, order_type=OrderType.LIMIT,
                limit_price=lim, sl=sl, tp=tp,
                placed_bar=i, tag=zone_id, expiry_bars=expiry,
            )
            if order is not None:
                self._used_zones.add(zone_id)
            return

        # Bull IFVG → long
        for idx in range(len(self._act_bull) - 1, -1, -1):
            inv_bar, zbot, ztop, orig_f = self._act_bull[idx]
            lim     = ztop - entry_pct * (ztop - zbot)
            zone_id = f"bull_{self._ts[orig_f].isoformat()}"
            if zone_id in self._used_zones:
                continue
            if close_i < zbot:
                continue
            sl      = zbot - atr_sl * atr_i
            tp      = lim  + tp_atr * atr_i
            sl_dist = abs(lim - sl)
            if sl_dist <= 0 or abs(tp - lim) / sl_dist > max_rr:
                continue
            order = broker.place_order(
                side=OrderSide.LONG, order_type=OrderType.LIMIT,
                limit_price=lim, sl=sl, tp=tp,
                placed_bar=i, tag=zone_id, expiry_bars=expiry,
            )
            if order is not None:
                self._used_zones.add(zone_id)
            return
