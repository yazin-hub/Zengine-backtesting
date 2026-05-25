"""
engine/broker.py — Virtual Broker
===================================

Handles order placement, limit fills, SL/TP management, position sizing,
spread, slippage, and trailing stops.

Execution rules (conservative by design):
  - Limit orders fill on bar AFTER placement (never same bar as signal)
  - If bar opens through the limit (gap), fill at open price (not limit)
  - SL and TP both hit on same bar → SL wins (worst-case assumption)
  - Commission applied at fill (entry) and at exit (round-trip split evenly)

Spread model:
  - LONG entry:   buy at ask  = price + spread   (pays spread on entry)
  - SHORT entry:  sell at bid = price             (no spread on entry)
  - LONG exit:    sell at bid = sl/tp price       (no spread on exit)
  - SHORT exit:   buy  at ask = sl/tp + spread    (pays spread on exit)
  Net effect: one spread per round trip regardless of direction — same as live.

Trailing stop model:
  - Manual:  strategy calls broker.trail_sl(new_sl) in on_bar()
  - Auto:    set trail_pct > 0 in BrokerConfig; broker trails automatically
             trail_activation_pct controls when auto-trail activates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math
import pandas as pd


class OrderSide(str, Enum):
    LONG  = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"


class OrderStatus(str, Enum):
    PENDING  = "pending"
    OPEN     = "open"
    CLOSED   = "closed"
    EXPIRED  = "expired"
    CANCELED = "canceled"


@dataclass
class Order:
    side:        OrderSide
    order_type:  OrderType
    limit_price: float          # for market orders, use current close
    sl:          float
    tp:          float
    size:        float          # position size in units (oz, contracts, shares, etc)
    placed_bar:  int
    tag:         str   = ""     # strategy-defined label (e.g. zone_id)
    expiry_bars: int   = 50     # cancel if not filled within N bars
    # filled in by broker
    status:      OrderStatus = OrderStatus.PENDING
    fill_bar:    int   = -1
    fill_price:  float = math.nan
    exit_bar:    int   = -1
    exit_price:  float = math.nan
    exit_reason: str   = ""
    pnl_gross:   float = math.nan
    pnl_net:     float = math.nan
    commission:  float = 0.0


# Alias for clarity
Trade = Order


@dataclass
class BrokerConfig:
    """
    Broker-level configuration. Strategy-agnostic.

    Args:
        commission_pct      : commission as % of trade value (e.g. 0.001 = 0.1%)
        commission_flat     : flat $ per 'lot_size' units (e.g. $6 per 100 oz lot)
        lot_size            : units per "lot" for flat commission (e.g. 100 oz)
        max_concurrent      : max simultaneous open positions
        risk_usd            : fixed $ risk per trade (used if size_mode='fixed_risk')
        size_mode           : 'fixed_risk' | 'fixed_size' | 'pct_equity'
        fixed_size          : units to trade (if size_mode='fixed_size')
        pct_equity_risk     : fraction of equity to risk (if size_mode='pct_equity')
        min_size            : minimum position size (default 1)
        slippage_pct        : slippage as fraction of price (applied at entry, market only)
        slippage_fixed      : fixed slippage in price units (applied at entry, market only)
                              combined with slippage_pct — whichever is larger or both additive
        spread              : bid/ask spread in price units (e.g. 0.30 for XAUUSD).
                              LONG entries fill at ask (price + spread).
                              SHORT exits fill at ask (sl/tp + spread).
                              Net cost = one spread per round trip.
        trail_pct           : auto-trailing stop distance as fraction of price
                              (e.g. 0.001 = trail at 0.1% below peak). 0 = disabled.
        trail_activation_pct: auto-trail activates when trade profit reaches this
                              fraction of entry price (e.g. 0.002 = 0.2%). Defaults
                              to trail_pct so trail activates immediately at open.
        intrabar_path_model : Use OHLC bar direction to infer SL/TP order when both
                              are hit in the same bar. Bullish bar (close>=open): low
                              arrived before high, so LONG SL hits before TP. Bearish
                              bar: high arrived before low, so LONG TP hits before SL.
                              More realistic than always assuming SL wins (conservative).
                              Default False for backward compatibility.
        gap_fill            : When a bar opens through an SL or TP level (gap/spike),
                              fill at the bar open price rather than the SL/TP level.
                              This is what happens in live trading — you get the gap
                              open, not your order price. Default True.
        spread_schedule     : Dict {hour (int, UTC): spread_multiplier (float)}.
                              The base spread is multiplied by the factor active at
                              each bar's UTC hour. Use to model wider spreads during
                              Asian session or around market open/close.
                              Example: {0: 2.0, 8: 1.0, 17: 1.5, 22: 2.0}
                              Empty dict = constant spread (default).
        slippage_atr_mult   : Scale market-order slippage by a multiple of ATR.
                              Total slippage = slippage_fixed + slippage_pct*price
                                              + slippage_atr_mult * ATR.
                              ATR is provided by the engine each bar. 0 = disabled.
    """
    commission_pct:        float = 0.0
    commission_flat:       float = 6.0      # $ per lot round-trip
    lot_size:              float = 100.0    # units per lot (100 oz for XAUUSD)
    max_concurrent:        int   = 1
    risk_usd:              float = 20.0
    size_mode:             str   = "fixed_risk"   # fixed_risk | fixed_size | pct_equity
    fixed_size:            float = 1.0
    pct_equity_risk:       float = 0.01
    min_size:              float = 1.0
    slippage_pct:          float = 0.0
    slippage_fixed:        float = 0.0      # price units, additive with slippage_pct
    spread:                float = 0.0      # bid/ask spread in price units
    trail_pct:             float = 0.0      # 0 = no auto-trail
    trail_activation_pct:  float = 0.0      # activate auto-trail when profit >= this
    intrabar_path_model:   bool  = False    # use OHLC direction to order SL/TP
    gap_fill:              bool  = True     # fill at open if price gaps through SL/TP
    spread_schedule:       dict  = field(default_factory=dict)  # {hour_utc: multiplier}
    slippage_atr_mult:     float = 0.0      # extra slippage as multiple of ATR


class Broker:
    """
    Virtual broker — fully strategy-agnostic.

    The engine calls `broker.on_bar(i, open_, high, low, close, equity)` each
    bar. The broker tries to fill any pending order, manages open positions,
    applies trailing stops, and returns updated equity.

    Strategies call `broker.place_order(...)` to submit orders.
    For manual trailing stops, call `broker.trail_sl(new_sl)` in on_bar().
    For breakeven, call `broker.move_sl_to_breakeven()`.
    """

    def __init__(self, config: BrokerConfig, starting_equity: float) -> None:
        self.config   = config
        self.equity   = starting_equity
        self._pending: Optional[Order] = None
        self._open:    list[Order]     = []
        self._history: list[Order]     = []

    # ── Strategy interface ────────────────────────────────────────────────────

    def place_order(self, side: OrderSide, order_type: OrderType,
                    limit_price: float, sl: float, tp: float,
                    placed_bar: int, tag: str = "",
                    expiry_bars: int = 50,
                    size: Optional[float] = None) -> Optional[Order]:
        """
        Place a limit or market order.
        Returns the Order object (status=PENDING) or None if rejected
        (e.g. max_concurrent reached or pending order already exists).
        """
        cfg = self.config
        if self._pending is not None:
            return None   # already have a pending order
        if len(self._open) >= cfg.max_concurrent:
            return None   # position limit reached

        # Position sizing — uses limit_price (mid) as the reference for SL distance
        if size is None:
            size = self._calc_size(limit_price, sl)
        if size <= 0:
            return None

        order = Order(
            side=side, order_type=order_type,
            limit_price=limit_price, sl=sl, tp=tp,
            size=size, placed_bar=placed_bar,
            tag=tag, expiry_bars=expiry_bars,
        )
        self._pending = order
        return order

    def cancel_pending(self) -> None:
        """Cancel the pending order if one exists."""
        if self._pending is not None:
            self._pending.status = OrderStatus.CANCELED
            self._history.append(self._pending)
            self._pending = None

    def trail_sl(self, new_sl: float,
                 trade: Optional[Order] = None) -> bool:
        """
        Manually update the stop-loss of an open trade.

        Called from strategy.on_bar() to implement custom trailing logic.
        Only moves the SL in the favourable direction — you cannot widen it.

        Args:
            new_sl : new stop-loss price
            trade  : specific trade to update (default = most recently filled)

        Returns:
            True if SL was updated, False if rejected (wrong direction or no trade).
        """
        t = trade if trade is not None else (self._open[-1] if self._open else None)
        if t is None:
            return False
        if t.side == OrderSide.LONG and new_sl > t.sl:
            t.sl = new_sl
            return True
        if t.side == OrderSide.SHORT and new_sl < t.sl:
            t.sl = new_sl
            return True
        return False   # would widen the stop — reject silently

    def move_sl_to_breakeven(self, trade: Optional[Order] = None) -> bool:
        """
        Move stop-loss to the fill price (breakeven) for an open trade.

        Only moves the SL if we're currently in profit — prevents accidentally
        worsening the stop on a trade that's underwater.

        Args:
            trade : specific trade (default = most recently filled)

        Returns:
            True if SL was updated, False otherwise.
        """
        t = trade if trade is not None else (self._open[-1] if self._open else None)
        if t is None or math.isnan(t.fill_price):
            return False
        return self.trail_sl(t.fill_price, trade=t)

    @property
    def pending(self) -> Optional[Order]:
        return self._pending

    @property
    def open_positions(self) -> list[Order]:
        return list(self._open)

    @property
    def n_open(self) -> int:
        return len(self._open)

    @property
    def has_capacity(self) -> bool:
        return (self._pending is None and
                len(self._open) < self.config.max_concurrent)

    @property
    def closed_trades(self) -> list[Order]:
        return [o for o in self._history if o.status == OrderStatus.CLOSED]

    # ── Engine interface ──────────────────────────────────────────────────────

    def on_bar(self, bar_idx: int,
               open_: float, high: float, low: float, close: float,
               bar_time: Optional["pd.Timestamp"] = None,
               atr: Optional[float] = None) -> None:
        """
        Called by engine for each bar. Updates pending fill and manages positions.
        Must be called AFTER strategy.on_bar() so strategy can't react to fills
        that haven't happened yet.

        Args:
            bar_idx  : current bar index (0-based)
            open_    : bar open price
            high     : bar high
            low      : bar low
            close    : bar close
            bar_time : UTC timestamp of this bar (enables spread_schedule lookup)
            atr      : current ATR value from engine (enables slippage_atr_mult)

        Critical look-ahead rules enforced here:
          1. Orders placed at bar i are only eligible for fill on bar i+1 or later
             (placed_bar < bar_idx, strict). This prevents same-bar signal+fill.
          2. Newly filled orders are NOT checked for SL/TP on their fill bar —
             they enter SL/TP monitoring starting the following bar.
          3. Auto-trailing applies only to orders NOT filled this bar.
        """
        cfg = self.config

        # Resolve effective spread for this bar (applies spread_schedule if set)
        eff_spread = self._get_effective_spread(bar_time)

        # 1. Try to fill pending order — only on bars AFTER placement (strict <)
        just_filled: set[int] = set()
        if self._pending is not None and self._pending.placed_bar < bar_idx:
            t = self._pending
            filled  = False
            fill_px = math.nan

            if t.order_type == OrderType.MARKET:
                # Market: fill at open, applying spread, fixed, pct, and ATR slippage
                base    = open_
                slip    = (base * cfg.slippage_pct
                           + cfg.slippage_fixed
                           + (atr * cfg.slippage_atr_mult if atr and cfg.slippage_atr_mult else 0.0))
                fill_px = self._apply_spread_entry(base + slip, t.side, eff_spread)
                filled  = True

            elif t.order_type == OrderType.LIMIT:
                if t.side == OrderSide.LONG and low <= t.limit_price:
                    # Gap down: fill at open if open already below limit
                    base    = open_ if open_ <= t.limit_price else t.limit_price
                    fill_px = self._apply_spread_entry(base, t.side, eff_spread)
                    filled  = True
                elif t.side == OrderSide.SHORT and high >= t.limit_price:
                    base    = open_ if open_ >= t.limit_price else t.limit_price
                    fill_px = self._apply_spread_entry(base, t.side, eff_spread)
                    filled  = True
                elif (bar_idx - t.placed_bar) >= t.expiry_bars:
                    t.status = OrderStatus.EXPIRED
                    self._history.append(t)
                    self._pending = None

            if filled:
                t.fill_bar   = bar_idx
                t.fill_price = fill_px
                t.status     = OrderStatus.OPEN
                comm_entry   = self._calc_commission(t.size) * 0.5
                t.commission += comm_entry
                self.equity  -= comm_entry
                just_filled.add(id(t))
                self._open.append(t)
                self._pending = None

        # 2. Manage open positions — skip SL/TP for orders filled this same bar.
        #
        #    Per-trade order of operations (matters for correctness):
        #      a) Gap fill check  — against the SL/TP that existed at bar open.
        #                           Must run BEFORE trail so we don't accidentally
        #                           trail the SL above the open price and then
        #                           immediately fire a false gap-fill exit.
        #      b) Auto-trail      — moves SL using bar high/low (intrabar).
        #                           Only runs if no gap fill this bar.
        #      c) Normal SL/TP    — checks bar high/low against the (possibly
        #                           updated) SL/TP after trailing.
        still_open: list[Order] = []
        for t in self._open:
            if id(t) in just_filled:
                still_open.append(t)
                continue

            exit_px: float = math.nan
            reason:  str   = ""

            # ── a) Gap fill check (against pre-trail SL/TP) ─────────────────
            # If bar opens through SL or TP (weekend gap, news spike), the real
            # fill is at the bar open — not the original SL/TP price.
            if cfg.gap_fill:
                if t.side == OrderSide.LONG:
                    if open_ <= t.sl:          # gapped down through SL
                        exit_px = self._apply_spread_exit(open_, t.side, eff_spread)
                        reason  = "sl"
                    elif open_ >= t.tp:        # gapped up through TP
                        exit_px = self._apply_spread_exit(open_, t.side, eff_spread)
                        reason  = "tp"
                else:
                    if open_ >= t.sl:          # gapped up through SL (short)
                        exit_px = self._apply_spread_exit(open_, t.side, eff_spread)
                        reason  = "sl"
                    elif open_ <= t.tp:        # gapped down through TP (short)
                        exit_px = self._apply_spread_exit(open_, t.side, eff_spread)
                        reason  = "tp"

            # ── b) Auto-trailing stop (only when no gap fill this bar) ───────
            #    Uses bar high/low as the reference point for trailing distance.
            #    Only activates when profit has reached trail_activation_pct.
            if not reason and cfg.trail_pct > 0:
                if t.side == OrderSide.LONG:
                    profit_pct = (high - t.fill_price) / t.fill_price
                    if profit_pct >= cfg.trail_activation_pct:
                        new_sl = high * (1.0 - cfg.trail_pct)
                        if new_sl > t.sl:
                            t.sl = new_sl
                else:
                    profit_pct = (t.fill_price - low) / t.fill_price
                    if profit_pct >= cfg.trail_activation_pct:
                        new_sl = low * (1.0 + cfg.trail_pct)
                        if new_sl < t.sl:
                            t.sl = new_sl

            # ── c) Normal intrabar check (only if no gap fill) ───────────────
            if not reason:
                sl_hit = tp_hit = False
                if t.side == OrderSide.LONG:
                    sl_hit = low  <= t.sl
                    tp_hit = high >= t.tp
                else:
                    sl_hit = high >= t.sl
                    tp_hit = low  <= t.tp

                if sl_hit or tp_hit:
                    if sl_hit and tp_hit:
                        # Both in same bar — use intrabar path model if enabled,
                        # otherwise conservative (SL wins).
                        # Intrabar path model: bullish bar → low came first →
                        # SL hit before TP for LONG. Bearish bar → high came first
                        # → TP hit before SL for LONG.  Mirror logic for SHORT.
                        if cfg.intrabar_path_model:
                            bullish = close >= open_
                            if t.side == OrderSide.LONG:
                                # bullish: low first → SL wins; bearish: high first → TP wins
                                first_sl = bullish
                            else:
                                # bearish: high first → SL wins; bullish: low first → TP wins
                                first_sl = not bullish
                        else:
                            first_sl = True   # conservative: SL always wins

                        if first_sl:
                            exit_px = self._apply_spread_exit(t.sl, t.side, eff_spread)
                            reason  = "sl"
                        else:
                            exit_px = self._apply_spread_exit(t.tp, t.side, eff_spread)
                            reason  = "tp"
                    elif sl_hit:
                        exit_px = self._apply_spread_exit(t.sl, t.side, eff_spread)
                        reason  = "sl"
                    else:
                        exit_px = self._apply_spread_exit(t.tp, t.side, eff_spread)
                        reason  = "tp"

            if reason:
                gross = ((exit_px - t.fill_price) if t.side == OrderSide.LONG
                         else (t.fill_price - exit_px)) * t.size
                comm_exit    = self._calc_commission(t.size) * 0.5
                t.commission += comm_exit
                t.exit_bar    = bar_idx
                t.exit_price  = exit_px
                t.exit_reason = reason
                t.pnl_gross   = gross
                t.pnl_net     = gross - comm_exit
                t.status      = OrderStatus.CLOSED
                self.equity  += t.pnl_net
                self._history.append(t)
            else:
                still_open.append(t)
        self._open = still_open

    def close_all_at(self, bar_idx: int, price: float,
                     bar_time: Optional["pd.Timestamp"] = None) -> None:
        """Close all open positions at a given price (end-of-data MTM)."""
        eff_spread = self._get_effective_spread(bar_time)
        for t in self._open:
            exit_px = self._apply_spread_exit(price, t.side, eff_spread)
            gross   = ((exit_px - t.fill_price) if t.side == OrderSide.LONG
                       else (t.fill_price - exit_px)) * t.size
            comm          = self._calc_commission(t.size) * 0.5
            t.commission += comm
            t.exit_bar    = bar_idx
            t.exit_price  = exit_px
            t.exit_reason = "end_of_data"
            t.pnl_gross   = gross
            t.pnl_net     = gross - comm
            t.status      = OrderStatus.CLOSED
            self.equity  += t.pnl_net
            self._history.append(t)
        self._open = []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_effective_spread(self,
                               bar_time: Optional["pd.Timestamp"] = None) -> float:
        """
        Return the spread for this bar, applying the spread_schedule multiplier
        if one is configured.

        spread_schedule is a dict {hour_utc (int): multiplier (float)}.
        The multiplier whose key is the largest hour ≤ current bar hour applies.
        Wraps around midnight: if no key ≤ current hour, uses the largest key.

        Example schedule: {0: 2.0, 8: 1.0, 17: 1.5, 22: 2.0}
          - 00:00–07:59 UTC: 2× spread  (Asian off-hours, wide spread)
          - 08:00–16:59 UTC: 1× spread  (London/NY overlap, tight)
          - 17:00–21:59 UTC: 1.5× spread
          - 22:00–23:59 UTC: 2× spread  (pre-Asia)
        """
        base = self.config.spread
        if not base or not self.config.spread_schedule or bar_time is None:
            return base
        hour = bar_time.hour
        sched = self.config.spread_schedule
        # Find largest key ≤ current hour; wrap to max key if none found
        candidates = [h for h in sched if h <= hour]
        key = max(candidates) if candidates else max(sched.keys())
        return base * sched[key]

    def _apply_spread_entry(self, price: float, side: OrderSide,
                             spread: Optional[float] = None) -> float:
        """
        Apply bid/ask spread at entry.

        LONG entries fill at ask (price + spread) — you pay more to get in.
        SHORT entries fill at bid (price) — no spread cost on entry.

        This ensures spread is always paid exactly once per round trip.
        """
        eff = spread if spread is not None else self.config.spread
        if eff <= 0 or side == OrderSide.SHORT:
            return price
        return price + eff

    def _apply_spread_exit(self, price: float, side: OrderSide,
                            spread: Optional[float] = None) -> float:
        """
        Apply bid/ask spread at exit (SL, TP, or end-of-data).

        LONG exits sell at bid (price) — no spread cost on exit.
        SHORT exits buy back at ask (price + spread) — pays spread on exit.
        """
        eff = spread if spread is not None else self.config.spread
        if eff <= 0 or side == OrderSide.LONG:
            return price
        return price + eff

    def _calc_size(self, limit: float, sl: float) -> float:
        cfg     = self.config
        sl_dist = abs(limit - sl)
        if sl_dist <= 0:
            return cfg.min_size

        if cfg.size_mode == "fixed_risk":
            raw = cfg.risk_usd / (sl_dist * cfg.lot_size)
            return max(cfg.min_size, round(raw))

        elif cfg.size_mode == "fixed_size":
            return cfg.fixed_size

        elif cfg.size_mode == "pct_equity":
            risk_amt = self.equity * cfg.pct_equity_risk
            raw = risk_amt / (sl_dist * cfg.lot_size)
            return max(cfg.min_size, round(raw))

        return cfg.min_size

    def _calc_commission(self, size: float) -> float:
        cfg = self.config
        flat = (size / cfg.lot_size) * cfg.commission_flat
        pct  = size * cfg.commission_pct
        return flat + pct
