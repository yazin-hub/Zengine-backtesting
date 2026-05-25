"""
engine/strategy.py — BaseStrategy Abstract Interface
======================================================

Every strategy must subclass BaseStrategy and implement at minimum:
  - prepare(feed)   : compute causal indicators, attach to feed
  - on_bar(feed, broker) : called each bar, return optional order params

The engine guarantees:
  - feed._cursor == i when on_bar is called (bar i, 0-indexed)
  - feed blocks access to any bar > i via LookAheadError
  - broker.on_bar() is called AFTER on_bar(), so fills happen next bar

To create a strategy:
    class MyStrategy(BaseStrategy):
        NAME   = "My Strategy"
        PARAMS = {"fast": 10, "slow": 50}

        def prepare(self, feed: DataFeed) -> None:
            # Compute all indicators causally once here
            close = feed.close._data          # full array — OK here, engine
                                               # hasn't started yet
            ema = pd.Series(close).ewm(span=self.params['fast']).mean().values
            feed._attach('ema_fast', ema)

        def on_bar(self, feed: DataFeed, broker: Broker) -> None:
            if feed.n_bars < 50:
                return
            if not broker.has_capacity:
                return
            fast = feed['ema_fast'][feed.i]
            prev = feed['ema_fast'][feed.i - 1]
            if prev < 0 and fast > 0:   # crossover
                broker.place_order(...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .data import DataFeed
from .broker import Broker


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Subclass this and implement `prepare` and `on_bar`.
    Optionally override `on_fill` and `on_close` for event hooks.
    """

    #: Strategy display name (used in UI)
    NAME: str = "Unnamed Strategy"

    #: Default parameters — override in subclass.
    #: Keys become sliders/inputs in the Streamlit UI automatically.
    #: Each value can be a scalar (default) or a dict:
    #:   {"default": 10, "min": 1, "max": 50, "step": 1, "label": "Fast EMA"}
    PARAMS: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        # Extract scalar defaults from the PARAMS template
        # (PARAMS values may be plain scalars or spec-dicts like
        #  {"default": 10, "min": 1, "max": 50, ...})
        defaults: dict[str, Any] = {}
        for k, v in self.PARAMS.items():
            defaults[k] = v["default"] if isinstance(v, dict) else v

        # Normalise any overrides — callers may accidentally pass the full
        # PARAMS template dict (e.g. run_splits(df, S, S.PARAMS, ...)) which
        # contains spec-dicts rather than scalars.  Extract the default in
        # that case so self.params always contains plain values.
        resolved: dict[str, Any] = {}
        for k, v in (params or {}).items():
            resolved[k] = v["default"] if isinstance(v, dict) else v

        self.params = {**defaults, **resolved}

    @abstractmethod
    def prepare(self, feed: DataFeed) -> None:
        """
        Called ONCE before the bar loop starts.

        This is the right place to:
          - Compute causal indicators over the full array (ATR, EMA, etc.)
          - Attach them to feed via feed._attach('name', array)
          - Initialise any strategy state (zone lists, counters, etc.)

        IMPORTANT: access feed.close._data (raw numpy) here, not feed.close[i],
        since the cursor hasn't started yet. The indicators must be computed
        causally (no look-ahead) — the engine does NOT verify indicator
        computation, only index access during on_bar.
        """

    @abstractmethod
    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        """
        Called every bar after feed._cursor is advanced to bar i.

        - Read feed.close[i], feed['atr'][i], etc. (guarded — future = error)
        - Call broker.place_order(...) to submit orders
        - Do NOT read feed.close[i + 1] — LookAheadError will be raised

        Broker fills happen AFTER this method returns (next bar), so:
          - broker.pending is the order just placed (not yet filled)
          - broker.open_positions are currently filled trades
        """

    def on_fill(self, order: Any, feed: DataFeed) -> None:
        """
        Optional hook: called when a pending order is filled.
        Override to track fills, adjust stops, etc.
        Default: no-op.
        """

    def on_close(self, trade: Any, feed: DataFeed) -> None:
        """
        Optional hook: called when a trade is closed (SL, TP, or end-of-data).
        Override to log, update state, etc.
        Default: no-op.
        """

    def __repr__(self) -> str:
        return f"{self.NAME}({self.params})"
