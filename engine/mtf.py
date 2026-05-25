"""
engine/mtf.py — Multi-Timeframe Support
=========================================

Allows a strategy to access multiple timeframes simultaneously with full
look-ahead protection across all of them.

Core guarantee:
  When the primary feed (e.g. M1) is at bar corresponding to time T, each
  secondary feed (H1, M15, etc.) is only visible up to and including the last
  bar that COMPLETED before time T. The current bar of any higher timeframe
  is still in progress and cannot be seen.

  Example: primary M1 cursor at 14:35
    - H1 bar 14:00–14:59 is currently in progress → NOT visible
    - Last visible H1 bar: 13:00–13:59 (completed at 14:00)
    - M15 bar 14:30–14:44 is in progress → NOT visible
    - Last visible M15 bar: 14:15–14:29 (completed at 14:30)

This is stricter than most multi-timeframe implementations which incorrectly
show the current higher-timeframe bar as it forms.

Usage in a strategy:
    class MyMTFStrategy(MTFStrategy):
        TIMEFRAMES = ["M1", "H1"]   # first = primary

        def prepare_mtf(self, feeds: dict[str, DataFeed]) -> None:
            h1_close = feeds["H1"].close._data
            feeds["H1"]._attach("h1_ema50", ind.ema(h1_close, 50))
            m1_close = feeds["M1"].close._data
            feeds["M1"]._attach("atr", ind.atr(...))

        def on_bar_mtf(self, feeds: dict[str, DataFeed], broker: Broker) -> None:
            i = feeds["M1"].i
            h1_bias = feeds["H1"]["h1_ema50"][feeds["H1"].i]   # last COMPLETE H1 bar
            m1_close = feeds["M1"].close[i]
            ...

See strategies/ma_cross_mtf.py for a complete example.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DataFeed
from .broker import Broker
from .strategy import BaseStrategy


# ── Cursor alignment ───────────────────────────────────────────────────────────

def build_secondary_cursor_map(
    primary_times:   pd.DatetimeIndex,
    secondary_times: pd.DatetimeIndex,
) -> np.ndarray:
    """
    For each primary bar i, compute the index of the last COMPLETED secondary
    bar — i.e. the last secondary bar whose opening time is strictly before
    primary_times[i].

    The "strictly before" rule ensures the current secondary bar (which may
    only be partially formed) is never visible from the primary feed.

    Returns:
        Array of shape (len(primary_times),) with int32 values.
        Value -1 means no secondary bar is visible yet (primary is before all
        secondary bars).

    Example (M1 primary, H1 secondary):
        primary  = [14:00, 14:01, ..., 14:59, 15:00, ...]
        secondary = [13:00, 14:00, 15:00, ...]
        cursor_map[bar@14:00] = index of 13:00  (14:00 H1 just started, not complete)
        cursor_map[bar@14:59] = index of 13:00  (14:00 H1 still in progress)
        cursor_map[bar@15:00] = index of 14:00  (14:00 H1 just completed)
    """
    # Convert to int64 nanoseconds for fast numpy searchsorted
    p_ns = primary_times.astype("int64").to_numpy()
    s_ns = secondary_times.astype("int64").to_numpy()

    # A secondary bar at index i is COMPLETE only when its NEXT bar starts.
    # So we compare against "close times" = [s_ns[1], s_ns[2], ..., inferred_last_close].
    # We infer the last close from the observed period (s_ns[-1] - s_ns[-2]).
    # This prevents looking at a partially-formed higher-timeframe bar — the
    # standard mistake in most open-source MTF implementations.
    #
    # searchsorted(close_ns, T, 'right') counts how many bars have closed by T.
    # Subtract 1 for the 0-based index.  Result of -1 = no bar closed yet.
    n = len(s_ns)
    close_ns = np.empty(n, dtype=np.int64)
    if n > 1:
        close_ns[:-1] = s_ns[1:]
        period = s_ns[-1] - s_ns[-2]          # infer period from last two bars
        close_ns[-1] = s_ns[-1] + period
    else:
        # Single secondary bar: use primary bar's resolution as fallback period
        p_period = p_ns[1] - p_ns[0] if len(p_ns) > 1 else int(60e9)
        close_ns[0] = s_ns[0] + p_period

    idx = np.searchsorted(close_ns, p_ns, side="right") - 1   # -1 means nothing visible yet
    return idx.astype(np.int32)


class MultiTimeframeFeed:
    """
    Container for multiple DataFeeds aligned to a primary timeframe.

    The engine calls `advance(primary_bar_idx)` each bar. Each secondary feed's
    cursor is automatically set to the last completed bar relative to the
    primary bar's timestamp.

    Strategies access feeds via indexing: `feeds["H1"].close[feeds["H1"].i]`
    """

    def __init__(
        self,
        primary_key:    str,
        feeds:          dict[str, DataFeed],
        cursor_maps:    dict[str, np.ndarray],
    ) -> None:
        self._primary_key = primary_key
        self._feeds       = feeds
        self._cursor_maps = cursor_maps   # secondary_key → cursor_map array

    def advance(self, primary_bar_idx: int) -> None:
        """Advance primary cursor and all secondary cursors for this primary bar."""
        self._feeds[self._primary_key]._advance(primary_bar_idx)
        for key, cmap in self._cursor_maps.items():
            secondary_idx = int(cmap[primary_bar_idx])
            self._feeds[key]._advance(secondary_idx)

    def __getitem__(self, key: str) -> DataFeed:
        return self._feeds[key]

    @property
    def primary(self) -> DataFeed:
        return self._feeds[self._primary_key]

    @property
    def primary_key(self) -> str:
        return self._primary_key

    def keys(self):
        return self._feeds.keys()


def build_mtf_feeds(
    dfs:         dict[str, pd.DataFrame],
    primary_key: str,
    symbol:      str = "UNKNOWN",
) -> MultiTimeframeFeed:
    """
    Build a MultiTimeframeFeed from a dict of DataFrames.

    Args:
        dfs         : Dict of timeframe_label → OHLCV DataFrame with DatetimeIndex
        primary_key : Which timeframe is the primary (drives the bar loop)
        symbol      : Instrument name (for DataFeed metadata)

    Returns:
        MultiTimeframeFeed ready to be passed to run_backtest_mtf()
    """
    if primary_key not in dfs:
        raise KeyError(
            f"primary_key '{primary_key}' not found in dfs. "
            f"Available keys: {list(dfs.keys())}"
        )

    feeds:       dict[str, DataFeed]   = {}
    cursor_maps: dict[str, np.ndarray] = {}
    primary_times = dfs[primary_key].index

    for key, df in dfs.items():
        feeds[key] = DataFeed(df, symbol=symbol, timeframe=key)

    for key in dfs:
        if key == primary_key:
            continue
        cursor_maps[key] = build_secondary_cursor_map(
            primary_times, dfs[key].index
        )

    return MultiTimeframeFeed(primary_key, feeds, cursor_maps)


# ── MTFStrategy base class ────────────────────────────────────────────────────

class MTFStrategy(BaseStrategy):
    """
    Extended BaseStrategy for multi-timeframe strategies.

    Override `prepare_mtf` and `on_bar_mtf` instead of `prepare` and `on_bar`.
    The default implementations of `prepare` and `on_bar` delegate to the MTF
    versions using only the primary feed (backward-compatible with single-TF use).

    Subclass interface:
        TIMEFRAMES = ["M1", "H1"]   # first entry = primary timeframe

        def prepare_mtf(self, feeds: dict[str, DataFeed]) -> None:
            ...compute indicators on all feeds...

        def on_bar_mtf(self, feeds: dict[str, DataFeed], broker: Broker) -> None:
            ...signal logic using all feeds...
    """

    #: List of required timeframes. First = primary (drives the loop).
    TIMEFRAMES: list[str] = []

    def prepare_mtf(self, feeds: dict[str, DataFeed]) -> None:
        """
        Called ONCE before the bar loop with all feeds available.
        Access raw arrays via feed.close._data (no cursor guard here).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement prepare_mtf(feeds)"
        )

    def on_bar_mtf(self, feeds: dict[str, DataFeed], broker: Broker) -> None:
        """
        Called every primary bar. All secondary feeds are cursor-guarded.
        Access indicators via feeds['H1']['ema50'][feeds['H1'].i]
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement on_bar_mtf(feeds, broker)"
        )

    # Satisfy BaseStrategy ABC — delegates to MTF methods using primary-only feed
    def prepare(self, feed: DataFeed) -> None:
        self.prepare_mtf({self.TIMEFRAMES[0] if self.TIMEFRAMES else "primary": feed})

    def on_bar(self, feed: DataFeed, broker: Broker) -> None:
        self.on_bar_mtf({self.TIMEFRAMES[0] if self.TIMEFRAMES else "primary": feed}, broker)
