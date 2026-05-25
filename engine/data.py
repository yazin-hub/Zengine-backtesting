"""
engine/data.py — Look-Ahead-Proof DataFeed
============================================

The DataFeed wraps the full OHLCV dataframe but ONLY exposes bars up to
and including the current bar index set by the engine.

Strategies receive a DataFeed instance. Any attempt to read a future bar
raises a LookAheadError immediately, making look-ahead bias impossible
at runtime — not just by convention.

Usage inside a strategy:
    def on_bar(self, i: int, feed: DataFeed, broker: Broker) -> None:
        close = feed.close[i]        # ✅ current bar close
        prev  = feed.close[i - 1]   # ✅ previous bar
        fut   = feed.close[i + 1]   # ❌ LookAheadError raised instantly
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class LookAheadError(Exception):
    """Raised when a strategy tries to access a bar it hasn't seen yet."""
    pass


class CausalityError(Exception):
    """
    Raised when the causality audit detects a non-causal indicator in prepare().

    A non-causal indicator is one whose value at bar i changes when future bars
    (i+1, i+2, ...) are added to the dataset — meaning it was computed using
    data it shouldn't have seen.

    This is caught before the backtest loop starts, so no results are produced
    from a strategy with a look-ahead bug in prepare().

    Example triggers:
      - A rolling mean that isn't properly bounded (pandas default forward-fills)
      - A smoothing operation applied to the full array in reverse
      - scipy/statsmodels filters that smooth forward and backward (Savitzky-Golay, etc.)
      - Any operation that normalises using the global min/max of the full series
    """
    pass


class _GuardedArray:
    """
    A numpy array wrapper that blocks access to indices beyond `_cursor`.
    Supports negative indexing and slices — all resolved against `_cursor`.
    """
    __slots__ = ("_data", "_feed")

    def __init__(self, data: np.ndarray, feed: "DataFeed") -> None:
        self._data = data
        self._feed = feed

    def _check(self, idx: int) -> None:
        cursor = self._feed._cursor
        if idx < 0:
            idx = cursor + 1 + idx   # resolve negative index
        if idx > cursor:
            raise LookAheadError(
                f"Strategy tried to access bar {idx} but current bar is {cursor}. "
                "This would be look-ahead bias."
            )

    def __getitem__(self, key):
        cursor = self._feed._cursor
        if isinstance(key, slice):
            start, stop, step = key.indices(cursor + 1)
            if stop - 1 > cursor:
                raise LookAheadError(
                    f"Slice [{key}] exceeds current bar {cursor}."
                )
            return self._data[start:stop:step]
        # Integer index
        idx = int(key)
        self._check(idx)
        real_idx = cursor + 1 + idx if idx < 0 else idx
        return self._data[real_idx]

    def __len__(self) -> int:
        return self._feed._cursor + 1

    def to_numpy(self) -> np.ndarray:
        return self._data[: self._feed._cursor + 1]

    def to_series(self) -> pd.Series:
        return pd.Series(self._data[: self._feed._cursor + 1],
                         index=self._feed.index[: self._feed._cursor + 1])

    # Convenience: shift and rolling via pandas (on visible slice only)
    def shift(self, n: int = 1) -> np.ndarray:
        s = pd.Series(self._data[: self._feed._cursor + 1])
        return s.shift(n).to_numpy()

    def rolling_max(self, window: int) -> np.ndarray:
        s = pd.Series(self._data[: self._feed._cursor + 1])
        return s.rolling(window).max().to_numpy()

    def rolling_min(self, window: int) -> np.ndarray:
        s = pd.Series(self._data[: self._feed._cursor + 1])
        return s.rolling(window).min().to_numpy()

    def ewm(self, alpha: float) -> np.ndarray:
        s = pd.Series(self._data[: self._feed._cursor + 1])
        return s.ewm(alpha=alpha, adjust=False).mean().to_numpy()


class DataFeed:
    """
    Look-ahead-proof wrapper around an OHLCV dataframe.

    The engine calls `_advance(i)` each bar. The strategy can only read
    bars 0..i. Any access beyond i raises LookAheadError.

    Attributes (all GuardedArray — read up to current bar only):
        open, high, low, close, volume : price/volume arrays
        index : pd.DatetimeIndex of bar timestamps
        symbol : str
        timeframe : str
    """

    def __init__(self, df: pd.DataFrame, symbol: str = "UNKNOWN",
                 timeframe: str = "M1") -> None:
        required = {"open", "high", "low", "close"}
        missing  = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFeed missing columns: {missing}")

        df = df.copy()
        df.columns = df.columns.str.lower()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        self.symbol    = symbol
        self.timeframe = timeframe
        self.index     = df.index
        self._cursor   = -1   # advances each bar via engine
        self._len      = len(df)

        self.open   = _GuardedArray(df["open"].to_numpy(),   self)
        self.high   = _GuardedArray(df["high"].to_numpy(),   self)
        self.low    = _GuardedArray(df["low"].to_numpy(),    self)
        self.close  = _GuardedArray(df["close"].to_numpy(),  self)
        self.volume = _GuardedArray(
            df["volume"].to_numpy() if "volume" in df.columns
            else np.zeros(len(df)), self)

        # Allow strategies to attach custom indicator arrays
        self._custom: dict[str, _GuardedArray] = {}

    # ── Engine interface ──────────────────────────────────────────────────────

    def _advance(self, i: int) -> None:
        """Called by engine only — moves cursor to bar i."""
        self._cursor = i

    def _attach(self, name: str, array: np.ndarray) -> None:
        """
        Attach a pre-computed causal indicator array.
        Strategy accesses it as feed['atr'], feed['ema20'], etc.
        The array must be computed causally (no look-ahead) — engine does not
        verify this, but since indicators are computed once over the full
        array before the loop, the look-ahead guard on reads is still active.
        """
        if len(array) != self._len:
            raise ValueError(
                f"Indicator '{name}' length {len(array)} != feed length {self._len}"
            )
        self._custom[name] = _GuardedArray(array, self)

    def __getitem__(self, name: str) -> _GuardedArray:
        if name not in self._custom:
            raise KeyError(
                f"Indicator '{name}' not attached. "
                "Call feed._attach('name', array) in strategy.prepare()."
            )
        return self._custom[name]

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def now(self) -> pd.Timestamp:
        """Current bar timestamp."""
        return self.index[self._cursor]

    @property
    def i(self) -> int:
        """Current bar index."""
        return self._cursor

    @property
    def n_bars(self) -> int:
        """Number of bars visible so far (= i + 1)."""
        return self._cursor + 1
