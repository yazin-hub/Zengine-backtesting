"""
engine/indicators.py — Causal Indicator Library
=================================================

All functions here are CAUSAL — they only use data available up to bar i.
Designed to be called inside strategy.prepare() over the full array, then
attached to the feed for bar-by-bar access.

Every function takes numpy arrays and returns a numpy array of the same length.
NaN fills the warmup period where insufficient data exists.

Usage in a strategy:
    def prepare(self, feed: DataFeed) -> None:
        close = feed.close._data
        high  = feed.high._data
        low   = feed.low._data
        feed._attach('atr',    ind.atr(high, low, close))
        feed._attach('ema20',  ind.ema(close, 20))
        feed._attach('rsi14',  ind.rsi(close, 14))
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ── Trend ─────────────────────────────────────────────────────────────────────

def ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    return pd.Series(close).ewm(span=period, adjust=False).mean().to_numpy()


def sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(close).rolling(period).mean().to_numpy()


def wma(close: np.ndarray, period: int) -> np.ndarray:
    """Weighted moving average (linear weights)."""
    weights = np.arange(1, period + 1, dtype=float)
    def _wma(x):
        return np.dot(x, weights) / weights.sum()
    return pd.Series(close).rolling(period).apply(_wma, raw=True).to_numpy()


def dema(close: np.ndarray, period: int) -> np.ndarray:
    """Double EMA."""
    e = ema(close, period)
    return 2 * e - ema(e, period)


def tema(close: np.ndarray, period: int) -> np.ndarray:
    """Triple EMA."""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


# ── Volatility ────────────────────────────────────────────────────────────────

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14, method: str = "ewm") -> np.ndarray:
    """
    Average True Range.
    method: 'ewm' (default, matches most live platforms) or 'sma' (Wilder's).
    Prior close used for TR — no look-ahead.
    """
    hl  = high - low
    hpc = np.abs(high - pd.Series(close).shift(1).to_numpy())
    lpc = np.abs(low  - pd.Series(close).shift(1).to_numpy())
    tr  = np.nanmax(np.stack([hl, hpc, lpc], axis=1), axis=1)
    tr_s = pd.Series(tr)
    if method == "ewm":
        return tr_s.ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    return tr_s.rolling(period).mean().to_numpy()


def bollinger(close: np.ndarray, period: int = 20, std_dev: float = 2.0):
    """
    Bollinger Bands.
    Returns (middle, upper, lower) numpy arrays.
    """
    s   = pd.Series(close)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std(ddof=0)
    return mid.to_numpy(), (mid + std_dev * std).to_numpy(), (mid - std_dev * std).to_numpy()


def atr_bands(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              period: int = 14, multiplier: float = 2.0):
    """
    ATR-based bands (Keltner-style).
    Returns (upper, lower) numpy arrays.
    """
    mid = ema(close, period)
    a   = atr(high, low, close, period)
    return mid + multiplier * a, mid - multiplier * a


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder's smoothing)."""
    delta  = pd.Series(close).diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).to_numpy()


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD.
    Returns (macd_line, signal_line, histogram) numpy arrays.
    """
    fast_ema   = ema(close, fast)
    slow_ema   = ema(close, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               k_period: int = 14, d_period: int = 3):
    """
    Stochastic Oscillator.
    Returns (%K, %D) numpy arrays.
    """
    s_high = pd.Series(high).rolling(k_period).max()
    s_low  = pd.Series(low).rolling(k_period).min()
    k      = 100 * (pd.Series(close) - s_low) / (s_high - s_low)
    d      = k.rolling(d_period).mean()
    return k.to_numpy(), d.to_numpy()


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 20) -> np.ndarray:
    """Commodity Channel Index."""
    tp    = (high + low + close) / 3
    tp_s  = pd.Series(tp)
    ma    = tp_s.rolling(period).mean()
    mad   = tp_s.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp_s - ma) / (0.015 * mad)).to_numpy()


def mfi(high, low, close, volume, period: int = 14) -> np.ndarray:
    """Money Flow Index."""
    tp   = (high + low + close) / 3
    mf   = tp * volume
    pos  = pd.Series(np.where(np.diff(tp, prepend=tp[0]) > 0, mf, 0))
    neg  = pd.Series(np.where(np.diff(tp, prepend=tp[0]) < 0, mf, 0))
    pmf  = pos.rolling(period).sum()
    nmf  = neg.rolling(period).sum()
    mr   = pmf / nmf.replace(0, np.nan)
    return (100 - 100 / (1 + mr)).to_numpy()


# ── Trend / Structure ─────────────────────────────────────────────────────────

def swing_high(high: np.ndarray, lookback: int) -> np.ndarray:
    """
    Rolling swing high over the last `lookback` bars, shifted by 1.
    At bar i: max(high[i-lookback .. i-1]) — bar i NOT included.
    """
    return pd.Series(high).rolling(lookback).max().shift(1).to_numpy()


def swing_low(low: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling swing low, shifted by 1 bar (causal)."""
    return pd.Series(low).rolling(lookback).min().shift(1).to_numpy()


def higher_highs(high: np.ndarray, low: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Returns bool array: True where current bar makes a higher high and higher low."""
    sh = swing_high(high, lookback)
    sl = swing_low(low, lookback)
    hh = (high > sh) & (low > sl)
    return hh.astype(float)


def lower_lows(high: np.ndarray, low: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Returns bool array: True where current bar makes a lower high and lower low."""
    sh = swing_high(high, lookback)
    sl = swing_low(low, lookback)
    ll = (high < sh) & (low < sl)
    return ll.astype(float)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14):
    """
    Average Directional Index.
    Returns (adx, +di, -di) numpy arrays.
    """
    h    = pd.Series(high)
    lo   = pd.Series(low)
    c    = pd.Series(close)
    tr   = pd.concat([h - lo,
                       (h - c.shift()).abs(),
                       (lo - c.shift()).abs()], axis=1).max(axis=1)
    dm_p = np.where((h.diff() > lo.diff().abs()) & (h.diff() > 0), h.diff(), 0)
    dm_n = np.where((lo.diff().abs() > h.diff()) & (lo.diff() < 0), lo.diff().abs(), 0)
    atr_ = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    di_p = 100 * pd.Series(dm_p).ewm(alpha=1 / period, adjust=False).mean() / atr_
    di_n = 100 * pd.Series(dm_n).ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx   = 100 * (di_p - di_n).abs() / (di_p + di_n)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_.to_numpy(), di_p.to_numpy(), di_n.to_numpy()


# ── ICT / Price Action helpers ────────────────────────────────────────────────

def fair_value_gaps(high: np.ndarray, low: np.ndarray):
    """
    3-bar Fair Value Gap detection.
    Bear FVG: high[i] < low[i-2]
    Bull FVG: low[i]  > high[i-2]

    Returns:
        bear_fvg : bool array
        bull_fvg : bool array
        bfvg_top, bfvg_bot : bear FVG zone boundaries
        ufvg_top, ufvg_bot : bull FVG zone boundaries
    """
    N       = len(high)
    pad2    = np.full(2, np.nan)
    low_m2  = np.concatenate([pad2, low[:-2]])
    high_m2 = np.concatenate([pad2, high[:-2]])
    bear_fvg = np.zeros(N, bool)
    bear_fvg[2:] = high[2:] < low_m2[2:]
    bull_fvg = np.zeros(N, bool)
    bull_fvg[2:] = low[2:]  > high_m2[2:]
    bfvg_top = np.where(bear_fvg, low_m2,  np.nan)
    bfvg_bot = np.where(bear_fvg, high,    np.nan)
    ufvg_top = np.where(bull_fvg, low,     np.nan)
    ufvg_bot = np.where(bull_fvg, high_m2, np.nan)
    return bear_fvg, bull_fvg, bfvg_top, bfvg_bot, ufvg_top, ufvg_bot


def liquidity_sweeps(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     atr_arr: np.ndarray, lookback: int = 15, min_atr: float = 0.5):
    """
    ICT-style liquidity sweeps.
    Bear sweep: spike above swing_high + close back below.
    Bull sweep: spike below swing_low + close back above.
    """
    sh      = swing_high(high, lookback)
    sl      = swing_low(low, lookback)
    valid   = ~(np.isnan(atr_arr) | np.isnan(sh) | (atr_arr == 0))
    bear_sw = valid & (high > sh) & (close < sh) & ((high - sh) >= min_atr * atr_arr)
    bull_sw = valid & (low  < sl) & (close > sl) & ((sl - low)  >= min_atr * atr_arr)
    bear_sw[:lookback] = False
    bull_sw[:lookback] = False
    return bear_sw.astype(bool), bull_sw.astype(bool)


def session_mask(index: "pd.DatetimeIndex", session: str) -> np.ndarray:
    """
    Returns bool array: True during the specified session.
    Sessions: 'london' (07-12 UTC), 'ny' (13-17 UTC), 'asia' (00-07 UTC),
              'both' (london + ny), 'all' (always True).
    """
    h = index.hour
    masks = {
        "london": (h >= 7)  & (h < 12),
        "ny":     (h >= 13) & (h < 17),
        "asia":   (h >= 0)  & (h < 7),
        "both":   ((h >= 7) & (h < 12)) | ((h >= 13) & (h < 17)),
        "all":    np.ones(len(index), bool),
    }
    if session not in masks:
        raise ValueError(f"Unknown session '{session}'. Choose from {list(masks.keys())}")
    return np.asarray(masks[session])
