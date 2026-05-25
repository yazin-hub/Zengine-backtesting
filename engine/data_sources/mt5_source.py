"""
engine/data_sources/mt5_source.py — MetaTrader 5 data source
=============================================================
Connects to the MT5 terminal running on the local Windows machine and
retrieves OHLC bar data with per-bar spread, plus broker config defaults
for automatic BrokerConfig population.

No login credentials are required — the terminal handles broker
authentication.  This module communicates only with the local terminal
process (like an IPC call), not directly with the broker's server.

What you get
------------
- OHLC bars       : typically 7–10 years of M1 history (broker-dependent)
- Spread per bar  : actual bid/ask spread in price units at each bar
- Symbol info     : contract size, commission, tick size for BrokerConfig
- Spread schedule : auto-derived hourly multipliers from bar spread data

Limitations
-----------
- Windows only (the MT5 terminal application is Windows-only)
- Tick data available via this module is NOT implemented; MT5 brokers
  typically only keep 2–4 weeks of tick history, which is insufficient
  for multi-year backtests.

Requires
--------
    pip install MetaTrader5        (Windows only)
    MT5 terminal running and logged in

Reference
---------
MetaTrader5 Python API: https://www.mql5.com/en/docs/python_metatrader5
Based on usage patterns from Dobrovolsky et al. (2021) — "Automated
Trading with Python and MetaTrader 5" (MQL5 community documentation).
"""

from __future__ import annotations

import platform
from datetime import datetime

import numpy as np
import pandas as pd


# Guard import — MT5 package is Windows-only.  All other modules work
# fine on Mac/Linux even if this import fails.
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None          # type: ignore[assignment]
    _MT5_AVAILABLE = False


# Timeframe string → MT5 timeframe constant mapping.
# Populated lazily (after confirming MT5 is importable).
_TF_MAP: dict[str, int] = {}


def _build_tf_map() -> None:
    """Populate _TF_MAP from MT5 constants (no-op if MT5 not importable)."""
    global _TF_MAP
    if not _MT5_AVAILABLE or mt5 is None or _TF_MAP:
        return
    _TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }


class MT5Source:
    """
    MetaTrader 5 data source.

    Pulls OHLC bar data (with per-bar spread) and broker symbol info
    directly from a running MT5 terminal.  No credentials required.

    Usage
    -----
    src = MT5Source()
    if src.is_available():
        df       = src.fetch_bars("XAUUSD", "M1", from_dt, to_dt)
        defaults = src.get_broker_defaults("XAUUSD")
        schedule = src.build_spread_schedule(df)
    """

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Return True if MT5 is importable, we're on Windows, and the terminal
        responds to an initialize() handshake.

        Safe to call on Mac/Linux — returns False without crashing.
        """
        if not _MT5_AVAILABLE:
            return False
        if platform.system() != "Windows":
            return False
        try:
            ok = mt5.initialize()
            if ok:
                mt5.shutdown()
            return bool(ok)
        except Exception:
            return False

    @staticmethod
    def platform_note() -> str:
        """
        Return a human-readable string explaining why MT5 is (un)available,
        or empty string if it's fully available.
        """
        if platform.system() != "Windows":
            return (
                "MT5 terminal is Windows-only.\n\n"
                "**Options for Mac / Linux:**\n"
                "• Upload a CSV exported from MT5 (File → Export)\n"
                "• Use Yahoo Finance below (daily bars, no account needed)\n"
                "• Run MT5 on a Windows VPS, export data as CSV and copy here\n"
                "• Use Wine (advanced) to run MT5 on Mac"
            )
        if not _MT5_AVAILABLE:
            return "Run `pip install MetaTrader5` then restart the app."
        return ""

    # ── Bar data ──────────────────────────────────────────────────────────────

    def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """
        Pull OHLC bars from the MT5 terminal.

        Args:
            symbol    : MT5 symbol name, e.g. "XAUUSD"
            timeframe : "M1", "M5", "M15", "M30", "H1", "H4", "D1"
            from_date : start of range (UTC-aware or naive, treated as UTC)
            to_date   : end of range

        Returns:
            DataFrame with a UTC-aware DatetimeIndex and columns:
            open, high, low, close, volume, spread
            spread is in price units (MT5 spread_points × tick_size).
            e.g. XAUUSD: 30 points × 0.01 = $0.30.

        Raises:
            RuntimeError  if MT5 unavailable, terminal not running, or no data.
            ValueError    if timeframe string is unrecognised.
        """
        if not _MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package not installed.\n"
                "Install with: pip install MetaTrader5  (Windows only)"
            )

        _build_tf_map()
        tf = _TF_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Supported: {list(_TF_MAP.keys())}"
            )

        # MT5 copy_rates_range expects UTC-naive datetimes
        from_dt = from_date.replace(tzinfo=None) if getattr(from_date, "tzinfo", None) else from_date
        to_dt   = to_date.replace(tzinfo=None)   if getattr(to_date,   "tzinfo", None) else to_date

        if not mt5.initialize():
            raise RuntimeError(
                "MT5 terminal not running or failed to connect.\n"
                "Open MetaTrader 5 and log in, then try again.\n"
                f"MT5 error: {mt5.last_error()}"
            )

        try:
            rates = mt5.copy_rates_range(symbol, tf, from_dt, to_dt)
            if rates is None or len(rates) == 0:
                raise RuntimeError(
                    f"No data returned for {symbol} {timeframe} "
                    f"{from_dt.date()} → {to_dt.date()}.\n"
                    f"Check the symbol name and date range.\n"
                    f"MT5 error: {mt5.last_error()}"
                )

            # Get tick size to convert spread from integer points to price units.
            # XAUUSD: point = 0.01, so spread=30 → $0.30
            info  = mt5.symbol_info(symbol)
            point = info.point if info else 0.01

            df = pd.DataFrame(rates)
            df.index      = pd.to_datetime(df["time"], unit="s", utc=True)
            df.index.name = None

            # spread in MT5 rates is an integer (points) — convert to price units
            df["spread"] = (df["spread"] * point).round(5)

            return df[["open", "high", "low", "close", "tick_volume", "spread"]].rename(
                columns={"tick_volume": "volume"}
            )

        finally:
            mt5.shutdown()

    # ── Broker defaults ───────────────────────────────────────────────────────

    def get_broker_defaults(self, symbol: str) -> dict:
        """
        Extract BrokerConfig-compatible defaults from MT5 symbol info.

        Returns a dict with keys that match BrokerConfig fields, ready to
        pre-populate the UI.  Returns empty dict if MT5 is unavailable.

        Fields returned
        ---------------
        spread          — current typical spread in price units
        commission_flat — per-lot commission (from broker's symbol info)
        lot_size        — contract size (e.g. 100 oz for XAUUSD)
        slippage_fixed  — conservative default: 5 ticks
        """
        if not _MT5_AVAILABLE or not mt5.initialize():
            return {}

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return {}

            point      = getattr(info, "point",                   0.01)
            spread_pts = getattr(info, "spread",                   30)
            contract   = getattr(info, "trade_contract_size",     100.0)
            # trade_commission_value may not exist on all broker builds
            commission = abs(getattr(info, "trade_commission_value", 0.0) or 0.0)

            return {
                "spread":          round(spread_pts * point, 5),
                "commission_flat": round(commission, 2),
                "lot_size":        float(contract),
                "slippage_fixed":  round(point * 5, 5),    # 5 ticks = conservative
            }
        finally:
            mt5.shutdown()

    # ── Spread schedule ───────────────────────────────────────────────────────

    def build_spread_schedule(self, df: pd.DataFrame) -> dict[int, float]:
        """
        Derive a spread_schedule multiplier dict from fetched bar data.

        Groups bars by UTC hour, computes mean spread per hour, then
        normalises so the tightest (lowest-spread) hour = 1.0.

        Args:
            df : DataFrame as returned by fetch_bars() — must have 'spread' column

        Returns:
            {hour_utc (int): multiplier (float)} for BrokerConfig.spread_schedule.
            Empty dict if no valid spread column.

        Example output for XAUUSD (IC Markets):
            {0: 2.1, 1: 2.3, ..., 8: 1.0, ..., 17: 1.4, ..., 22: 2.0}
        """
        if "spread" not in df.columns or df["spread"].isna().all():
            return {}

        idx = (
            df.index
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.to_datetime(df.index, utc=True)
        )
        hourly_mean = df["spread"].groupby(idx.hour).mean()

        if hourly_mean.max() == 0:
            return {}

        valid       = hourly_mean[hourly_mean > 0]
        min_spread  = valid.min()
        if np.isnan(min_spread) or min_spread == 0:
            return {}

        return {
            int(h): round(float(s / min_spread), 3)
            for h, s in hourly_mean.items()
            if not np.isnan(s) and s > 0
        }

    # ── Symbol browser ────────────────────────────────────────────────────────

    def list_symbols(self, pattern: str = "") -> list[str]:
        """
        List symbols available in the connected terminal.

        Args:
            pattern : optional filter string (e.g. "XAU", "USD"). Empty = all.

        Returns sorted list of symbol names, or empty list if unavailable.
        """
        if not _MT5_AVAILABLE or not mt5.initialize():
            return []
        try:
            syms = mt5.symbols_get(pattern) or []
            return sorted(s.name for s in syms)
        finally:
            mt5.shutdown()
