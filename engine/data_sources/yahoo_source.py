"""
engine/data_sources/yahoo_source.py — Yahoo Finance data source
================================================================
Cross-platform market data via yfinance.  Works on Mac, Linux, and Windows.
No account or API key required — data is fetched from Yahoo Finance's
public API.

Best suited for
---------------
- Strategy development and quick validation on Mac/Linux
- Daily bar backtests (unlimited history, ~20 years)
- Multi-day/swing strategies that don't need M1 data

Limitations vs MT5
------------------
| Feature          | Yahoo Finance        | MT5                         |
|------------------|----------------------|-----------------------------|
| M1 bars          | Last 7 days only     | ~7–10 years                 |
| Daily bars       | ~20 years ✓          | ~10 years ✓                 |
| Spread per bar   | Not available        | Full history ✓              |
| Broker defaults  | Not available        | Commission, lot size ✓      |
| Cross-platform   | ✓ Mac / Linux / Win  | Windows only                |

Common gold / XAUUSD symbols
-----------------------------
GC=F      — Gold Futures (CME, USD/troy oz) — most liquid, best proxy
XAUUSD=X  — Spot Gold (interbank); not always available
GLD       — SPDR Gold Trust ETF (highly liquid, ~1/10 oz per share)
IAU       — iShares Gold Trust ETF (alternative to GLD)

Requires
--------
pip install yfinance
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None           # type: ignore[assignment]
    _YF_AVAILABLE = False


# yfinance interval string → maximum lookback in days (Yahoo Finance API limits).
# None means effectively unlimited (daily and coarser).
_INTERVAL_MAX_DAYS: dict[str, Optional[int]] = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h":  730,
    "90m": 60,
    "1d":  None,
    "5d":  None,
    "1wk": None,
    "1mo": None,
}

# Pre-populated common gold symbols shown in the UI dropdown.
GOLD_SYMBOLS: dict[str, str] = {
    "GC=F":     "Gold Futures (CME) — most liquid",
    "XAUUSD=X": "Spot Gold — interbank",
    "GLD":      "SPDR Gold ETF — highly liquid proxy",
    "IAU":      "iShares Gold ETF — alternative proxy",
}


class YahooSource:
    """
    Yahoo Finance data source via yfinance.

    Cross-platform — works on Mac, Linux, Windows without any account,
    API key, or running terminal.  Best suited for daily-bar strategies
    or quick M1 validation on recent data.

    Usage
    -----
    src = YahooSource()
    if src.is_available():
        df = src.fetch_bars("GC=F", "1d", from_dt, to_dt)
        warn = YahooSource.interval_warning("1m", from_dt, to_dt)
    """

    def is_available(self) -> bool:
        """Return True if the yfinance package is installed."""
        return _YF_AVAILABLE

    @staticmethod
    def platform_note() -> str:
        if not _YF_AVAILABLE:
            return "Run `pip install yfinance` to enable Yahoo Finance as a data source."
        return ""

    @staticmethod
    def interval_warning(interval: str, from_date: datetime, to_date: datetime) -> Optional[str]:
        """
        Return a warning string if the requested interval/range exceeds Yahoo's
        API limits, otherwise None.

        Yahoo Finance silently returns an empty result (no error) when you ask
        for M1 data older than 7 days — this warning surfaces that to the user.
        """
        max_days = _INTERVAL_MAX_DAYS.get(interval)
        if max_days is None:
            return None     # daily and coarser — no lookback limit

        now      = datetime.now()
        from_dt  = from_date.replace(tzinfo=None) if getattr(from_date, "tzinfo", None) else from_date
        lookback = (now - from_dt).days

        if lookback > max_days:
            return (
                f"⚠️ Yahoo Finance limits **{interval}** data to the last "
                f"**{max_days} days**.  Your requested range covers {lookback} days — "
                f"data before {(now - __import__('datetime').timedelta(days=max_days)).date()} "
                f"will be missing.\n\n"
                f"For long-history M1 backtests, export from MT5 as a CSV and use "
                f"the **CSV** source."
            )
        return None

    def fetch_bars(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """
        Download OHLCV bars from Yahoo Finance.

        Args:
            symbol   : Yahoo Finance ticker, e.g. "GC=F", "XAUUSD=X", "GLD"
            interval : "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"
            from_date: start of range (UTC-aware or naive)
            to_date  : end of range

        Returns:
            DataFrame with a UTC-aware DatetimeIndex and columns:
            open, high, low, close, volume
            Note: no 'spread' column — Yahoo Finance doesn't provide bid/ask.

        Raises:
            RuntimeError if yfinance is not installed or no data is returned.
        """
        if not _YF_AVAILABLE:
            raise RuntimeError(
                "yfinance not installed.\n"
                "Install with: pip install yfinance"
            )

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=from_date,
            end=to_date,
            interval=interval,
            auto_adjust=True,
            prepost=False,
        )

        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for {symbol} ({interval}) "
                f"{from_date} → {to_date}.\n"
                f"Check the symbol name and date range.  "
                f"For intraday intervals, Yahoo Finance only provides recent history."
            )

        # Normalise column names to lowercase
        df.columns = df.columns.str.lower()

        # Ensure UTC-aware DatetimeIndex
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.sort_index(inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        # Keep only the standard OHLCV columns (drop dividends / stock splits)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    def get_broker_defaults(self, symbol: str = "") -> dict:
        """Yahoo Finance provides no broker metadata — return empty defaults."""
        return {}

    def build_spread_schedule(self, df: pd.DataFrame) -> dict:
        """Yahoo Finance has no spread data — return empty schedule."""
        return {}
