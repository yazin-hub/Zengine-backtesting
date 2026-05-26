"""
connectors/base.py — BaseConnector ABC
=======================================

Abstract interface for ZEngine live execution connectors.

Any broker or exchange connector must implement this interface to be compatible
with ZEngine live runners. The contract ensures strategy portability: write
once against BaseConnector, deploy to any supported platform.

Implemented connectors:
    CTraderConnector  — cTrader Open API (Spotware)  [connectors/ctrader.py]

Planned connectors:
    BinanceConnector  — Binance Futures
    IBKRConnector     — Interactive Brokers TWS
    OANDAConnector    — OANDA v20

Example skeleton for a custom connector:

    class MyBrokerConnector(BaseConnector):
        def connect(self)    -> None:        ...
        def disconnect(self) -> None:        ...
        def is_connected(self) -> bool:      ...
        def fetch_ohlcv(self, symbol, timeframe, limit=500) -> pd.DataFrame: ...
        def place_market_order(self, symbol, side, size, sl, tp, comment="") -> OrderResult | None: ...
        def close_position(self, position_id, size=None) -> bool: ...
        def get_open_positions(self, symbol=None) -> list[Position]: ...
        def get_account_summary(self) -> dict: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """
    Result of a successfully placed and filled market order.

    Attributes:
        position_id : Broker-assigned position identifier.
        order_id    : Broker-assigned order identifier.
        symbol      : Normalised symbol name (e.g. "BTCUSD").
        side        : "buy" or "sell".
        size        : Filled size in instrument base units.
        fill_price  : Actual fill price (0.0 if not reported by broker).
        sl          : Stop-loss price applied to the position (None if unset).
        tp          : Take-profit price applied to the position (None if unset).
        timestamp   : UTC datetime of fill (or placement time if fill time unavailable).
        comment     : Order label visible in broker trade history.
    """
    position_id : int
    order_id    : int
    symbol      : str
    side        : str               # "buy" | "sell"
    size        : float
    fill_price  : float
    sl          : float | None = None
    tp          : float | None = None
    timestamp   : datetime | None = None
    comment     : str = ""


@dataclass
class Position:
    """
    An open position as reported by the broker.

    Attributes:
        position_id : Broker-assigned position identifier.
        symbol      : Normalised symbol name.
        side        : "buy" (long) or "sell" (short).
        size        : Current position size in instrument base units.
        entry_price : Average entry price.
        sl          : Current stop-loss price (None if not set).
        tp          : Current take-profit price (None if not set).
        comment     : Label attached at order placement.
    """
    position_id : int
    symbol      : str
    side        : str               # "buy" | "sell"
    size        : float
    entry_price : float
    sl          : float | None = None
    tp          : float | None = None
    comment     : str = ""


# ── BaseConnector ABC ─────────────────────────────────────────────────────────

class BaseConnector(ABC):
    """
    Abstract base class for ZEngine live execution connectors.

    A connector bridges ZEngine strategy signals to a live broker or exchange.
    Subclasses must implement all abstract methods. Two optional methods
    (cancel_pending_order, get_latest_price) have default implementations
    that may be overridden.

    Thread-safety contract:
        All public methods must be callable from the main (strategy) thread.
        Internal network I/O — reactor loops, websockets, polling threads —
        must run in daemon threads and not block the caller beyond the
        documented timeout.

    Timeframe convention:
        "1m", "5m", "15m", "1h", "4h", "1d"  (lowercase, no spaces).

    Symbol convention:
        Normalised form — no slashes, uppercase: "BTCUSD", "ETHUSD", "XAUUSD".
        Connectors normalise internally; callers may pass "BTC/USD" or "btcusd".
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection and authenticate with the broker.

        Blocks until connected and authenticated, or raises:
            ConnectionError  — TCP/network failure or timeout.
            RuntimeError     — Authentication rejected by broker.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """
        Gracefully close the connection and release all resources.
        Safe to call even if not currently connected.
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if currently connected AND authenticated."""

    # ── Market data ───────────────────────────────────────────────────────────

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol    : str,
        timeframe : str,
        limit     : int = 500,
    ) -> pd.DataFrame:
        """
        Return the last `limit` completed OHLCV bars for `symbol`.

        The incomplete (currently forming) bar is NOT included. Bars are
        sorted oldest-first with a UTC DatetimeIndex.

        Args:
            symbol    : Instrument name (e.g. "BTCUSD", "XAUUSD").
            timeframe : Bar period — "1m", "5m", "15m", "1h", "4h", "1d".
            limit     : Number of completed bars to return.

        Returns:
            pd.DataFrame with UTC DatetimeIndex and float64 columns:
                open, high, low, close, volume
            Sorted oldest-first. May contain fewer than `limit` rows if the
            broker does not have sufficient history.

        Raises:
            ValueError   : Symbol or timeframe not supported by this connector.
            RuntimeError : Connector is not connected.
        """

    # ── Order management ──────────────────────────────────────────────────────

    @abstractmethod
    def place_market_order(
        self,
        symbol  : str,
        side    : str,
        size    : float,
        sl      : float | None = None,
        tp      : float | None = None,
        comment : str = "",
    ) -> OrderResult | None:
        """
        Place a market order and return the fill result.

        Args:
            symbol  : Instrument name (normalised internally).
            side    : "buy" or "sell".
            size    : Order size in instrument base units.
                      (BTC lots for BTCUSD, oz for XAUUSD, etc.)
            sl      : Absolute stop-loss price. None = no stop loss.
            tp      : Absolute take-profit price. None = no take profit.
            comment : Order label visible in broker trade history.

        Returns:
            OrderResult on successful fill, None on any failure.
            Connector logs a detailed error message on None.

        Raises:
            RuntimeError : Connector is not connected.
            ValueError   : Invalid side, size <= 0.
        """

    @abstractmethod
    def close_position(
        self,
        position_id : int,
        size        : float | None = None,
    ) -> bool:
        """
        Close an open position fully or partially.

        Args:
            position_id : Broker-assigned position identifier.
            size        : Units to close. None = close the full position.

        Returns:
            True on success, False on failure.
            Connector logs a detailed error on False.

        Raises:
            RuntimeError : Connector is not connected.
        """

    @abstractmethod
    def get_open_positions(
        self,
        symbol : str | None = None,
    ) -> list[Position]:
        """
        Return all currently open positions.

        Args:
            symbol : Filter to this instrument. None = return all positions.

        Returns:
            List of Position objects. Empty list if no open positions.

        Raises:
            RuntimeError : Connector is not connected.
        """

    # ── Account ───────────────────────────────────────────────────────────────

    @abstractmethod
    def get_account_summary(self) -> dict:
        """
        Return current account balance and equity.

        Returns:
            dict with at minimum:
                {"balance": float, "equity": float, "currency": str}
            Additional broker-specific keys are permitted.

        Raises:
            RuntimeError : Connector is not connected.
        """

    # ── Optional overrides ────────────────────────────────────────────────────

    def cancel_pending_order(self, order_id: int) -> bool:
        """
        Cancel a pending (limit or stop) order by ID.

        Default: raises NotImplementedError.
        Override in connectors that support pending order management.

        Returns:
            True on successful cancellation, False on failure.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement cancel_pending_order(). "
            "Override this method if the broker supports pending order cancellation."
        )

    def get_latest_price(self, symbol: str) -> dict | None:
        """
        Return the latest bid/ask price for `symbol`.

        Default: derives mid price from the most recent 1m bar close.
        Override for connectors with live tick subscriptions.

        Returns:
            {"bid": float, "ask": float}  or  None if unavailable.
        """
        try:
            df = self.fetch_ohlcv(symbol, "1m", limit=1)
            if df is not None and not df.empty:
                mid = float(df["close"].iloc[-1])
                return {"bid": mid, "ask": mid}
        except Exception:
            pass
        return None

    def __repr__(self) -> str:
        status = "connected" if self.is_connected() else "disconnected"
        return f"{self.__class__.__name__}({status})"
