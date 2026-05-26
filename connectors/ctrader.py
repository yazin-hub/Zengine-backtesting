"""
connectors/ctrader.py — cTrader Open API Connector
====================================================

Implements BaseConnector for the cTrader Open API (Spotware).
Uses the official `ctrader-open-api` Python package with Twisted reactor.

Install dependencies:
    pip install ctrader-open-api twisted

    Or via pyproject.toml extras:
    pip install "zengine[ctrader]"

Authentication:
    Requires a static access_token obtained via the Spotware OAuth2 flow
    (https://help.ctrader.com/open-api/account-authentication/).
    Token refresh is NOT handled here — tokens for personal accounts
    typically last 1 month. Obtain a new token and reconnect when needed.

Connection model:
    The cTrader Open API uses a persistent TCP connection running on a
    Twisted reactor in a background daemon thread. All public methods are
    synchronous — they use a threading.Event bridge to wait for Deferred
    callbacks from the reactor thread.

Order execution:
    Market orders only (PROTO_OA_NEW_ORDER_REQ with MARKET type).
    Absolute SL/TP cannot be set at market order creation — they are applied
    via post-fill position amendment (ProtoOAAmendPositionSLTPReq).
    Amendment is retried up to 3 times with a 2-second delay.
    If all retries fail, the unprotected position is closed immediately.

OHLCV data:
    Uses ProtoOAGetTrendbarsReq. cTrader stores trendbar prices ×100000
    regardless of symbol pip digits — divisor is always 100_000.
    Reference: https://help.ctrader.com/open-api/messages/#ProtoOATrendbar

Example:
    from connectors.ctrader import CTraderConnector

    conn = CTraderConnector(
        client_id="your_client_id",
        client_secret="your_secret",
        account_id=12345678,
        access_token="your_access_token",
        env="demo",
    )
    conn.connect()
    df = conn.fetch_ohlcv("BTCUSD", "15m", limit=200)
    result = conn.place_market_order("BTCUSD", "buy", size=0.01, sl=90_000.0)
    conn.disconnect()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .base import BaseConnector, OrderResult, Position

_log = logging.getLogger(__name__)

# cTrader volume = size × 100 (centilots).
# e.g. 0.01 BTC → volume = 1
_VOLUME_MULTIPLIER: int = 100

# SL/TP amendment retry config (market orders need post-fill amendment)
_SLTP_MAX_ATTEMPTS: int   = 3
_SLTP_RETRY_DELAY:  float = 2.0

# cTrader always stores trendbar prices ×100_000, regardless of symbol digits.
# Using sym.digits for divisor gives 1000× wrong prices on commodities (e.g. XAUUSD).
_PRICE_DIVISOR: int = 100_000

# Timeframe string → (cTrader period enum value, minutes per bar)
_PERIOD_MAP: dict[str, tuple[Any, int]] = {
    "1m" : (None, 1),    # enum values resolved at connect() time
    "5m" : (None, 5),
    "15m": (None, 15),
    "1h" : (None, 60),
    "4h" : (None, 240),
    "1d" : (None, 1440),
}


def _normalise(symbol: str) -> str:
    """Remove slashes, uppercase — cTrader canonical symbol format."""
    return symbol.replace("/", "").upper()


class CTraderConnector(BaseConnector):
    """
    cTrader Open API connector for ZEngine.

    Implements BaseConnector using the official Spotware ctrader-open-api
    package. Handles authentication, OHLCV fetching via trendbars, and
    market order placement with SL/TP post-fill amendment.

    Args:
        client_id         : OAuth2 client ID from Spotware Connect.
        client_secret     : OAuth2 client secret.
        account_id        : Numeric cTrader account ID.
        access_token      : Bearer token from OAuth2 flow.
        env               : "demo" or "live".
        connect_timeout   : Seconds to wait for initial TCP connection (default 15).
        reconnect_delay   : Seconds to wait before re-authenticating after a TCP
                            reconnect. A small pause lets the TCP layer stabilise
                            before sending protobuf messages (default 2.0).
        on_reconnect      : Optional callback called after every successful
                            re-authentication. Signature: () -> None.
                            Use this to send a Telegram alert, reset state, etc.
    """

    def __init__(
        self,
        client_id       : str,
        client_secret   : str,
        account_id      : int,
        access_token    : str,
        env             : str                    = "demo",
        connect_timeout : float                  = 15.0,
        reconnect_delay : float                  = 2.0,
        on_reconnect    : Callable[[], None] | None = None,
    ) -> None:
        if env not in ("demo", "live"):
            raise ValueError(f"env must be 'demo' or 'live', got {env!r}")
        if connect_timeout <= 0:
            raise ValueError(f"connect_timeout must be positive, got {connect_timeout}")
        if reconnect_delay < 0:
            raise ValueError(f"reconnect_delay must be >= 0, got {reconnect_delay}")

        self._client_id       = str(client_id)
        self._client_secret   = str(client_secret)
        self._account_id      = int(account_id)
        self._access_token    = str(access_token)
        self._env             = env
        self._connect_timeout = connect_timeout
        self._reconnect_delay = reconnect_delay
        self._on_reconnect    = on_reconnect

        # Set at connect() time
        self._client          : Any                    = None
        self._reactor_thread  : threading.Thread | None = None
        self._connected_evt   = threading.Event()
        self._authenticated   = False
        self._is_reconnection : bool                   = False   # True after first connect
        self._symbol_map      : dict[str, dict]        = {}      # norm → {id, digits, name}
        self._id_to_norm      : dict[int, str]         = {}      # symbolId → norm

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Connect to cTrader API and authenticate.

        Blocks until connected and authenticated, or raises on failure.
        After the initial connection, TCP reconnection is handled automatically
        by Twisted's ReconnectingClientFactory. Re-authentication runs in a
        background thread each time the connection is re-established.

        Raises:
            ImportError     : ctrader-open-api or twisted not installed.
            ConnectionError : TCP connection timeout.
            RuntimeError    : Authentication rejected.
        """
        try:
            from ctrader_open_api import Client, TcpProtocol, EndPoints
        except ImportError as exc:
            raise ImportError(
                "ctrader-open-api is not installed.\n"
                "Install with:  pip install ctrader-open-api twisted\n"
                "Or:            pip install 'zengine[ctrader]'"
            ) from exc

        host = (
            EndPoints.PROTOBUF_LIVE_HOST
            if self._env == "live"
            else EndPoints.PROTOBUF_DEMO_HOST
        )

        self._client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        self._reactor_thread = threading.Thread(
            target=self._run_reactor, daemon=True, name="ctrader-reactor"
        )
        self._reactor_thread.start()

        _log.info(
            "Waiting for cTrader %s connection (timeout=%ss)...",
            self._env, self._connect_timeout,
        )
        if not self._connected_evt.wait(timeout=self._connect_timeout):
            raise ConnectionError(
                f"Timeout connecting to cTrader {self._env} API "
                f"after {self._connect_timeout}s. Check network connectivity."
            )

        _log.info("Authenticating account %d...", self._account_id)
        self._authenticate()

    def disconnect(self) -> None:
        """Logout and stop the Twisted reactor cleanly."""
        if self._authenticated:
            try:
                self._logout()
            except Exception as exc:
                _log.warning("Logout failed (non-critical): %s", exc)

        self._authenticated = False
        self._connected_evt.clear()

        try:
            from twisted.internet import reactor
            reactor.callFromThread(reactor.stop)
        except Exception:
            pass

        _log.info("CTraderConnector disconnected.")

    def is_connected(self) -> bool:
        """Return True if TCP is up AND account is authenticated."""
        return self._authenticated and self._connected_evt.is_set()

    # ── Twisted internals ─────────────────────────────────────────────────────

    def _run_reactor(self) -> None:
        from twisted.internet import reactor
        self._client.startService()
        reactor.run(installSignalHandlers=False)

    def _on_connected(self, client: Any) -> None:
        """
        Called by Twisted on every successful TCP connection — initial and reconnects.

        On the first call: just set the event so connect() can proceed.
        On subsequent calls (reconnects): spawn _reauthenticate() in a daemon
        thread so the reactor thread is never blocked.
        """
        self._connected_evt.set()
        if self._is_reconnection:
            _log.info("TCP reconnected to cTrader %s — re-authenticating...", self._env)
            t = threading.Thread(target=self._reauthenticate, daemon=True,
                                 name="ctrader-reauth")
            t.start()
        else:
            _log.info("TCP connection established to cTrader %s.", self._env)
        # Mark True so every call after the first is treated as a reconnect
        self._is_reconnection = True

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        """
        Called by Twisted on TCP disconnect.
        Twisted's ReconnectingClientFactory will automatically attempt to
        restore the TCP connection — we just clear auth state here.
        """
        self._connected_evt.clear()
        self._authenticated = False
        _log.warning(
            "Disconnected from cTrader %s: %s  (Twisted will reconnect automatically)",
            self._env, reason,
        )

    def _on_message(self, client: Any, message: Any) -> None:
        # Push events (SL/TP fills, etc.) — request-response pairs are
        # resolved by _send_sync() via Deferred callbacks, so nothing to
        # dispatch here for a polling-based live runner.
        pass

    def _reauthenticate(self) -> None:
        """
        Re-run the full authentication sequence after a TCP reconnect.

        Waits reconnect_delay seconds to let the TCP layer stabilise, then
        calls _authenticate() which re-runs app auth, account auth, and reloads
        the symbol cache. Calls the on_reconnect callback on success.

        Mirrors the _reauthenticate() pattern from the reference CTraderAdapter.
        """
        try:
            if self._reconnect_delay > 0:
                time.sleep(self._reconnect_delay)
            _log.info("Re-authenticating account %d...", self._account_id)
            self._authenticate()
            _log.info("Re-authentication successful — connector is live again.")
            if self._on_reconnect is not None:
                try:
                    self._on_reconnect()
                except Exception as exc:
                    _log.warning("on_reconnect callback raised: %s", exc)
        except Exception as exc:
            self._authenticated = False
            _log.error("Re-authentication failed: %s", exc, exc_info=True)

    def _send_sync(self, request: Any, timeout: float = 30.0) -> Any:
        """
        Send a protobuf request and block until the response arrives.

        Thread-safe bridge from the synchronous caller to the Twisted reactor.
        Uses a queue + threading.Event to avoid busy-waiting.

        Raises:
            TimeoutError : No response within (timeout + 5s) seconds.
            Exception    : Any error returned by the Deferred errback.
        """
        from twisted.internet import reactor

        resp_q   = queue.Queue()
        done_evt = threading.Event()

        def _go() -> None:
            d = self._client.send(request, responseTimeoutInSeconds=int(timeout))
            d.addCallbacks(
                lambda r: (resp_q.put(r),                 done_evt.set()),
                lambda f: (resp_q.put(Exception(str(f))), done_evt.set()),
            )

        reactor.callFromThread(_go)

        if not done_evt.wait(timeout=timeout + 5.0):
            raise TimeoutError(
                f"cTrader request timed out after {timeout}s: "
                f"{type(request).__name__}"
            )

        result = resp_q.get_nowait()
        if isinstance(result, Exception):
            raise result
        return result

    # ── Authentication ────────────────────────────────────────────────────────

    def _authenticate(self) -> None:
        """Full two-step auth: app → account. Loads symbol cache on success."""
        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        # Step 1: application auth (client_id + client_secret)
        req = Msg.ProtoOAApplicationAuthReq()
        req.clientId     = self._client_id
        req.clientSecret = self._client_secret
        resp = self._send_sync(req, timeout=60.0)

        if resp.payloadType == Model.PROTO_OA_ERROR_RES:
            err = Msg.ProtoOAErrorRes()
            err.ParseFromString(resp.payload)
            raise RuntimeError(
                f"App auth failed: {err.errorCode} — {err.description}"
            )
        _log.info("Application authenticated.")

        # Step 2: account auth (account_id + access_token)
        req2 = Msg.ProtoOAAccountAuthReq()
        req2.ctidTraderAccountId = self._account_id
        req2.accessToken         = self._access_token
        resp2 = self._send_sync(req2, timeout=60.0)

        if resp2.payloadType != Model.PROTO_OA_ACCOUNT_AUTH_RES:
            err2 = Msg.ProtoOAErrorRes()
            try:
                err2.ParseFromString(resp2.payload)
            except Exception:
                pass
            raise RuntimeError(
                f"Account auth failed (payloadType={resp2.payloadType}): "
                f"{getattr(err2, 'errorCode', '?')} — {getattr(err2, 'description', '?')}"
            )
        _log.info("Account %d authenticated.", self._account_id)

        self._load_symbol_cache()
        self._authenticated = True
        _log.info("CTraderConnector ready.")

    def _logout(self) -> None:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as Msg
        req = Msg.ProtoOAAccountLogoutReq()
        req.ctidTraderAccountId = self._account_id
        try:
            self._send_sync(req, timeout=10.0)
        except Exception:
            pass

    def _load_symbol_cache(self) -> None:
        """
        Fetch all symbols to build name→ID and ID→name mappings.

        Called once at authentication. Subsequent calls are not needed
        unless the account's available instruments change.
        """
        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        req = Msg.ProtoOASymbolsListReq()
        req.ctidTraderAccountId    = self._account_id
        req.includeArchivedSymbols = False
        resp = self._send_sync(req, timeout=30.0)

        if resp.payloadType != Model.PROTO_OA_SYMBOLS_LIST_RES:
            _log.warning("Unexpected symbols list response: %s", resp.payloadType)
            return

        payload = Msg.ProtoOASymbolsListRes()
        payload.ParseFromString(resp.payload)

        self._symbol_map = {}
        self._id_to_norm = {}

        for sym in payload.symbol:
            name  = sym.symbolName
            norm  = _normalise(name)
            upper = name.upper()

            # Display precision — used for SL/TP rounding only (not for price divisor).
            # Note: cTrader ALWAYS stores raw prices ×100000 regardless of digits.
            if any(t in upper for t in ("BTC", "ETH", "XAU", "XAG", "OIL", "WTI")):
                digits = 2
            elif "JPY" in upper:
                digits = 3
            else:
                digits = 5

            self._symbol_map[norm]       = {"id": sym.symbolId, "digits": digits, "name": name}
            self._id_to_norm[sym.symbolId] = norm

        _log.info("Symbol cache: %d instruments loaded.", len(self._symbol_map))

    # ── BaseConnector — market data ───────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol    : str,
        timeframe : str,
        limit     : int = 500,
    ) -> pd.DataFrame:
        """
        Return the last `limit` completed OHLCV bars as a DataFrame.

        Uses ProtoOAGetTrendbarsReq. All trendbar prices are stored ×100_000
        by cTrader regardless of symbol digits — divisor is always 100_000.

        Returns:
            DataFrame with UTC DatetimeIndex, columns: open high low close volume.
            Sorted oldest-first. Empty DataFrame if no bars available.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected. Call connect() first.")

        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        # Resolve period enum values lazily (Model imported here to avoid
        # top-level import of ctrader_open_api at module load time)
        period_map: dict[str, tuple[Any, int]] = {
            "1m" : (Model.M1,  1),
            "5m" : (Model.M5,  5),
            "15m": (Model.M15, 15),
            "1h" : (Model.H1,  60),
            "4h" : (Model.H4,  240),
            "1d" : (Model.D1,  1440),
        }

        norm = _normalise(symbol)
        if norm not in self._symbol_map:
            raise ValueError(
                f"Symbol {symbol!r} not in account symbol cache. "
                "Check the symbol name matches your broker's naming."
            )
        if timeframe not in period_map:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}. "
                f"Supported: {sorted(period_map)}"
            )

        period, mins = period_map[timeframe]
        sym_id       = self._symbol_map[norm]["id"]
        to_ms        = int(time.time() * 1000)
        # 4× window ensures weekends / market closures don't reduce bar count
        from_ms      = to_ms - (limit * mins * 60 * 1000 * 4)

        req = Msg.ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self._account_id
        req.fromTimestamp       = from_ms
        req.toTimestamp         = to_ms
        req.period              = period
        req.symbolId            = sym_id
        req.count               = limit

        # Scale timeout with bar count — large requests take longer
        timeout = float(max(30.0, min(limit / 100, 60.0)))
        resp    = self._send_sync(req, timeout=timeout)

        if resp.payloadType != Model.PROTO_OA_GET_TRENDBARS_RES:
            _log.warning("Unexpected trendbars response: %s", resp.payloadType)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        payload = Msg.ProtoOAGetTrendbarsRes()
        payload.ParseFromString(resp.payload)
        _log.info("Received %d %s bars for %s.", len(payload.trendbar), timeframe, norm)

        rows = []
        for bar in payload.trendbar:
            low_raw = bar.low
            o   = (low_raw + bar.deltaOpen)  / _PRICE_DIVISOR
            h   = (low_raw + bar.deltaHigh)  / _PRICE_DIVISOR
            low = low_raw                    / _PRICE_DIVISOR
            c   = (low_raw + bar.deltaClose) / _PRICE_DIVISOR
            ts = pd.Timestamp(bar.utcTimestampInMinutes * 60, unit="s", tz="UTC")
            rows.append({
                "time"  : ts,
                "open"  : o,
                "high"  : h,
                "low"   : low,
                "close" : c,
                "volume": float(bar.volume),
            })

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).set_index("time").sort_index()
        df.index.name = None
        return df

    # ── BaseConnector — order management ──────────────────────────────────────

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
        Place a market order and apply SL/TP via post-fill amendment.

        cTrader MARKET orders do not accept absolute SL/TP at creation time —
        they are applied separately via ProtoOAAmendPositionSLTPReq after the
        position opens. If amendment fails after _SLTP_MAX_ATTEMPTS retries,
        the position is closed immediately to prevent unprotected exposure.

        Returns:
            OrderResult on success, None on any failure.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected. Call connect() first.")
        if side.lower() not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        norm = _normalise(symbol)
        if norm not in self._symbol_map:
            _log.error("Symbol %r not in symbol cache.", symbol)
            return None

        sym_data = self._symbol_map[norm]
        prec     = sym_data["digits"]

        req = Msg.ProtoOANewOrderReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId            = sym_data["id"]
        req.orderType           = Model.MARKET
        req.tradeSide           = Model.BUY if side.lower() == "buy" else Model.SELL
        req.volume              = int(size * _VOLUME_MULTIPLIER)
        req.comment             = comment or "ZEngine"

        try:
            resp = self._send_sync(req, timeout=15.0)
        except Exception as exc:
            _log.error("Market order failed (%s %s %s): %s", side, size, norm, exc)
            return None

        if resp.payloadType != Model.PROTO_OA_EXECUTION_EVENT:
            _log.error(
                "Unexpected order response (payloadType=%s) for %s %s %s.",
                resp.payloadType, side, size, norm,
            )
            return None

        payload = Msg.ProtoOAExecutionEvent()
        payload.ParseFromString(resp.payload)
        order_id   = payload.order.orderId
        pos_id     : int | None = None
        fill_price : float      = 0.0

        if hasattr(payload, "position") and payload.position and payload.position.positionId:
            pos_id     = payload.position.positionId
            fill_price = float(payload.position.price) if payload.position.price else 0.0

        _log.info(
            "Market order filled: orderId=%d  %s %.4f %s  pos=%s  price=%.5f",
            order_id, side, size, norm, pos_id, fill_price,
        )

        # Apply SL/TP via post-fill amendment (required for MARKET orders)
        sl_r = round(sl, prec) if sl is not None else None
        tp_r = round(tp, prec) if tp is not None else None

        if pos_id and (sl_r is not None or tp_r is not None):
            ok = self._apply_sltp_with_retry(pos_id, sl_r, tp_r)
            if not ok:
                _log.error(
                    "SL/TP amendment failed after %d attempts on pos %d — "
                    "closing to prevent unprotected exposure.",
                    _SLTP_MAX_ATTEMPTS, pos_id,
                )
                self.close_position(pos_id)
                return None

        return OrderResult(
            position_id = pos_id or 0,
            order_id    = order_id,
            symbol      = norm,
            side        = side.lower(),
            size        = size,
            fill_price  = fill_price,
            sl          = sl_r,
            tp          = tp_r,
            timestamp   = datetime.now(tz=timezone.utc),
            comment     = comment,
        )

    def close_position(
        self,
        position_id : int,
        size        : float | None = None,
    ) -> bool:
        """Close a position fully (default) or partially."""
        if not self.is_connected():
            raise RuntimeError("Not connected. Call connect() first.")

        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        actual_size = size
        if actual_size is None:
            for pos in self.get_open_positions():
                if pos.position_id == position_id:
                    actual_size = pos.size
                    break
        if actual_size is None:
            _log.error("Cannot close position %d: size unknown.", position_id)
            return False

        req = Msg.ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self._account_id
        req.positionId          = int(position_id)
        req.volume              = int(actual_size * _VOLUME_MULTIPLIER)

        try:
            resp = self._send_sync(req, timeout=15.0)
            if resp.payloadType == Model.PROTO_OA_EXECUTION_EVENT:
                _log.info("Position %d closed.", position_id)
                return True
            _log.error(
                "Unexpected close response for position %d: payloadType=%s",
                position_id, resp.payloadType,
            )
        except Exception as exc:
            _log.error("Failed to close position %d: %s", position_id, exc)
        return False

    def get_open_positions(
        self,
        symbol : str | None = None,
    ) -> list[Position]:
        """Return open positions, optionally filtered by symbol."""
        if not self.is_connected():
            raise RuntimeError("Not connected. Call connect() first.")

        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        req = Msg.ProtoOAReconcileReq()
        req.ctidTraderAccountId = self._account_id

        try:
            resp = self._send_sync(req, timeout=15.0)
        except Exception as exc:
            _log.error("Failed to fetch positions: %s", exc)
            return []

        if resp.payloadType != Model.PROTO_OA_RECONCILE_RES:
            return []

        payload = Msg.ProtoOAReconcileRes()
        payload.ParseFromString(resp.payload)

        filter_norm = _normalise(symbol) if symbol else None
        positions   = []

        for p in payload.position:
            sym_norm = self._id_to_norm.get(p.tradeData.symbolId, "UNKNOWN")
            if filter_norm and sym_norm != filter_norm:
                continue
            side = "buy" if p.tradeData.tradeSide == Model.BUY else "sell"
            positions.append(Position(
                position_id = p.positionId,
                symbol      = sym_norm,
                side        = side,
                size        = p.tradeData.volume / _VOLUME_MULTIPLIER,
                entry_price = float(p.price),
                sl          = float(p.stopLoss)   if p.stopLoss   else None,
                tp          = float(p.takeProfit) if p.takeProfit else None,
                comment     = getattr(p.tradeData, "comment", ""),
            ))
        return positions

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        """Return account balance. Equity approximated as balance (no floating P&L)."""
        if not self.is_connected():
            raise RuntimeError("Not connected. Call connect() first.")

        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        req = Msg.ProtoOATraderReq()
        req.ctidTraderAccountId = self._account_id

        try:
            resp = self._send_sync(req, timeout=15.0)
        except Exception as exc:
            _log.error("Failed to fetch account info: %s", exc)
            return {"balance": 0.0, "equity": 0.0, "currency": "USD"}

        if resp.payloadType == Model.PROTO_OA_TRADER_RES:
            payload = Msg.ProtoOATraderRes()
            payload.ParseFromString(resp.payload)
            trader = payload.trader
            digits = getattr(trader, "moneyDigits", 2)
            bal    = trader.balance / (10 ** digits)
            return {"balance": bal, "equity": bal, "currency": "USD"}

        return {"balance": 0.0, "equity": 0.0, "currency": "USD"}

    # ── Optional overrides ────────────────────────────────────────────────────

    def cancel_pending_order(self, order_id: int) -> bool:
        """Cancel a pending (limit/stop) order by ID."""
        if not self.is_connected():
            raise RuntimeError("Not connected. Call connect() first.")

        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        req = Msg.ProtoOACancelOrderReq()
        req.ctidTraderAccountId = self._account_id
        req.orderId             = int(order_id)

        try:
            resp = self._send_sync(req, timeout=15.0)
            if resp.payloadType == Model.PROTO_OA_EXECUTION_EVENT:
                _log.info("Order %d cancelled.", order_id)
                return True
            _log.warning(
                "Unexpected cancel response for order %d: %s", order_id, resp.payloadType
            )
        except Exception as exc:
            _log.error("Failed to cancel order %d: %s", order_id, exc)
        return False

    # ── SL/TP amendment ───────────────────────────────────────────────────────

    def _apply_sltp(
        self,
        position_id : int,
        sl_price    : float | None,
        tp_price    : float | None,
    ) -> bool:
        """Single attempt to apply SL/TP via position amendment."""
        from ctrader_open_api.messages import (
            OpenApiMessages_pb2      as Msg,
            OpenApiModelMessages_pb2 as Model,
        )

        req = Msg.ProtoOAAmendPositionSLTPReq()
        req.ctidTraderAccountId = self._account_id
        req.positionId          = position_id
        if sl_price is not None:
            req.stopLoss   = sl_price
        if tp_price is not None:
            req.takeProfit = tp_price

        _log.info("Amending position %d: SL=%s  TP=%s", position_id, sl_price, tp_price)
        try:
            resp = self._send_sync(req, timeout=15.0)
            if resp and resp.payloadType == Model.PROTO_OA_EXECUTION_EVENT:
                _log.info("SL/TP applied to position %d.", position_id)
                return True
        except Exception as exc:
            _log.error("SL/TP amendment exception on position %d: %s", position_id, exc)
        return False

    def _apply_sltp_with_retry(
        self,
        position_id : int,
        sl_price    : float | None,
        tp_price    : float | None,
    ) -> bool:
        """Retry SL/TP amendment up to _SLTP_MAX_ATTEMPTS times."""
        for attempt in range(1, _SLTP_MAX_ATTEMPTS + 1):
            if attempt > 1:
                _log.info(
                    "SL/TP retry %d/%d for position %d (%.1fs delay)...",
                    attempt, _SLTP_MAX_ATTEMPTS, position_id, _SLTP_RETRY_DELAY,
                )
                time.sleep(_SLTP_RETRY_DELAY)
            if self._apply_sltp(position_id, sl_price, tp_price):
                return True
        return False
