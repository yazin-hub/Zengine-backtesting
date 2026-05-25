"""
engine/data_sources/ctrader_source.py — cTrader Open API data source
=====================================================================
Fetches historical OHLCV bar data with real bid/ask spread per bar from
the cTrader Open API.  Works on Mac, Linux, and Windows — no trading
terminal installation required.

Authentication
--------------
Uses OAuth2 with a local callback server.  One-time setup:
  1. Register a free application at https://openapi.ctrader.com/
  2. Set the Redirect URI to exactly: http://localhost:8182/callback
  3. Copy your Client ID and Client Secret into the ZEngine UI
  4. Click "Authenticate" — browser opens, you log in once, tokens saved

Tokens are stored in ~/.zengine/ctrader_tokens.json and auto-refreshed
for ~60 days before re-authentication is needed.

Technical approach
------------------
The cTrader Open API uses SSL TCP + protobuf (not HTTP/REST).
Message format: [4-byte big-endian length][ProtoMessage envelope bytes].

This module implements a minimal asyncio TCP client that:
  1. Opens an SSL connection to demo.ctraderapi.com:5035 (or live)
  2. Authenticates the application and account
  3. Resolves symbol IDs
  4. Fetches trendbar (OHLCV) data in paginated 5000-bar chunks
  5. Fetches bid + ask bars separately → spread = ask_open - bid_open
  6. Disconnects cleanly

We import only the generated protobuf message classes from ctrader-open-api,
not its Twisted networking layer.  Our own asyncio client handles transport.

Why real spread is better than fixed
-------------------------------------
Connecting your live cTrader account (same one you trade with) gives you
the exact spread your account received at every minute bar.  ECN/raw spread
accounts have tighter spreads than standard accounts.  Demo accounts use
synthetic spreads that are typically 20–40% tighter than live.

Requires
--------
    pip install ctrader-open-api requests
    (installed via: pip install "zengine[ctrader]")

References
----------
cTrader Open API documentation: https://help.ctrader.com/open-api/
Spotware OpenApiPy examples: https://github.com/spotware/OpenApiPy
"""

from __future__ import annotations

import asyncio
import json
import ssl
import struct
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import numpy as np
import pandas as pd

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None   # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

try:
    # We only import the generated protobuf message classes, not the Twisted client
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOAGetTrendbarsReq,
        ProtoOAGetTrendbarsRes,
        ProtoOAErrorRes,
    )
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
    _CTRADER_AVAILABLE = True
except ImportError:
    _CTRADER_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────

DEMO_HOST = "demo.ctraderapi.com"
LIVE_HOST = "live.ctraderapi.com"
PORT      = 5035                          # SSL TCP port

# Spotware migrated OAuth2 in 2024 — old connect.spotware.com URLs return 404.
# Auth redirect:  https://id.ctrader.com/my/settings/openapi/grantingaccess/
# Token exchange: https://openapi.ctrader.com/apps/token  (GET for code, POST for refresh)
AUTH_URL  = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
TOKEN_URL = "https://openapi.ctrader.com/apps/token"

REDIRECT_URI   = "http://localhost:8182/callback"   # must match app registration
CALLBACK_PORT  = 8182
TOKEN_FILE     = Path.home() / ".zengine" / "ctrader_tokens.json"

# cTrader payload type constants (from OpenApiMessages.proto)
_PT_APP_AUTH_REQ        = 2100
_PT_APP_AUTH_RES        = 2101
_PT_ACCOUNT_AUTH_REQ    = 2102
_PT_ACCOUNT_AUTH_RES    = 2103
_PT_GET_ACCOUNTS_REQ    = 2149
_PT_GET_ACCOUNTS_RES    = 2150
_PT_SYMBOLS_LIST_REQ    = 2114
_PT_SYMBOLS_LIST_RES    = 2115
_PT_GET_TRENDBARS_REQ   = 2137
_PT_GET_TRENDBARS_RES   = 2138
_PT_REFRESH_TOKEN_REQ   = 2072
_PT_REFRESH_TOKEN_RES   = 2073
_PT_ERROR_RES           = 2142
_PT_HEARTBEAT           = 51

# Timeframe string → ProtoOATrendbarPeriod enum value
# Populated lazily after confirming ctrader-open-api is importable
_TF_MAP: dict[str, int] = {}

# Trendbar period → bar duration in seconds (for timestamp stepping)
_TF_SECONDS: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "H12": 43200,
    "D1": 86400, "W1": 604800,
}

# Max bars per trendbar request (cTrader server limit)
_MAX_BARS_PER_REQUEST = 5000


def _build_tf_map() -> None:
    global _TF_MAP
    if not _CTRADER_AVAILABLE or _TF_MAP:
        return
    _TF_MAP = {
        "M1":  ProtoOATrendbarPeriod.Value("M1"),
        "M5":  ProtoOATrendbarPeriod.Value("M5"),
        "M15": ProtoOATrendbarPeriod.Value("M15"),
        "M30": ProtoOATrendbarPeriod.Value("M30"),
        "H1":  ProtoOATrendbarPeriod.Value("H1"),
        "H4":  ProtoOATrendbarPeriod.Value("H4"),
        "H12": ProtoOATrendbarPeriod.Value("H12"),
        "D1":  ProtoOATrendbarPeriod.Value("D1"),
        "W1":  ProtoOATrendbarPeriod.Value("W1"),
    }


# ── Low-level asyncio TCP client ───────────────────────────────────────────────

async def _send(writer: asyncio.StreamWriter, payload_type: int, message) -> None:
    """Serialise a protobuf message in a ProtoMessage envelope and send over TCP."""
    envelope = ProtoMessage(
        payloadType=payload_type,
        payload=message.SerializeToString(),
    )
    raw  = envelope.SerializeToString()
    data = struct.pack(">I", len(raw)) + raw
    writer.write(data)
    await writer.drain()


async def _recv(reader: asyncio.StreamReader) -> ProtoMessage:
    """Read one framed ProtoMessage from the TCP stream."""
    header = await asyncio.wait_for(reader.readexactly(4), timeout=30.0)
    length = struct.unpack(">I", header)[0]
    raw    = await asyncio.wait_for(reader.readexactly(length), timeout=30.0)
    msg    = ProtoMessage()
    msg.ParseFromString(raw)
    return msg


async def _recv_type(reader: asyncio.StreamReader, expected_type: int,
                     timeout: float = 30.0) -> ProtoMessage:
    """
    Receive messages until we get one with the expected payloadType.
    Heartbeats and unrelated messages are silently discarded.
    Raises RuntimeError on error response or timeout.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for cTrader message type {expected_type}")
        try:
            msg = await asyncio.wait_for(_recv(reader), timeout=remaining)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timed out waiting for cTrader message type {expected_type}")

        if msg.payloadType == _PT_ERROR_RES:
            err = ProtoOAErrorRes()
            err.ParseFromString(msg.payload)
            raise RuntimeError(
                f"cTrader API error {err.errorCode}: {err.description}"
            )
        if msg.payloadType == _PT_HEARTBEAT:
            continue    # ignore heartbeats
        if msg.payloadType == expected_type:
            return msg
        # Other message types — skip (common: status/notification messages)


async def _open_connection(host: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open an SSL TCP connection to the cTrader API server."""
    ctx = ssl.create_default_context()
    return await asyncio.open_connection(host, PORT, ssl=ctx)


# ── OAuth2 helpers ─────────────────────────────────────────────────────────────

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that captures the OAuth2 authorization code."""
    auth_code: Optional[str] = None
    error: Optional[str]     = None

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if "code" in params:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style='font-family:sans-serif;text-align:center;padding:60px'>
                <h2 style='color:#26a69a'>&#10003; Authenticated!</h2>
                <p>You can close this tab and return to ZEngine.</p>
                </body></html>
            """)
        elif "error" in params:
            _OAuthCallbackHandler.error = params.get("error_description", ["Unknown error"])[0]
            self.send_response(400)
            self.end_headers()
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, *args):
        pass    # suppress default request logging


def _normalize_tokens(raw: dict) -> dict:
    """
    Normalize a cTrader token response to consistent snake_case keys.

    The new openapi.ctrader.com endpoint returns camelCase keys
    (accessToken, refreshToken, expiresIn) whereas the old
    connect.spotware.com endpoint used snake_case.  We normalise here so
    the rest of the codebase can always use access_token / refresh_token.
    """
    return {
        "access_token":  raw.get("accessToken")  or raw.get("access_token",  ""),
        "refresh_token": raw.get("refreshToken") or raw.get("refresh_token", ""),
        "expires_in":    raw.get("expiresIn")    or raw.get("expires_in",    0),
        "token_type":    raw.get("tokenType")    or raw.get("token_type",    ""),
    }


def _run_oauth_flow(client_id: str, client_secret: str,
                    timeout: int = 120) -> dict:
    """
    Run the full OAuth2 authorization code flow:
      1. Start a local HTTP server on CALLBACK_PORT
      2. Open browser to cTrader's authorization URL
      3. Wait for the callback (user logs in and approves)
      4. Exchange the authorization code for access + refresh tokens
      5. Return token dict

    Args:
        client_id     : cTrader application client ID
        client_secret : cTrader application client secret
        timeout       : seconds to wait for browser callback (default 120)

    Returns:
        dict with keys: access_token, refresh_token, expires_in, token_type

    Raises:
        RuntimeError on auth failure or timeout.
    """
    if not _REQUESTS_AVAILABLE:
        raise RuntimeError(
            "requests package not installed.\n"
            "Install with: pip install requests"
        )

    # Reset handler state from any previous run
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.error     = None

    # Start local callback server in a background thread
    server = HTTPServer(("localhost", CALLBACK_PORT), _OAuthCallbackHandler)
    server.timeout = 1.0   # poll every second so we can check for the code

    stop_event = threading.Event()

    def _serve():
        while not stop_event.is_set():
            server.handle_request()
        server.server_close()

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()

    # Build authorization URL and open browser.
    # `product=web` strips header/footer — looks cleaner in a local browser tab.
    auth_params = {
        "client_id":     client_id,
        "redirect_uri":  REDIRECT_URI,
        "scope":         "trading",
        "product":       "web",
    }
    webbrowser.open(f"{AUTH_URL}?{urlencode(auth_params)}")

    # Wait for authorization code to arrive
    import time
    start = time.time()
    while time.time() - start < timeout:
        if _OAuthCallbackHandler.auth_code:
            break
        if _OAuthCallbackHandler.error:
            stop_event.set()
            raise RuntimeError(f"cTrader OAuth2 error: {_OAuthCallbackHandler.error}")
        time.sleep(0.5)
    else:
        stop_event.set()
        raise TimeoutError(
            f"Browser authentication timed out after {timeout}s. "
            "Complete the login in the browser window and try again."
        )

    stop_event.set()
    code = _OAuthCallbackHandler.auth_code

    # Exchange authorization code for tokens.
    # New endpoint uses GET with query params (not POST + form body).
    resp = _requests.get(TOKEN_URL, params={
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  REDIRECT_URI,
        "client_id":     client_id,
        "client_secret": client_secret,
    }, timeout=15)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed (HTTP {resp.status_code}): {resp.text}"
        )

    # Normalize camelCase keys (accessToken → access_token) so the rest of
    # the codebase always reads access_token / refresh_token consistently.
    return _normalize_tokens(resp.json())


def _refresh_access_token(client_id: str, client_secret: str,
                           refresh_token: str) -> dict:
    """Exchange a refresh token for a new access token."""
    if not _REQUESTS_AVAILABLE:
        raise RuntimeError("requests package not installed.")

    # Refresh uses POST but params go in the query string, not the body.
    resp = _requests.post(TOKEN_URL, params={
        "grant_type":    "refresh_token",
        "refresh_token": refresh_token,
        "client_id":     client_id,
        "client_secret": client_secret,
    }, timeout=15)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Token refresh failed (HTTP {resp.status_code}): {resp.text}"
        )
    return _normalize_tokens(resp.json())


# ── CTraderSource ──────────────────────────────────────────────────────────────

class CTraderSource:
    """
    cTrader Open API data source.

    Cross-platform — works on Mac, Linux, and Windows without any
    trading terminal installed.  Uses OAuth2 for authentication and
    SSL TCP for data retrieval.

    Spread accuracy
    ---------------
    cTrader returns separate bid and ask trendbar series.  We fetch both
    and compute spread = ask_open − bid_open at every bar.  This is exact
    bid/ask spread for your specific account and broker — more accurate
    than any fixed or estimated spread value.

    Connecting your live account (same as you trade with) gives you the
    exact execution environment your strategy will face.  Demo accounts
    have synthetic spreads that underestimate live costs.

    Usage
    -----
    src = CTraderSource(client_id="...", client_secret="...", use_demo=True)
    if not src.is_authenticated():
        src.authenticate()                    # opens browser once
    df       = src.fetch_bars("XAUUSD", "M1", from_dt, to_dt)
    defaults = src.get_broker_defaults("XAUUSD")
    schedule = src.build_spread_schedule(df)
    """

    def __init__(self,
                 client_id:     str  = "",
                 client_secret: str  = "",
                 use_demo:      bool = True) -> None:
        self.client_id     = client_id
        self.client_secret = client_secret
        self.use_demo      = use_demo

    # ── Availability & authentication ─────────────────────────────────────────

    def is_available(self) -> bool:
        """True if ctrader-open-api and requests are installed."""
        return _CTRADER_AVAILABLE and _REQUESTS_AVAILABLE

    @staticmethod
    def platform_note() -> str:
        """Return a human-readable note about what's missing, or '' if ready."""
        missing = []
        if not _CTRADER_AVAILABLE:
            missing.append("`ctrader-open-api`")
        if not _REQUESTS_AVAILABLE:
            missing.append("`requests`")
        if missing:
            return (
                f"Install missing packages: pip install {' '.join(missing)}\n"
                "Or: pip install \"zengine[ctrader]\""
            )
        return ""

    def is_authenticated(self) -> bool:
        """True if valid tokens exist on disk (access token present)."""
        tokens = self._load_tokens()
        return bool(tokens.get("access_token"))

    def authenticate(self, timeout: int = 120) -> None:
        """
        Run the OAuth2 browser flow.  Opens the browser, waits for the user
        to log in on cTrader's website, then saves tokens to disk.

        The browser is opened to:
          https://connect.spotware.com/apps/authorize?client_id=...

        cTrader redirects back to http://localhost:8182/callback
        with the authorization code.  ZEngine catches this, exchanges the
        code for access + refresh tokens, and saves them to:
          ~/.zengine/ctrader_tokens.json

        Args:
            timeout : seconds to wait for browser completion (default 120)

        Raises:
            RuntimeError if packages missing, auth fails, or user cancels.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("client_id and client_secret must be set before authenticating.")

        tokens = _run_oauth_flow(self.client_id, self.client_secret, timeout)
        self._save_tokens(tokens)

    def refresh_tokens(self) -> bool:
        """
        Try to refresh the access token using the stored refresh token.
        Returns True on success, False if refresh token is expired/missing.
        """
        tokens = self._load_tokens()
        if not tokens.get("refresh_token"):
            return False
        try:
            new_tokens = _refresh_access_token(
                self.client_id, self.client_secret,
                tokens["refresh_token"],
            )
            # Merge — keep refresh_token if new response doesn't include one
            if "refresh_token" not in new_tokens:
                new_tokens["refresh_token"] = tokens["refresh_token"]
            self._save_tokens(new_tokens)
            return True
        except RuntimeError:
            return False

    # ── Bar data ──────────────────────────────────────────────────────────────

    def fetch_bars(
        self,
        symbol:    str,
        timeframe: str,
        from_date: datetime,
        to_date:   datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars with per-bar bid/ask spread from cTrader.

        Fetches bid trendbar and ask trendbar series separately, then
        computes spread = ask_open − bid_open at every bar.

        Args:
            symbol    : cTrader symbol name, e.g. "XAUUSD"
            timeframe : "M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1"
            from_date : start of range (UTC-aware or naive, treated as UTC)
            to_date   : end of range

        Returns:
            DataFrame with UTC DatetimeIndex and columns:
            open, high, low, close, volume, spread
            spread is in price units (ask_open - bid_open).

        Raises:
            RuntimeError  if not authenticated, API error, or no data returned.
            ValueError    if timeframe is unrecognised.
        """
        if not _CTRADER_AVAILABLE:
            raise RuntimeError("ctrader-open-api not installed. Run: pip install \"zengine[ctrader]\"")

        if not self.is_authenticated():
            raise RuntimeError(
                "Not authenticated. Call authenticate() first or use the UI's "
                "🔑 Authenticate button."
            )

        _build_tf_map()
        tf_value = _TF_MAP.get(timeframe.upper())
        if tf_value is None:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Supported: {list(_TF_MAP.keys())}"
            )

        # Ensure UTC timestamps in milliseconds
        from_ms = int(_to_utc(from_date).timestamp() * 1000)
        to_ms   = int(_to_utc(to_date).timestamp()   * 1000)

        tokens     = self._load_tokens()
        access_tok = tokens["access_token"]
        host       = DEMO_HOST if self.use_demo else LIVE_HOST

        try:
            return asyncio.run(self._async_fetch_bars(
                host, access_tok, symbol, tf_value, timeframe, from_ms, to_ms
            ))
        except RuntimeError as e:
            if "token" in str(e).lower() or "auth" in str(e).lower():
                if self.refresh_tokens():
                    tokens     = self._load_tokens()
                    access_tok = tokens["access_token"]
                    return asyncio.run(self._async_fetch_bars(
                        host, access_tok, symbol, tf_value, timeframe, from_ms, to_ms
                    ))
            raise

    async def _async_fetch_bars(
        self, host: str, access_token: str,
        symbol: str, tf_value: int, tf_str: str,
        from_ms: int, to_ms: int,
    ) -> pd.DataFrame:
        """Async implementation — connects, authenticates, fetches, disconnects."""
        reader, writer = await _open_connection(host)
        try:
            # 1. Application auth
            await _send(writer, _PT_APP_AUTH_REQ,
                        ProtoOAApplicationAuthReq(
                            clientId=self.client_id,
                            clientSecret=self.client_secret,
                        ))
            await _recv_type(reader, _PT_APP_AUTH_RES)

            # 2. Get accounts and select the right one (demo vs live)
            await _send(writer, _PT_GET_ACCOUNTS_REQ,
                        ProtoOAGetAccountListByAccessTokenReq(
                            accessToken=access_token
                        ))
            accts_msg = await _recv_type(reader, _PT_GET_ACCOUNTS_RES)
            accts_res = ProtoOAGetAccountListByAccessTokenRes()
            accts_res.ParseFromString(accts_msg.payload)

            accounts = [a for a in accts_res.ctidTraderAccount
                        if a.isLive == (not self.use_demo)]
            if not accounts:
                mode = "demo" if self.use_demo else "live"
                raise RuntimeError(
                    f"No {mode} accounts found for this access token. "
                    f"Make sure you have a {mode} account linked to your cTrader ID."
                )
            account_id = accounts[0].ctidTraderAccountId

            # 3. Account auth
            await _send(writer, _PT_ACCOUNT_AUTH_REQ,
                        ProtoOAAccountAuthReq(
                            ctidTraderAccountId=account_id,
                            accessToken=access_token,
                        ))
            await _recv_type(reader, _PT_ACCOUNT_AUTH_RES)

            # 4. Resolve symbol ID
            await _send(writer, _PT_SYMBOLS_LIST_REQ,
                        ProtoOASymbolsListReq(ctidTraderAccountId=account_id))
            syms_msg = await _recv_type(reader, _PT_SYMBOLS_LIST_RES)
            syms_res = ProtoOASymbolsListRes()
            syms_res.ParseFromString(syms_msg.payload)

            sym_obj = next(
                (s for s in syms_res.symbol if s.symbolName == symbol), None
            )
            if sym_obj is None:
                available = sorted(s.symbolName for s in syms_res.symbol)[:20]
                raise RuntimeError(
                    f"Symbol '{symbol}' not found. "
                    f"First 20 available: {available}"
                )
            symbol_id = sym_obj.symbolId
            digits    = sym_obj.digits

            # 5. Fetch bid bars (paginated)
            bid_bars = await self._fetch_all_bars(
                reader, writer, account_id, symbol_id,
                tf_value, tf_str, from_ms, to_ms, bid=True
            )
            # 6. Fetch ask bars for spread computation
            ask_bars = await self._fetch_all_bars(
                reader, writer, account_id, symbol_id,
                tf_value, tf_str, from_ms, to_ms, bid=False
            )

            return _build_dataframe(bid_bars, ask_bars, digits)

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _fetch_all_bars(
        self, reader, writer,
        account_id: int, symbol_id: int,
        tf_value: int, tf_str: str,
        from_ms: int, to_ms: int,
        bid: bool,
    ) -> list:
        """
        Fetch trendbar data in paginated chunks of _MAX_BARS_PER_REQUEST.

        cTrader returns bars in reverse chronological order per request.
        We collect all chunks and return a flat list in chronological order.
        """
        bar_duration_ms = _TF_SECONDS.get(tf_str, 60) * 1000
        all_bars = []
        chunk_to_ms = to_ms

        while True:
            req = ProtoOAGetTrendbarsReq(
                ctidTraderAccountId=account_id,
                symbolId=symbol_id,
                period=tf_value,
                fromTimestamp=from_ms,
                toTimestamp=chunk_to_ms,
                count=_MAX_BARS_PER_REQUEST,
            )
            # Note: bidOrAsk field not available in all API versions;
            # we request bid bars by default and ask bars by setting the flag
            # if supported.  For older API versions, both calls return bid bars
            # and spread will be zero (graceful degradation).
            if not bid:
                try:
                    req.bidOrAsk = 2   # 1=bid, 2=ask  (field added in newer proto versions)
                except AttributeError:
                    pass   # older API version — ask bars unavailable, spread will be 0

            await _send(writer, _PT_GET_TRENDBARS_REQ, req)
            res_msg = await _recv_type(reader, _PT_GET_TRENDBARS_RES)
            res     = ProtoOAGetTrendbarsRes()
            res.ParseFromString(res_msg.payload)

            bars = list(res.trendbar)
            if not bars:
                break

            all_bars.extend(bars)

            # Oldest bar in this chunk tells us where to start the next request
            oldest_ts_ms = bars[-1].utcTimestampInMinutes * 60 * 1000
            if oldest_ts_ms <= from_ms:
                break   # reached the start of our range
            if len(bars) < _MAX_BARS_PER_REQUEST:
                break   # server returned fewer bars than asked — we have everything

            # Step back for next chunk
            chunk_to_ms = oldest_ts_ms - bar_duration_ms

        # Sort chronologically (requests return newest-first per chunk)
        all_bars.sort(key=lambda b: b.utcTimestampInMinutes)
        return all_bars

    # ── Broker defaults ───────────────────────────────────────────────────────

    def get_broker_defaults(self, symbol: str) -> dict:
        """
        Extract BrokerConfig-compatible defaults from cTrader symbol info.

        Connects briefly to the API to read symbol metadata:
            spread          — typical spread in price units
            commission_flat — per-lot commission
            lot_size        — contract size (e.g. 100000 for EURUSD, 100 for XAUUSD)
            slippage_fixed  — 5 ticks as a conservative default

        Returns empty dict if not authenticated or API unavailable.
        """
        if not _CTRADER_AVAILABLE or not self.is_authenticated():
            return {}
        try:
            return asyncio.run(self._async_get_defaults(symbol))
        except Exception:
            return {}

    async def _async_get_defaults(self, symbol: str) -> dict:
        tokens     = self._load_tokens()
        access_tok = tokens["access_token"]
        host       = DEMO_HOST if self.use_demo else LIVE_HOST

        reader, writer = await _open_connection(host)
        try:
            await _send(writer, _PT_APP_AUTH_REQ,
                        ProtoOAApplicationAuthReq(
                            clientId=self.client_id,
                            clientSecret=self.client_secret,
                        ))
            await _recv_type(reader, _PT_APP_AUTH_RES)

            await _send(writer, _PT_GET_ACCOUNTS_REQ,
                        ProtoOAGetAccountListByAccessTokenReq(accessToken=access_tok))
            accts_msg = await _recv_type(reader, _PT_GET_ACCOUNTS_RES)
            accts_res = ProtoOAGetAccountListByAccessTokenRes()
            accts_res.ParseFromString(accts_msg.payload)
            accounts  = [a for a in accts_res.ctidTraderAccount
                         if a.isLive == (not self.use_demo)]
            if not accounts:
                return {}
            account_id = accounts[0].ctidTraderAccountId

            await _send(writer, _PT_ACCOUNT_AUTH_REQ,
                        ProtoOAAccountAuthReq(
                            ctidTraderAccountId=account_id,
                            accessToken=access_tok,
                        ))
            await _recv_type(reader, _PT_ACCOUNT_AUTH_RES)

            await _send(writer, _PT_SYMBOLS_LIST_REQ,
                        ProtoOASymbolsListReq(ctidTraderAccountId=account_id))
            syms_msg = await _recv_type(reader, _PT_SYMBOLS_LIST_RES)
            syms_res = ProtoOASymbolsListRes()
            syms_res.ParseFromString(syms_msg.payload)

            sym_obj = next(
                (s for s in syms_res.symbol if s.symbolName == symbol), None
            )
            if sym_obj is None:
                return {}

            point = 10 ** (-sym_obj.digits)
            return {
                "spread":          round(sym_obj.spread * point, 5),
                "commission_flat": round(getattr(sym_obj, "commissionValueInUSD", 0.0) or 0.0, 2),
                "lot_size":        float(getattr(sym_obj, "lotSize", 100000)),
                "slippage_fixed":  round(point * 5, 5),
            }
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # ── Spread schedule ───────────────────────────────────────────────────────

    def build_spread_schedule(self, df: pd.DataFrame) -> dict[int, float]:
        """
        Derive a spread_schedule multiplier dict from fetched bar data.

        Since cTrader provides per-bar spread (ask_open − bid_open), this
        gives an accurate picture of how spread varies by time of day.
        Normalises so the tightest (lowest-spread) hour = 1.0.

        Args:
            df : DataFrame as returned by fetch_bars() — must have 'spread' column

        Returns:
            {hour_utc (int): multiplier (float)} for BrokerConfig.spread_schedule
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

        valid      = hourly_mean[hourly_mean > 0]
        min_spread = valid.min()
        if np.isnan(min_spread) or min_spread == 0:
            return {}

        return {
            int(h): round(float(s / min_spread), 3)
            for h, s in hourly_mean.items()
            if not np.isnan(s) and s > 0
        }

    def list_symbols(self) -> list[str]:
        """
        Return a list of available symbol names for the connected account.
        Returns empty list if not authenticated.
        """
        if not _CTRADER_AVAILABLE or not self.is_authenticated():
            return []
        try:
            return asyncio.run(self._async_list_symbols())
        except Exception:
            return []

    async def _async_list_symbols(self) -> list[str]:
        tokens     = self._load_tokens()
        access_tok = tokens["access_token"]
        host       = DEMO_HOST if self.use_demo else LIVE_HOST

        reader, writer = await _open_connection(host)
        try:
            await _send(writer, _PT_APP_AUTH_REQ,
                        ProtoOAApplicationAuthReq(
                            clientId=self.client_id,
                            clientSecret=self.client_secret,
                        ))
            await _recv_type(reader, _PT_APP_AUTH_RES)

            await _send(writer, _PT_GET_ACCOUNTS_REQ,
                        ProtoOAGetAccountListByAccessTokenReq(accessToken=access_tok))
            accts_msg = await _recv_type(reader, _PT_GET_ACCOUNTS_RES)
            accts_res = ProtoOAGetAccountListByAccessTokenRes()
            accts_res.ParseFromString(accts_msg.payload)
            accounts  = [a for a in accts_res.ctidTraderAccount
                         if a.isLive == (not self.use_demo)]
            if not accounts:
                return []
            account_id = accounts[0].ctidTraderAccountId

            await _send(writer, _PT_ACCOUNT_AUTH_REQ,
                        ProtoOAAccountAuthReq(
                            ctidTraderAccountId=account_id,
                            accessToken=access_tok,
                        ))
            await _recv_type(reader, _PT_ACCOUNT_AUTH_RES)

            await _send(writer, _PT_SYMBOLS_LIST_REQ,
                        ProtoOASymbolsListReq(ctidTraderAccountId=account_id))
            syms_msg = await _recv_type(reader, _PT_SYMBOLS_LIST_RES)
            syms_res = ProtoOASymbolsListRes()
            syms_res.ParseFromString(syms_msg.payload)
            return sorted(s.symbolName for s in syms_res.symbol)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # ── Token storage ─────────────────────────────────────────────────────────

    def _load_tokens(self) -> dict:
        if TOKEN_FILE.exists():
            try:
                return json.loads(TOKEN_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_tokens(self, tokens: dict) -> None:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(tokens, indent=2))

    def clear_tokens(self) -> None:
        """Remove stored tokens (force re-authentication on next use)."""
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()


# ── Price decoding ─────────────────────────────────────────────────────────────

def _to_utc(dt: datetime) -> datetime:
    """Ensure a datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_dataframe(
    bid_bars: list,
    ask_bars: list,
    digits: int,
) -> pd.DataFrame:
    """
    Convert cTrader trendbar lists to a DataFrame.

    cTrader trendbars use delta encoding:
      open  = low + deltaOpen
      high  = low + deltaHigh
      close = low + deltaClose
    All values are raw integers; divide by 10^digits for actual price.

    Spread is computed as ask_open − bid_open at each bar.
    If ask bars are unavailable (older API version), spread = 0.
    """
    if not bid_bars:
        raise RuntimeError("No bar data returned from cTrader API.")

    divisor = 10 ** digits

    def _decode(bars):
        records = []
        for b in bars:
            low_raw   = b.low
            open_raw  = low_raw + b.deltaOpen
            high_raw  = low_raw + b.deltaHigh
            close_raw = low_raw + b.deltaClose
            ts = datetime.fromtimestamp(
                b.utcTimestampInMinutes * 60, tz=timezone.utc
            )
            records.append({
                "time":   ts,
                "open":   open_raw  / divisor,
                "high":   high_raw  / divisor,
                "low":    low_raw   / divisor,
                "close":  close_raw / divisor,
                "volume": getattr(b, "volume", 0),
            })
        return records

    bid_records = _decode(bid_bars)

    # Build ask open prices for spread computation
    ask_open_by_ts: dict = {}
    if ask_bars:
        divisor_ask = divisor
        for b in ask_bars:
            ts  = datetime.fromtimestamp(b.utcTimestampInMinutes * 60, tz=timezone.utc)
            ask_open_by_ts[ts] = (b.low + b.deltaOpen) / divisor_ask

    df = pd.DataFrame(bid_records)
    df.set_index("time", inplace=True)
    df.index.name = None

    # Per-bar spread: ask_open - bid_open (0 if ask bars unavailable)
    df["spread"] = df.index.map(
        lambda ts: round(ask_open_by_ts.get(ts, df.at[ts, "open"]) - df.at[ts, "open"], 5)
    )
    # Floor at 0 (rounding artefacts can produce tiny negatives)
    df["spread"] = df["spread"].clip(lower=0.0)

    df.sort_index(inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    return df[["open", "high", "low", "close", "volume", "spread"]]
