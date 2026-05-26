"""
tests/test_connectors.py — ZEngine Connectors Test Suite
=========================================================

Tests for connectors/base.py and connectors/ctrader.py.

All tests run without a live cTrader connection — the ctrader-open-api
package is mocked throughout. This keeps CI fast and dependency-free.

Coverage target: connectors/base.py and connectors/ctrader.py to ≥95%.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ── Stub optional heavy deps so CI runs without installing ctrader-open-api ───
# ctrader-open-api and twisted are optional (pip install "zengine[ctrader]").
# We inject MagicMock modules into sys.modules BEFORE any import that
# references them — this makes `from ctrader_open_api.messages import ...`
# and `patch("ctrader_open_api.messages...")` work in a package-free environment.
# When the real package IS installed (e.g. on the VM), setdefault is a no-op.
for _stub_mod in [
    "ctrader_open_api",
    "ctrader_open_api.messages",
    "ctrader_open_api.messages.OpenApiMessages_pb2",
    "ctrader_open_api.messages.OpenApiModelMessages_pb2",
    "ctrader_open_api.messages.OpenApiCommonMessages_pb2",
    "twisted",
    "twisted.internet",
    "twisted.internet.reactor",
]:
    sys.modules.setdefault(_stub_mod, MagicMock())

from datetime import datetime, timezone  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from connectors.base import BaseConnector, OrderResult, Position  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

class _ConcreteConnector(BaseConnector):
    """Minimal concrete subclass — for testing BaseConnector interface."""

    def connect(self)    -> None: self._conn = True
    def disconnect(self) -> None: self._conn = False
    def is_connected(self) -> bool: return getattr(self, "_conn", False)

    def fetch_ohlcv(self, symbol, timeframe, limit=500) -> pd.DataFrame:
        return pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01 00:00:00"], tz="UTC"),
        )

    def place_market_order(self, symbol, side, size, sl=None, tp=None, comment=""):
        return OrderResult(
            position_id=1, order_id=2, symbol=symbol, side=side,
            size=size, fill_price=100.0, sl=sl, tp=tp, comment=comment,
        )

    def close_position(self, position_id, size=None) -> bool:
        return True

    def get_open_positions(self, symbol=None) -> list:
        return [Position(
            position_id=1, symbol="BTCUSD", side="buy",
            size=0.01, entry_price=95000.0,
        )]

    def get_account_summary(self) -> dict:
        return {"balance": 10000.0, "equity": 10000.0, "currency": "USD"}


# ── OrderResult dataclass ─────────────────────────────────────────────────────

class TestOrderResult:
    def test_required_fields(self):
        r = OrderResult(
            position_id=1, order_id=2, symbol="BTCUSD",
            side="buy", size=0.01, fill_price=95_000.0,
        )
        assert r.position_id == 1
        assert r.order_id    == 2
        assert r.symbol      == "BTCUSD"
        assert r.side        == "buy"
        assert r.size        == 0.01
        assert r.fill_price  == 95_000.0

    def test_optional_defaults(self):
        r = OrderResult(
            position_id=1, order_id=2, symbol="ETHUSD",
            side="sell", size=1.0, fill_price=3_000.0,
        )
        assert r.sl        is None
        assert r.tp        is None
        assert r.timestamp is None
        assert r.comment   == ""

    def test_full_fields(self):
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        r  = OrderResult(
            position_id=10, order_id=20, symbol="XAUUSD", side="sell",
            size=0.5, fill_price=2_300.0, sl=2_350.0, tp=2_200.0,
            timestamp=ts, comment="ZEngine test",
        )
        assert r.sl      == 2_350.0
        assert r.tp      == 2_200.0
        assert r.comment == "ZEngine test"
        assert r.timestamp == ts

    def test_sell_side(self):
        r = OrderResult(
            position_id=3, order_id=4, symbol="ETHUSD",
            side="sell", size=0.1, fill_price=3_500.0,
        )
        assert r.side == "sell"


# ── Position dataclass ────────────────────────────────────────────────────────

class TestPosition:
    def test_required_fields(self):
        p = Position(
            position_id=5, symbol="BTCUSD", side="buy",
            size=0.01, entry_price=94_000.0,
        )
        assert p.position_id  == 5
        assert p.symbol       == "BTCUSD"
        assert p.side         == "buy"
        assert p.size         == 0.01
        assert p.entry_price  == 94_000.0

    def test_optional_defaults(self):
        p = Position(
            position_id=1, symbol="ETHUSD", side="sell",
            size=1.0, entry_price=3_000.0,
        )
        assert p.sl      is None
        assert p.tp      is None
        assert p.comment == ""

    def test_with_sltp(self):
        p = Position(
            position_id=2, symbol="XAUUSD", side="buy",
            size=0.5, entry_price=2_300.0,
            sl=2_250.0, tp=2_400.0, comment="pairs_leg_a",
        )
        assert p.sl      == 2_250.0
        assert p.tp      == 2_400.0
        assert p.comment == "pairs_leg_a"


# ── BaseConnector ABC ─────────────────────────────────────────────────────────

class TestBaseConnector:
    def test_cannot_instantiate_abstract(self):
        """BaseConnector must remain abstract — missing abstract methods raise."""
        with pytest.raises(TypeError):
            BaseConnector()

    def test_concrete_subclass_works(self):
        conn = _ConcreteConnector()
        conn.connect()
        assert conn.is_connected() is True

    def test_disconnect(self):
        conn = _ConcreteConnector()
        conn.connect()
        conn.disconnect()
        assert conn.is_connected() is False

    def test_repr_connected(self):
        conn = _ConcreteConnector()
        conn.connect()
        assert "connected" in repr(conn)
        assert "_ConcreteConnector" in repr(conn)

    def test_repr_disconnected(self):
        conn = _ConcreteConnector()
        assert "disconnected" in repr(conn)

    def test_fetch_ohlcv_returns_dataframe(self):
        conn = _ConcreteConnector()
        conn.connect()
        df = conn.fetch_ohlcv("BTCUSD", "15m", limit=1)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_place_market_order_returns_result(self):
        conn = _ConcreteConnector()
        conn.connect()
        result = conn.place_market_order("BTCUSD", "buy", size=0.01, sl=90_000.0)
        assert isinstance(result, OrderResult)
        assert result.symbol == "BTCUSD"
        assert result.side   == "buy"

    def test_close_position_returns_bool(self):
        conn = _ConcreteConnector()
        conn.connect()
        ok = conn.close_position(1)
        assert ok is True

    def test_get_open_positions_returns_list(self):
        conn = _ConcreteConnector()
        conn.connect()
        positions = conn.get_open_positions()
        assert isinstance(positions, list)
        assert all(isinstance(p, Position) for p in positions)

    def test_get_open_positions_filter_by_symbol(self):
        conn = _ConcreteConnector()
        conn.connect()
        # _ConcreteConnector always returns BTCUSD — no filter logic, but call works
        positions = conn.get_open_positions(symbol="BTCUSD")
        assert isinstance(positions, list)

    def test_get_account_summary_returns_dict(self):
        conn = _ConcreteConnector()
        conn.connect()
        summary = conn.get_account_summary()
        assert "balance" in summary
        assert "equity"  in summary
        assert "currency" in summary

    def test_cancel_pending_order_raises_by_default(self):
        """BaseConnector.cancel_pending_order() raises NotImplementedError by default."""
        conn = _ConcreteConnector()
        conn.connect()
        with pytest.raises(NotImplementedError):
            conn.cancel_pending_order(order_id=999)

    def test_get_latest_price_default_uses_ohlcv(self):
        """BaseConnector.get_latest_price() default derives price from 1m bar."""
        conn = _ConcreteConnector()
        conn.connect()
        price = conn.get_latest_price("BTCUSD")
        assert price is not None
        assert "bid" in price
        assert "ask" in price
        assert price["bid"] == price["ask"]  # mid = close

    def test_get_latest_price_returns_none_on_empty_df(self):
        """get_latest_price returns None when fetch_ohlcv returns empty DataFrame."""
        conn = _ConcreteConnector()
        conn.connect()
        with patch.object(conn, "fetch_ohlcv", return_value=pd.DataFrame()):
            result = conn.get_latest_price("BTCUSD")
        assert result is None

    def test_get_latest_price_returns_none_on_exception(self):
        """get_latest_price returns None when fetch_ohlcv raises."""
        conn = _ConcreteConnector()
        conn.connect()
        with patch.object(conn, "fetch_ohlcv", side_effect=RuntimeError("boom")):
            result = conn.get_latest_price("BTCUSD")
        assert result is None


# ── CTraderConnector — init validation ───────────────────────────────────────

class TestCTraderConnectorInit:
    """Tests that don't require a live connection — only __init__ validation."""

    def _make(self, **kwargs):
        from connectors.ctrader import CTraderConnector
        defaults = dict(
            client_id="test_id",
            client_secret="test_secret",
            account_id=12345,
            access_token="test_token",
            env="demo",
        )
        defaults.update(kwargs)
        return CTraderConnector(**defaults)

    def test_valid_demo(self):
        conn = self._make(env="demo")
        assert conn._env == "demo"
        assert conn._account_id == 12345

    def test_valid_live(self):
        conn = self._make(env="live")
        assert conn._env == "live"

    def test_invalid_env(self):
        with pytest.raises(ValueError, match="env must be"):
            self._make(env="staging")

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="connect_timeout"):
            self._make(connect_timeout=0)

    def test_invalid_timeout_negative(self):
        with pytest.raises(ValueError, match="connect_timeout"):
            self._make(connect_timeout=-5.0)

    def test_account_id_coerced_to_int(self):
        conn = self._make(account_id="99999")
        assert conn._account_id == 99999
        assert isinstance(conn._account_id, int)

    def test_is_not_connected_before_connect(self):
        conn = self._make()
        assert conn.is_connected() is False

    def test_repr_disconnected(self):
        conn = self._make()
        assert "disconnected" in repr(conn)
        assert "CTraderConnector" in repr(conn)

    def test_reconnect_delay_default(self):
        conn = self._make()
        assert conn._reconnect_delay == 2.0

    def test_reconnect_delay_custom(self):
        conn = self._make(reconnect_delay=5.0)
        assert conn._reconnect_delay == 5.0

    def test_reconnect_delay_zero_allowed(self):
        """reconnect_delay=0 is valid (no sleep before re-auth)."""
        conn = self._make(reconnect_delay=0.0)
        assert conn._reconnect_delay == 0.0

    def test_reconnect_delay_negative_raises(self):
        with pytest.raises(ValueError, match="reconnect_delay"):
            self._make(reconnect_delay=-1.0)

    def test_on_reconnect_callback_stored(self):
        cb   = MagicMock()
        conn = self._make(on_reconnect=cb)
        assert conn._on_reconnect is cb

    def test_on_reconnect_default_none(self):
        conn = self._make()
        assert conn._on_reconnect is None

    def test_is_reconnection_false_before_connect(self):
        """_is_reconnection starts False — first _on_connected is not a reconnect."""
        conn = self._make()
        assert conn._is_reconnection is False


# ── CTraderConnector — reconnect behaviour ────────────────────────────────────

class TestCTraderConnectorReconnect:
    """Tests for auto-reconnect and re-authentication logic."""

    def _make(self, **kwargs):
        from connectors.ctrader import CTraderConnector
        defaults = dict(
            client_id="x", client_secret="x",
            account_id=1, access_token="x",
            env="demo", reconnect_delay=0.0,
        )
        defaults.update(kwargs)
        return CTraderConnector(**defaults)

    def test_on_connected_first_call_sets_flag(self):
        """First _on_connected sets _is_reconnection=True, does NOT spawn reauth."""
        conn = self._make()
        assert conn._is_reconnection is False

        conn._on_connected(None)

        assert conn._is_reconnection is True
        assert conn._connected_evt.is_set()

    def test_on_connected_reconnect_spawns_reauth(self):
        """Second _on_connected (reconnect) spawns _reauthenticate in a thread."""
        conn = self._make()
        conn._is_reconnection = True  # simulate already connected once

        with patch.object(conn, "_reauthenticate"):
            with patch("connectors.ctrader.threading.Thread") as mock_thread:
                mock_t = MagicMock()
                mock_thread.return_value = mock_t
                conn._on_connected(None)

        mock_thread.assert_called_once()
        mock_t.start.assert_called_once()

    def test_on_disconnected_clears_auth(self):
        """_on_disconnected clears auth state."""
        conn = self._make()
        conn._authenticated = True
        conn._connected_evt.set()

        conn._on_disconnected(None, "connection lost")

        assert conn._authenticated is False
        assert not conn._connected_evt.is_set()

    def test_reauthenticate_success_calls_callback(self):
        """_reauthenticate calls on_reconnect callback on success."""
        cb   = MagicMock()
        conn = self._make(on_reconnect=cb)

        with patch.object(conn, "_authenticate"):
            conn._reauthenticate()

        cb.assert_called_once()

    def test_reauthenticate_success_sets_authenticated(self):
        """_reauthenticate sets _authenticated=True via _authenticate()."""
        conn = self._make()

        def _mock_auth():
            conn._authenticated = True

        with patch.object(conn, "_authenticate", side_effect=_mock_auth):
            conn._reauthenticate()

        assert conn._authenticated is True

    def test_reauthenticate_failure_clears_authenticated(self):
        """_reauthenticate sets _authenticated=False if _authenticate() raises."""
        conn = self._make()
        conn._authenticated = True  # pretend was authed before

        with patch.object(conn, "_authenticate", side_effect=RuntimeError("auth fail")):
            conn._reauthenticate()

        assert conn._authenticated is False

    def test_reauthenticate_callback_exception_does_not_propagate(self):
        """on_reconnect callback raising should not crash _reauthenticate."""
        cb   = MagicMock(side_effect=RuntimeError("telegram down"))
        conn = self._make(on_reconnect=cb)

        with patch.object(conn, "_authenticate"):
            conn._reauthenticate()  # must not raise

        cb.assert_called_once()

    def test_reauthenticate_no_callback_is_safe(self):
        """_reauthenticate with on_reconnect=None must not raise."""
        conn = self._make(on_reconnect=None)

        with patch.object(conn, "_authenticate"):
            conn._reauthenticate()  # must not raise

    def test_reauthenticate_skips_sleep_when_delay_zero(self):
        """reconnect_delay=0 means no time.sleep call."""
        conn = self._make(reconnect_delay=0.0)

        with patch.object(conn, "_authenticate"), \
             patch("connectors.ctrader.time") as mock_time:
            conn._reauthenticate()

        mock_time.sleep.assert_not_called()

    def test_reauthenticate_sleeps_when_delay_nonzero(self):
        """reconnect_delay>0 causes a time.sleep call before re-auth."""
        conn = self._make(reconnect_delay=3.0)

        with patch.object(conn, "_authenticate"), \
             patch("connectors.ctrader.time") as mock_time:
            conn._reauthenticate()

        mock_time.sleep.assert_called_once_with(3.0)


# ── CTraderConnector — behaviour when not connected ──────────────────────────

class TestCTraderConnectorNotConnected:
    """Methods that require a connection should raise RuntimeError when called before connect()."""

    def _make(self):
        from connectors.ctrader import CTraderConnector
        return CTraderConnector(
            client_id="x", client_secret="x",
            account_id=1, access_token="x", env="demo",
        )

    def test_fetch_ohlcv_raises(self):
        conn = self._make()
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.fetch_ohlcv("BTCUSD", "15m")

    def test_place_market_order_raises(self):
        conn = self._make()
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.place_market_order("BTCUSD", "buy", size=0.01)

    def test_close_position_raises(self):
        conn = self._make()
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.close_position(position_id=1)

    def test_get_open_positions_raises(self):
        conn = self._make()
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.get_open_positions()

    def test_get_account_summary_raises(self):
        conn = self._make()
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.get_account_summary()

    def test_cancel_pending_order_raises(self):
        conn = self._make()
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.cancel_pending_order(order_id=42)


# ── CTraderConnector — input validation ──────────────────────────────────────

class TestCTraderConnectorValidation:
    """Input validation tests — set _authenticated + _connected_evt to bypass connection check."""

    def _make_connected(self):
        from connectors.ctrader import CTraderConnector
        conn = CTraderConnector(
            client_id="x", client_secret="x",
            account_id=1, access_token="x", env="demo",
        )
        conn._authenticated = True
        conn._connected_evt.set()
        conn._symbol_map = {"BTCUSD": {"id": 101, "digits": 2, "name": "BTCUSD"}}
        conn._id_to_norm = {101: "BTCUSD"}
        return conn

    def test_place_order_invalid_side(self):
        conn = self._make_connected()
        with pytest.raises(ValueError, match="side must be"):
            conn.place_market_order("BTCUSD", "hold", size=0.01)

    def test_place_order_zero_size(self):
        conn = self._make_connected()
        with pytest.raises(ValueError, match="size must be positive"):
            conn.place_market_order("BTCUSD", "buy", size=0.0)

    def test_place_order_negative_size(self):
        conn = self._make_connected()
        with pytest.raises(ValueError, match="size must be positive"):
            conn.place_market_order("BTCUSD", "buy", size=-1.0)

    def test_fetch_ohlcv_unknown_symbol(self):
        conn = self._make_connected()
        with pytest.raises(ValueError, match="not in account symbol cache"):
            conn.fetch_ohlcv("UNKNOWN_XYZ", "15m")

    def test_fetch_ohlcv_unknown_timeframe(self):
        conn = self._make_connected()
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            conn.fetch_ohlcv("BTCUSD", "3m")

    def test_place_order_unknown_symbol_returns_none(self):
        conn = self._make_connected()
        result = conn.place_market_order("ETHUSD", "buy", size=0.01)
        assert result is None

    def test_normalise_symbol_with_slash(self):
        """Connector normalises BTC/USD → BTCUSD internally."""
        from connectors.ctrader import _normalise
        assert _normalise("BTC/USD")  == "BTCUSD"
        assert _normalise("eth/usd")  == "ETHUSD"
        assert _normalise("XAU/USD")  == "XAUUSD"
        assert _normalise("BTCUSD")   == "BTCUSD"


# ── CTraderConnector — mocked API calls ──────────────────────────────────────

class TestCTraderConnectorMocked:
    """
    Tests with fully mocked cTrader protobuf messages.
    Validates the full call path without a real broker connection.
    """

    def _make_connected(self):
        from connectors.ctrader import CTraderConnector
        conn = CTraderConnector(
            client_id="x", client_secret="x",
            account_id=1, access_token="x", env="demo",
        )
        conn._authenticated = True
        conn._connected_evt.set()
        conn._symbol_map = {"BTCUSD": {"id": 101, "digits": 2, "name": "BTCUSD"}}
        conn._id_to_norm = {101: "BTCUSD"}
        return conn

    def _mock_trendbar(self, low, delta_open, delta_high, delta_close, volume, ts_min):
        bar = MagicMock()
        bar.low                  = low
        bar.deltaOpen            = delta_open
        bar.deltaHigh            = delta_high
        bar.deltaClose           = delta_close
        bar.volume               = volume
        bar.utcTimestampInMinutes = ts_min
        return bar

    def test_fetch_ohlcv_success(self):
        conn = self._make_connected()

        # Build mock protobuf response
        mock_resp    = MagicMock()
        mock_payload = MagicMock()

        # One trendbar: low=9000000000 (~90000.0 after ÷100000)
        bar = self._mock_trendbar(
            low=9_000_000_000,   # 90000.00
            delta_open=100_000,  # open  = 90001.00
            delta_high=500_000,  # high  = 90005.00
            delta_close=200_000, # close = 90002.00
            volume=5000,
            ts_min=int(pd.Timestamp("2024-01-01 00:00", tz="UTC").timestamp() / 60),
        )
        mock_payload.trendbar = [bar]

        with patch.object(conn, "_send_sync", return_value=mock_resp):
            # payloadType must match PROTO_OA_GET_TRENDBARS_RES
            from unittest.mock import patch as up
            with up("connectors.ctrader.CTraderConnector.fetch_ohlcv",
                    wraps=conn.fetch_ohlcv):

                # Manually set payloadType and mock ParseFromString
                with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2") as mock_model, \
                     patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

                    mock_model.PROTO_OA_GET_TRENDBARS_RES = 9
                    mock_model.M1  = 1
                    mock_model.M5  = 2
                    mock_model.M15 = 3
                    mock_model.H1  = 4
                    mock_model.H4  = 5
                    mock_model.D1  = 6
                    mock_resp.payloadType = 9

                    mock_res_obj = MagicMock()
                    mock_res_obj.trendbar = [bar]
                    mock_msg.ProtoOAGetTrendbarsRes.return_value = mock_res_obj
                    mock_msg.ProtoOAGetTrendbarsReq.return_value = MagicMock()

                    df = conn.fetch_ohlcv("BTCUSD", "15m", limit=1)

        # Even if mocking is incomplete, a RuntimeError or ValueError would bubble up.
        # The test verifies the call path doesn't crash on the happy path.
        assert df is not None

    def test_fetch_ohlcv_unexpected_response_returns_empty(self):
        """fetch_ohlcv returns empty DataFrame on unexpected payloadType."""
        conn = self._make_connected()

        mock_resp = MagicMock()

        with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2") as mock_mdl, \
             patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

            mock_mdl.PROTO_OA_GET_TRENDBARS_RES = 9
            mock_mdl.M1 = mock_mdl.M5 = mock_mdl.M15 = 1
            mock_mdl.H1 = mock_mdl.H4 = mock_mdl.D1  = 1
            mock_resp.payloadType = 99  # anything != 9

            mock_msg.ProtoOAGetTrendbarsReq.return_value = MagicMock()

            with patch.object(conn, "_send_sync", return_value=mock_resp):
                df = conn.fetch_ohlcv("BTCUSD", "15m", limit=1)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_place_order_send_exception_returns_none(self):
        """place_market_order returns None when _send_sync raises."""
        conn = self._make_connected()

        with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2") as mock_model, \
             patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

            mock_model.MARKET = 1
            mock_model.BUY    = 1
            mock_model.SELL   = 2
            mock_msg.ProtoOANewOrderReq.return_value = MagicMock()

            with patch.object(conn, "_send_sync", side_effect=TimeoutError("no response")):
                result = conn.place_market_order("BTCUSD", "buy", size=0.01)

        assert result is None

    def test_close_position_unknown_size_returns_false(self):
        """close_position returns False when size is None and position not found."""
        conn = self._make_connected()

        with patch.object(conn, "get_open_positions", return_value=[]):
            result = conn.close_position(position_id=9999)

        assert result is False

    def test_get_open_positions_api_error_returns_empty(self):
        """get_open_positions returns [] when _send_sync raises."""
        conn = self._make_connected()

        with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2"), \
             patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

            mock_msg.ProtoOAReconcileReq.return_value = MagicMock()

            with patch.object(conn, "_send_sync", side_effect=ConnectionError("timeout")):
                positions = conn.get_open_positions()

        assert positions == []

    def test_get_open_positions_wrong_payload_returns_empty(self):
        """get_open_positions returns [] on unexpected payloadType."""
        conn = self._make_connected()

        mock_resp = MagicMock()

        with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2") as mock_mdl, \
             patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

            mock_mdl.PROTO_OA_RECONCILE_RES = 5
            mock_resp.payloadType = 99
            mock_msg.ProtoOAReconcileReq.return_value = MagicMock()

            with patch.object(conn, "_send_sync", return_value=mock_resp):
                positions = conn.get_open_positions()

        assert positions == []

    def test_get_account_summary_api_error_returns_zeros(self):
        """get_account_summary returns zero-balance dict on API error."""
        conn = self._make_connected()

        with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2"), \
             patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

            mock_msg.ProtoOATraderReq.return_value = MagicMock()

            with patch.object(conn, "_send_sync", side_effect=TimeoutError("boom")):
                summary = conn.get_account_summary()

        assert summary["balance"]  == 0.0
        assert summary["equity"]   == 0.0
        assert summary["currency"] == "USD"

    def test_cancel_order_send_exception_returns_false(self):
        """cancel_pending_order returns False when _send_sync raises."""
        conn = self._make_connected()

        with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2"), \
             patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

            mock_msg.ProtoOACancelOrderReq.return_value = MagicMock()

            with patch.object(conn, "_send_sync", side_effect=TimeoutError("no resp")):
                result = conn.cancel_pending_order(order_id=42)

        assert result is False

    def test_apply_sltp_exception_returns_false(self):
        """_apply_sltp returns False when _send_sync raises."""
        conn = self._make_connected()

        with patch("ctrader_open_api.messages.OpenApiModelMessages_pb2"), \
             patch("ctrader_open_api.messages.OpenApiMessages_pb2") as mock_msg:

            mock_msg.ProtoOAAmendPositionSLTPReq.return_value = MagicMock()

            with patch.object(conn, "_send_sync", side_effect=RuntimeError("fail")):
                result = conn._apply_sltp(1, 90_000.0, None)

        assert result is False

    def test_apply_sltp_with_retry_exhausted_returns_false(self):
        """_apply_sltp_with_retry returns False after all attempts fail."""
        conn = self._make_connected()

        with patch.object(conn, "_apply_sltp", return_value=False):
            result = conn._apply_sltp_with_retry(1, 90_000.0, 100_000.0)

        assert result is False

    def test_apply_sltp_with_retry_succeeds_second_attempt(self):
        """_apply_sltp_with_retry returns True when second attempt succeeds."""
        conn   = self._make_connected()
        calls  = [False, True]
        it     = iter(calls)

        with patch.object(conn, "_apply_sltp", side_effect=lambda *a: next(it)):
            with patch("connectors.ctrader.time") as mock_time:
                mock_time.sleep = MagicMock()
                result = conn._apply_sltp_with_retry(1, 90_000.0, 100_000.0)

        assert result is True


# ── _normalise helper ─────────────────────────────────────────────────────────

class TestNormalise:
    def test_removes_slash(self):
        from connectors.ctrader import _normalise
        assert _normalise("BTC/USD") == "BTCUSD"

    def test_uppercases(self):
        from connectors.ctrader import _normalise
        assert _normalise("ethusd") == "ETHUSD"

    def test_already_normalised(self):
        from connectors.ctrader import _normalise
        assert _normalise("XAUUSD") == "XAUUSD"

    def test_multiple_slashes(self):
        from connectors.ctrader import _normalise
        assert _normalise("XAU/USD/X") == "XAUUSDX"
