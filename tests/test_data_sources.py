"""
tests/test_data_sources.py — Unit tests for engine/data_sources/
=================================================================
All external dependencies (MetaTrader5, ctrader-open-api, yfinance) are mocked
so these tests run on any platform without a live terminal or network.

Tests cover:
  - CSVSource:      loading from path and bytes; normalisation; error cases
  - MT5Source:      platform detection; bar fetch; broker defaults; spread schedule
  - CTraderSource:  availability; token file; fetch_bars; build_dataframe helper
  - YahooSource:    availability check; interval warnings; bar fetch; normalisation
  - Package:        available_sources() returns the right set per platform
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from engine.data_sources.csv_source      import CSVSource
from engine.data_sources.mt5_source      import MT5Source
from engine.data_sources.ctrader_source  import CTraderSource, _build_dataframe
from engine.data_sources.yahoo_source    import YahooSource, GOLD_SYMBOLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv_csv(n: int = 5) -> str:
    """Build a minimal OHLCV CSV string for testing."""
    lines = ["datetime,Open,High,Low,Close,Volume"]
    for i in range(n):
        ts = f"2024-01-{i+1:02d} 00:00:00"     # zero-pad day so Jan 10 → 2024-01-10
        lines.append(f"{ts},{2000+i},{2010+i},{1990+i},{2005+i},{1000}")
    return "\n".join(lines)


def _make_ohlcv_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a UTC DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "open":   np.arange(2000, 2000 + n, dtype=float),
        "high":   np.arange(2010, 2010 + n, dtype=float),
        "low":    np.arange(1990, 1990 + n, dtype=float),
        "close":  np.arange(2005, 2005 + n, dtype=float),
        "volume": np.ones(n) * 100,
        "spread": np.full(n, 0.30),
    }, index=idx)


# ── CSVSource ─────────────────────────────────────────────────────────────────

class TestCSVSource:

    def test_is_always_available(self):
        assert CSVSource().is_available() is True

    def test_platform_note_empty(self):
        assert CSVSource().platform_note() == ""

    def test_load_from_bytes(self):
        src = CSVSource()
        raw = _make_ohlcv_csv(10).encode()
        df  = src.fetch_bars(raw)
        assert len(df) == 10
        assert list(df.columns[:4]) == ["open", "high", "low", "close"]

    def test_load_from_bytesio(self):
        src = CSVSource()
        buf = io.BytesIO(_make_ohlcv_csv(5).encode())
        df  = src.fetch_bars(buf)
        assert len(df) == 5

    def test_datetime_index_is_utc(self):
        src = CSVSource()
        df  = src.fetch_bars(_make_ohlcv_csv(3).encode())
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_columns_normalised_to_lowercase(self):
        csv_text = "DateTime,OPEN,HIGH,LOW,CLOSE,VOLUME\n2024-01-01,2000,2010,1990,2005,100\n"
        src = CSVSource()
        df  = src.fetch_bars(csv_text.encode())
        assert "open" in df.columns
        assert "OPEN" not in df.columns

    def test_missing_ohlc_column_raises(self):
        csv_text = "datetime,open,high,low\n2024-01-01,2000,2010,1990\n"
        src = CSVSource()
        with pytest.raises(ValueError, match="close"):
            src.fetch_bars(csv_text.encode())

    def test_sorted_by_index(self):
        # Rows deliberately out of order
        csv = (
            "datetime,open,high,low,close\n"
            "2024-01-03,2000,2010,1990,2005\n"
            "2024-01-01,2001,2011,1991,2006\n"
            "2024-01-02,2002,2012,1992,2007\n"
        )
        df = CSVSource().fetch_bars(csv.encode())
        assert df.index.is_monotonic_increasing

    def test_nan_rows_dropped(self):
        csv = (
            "datetime,open,high,low,close\n"
            "2024-01-01,2000,2010,1990,2005\n"
            "2024-01-02,,,, \n"
            "2024-01-03,2001,2011,1991,2006\n"
        )
        df = CSVSource().fetch_bars(csv.encode())
        assert len(df) == 2

    def test_broker_defaults_empty(self):
        assert CSVSource().get_broker_defaults() == {}

    def test_spread_schedule_empty(self):
        df = _make_ohlcv_df()
        assert CSVSource().build_spread_schedule(df) == {}


# ── MT5Source ─────────────────────────────────────────────────────────────────

class TestMT5SourceAvailability:

    def test_unavailable_on_non_windows(self):
        """On Linux/Mac the source must report unavailable, regardless of package."""
        with patch("engine.data_sources.mt5_source.platform.system", return_value="Darwin"):
            src = MT5Source()
            assert src.is_available() is False

    def test_unavailable_when_package_missing(self):
        """If MetaTrader5 is not importable, is_available() returns False."""
        with patch("engine.data_sources.mt5_source._MT5_AVAILABLE", False):
            src = MT5Source()
            assert src.is_available() is False

    def test_platform_note_non_windows(self):
        with patch("engine.data_sources.mt5_source.platform.system", return_value="Darwin"):
            note = MT5Source.platform_note()
            assert "Windows" in note
            # Should mention CSV and Yahoo Finance as alternatives
            assert "CSV" in note or "Yahoo" in note

    def test_platform_note_windows_no_package(self):
        with (
            patch("engine.data_sources.mt5_source.platform.system", return_value="Windows"),
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", False),
        ):
            note = MT5Source.platform_note()
            assert "pip install" in note

    def test_is_available_windows_terminal_running(self):
        """Simulate a successful Windows + terminal connection."""
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True
        mock_mt5.shutdown.return_value   = None
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source.platform.system", return_value="Windows"),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            assert MT5Source().is_available() is True

    def test_is_available_windows_terminal_not_running(self):
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = False
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source.platform.system", return_value="Windows"),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            assert MT5Source().is_available() is False


class TestMT5SourceFetchBars:
    """fetch_bars() tests using a fully mocked MT5 module."""

    def _mock_rates(self, n: int = 5):
        """Structured array that matches MetaTrader5 copy_rates_range output."""
        import numpy as np
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
        records = []
        for i in range(n):
            records.append((
                base_ts + i * 60,   # time  (unix seconds)
                2000.0 + i,         # open
                2010.0 + i,         # high
                1990.0 + i,         # low
                2005.0 + i,         # close
                1000,               # tick_volume
                30,                 # spread (integer points)
                0,                  # real_volume
            ))
        dtype = np.dtype([
            ("time", np.int64), ("open", np.float64), ("high", np.float64),
            ("low",  np.float64), ("close", np.float64), ("tick_volume", np.int64),
            ("spread", np.int32), ("real_volume", np.int64),
        ])
        return np.array(records, dtype=dtype)

    def _mock_symbol_info(self, point=0.01, spread=30, contract_size=100.0, commission=6.0):
        info = SimpleNamespace(
            point=point,
            spread=spread,
            trade_contract_size=contract_size,
            trade_commission_value=commission,
        )
        return info

    def _make_mt5_mock(self, n=5):
        m = MagicMock()
        m.initialize.return_value = True
        m.shutdown.return_value   = None
        m.copy_rates_range.return_value = self._mock_rates(n)
        m.symbol_info.return_value      = self._mock_symbol_info()
        m.last_error.return_value       = (0, "OK")
        return m

    def test_returns_dataframe_with_correct_columns(self):
        mock_mt5 = self._make_mt5_mock()
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source._TF_MAP", {"M1": 1}),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            df = MT5Source().fetch_bars(
                "XAUUSD", "M1",
                datetime(2024, 1, 1), datetime(2024, 1, 2)
            )
        assert set(df.columns) == {"open", "high", "low", "close", "volume", "spread"}

    def test_spread_converted_to_price_units(self):
        """30 points × 0.01 point_size = $0.30 spread."""
        mock_mt5 = self._make_mt5_mock()
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source._TF_MAP", {"M1": 1}),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            df = MT5Source().fetch_bars(
                "XAUUSD", "M1",
                datetime(2024, 1, 1), datetime(2024, 1, 2)
            )
        assert df["spread"].iloc[0] == pytest.approx(0.30, rel=1e-4)

    def test_index_is_utc_datetime(self):
        mock_mt5 = self._make_mt5_mock()
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source._TF_MAP", {"M1": 1}),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            df = MT5Source().fetch_bars(
                "XAUUSD", "M1",
                datetime(2024, 1, 1), datetime(2024, 1, 2)
            )
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_invalid_timeframe_raises(self):
        with patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True):
            with pytest.raises(ValueError, match="Unknown timeframe"):
                MT5Source().fetch_bars("XAUUSD", "W1", datetime(2024, 1, 1), datetime(2024, 2, 1))

    def test_package_not_installed_raises(self):
        with patch("engine.data_sources.mt5_source._MT5_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="not installed"):
                MT5Source().fetch_bars("XAUUSD", "M1", datetime(2024, 1, 1), datetime(2024, 1, 2))

    def test_shutdown_called_even_on_error(self):
        """Ensure mt5.shutdown() is always called (context manager pattern)."""
        mock_mt5 = self._make_mt5_mock()
        mock_mt5.copy_rates_range.return_value = None  # simulate no data
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source._TF_MAP", {"M1": 1}),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            with pytest.raises(RuntimeError):
                MT5Source().fetch_bars("XAUUSD", "M1", datetime(2024, 1, 1), datetime(2024, 1, 2))
        mock_mt5.shutdown.assert_called()


class TestMT5BrokerDefaults:

    def _make_info_mock(self):
        return SimpleNamespace(
            point=0.01,
            spread=30,
            trade_contract_size=100.0,
            trade_commission_value=6.0,
        )

    def test_returns_expected_keys(self):
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True
        mock_mt5.symbol_info.return_value = self._make_info_mock()
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            defaults = MT5Source().get_broker_defaults("XAUUSD")
        assert "spread"          in defaults
        assert "commission_flat" in defaults
        assert "lot_size"        in defaults
        assert "slippage_fixed"  in defaults

    def test_spread_matches_symbol_info(self):
        """spread = 30 points × 0.01 = 0.30"""
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True
        mock_mt5.symbol_info.return_value = self._make_info_mock()
        with (
            patch("engine.data_sources.mt5_source._MT5_AVAILABLE", True),
            patch("engine.data_sources.mt5_source.mt5", mock_mt5),
        ):
            defaults = MT5Source().get_broker_defaults("XAUUSD")
        assert defaults["spread"] == pytest.approx(0.30, rel=1e-4)

    def test_empty_dict_when_package_missing(self):
        with patch("engine.data_sources.mt5_source._MT5_AVAILABLE", False):
            assert MT5Source().get_broker_defaults("XAUUSD") == {}


class TestMT5SpreadSchedule:

    def test_schedule_normalised_to_1_at_minimum(self):
        df = _make_ohlcv_df(24 * 60)  # 24 hours of M1 bars
        # Manually set hour-dependent spread
        df["spread"] = (df.index.hour + 1) * 0.10   # hour 0 → 0.10, hour 23 → 2.40
        sched = MT5Source().build_spread_schedule(df)
        assert sched[0] == pytest.approx(1.0, rel=1e-2), "Cheapest hour should be 1.0"

    def test_schedule_covers_all_hours_present(self):
        df = _make_ohlcv_df(24 * 60)
        df["spread"] = 0.30
        sched = MT5Source().build_spread_schedule(df)
        assert len(sched) == 24

    def test_empty_when_no_spread_column(self):
        df = _make_ohlcv_df()
        df.drop(columns=["spread"], inplace=True)
        assert MT5Source().build_spread_schedule(df) == {}

    def test_empty_when_spread_all_zero(self):
        df = _make_ohlcv_df()
        df["spread"] = 0.0
        assert MT5Source().build_spread_schedule(df) == {}


# ── CTraderSource ─────────────────────────────────────────────────────────────

class TestCTraderSourceAvailability:

    def test_available_when_packages_installed(self):
        with (
            patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", True),
            patch("engine.data_sources.ctrader_source._REQUESTS_AVAILABLE", True),
        ):
            assert CTraderSource().is_available() is True

    def test_unavailable_when_ctrader_package_missing(self):
        with patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", False):
            assert CTraderSource().is_available() is False

    def test_unavailable_when_requests_missing(self):
        with (
            patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", True),
            patch("engine.data_sources.ctrader_source._REQUESTS_AVAILABLE", False),
        ):
            assert CTraderSource().is_available() is False

    def test_platform_note_lists_missing_ctrader_package(self):
        with (
            patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", False),
            patch("engine.data_sources.ctrader_source._REQUESTS_AVAILABLE", True),
        ):
            note = CTraderSource.platform_note()
            assert "ctrader-open-api" in note
            assert "pip install" in note

    def test_platform_note_empty_when_all_available(self):
        with (
            patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", True),
            patch("engine.data_sources.ctrader_source._REQUESTS_AVAILABLE", True),
        ):
            assert CTraderSource.platform_note() == ""


class TestCTraderTokens:

    def test_not_authenticated_without_token_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE",
            tmp_path / "tokens.json"
        )
        assert CTraderSource().is_authenticated() is False

    def test_authenticated_when_token_file_present(self, tmp_path, monkeypatch):
        token_file = tmp_path / "tokens.json"
        token_file.write_text(json.dumps({"access_token": "abc123"}))
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE", token_file
        )
        assert CTraderSource().is_authenticated() is True

    def test_clear_tokens_removes_file(self, tmp_path, monkeypatch):
        token_file = tmp_path / "tokens.json"
        token_file.write_text(json.dumps({"access_token": "abc123"}))
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE", token_file
        )
        CTraderSource().clear_tokens()
        assert not token_file.exists()

    def test_load_tokens_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE",
            tmp_path / "nonexistent.json"
        )
        assert CTraderSource()._load_tokens() == {}

    def test_save_then_load_round_trip(self, tmp_path, monkeypatch):
        token_file = tmp_path / ".zengine" / "ctrader_tokens.json"
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE", token_file
        )
        src = CTraderSource()
        src._save_tokens({"access_token": "tok123", "refresh_token": "ref456"})
        loaded = src._load_tokens()
        assert loaded["access_token"]  == "tok123"
        assert loaded["refresh_token"] == "ref456"


class TestCTraderFetchBars:

    def test_raises_when_ctrader_package_missing(self):
        with patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="not installed"):
                CTraderSource().fetch_bars(
                    "XAUUSD", "M1", datetime(2024, 1, 1), datetime(2024, 1, 2)
                )

    def test_raises_when_not_authenticated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE",
            tmp_path / "tokens.json"
        )
        with patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", True):
            with pytest.raises(RuntimeError, match="Not authenticated"):
                CTraderSource().fetch_bars(
                    "XAUUSD", "M1", datetime(2024, 1, 1), datetime(2024, 1, 2)
                )

    def test_raises_on_unknown_timeframe(self, tmp_path, monkeypatch):
        token_file = tmp_path / "tokens.json"
        token_file.write_text(json.dumps({"access_token": "tok"}))
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE", token_file
        )
        with (
            patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", True),
            patch("engine.data_sources.ctrader_source._TF_MAP", {"M1": 1, "D1": 5}),
        ):
            with pytest.raises(ValueError, match="Unknown timeframe"):
                CTraderSource().fetch_bars(
                    "XAUUSD", "W2", datetime(2024, 1, 1), datetime(2024, 1, 2)
                )

    def test_fetch_bars_returns_dataframe(self, tmp_path, monkeypatch):
        """fetch_bars() with mocked _async_fetch_bars returns expected columns."""
        token_file = tmp_path / "tokens.json"
        token_file.write_text(json.dumps({"access_token": "tok"}))
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE", token_file
        )
        mock_df = _make_ohlcv_df(10)

        # Patch the async method so asyncio.run returns our mock DataFrame
        async def _fake_fetch(*args, **kwargs):
            return mock_df

        with (
            patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", True),
            patch("engine.data_sources.ctrader_source._TF_MAP", {"M1": 1}),
            patch.object(CTraderSource, "_async_fetch_bars", _fake_fetch),
        ):
            df = CTraderSource(client_id="id", client_secret="sec").fetch_bars(
                "XAUUSD", "M1", datetime(2024, 1, 1), datetime(2024, 1, 11)
            )
        assert set(df.columns) == {"open", "high", "low", "close", "volume", "spread"}
        assert len(df) == 10

    def test_broker_defaults_empty_when_not_authenticated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "engine.data_sources.ctrader_source.TOKEN_FILE",
            tmp_path / "tokens.json"
        )
        with patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", True):
            assert CTraderSource().get_broker_defaults("XAUUSD") == {}

    def test_broker_defaults_empty_when_package_missing(self):
        with patch("engine.data_sources.ctrader_source._CTRADER_AVAILABLE", False):
            assert CTraderSource().get_broker_defaults("XAUUSD") == {}


class TestBuildDataframe:
    """Tests for _build_dataframe() — the delta-decoding / spread computation helper."""

    def _make_bar(self, ts_minutes=1000, low=200000,
                  delta_open=0, delta_high=1000, delta_close=500, volume=100):
        """Build a SimpleNamespace that quacks like a ProtoOATrendbar."""
        return SimpleNamespace(
            utcTimestampInMinutes=ts_minutes,
            low=delta_open,   # unused alias — low is the named field
            deltaOpen=delta_open,
            deltaHigh=delta_high,
            deltaClose=delta_close,
            volume=volume,
        )

    def _make_real_bar(self, ts_minutes=1000, low_raw=200000,
                       delta_open=0, delta_high=1000, delta_close=500, volume=100):
        """Build a bar with a proper .low field (raw integer price units)."""
        return SimpleNamespace(
            utcTimestampInMinutes=ts_minutes,
            low=low_raw,
            deltaOpen=delta_open,
            deltaHigh=delta_high,
            deltaClose=delta_close,
            volume=volume,
        )

    def test_raises_on_empty_bid_bars(self):
        with pytest.raises(RuntimeError, match="No bar data"):
            _build_dataframe([], [], digits=2)

    def test_delta_decoding_open(self):
        """open = (low_raw + deltaOpen) / 10^digits"""
        # low=200000, deltaOpen=300, digits=2 → open = 200300/100 = 2003.0
        bar = self._make_real_bar(low_raw=200000, delta_open=300,
                                  delta_high=1000, delta_close=500)
        df = _build_dataframe([bar], [], digits=2)
        assert df["open"].iloc[0]  == pytest.approx(2003.0,  rel=1e-4)
        assert df["high"].iloc[0]  == pytest.approx(2010.0,  rel=1e-4)   # (200000+1000)/100
        assert df["low"].iloc[0]   == pytest.approx(2000.0,  rel=1e-4)   # 200000/100
        assert df["close"].iloc[0] == pytest.approx(2005.0,  rel=1e-4)   # (200000+500)/100

    def test_spread_is_ask_open_minus_bid_open(self):
        """spread = ask_open − bid_open for matching timestamps."""
        # bid bar: low=200000, deltaOpen=0  → bid_open = 200000/100 = 2000.0
        # ask bar: low=200030, deltaOpen=0  → ask_open = 200030/100 = 2000.30
        # expected spread = 0.30
        bid_bar = self._make_real_bar(ts_minutes=1000, low_raw=200000, delta_open=0)
        ask_bar = self._make_real_bar(ts_minutes=1000, low_raw=200030, delta_open=0)
        df = _build_dataframe([bid_bar], [ask_bar], digits=2)
        assert df["spread"].iloc[0] == pytest.approx(0.30, rel=1e-4)

    def test_spread_zero_when_no_ask_bars(self):
        """When ask bars are unavailable, spread should be 0."""
        bar = self._make_real_bar(ts_minutes=1000, low_raw=200000, delta_open=0)
        df = _build_dataframe([bar], [], digits=2)
        assert df["spread"].iloc[0] == pytest.approx(0.0, abs=1e-5)

    def test_spread_never_negative(self):
        """Rounding artefacts must not produce negative spread values."""
        # Ask open slightly below bid open due to float rounding — should be clipped to 0
        bid_bar = self._make_real_bar(ts_minutes=1000, low_raw=200005, delta_open=0)
        ask_bar = self._make_real_bar(ts_minutes=1000, low_raw=200000, delta_open=0)
        df = _build_dataframe([bid_bar], [ask_bar], digits=2)
        assert df["spread"].iloc[0] >= 0.0

    def test_index_is_utc_aware(self):
        bar = self._make_real_bar(ts_minutes=1000, low_raw=200000)
        df = _build_dataframe([bar], [], digits=2)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_output_columns_exact(self):
        bar = self._make_real_bar(ts_minutes=1000, low_raw=200000)
        df = _build_dataframe([bar], [], digits=2)
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "spread"]

    def test_multiple_bars_sorted_chronologically(self):
        # Bars deliberately in reverse order to verify sort
        bars = [
            self._make_real_bar(ts_minutes=1002, low_raw=200200),
            self._make_real_bar(ts_minutes=1000, low_raw=200000),
            self._make_real_bar(ts_minutes=1001, low_raw=200100),
        ]
        df = _build_dataframe(bars, [], digits=2)
        assert df.index.is_monotonic_increasing


class TestCTraderSpreadSchedule:

    def test_schedule_normalised_to_1_at_minimum_hour(self):
        df = _make_ohlcv_df(24 * 60)   # 24 hours of M1 bars
        # Spread increases with hour → hour 0 should be the minimum = 1.0
        df["spread"] = (df.index.hour + 1) * 0.10
        sched = CTraderSource().build_spread_schedule(df)
        assert sched[0] == pytest.approx(1.0, rel=1e-2)

    def test_schedule_covers_all_hours_in_data(self):
        df = _make_ohlcv_df(24 * 60)
        df["spread"] = 0.30
        sched = CTraderSource().build_spread_schedule(df)
        assert len(sched) == 24

    def test_empty_when_no_spread_column(self):
        df = _make_ohlcv_df()
        df.drop(columns=["spread"], inplace=True)
        assert CTraderSource().build_spread_schedule(df) == {}

    def test_empty_when_spread_all_zero(self):
        df = _make_ohlcv_df()
        df["spread"] = 0.0
        assert CTraderSource().build_spread_schedule(df) == {}


# ── YahooSource ───────────────────────────────────────────────────────────────

class TestYahooSourceAvailability:

    def test_available_when_yfinance_installed(self):
        with patch("engine.data_sources.yahoo_source._YF_AVAILABLE", True):
            assert YahooSource().is_available() is True

    def test_unavailable_when_yfinance_missing(self):
        with patch("engine.data_sources.yahoo_source._YF_AVAILABLE", False):
            assert YahooSource().is_available() is False

    def test_platform_note_missing_package(self):
        with patch("engine.data_sources.yahoo_source._YF_AVAILABLE", False):
            note = YahooSource.platform_note()
            assert "pip install" in note

    def test_platform_note_empty_when_available(self):
        with patch("engine.data_sources.yahoo_source._YF_AVAILABLE", True):
            assert YahooSource.platform_note() == ""


class TestYahooIntervalWarning:

    def test_no_warning_for_daily(self):
        from_dt = datetime(2020, 1, 1)
        to_dt   = datetime(2024, 1, 1)
        assert YahooSource.interval_warning("1d", from_dt, to_dt) is None

    def test_warning_for_1m_old_data(self):
        from_dt = datetime(2020, 1, 1)    # way beyond 7-day limit
        to_dt   = datetime(2024, 1, 1)
        warn = YahooSource.interval_warning("1m", from_dt, to_dt)
        assert warn is not None
        assert "7 days" in warn

    def test_no_warning_for_1m_recent_data(self):
        from datetime import timedelta
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=3)     # within 7-day window
        assert YahooSource.interval_warning("1m", from_dt, to_dt) is None

    def test_warning_for_5m_60_day_limit(self):
        from_dt = datetime(2023, 1, 1)
        to_dt   = datetime(2024, 1, 1)
        warn = YahooSource.interval_warning("5m", from_dt, to_dt)
        assert warn is not None
        assert "60 days" in warn


class TestYahooSourceFetchBars:

    def _make_yf_df(self, n: int = 10) -> pd.DataFrame:
        """Build a DataFrame that mimics yfinance output."""
        idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
        return pd.DataFrame({
            "Open":     np.arange(2000, 2000 + n, dtype=float),
            "High":     np.arange(2010, 2010 + n, dtype=float),
            "Low":      np.arange(1990, 1990 + n, dtype=float),
            "Close":    np.arange(2005, 2005 + n, dtype=float),
            "Volume":   np.ones(n) * 1000,
            "Dividends":np.zeros(n),
        }, index=idx)

    def test_returns_ohlcv_columns_only(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_yf_df()
        mock_yf.Ticker.return_value = mock_ticker

        with (
            patch("engine.data_sources.yahoo_source._YF_AVAILABLE", True),
            patch("engine.data_sources.yahoo_source.yf", mock_yf),
        ):
            df = YahooSource().fetch_bars(
                "GC=F", "1d",
                datetime(2024, 1, 1), datetime(2024, 1, 11)
            )

        assert "open"      in df.columns
        assert "dividends" not in df.columns

    def test_index_utc_aware(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_yf_df()
        mock_yf.Ticker.return_value = mock_ticker

        with (
            patch("engine.data_sources.yahoo_source._YF_AVAILABLE", True),
            patch("engine.data_sources.yahoo_source.yf", mock_yf),
        ):
            df = YahooSource().fetch_bars("GC=F", "1d", datetime(2024, 1, 1), datetime(2024, 1, 11))

        assert df.index.tz is not None

    def test_package_missing_raises(self):
        with patch("engine.data_sources.yahoo_source._YF_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="not installed"):
                YahooSource().fetch_bars("GC=F", "1d", datetime(2024, 1, 1), datetime(2024, 1, 11))

    def test_empty_result_raises(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        with (
            patch("engine.data_sources.yahoo_source._YF_AVAILABLE", True),
            patch("engine.data_sources.yahoo_source.yf", mock_yf),
        ):
            with pytest.raises(RuntimeError, match="No data"):
                YahooSource().fetch_bars("INVALID", "1d", datetime(2024, 1, 1), datetime(2024, 1, 11))

    def test_broker_defaults_always_empty(self):
        assert YahooSource().get_broker_defaults("GC=F") == {}

    def test_spread_schedule_always_empty(self):
        df = _make_ohlcv_df()
        assert YahooSource().build_spread_schedule(df) == {}


# ── Gold symbols dict ─────────────────────────────────────────────────────────

class TestGoldSymbols:

    def test_gcf_in_gold_symbols(self):
        assert "GC=F" in GOLD_SYMBOLS

    def test_xauusd_in_gold_symbols(self):
        assert "XAUUSD=X" in GOLD_SYMBOLS

    def test_all_entries_have_descriptions(self):
        for sym, desc in GOLD_SYMBOLS.items():
            assert isinstance(desc, str) and len(desc) > 0
