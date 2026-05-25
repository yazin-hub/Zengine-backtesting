"""
engine/data_sources/csv_source.py — CSV / file-upload data source
===================================================================
Thin wrapper around the existing CSV loading logic, exposing it through
the DataSource interface so all sources are treated uniformly.

Accepts a local file path (string or Path) or in-memory bytes from a
Streamlit file uploader widget.

Always available — no external dependencies.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pandas as pd


class CSVSource:
    """
    Load OHLCV bar data from a local CSV file or an in-memory buffer.

    Expected CSV format
    -------------------
    - First column : datetime (any parseable format; UTC assumed if tz-naive)
    - Remaining    : Open, High, Low, Close, Volume  (Volume optional)
    - Column names : case-insensitive (normalised to lowercase internally)

    No external dependencies — always available on every platform.
    """

    def is_available(self) -> bool:
        return True

    @staticmethod
    def platform_note() -> str:
        return ""

    def fetch_bars(
        self,
        path_or_buffer: Union[str, Path, io.IOBase, bytes],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load bars from a CSV path or buffer.

        Args:
            path_or_buffer : file path (str/Path), BytesIO, or raw bytes
                             (e.g. from st.file_uploader().read())

        Returns:
            DataFrame with a UTC-aware DatetimeIndex and lowercase columns:
            open, high, low, close, [volume]

        Raises:
            FileNotFoundError if a path is given but doesn't exist.
            ValueError        if required OHLC columns are missing.
        """
        if isinstance(path_or_buffer, bytes):
            path_or_buffer = io.BytesIO(path_or_buffer)

        df = pd.read_csv(path_or_buffer, parse_dates=[0], index_col=0)
        df.columns = df.columns.str.lower()

        required = {"open", "high", "low", "close"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found: {list(df.columns)}"
            )

        df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        return df

    def get_broker_defaults(self, **kwargs) -> dict:
        """CSV files carry no broker metadata — caller keeps manual settings."""
        return {}

    def build_spread_schedule(self, df: pd.DataFrame) -> dict:
        """CSV files have no spread column — return empty schedule."""
        return {}
