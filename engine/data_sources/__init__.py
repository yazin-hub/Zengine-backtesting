"""
engine/data_sources — Pluggable data source abstraction
==========================================================

All data sources expose a common interface so the engine and UI stay
source-agnostic — they call source.fetch_bars(...) regardless of whether
data comes from a local file, an MT5 terminal, a cTrader account, or a web API.

Available sources
-----------------
CSVSource      — load from a local CSV or uploaded file bytes  (always available)
MT5Source      — pull from a running MetaTrader 5 terminal     (Windows only)
CTraderSource  — pull via cTrader Open API (OAuth2)            (cross-platform ✅)
YahooSource    — pull via yfinance                             (cross-platform ✅;
                 M1 data limited to last 7 days, daily unlimited)

Quick usage
-----------
from engine.data_sources import CSVSource, MT5Source, CTraderSource, YahooSource

src = CTraderSource(client_id="...", client_secret="...", use_demo=True)
if not src.is_authenticated():
    src.authenticate()             # opens browser once, saves tokens
df       = src.fetch_bars("XAUUSD", "M1", from_dt, to_dt)
defaults = src.get_broker_defaults("XAUUSD")
schedule = src.build_spread_schedule(df)
"""

from engine.data_sources.csv_source      import CSVSource
from engine.data_sources.mt5_source      import MT5Source
from engine.data_sources.ctrader_source  import CTraderSource
from engine.data_sources.yahoo_source    import YahooSource


def available_sources() -> dict[str, object]:
    """
    Return {display_name: source_instance} for every source usable on this machine.

    CSVSource is always present.  MT5 requires Windows + the terminal running.
    CTraderSource requires ctrader-open-api + requests (cross-platform).
    YahooSource requires yfinance (cross-platform).
    """
    sources: dict[str, object] = {"📄 CSV / File Upload": CSVSource()}

    # MetaTrader 5 — Windows only
    mt5_src = MT5Source()
    if mt5_src.is_available():
        sources["🖥️ MetaTrader 5 (connected)"] = mt5_src
    elif MT5Source.platform_note():
        # Show on non-Windows so the UI can display the platform note and alternatives
        sources["🖥️ MetaTrader 5 (unavailable on this OS)"] = mt5_src

    # cTrader — cross-platform, OAuth2
    ct_src = CTraderSource()
    if ct_src.is_available():
        sources["🔐 cTrader (cross-platform)"] = ct_src
    elif CTraderSource.platform_note():
        # Show even when packages are missing so the user sees the install instructions
        sources["🔐 cTrader (install packages)"] = ct_src

    # Yahoo Finance — cross-platform, free, no account
    yahoo_src = YahooSource()
    if yahoo_src.is_available():
        sources["🌐 Yahoo Finance"] = yahoo_src

    return sources


__all__ = ["CSVSource", "MT5Source", "CTraderSource", "YahooSource", "available_sources"]
