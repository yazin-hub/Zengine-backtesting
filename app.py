"""
app.py — ZEngine Strategy Lab
================================
Streamlit UI for the look-ahead-free backtesting engine.

Run with:
    streamlit run app.py

Features:
  - Strategy selector (auto-discovers from strategies/registry)
  - Parameter sidebar (auto-generated from strategy PARAMS)
  - IS / OOS / FWD split configuration
  - Broker / risk settings
  - Equity curve (all splits overlaid)
  - Metrics table comparison
  - Trade browser with filters
  - Per-session and per-direction breakdown
  - Compare mode: run two param sets side by side
  - Export trades and equity to CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, date, timedelta

from engine.backtest import run_splits, BacktestResult
from engine.broker import BrokerConfig
from engine.metrics import compute_metrics, trade_log_df
from engine.walkforward import run_walk_forward
from engine.monte_carlo import run_monte_carlo
from engine.portfolio import PortfolioConfig, run_backtest_portfolio
from engine.funded import (
    FundedAccountConfig, FIRM_PRESETS, run_funded_backtest,
)
from engine.data_sources import CSVSource, MT5Source, CTraderSource, YahooSource, available_sources
from engine.data_sources.yahoo_source import GOLD_SYMBOLS
from strategies import REGISTRY

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ZEngine — Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours ─────────────────────────────────────────────────────────────
COLORS = {
    "IS":  "#4C9BE8",
    "OOS": "#F4A261",
    "FWD": "#2A9D8F",
    "A":   "#4C9BE8",
    "B":   "#E76F51",
    "win": "#2A9D8F",
    "loss":"#E76F51",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("⚙️ ZEngine Strategy Lab")
    st.sidebar.caption("Look-ahead-free backtesting · Open source")
    st.sidebar.divider()

    # ── Data Source ───────────────────────────────────────────────────────────
    st.sidebar.subheader("📂 Data")

    _sources      = available_sources()
    _source_names = list(_sources.keys())
    _source_sel   = st.sidebar.radio(
        "Source", _source_names, key="data_source_radio",
        help="MT5 requires the MetaTrader 5 terminal running on Windows. "
             "Yahoo Finance works on all platforms (daily bars; M1 last 7 days only). "
             "CSV works everywhere.",
    )
    _source_obj = _sources[_source_sel]

    # ── CSV ───────────────────────────────────────────────────────────────────
    if "CSV" in _source_sel:
        _use_uploader = st.sidebar.checkbox("Upload file instead of path", key="csv_upload_mode")
        if _use_uploader:
            _uploaded = st.sidebar.file_uploader(
                "Upload CSV", type=["csv", "txt"], key="csv_upload_file",
            )
            data_source_cfg = {"type": "csv_upload", "uploaded": _uploaded}
        else:
            _csv_path = st.sidebar.text_input(
                "CSV path", value="",
                placeholder="/path/to/data.csv",
                help="OHLCV CSV with DateTime index. "
                     "Columns: Open, High, Low, Close, Volume (Volume optional).",
                key="csv_path",
            )
            data_source_cfg = {"type": "csv_path", "path": _csv_path}

    # ── MetaTrader 5 ──────────────────────────────────────────────────────────
    elif "MetaTrader" in _source_sel:
        _note = MT5Source.platform_note()
        if _note:
            st.sidebar.warning(_note)
            data_source_cfg = {"type": "mt5_unavailable"}
        else:
            _mt5_sym = st.sidebar.text_input("Symbol", "XAUUSD", key="mt5_symbol")
            _mt5_tf  = st.sidebar.selectbox(
                "Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                key="mt5_tf",
            )
            _mt5_col1, _mt5_col2 = st.sidebar.columns(2)
            with _mt5_col1:
                _mt5_from = st.date_input("From", value=datetime(2020, 1, 1), key="mt5_from")
            with _mt5_col2:
                _mt5_to   = st.date_input("To",   value=datetime.today(),     key="mt5_to")

            if st.sidebar.button("📥 Fetch from MT5", key="mt5_fetch_btn", use_container_width=True):
                with st.sidebar.status("Connecting to MT5 terminal...") as _s:
                    try:
                        _src   = MT5Source()
                        _df_mt5 = _src.fetch_bars(
                            _mt5_sym, _mt5_tf,
                            datetime.combine(_mt5_from, datetime.min.time()),
                            datetime.combine(_mt5_to,   datetime.max.time()),
                        )
                        _defs   = _src.get_broker_defaults(_mt5_sym)
                        _sched  = _src.build_spread_schedule(_df_mt5)
                        st.session_state["loaded_df"]       = _df_mt5
                        st.session_state["mt5_defaults"]    = _defs
                        st.session_state["mt5_schedule"]    = _sched
                        st.session_state["data_source_tag"] = (
                            f"MT5 · {_mt5_sym} {_mt5_tf} · "
                            f"{_df_mt5.index[0].date()} → {_df_mt5.index[-1].date()} · "
                            f"{len(_df_mt5):,} bars"
                        )
                        _s.update(label="✅ MT5 data loaded!", state="complete")
                    except Exception as _e:
                        _s.update(label=f"❌ {_e}", state="error")

            # Show status if data already fetched
            if "data_source_tag" in st.session_state and "MT5" in st.session_state["data_source_tag"]:
                st.sidebar.success(st.session_state["data_source_tag"])
                # Surface auto-filled broker defaults so user knows what was applied
                _defs_loaded = st.session_state.get("mt5_defaults", {})
                if _defs_loaded:
                    st.sidebar.caption(
                        f"Auto-filled from MT5 · "
                        f"Spread: {_defs_loaded.get('spread', '–')} · "
                        f"Commission: ${_defs_loaded.get('commission_flat', '–')}/lot · "
                        f"Lot: {_defs_loaded.get('lot_size', '–')} units"
                    )

            data_source_cfg = {"type": "mt5_loaded"}

    # ── cTrader ───────────────────────────────────────────────────────────────
    elif "cTrader" in _source_sel:
        _ct_note = CTraderSource.platform_note()
        if _ct_note:
            st.sidebar.warning(_ct_note)
            data_source_cfg = {"type": "ctrader_unavailable"}
        else:
            st.sidebar.caption(
                "Register at [openapi.ctrader.com](https://openapi.ctrader.com) "
                "· Redirect URI: `http://localhost:8182/callback`"
            )
            _ct_id  = st.sidebar.text_input(
                "Client ID", value=st.session_state.get("ct_client_id", ""),
                key="ct_client_id",
                help="From openapi.ctrader.com → your application",
            )
            _ct_sec = st.sidebar.text_input(
                "Client Secret", value=st.session_state.get("ct_client_secret", ""),
                type="password", key="ct_client_secret",
            )
            _ct_mode = st.sidebar.radio(
                "Account type", ["Demo", "Live"],
                key="ct_demo_mode", horizontal=True,
                help="Demo spreads are ~20–40% tighter than live. Use Live for production accuracy.",
            )
            _ct_use_demo = (_ct_mode == "Demo")
            _ct_src = CTraderSource(
                client_id=_ct_id, client_secret=_ct_sec, use_demo=_ct_use_demo
            )

            # Auth status badge
            if _ct_src.is_authenticated():
                st.sidebar.success("✅ Authenticated — tokens saved")
            else:
                st.sidebar.info("⚠️ Not authenticated — click 🔑 Authenticate below")

            # Authenticate button — runs OAuth2 browser flow
            if st.sidebar.button("🔑 Authenticate", key="ct_auth_btn", use_container_width=True):
                if not _ct_id or not _ct_sec:
                    st.sidebar.error("Enter your Client ID and Client Secret first.")
                else:
                    with st.sidebar.status("Opening browser for cTrader login...") as _s:
                        try:
                            _ct_src.authenticate()
                            _s.update(label="✅ Authenticated!", state="complete")
                            st.rerun()
                        except Exception as _e:
                            _s.update(label=f"❌ {_e}", state="error")

            _ct_sym = st.sidebar.text_input("Symbol", "XAUUSD", key="ct_symbol")
            _ct_tf  = st.sidebar.selectbox(
                "Timeframe",
                ["M1", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1"],
                key="ct_tf",
            )
            _ct_col1, _ct_col2 = st.sidebar.columns(2)
            with _ct_col1:
                _ct_from = st.date_input("From", value=datetime(2024, 1, 1), key="ct_from")
            with _ct_col2:
                _ct_to   = st.date_input("To",   value=datetime.today(),     key="ct_to")

            if st.sidebar.button("📥 Fetch from cTrader", key="ct_fetch_btn", use_container_width=True):
                if not _ct_src.is_authenticated():
                    st.sidebar.error("Authenticate first using the 🔑 button above.")
                else:
                    with st.sidebar.status("Connecting to cTrader API...") as _s:
                        try:
                            _df_ct  = _ct_src.fetch_bars(
                                _ct_sym, _ct_tf,
                                datetime.combine(_ct_from, datetime.min.time()),
                                datetime.combine(_ct_to,   datetime.max.time()),
                            )
                            _defs   = _ct_src.get_broker_defaults(_ct_sym)
                            _sched  = _ct_src.build_spread_schedule(_df_ct)
                            st.session_state["loaded_df"]       = _df_ct
                            st.session_state["mt5_defaults"]    = _defs
                            st.session_state["mt5_schedule"]    = _sched
                            st.session_state["data_source_tag"] = (
                                f"cTrader · {_ct_mode} · {_ct_sym} {_ct_tf} · "
                                f"{_df_ct.index[0].date()} → {_df_ct.index[-1].date()} · "
                                f"{len(_df_ct):,} bars"
                            )
                            _s.update(label="✅ cTrader data loaded!", state="complete")
                        except Exception as _e:
                            _s.update(label=f"❌ {_e}", state="error")

            if "data_source_tag" in st.session_state and "cTrader" in st.session_state.get("data_source_tag", ""):
                st.sidebar.success(st.session_state["data_source_tag"])
                _defs_loaded = st.session_state.get("mt5_defaults", {})
                if _defs_loaded:
                    st.sidebar.caption(
                        f"Auto-filled from cTrader · "
                        f"Spread: {_defs_loaded.get('spread', '–')} · "
                        f"Commission: ${_defs_loaded.get('commission_flat', '–')}/lot · "
                        f"Lot: {_defs_loaded.get('lot_size', '–')} units"
                    )

            data_source_cfg = {"type": "ctrader_loaded"}

    # ── Yahoo Finance ─────────────────────────────────────────────────────────
    elif "Yahoo" in _source_sel:
        _note_yf = YahooSource.platform_note()
        if _note_yf:
            st.sidebar.warning(_note_yf)
            data_source_cfg = {"type": "yahoo_unavailable"}
        else:
            _yf_sym_preset = st.sidebar.selectbox(
                "Symbol preset",
                list(GOLD_SYMBOLS.keys()),
                format_func=lambda s: f"{s} — {GOLD_SYMBOLS[s]}",
                key="yf_sym_preset",
            )
            _yf_sym = st.sidebar.text_input(
                "Or type custom symbol", value=_yf_sym_preset, key="yf_sym_custom",
            )
            _yf_interval = st.sidebar.selectbox(
                "Interval",
                ["1d", "1h", "30m", "15m", "5m", "1m"],
                index=0,
                key="yf_interval",
                help="1m/5m/15m/30m are limited to recent weeks by Yahoo. Use 1d for long history.",
            )
            _yf_col1, _yf_col2 = st.sidebar.columns(2)
            with _yf_col1:
                _yf_from = st.date_input("From", value=datetime(2020, 1, 1), key="yf_from")
            with _yf_col2:
                _yf_to   = st.date_input("To",   value=datetime.today(),     key="yf_to")

            # Interval warning
            _yf_warn = YahooSource.interval_warning(
                _yf_interval,
                datetime.combine(_yf_from, datetime.min.time()),
                datetime.combine(_yf_to,   datetime.max.time()),
            )
            if _yf_warn:
                st.sidebar.warning(_yf_warn)

            if st.sidebar.button("📥 Fetch from Yahoo", key="yf_fetch_btn", use_container_width=True):
                with st.sidebar.status("Downloading from Yahoo Finance...") as _s:
                    try:
                        _df_yf = YahooSource().fetch_bars(
                            _yf_sym, _yf_interval,
                            datetime.combine(_yf_from, datetime.min.time()),
                            datetime.combine(_yf_to,   datetime.max.time()),
                        )
                        st.session_state["loaded_df"]       = _df_yf
                        st.session_state["mt5_defaults"]    = {}
                        st.session_state["mt5_schedule"]    = {}
                        st.session_state["data_source_tag"] = (
                            f"Yahoo · {_yf_sym} {_yf_interval} · "
                            f"{_df_yf.index[0].date()} → {_df_yf.index[-1].date()} · "
                            f"{len(_df_yf):,} bars"
                        )
                        _s.update(label="✅ Yahoo data loaded!", state="complete")
                    except Exception as _e:
                        _s.update(label=f"❌ {_e}", state="error")

            if "data_source_tag" in st.session_state and "Yahoo" in st.session_state.get("data_source_tag", ""):
                st.sidebar.success(st.session_state["data_source_tag"])

            data_source_cfg = {"type": "yahoo_loaded"}

    # ── Strategy ──────────────────────────────────────────────────────────────
    st.sidebar.subheader("🧠 Strategy")
    strategy_name = st.sidebar.selectbox("Strategy", list(REGISTRY.keys()))
    strategy_cls  = REGISTRY[strategy_name]

    # Auto-generate param controls from PARAMS definition
    params = {}
    with st.sidebar.expander("Parameters", expanded=True):
        for key, spec in strategy_cls.PARAMS.items():
            if isinstance(spec, dict):
                label   = spec.get("label", key)
                default = spec.get("default")
                if "options" in spec:
                    params[key] = st.selectbox(label, spec["options"],
                                               index=spec["options"].index(default)
                                               if default in spec["options"] else 0,
                                               key=f"param_{key}")
                elif isinstance(default, float):
                    params[key] = st.slider(label, float(spec.get("min", 0)),
                                            float(spec.get("max", 10)),
                                            float(default), float(spec.get("step", 0.1)),
                                            key=f"param_{key}")
                elif isinstance(default, int):
                    params[key] = st.slider(label, int(spec.get("min", 1)),
                                            int(spec.get("max", 100)),
                                            int(default), int(spec.get("step", 1)),
                                            key=f"param_{key}")
                else:
                    params[key] = st.text_input(label, str(default), key=f"param_{key}")
            else:
                params[key] = spec  # plain default

    # ── Broker ────────────────────────────────────────────────────────────────
    # Pull any auto-filled defaults from a prior MT5 fetch (empty dict otherwise)
    _mt5_def = st.session_state.get("mt5_defaults", {})

    st.sidebar.subheader("💼 Broker / Risk")
    starting_equity = st.sidebar.number_input("Starting Equity ($)", 1000, 1_000_000, 5000, 500)
    risk_usd        = st.sidebar.number_input("Risk per Trade ($)", 1, 10_000, 20, 1)
    max_concurrent  = st.sidebar.slider("Max Concurrent Positions", 1, 10, 2)
    commission_flat = st.sidebar.number_input(
        "Commission ($ per lot)", 0.0, 50.0,
        float(_mt5_def.get("commission_flat", 6.0)), 0.5,
    )
    lot_size = st.sidebar.number_input(
        "Lot Size (units)", 1.0, 10_000.0,
        float(_mt5_def.get("lot_size", 100.0)), 1.0,
    )
    warmup_bars     = st.sidebar.slider("Warmup Bars", 50, 500, 200, 25)

    with st.sidebar.expander("🔬 Realistic Fill Settings", expanded=False):
        _mt5_spread_default = float(_mt5_def.get("spread", 0.0))
        spread = st.number_input(
            "Spread (price units)", 0.0, 10.0, _mt5_spread_default, 0.05,
            help="Bid/ask spread. XAUUSD typical: 0.20–0.50. "
                 "Auto-filled from MT5 if connected. "
                 "LONG entries fill at ask (price+spread), SHORT exits at ask. "
                 "Net cost = one spread per round trip.",
            key="spread",
        )
        slippage_fixed = st.number_input(
            "Slippage fixed (price units)", 0.0, 5.0, 0.0, 0.05,
            help="Extra adverse slippage on market order entry, in price units.",
            key="slip_fixed",
        )
        slippage_pct   = st.number_input(
            "Slippage % (fraction)", 0.0, 0.01, 0.0, 0.0001,
            format="%.4f",
            help="Slippage as fraction of price (e.g. 0.0002 = 0.02%).",
            key="slip_pct",
        )
        slippage_atr_mult = st.number_input(
            "Slippage ATR multiplier", 0.0, 2.0, 0.0, 0.05,
            help="Scales slippage by ATR. During high volatility slippage grows "
                 "automatically. Total slip = fixed + pct×price + atr_mult×ATR(14).",
            key="slip_atr",
        )
        trail_pct      = st.number_input(
            "Auto-trail distance %", 0.0, 5.0, 0.0, 0.1,
            help="0 = disabled. E.g. 0.5 = trail SL at 0.5% below bar high for LONG.",
            key="trail_pct",
        )
        trail_actv_pct = st.number_input(
            "Trail activation % profit", 0.0, 5.0, 0.0, 0.1,
            help="Auto-trail activates once trade profit reaches this % of entry price.",
            key="trail_actv",
        )
        intrabar_model = st.checkbox(
            "Intrabar path model", value=False,
            help="Use OHLC bar direction to determine whether SL or TP hit first "
                 "when both are within the bar's range. Bullish bar → low came first "
                 "(SL more likely for LONG). Bearish bar → high came first (TP more "
                 "likely for LONG). More realistic than always assuming SL wins.",
            key="intrabar_model",
        )
        gap_fill = st.checkbox(
            "Gap fill protection", value=True,
            help="When a bar opens through your SL or TP (weekend gap, news spike), "
                 "fill at the bar open price — not the original SL/TP level. "
                 "This is what happens in live trading.",
            key="gap_fill",
        )
        st.markdown("**Time-of-day spread schedule (UTC hours)**")
        # Pre-populate text area from MT5 auto-derived schedule if available
        _mt5_sched = st.session_state.get("mt5_schedule", {})
        _sched_default = "\n".join(
            f"{h}:{m}" for h, m in sorted(_mt5_sched.items())
        ) if _mt5_sched else ""
        if _mt5_sched:
            st.caption(
                f"✅ Auto-filled from MT5 data ({len(_mt5_sched)} hours). "
                "You can edit the values below."
            )
        else:
            st.caption(
                "Multiplies base spread by hour. Leave blank for constant spread. "
                "Format: `hour:multiplier` per line. Example:\n"
                "`0:2.0` (midnight–07:59 = 2×)\n"
                "`8:1.0` (08:00–16:59 = normal)\n"
                "`17:1.5` (17:00–21:59 = 1.5×)"
            )
        sched_raw = st.text_area(
            "Spread schedule", value=_sched_default, height=80, key="spread_sched",
            label_visibility="collapsed",
        )
        spread_schedule = {}
        for line in sched_raw.strip().splitlines():
            try:
                h, m = line.strip().split(":")
                spread_schedule[int(h.strip())] = float(m.strip())
            except Exception:
                pass

    broker_cfg = BrokerConfig(
        commission_flat       = commission_flat,
        lot_size              = lot_size,
        max_concurrent        = max_concurrent,
        risk_usd              = risk_usd,
        size_mode             = "fixed_risk",
        spread                = float(spread),
        slippage_fixed        = float(slippage_fixed),
        slippage_pct          = float(slippage_pct),
        slippage_atr_mult     = float(slippage_atr_mult),
        trail_pct             = float(trail_pct) / 100.0,
        trail_activation_pct  = float(trail_actv_pct) / 100.0,
        intrabar_path_model   = bool(intrabar_model),
        gap_fill              = bool(gap_fill),
        spread_schedule       = spread_schedule,
    )

    # ── Date Splits ───────────────────────────────────────────────────────────
    # Auto-detect sensible defaults from the last loaded DataFrame so splits
    # always match the user's actual data range rather than 2020 hardcodes.
    # Uses 80 / 20 IS / OOS split of whatever data is loaded.
    _prev_df = st.session_state.get("df")
    if _prev_df is not None:
        _d0  = _prev_df.index[0].date()
        _d1  = _prev_df.index[-1].date()
        _gap = (_d1 - _d0).days
        # 80 % IS, last 20 % OOS, anything after data end = FWD
        _oos_start_def = _d0 + timedelta(days=int(_gap * 0.8))
        _is_start_def  = _d0
        _is_end_def    = _oos_start_def - timedelta(days=1)
        _oos_end_def   = _d1
        _fwd_start_def = _d1 + timedelta(days=1)
        # When data changes, reset the date pickers to the new range
        _sig = f"{_d0}_{_d1}"
        if st.session_state.get("_split_data_sig") != _sig:
            st.session_state["_split_data_sig"] = _sig
            for _k in ("_is_start", "_is_end", "_oos_start", "_oos_end", "_fwd_start"):
                if _k in st.session_state:
                    del st.session_state[_k]
    else:
        _today = date.today()
        _is_start_def  = date(_today.year - 4, 1, 1)
        _is_end_def    = date(_today.year - 1, 12, 31)
        _oos_start_def = date(_today.year, 1, 1)
        _oos_end_def   = date(_today.year, 12, 31)
        _fwd_start_def = date(_today.year + 1, 1, 1)

    st.sidebar.subheader("📅 Date Splits")
    with st.sidebar.expander("Configure splits", expanded=True):
        is_start  = st.date_input("IS start",  value=_is_start_def,  key="_is_start")
        is_end    = st.date_input("IS end",    value=_is_end_def,    key="_is_end")
        oos_start = st.date_input("OOS start", value=_oos_start_def, key="_oos_start")
        oos_end   = st.date_input("OOS end",   value=_oos_end_def,   key="_oos_end")
        fwd_start = st.date_input("FWD start", value=_fwd_start_def, key="_fwd_start")
        st.caption("Dates auto-adjust to your data range on each new load.")

    splits = {
        f"IS  ({is_start.year}–{is_end.year})":  (str(is_start), str(is_end)),
        f"OOS ({oos_start.year})":                (str(oos_start), str(oos_end)),
        f"FWD ({fwd_start.year}–now)":            (str(fwd_start), None),
    }

    # ── Run button ────────────────────────────────────────────────────────────
    st.sidebar.divider()
    run = st.sidebar.button("▶ Run Backtest", type="primary", use_container_width=True)

    return {
        "data_source_cfg": data_source_cfg,
        "strategy_cls":    strategy_cls,
        "strategy_name":   strategy_name,
        "params":          params,
        "broker_cfg":      broker_cfg,
        "starting_equity": starting_equity,
        "warmup_bars":     warmup_bars,
        "splits":          splits,
        "run":             run,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading CSV...")
def _load_csv_path(path: str) -> pd.DataFrame:
    return CSVSource().fetch_bars(path)


@st.cache_data(show_spinner="Loading CSV...")
def _load_csv_bytes(data: bytes) -> pd.DataFrame:
    return CSVSource().fetch_bars(data)


def load_data(cfg: dict) -> tuple[pd.DataFrame | None, str]:
    """
    Dispatch to the right data loader based on cfg["data_source_cfg"].

    Returns (df, status_message) where df is None if no data is ready yet.
    """
    src_cfg = cfg.get("data_source_cfg", {})
    src_type = src_cfg.get("type", "")

    if src_type == "csv_path":
        path = src_cfg.get("path", "").strip()
        if not path:
            return None, "csv_no_path"
        if not Path(path).exists():
            return None, f"csv_not_found:{path}"
        try:
            return _load_csv_path(path), "ok"
        except Exception as e:
            return None, f"error:{e}"

    elif src_type == "csv_upload":
        uploaded = src_cfg.get("uploaded")
        if uploaded is None:
            return None, "csv_no_upload"
        try:
            return _load_csv_bytes(uploaded.read()), "ok"
        except Exception as e:
            return None, f"error:{e}"

    elif src_type in ("mt5_loaded", "yahoo_loaded", "ctrader_loaded"):
        df = st.session_state.get("loaded_df")
        if df is None:
            return None, "fetch_first"
        return df, "ok"

    elif src_type in ("mt5_unavailable", "yahoo_unavailable", "ctrader_unavailable"):
        return None, "source_unavailable"

    return None, "no_source"


# ── Tabs ──────────────────────────────────────────────────────────────────────

def tab_overview(results: dict[str, BacktestResult]):
    """Metrics table + headline numbers."""
    st.subheader("📊 Performance Overview")

    # Headline metrics for each split
    cols = st.columns(len(results))
    for col, (label, res) in zip(cols, results.items()):
        m = compute_metrics(res)
        with col:
            st.markdown(f"**{label}**")
            st.metric("Total PnL", f"${m.get('total_pnl_$', 0):+,.2f}")
            st.metric("Win Rate",  f"{m.get('win_rate_%', 0):.1f}%")
            st.metric("Sharpe",    f"{m.get('sharpe_ann', 0):.2f}")
            st.metric("Max DD",    f"{m.get('max_dd_%', 0):.1f}%")
            st.metric("Trades",    m.get("trades", 0))

    st.divider()

    # Full metrics comparison table
    rows = []
    for label, res in results.items():
        m = compute_metrics(res)
        rows.append(m)
    if rows:
        df_m = pd.DataFrame(rows).set_index("label").drop(columns=["strategy"], errors="ignore")
        st.dataframe(df_m.T, use_container_width=True)


def tab_equity(results: dict[str, BacktestResult]):
    """Equity curves for all splits."""
    st.subheader("📈 Equity Curves")

    fig = go.Figure()
    for label, res in results.items():
        split_key = label.split()[0]
        color = COLORS.get(split_key, "#888")
        ts    = res.timestamps
        eq    = res.equity_curve
        valid = ~np.isnan(eq)
        fig.add_trace(go.Scatter(
            x=ts[valid], y=eq[valid],
            name=label, line=dict(color=color, width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Equity ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", height=450,
        template="plotly_dark" if _is_dark() else "plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    st.subheader("📉 Drawdown")
    fig2 = go.Figure()
    for label, res in results.items():
        split_key = label.split()[0]
        color = COLORS.get(split_key, "#888")
        ts  = res.timestamps
        eq  = res.equity_curve
        valid = ~np.isnan(eq)
        eq_v  = eq[valid]
        ts_v  = ts[valid]
        roll  = np.maximum.accumulate(eq_v)
        dd    = (eq_v - roll) / roll * 100
        fig2.add_trace(go.Scatter(
            x=ts_v, y=dd, name=label, fill="tozeroy",
            line=dict(color=color, width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        ))
    fig2.update_layout(
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=250, template="plotly_dark" if _is_dark() else "plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)


def tab_trades(results: dict[str, BacktestResult]):
    """Trade browser with filters."""
    st.subheader("📋 Trade Browser")

    split_sel = st.selectbox("Select split", list(results.keys()), key="trade_split")
    res = results[split_sel]
    df_t = trade_log_df(res)

    if df_t.empty:
        st.info("No closed trades in this split.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dir_filter = st.multiselect("Direction", ["LONG", "SHORT"],
                                    default=["LONG", "SHORT"])
    with col2:
        reason_filter = st.multiselect("Exit reason",
                                       df_t["exit_reason"].unique().tolist(),
                                       default=df_t["exit_reason"].unique().tolist())
    with col3:
        pnl_filter = st.slider("PnL range ($)",
                               float(df_t["pnl_net_$"].min()),
                               float(df_t["pnl_net_$"].max()),
                               (float(df_t["pnl_net_$"].min()), float(df_t["pnl_net_$"].max())))

    mask = (
        df_t["direction"].isin(dir_filter) &
        df_t["exit_reason"].isin(reason_filter) &
        df_t["pnl_net_$"].between(*pnl_filter)
    )
    df_filtered = df_t[mask]

    # Colour rows by PnL
    def _color(val):
        if isinstance(val, (int, float)):
            return "color: #2A9D8F" if val > 0 else "color: #E76F51"
        return ""

    st.dataframe(
        df_filtered.style.map(_color, subset=["pnl_net_$"]),
        use_container_width=True, height=400,
    )

    # PnL distribution
    fig = px.histogram(df_filtered, x="pnl_net_$", nbins=50,
                       color_discrete_sequence=["#4C9BE8"],
                       title="PnL Distribution")
    fig.update_layout(height=300,
                      template="plotly_dark" if _is_dark() else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Export
    csv = df_filtered.to_csv(index=False)
    st.download_button("⬇ Download CSV", csv,
                       f"trades_{split_sel.strip()}.csv", "text/csv")


def tab_breakdown(results: dict[str, BacktestResult]):
    """Per-session and per-direction breakdown."""
    st.subheader("🔍 Breakdown Analysis")

    split_sel = st.selectbox("Select split", list(results.keys()), key="bd_split")
    res = results[split_sel]
    df_t = trade_log_df(res)

    if df_t.empty:
        st.info("No closed trades to analyse.")
        return

    col1, col2 = st.columns(2)

    # Direction breakdown
    with col1:
        st.markdown("**By Direction**")
        dir_stats = df_t.groupby("direction").agg(
            trades=("pnl_net_$", "count"),
            win_rate=("pnl_net_$", lambda x: (x > 0).mean() * 100),
            total_pnl=("pnl_net_$", "sum"),
            avg_pnl=("pnl_net_$", "mean"),
        ).round(2)
        st.dataframe(dir_stats, use_container_width=True)

        fig = px.bar(dir_stats, y="total_pnl", color="total_pnl",
                     color_continuous_scale=["#E76F51","#2A9D8F"],
                     title="Total PnL by Direction")
        fig.update_layout(height=250, showlegend=False,
                          template="plotly_dark" if _is_dark() else "plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Exit reason breakdown
    with col2:
        st.markdown("**By Exit Reason**")
        ex_stats = df_t.groupby("exit_reason").agg(
            trades=("pnl_net_$", "count"),
            total_pnl=("pnl_net_$", "sum"),
            avg_pnl=("pnl_net_$", "mean"),
        ).round(2)
        st.dataframe(ex_stats, use_container_width=True)

        fig2 = px.pie(ex_stats.reset_index(), values="trades",
                      names="exit_reason", title="Trade Count by Exit",
                      color_discrete_sequence=px.colors.qualitative.Safe)
        fig2.update_layout(height=250,
                           template="plotly_dark" if _is_dark() else "plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly PnL heatmap
    st.markdown("**Monthly PnL**")
    df_t2 = df_t.copy()
    df_t2["exit_time"] = pd.to_datetime(df_t2["exit_time"], utc=True, errors="coerce")
    df_t2 = df_t2.dropna(subset=["exit_time"])
    df_t2["year"]  = df_t2["exit_time"].dt.year
    df_t2["month"] = df_t2["exit_time"].dt.strftime("%b")
    monthly = df_t2.groupby(["year","month"])["pnl_net_$"].sum().unstack(fill_value=0)
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = monthly.reindex(columns=[m for m in month_order if m in monthly.columns])
    fig3 = px.imshow(monthly, color_continuous_scale="RdYlGn",
                     aspect="auto", title="Monthly PnL ($)")
    fig3.update_layout(height=200 + 40 * len(monthly),
                       template="plotly_dark" if _is_dark() else "plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # Cumulative PnL by direction
    st.markdown("**Cumulative PnL by Direction**")
    df_t["fill_time_dt"] = pd.to_datetime(df_t["fill_time"], utc=True, errors="coerce")
    df_t = df_t.dropna(subset=["fill_time_dt"]).sort_values("fill_time_dt")
    for direction, grp in df_t.groupby("direction"):
        df_t.loc[grp.index, "cumulative_pnl"] = grp["pnl_net_$"].cumsum()
    fig4 = px.line(df_t, x="fill_time_dt", y="cumulative_pnl",
                   color="direction",
                   color_discrete_map={"LONG": COLORS["win"], "SHORT": COLORS["loss"]},
                   title="Cumulative PnL — Long vs Short")
    fig4.update_layout(height=300,
                       template="plotly_dark" if _is_dark() else "plotly_white")
    st.plotly_chart(fig4, use_container_width=True)


def tab_compare(cfg: dict, df: pd.DataFrame):
    """Run two param sets and compare side by side."""
    st.subheader("⚡ Compare Two Param Sets")
    st.caption("Tweak params for Set B — Set A uses the current sidebar settings.")

    strategy_cls = cfg["strategy_cls"]

    st.markdown("**Set B — Override Parameters**")
    params_b = {}
    cols = st.columns(3)
    col_idx = 0
    for key, spec in strategy_cls.PARAMS.items():
        with cols[col_idx % 3]:
            if isinstance(spec, dict):
                label   = spec.get("label", key)
                default = cfg["params"].get(key, spec.get("default"))
                if "options" in spec:
                    params_b[key] = st.selectbox(
                        f"B: {label}", spec["options"],
                        index=spec["options"].index(default) if default in spec["options"] else 0,
                        key=f"cmp_{key}")
                elif isinstance(default, float):
                    params_b[key] = st.slider(
                        f"B: {label}", float(spec.get("min",0)),
                        float(spec.get("max",10)), float(default),
                        float(spec.get("step",0.1)), key=f"cmp_{key}")
                elif isinstance(default, int):
                    params_b[key] = st.slider(
                        f"B: {label}", int(spec.get("min",1)),
                        int(spec.get("max",100)), int(default),
                        int(spec.get("step",1)), key=f"cmp_{key}")
                else:
                    params_b[key] = default
            else:
                params_b[key] = spec
        col_idx += 1

    if st.button("▶ Run Comparison", type="primary"):
        with st.spinner("Running Set A ..."):
            res_a   = run_splits(df, strategy_cls, cfg["params"],
                                 cfg["broker_cfg"], cfg["starting_equity"],
                                 cfg["warmup_bars"], cfg["splits"],
                                 verbose=False)
        with st.spinner("Running Set B ..."):
            res_b   = run_splits(df, strategy_cls, params_b,
                                 cfg["broker_cfg"], cfg["starting_equity"],
                                 cfg["warmup_bars"], cfg["splits"],
                                 verbose=False)

        # Metrics table A vs B
        rows_a = [compute_metrics(r) for r in res_a.values()]
        rows_b = [compute_metrics(r) for r in res_b.values()]

        if rows_a and rows_b:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Set A (sidebar params)**")
                df_a = pd.DataFrame(rows_a).set_index("label")
                st.dataframe(df_a.T, use_container_width=True)
            with col2:
                st.markdown("**Set B (compare params)**")
                df_b = pd.DataFrame(rows_b).set_index("label")
                st.dataframe(df_b.T, use_container_width=True)

        # Equity comparison chart (first split only)
        if res_a and res_b:
            label_a = list(res_a.keys())[0]
            r_a = res_a[label_a]
            r_b = list(res_b.values())[0]
            fig = go.Figure()
            ts_a = r_a.timestamps
            eq_a = r_a.equity_curve
            ts_b = r_b.timestamps
            eq_b = r_b.equity_curve
            valid_a = ~np.isnan(eq_a)
            valid_b = ~np.isnan(eq_b)
            fig.add_trace(go.Scatter(x=ts_a[valid_a], y=eq_a[valid_a],
                                     name="Set A", line=dict(color=COLORS["A"])))
            fig.add_trace(go.Scatter(x=ts_b[valid_b], y=eq_b[valid_b],
                                     name="Set B", line=dict(color=COLORS["B"])))
            fig.update_layout(title=f"Equity Comparison — {label_a}",
                              height=350,
                              template="plotly_dark" if _is_dark() else "plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ── Price chart tab (MT5-style trade overlay) ────────────────────────────────

def tab_chart(results: dict[str, BacktestResult], df_full: pd.DataFrame):
    """
    MT5-style candlestick chart with trade overlays.

    Shows:
      ▲  Blue   up-triangle   = LONG entry
      ▼  Red    down-triangle = SHORT entry
      ●  Green  circle        = winning exit
      ●  Red    circle        = losing exit
      ── Thin coloured line   = trade duration (green win, red loss)
      -- Dashed red line      = SL level from entry to exit
      -- Dashed green line    = TP level from entry to exit
    """
    st.subheader("📉 Price Chart")
    st.caption(
        "Candlestick chart with all trades overlaid — similar to MT5/MT4 trade history. "
        "Use the range slider below the chart to zoom in on specific periods."
    )

    if not results:
        st.info("Run a backtest first to see trades on the chart.")
        return

    # Split selector
    split_sel = st.selectbox("Select split", list(results.keys()), key="chart_split")
    res = results[split_sel]

    # Filter df_full to the split's time range
    ts_start = res.timestamps[0]
    ts_end   = res.timestamps[-1]
    df_split = df_full.loc[
        (df_full.index >= ts_start) & (df_full.index <= ts_end)
    ].copy()

    if df_split.empty:
        st.warning("No price data found for the selected split range.")
        return

    # ── Bar range control ─────────────────────────────────────────────────────
    n_bars = len(df_split)
    st.caption(f"Split contains **{n_bars:,}** bars ({ts_start.date()} → {ts_end.date()})")

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        max_candles = st.slider(
            "Candles to display", 100, min(5_000, n_bars),
            value=min(2_000, n_bars), step=100,
            help="Fewer candles = faster render. Use the Plotly range slider below to zoom.",
        )
    with col_r2:
        view_end = st.selectbox(
            "View window",
            ["All trades", "Last N bars", "First N bars"],
            index=0,   # default: All trades — so trades are always visible on first open
        )

    # Select which bars to show
    trades_df = trade_log_df(res)
    if view_end == "Last N bars":
        df_view = df_split.iloc[-max_candles:]
    elif view_end == "First N bars":
        df_view = df_split.iloc[:max_candles]
    else:  # "All trades" — span from first fill to last exit, capped for render perf
        if not trades_df.empty:
            t_start = pd.to_datetime(trades_df["fill_time"].min(), utc=True)
            t_end   = pd.to_datetime(trades_df["exit_time"].max(),  utc=True)
            # Small padding so entry/exit markers aren't right at the edge
            pad     = pd.Timedelta(minutes=max(10, max_candles // 10))
            df_view = df_split.loc[
                (df_split.index >= t_start - pad) &
                (df_split.index <= t_end   + pad)
            ]
            # When trade history spans more bars than max_candles, start from
            # the FIRST trade so the user immediately sees trades on open
            if len(df_view) > max_candles:
                df_view = df_view.iloc[:max_candles]
        else:
            df_view = df_split.iloc[:max_candles]

    template = "plotly_dark" if _is_dark() else "plotly_white"

    # ── Build candlestick figure ───────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x    = df_view.index,
        open = df_view["open"],
        high = df_view["high"],
        low  = df_view["low"],
        close= df_view["close"],
        name = "Price",
        increasing_line_color = "#26a69a",
        decreasing_line_color = "#ef5350",
        showlegend = False,
    ))

    # ── Overlay trades ────────────────────────────────────────────────────────
    if trades_df.empty:
        st.info("No closed trades in this split.")
    else:
        view_start_ts = df_view.index[0]
        view_end_ts   = df_view.index[-1]

        trades_df["fill_dt"] = pd.to_datetime(trades_df["fill_time"], utc=True,
                                               errors="coerce")
        trades_df["exit_dt"] = pd.to_datetime(trades_df["exit_time"], utc=True,
                                               errors="coerce")

        # In "All trades" mode show every trade whose entry OR exit is in the window;
        # in "Last/First N bars" mode filter strictly to the visible candle range.
        if view_end == "All trades":
            visible = trades_df[
                (trades_df["fill_dt"] <= view_end_ts) &
                (trades_df["exit_dt"]  >= view_start_ts)
            ].copy()
        else:
            visible = trades_df[
                (trades_df["fill_dt"] >= view_start_ts) &
                (trades_df["fill_dt"] <= view_end_ts)
            ].copy()

        # ── Trade display filter (multiselect) ────────────────────────────────
        _all_filters = ["Wins ✅", "Losses ❌", "Longs ▲", "Shorts ▼"]
        trade_filters = st.multiselect(
            "Show trades",
            _all_filters,
            default=_all_filters,
            help="Filter which trade types are drawn on the chart. "
                 "Click the legend entries (Win / Loss) to toggle visibility without re-rendering.",
        )
        # Apply direction / outcome filters to the visible slice
        if "Wins ✅" not in trade_filters:
            visible = visible[visible["pnl_net_$"] <= 0]
        if "Losses ❌" not in trade_filters:
            visible = visible[visible["pnl_net_$"] > 0]
        if "Longs ▲" not in trade_filters:
            visible = visible[visible["direction"] != "LONG"]
        if "Shorts ▼" not in trade_filters:
            visible = visible[visible["direction"] != "SHORT"]

        n_visible = len(visible)
        st.caption(
            f"Showing **{n_visible}** of {len(trades_df)} trades in this window. "
            "Use the range slider below to scroll through the full split."
        )

        for _, t in visible.iterrows():
            is_long = t["direction"] == "LONG"
            is_win  = t["pnl_net_$"] > 0
            fill_dt = t["fill_dt"]
            exit_dt = t["exit_dt"]
            fp      = t["fill_price"]
            ep      = t["exit_price"]
            sl_lvl  = t["sl"]
            tp_lvl  = t["tp"]
            reason  = t["exit_reason"]

            trade_color = "#26a69a" if is_win else "#ef5350"
            entry_color = "#1565C0" if is_long else "#E65100"   # blue LONG, orange SHORT
            # legendgroup links ALL 5 traces for this trade so a single legend
            # click (Win / Loss) toggles the entire trade at once — Plotly
            # only hides traces that belong to the clicked group.
            lg = "win" if is_win else "loss"

            # ── Entry to exit connecting line ──────────────────────────────
            fig.add_trace(go.Scatter(
                x=[fill_dt, exit_dt], y=[fp, ep],
                mode="lines",
                line=dict(color=trade_color, width=1.5, dash="solid"),
                legendgroup=lg,
                showlegend=False,
                hoverinfo="skip",
            ))

            # ── SL level (dashed red, entry → exit) ───────────────────────
            fig.add_trace(go.Scatter(
                x=[fill_dt, exit_dt], y=[sl_lvl, sl_lvl],
                mode="lines",
                line=dict(color="#ef5350", width=1, dash="dot"),
                legendgroup=lg,
                showlegend=False,
                hoverinfo="skip",
            ))

            # ── TP level (dashed green, entry → exit) ─────────────────────
            fig.add_trace(go.Scatter(
                x=[fill_dt, exit_dt], y=[tp_lvl, tp_lvl],
                mode="lines",
                line=dict(color="#26a69a", width=1, dash="dot"),
                legendgroup=lg,
                showlegend=False,
                hoverinfo="skip",
            ))

            # ── Entry marker ───────────────────────────────────────────────
            entry_symbol = "triangle-up" if is_long else "triangle-down"
            entry_offset = -abs(fp - sl_lvl) * 0.15   # nudge below/above candle
            fig.add_trace(go.Scatter(
                x=[fill_dt],
                y=[fp + (entry_offset if is_long else -entry_offset)],
                mode="markers",
                marker=dict(
                    symbol=entry_symbol, size=12,
                    color=entry_color,
                    line=dict(color="white", width=1),
                ),
                name=f"{'LONG' if is_long else 'SHORT'} entry",
                legendgroup=lg,
                showlegend=False,
                hovertemplate=(
                    f"<b>{'LONG' if is_long else 'SHORT'} ENTRY</b><br>"
                    f"Time: {fill_dt}<br>"
                    f"Fill: {fp:.4f}<br>"
                    f"SL: {sl_lvl:.4f}<br>"
                    f"TP: {tp_lvl:.4f}"
                    "<extra></extra>"
                ),
            ))

            # ── Exit marker ────────────────────────────────────────────────
            exit_symbol = "circle" if is_win else "x"
            fig.add_trace(go.Scatter(
                x=[exit_dt],
                y=[ep],
                mode="markers",
                marker=dict(
                    symbol=exit_symbol, size=10,
                    color=trade_color,
                    line=dict(color="white", width=1),
                ),
                legendgroup=lg,
                showlegend=False,
                hovertemplate=(
                    f"<b>EXIT ({reason.upper()})</b><br>"
                    f"Time: {exit_dt}<br>"
                    f"Price: {ep:.4f}<br>"
                    f"PnL: ${t['pnl_net_$']:+.2f}"
                    "<extra></extra>"
                ),
            ))

    # ── Legend entries — linked to legendgroup so clicking them toggles all
    # traces that belong to the same group (every line + marker for that trade).
    # Single-clicking hides the group; double-clicking isolates it.
    for sym, clr, name, lg in [
        ("circle", "#26a69a", "✅ Wins",   "win"),
        ("x",      "#ef5350", "❌ Losses", "loss"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=sym, size=10, color=clr),
            name=name,
            legendgroup=lg,
            showlegend=True,
        ))

    fig.update_layout(
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.06,
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        template=template,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Quick trade stats for visible window ──────────────────────────────────
    if not trades_df.empty and n_visible > 0:
        wins_v   = (visible["pnl_net_$"] > 0).sum()
        total_pnl = visible["pnl_net_$"].sum()
        st.caption(
            f"Visible window: **{n_visible}** trades | "
            f"**{wins_v}W / {n_visible - wins_v}L** | "
            f"Net PnL: **${total_pnl:+,.2f}**"
        )

    # ── Trade table for visible window ────────────────────────────────────────
    with st.expander("Trade details (visible window)", expanded=False):
        if not trades_df.empty and n_visible > 0:
            show_cols = ["direction", "fill_time", "fill_price",
                         "sl", "tp", "exit_time", "exit_price",
                         "exit_reason", "pnl_net_$"]
            available = [c for c in show_cols if c in visible.columns]
            def _color_pnl(val):
                if isinstance(val, (int, float)):
                    return "color: #26a69a" if val > 0 else "color: #ef5350"
                return ""
            st.dataframe(
                visible[available].style.map(_color_pnl, subset=["pnl_net_$"]),
                use_container_width=True,
            )
        else:
            st.info("No trades in visible window.")


# ── Walk-Forward tab ──────────────────────────────────────────────────────────

def _build_default_grid(strategy_cls, current_params: dict) -> dict:
    """
    Auto-build a small param_grid from a strategy's PARAMS spec.

    For each int/float param: generates 3 values around the current setting.
    For option params: uses all declared options.
    Keeps grids small (≤ a few hundred combos) so WFA runs in reasonable time.
    """
    grid = {}
    for key, spec in strategy_cls.PARAMS.items():
        if not isinstance(spec, dict):
            grid[key] = [spec]
            continue
        default = current_params.get(key, spec.get("default"))
        if "options" in spec:
            grid[key] = spec["options"]
        elif isinstance(default, int):
            step = int(spec.get("step", 1))
            lo   = int(spec.get("min", max(1, default - step * 2)))
            hi   = int(spec.get("max", default + step * 2))
            vals = sorted({max(lo, default - step * 2), default,
                           min(hi, default + step * 2)})
            grid[key] = vals
        elif isinstance(default, float):
            step = float(spec.get("step", 0.1))
            lo   = float(spec.get("min", default - step * 2))
            hi   = float(spec.get("max", default + step * 2))
            vals = sorted({round(max(lo, default - step * 2), 6),
                           round(default, 6),
                           round(min(hi, default + step * 2), 6)})
            grid[key] = vals
        else:
            grid[key] = [default]
    return grid


def tab_walkforward(cfg: dict, df: pd.DataFrame):
    """Walk-Forward Optimisation tab — rolling IS/OOS window analysis."""
    import json
    st.subheader("🔄 Walk-Forward Optimisation")
    st.caption(
        "Repeatedly optimise on an In-Sample window, then validate on the "
        "Out-of-Sample window immediately following it.  "
        "A high **score ratio** (OOS/IS) signals a robust, non-overfit strategy."
    )

    strategy_cls = cfg["strategy_cls"]

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        window_bars = st.number_input(
            "IS window (bars)", min_value=500, max_value=500_000,
            value=10_000, step=1_000,
            help="Number of bars in each In-Sample optimisation window.",
        )
        step_bars = st.number_input(
            "Step / OOS window (bars)", min_value=100, max_value=100_000,
            value=2_000, step=500,
            help="Size of each OOS window (also the roll-forward step).",
        )
        anchored = st.checkbox(
            "Anchored mode", value=False,
            help="IS window grows from a fixed start instead of rolling.",
        )
    with col_cfg2:
        score_fn = st.selectbox(
            "Score function",
            ["sharpe", "pf", "calmar", "pnl", "sortino", "win_rate"],
            index=0,
            help="Metric used to rank parameter combos on IS data.",
        )
        max_combos = st.number_input(
            "Max param combos", min_value=1, max_value=2_000,
            value=50, step=10,
            help="Cap on the grid search to keep runtime reasonable.",
        )

    # Param grid editor
    st.markdown("**Parameter Grid** (edit JSON below)")
    default_grid = _build_default_grid(strategy_cls, cfg["params"])
    grid_str = st.text_area(
        "param_grid (JSON)",
        value=json.dumps(default_grid, indent=2),
        height=160,
        help="Dict mapping param name → list of values to try.",
    )
    try:
        param_grid = json.loads(grid_str)
        st.caption(f"✅ Valid grid — up to "
                   f"{int(np.prod([len(v) for v in param_grid.values()])):,} combos "
                   f"(capped at {max_combos})")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return

    # Performance note — walk-forward is CPU-bound and runs synchronously in
    # Streamlit; the browser tab will appear frozen until all windows complete.
    total_combos = int(np.prod([len(v) for v in param_grid.values()]))
    n_windows_est = max(1, (len(df) - window_bars) // step_bars)
    total_runs = min(total_combos, max_combos) * n_windows_est
    if total_runs > 500:
        st.warning(
            f"⚠️ **Performance notice** — this grid will run up to **{total_runs:,} backtests** "
            f"({min(total_combos, max_combos):,} combos × {n_windows_est} windows). "
            "The browser tab will be unresponsive until it finishes. "
            "Reduce the grid size or bar counts to keep it under ~500 total runs for a snappy experience."
        )

    if not st.button("▶ Run Walk-Forward", type="primary", key="wf_run"):
        st.info("Configure the grid above and click **▶ Run Walk-Forward**.")
        return

    with st.spinner("Running walk-forward optimisation …"):
        try:
            wf = run_walk_forward(
                df           = df,
                strategy_class = strategy_cls,
                param_grid   = param_grid,
                broker_config = cfg["broker_cfg"],
                window_bars  = int(window_bars),
                step_bars    = int(step_bars),
                score_fn     = score_fn,
                starting_equity = cfg["starting_equity"],
                warmup_bars  = cfg["warmup_bars"],
                anchored     = anchored,
                max_combos   = int(max_combos),
                verbose      = False,
            )
        except Exception as e:
            st.error(f"Walk-forward error: {e}")
            st.exception(e)
            return

    st.success(
        f"Done — {len(wf.windows)} windows | "
        f"avg score ratio {wf.avg_score_ratio:.3f}"
    )

    # ── IS vs OOS score chart ──────────────────────────────────────────────────
    st.subheader("IS vs OOS Scores per Window")
    win_data = pd.DataFrame([
        {"Window": i + 1, "IS score": w.is_score, "OOS score": w.oos_score,
         "Score ratio": w.score_ratio}
        for i, w in enumerate(wf.windows)
    ])
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Bar(
        x=win_data["Window"], y=win_data["IS score"],
        name="IS", marker_color=COLORS["IS"], opacity=0.8,
    ))
    fig_scores.add_trace(go.Bar(
        x=win_data["Window"], y=win_data["OOS score"],
        name="OOS", marker_color=COLORS["OOS"], opacity=0.8,
    ))
    fig_scores.update_layout(
        barmode="group", xaxis_title="Window #",
        yaxis_title=score_fn, height=350,
        template="plotly_dark" if _is_dark() else "plotly_white",
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    # ── Combined OOS equity curve ──────────────────────────────────────────────
    if len(wf.combined_oos_equity) > 1:
        st.subheader("Combined OOS Equity")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=wf.combined_oos_equity,
            name="Combined OOS", line=dict(color=COLORS["OOS"], width=2),
            hovertemplate="Trade %{x}<br>$%{y:,.2f}<extra></extra>",
        ))
        fig_eq.update_layout(
            xaxis_title="Trade #", yaxis_title="Equity ($)",
            height=300,
            template="plotly_dark" if _is_dark() else "plotly_white",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Best params per window ─────────────────────────────────────────────────
    st.subheader("Best Parameters per Window")
    params_rows = []
    for i, w in enumerate(wf.windows):
        row = {"Window": i + 1, "IS Score": round(w.is_score, 4),
               "OOS Score": round(w.oos_score, 4),
               "Ratio": round(w.score_ratio, 3)}
        row.update(w.best_params)
        params_rows.append(row)
    st.dataframe(pd.DataFrame(params_rows), use_container_width=True)

    # ── Summary text ──────────────────────────────────────────────────────────
    with st.expander("Walk-Forward Summary", expanded=False):
        st.text(wf.summary())


# ── Monte Carlo tab ───────────────────────────────────────────────────────────

def tab_montecarlo(results: dict[str, BacktestResult], cfg: dict):
    """Monte Carlo Simulation tab — bootstrap equity fan chart and risk stats."""
    st.subheader("🎲 Monte Carlo Simulation")
    st.caption(
        "Bootstraps the **trade PnL sequence** N times to estimate the distribution "
        "of possible outcomes.  The fan chart shows percentile bands of simulated "
        "equity curves.  **Prob Ruin** = fraction of simulations that hit the ruin floor."
    )

    if not results:
        st.info("Run a backtest first (▶ Run Backtest in sidebar).")
        return

    # Pick which split to simulate on
    split_sel = st.selectbox(
        "Simulate from split", list(results.keys()), key="mc_split"
    )
    res = results[split_sel]

    if res.n_trades == 0:
        st.warning("No closed trades in this split — cannot run Monte Carlo.")
        return

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        n_sims = st.number_input(
            "Number of simulations", min_value=100, max_value=10_000,
            value=1_000, step=100,
        )
        seed = st.number_input("Random seed", min_value=0, max_value=99_999,
                                value=42, step=1)
    with col_mc2:
        ruin_pct = st.slider(
            "Ruin floor (% of starting equity)", 10, 90, 50,
            help="A simulation 'ruins' if equity ever falls below this % of starting.",
        )

    if not st.button("▶ Run Monte Carlo", type="primary", key="mc_run"):
        st.info(f"Using {res.n_trades} closed trades from **{split_sel}**. "
                "Click **▶ Run Monte Carlo** to simulate.")
        return

    starting = cfg.get("starting_equity", 5_000.0)
    ruin_floor = starting * (ruin_pct / 100.0)

    with st.spinner(f"Running {n_sims:,} simulations …"):
        try:
            mc = run_monte_carlo(
                result         = res,
                n_sims         = int(n_sims),
                starting_equity = starting,
                ruin_floor     = ruin_floor,
                seed           = int(seed),
                verbose        = False,
            )
        except Exception as e:
            st.error(f"Monte Carlo error: {e}")
            st.exception(e)
            return

    # ── Headline risk stats ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    bands   = mc.equity_bands()
    final   = mc.sim_curves[:, -1]
    med_eq  = float(np.median(final))
    c1.metric("Median Final Equity", f"${med_eq:,.2f}",
              delta=f"{(med_eq/starting - 1)*100:+.1f}%")
    c2.metric("Prob Ruin", f"{mc.prob_ruin*100:.1f}%",
              delta=f"floor ${ruin_floor:,.0f}",
              delta_color="inverse" if mc.prob_ruin > 0.1 else "normal")
    c3.metric("Median Max DD", f"{float(np.median(mc.max_drawdowns))*100:.1f}%")
    c4.metric("Median Sharpe", f"{float(np.median(mc.sharpes)):.2f}")

    # ── Equity fan chart ──────────────────────────────────────────────────────
    st.subheader("Equity Fan Chart")
    pct_labels = [5, 25, 50, 75, 95]
    x_trades = list(range(mc.sim_curves.shape[1]))

    fig_fan = go.Figure()
    # Fill bands: 5–95 (outer), 25–75 (inner)
    fig_fan.add_trace(go.Scatter(
        x=x_trades + x_trades[::-1],
        y=list(bands[95]) + list(bands[5])[::-1],
        fill="toself", fillcolor="rgba(78,160,230,0.12)",
        line=dict(width=0), name="5–95 pct", showlegend=True,
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_trades + x_trades[::-1],
        y=list(bands[75]) + list(bands[25])[::-1],
        fill="toself", fillcolor="rgba(78,160,230,0.25)",
        line=dict(width=0), name="25–75 pct", showlegend=True,
    ))
    # Median line
    fig_fan.add_trace(go.Scatter(
        x=x_trades, y=list(bands[50]),
        line=dict(color=COLORS["OOS"], width=2), name="Median",
    ))
    # Ruin floor reference
    fig_fan.add_hline(
        y=ruin_floor, line_dash="dash", line_color="#E76F51",
        annotation_text=f"Ruin floor ${ruin_floor:,.0f}",
        annotation_position="bottom right",
    )
    fig_fan.update_layout(
        xaxis_title="Trade #", yaxis_title="Equity ($)",
        height=380, hovermode="x unified",
        template="plotly_dark" if _is_dark() else "plotly_white",
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # ── Distribution charts ────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("**Max Drawdown Distribution**")
        fig_dd = px.histogram(
            x=mc.max_drawdowns * 100, nbins=50,
            color_discrete_sequence=["#E76F51"],
            labels={"x": "Max Drawdown (%)"},
        )
        fig_dd.update_layout(
            height=280, showlegend=False,
            template="plotly_dark" if _is_dark() else "plotly_white",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_h2:
        st.markdown("**Annualised Sharpe Distribution**")
        sharpe_clean = mc.sharpes[np.isfinite(mc.sharpes)]
        fig_sh = px.histogram(
            x=sharpe_clean, nbins=50,
            color_discrete_sequence=[COLORS["IS"]],
            labels={"x": "Sharpe Ratio"},
        )
        fig_sh.update_layout(
            height=280, showlegend=False,
            template="plotly_dark" if _is_dark() else "plotly_white",
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    # Percentile table
    st.subheader("Final Equity Percentiles")
    pct_df = pd.DataFrame({
        "Percentile": [f"p{p}" for p in pct_labels],
        "Final Equity ($)": [round(float(bands[p][-1]), 2) for p in pct_labels],
        "Return (%)": [round((float(bands[p][-1]) / starting - 1) * 100, 2)
                       for p in pct_labels],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

    with st.expander("Monte Carlo Summary", expanded=False):
        st.text(mc.summary())


# ── Portfolio tab ─────────────────────────────────────────────────────────────

def tab_portfolio(cfg: dict):
    """
    Portfolio backtesting tab — run multiple strategies on multiple symbols
    with shared capital and a portfolio-level risk cap.
    """
    st.subheader("💼 Portfolio Backtesting")
    st.caption(
        "Run several strategies on different symbols simultaneously with "
        "**shared equity**.  The portfolio-level risk cap prevents any single "
        "symbol from consuming all available capital."
    )

    strategy_cls = cfg["strategy_cls"]
    _sidebar_df_available = st.session_state.get("loaded_df") is not None

    # ── Symbol / data paths ────────────────────────────────────────────────────
    st.markdown("**Add Symbols**")
    st.caption(
        "Each symbol can use a **CSV file** or the **sidebar-fetched dataset** "
        "(MT5 / cTrader / Yahoo).  The strategy from the sidebar is applied to "
        "all symbols by default — you can override per symbol below."
    )
    if _sidebar_df_available:
        st.success(
            "✅ Sidebar data loaded — you can assign it to any symbol below.",
            icon="📊",
        )
    else:
        st.info(
            "No sidebar data loaded.  Fetch data from MT5 / cTrader / Yahoo in "
            "the sidebar, or supply CSV paths directly.",
            icon="ℹ️",
        )

    # Dynamic symbol table
    if "portfolio_symbols" not in st.session_state:
        st.session_state["portfolio_symbols"] = [
            {"symbol": "SYM_A", "path": "", "use_sidebar": False,
             "strategy": strategy_cls.NAME}
        ]

    syms = st.session_state["portfolio_symbols"]

    # Ensure legacy rows (created before use_sidebar existed) have the key
    for row in syms:
        row.setdefault("use_sidebar", False)

    # Render one row per symbol
    rows_to_delete = []
    for idx, row in enumerate(syms):
        c1, c2, c3, c4 = st.columns([1.5, 3, 2, 0.5])
        with c1:
            syms[idx]["symbol"] = st.text_input(
                "Symbol", row["symbol"], key=f"sym_name_{idx}"
            )
        with c2:
            use_sb = st.checkbox(
                "Use sidebar data",
                value=row.get("use_sidebar", False),
                key=f"use_sb_{idx}",
                disabled=not _sidebar_df_available,
                help="Use the dataset fetched in the sidebar (MT5 / cTrader / Yahoo).",
            )
            syms[idx]["use_sidebar"] = use_sb
            if not use_sb:
                syms[idx]["path"] = st.text_input(
                    "CSV path", row.get("path", ""),
                    placeholder="/path/to/data.csv",
                    key=f"sym_path_{idx}",
                )
            else:
                st.caption("← sidebar dataset will be used")
        with c3:
            syms[idx]["strategy"] = st.selectbox(
                "Strategy", list(REGISTRY.keys()),
                index=list(REGISTRY.keys()).index(row["strategy"])
                if row["strategy"] in REGISTRY else 0,
                key=f"sym_strat_{idx}",
            )
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✕", key=f"del_{idx}") and len(syms) > 1:
                rows_to_delete.append(idx)

    for i in reversed(rows_to_delete):
        syms.pop(i)

    if st.button("＋ Add symbol", key="add_sym"):
        syms.append({"symbol": f"SYM_{chr(65+len(syms))}",
                     "path": "", "use_sidebar": False,
                     "strategy": strategy_cls.NAME})
        st.rerun()

    # ── Portfolio risk controls ────────────────────────────────────────────────
    st.markdown("**Portfolio Risk Controls**")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        port_equity = st.number_input(
            "Starting Equity ($)", 1_000, 1_000_000,
            cfg.get("starting_equity", 10_000), 1_000, key="port_equity",
        )
    with col_p2:
        max_positions = st.number_input(
            "Max Total Positions", 1, 20, len(syms) * 2, 1,
            help="Maximum open positions across ALL symbols simultaneously.",
        )
    with col_p3:
        max_risk_usd = st.number_input(
            "Max Total Risk ($)", 0, 100_000, 0, 10,
            help="Portfolio risk cap in $. 0 = no cap.",
        )
    warmup_bars_p = st.slider("Warmup Bars", 20, 500, cfg.get("warmup_bars", 200),
                               25, key="port_warmup")

    if not st.button("▶ Run Portfolio Backtest", type="primary", key="port_run"):
        st.info(
            f"Configure {len(syms)} symbol(s) above and click "
            "**▶ Run Portfolio Backtest**."
        )
        return

    # ── Validate all sources ───────────────────────────────────────────────────
    errors = []
    _sidebar_df = st.session_state.get("loaded_df")
    for row in syms:
        if row.get("use_sidebar"):
            if _sidebar_df is None:
                errors.append(
                    f"**{row['symbol']}** — 'Use sidebar data' is checked but "
                    "no data has been fetched in the sidebar yet."
                )
        else:
            path = row.get("path", "").strip()
            if not path:
                errors.append(f"**{row['symbol']}** — CSV path is empty")
            elif not Path(path).exists():
                errors.append(f"**{row['symbol']}** — file not found: `{path}`")
    if errors:
        for e in errors:
            st.error(e)
        return

    # ── Load data ──────────────────────────────────────────────────────────────
    dfs: dict[str, pd.DataFrame] = {}
    for row in syms:
        try:
            if row.get("use_sidebar"):
                # Multiple symbols can share the sidebar dataset (e.g. running
                # different strategies on the same instrument for comparison)
                dfs[row["symbol"]] = _sidebar_df.copy()
            else:
                dfs[row["symbol"]] = _load_csv_path(row["path"])
        except Exception as e:
            st.error(f"Failed to load **{row['symbol']}**: {e}")
            return

    # ── Build broker configs (reuse all sidebar broker settings per symbol) ───
    _bc = cfg["broker_cfg"]
    broker_cfgs = {
        row["symbol"]: BrokerConfig(
            commission_flat      = _bc.commission_flat,
            lot_size             = _bc.lot_size,
            max_concurrent       = _bc.max_concurrent,
            risk_usd             = _bc.risk_usd,
            size_mode            = _bc.size_mode,
            spread               = _bc.spread,
            slippage_fixed       = _bc.slippage_fixed,
            slippage_pct         = _bc.slippage_pct,
            slippage_atr_mult    = _bc.slippage_atr_mult,
            trail_pct            = _bc.trail_pct,
            trail_activation_pct = _bc.trail_activation_pct,
            intrabar_path_model  = _bc.intrabar_path_model,
            gap_fill             = _bc.gap_fill,
            spread_schedule      = _bc.spread_schedule,
        )
        for row in syms
    }

    strategies = {
        row["symbol"]: REGISTRY[row["strategy"]]() for row in syms
    }

    pcfg = PortfolioConfig(
        symbols             = [row["symbol"] for row in syms],
        broker_configs      = broker_cfgs,
        starting_equity     = float(port_equity),
        warmup_bars         = warmup_bars_p,
        max_total_positions = int(max_positions),
        max_total_risk_usd  = float(max_risk_usd) if max_risk_usd > 0 else float("inf"),
    )

    with st.spinner("Running portfolio backtest …"):
        try:
            port_result = run_backtest_portfolio(
                dfs          = dfs,
                strategies   = strategies,
                portfolio_cfg = pcfg,
                verbose      = False,
            )
        except Exception as e:
            st.error(f"Portfolio backtest error: {e}")
            st.exception(e)
            return

    st.success(
        f"Done — {sum(r.n_trades for r in port_result.symbol_results.values()):,} "
        f"total trades across {len(syms)} symbol(s)"
    )

    # ── Shared equity curve ────────────────────────────────────────────────────
    st.subheader("Shared Equity Curve")
    eq = np.array(port_result.equity_curve)
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        y=eq, name="Portfolio Equity",
        line=dict(color=COLORS["IS"], width=2),
        hovertemplate="Event %{x}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig_eq.add_hline(y=port_equity, line_dash="dash", line_color="#888",
                     annotation_text=f"Start ${port_equity:,.0f}")
    fig_eq.update_layout(
        xaxis_title="Bar events (all symbols, chronological)",
        yaxis_title="Shared Equity ($)", height=350,
        template="plotly_dark" if _is_dark() else "plotly_white",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Per-symbol metrics ────────────────────────────────────────────────────
    st.subheader("Per-Symbol Performance")
    from engine.metrics import compute_metrics
    sym_rows = []
    for sym, res in port_result.symbol_results.items():
        m = compute_metrics(res)
        m["symbol"] = sym
        sym_rows.append(m)

    if sym_rows:
        df_sym = pd.DataFrame(sym_rows).set_index("symbol")
        display_cols = ["trades", "win_rate_%", "sharpe_ann",
                        "max_dd_%", "total_pnl_$", "profit_factor"]
        available = [c for c in display_cols if c in df_sym.columns]
        st.dataframe(df_sym[available].round(3), use_container_width=True)

        # PnL comparison bar chart
        if "total_pnl_$" in df_sym.columns:
            fig_bar = px.bar(
                df_sym.reset_index(), x="symbol", y="total_pnl_$",
                color="total_pnl_$",
                color_continuous_scale=["#E76F51", "#2A9D8F"],
                title="Total PnL by Symbol ($)",
            )
            fig_bar.update_layout(
                height=300, showlegend=False,
                template="plotly_dark" if _is_dark() else "plotly_white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Portfolio summary metrics ──────────────────────────────────────────────
    st.subheader("Portfolio Summary")
    final_eq = eq[-1] if len(eq) > 0 else port_equity
    total_return = (final_eq / port_equity - 1) * 100
    total_trades = sum(r.n_trades for r in port_result.symbol_results.values())
    peak   = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min() * 100) if len(eq) > 1 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Equity",   f"${final_eq:,.2f}",
              delta=f"{total_return:+.1f}%")
    c2.metric("Total Trades",   f"{total_trades:,}")
    c3.metric("Max Drawdown",   f"{max_dd:.1f}%")
    c4.metric("Symbols",        f"{len(syms)}")


# ── Funded account tab ───────────────────────────────────────────────────────

def tab_funded(results: dict, cfg: dict):
    """
    Funded Account Simulation tab.

    Applies prop-firm drawdown and daily-loss rules to any backtest split,
    returning a pass/fail verdict, failure details, and payout history.
    """
    st.subheader("🏦 Funded Account Simulation")
    st.caption(
        "Simulate how your strategy would perform under **prop-firm rules**: "
        "daily loss limits, max drawdown (absolute or trailing), profit targets, "
        "and payout cycles.  Choose a firm preset or configure custom rules."
    )

    if not results:
        st.info("Run a backtest first (sidebar → **▶ Run Backtest**).")
        return

    # ── Firm / phase selector ─────────────────────────────────────────────────
    col_preset, col_split = st.columns([2, 1])
    with col_preset:
        preset_name = st.selectbox(
            "Firm preset",
            list(FIRM_PRESETS.keys()),
            index=0,
            help="Select a prop-firm preset, or choose 'Custom' to set your own rules.",
        )
    with col_split:
        split_label = st.selectbox(
            "Apply to split",
            list(results.keys()),
            index=0,
            help="Which backtest split (IS/OOS/FWD) to apply the rules to.",
        )

    base_cfg  = FIRM_PRESETS[preset_name]
    is_custom = (preset_name == "Custom")

    # ── Config editor ─────────────────────────────────────────────────────────
    with st.expander(
        "⚙️ Rule Configuration" + (" (editing — Custom mode)" if is_custom else ""),
        expanded=is_custom,
    ):
        rc1, rc2 = st.columns(2)
        with rc1:
            f_equity = st.number_input(
                "Starting Equity ($)", 1_000, 500_000,
                int(base_cfg.starting_equity), 1_000,
                key="fa_equity",
            )
            f_dd_pct = st.slider(
                "Max Drawdown (%)", 1, 30, int(base_cfg.max_drawdown_pct * 100),
                key="fa_dd",
            )
            f_dd_type = st.radio(
                "Drawdown type",
                ["absolute", "trailing"],
                index=0 if base_cfg.drawdown_type == "absolute" else 1,
                horizontal=True,
                key="fa_dd_type",
                help=(
                    "**absolute** — floor fixed at initial balance (FTMO-style).  "
                    "**trailing** — floor tracks the all-time equity high (Topstep-style)."
                ),
            )
            f_daily_pct = st.slider(
                "Daily Loss Limit (%) — 0 = disabled",
                0, 20, int(base_cfg.daily_loss_limit_pct * 100),
                key="fa_daily",
            )
        with rc2:
            f_target_pct = st.slider(
                "Profit Target (%) — 0 = no target (funded phase)",
                0, 30, int(base_cfg.profit_target_pct * 100),
                key="fa_target",
            )
            f_min_days = st.number_input(
                "Min Trading Days", 1, 60, base_cfg.min_trading_days,
                key="fa_min_days",
                help="Minimum distinct calendar days with a closed trade before profit target counts.",
            )
            f_split_pct = st.slider(
                "Profit Split (trader %)", 50, 100, int(base_cfg.profit_split_pct * 100),
                key="fa_split",
            )
            f_payout_days = st.number_input(
                "Payout Frequency (days)", 1, 90, base_cfg.payout_frequency_days,
                key="fa_payout_days",
            )
            f_reset = st.checkbox(
                "Reset equity after payout",
                value=base_cfg.reset_on_payout,
                key="fa_reset",
                help=(
                    "Re-anchor account to starting equity after each payout.  "
                    "Most firms do NOT do this — leave unchecked unless you know "
                    "your firm resets capital."
                ),
            )

    # Always build config from widget values so it reflects any edits
    try:
        funded_cfg = FundedAccountConfig(
            starting_equity       = float(f_equity),
            daily_loss_limit_pct  = f_daily_pct / 100.0,
            max_drawdown_pct      = f_dd_pct / 100.0,
            drawdown_type         = f_dd_type,
            profit_target_pct     = f_target_pct / 100.0,
            min_trading_days      = int(f_min_days),
            profit_split_pct      = f_split_pct / 100.0,
            payout_frequency_days = int(f_payout_days),
            reset_on_payout       = f_reset,
        )
    except ValueError as e:
        st.error(f"Invalid configuration: {e}")
        return

    # ── Key parameters summary ─────────────────────────────────────────────────
    km1, km2, km3, km4 = st.columns(4)
    km1.metric("Max Drawdown",   f"{funded_cfg.max_drawdown_pct*100:.0f}%",
               help=f"${funded_cfg.max_loss_abs:,.0f} ({funded_cfg.drawdown_type})")
    km2.metric("Daily Loss Limit",
               f"{funded_cfg.daily_loss_limit_pct*100:.0f}%" if funded_cfg.daily_loss_limit_pct > 0 else "Off",
               help=f"${funded_cfg.daily_loss_abs:,.0f}" if funded_cfg.daily_loss_limit_pct > 0 else "No daily limit")
    km3.metric("Profit Target",
               f"{funded_cfg.profit_target_pct*100:.0f}%" if funded_cfg.profit_target_pct > 0 else "None",
               help=f"${funded_cfg.profit_target_abs:,.0f}" if funded_cfg.profit_target_pct > 0 else "Funded phase — payout on schedule")
    km4.metric("Trader Split",   f"{funded_cfg.profit_split_pct*100:.0f}%")

    st.markdown("---")

    if not st.button("▶ Run Funded Simulation", type="primary", key="fa_run"):
        st.info("Configure rules above and click **▶ Run Funded Simulation**.")
        return

    # ── Run simulation ────────────────────────────────────────────────────────
    with st.spinner("Simulating funded account rules ..."):
        result     = results[split_label]
        try:
            funded = run_funded_backtest(result, funded_cfg)
        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.exception(e)
            return

    # ── Pass / fail badge ─────────────────────────────────────────────────────
    if funded.passed:
        st.success(
            f"✅ **PASSED** — Strategy survives {split_label} under these rules.\n\n"
            + (f"Profit target ({funded_cfg.profit_target_pct*100:.0f}%) reached "
               f"after {funded.trading_days_active} trading day(s)."
               if funded_cfg.profit_target_pct > 0
               else f"No breach detected across {funded.trading_days_active} active trading day(s)."),
        )
    else:
        st.error(
            f"❌ **FAILED** — Strategy would have breached the rules.\n\n"
            f"{funded.failure_reason}"
        )

    # ── Key results metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Equity",        f"${funded.peak_equity:,.2f}")
    m2.metric("Max DD Reached",     f"{funded.max_drawdown_reached_pct:.1f}%",
              delta=f"Limit: {funded_cfg.max_drawdown_pct*100:.0f}%",
              delta_color="inverse" if funded.max_drawdown_reached_pct >= funded_cfg.max_drawdown_pct * 100
              else "off")
    m3.metric("Trading Days",       str(funded.trading_days_active),
              delta=f"Min required: {funded_cfg.min_trading_days}" if funded_cfg.profit_target_pct > 0 else None)
    m4.metric("Total Payout",       f"${funded.total_payout_to_trader:,.2f}",
              help=f"At {funded_cfg.profit_split_pct*100:.0f}% profit split")

    # ── Equity curve comparison ───────────────────────────────────────────────
    st.markdown("**Equity Curve — Strategy vs Funded Account Limits**")
    ts  = result.timestamps
    raw = result.equity_curve
    # Scale raw to funded account space (same scaling as FundedResult)
    valid_mask = ~np.isnan(raw)
    if valid_mask.any():
        bt_origin = float(raw[valid_mask][0])
        scale     = funded_cfg.starting_equity / bt_origin if bt_origin != 0 else 1.0
    else:
        scale = 1.0

    fig_fa = go.Figure()

    # Strategy equity (scaled)
    fig_fa.add_trace(go.Scatter(
        x=ts, y=raw * scale,
        mode="lines", name="Strategy equity",
        line=dict(color=COLORS["long"], width=1.5),
    ))

    # Funded equity (flattened after breach)
    fig_fa.add_trace(go.Scatter(
        x=ts, y=funded.funded_equity_curve,
        mode="lines", name="Funded equity",
        line=dict(color=COLORS["equity"], width=2),
    ))

    # Max drawdown floor line
    if funded_cfg.drawdown_type == "absolute":
        floor_val = funded_cfg.starting_equity * (1.0 - funded_cfg.max_drawdown_pct)
        fig_fa.add_hline(
            y=floor_val,
            line_dash="dash", line_color="red", line_width=1,
            annotation_text=f"Max DD floor ${floor_val:,.0f}",
            annotation_position="bottom right",
        )
    else:
        # Trailing floor — recompute for display
        hwm_arr   = np.maximum.accumulate(
            np.where(np.isnan(funded.funded_equity_curve),
                     funded_cfg.starting_equity, funded.funded_equity_curve)
        )
        floor_arr = hwm_arr * (1.0 - funded_cfg.max_drawdown_pct)
        fig_fa.add_trace(go.Scatter(
            x=ts, y=floor_arr,
            mode="lines", name="Trailing DD floor",
            line=dict(color="red", dash="dash", width=1),
        ))

    # Daily loss floor band (if enabled)
    if funded_cfg.daily_loss_limit_pct > 0.0:
        fig_fa.add_hline(
            y=funded_cfg.starting_equity - funded_cfg.daily_loss_abs,
            line_dash="dot", line_color="orange", line_width=1,
            annotation_text="Max single-day loss floor",
            annotation_position="top right",
        )

    # Failure marker
    if funded.failure_bar is not None and funded.failure_bar < len(ts):
        fig_fa.add_vline(
            x=ts[funded.failure_bar],
            line_dash="dash", line_color="red", line_width=2,
            annotation_text="❌ Breach",
            annotation_position="top right",
        )

    # Profit target line
    if funded_cfg.profit_target_pct > 0.0:
        fig_fa.add_hline(
            y=funded_cfg.starting_equity * (1.0 + funded_cfg.profit_target_pct),
            line_dash="dash", line_color="green", line_width=1,
            annotation_text=f"Profit target ${funded_cfg.profit_target_abs:,.0f}",
            annotation_position="bottom right",
        )

    # Payout markers
    for pev in funded.payout_events:
        if pev.bar_index < len(ts):
            fig_fa.add_vline(
                x=ts[pev.bar_index],
                line_dash="dot", line_color="gold", line_width=1,
                annotation_text=f"💰 ${pev.trader_share:,.0f}",
                annotation_position="top left",
            )

    fig_fa.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=50, r=30, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_fa, use_container_width=True)

    # ── Payout history table ───────────────────────────────────────────────────
    if funded.payout_events:
        st.markdown("**Payout History**")
        payout_rows = [
            {
                "Cycle":          p.cycle_number,
                "Date":           p.date,
                "Equity Before":  f"${p.equity_before:,.2f}",
                "Gross Profit":   f"${p.profit_gross:,.2f}",
                "Trader Share":   f"${p.trader_share:,.2f}",
                "Firm Share":     f"${p.firm_share:,.2f}",
                "Equity After":   f"${p.equity_after:,.2f}",
            }
            for p in funded.payout_events
        ]
        st.dataframe(pd.DataFrame(payout_rows), use_container_width=True, hide_index=True)
    else:
        st.caption(
            "No payout events recorded.  "
            "Payouts trigger when profit > 0 at each payout-frequency boundary."
        )

    # ── Rule summary ──────────────────────────────────────────────────────────
    with st.expander("📋 Rule Summary"):
        dd_ref = ("fixed at starting_equity"
                  if funded_cfg.drawdown_type == "absolute"
                  else "trailing from all-time equity high")
        st.markdown(f"""
| Rule | Limit | $ Value |
|------|-------|---------|
| Max drawdown ({funded_cfg.drawdown_type}) | {funded_cfg.max_drawdown_pct*100:.1f}% | ${funded_cfg.max_loss_abs:,.0f} |
| Daily loss limit | {funded_cfg.daily_loss_limit_pct*100:.1f}% | ${funded_cfg.daily_loss_abs:,.0f} |
| Profit target | {funded_cfg.profit_target_pct*100:.1f}% | ${funded_cfg.profit_target_abs:,.0f} |
| Min trading days | {funded_cfg.min_trading_days} | — |
| Profit split | {funded_cfg.profit_split_pct*100:.0f}% trader | — |
| Payout every | {funded_cfg.payout_frequency_days} days | — |

Drawdown reference: *{dd_ref}*
        """)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _show_quickstart():
    with st.expander("Quick Start", expanded=True):
        st.markdown("""
1. **Set data path** — point to your OHLCV CSV (`DateTime, Open, High, Low, Close, Volume`)
2. **Pick a strategy** — IFVG M1 Scalping or EMA Crossover (or add your own in `strategies/`)
3. **Tune parameters** — sliders are auto-generated from the strategy definition
4. **Configure broker** — risk per trade, commission, max positions
5. **Click ▶ Run Backtest** — results appear across all tabs

**Adding a new strategy:**
```python
# strategies/my_strategy.py
from engine.strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    NAME = "My Strategy"
    PARAMS = {"period": {"default": 20, "min": 5, "max": 100, "step": 1, "label": "Period"}}
    def prepare(self, feed): ...
    def on_bar(self, feed, broker): ...
```
Then add it to `strategies/__init__.py` → `REGISTRY`.
        """)


def _is_dark() -> bool:
    """Rough dark-mode detection via streamlit theme."""
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = sidebar()

    st.title("📈 ZEngine — Strategy Lab")
    st.caption(
        "🔒 Look-ahead bias is **structurally impossible** — "
        "the engine's DataFeed blocks any access to future bars at runtime."
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    df, status = load_data(cfg)

    if status == "csv_no_path":
        st.info("👈 Choose a data source in the sidebar to get started.")
        _show_quickstart()
        return
    elif status.startswith("csv_not_found:"):
        path = status.split(":", 1)[1]
        st.warning(f"Data file not found: `{path}`")
        st.info("Check the CSV path in the sidebar.")
        _show_quickstart()
        return
    elif status == "csv_no_upload":
        st.info("👈 Upload a CSV file in the sidebar.")
        _show_quickstart()
        return
    elif status == "fetch_first":
        st.info("👈 Click **📥 Fetch** in the sidebar to load data from MT5, cTrader, or Yahoo Finance.")
        _show_quickstart()
        return
    elif status == "source_unavailable":
        st.warning(
            "The selected data source is not available on this platform. "
            "Switch to **CSV** or **Yahoo Finance** in the sidebar."
        )
        _show_quickstart()
        return
    elif status.startswith("error:"):
        st.error(f"Failed to load data: {status[6:]}")
        return

    # Show data summary banner
    src_tag = st.session_state.get("data_source_tag", "")
    if src_tag:
        st.caption(f"📊 {src_tag}")
    else:
        st.caption(
            f"📁 {len(df):,} bars | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )

    # Run on button press
    if cfg["run"]:
        with st.spinner("Running backtest ..."):
            try:
                results = run_splits(
                    df, cfg["strategy_cls"], cfg["params"],
                    cfg["broker_cfg"], cfg["starting_equity"],
                    cfg["warmup_bars"], cfg["splits"],
                    verbose=False,
                )
                st.session_state["results"] = results
                st.session_state["cfg"]     = cfg
                st.session_state["df"]      = df
            except Exception as e:
                st.error(f"Backtest error: {e}")
                st.exception(e)
                return
        st.success(f"Done — {sum(r.n_trades for r in results.values()):,} total closed trades")

    results = st.session_state.get("results", {})
    cfg_s   = st.session_state.get("cfg", cfg)
    df_s    = st.session_state.get("df", df)

    if not results:
        st.info("👈 Configure your strategy in the sidebar and click **▶ Run Backtest**.")
        _show_quickstart()
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Overview", "📉 Chart", "📈 Equity", "📋 Trades", "🔍 Breakdown",
        "⚡ Compare", "🔄 Walk-Forward", "🎲 Monte Carlo", "💼 Portfolio",
        "🏦 Funded",
    ])

    with tab1:
        tab_overview(results)
    with tab2:
        tab_chart(results, df_s)
    with tab3:
        tab_equity(results)
    with tab4:
        tab_trades(results)
    with tab5:
        tab_breakdown(results)
    with tab6:
        tab_compare(cfg_s, df_s)
    with tab7:
        tab_walkforward(cfg_s, df_s)
    with tab8:
        tab_montecarlo(results, cfg_s)
    with tab9:
        tab_portfolio(cfg_s)
    with tab10:
        tab_funded(results, cfg_s)


if __name__ == "__main__":
    main()
