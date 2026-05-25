"""
run_pairs_backtest.py — BTC/ETH Spread Strategy Backtest
=========================================================

Runs the BTC/ETH z-score mean reversion strategy on 15m CSV data.

Usage:
    python run_pairs_backtest.py

Expects:
    BTCUSD_M15.csv and ETHUSD_M15.csv in the same directory
    (or update the paths below)

Output:
    - Console summary (trades, win rate, PnL, drawdown)
    - equity_curve.csv  (for further analysis)
    - zscore_history.csv (spread and z-score per bar)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configure logging so engine progress is visible ──────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)

# ── Add zengine root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.broker import BrokerConfig
from engine.pairs import run_backtest_pairs
from strategies.btc_eth_spread import BTCETHSpreadStrategy

# ── Data paths ────────────────────────────────────────────────────────────────
BTC_CSV = ROOT / "BTCUSD_M15.csv"
ETH_CSV = ROOT / "ETHUSD_M15.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV CSV with DateTime index."""
    df = pd.read_csv(path, index_col="DateTime")
    df.index = pd.to_datetime(df.index, format="mixed", utc=True)
    df.columns = df.columns.str.lower()
    return df


def main() -> None:
    print("\n" + "=" * 60)
    print("  BTC/ETH Spread Strategy — ZEngine Backtest")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {BTC_CSV.name} ...")
    df_btc = load_csv(BTC_CSV)
    print(f"  BTC: {len(df_btc):,} bars | {df_btc.index[0].date()} → {df_btc.index[-1].date()}")

    print(f"Loading {ETH_CSV.name} ...")
    df_eth = load_csv(ETH_CSV)
    print(f"  ETH: {len(df_eth):,} bars | {df_eth.index[0].date()} → {df_eth.index[-1].date()}")

    # ── Strategy configuration ────────────────────────────────────────────────
    strategy = BTCETHSpreadStrategy(params={
        "hedge_ratio": 18.0,   # Spread = BTC - 18 * ETH
        "lookback":    100,    # 100-bar rolling window (~25 hours on M15)
        "entry_z":     2.0,    # Enter when |Z| > 2.0
        "exit_z":      0.3,    # Exit when |Z| < 0.3
        "sl_z":        3.5,    # Emergency stop at |Z| > 3.5
        "volume_btc":  0.01,   # 0.01 BTC per trade (~$300-600 notional)
    })

    # ── Broker configuration ──────────────────────────────────────────────────
    # Realistic crypto exchange fees: 0.05% taker fee each side
    # Based on typical cTrader/exchange commission structure
    broker_cfg = BrokerConfig(
        commission_pct  = 0.0005,   # 0.05% per fill (taker fee)
        slippage_pct    = 0.0001,   # 0.01% slippage per fill
        max_concurrent  = 1,        # one pairs trade at a time
    )

    # ── Run backtest ──────────────────────────────────────────────────────────
    print("\nRunning backtest ...\n")
    result = run_backtest_pairs(
        df_a            = df_btc,
        df_b            = df_eth,
        strategy        = strategy,
        broker_config_a = broker_cfg,
        broker_config_b = broker_cfg,
        starting_equity = 10_000.0,
        warmup_bars     = 100,
        label           = "BTC/ETH v1",
        symbol_a        = "BTCUSD",
        symbol_b        = "ETHUSD",
        verbose         = True,
    )

    # ── Per-trade breakdown ───────────────────────────────────────────────────
    print("\n── Per-trade P&L (combined legs) ──")
    print(f"  {'#':<4} {'BTC PnL':>10} {'ETH PnL':>10} {'Combined':>10}  Exit reason")
    print(f"  {'-'*52}")
    for i, (ta, tb) in enumerate(zip(result.closed_trades_a, result.closed_trades_b), 1):
        combined = ta.pnl_net + tb.pnl_net
        marker   = "✅" if combined > 0 else "❌"
        print(f"  {i:<4} {ta.pnl_net:>10.2f} {tb.pnl_net:>10.2f} {combined:>10.2f}  "
              f"{marker} {ta.exit_reason}")
        if i >= 30:
            remaining = result.n_pairs_trades - 30
            if remaining > 0:
                print(f"  ... {remaining} more trades not shown")
            break

    # ── Z-score stats ─────────────────────────────────────────────────────────
    valid_z = result.zscore_history[~np.isnan(result.zscore_history)]
    if len(valid_z):
        print(f"\n── Z-score statistics ──")
        print(f"  Mean : {np.mean(valid_z):>8.4f}")
        print(f"  Std  : {np.std(valid_z):>8.4f}")
        print(f"  Min  : {np.min(valid_z):>8.4f}")
        print(f"  Max  : {np.max(valid_z):>8.4f}")
        pct_above2 = np.mean(np.abs(valid_z) > 2.0) * 100
        print(f"  |Z|>2: {pct_above2:>7.2f}% of bars (potential entry signals)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = ROOT

    eq_df = pd.DataFrame({
        "timestamp":    result.timestamps,
        "equity":       result.equity_curve,
        "spread":       result.spread_history,
        "zscore":       result.zscore_history,
    }).set_index("timestamp")
    eq_df.to_csv(out_dir / "pairs_equity_curve.csv")
    print(f"\nSaved: pairs_equity_curve.csv")
    print("Done.")


if __name__ == "__main__":
    main()
