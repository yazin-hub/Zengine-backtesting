"""
ZEngine — Open-Source Event-Driven Backtesting Engine
Look-ahead bias is structurally impossible by design.
"""
# Explicit re-exports so that `from engine import X` works for users
# and ruff recognises these as intentional public-API re-exports.
from .backtest import (
    run_backtest as run_backtest,
    run_splits as run_splits,
    run_backtest_mtf as run_backtest_mtf,
    audit_indicator_causality as audit_indicator_causality,
)
from .strategy import BaseStrategy as BaseStrategy
from .data import (
    DataFeed as DataFeed,
    LookAheadError as LookAheadError,
    CausalityError as CausalityError,
)
from .broker import (
    Broker as Broker,
    BrokerConfig as BrokerConfig,
    Order as Order,
    OrderSide as OrderSide,
    OrderType as OrderType,
    Trade as Trade,
)
from .metrics import compute_metrics as compute_metrics
from .walkforward import (
    run_walk_forward as run_walk_forward,
    WalkForwardResult as WalkForwardResult,
    WalkForwardWindow as WalkForwardWindow,
)
from .monte_carlo import (
    run_monte_carlo as run_monte_carlo,
    MonteCarloResult as MonteCarloResult,
)
from .mtf import (
    MTFStrategy as MTFStrategy,
    MultiTimeframeFeed as MultiTimeframeFeed,
    build_mtf_feeds as build_mtf_feeds,
)
from .portfolio import (
    Portfolio as Portfolio,
    PortfolioConfig as PortfolioConfig,
    PortfolioResult as PortfolioResult,
    run_backtest_portfolio as run_backtest_portfolio,
)
from .funded import (
    FundedAccountConfig as FundedAccountConfig,
    FundedResult as FundedResult,
    PayoutEvent as PayoutEvent,
    FIRM_PRESETS as FIRM_PRESETS,
    run_funded_backtest as run_funded_backtest,
)
