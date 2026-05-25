# Contributing to ZEngine

Thank you for considering a contribution to ZEngine! This document covers how to set up a development environment, the conventions we follow, and how to submit a pull request.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Running the Tests](#running-the-tests)
4. [Code Style](#code-style)
5. [Adding a Strategy](#adding-a-strategy)
6. [Adding a Data Source](#adding-a-data-source)
7. [Pull Request Checklist](#pull-request-checklist)
8. [Reporting Bugs](#reporting-bugs)

---

## Getting Started

```bash
git clone https://github.com/<your-fork>/zengine.git
cd zengine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

That installs ZEngine in editable mode along with pytest, pytest-cov, and ruff.

---

## Development Setup

**Optional data-source extras** (only if you need them locally):

| Extra | Installs | Platform |
|-------|----------|----------|
| `.[mt5]` | MetaTrader5 | Windows only |
| `.[ctrader]` | ctrader-open-api, requests | All |
| `.[yahoo]` | yfinance | All |
| `.[all-sources]` | All of the above | Windows + All |

```bash
pip install -e ".[yahoo]"         # e.g. add Yahoo Finance support
```

Launch the Strategy Lab UI:

```bash
streamlit run app.py
# or, after pip install:
zengine-ui
```

---

## Running the Tests

```bash
pytest                            # run all tests
pytest -k test_engine             # run a specific module
pytest --cov=engine --cov-report=term-missing   # with coverage
```

CI runs the full suite on Python 3.10, 3.11, and 3.12 via GitHub Actions on every push and PR. All tests must pass before a PR can be merged.

---

## Code Style

ZEngine uses **ruff** for linting (line length 100, target py310):

```bash
ruff check engine/ strategies/ tests/
ruff check --fix engine/ strategies/ tests/   # auto-fix safe issues
```

A few conventions we ask you to follow:

- **Type annotations on all public functions.** Return type included.
- **Docstrings on all public classes and functions** (one-liner is fine for simple helpers).
- **No look-ahead in engine code** — the whole point of this project is structural look-ahead freedom. Any PR that reads a future bar (`data[i+1]`, `data.iloc[-1]` inside the strategy loop, etc.) will be rejected.
- **Causal access only.** The `DataFeed` guard raises `LookAheadError` at runtime for any index >= current bar. Keep that invariant intact.
- Prefer immutable `dataclass(frozen=True)` for configuration objects.
- No `print()` in engine or strategy code — use Python `logging`.

---

## Adding a Strategy

1. Create `strategies/my_strategy.py`:

```python
from engine.strategy import Strategy, Params
from engine.data import DataFeed
from engine.broker import Broker

class MyStrategy(Strategy):
    NAME = "My Strategy"
    PARAMS: Params = {
        "fast_period": 10,
        "slow_period": 30,
    }

    def on_bar(self, bar: int, feed: DataFeed, broker: Broker) -> None:
        # Only access feed[bar] or earlier — never feed[bar+1]
        fast_ma = feed.close[:bar+1][-self.params["fast_period"]:].mean()
        slow_ma = feed.close[:bar+1][-self.params["slow_period"]:].mean()
        ...
```

2. Register it in `strategies/__init__.py`:

```python
from .my_strategy import MyStrategy

REGISTRY = {
    ...,
    MyStrategy.NAME: MyStrategy,
}
```

3. Add tests in `tests/test_engine.py` or a new `tests/test_my_strategy.py`. Verify:
   - Strategy runs without `LookAheadError`
   - Trade count is non-zero on a synthetic OHLCV fixture
   - Metrics dict has expected keys

---

## Adding a Data Source

1. Subclass `engine.data_sources.base.BaseDataSource` (or follow the pattern of `CSVSource`/`YahooSource`).
2. Implement `is_available() -> bool`, `fetch_bars(symbol, ...) -> pd.DataFrame`, and `get_broker_defaults(symbol) -> dict`.
3. Register in `engine/data_sources/__init__.py` inside `available_sources()`.
4. Add at least:
   - An `is_available()` test that mocks the optional import
   - A `fetch_bars()` test that returns a correct OHLCV DataFrame
5. Document any required credentials or setup in `README.md` under **Data Sources**.

---

## Pull Request Checklist

Before opening a PR, please confirm:

- [ ] `pytest` passes locally (all existing + new tests)
- [ ] `ruff check engine/ strategies/ tests/` reports no errors
- [ ] New public functions/classes have docstrings and type annotations
- [ ] No future-bar access introduced in engine or strategy code
- [ ] `CHANGELOG.md` entry added (create the file if it doesn't exist yet)
- [ ] PR description explains *what* changed and *why*

---

## Reporting Bugs

Please open a GitHub issue with:

1. ZEngine version (`pip show zengine`)
2. Python version and OS
3. Minimal reproducible example — ideally a failing test or a short script
4. Full traceback

---

Thanks again — every contribution, from a one-line typo fix to a new data source, is appreciated.
