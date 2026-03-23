"""
pytest Configuration & Shared Fixtures
Runs before any test file. Sets up the test environment.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

# ── Force test-safe environment ───────────────────────────────────────────────
os.environ["PAPER_TRADE"]  = "true"
os.environ["GROQ_API_KEY"] = "test_key_not_real"

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_ohlcv_up():
    """60-bar uptrending OHLCV DataFrame."""
    np.random.seed(1)
    prices = [1000.0]
    for _ in range(59):
        prices.append(max(10, prices[-1] + np.random.normal(1, 1.5)))
    return _make_df(prices)


@pytest.fixture(scope="session")
def sample_ohlcv_down():
    """60-bar downtrending OHLCV DataFrame."""
    np.random.seed(2)
    prices = [1000.0]
    for _ in range(59):
        prices.append(max(10, prices[-1] + np.random.normal(-1, 1.5)))
    return _make_df(prices)


@pytest.fixture(scope="session")
def sample_ohlcv_flat():
    """60-bar flat OHLCV DataFrame."""
    np.random.seed(3)
    prices = [500.0 + np.random.normal(0, 2) for _ in range(60)]
    return _make_df(prices)


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite DB path for RiskManagerAgent tests."""
    import config
    original = config.DB_PATH
    config.DB_PATH = str(tmp_path / "test_trades.db")
    yield config.DB_PATH
    config.DB_PATH = original


@pytest.fixture
def risk_manager(tmp_db):
    """Fresh RiskManagerAgent with isolated test DB."""
    from agents.risk_manager import RiskManagerAgent
    return RiskManagerAgent(capital=100_000)


@pytest.fixture
def paper_executor():
    """TradeExecutorAgent in paper trading mode."""
    from agents.trade_executor import TradeExecutorAgent
    return TradeExecutorAgent()


@pytest.fixture
def circuit_breaker():
    """CircuitBreaker with a mock notify function."""
    from utils.circuit_breaker import CircuitBreaker
    alerts = []
    return CircuitBreaker(
        capital=100_000,
        notify_fn=lambda m: alerts.append(m),
        on_halt=lambda: None,
    ), alerts


@pytest.fixture
def market_calendar():
    """MarketCalendar instance (uses hardcoded holidays in tests)."""
    from utils.market_calendar import MarketCalendar
    return MarketCalendar()


def _make_df(prices) -> pd.DataFrame:
    """Helper: turn a list of prices into an OHLCV DataFrame."""
    data = []
    for p in prices:
        vol = np.random.uniform(0.5, 2.0)
        data.append({
            "open":   p + np.random.uniform(-1, 1),
            "high":   p + abs(np.random.normal(0, 1.5)),
            "low":    p - abs(np.random.normal(0, 1.5)),
            "close":  p,
            "volume": int(1_000_000 * vol),
        })
    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=len(prices), freq="D")
    return df


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "live: requires live API keys (skip in CI)")
    config.addinivalue_line("markers", "integration: end-to-end integration tests")
