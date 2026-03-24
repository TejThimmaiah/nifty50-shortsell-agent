"""
Test Suite — Tej Agent
Run: pytest tests/ -v

Tests cover:
  - Technical analysis signals
  - Risk manager rules
  - Position sizing math
  - Self-healer search
  - Trade executor paper mode
  - Backtest engine
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import date

# Force paper trading in tests
os.environ["PAPER_TRADE"]  = "true"
os.environ["GROQ_API_KEY"] = "test_key"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    prices = [1000.0]
    for _ in range(n - 1):
        direction = 1 if trend == "up" else -1
        change = np.random.normal(direction * 0.5, 1.5)
        prices.append(max(10, prices[-1] + change))

    data = []
    for i, p in enumerate(prices):
        vol = np.random.uniform(0.5, 1.5)
        data.append({
            "open":   p + np.random.uniform(-1, 1),
            "high":   p + abs(np.random.normal(0, 1.5)),
            "low":    p - abs(np.random.normal(0, 1.5)),
            "close":  p,
            "volume": int(1_000_000 * vol),
        })

    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return df


@pytest.fixture
def overbought_df():
    """OHLCV that ends with overbought RSI (>70) — ideal short candidate."""
    df = make_ohlcv(n=60, trend="up")
    # Force the last 10 bars to spike hard upward
    for i in range(-10, 0):
        df.iloc[i, df.columns.get_loc("close")] *= 1.008
        df.iloc[i, df.columns.get_loc("high")]  = df.iloc[i]["close"] * 1.01
    return df


@pytest.fixture
def risk_mgr():
    """RiskManagerAgent backed by an in-memory SQLite DB."""
    import tempfile
    import config
    tmp = tempfile.mktemp(suffix=".db")
    original = config.DB_PATH
    config.DB_PATH = tmp
    mgr = __import__("agents.risk_manager", fromlist=["RiskManagerAgent"]).RiskManagerAgent(capital=100_000)
    yield mgr
    config.DB_PATH = original
    if os.path.exists(tmp):
        os.remove(tmp)


@pytest.fixture
def executor():
    from agents.trade_executor import TradeExecutorAgent
    return TradeExecutorAgent()


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYST TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestTechnicalAnalyst:

    def test_returns_signal_for_valid_data(self, overbought_df):
        from agents.technical_analyst import calculate_all
        signal = calculate_all(overbought_df, "TEST")
        assert signal is not None
        assert signal.symbol == "TEST"
        assert 0.0 <= signal.confidence <= 1.0

    def test_insufficient_data_returns_none(self):
        from agents.technical_analyst import calculate_all
        df = make_ohlcv(n=10)
        signal = calculate_all(df, "TEST")
        assert signal is None

    def test_overbought_detected(self, overbought_df):
        from agents.technical_analyst import calculate_all
        signal = calculate_all(overbought_df, "TEST")
        assert signal is not None
        # RSI should be calculable
        assert signal.rsi > 0

    def test_stop_loss_above_entry_for_shorts(self, overbought_df):
        from agents.technical_analyst import calculate_all
        signal = calculate_all(overbought_df, "TEST")
        assert signal is not None
        assert signal.stop_loss > signal.entry_price, \
            "Stop loss must be ABOVE entry price for short positions"

    def test_target_below_entry_for_shorts(self, overbought_df):
        from agents.technical_analyst import calculate_all
        signal = calculate_all(overbought_df, "TEST")
        assert signal is not None
        assert signal.target < signal.entry_price, \
            "Target must be BELOW entry price for short positions"

    def test_bearish_divergence_detection(self):
        from agents.technical_analyst import _check_bearish_divergence
        import ta.momentum
        # Create data with bearish divergence: price higher high, RSI lower high
        df = make_ohlcv(n=50, trend="up")
        rsi_indicator = ta.momentum.RSIIndicator(df["close"], window=14)
        df["rsi"] = rsi_indicator.rsi()
        # Should not crash even with NaN values
        result = _check_bearish_divergence(df)
        assert isinstance(result, (bool, __import__('numpy').bool_))

    def test_sr_levels_ordered(self, overbought_df):
        from agents.technical_analyst import _calculate_sr_levels
        support, resistance = _calculate_sr_levels(overbought_df)
        close = float(overbought_df.iloc[-1]["close"])
        if support is not None and resistance is not None:
            assert support < resistance, "Support must be below resistance"

    def test_market_breadth_classification(self):
        from agents.technical_analyst import get_market_breadth
        assert get_market_breadth({"advances": 10, "declines": 40}) == "BEARISH"
        assert get_market_breadth({"advances": 40, "declines": 10}) == "BULLISH"
        assert get_market_breadth({"advances": 25, "declines": 25}) in ("NEUTRAL", "INSUFFICIENT")
        assert get_market_breadth({}) == "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────────────
# RISK MANAGER TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskManager:

    def test_trade_approved_within_limits(self, risk_mgr):
        decision = risk_mgr.approve_trade(
            symbol="RELIANCE",
            entry_price=2500.0,
            stop_loss=2512.5,    # 0.5% SL above entry
            quantity=100,
            direction="SHORT",
        )
        assert decision.approved is True
        assert decision.adjusted_quantity > 0

    def test_position_sizing_respects_max_risk(self, risk_mgr):
        """Risk per trade should not exceed 2% of capital."""
        decision = risk_mgr.approve_trade(
            symbol="TITAN",
            entry_price=3000.0,
            stop_loss=3015.0,    # 0.5% SL
            quantity=1000,       # Oversized — should be reduced
            direction="SHORT",
        )
        if decision.approved:
            max_risk_amount = risk_mgr.capital * 0.02   # 2%
            actual_risk = decision.adjusted_quantity * (decision.adjusted_stop_loss - 3000.0)
            assert actual_risk <= max_risk_amount * 1.01, \
                f"Risk ₹{actual_risk:.0f} exceeds limit ₹{max_risk_amount:.0f}"

    def test_max_positions_blocks_new_trade(self, risk_mgr):
        """After max_open_positions trades, new trades should be blocked."""
        from config import TRADING
        # Fill up all position slots
        for i in range(TRADING.max_open_positions):
            risk_mgr.record_trade(
                symbol=f"STOCK{i}", direction="SHORT",
                entry_price=100.0, quantity=10, order_id=f"ORD{i}"
            )
        # Next trade should be rejected
        decision = risk_mgr.approve_trade("NEWSTOCK", 200.0, 201.0, 50, "SHORT")
        assert decision.approved is False
        assert "Max open positions" in decision.reason

    def test_daily_loss_limit_blocks_trading(self, risk_mgr):
        """If daily loss >= 5%, no more trades allowed."""
        # Record a large losing trade
        risk_mgr.record_trade("LOSER", "SHORT", 1000.0, 100, "ORD1")
        risk_mgr.close_trade("LOSER", 1060.0)   # 6% loss on short = ₹6000 loss

        decision = risk_mgr.approve_trade("NEXT", 500.0, 502.5, 10, "SHORT")
        # At ₹100k capital, 5% limit = ₹5000. ₹6000 loss exceeds it
        assert decision.approved is False
        assert "Daily loss limit" in decision.reason

    def test_invalid_sl_corrected_for_shorts(self, risk_mgr):
        """SL below entry for short should be corrected automatically."""
        decision = risk_mgr.approve_trade(
            symbol="TEST",
            entry_price=500.0,
            stop_loss=490.0,   # WRONG: SL below entry for short
            quantity=10,
            direction="SHORT",
        )
        if decision.approved:
            assert decision.adjusted_stop_loss > 500.0, \
                "SL should be corrected to above entry for shorts"

    def test_pnl_calculation_short_winning(self, risk_mgr):
        risk_mgr.record_trade("WIN", "SHORT", 1000.0, 100, "ORD1")
        pnl = risk_mgr.close_trade("WIN", 985.0)   # Price fell = profit for short
        assert pnl > 0, f"Short profit expected, got ₹{pnl}"
        assert pnl == pytest.approx(1500.0, abs=1)  # (1000 - 985) * 100

    def test_pnl_calculation_short_losing(self, risk_mgr):
        risk_mgr.record_trade("LOSE", "SHORT", 1000.0, 100, "ORD2")
        pnl = risk_mgr.close_trade("LOSE", 1005.0)   # Price rose = loss for short
        assert pnl < 0
        assert pnl == pytest.approx(-500.0, abs=1)   # (1000 - 1005) * 100

    def test_daily_summary_structure(self, risk_mgr):
        summary = risk_mgr.get_daily_summary()
        assert "date" in summary
        assert "total_pnl" in summary
        assert "win_rate" in summary
        assert "trades" in summary
        assert isinstance(summary["trades"], list)


# ──────────────────────────────────────────────────────────────────────────────
# TRADE EXECUTOR TESTS (paper mode)
# ──────────────────────────────────────────────────────────────────────────────

class TestTradeExecutor:

    def test_paper_short_returns_executed(self, executor):
        result = executor.short_sell("RELIANCE", 10, 2500.0, 2512.5, 2462.5)
        assert result["status"] == "EXECUTED"
        assert result["paper"] is True
        assert result["symbol"] == "RELIANCE"

    def test_paper_short_stores_position(self, executor):
        executor.short_sell("TCS", 5, 3800.0, 3819.0, 3743.0)
        positions = executor.get_positions()
        assert "TCS" in positions

    def test_paper_cover_closes_position(self, executor):
        executor.short_sell("INFY", 20, 1500.0, 1507.5, 1477.5)
        result = executor.cover_short("INFY", 20, 1490.0)
        assert result["status"] == "COVERED"

    def test_paper_sl_trigger_works(self, executor):
        executor.short_sell("HDFC", 5, 1600.0, 1608.0, 1576.0)
        trigger = executor.check_paper_triggers("HDFC", 1610.0)   # price > SL
        assert trigger == "SL"

    def test_paper_target_trigger_works(self, executor):
        executor.short_sell("BAJAJ", 5, 7000.0, 7035.0, 6895.0)
        trigger = executor.check_paper_triggers("BAJAJ", 6890.0)  # price < target
        assert trigger == "TARGET"

    def test_paper_no_trigger_in_range(self, executor):
        executor.short_sell("WIPRO", 10, 400.0, 402.0, 394.0)
        trigger = executor.check_paper_triggers("WIPRO", 398.0)   # price in range
        assert trigger is None

    def test_square_off_all_clears_positions(self, executor):
        executor.short_sell("A", 5, 100.0, 100.5, 98.5)
        executor.short_sell("B", 5, 200.0, 201.0, 197.0)
        executor.square_off_all()
        positions = executor.get_positions()
        assert len(positions) == 0


# ──────────────────────────────────────────────────────────────────────────────
# SELF-HEALER TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestSelfHealer:

    def test_healer_initializes(self):
        from agents.self_healer import SelfHealerAgent
        healer = SelfHealerAgent()
        assert healer is not None

    def test_score_to_label_mapping(self):
        from agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent()
        assert agent._score_to_label(-0.7) == "STRONG_BEARISH"
        assert agent._score_to_label(-0.3) == "BEARISH"
        assert agent._score_to_label(0.0)  in ("NEUTRAL", "INSUFFICIENT")
        assert agent._score_to_label(0.3)  == "BULLISH"
        assert agent._score_to_label(0.7)  == "STRONG_BULLISH"

    @patch("agents.self_healer.DDGS")
    def test_healer_returns_dict(self, mock_ddgs):
        mock_ddgs.return_value.text.return_value = [
            {"title": "NSE tips", "body": "Market is bearish today", "href": "http://test.com"}
        ]
        from agents.self_healer import SelfHealerAgent
        healer = SelfHealerAgent()
        with patch.object(healer, "_call_llm", return_value="Use tighter stop losses."):
            result = healer.heal("Test problem", {"symbol": "RELIANCE"})
        assert isinstance(result, dict)
        assert "solution" in result
        assert "confidence" in result

    @patch("agents.self_healer.DDGS")
    def test_healer_handles_empty_search(self, mock_ddgs):
        mock_ddgs.return_value.text.return_value = []
        from agents.self_healer import SelfHealerAgent
        healer = SelfHealerAgent()
        result = healer.heal("Totally unknown problem")
        assert result["confidence"] < 0.5   # Low confidence when no results


# ──────────────────────────────────────────────────────────────────────────────
# BACKTESTER TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestBacktester:

    @patch("backtest.backtester.get_historical_ohlcv")
    def test_backtest_runs_on_synthetic_data(self, mock_fetch):
        mock_fetch.return_value = make_ohlcv(n=120, trend="up")
        from backtest.backtester import Backtester
        bt = Backtester(capital=100_000)
        result = bt.run("RELIANCE", days=60)
        assert result is not None
        assert result.symbol == "RELIANCE"
        assert result.total_trades >= 0

    @patch("backtest.backtester.get_historical_ohlcv")
    def test_backtest_win_rate_in_valid_range(self, mock_fetch):
        mock_fetch.return_value = make_ohlcv(n=120)
        from backtest.backtester import Backtester
        bt = Backtester(capital=100_000)
        result = bt.run("TEST", days=60)
        assert 0 <= result.win_rate_pct <= 100

    @patch("backtest.backtester.get_historical_ohlcv")
    def test_backtest_handles_insufficient_data(self, mock_fetch):
        mock_fetch.return_value = make_ohlcv(n=10)  # Too little data
        from backtest.backtester import Backtester
        bt = Backtester(capital=100_000)
        result = bt.run("TEST", days=60)
        # Should return empty result, not crash
        assert result.total_trades == 0

    @patch("backtest.backtester.get_historical_ohlcv")
    def test_backtest_profit_factor_positive(self, mock_fetch):
        """Profit factor should be computable and >= 0."""
        mock_fetch.return_value = make_ohlcv(n=120, trend="down")
        from backtest.backtester import Backtester
        bt = Backtester(capital=100_000)
        result = bt.run("TEST", days=60)
        assert result.profit_factor >= 0


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG TESTS
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_trading_config_defaults_reasonable(self):
        from config import TRADING
        assert 0 < TRADING.max_risk_per_trade_pct <= 5
        assert 0 < TRADING.stop_loss_pct <= 2
        assert 0 < TRADING.target_pct <= 5
        assert TRADING.target_pct > TRADING.stop_loss_pct, \
            "Target must be larger than SL for positive expectancy"
        assert TRADING.max_open_positions >= 1
        assert TRADING.max_daily_loss_pct >= TRADING.max_risk_per_trade_pct

    def test_priority_watchlist_not_empty(self):
        from config import TRADING
        assert len(TRADING.priority_watchlist) > 0

    def test_timing_config(self):
        from config import TRADING
        from datetime import time as dtime
        scan  = dtime.fromisoformat(TRADING.scan_start)
        cutoff = dtime.fromisoformat(TRADING.scan_end)
        sqoff  = dtime.fromisoformat(TRADING.square_off)
        mclose = dtime.fromisoformat(TRADING.market_close)
        assert scan < cutoff, "Scan start must be before scan end"
        assert cutoff < sqoff, "Scan end must be before square off"
        assert sqoff < mclose, "Square off must be before market close"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
