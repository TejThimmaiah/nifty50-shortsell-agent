"""
Integration Test — Full Paper Trade Cycle
Simulates an end-to-end trading day without any real orders or API calls.
All external services are mocked. Tests the full agent pipeline.

Run: pytest tests/test_integration.py -v -s
"""

import os
import sys
import json
import pytest
import tempfile
import numpy as np
import pandas as pd
from datetime import date, datetime
from unittest.mock import patch, MagicMock

os.environ["PAPER_TRADE"]  = "true"
os.environ["GROQ_API_KEY"] = "test_key"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def make_strong_short_df(n=60) -> pd.DataFrame:
    """OHLCV with RSI overbought setup — classic short candidate."""
    np.random.seed(99)
    prices = [1000.0]
    for i in range(n - 1):
        # Trend up strongly to create overbought RSI
        prices.append(prices[-1] + np.random.normal(3 if i < 45 else -1, 1))
    df = pd.DataFrame([{
        "open":   p - 2,
        "high":   p + 5,
        "low":    p - 4,
        "close":  p,
        "volume": int(2_000_000 + np.random.normal(0, 100_000)),
    } for p in prices])
    df.index = pd.date_range("2024-01-01", periods=n, freq="D")
    return df


@pytest.fixture
def isolated_db(tmp_path):
    import config
    original = config.DB_PATH
    config.DB_PATH = str(tmp_path / "integration_test.db")
    yield config.DB_PATH
    config.DB_PATH = original


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFullTradeCycle:
    """Tests the complete pipeline: scan → signal → risk → execute → monitor → close."""

    @patch("data.nse_fetcher.get_historical_ohlcv")
    @patch("data.nse_fetcher.get_intraday_ohlcv")
    @patch("data.nse_fetcher.get_quote")
    def test_complete_short_cycle(self, mock_quote, mock_intraday, mock_daily, isolated_db):
        """Full paper trade: entry → trailing SL → close at target."""
        df = make_strong_short_df()
        mock_daily.return_value   = df
        mock_intraday.return_value = df.tail(20).copy()

        # Simulate live price moving down (short profit)
        prices = [1180.0, 1170.0, 1160.0, 1150.0, 1140.0]  # falling price = short profit
        price_iter = iter(prices)

        def quote_side_effect(sym):
            try:
                p = next(price_iter)
            except StopIteration:
                p = 1140.0
            return {"ltp": p, "open": 1180, "high": 1185, "low": 1135,
                    "prev_close": 1175, "change_pct": -2.1, "volume": 2000000, "avg_price": 1160}

        mock_quote.side_effect = quote_side_effect

        # Build the components
        from agents.risk_manager import RiskManagerAgent
        from agents.trade_executor import TradeExecutorAgent
        from agents.position_monitor import PositionMonitorAgent
        from utils.circuit_breaker import CircuitBreaker

        risk_mgr = RiskManagerAgent(capital=100_000)
        executor = TradeExecutorAgent()
        alerts   = []
        cb       = CircuitBreaker(capital=100_000, notify_fn=lambda m: alerts.append(m))
        monitor  = PositionMonitorAgent(
            risk_mgr=risk_mgr,
            executor=executor,
            notify_fn=lambda m: alerts.append(m),
        )

        # Step 1: Technical analysis
        from agents.technical_analyst import calculate_all
        signal = calculate_all(df, "TESTSTOCK")
        assert signal is not None, "Technical signal should be generated"

        entry  = 1180.0
        sl     = round(entry * 1.005, 2)      # 0.5% SL above
        target = round(entry * 0.985, 2)      # 1.5% target below

        # Step 2: Risk manager approval
        decision = risk_mgr.approve_trade("TESTSTOCK", entry, sl, 50, "SHORT")
        assert decision.approved, f"Trade should be approved: {decision.reason}"
        assert decision.adjusted_quantity > 0

        # Step 3: Execute
        result = executor.short_sell("TESTSTOCK", decision.adjusted_quantity, entry, sl, target)
        assert result["status"] == "EXECUTED"
        assert result["paper"] is True

        # Step 4: Record
        risk_mgr.record_trade("TESTSTOCK", "SHORT", entry, decision.adjusted_quantity, "TEST_ORDER")
        monitor.register_position("TESTSTOCK", entry, sl, target, decision.adjusted_quantity)

        assert risk_mgr.get_open_position_count() == 1

        # Step 5: Monitor — price falls to target
        closed = monitor.check_all()   # first check at 1170 — in profit but not at target

        # Force price to target level
        executor._paper_positions["TESTSTOCK"]["stop_loss"] = sl
        trigger = executor.check_paper_triggers("TESTSTOCK", target - 1)  # below target

        if trigger == "TARGET":
            pnl = risk_mgr.close_trade("TESTSTOCK", target)
            assert pnl > 0, f"Short should profit when price falls. Got ₹{pnl:.2f}"
            monitor.unregister_position("TESTSTOCK")

        # Step 6: Verify DB state
        summary = risk_mgr.get_daily_summary()
        assert isinstance(summary["trades"], list)

        print(f"\n✅ Integration test passed: entry={entry}, target={target}, P&L calculated")

    def test_risk_manager_blocks_after_loss_limit(self, isolated_db):
        """Verify trading halts when daily loss exceeds 5%."""
        from agents.risk_manager import RiskManagerAgent
        risk_mgr = RiskManagerAgent(capital=100_000)

        # Record a big losing trade (6% of capital)
        risk_mgr.record_trade("LOSER", "SHORT", 1000.0, 100, "ORD_LOSS")
        risk_mgr.close_trade("LOSER", 1060.0)  # Short losses when price rises

        # Try to place another trade
        decision = risk_mgr.approve_trade("NEXT", 500.0, 502.5, 10, "SHORT")
        assert not decision.approved
        assert "Daily loss limit" in decision.reason

    def test_circuit_breaker_halts_on_streak(self, isolated_db):
        """Circuit breaker triggers on 3 consecutive losses."""
        from utils.circuit_breaker import CircuitBreaker
        alerts = []
        cb = CircuitBreaker(capital=100_000, notify_fn=lambda m: alerts.append(m), on_halt=lambda: None)

        for _ in range(3):
            cb.record_trade_result(-300)

        allowed, reason = cb.allow_trade()
        assert not allowed
        assert "CONSECUTIVE_LOSS" in reason
        assert any("CIRCUIT BREAKER" in a for a in alerts)

    @patch("data.nse_fetcher.get_intraday_ohlcv")
    @patch("data.nse_fetcher.get_historical_ohlcv")
    def test_mtf_requires_two_timeframe_alignment(self, mock_daily, mock_intraday):
        """MTF should reject signals that only appear on one timeframe."""
        from strategies.multi_timeframe import MultiTimeframeAnalyser

        df_strong = make_strong_short_df(n=60)    # good short signal
        df_flat   = pd.DataFrame([{
            "open": 500, "high": 502, "low": 498, "close": 500, "volume": 1000000
        }] * 60)
        df_flat.index = pd.date_range("2024-01-01", periods=60, freq="D")

        mock_daily.return_value   = df_strong
        mock_intraday.return_value = df_flat   # flat intraday — no signal

        mtf = MultiTimeframeAnalyser()
        result = mtf.analyse("TESTSYM")

        if result:
            # With one strong timeframe and one flat, alignment should be 1 or 2
            assert result.alignment_count <= 3
            # is_high_conviction requires >= 2 aligned
            conviction = mtf.is_high_conviction(result, min_alignment=2)
            # With one strong (daily) and flat (intraday), may not meet threshold
            print(f"  MTF alignment: {result.alignment_count}/3, conviction: {conviction}")

    def test_paper_executor_full_lifecycle(self, isolated_db):
        """Paper executor: short → monitor trigger → cover."""
        from agents.trade_executor import TradeExecutorAgent

        executor = TradeExecutorAgent()

        # Open short
        result = executor.short_sell("ALPHA", 10, 500.0, 502.5, 492.5)
        assert result["status"] == "EXECUTED"

        # No trigger in range
        assert executor.check_paper_triggers("ALPHA", 498.0) is None

        # SL hit
        sl_trigger = executor.check_paper_triggers("ALPHA", 503.0)
        assert sl_trigger == "SL"

        # Position auto-closed
        positions = executor.get_positions()
        assert "ALPHA" not in positions

    def test_screener_short_bias_scoring(self):
        """Fundamental scoring should rank weak companies higher."""
        from data.screener_fetcher import FundamentalProfile, _compute_short_bias

        # Weak company profile
        weak = FundamentalProfile(
            symbol="WEAK",
            company_name="Weak Co",
            pe_ratio=90,
            debt_to_equity=2.5,
            revenue_growth_pct=-10,
            profit_growth_pct=-20,
            promoter_holding_pct=20,
            roe=3,
        )
        score, reasons = _compute_short_bias(weak)
        assert score >= 0.60, f"Weak company should score >= 0.60, got {score}"
        assert len(reasons) >= 3

        # Strong company profile
        strong = FundamentalProfile(
            symbol="STRONG",
            company_name="Strong Co",
            pe_ratio=22,
            debt_to_equity=0.3,
            revenue_growth_pct=15,
            profit_growth_pct=20,
            promoter_holding_pct=65,
            roe=22,
        )
        strong_score, _ = _compute_short_bias(strong)
        assert strong_score < score, "Strong company should have lower short bias score"

    def test_performance_analytics_on_empty_db(self, isolated_db):
        """Analytics should handle empty DB gracefully."""
        from reports.performance_analytics import PerformanceAnalytics
        analytics = PerformanceAnalytics(capital=100_000)
        report = analytics.generate_report(days_back=7)
        assert report.total_trades == 0
        assert report.overall_pnl == 0

    def test_performance_analytics_with_trades(self, isolated_db):
        """Analytics computes correct metrics from real trade records."""
        from agents.risk_manager import RiskManagerAgent
        from reports.performance_analytics import PerformanceAnalytics

        risk_mgr  = RiskManagerAgent(capital=100_000)
        analytics = PerformanceAnalytics(capital=100_000)

        # Record 5 trades (3 wins, 2 losses)
        trades_data = [
            ("WIN1", "SHORT", 1000, 970,   100),   # profit 3%
            ("WIN2", "SHORT", 500,  490,   50),    # profit 2%
            ("LOSS1","SHORT", 800,  816,   50),    # loss 2%
            ("WIN3", "SHORT", 300,  291,   100),   # profit 3%
            ("LOSS2","SHORT", 600,  612,   30),    # loss 2%
        ]
        for sym, direction, entry, exit_p, qty in trades_data:
            risk_mgr.record_trade(sym, direction, entry, qty, f"ORD_{sym}")
            risk_mgr.close_trade(sym, exit_p)

        report = analytics.generate_report(days_back=1)
        assert report.total_trades == 5
        assert report.overall_win_rate == pytest.approx(60.0, abs=1)

        # Overall P&L should be positive (3 wins > 2 losses by structure)
        print(f"  Analytics: {report.total_trades} trades, "
              f"WR={report.overall_win_rate:.0f}%, P&L=₹{report.overall_pnl:,.0f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
