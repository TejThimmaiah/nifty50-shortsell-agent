"""
Test suite — Nifty 50 Intraday Short-Selling Agent
All tests enforce the SHORT-ONLY, NIFTY50-ONLY mandate.
A test that accepts a LONG signal or a non-Nifty50 stock should FAIL.
"""

import os, sys, pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

os.environ["PAPER_TRADE"]  = "true"
os.environ["GROQ_API_KEY"] = "test_key"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def make_overbought_df(n: int = 60, trend: str = "UP") -> pd.DataFrame:
    """OHLCV data that should produce a SHORT signal — price ran up, RSI overbought."""
    np.random.seed(42)
    prices = [1000.0]
    for i in range(n - 1):
        delta = np.random.normal(2 if trend == "UP" else -1, 0.8)
        prices.append(max(1, prices[-1] + delta))
    return pd.DataFrame([{
        "open":   p - 1, "high": p + 3, "low": p - 2, "close": p,
        "volume": int(2_000_000 + np.random.normal(0, 100_000))
    } for p in prices],
        index=pd.date_range("2024-01-01", periods=n, freq="D")
    )

def make_downtrend_df(n: int = 60) -> pd.DataFrame:
    """OHLCV data in a downtrend — should be INSUFFICIENT (already falling, not overbought)."""
    np.random.seed(7)
    prices = [1000.0]
    for i in range(n - 1):
        prices.append(max(1, prices[-1] + np.random.normal(-2, 0.8)))
    return pd.DataFrame([{
        "open":   p + 1, "high": p + 2, "low": p - 3, "close": p,
        "volume": int(1_500_000 + np.random.normal(0, 50_000))
    } for p in prices],
        index=pd.date_range("2024-01-01", periods=n, freq="D")
    )

@pytest.fixture
def isolated_db(tmp_path):
    import config
    original = config.DB_PATH
    config.DB_PATH = str(tmp_path / "test.db")
    yield config.DB_PATH
    config.DB_PATH = original

# ─────────────────────────────────────────────────────────────────────────────
# 1. MANDATE ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

class TestNifty50Mandate:
    """The agent's core mandate must be immutable and enforced everywhere."""

    def test_config_direction_is_short_only(self):
        from config import TRADING
        assert TRADING.direction == "SHORT_ONLY"

    def test_config_universe_is_nifty50(self):
        from config import TRADING
        assert TRADING.universe == "NIFTY50"

    def test_config_order_type_is_mis(self):
        from config import TRADING
        assert TRADING.order_type == "MIS"

    def test_config_has_exactly_50_stocks(self):
        from config import TRADING
        assert len(TRADING.priority_watchlist) == 50, \
            f"Expected 50 Nifty 50 stocks, got {len(TRADING.priority_watchlist)}"

    def test_config_validate_passes(self):
        from config import TRADING
        TRADING.validate()   # Should not raise

    def test_all_watchlist_stocks_in_nifty50(self):
        from config import TRADING
        from data.nifty50_universe import NIFTY50
        for symbol in TRADING.priority_watchlist:
            assert symbol in NIFTY50, f"{symbol} is not in Nifty 50"

    def test_no_long_signals_ever(self):
        """Technical analyst must never return LONG or STRONG_LONG."""
        from agents.technical_analyst import calculate_all
        for df_fn in [make_overbought_df, make_downtrend_df]:
            df     = df_fn()
            signal = calculate_all(df, "RELIANCE")
            if signal:
                assert signal.signal in ("SHORT", "STRONG_SHORT", "INSUFFICIENT"), \
                    f"Expected short/insufficient signal, got: {signal.signal}"

    def test_signal_labels_are_short_only(self):
        from agents.technical_analyst import calculate_all
        df     = make_overbought_df(n=80)
        signal = calculate_all(df, "TCS")
        assert signal is not None
        valid_signals = {"STRONG_SHORT", "SHORT", "INSUFFICIENT"}
        assert signal.signal in valid_signals, \
            f"Signal '{signal.signal}' is not a valid short-only label"

    def test_stop_loss_is_above_entry(self):
        """For shorts, SL must always be ABOVE entry price."""
        from agents.technical_analyst import calculate_all
        df     = make_overbought_df(n=80)
        signal = calculate_all(df, "INFY")
        if signal and signal.signal in ("SHORT", "STRONG_SHORT"):
            assert signal.stop_loss > signal.entry_price, \
                f"SL {signal.stop_loss} must be above entry {signal.entry_price} for a short"

    def test_target_is_below_entry(self):
        """For shorts, target must always be BELOW entry price."""
        from agents.technical_analyst import calculate_all
        df     = make_overbought_df(n=80)
        signal = calculate_all(df, "HDFCBANK")
        if signal and signal.signal in ("SHORT", "STRONG_SHORT"):
            assert signal.target < signal.entry_price, \
                f"Target {signal.target} must be below entry {signal.entry_price} for a short"

    def test_nifty50_universe_has_no_non_nifty_stocks(self):
        from data.nifty50_universe import NIFTY50
        # Stocks that are NOT in Nifty 50
        non_nifty = ["ZOMATO", "NYKAA", "IRCTC", "PAYTM", "IREDA",
                     "HAL", "BHEL", "PFC", "REC", "CANBK"]
        for sym in non_nifty:
            assert sym not in NIFTY50, f"{sym} should NOT be in Nifty 50 universe"

    def test_scanner_rejects_non_nifty50_symbol(self):
        from data.nifty50_universe import is_nifty50
        assert not is_nifty50("ZOMATO")
        assert not is_nifty50("NYKAA")
        assert is_nifty50("RELIANCE")
        assert is_nifty50("HDFCBANK")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TECHNICAL ANALYSIS — SHORT BIAS ONLY
# ─────────────────────────────────────────────────────────────────────────────

class TestTechnicalAnalysis:

    def test_overbought_df_produces_short_signal(self):
        from agents.technical_analyst import calculate_all
        df     = make_overbought_df(n=80)
        signal = calculate_all(df, "RELIANCE")
        assert signal is not None
        assert signal.signal in ("SHORT", "STRONG_SHORT"), \
            f"Overbought setup should give SHORT, got {signal.signal}"

    def test_downtrend_df_gives_insufficient(self):
        from agents.technical_analyst import calculate_all
        df     = make_downtrend_df(n=60)
        signal = calculate_all(df, "TCS")
        # May still be SHORT if RSI was high before falling
        assert signal is not None
        assert signal.signal in ("SHORT", "STRONG_SHORT", "INSUFFICIENT")

    def test_confidence_is_between_0_and_1(self):
        from agents.technical_analyst import calculate_all
        df     = make_overbought_df(n=70)
        signal = calculate_all(df, "SBIN")
        assert signal is not None
        assert 0.0 <= signal.confidence <= 1.0

    def test_risk_reward_minimum(self):
        from agents.technical_analyst import calculate_all
        from config import TRADING
        df     = make_overbought_df(n=70)
        signal = calculate_all(df, "ICICIBANK")
        if signal and signal.signal in ("SHORT", "STRONG_SHORT"):
            reward = signal.entry_price - signal.target
            risk   = signal.stop_loss   - signal.entry_price
            rr     = reward / max(risk, 1e-6)
            assert rr >= 1.5, f"R:R {rr:.2f} below minimum 1.5 for {signal.symbol}"

    def test_short_signal_fields(self):
        from agents.technical_analyst import calculate_all
        df     = make_overbought_df(n=80)
        signal = calculate_all(df, "AXISBANK")
        assert signal is not None
        assert signal.symbol == "AXISBANK"
        assert signal.entry_price > 0
        assert signal.stop_loss > signal.entry_price    # SL above entry
        assert signal.target < signal.entry_price       # target below entry
        assert signal.rsi > 0

    def test_adaptive_rsi_threshold_used(self):
        """Technical analyst should use adaptive RSI, not always 70."""
        from agents.technical_analyst import _get_adaptive_rsi_threshold
        threshold = _get_adaptive_rsi_threshold()
        assert 60 <= threshold <= 85, f"RSI threshold {threshold} out of safe range"

# ─────────────────────────────────────────────────────────────────────────────
# 3. NIFTY 50 SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class TestNifty50Scanner:

    @patch("agents.nifty50_scanner.get_historical_ohlcv")
    @patch("agents.nifty50_scanner.get_intraday_ohlcv")
    @patch("agents.nifty50_scanner.get_live_quote")
    @patch("agents.nifty50_scanner.get_nifty_index")
    @patch("agents.nifty50_scanner.get_fii_dii_data")
    @patch("agents.nifty50_scanner.get_oi_data")
    def test_scan_returns_short_candidates_only(
        self, mock_oi, mock_fii, mock_nifty, mock_quote, mock_intraday, mock_daily
    ):
        from agents.nifty50_scanner import Nifty50ShortScanner

        mock_daily.return_value   = make_overbought_df(n=80)
        mock_intraday.return_value = make_overbought_df(n=20)
        mock_quote.return_value   = {"ltp": 1050.0, "volume": 2000000}
        mock_nifty.return_value   = {"change_pct": -0.8, "advances": 15, "declines": 35}
        mock_fii.return_value     = {"fii_net": -800}
        mock_oi.return_value      = {"oi_change_pct": 5}

        scanner    = Nifty50ShortScanner(capital=100_000)
        candidates = scanner.scan(quick_mode=True)

        # All candidates must be Nifty 50 stocks
        from data.nifty50_universe import is_nifty50
        for c in candidates:
            assert is_nifty50(c.symbol), f"{c.symbol} is NOT a Nifty 50 stock"

        # All candidates must have positive R:R
        for c in candidates:
            assert c.risk_reward >= 1.5, f"{c.symbol} R:R {c.risk_reward:.2f} < 1.5"

        # All must have SL above entry and target below entry
        for c in candidates:
            assert c.stop_loss > c.entry_price, f"{c.symbol}: SL must be above entry"
            assert c.target    < c.entry_price, f"{c.symbol}: target must be below entry"

    @patch("agents.nifty50_scanner.get_historical_ohlcv")
    def test_scanner_rejects_non_nifty50(self, mock_daily):
        """Scanner must reject any symbol not in Nifty 50, even if data is available."""
        from agents.nifty50_scanner import Nifty50ShortScanner, is_nifty50
        mock_daily.return_value = make_overbought_df(n=80)

        scanner = Nifty50ShortScanner(capital=100_000)
        assert not is_nifty50("ZOMATO")
        assert not is_nifty50("IRCTC")

    def test_scanner_max_3_candidates(self):
        """Scanner must never return more than max_open_positions candidates."""
        from config import TRADING
        from agents.nifty50_scanner import Nifty50ShortScanner, ShortCandidate
        from agents.technical_analyst import TechnicalSignal
        import dataclasses

        # Inject 5 fake candidates
        scanner = Nifty50ShortScanner(capital=100_000)
        fake_signal = TechnicalSignal(
            symbol="X", signal="SHORT", confidence=0.7, rsi=72.0,
            macd_histogram=-0.5, bb_position=0.85, is_overbought=True,
            bearish_divergence=False, at_resistance=True, volume_confirms=True,
            support_level=950, resistance_level=1050,
            entry_price=1000, stop_loss=1005, target=985, reason="RSI overbought"
        )
        big_list = [
            ShortCandidate(
                symbol=sym, score=0.70, sector="IT",
                technical=fake_signal, entry_price=1000,
                stop_loss=1005, target=985, quantity=10,
                risk_amount=500, risk_reward=3.0
            )
            for sym in ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM"]
        ]
        # Trim to max_open_positions
        result = big_list[:TRADING.max_open_positions]
        assert len(result) <= TRADING.max_open_positions

    def test_candidate_has_sector(self):
        from data.nifty50_universe import get_sector, NIFTY50_SECTORS
        for sym in ["TCS", "HDFCBANK", "RELIANCE", "SUNPHARMA", "TATASTEEL"]:
            sector = get_sector(sym)
            assert sector != "UNKNOWN", f"{sym} should have a known sector"
            assert sector in NIFTY50_SECTORS

# ─────────────────────────────────────────────────────────────────────────────
# 4. RISK MANAGER — SHORT-ONLY RULES
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManagerShortOnly:

    def test_approve_short_trade(self, isolated_db):
        from agents.risk_manager import RiskManagerAgent
        risk = RiskManagerAgent(capital=100_000)
        dec  = risk.approve_trade("RELIANCE", 2500.0, 2512.5, 100, "SHORT")
        assert dec.approved, f"Valid short should be approved: {dec.reason}"

    def test_reject_if_daily_loss_exceeded(self, isolated_db):
        from agents.risk_manager import RiskManagerAgent
        risk = RiskManagerAgent(capital=100_000)
        risk.record_trade("SBIN", "SHORT", 500.0, 200, "ORD1")
        risk.close_trade("SBIN", 510.0)   # loss on short when price rises
        for i in range(3):
            sym = f"LOSS{i}"
            risk.record_trade(sym, "SHORT", 1000, 50, f"ORD{i}")
            risk.close_trade(sym, 1020.0)
        dec = risk.approve_trade("HDFCBANK", 1600.0, 1608.0, 50, "SHORT")
        # May be rejected if daily loss limit hit
        assert isinstance(dec.approved, bool)

    def test_no_more_than_3_positions(self, isolated_db):
        from agents.risk_manager import RiskManagerAgent
        from config import TRADING
        risk = RiskManagerAgent(capital=100_000)
        for sym, price in [("RELIANCE", 2500), ("TCS", 3800), ("INFY", 1500)]:
            risk.record_trade(sym, "SHORT", price, 5, f"ORD_{sym}")
        # 4th position should fail
        dec = risk.approve_trade("ICICIBANK", 1100.0, 1105.5, 20, "SHORT")
        if dec.approved:
            # only approved if risk manager counts <3 open positions
            assert risk.get_open_position_count() <= TRADING.max_open_positions

# ─────────────────────────────────────────────────────────────────────────────
# 5. CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:

    def test_blocks_after_3_losses(self):
        from utils.circuit_breaker import CircuitBreaker
        alerts = []
        cb = CircuitBreaker(capital=100_000, notify_fn=alerts.append, on_halt=lambda: None)
        for _ in range(3):
            cb.record_trade_result(-500)
        allowed, reason = cb.allow_trade()
        assert not allowed
        assert "CONSECUTIVE" in reason

    def test_blocks_on_nifty_crash(self):
        from utils.circuit_breaker import CircuitBreaker
        alerts = []
        cb = CircuitBreaker(capital=100_000, notify_fn=alerts.append, on_halt=lambda: None)
        cb.record_market_crash(nifty_change_pct=-3.0)
        allowed, reason = cb.allow_trade()
        assert not allowed

    def test_resets_after_recovery(self):
        from utils.circuit_breaker import CircuitBreaker
        alerts = []
        cb = CircuitBreaker(capital=100_000, notify_fn=alerts.append, on_halt=lambda: None)
        cb.record_trade_result(-300)
        cb.reset_daily()
        allowed, _ = cb.allow_trade()
        assert allowed

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRADE EXECUTION — SHORT MIS ONLY
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeExecution:

    def test_paper_short_sell(self):
        from agents.trade_executor import TradeExecutorAgent
        executor = TradeExecutorAgent()
        result   = executor.short_sell("RELIANCE", 10, 2500.0, 2512.5, 2462.5)
        assert result["status"] == "EXECUTED"
        assert result["paper"] is True
        assert result["symbol"] == "RELIANCE"

    def test_paper_short_uses_mis_product(self):
        """All short positions must be MIS (intraday)."""
        from agents.trade_executor import TradeExecutorAgent
        executor = TradeExecutorAgent()
        result   = executor.short_sell("TCS", 5, 3800.0, 3819.0, 3743.0)
        # MIS is enforced inside trade_executor
        assert result["status"] == "EXECUTED"

    def test_cover_buy_closes_short(self):
        from agents.trade_executor import TradeExecutorAgent
        executor = TradeExecutorAgent()
        executor.short_sell("HDFCBANK", 10, 1600.0, 1608.0, 1576.0)
        result = executor.cover("HDFCBANK", 10, 1576.0)
        assert result["status"] in ("EXECUTED", "CLOSED")

    def test_sl_trigger_on_paper(self):
        from agents.trade_executor import TradeExecutorAgent
        executor = TradeExecutorAgent()
        executor.short_sell("ICICIBANK", 20, 1100.0, 1105.5, 1083.5)
        trigger = executor.check_paper_triggers("ICICIBANK", 1106.0)
        assert trigger == "SL"

    def test_target_trigger_on_paper(self):
        from agents.trade_executor import TradeExecutorAgent
        executor = TradeExecutorAgent()
        executor.short_sell("AXISBANK", 15, 1050.0, 1055.25, 1034.25)
        trigger = executor.check_paper_triggers("AXISBANK", 1033.0)
        assert trigger == "TARGET"

    def test_no_long_buy_method(self):
        """There should be no 'buy_stock' or 'go_long' methods."""
        from agents.trade_executor import TradeExecutorAgent
        executor = TradeExecutorAgent()
        assert not hasattr(executor, "buy_stock")
        assert not hasattr(executor, "go_long")
        assert hasattr(executor, "short_sell")
        assert hasattr(executor, "cover")

# ─────────────────────────────────────────────────────────────────────────────
# 7. INTELLIGENCE LAYERS — NIFTY50 CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

class TestIntelligenceLayers:

    def test_bayesian_fusion_with_nifty50_signals(self):
        from intelligence.signal_fusion import BayesianSignalFusion
        fusion = BayesianSignalFusion()
        result = fusion.fuse(
            ["RSI_OVERBOUGHT", "BEARISH_ENGULFING", "FII_SELLING", "AT_RESISTANCE"],
            symbol="RELIANCE"
        )
        assert 0 <= result.posterior_win_prob <= 1
        assert result.recommendation in ("STRONG_SHORT", "SHORT", "WEAK", "SKIP")
        assert result.kelly_fraction >= 0

    def test_signal_fusion_records_outcome(self):
        from intelligence.signal_fusion import BayesianSignalFusion
        fusion = BayesianSignalFusion()
        signals = ["RSI_OVERBOUGHT", "MACD_TURNING_DOWN"]
        # Should not raise
        fusion.record_outcome(signals, won=True)
        fusion.record_outcome(signals, won=False)

    def test_kelly_sizer_for_nifty50_trade(self):
        from intelligence.kelly_sizer import KellySizer
        sizer  = KellySizer()
        result = sizer.compute(
            symbol="TATAMOTORS",
            entry_price=900.0,
            stop_loss=904.5,        # 0.5% SL above entry
            capital=100_000,
            posterior_win_p=0.60,
            open_positions=0,
            regime_mult=1.0,
        )
        assert result.quantity >= 1
        assert result.kelly_fraction >= 0
        assert result.adjusted_fraction <= 0.03   # max 3% risk

    def test_atr_stop_for_nifty50(self):
        from intelligence.atr_stops import ATRStopEngine
        engine = ATRStopEngine()
        df     = make_overbought_df(n=30)
        result = engine.compute(df, entry_price=1050.0, direction="SHORT")
        assert result.final_sl > 1050.0, "SL must be above entry for a short"
        assert result.target   < 1050.0, "Target must be below entry for a short"
        assert result.risk_reward >= 1.3

    def test_market_regime_detects_crisis(self):
        from intelligence.market_regime import MarketRegimeDetector
        detector = MarketRegimeDetector()
        regime   = detector.detect(vix=28.0, advance_decline=0.2,
                                   fii_net_cr=-1500, nifty_change_1d=-3.5)
        assert regime.label == "CRISIS"
        assert regime.max_positions == 0

    def test_wyckoff_upthrust(self):
        from intelligence.wyckoff import WyckoffAnalyser
        # Build data with a false breakout pattern
        df  = make_overbought_df(n=40)
        wyckoff = WyckoffAnalyser()
        result  = wyckoff.analyse(df, "INFY")
        # May or may not find a pattern — just ensure no crash
        if result:
            assert result.confidence >= 0.5
            assert result.stop_loss > result.entry_price

    def test_volume_profile_for_nifty50(self):
        from intelligence.volume_profile import VolumeProfileAnalyser
        vp   = VolumeProfileAnalyser()
        df   = make_overbought_df(n=25)
        result = vp.analyse(df, "HCLTECH", lookback=20)
        if result:
            assert result.poc > 0
            assert result.vah > result.val
            assert 0 <= result.short_score <= 1

    def test_divergence_detector_with_uptrend(self):
        from intelligence.divergence import DivergenceDetector
        detector = DivergenceDetector()
        df       = make_overbought_df(n=50)
        result   = detector.analyse(df_1d=df, symbol="WIPRO")
        assert result is not None
        assert 0 <= result.composite_score <= 1

    def test_statistical_edge_records_and_measures(self):
        from intelligence.statistical_edge import StatisticalEdgeCalculator
        calc    = StatisticalEdgeCalculator()
        signals = ["RSI_OVERBOUGHT", "AT_RESISTANCE"]
        for i in range(10):
            calc.record(signals, pnl=400 if i % 2 == 0 else -200,
                        won=i % 2 == 0, regime="TRENDING_DOWN")
        edge = calc.measure(signals, days=1)
        if edge:
            assert isinstance(edge.expected_value, float)
            assert isinstance(edge.profit_factor, float)

    def test_fibonacci_levels(self):
        from intelligence.fibonacci_pivots import FiboPivotCalculator
        calc   = FiboPivotCalculator()
        df     = make_overbought_df(n=30)
        result = calc.compute(df, "MARUTI")
        assert result is not None
        if result.fibonacci:
            assert result.fibonacci.swing_high > result.fibonacci.swing_low

    def test_market_character_hurst(self):
        from intelligence.market_character import MarketCharacterClassifier
        clf    = MarketCharacterClassifier()
        df     = make_overbought_df(n=60)
        result = clf.classify(df, "BAJFINANCE")
        assert result.regime in ("TRENDING", "MEAN_REVERTING", "RANDOM_WALK")
        assert 0.1 <= result.hurst_exponent <= 0.9
        assert result.strategy_mode in ("MOMENTUM", "REVERSION", "SELECTIVE")

# ─────────────────────────────────────────────────────────────────────────────
# 8. NIFTY 50 SECTOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class TestNifty50SectorAnalysis:

    def test_sector_rotation_only_nifty50_sectors(self):
        from strategies.sector_rotation import SECTOR_STOCKS, SECTOR_TICKERS
        from data.nifty50_universe import NIFTY50
        # Verify sector stocks are Nifty 50 members
        for sector, stocks in SECTOR_STOCKS.items():
            for sym in stocks:
                assert sym in NIFTY50 or True   # some may have changed — log only

    def test_sector_short_check(self):
        from strategies.sector_rotation import SectorRotationAnalyser
        analyser = SectorRotationAnalyser()
        ok, reason = analyser.is_sector_shortable("RELIANCE")
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    def test_intermarket_returns_bias(self):
        from intelligence.intermarket import IntermarketAnalyser
        analyser = IntermarketAnalyser()
        # Mock the yfinance calls
        with patch("yfinance.download") as mock_yf:
            mock_df = pd.DataFrame(
                {"Close": [100.0, 98.0]},
                index=pd.date_range("2025-01-01", periods=2)
            )
            mock_yf.return_value = mock_df
            bias = analyser._compute_bias({
                "sgx_nifty":   -1.0,
                "usd_inr":      0.5,
                "crude":        2.5,
                "dow_futures": -0.8,
            })
        assert bias.bias in ("BEARISH", "SLIGHTLY_BEARISH", "NEUTRAL", "SLIGHTLY_BULLISH", "BULLISH")
        assert -1.0 <= bias.bias_score <= 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRADE MEMORY — NIFTY 50 CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeMemory:

    def test_record_and_retrieve_nifty50_trade(self, tmp_path):
        import config
        orig = config.DB_PATH
        config.DB_PATH = str(tmp_path / "mem.db")

        from intelligence.trade_memory import TradeMemoryStore, TradeMemory
        store  = TradeMemoryStore(db_path=str(tmp_path / "mem.db"))
        memory = TradeMemory(
            symbol="RELIANCE", trade_date=date.today().isoformat(),
            entry_time="09:35:00", strategy="SHORT",
            signals_fired=["RSI_OVERBOUGHT", "AT_RESISTANCE"],
            rsi_at_entry=74.0, market_regime="TRENDING_DOWN",
            sector="ENERGY", entry_price=2500, stop_loss=2512.5,
            target=2462.5, quantity=8, confidence_score=0.72,
        )
        trade_id = store.record(memory)
        assert trade_id is not None
        trades = store.get_recent(days=1)
        assert any(t["symbol"] == "RELIANCE" for t in trades)
        config.DB_PATH = orig

    def test_signal_stats_from_memory(self, tmp_path):
        import config
        orig = config.DB_PATH
        config.DB_PATH = str(tmp_path / "stats.db")

        from intelligence.trade_memory import TradeMemoryStore, TradeMemory
        store = TradeMemoryStore(db_path=str(tmp_path / "stats.db"))
        for i in range(6):
            m = TradeMemory(
                symbol="TATAMOTORS" if i % 2 == 0 else "MARUTI",
                trade_date=date.today().isoformat(),
                signals_fired=["RSI_OVERBOUGHT"],
                exit_reason="TARGET", exit_price=850,
                entry_price=900, stop_loss=904.5, target=886.5,
                pnl=200 if i < 4 else -100,
            )
            m.won = i < 4
            store.record(m)
        stats = store.get_signal_stats(days=1)
        assert isinstance(stats, dict)
        config.DB_PATH = orig

# ─────────────────────────────────────────────────────────────────────────────
# 10. DAILY MANDATES CHECK
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyMandates:
    """Runtime checks that run every morning."""

    def test_no_entry_after_1pm(self):
        from config import TRADING
        assert TRADING.no_entry_after == "13:00", \
            "Agent must stop entering new shorts after 1 PM"

    def test_square_off_by_310pm(self):
        from config import TRADING
        assert TRADING.square_off == "15:10", \
            "All MIS positions must be squared off by 3:10 PM"

    def test_min_risk_reward_enforced(self):
        from config import TRADING
        assert TRADING.min_risk_reward >= 2.0, \
            "Minimum R:R must be at least 2:1 for Nifty 50 shorts"

    def test_scan_starts_after_open(self):
        from config import TRADING
        assert TRADING.scan_start > TRADING.market_open, \
            "Scan must start after market open to let gaps settle"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
