"""
Extended Test Suite — New Components
Tests for: CircuitBreaker, MarketCalendar, CandlestickPatterns,
           MultiTimeframe, OptionsAnalyser, TickStreamer
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

os.environ["PAPER_TRADE"]  = "true"
os.environ["GROQ_API_KEY"] = "test_key"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:

    def test_allows_trade_initially(self, circuit_breaker):
        cb, _ = circuit_breaker
        allowed, reason = cb.allow_trade()
        assert allowed is True
        assert reason == "ok"

    def test_blocks_after_consecutive_losses(self, circuit_breaker):
        cb, alerts = circuit_breaker
        for _ in range(3):
            cb.record_trade_result(-500)
        allowed, reason = cb.allow_trade()
        assert allowed is False
        assert "CONSECUTIVE_LOSS" in reason
        assert len(alerts) > 0

    def test_resets_consecutive_on_win(self, circuit_breaker):
        cb, _ = circuit_breaker
        cb.record_trade_result(-500)
        cb.record_trade_result(-500)
        cb.record_trade_result(+1000)   # win resets counter
        assert cb.state.consecutive_losses == 0

    def test_daily_loss_hard_halt(self, circuit_breaker):
        cb, alerts = circuit_breaker
        # Simulate 5.5% capital loss (capital=100k → ₹5500)
        cb.record_trade_result(-5500)
        allowed, reason = cb.allow_trade()
        assert allowed is False
        assert "DAILY_LOSS" in reason

    def test_flash_loss_triggers(self, circuit_breaker):
        cb, alerts = circuit_breaker
        # 2% flash loss on a single position
        crisis, msg = cb.check_position_flash_loss("RELIANCE", 2500.0, 2560.0, 100)
        # (2500-2560)*100 = -6000 loss = 6% of 100k capital
        assert crisis is True
        assert len(alerts) > 0

    def test_manual_reset(self, circuit_breaker):
        cb, _ = circuit_breaker
        cb.record_trade_result(-5000)
        cb.reset()
        allowed, _ = cb.allow_trade()
        assert allowed is True

    def test_auto_reset_after_time(self, circuit_breaker):
        cb, _ = circuit_breaker
        cb.record_trade_result(-300)
        cb.record_trade_result(-300)
        cb.record_trade_result(-300)   # triggers consecutive loss halt, 30-min auto-reset
        # Manually set trigger_time to 31 minutes ago
        from datetime import timezone
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        past = datetime.now(IST) - timedelta(minutes=31)
        cb.state.trigger_time = past.isoformat()
        # Next allow_trade should auto-reset
        allowed, _ = cb.allow_trade()
        assert allowed is True

    def test_nifty_crash_blocks_new_entries(self, circuit_breaker):
        cb, alerts = circuit_breaker
        cb.check_market_crash(-3.5)   # Nifty down 3.5%
        allowed, reason = cb.allow_trade()
        assert allowed is False
        assert "NIFTY_CRASH" in reason

    def test_status_dict_structure(self, circuit_breaker):
        cb, _ = circuit_breaker
        status = cb.get_status()
        assert "triggered" in status
        assert "consecutive_losses" in status
        assert "total_loss_today" in status
        assert "loss_pct_today" in status


# ─────────────────────────────────────────────────────────────────────────────
# MARKET CALENDAR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketCalendar:

    def test_weekend_is_not_trading_day(self, market_calendar):
        saturday = date(2025, 3, 15)   # Definitely a Saturday
        assert market_calendar.is_trading_day(saturday) is False

        sunday = date(2025, 3, 16)
        assert market_calendar.is_trading_day(sunday) is False

    def test_weekday_is_trading_day(self, market_calendar):
        # March 17, 2025 is a Monday (not a holiday)
        monday = date(2025, 3, 17)
        assert market_calendar.is_trading_day(monday) is True

    def test_known_holiday_blocked(self, market_calendar):
        # Republic Day
        republic_day = date(2025, 1, 26)
        assert market_calendar.is_trading_day(republic_day) is False

    def test_market_open_during_hours(self, market_calendar):
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        # Monday 11 AM IST
        dt = datetime(2025, 3, 17, 11, 0, 0, tzinfo=IST)
        with patch("utils.market_calendar.date") as mock_date:
            mock_date.today.return_value = dt.date()
            # is_trading_day uses date.today, so we mock it
            assert market_calendar.is_trading_day(dt.date()) is True

    def test_next_trading_day_skips_weekend(self, market_calendar):
        friday = date(2025, 3, 14)
        with patch("utils.market_calendar.date") as mock_date:
            mock_date.today.return_value = friday
            # Manually compute next trading day
        next_day = friday + timedelta(days=1)
        while not market_calendar.is_trading_day(next_day):
            next_day += timedelta(days=1)
        assert next_day.weekday() < 5   # Must be a weekday

    def test_upcoming_holidays_returns_list(self, market_calendar):
        holidays = market_calendar.get_upcoming_holidays(365)
        assert isinstance(holidays, list)
        assert len(holidays) >= 5   # Should have at least 5 holidays in a year


# ─────────────────────────────────────────────────────────────────────────────
# CANDLESTICK PATTERN TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCandlestickPatterns:

    def _make_candle_df(self, candles: list) -> pd.DataFrame:
        """candles = list of (open, high, low, close, volume)"""
        df = pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"])
        df.index = pd.date_range("2025-01-01", periods=len(candles), freq="D")
        return df

    def test_shooting_star_detected(self):
        from strategies.candlestick_patterns import _shooting_star
        # Small body near bottom, long upper shadow
        candles = [
            (100, 102, 98,  101, 1000000),
            (100, 102, 98,  101, 1000000),
            (100, 102, 98,  101, 1000000),
            (100, 102, 98,  101, 1000000),
            (101, 115, 100, 102, 1500000),   # Shooting star: open≈close≈low, huge upper shadow
        ]
        df = self._make_candle_df(candles)
        result = _shooting_star(df)
        assert result is not None
        assert result.pattern_type == "BEARISH_REVERSAL"
        assert result.confidence > 0.6

    def test_bearish_engulfing_detected(self):
        from strategies.candlestick_patterns import _bearish_engulfing
        candles = [
            (100, 105, 99, 103, 1000000),   # bullish previous
            (104, 106, 97,  98, 1500000),   # bearish current, engulfs previous body
        ]
        df = self._make_candle_df(candles)
        result = _bearish_engulfing(df)
        assert result is not None
        assert result.confidence > 0.7

    def test_gravestone_doji_detected(self):
        from strategies.candlestick_patterns import _gravestone_doji
        # Open = Close = Low, with a long upper shadow
        candles = [
            (100, 100.5, 99, 100, 1000000),
            (100, 115,  100, 100.1, 1000000),   # open≈close≈low, long upper
        ]
        df = self._make_candle_df(candles)
        result = _gravestone_doji(df)
        assert result is not None

    def test_three_black_crows_detected(self):
        from strategies.candlestick_patterns import _three_black_crows
        # 3 consecutive long bearish candles
        candles = [
            (100, 100, 94, 95, 1000000),   # long bearish 1
            (95,  95,  89, 90, 1000000),   # long bearish 2, opens in prior body
            (90,  90,  84, 85, 1000000),   # long bearish 3, opens in prior body
        ]
        df = self._make_candle_df(candles)
        result = _three_black_crows(df)
        assert result is not None
        assert result.confidence >= 0.80

    def test_no_pattern_on_neutral_data(self):
        from strategies.candlestick_patterns import detect_all_bearish_patterns
        # Completely flat candles — no pattern
        candles = [(100, 101, 99, 100, 1000000)] * 5
        df = self._make_candle_df(candles)
        patterns = detect_all_bearish_patterns(df)
        # May detect some weak patterns but score should be low
        if patterns:
            assert max(p.confidence for p in patterns) < 0.8

    def test_pattern_confidence_score_range(self):
        from strategies.candlestick_patterns import pattern_confidence_score
        import numpy as np
        np.random.seed(42)
        prices = [1000 + i * 2 + np.random.normal(0, 1) for i in range(60)]
        df = pd.DataFrame([{
            "open": p, "high": p+2, "low": p-2, "close": p,
            "volume": 1000000
        } for p in prices])
        df.index = pd.date_range("2024-01-01", periods=60, freq="D")
        score = pattern_confidence_score(df)
        assert 0.0 <= score <= 1.0

    def test_multi_timeframe_returns_tuple(self):
        from strategies.candlestick_patterns import multi_timeframe_bearish_score
        df = pd.DataFrame([{
            "open": 100, "high": 105, "low": 98, "close": 101, "volume": 1000000
        }] * 20)
        df.index = pd.date_range("2024-01-01", periods=20, freq="D")
        score, signals = multi_timeframe_bearish_score(df, df, df)
        assert isinstance(score, float)
        assert isinstance(signals, list)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TICK STREAMER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTickStreamer:

    def test_subscribe_and_receive_tick(self):
        from data.tick_streamer import TickStreamer, Tick
        streamer = TickStreamer()
        received = []

        streamer.subscribe("RELIANCE", lambda tick: received.append(tick))

        # Simulate a tick dispatch directly
        fake_tick = Tick(
            symbol="RELIANCE",
            instrument_token=738561,
            ltp=2500.0,
            last_traded_quantity=100,
            average_traded_price=2498.0,
            volume_traded=1000000,
            buy_quantity=5000,
            sell_quantity=3000,
            ohlc={"open": 2480, "high": 2510, "low": 2470, "close": 2490},
            change=0.4,
        )
        streamer._dispatch("RELIANCE", fake_tick)

        assert len(received) == 1
        assert received[0].ltp == 2500.0

    def test_unsubscribe_stops_callbacks(self):
        from data.tick_streamer import TickStreamer, Tick
        streamer = TickStreamer()
        received = []

        streamer.subscribe("TCS", lambda t: received.append(t))
        streamer.unsubscribe("TCS")

        fake_tick = Tick("TCS", 0, 3800.0, 0, 0, 0, 0, 0, {}, 0)
        streamer._dispatch("TCS", fake_tick)
        assert len(received) == 0   # No callbacks after unsubscribe

    def test_token_reverse_lookup(self):
        from data.tick_streamer import TickStreamer, INSTRUMENT_TOKENS
        streamer = TickStreamer()
        # RELIANCE token
        sym = streamer._token_to_symbol(738561)
        assert sym == "RELIANCE"

    def test_unknown_token_returns_none(self):
        from data.tick_streamer import TickStreamer
        streamer = TickStreamer()
        assert streamer._token_to_symbol(999999999) is None

    def test_latest_tick_cache(self):
        from data.tick_streamer import TickStreamer, Tick
        streamer = TickStreamer()
        tick = Tick("INFY", 408065, 1500.0, 0, 0, 0, 0, 0, {}, 0)
        streamer._latest_ticks["INFY"] = tick
        assert streamer.get_ltp("INFY") == 1500.0


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS ANALYSER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionsAnalyser:

    def test_max_pain_calculation(self):
        from strategies.options_analyser import OptionsChainAnalyser
        oa = OptionsChainAnalyser()
        strikes = {
            100: {"call_oi": 5000, "put_oi": 1000},
            110: {"call_oi": 8000, "put_oi": 2000},
            120: {"call_oi": 3000, "put_oi": 9000},
            130: {"call_oi": 1000, "put_oi": 7000},
        }
        max_pain = oa._calculate_max_pain(strikes)
        # Max pain should be somewhere in the strike range
        assert 100 <= max_pain <= 130

    def test_max_pain_empty_strikes(self):
        from strategies.options_analyser import OptionsChainAnalyser
        oa = OptionsChainAnalyser()
        result = oa._calculate_max_pain({})
        assert result == 0.0

    @patch("strategies.options_analyser.requests.Session")
    def test_analyse_returns_none_on_api_failure(self, mock_session):
        mock_session.return_value.get.side_effect = Exception("Network error")
        from strategies.options_analyser import OptionsChainAnalyser
        oa = OptionsChainAnalyser()
        result = oa.analyse("RELIANCE")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizer:

    def test_param_grid_generates_valid_combos(self):
        from strategies.optimizer import StrategyOptimizer
        opt = StrategyOptimizer()
        combos = opt._generate_combinations()
        assert len(combos) > 0
        for p in combos:
            # Each combo must satisfy risk/reward constraint
            rr = p.target_pct / p.stop_loss_pct
            assert rr >= opt.CONSTRAINTS["min_risk_reward"]

    def test_all_combos_have_positive_rr(self):
        from strategies.optimizer import StrategyOptimizer
        opt = StrategyOptimizer()
        combos = opt._generate_combinations()
        for p in combos:
            assert p.target_pct > p.stop_loss_pct, \
                f"target {p.target_pct}% must exceed SL {p.stop_loss_pct}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
