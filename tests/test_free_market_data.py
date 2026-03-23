"""
Tests — Free Market Data Engine
Validates the zero-cost NSE data stack: NSE API, OpenChart, jugaad-data, candle builder.
"""

import os
import sys
import time
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

os.environ["PAPER_TRADE"]  = "true"
os.environ["GROQ_API_KEY"] = "test_key"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# TICK DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestTickStructure:

    def test_tick_defaults(self):
        from data.free_market_data import Tick
        tick = Tick(symbol="RELIANCE")
        assert tick.symbol == "RELIANCE"
        assert tick.ltp == 0.0
        assert tick.volume_traded == 0
        assert tick.source == "nse_api"
        assert tick.ohlc == {}

    def test_tick_with_values(self):
        from data.free_market_data import Tick
        tick = Tick(
            symbol="TCS",
            ltp=3800.0,
            volume_traded=500000,
            change_pct=1.5,
            ohlc={"open": 3750, "high": 3820, "low": 3740, "close": 3790},
        )
        assert tick.ltp == 3800.0
        assert tick.ohlc["high"] == 3820


# ─────────────────────────────────────────────────────────────────────────────
# FREE TICK STREAMER
# ─────────────────────────────────────────────────────────────────────────────

class TestFreeTickStreamer:

    def test_subscribe_and_dispatch(self):
        from data.free_market_data import FreeTickStreamer, Tick
        streamer = FreeTickStreamer()
        received = []

        streamer.subscribe("RELIANCE", lambda t: received.append(t))
        tick = Tick(symbol="RELIANCE", ltp=2500.0, volume_traded=1_000_000)
        streamer._dispatch("RELIANCE", tick)

        assert len(received) == 1
        assert received[0].ltp == 2500.0

    def test_unsubscribe_stops_dispatch(self):
        from data.free_market_data import FreeTickStreamer, Tick
        streamer = FreeTickStreamer()
        received = []

        streamer.subscribe("TCS", lambda t: received.append(t))
        streamer.unsubscribe("TCS")

        tick = Tick(symbol="TCS", ltp=3800.0)
        streamer._dispatch("TCS", tick)
        assert len(received) == 0

    def test_multiple_subscribers_same_symbol(self):
        from data.free_market_data import FreeTickStreamer, Tick
        streamer = FreeTickStreamer()
        cb1_recv, cb2_recv = [], []

        streamer.subscribe("INFY", lambda t: cb1_recv.append(t))
        streamer.subscribe("INFY", lambda t: cb2_recv.append(t))

        tick = Tick(symbol="INFY", ltp=1500.0)
        streamer._dispatch("INFY", tick)

        assert len(cb1_recv) == 1
        assert len(cb2_recv) == 1

    def test_get_ltp_from_cache(self):
        from data.free_market_data import FreeTickStreamer, Tick
        streamer = FreeTickStreamer()
        tick = Tick(symbol="SBIN", ltp=800.0)
        streamer._latest_ticks["SBIN"] = tick

        assert streamer.get_ltp("SBIN") == 800.0

    def test_stats_structure(self):
        from data.free_market_data import FreeTickStreamer
        streamer = FreeTickStreamer()
        stats = streamer.get_stats()
        assert "polls" in stats
        assert "errors" in stats
        assert "ticks_dispatched" in stats
        assert "cost" in stats
        assert stats["cost"] == "₹0"

    def test_start_stop(self):
        from data.free_market_data import FreeTickStreamer
        streamer = FreeTickStreamer()
        streamer.start()
        assert streamer._running is True
        streamer.stop()
        assert streamer._running is False

    @patch("data.free_market_data.get_live_quote")
    def test_fetch_batch_dispatches_ticks(self, mock_quote):
        from data.free_market_data import FreeTickStreamer
        mock_quote.return_value = {
            "ltp": 2500.0, "open": 2480, "high": 2520, "low": 2470,
            "prev_close": 2490, "volume": 1000000, "change": 10,
            "change_pct": 0.4, "vwap": 2495, "source": "nse_api",
        }

        streamer = FreeTickStreamer()
        received = []
        streamer.subscribe("RELIANCE", lambda t: received.append(t))
        streamer._fetch_batch(["RELIANCE"])

        assert len(received) == 1
        assert received[0].ltp == 2500.0
        assert received[0].ohlc["open"] == 2480

    @patch("data.free_market_data.get_live_quote")
    def test_zero_ltp_not_dispatched(self, mock_quote):
        """Ticks with LTP=0 should be ignored (bad data)."""
        from data.free_market_data import FreeTickStreamer
        mock_quote.return_value = {"ltp": 0, "source": "nse_api"}

        streamer = FreeTickStreamer()
        received = []
        streamer.subscribe("BADSTOCK", lambda t: received.append(t))
        streamer._fetch_batch(["BADSTOCK"])

        assert len(received) == 0


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleBuilder:

    def _make_tick(self, ltp: float, minute_offset: int = 0, volume: int = 100_000):
        from data.free_market_data import Tick
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        ts = datetime(2025, 3, 10, 9, 20, 0, tzinfo=IST) + timedelta(minutes=minute_offset, seconds=1)
        return Tick(symbol="TEST", ltp=ltp, volume_traded=volume, timestamp=ts)

    def test_builds_first_candle(self):
        from data.free_market_data import CandleBuilder
        builder = CandleBuilder("TEST", interval_min=5)
        builder.on_tick(self._make_tick(100.0, 0))
        builder.on_tick(self._make_tick(102.0, 1))
        builder.on_tick(self._make_tick(99.0,  2))

        assert builder._current is not None
        assert builder._current["open"]  == 100.0
        assert builder._current["high"]  == 102.0
        assert builder._current["low"]   == 99.0
        assert builder._current["close"] == 99.0

    def test_closes_candle_at_interval(self):
        from data.free_market_data import CandleBuilder
        completed = []
        builder = CandleBuilder("TEST", interval_min=5, on_candle=completed.append)

        builder.on_tick(self._make_tick(100.0, 0))   # candle 9:20
        builder.on_tick(self._make_tick(105.0, 5))   # starts new candle 9:25 → closes previous

        assert len(completed) == 1
        assert completed[0]["open"]  == 100.0
        assert completed[0]["close"] == 100.0

    def test_volume_accumulates(self):
        from data.free_market_data import CandleBuilder
        builder = CandleBuilder("TEST", interval_min=5)
        builder.on_tick(self._make_tick(100.0, 0, volume=50_000))
        builder.on_tick(self._make_tick(101.0, 1, volume=50_000))
        assert builder._current["volume"] == 100_000

    def test_get_dataframe_returns_completed(self):
        from data.free_market_data import CandleBuilder
        builder = CandleBuilder("TEST", interval_min=5)
        builder.on_tick(self._make_tick(100.0, 0))
        builder.on_tick(self._make_tick(105.0, 5))   # closes first candle

        df = builder.get_dataframe()
        assert df is not None
        assert len(df) == 1
        assert "open" in df.columns

    def test_high_low_tracked_correctly(self):
        from data.free_market_data import CandleBuilder
        builder = CandleBuilder("TEST", interval_min=5)
        for price in [100, 115, 95, 108]:
            builder.on_tick(self._make_tick(price, 0))
        assert builder._current["high"] == 115
        assert builder._current["low"]  == 95


# ─────────────────────────────────────────────────────────────────────────────
# NSE SESSION
# ─────────────────────────────────────────────────────────────────────────────

class TestNSESession:

    @patch("data.free_market_data.requests.Session")
    def test_quote_parse(self, mock_session_cls):
        from data.free_market_data import _NSESession

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "priceInfo": {
                "lastPrice": 2500.0,
                "open": 2480.0,
                "change": 20.0,
                "pChange": 0.8,
                "previousClose": 2480.0,
                "vwap": 2495.0,
                "intraDayHighLow": {"max": 2520.0, "min": 2470.0},
            },
            "marketDeptOrderBook": {
                "tradeInfo": {"totalTradedVolume": 1_000_000}
            }
        }
        mock_session_cls.return_value.get.return_value = mock_resp

        session = _NSESession()
        data = session.get_quote("RELIANCE")
        assert data is not None
        assert data["priceInfo"]["lastPrice"] == 2500.0


# ─────────────────────────────────────────────────────────────────────────────
# FREE QUOTE FALLBACK CHAIN
# ─────────────────────────────────────────────────────────────────────────────

class TestFreeQuoteFallback:

    @patch("data.free_market_data._nse_session")
    def test_primary_nse_api_returns_quote(self, mock_nse):
        mock_nse.get_quote.return_value = {
            "priceInfo": {
                "lastPrice": 2500.0, "open": 2480, "change": 20,
                "pChange": 0.8, "previousClose": 2480, "vwap": 2495,
                "intraDayHighLow": {"max": 2520, "min": 2470},
            },
            "marketDeptOrderBook": {"tradeInfo": {"totalTradedVolume": 1_000_000}},
        }

        from data.free_market_data import get_live_quote
        result = get_live_quote("RELIANCE")

        assert result is not None
        assert result["ltp"] == 2500.0
        assert result["source"] == "nse_api"
        assert result["volume"] == 1_000_000

    @patch("data.free_market_data._nse_session")
    @patch("data.free_market_data._jugaad_quote")
    def test_falls_back_to_jugaad(self, mock_jugaad, mock_nse):
        mock_nse.get_quote.return_value = None   # Primary fails
        mock_jugaad.return_value = {
            "symbol": "TCS", "ltp": 3800.0, "open": 3750,
            "high": 3820, "low": 3740, "prev_close": 3780,
            "change_pct": 0.5, "volume": 0,
        }

        from data.free_market_data import get_live_quote
        result = get_live_quote("TCS")

        assert result is not None
        assert result["ltp"] == 3800.0
        assert result["source"] == "jugaad"

    @patch("data.free_market_data._nse_session")
    @patch("data.free_market_data._jugaad_quote")
    @patch("data.free_market_data._nsetools_quote")
    def test_falls_back_to_nsetools(self, mock_nsetools, mock_jugaad, mock_nse):
        mock_nse.get_quote.return_value   = None
        mock_jugaad.return_value          = None
        mock_nsetools.return_value        = {
            "symbol": "INFY", "ltp": 1500.0, "open": 1480,
            "high": 1520, "low": 1470, "prev_close": 1490,
            "change_pct": 0.7, "volume": 0,
        }

        from data.free_market_data import get_live_quote
        result = get_live_quote("INFY")

        assert result is not None
        assert result["ltp"] == 1500.0
        assert result["source"] == "nsetools"

    @patch("data.free_market_data._nse_session")
    @patch("data.free_market_data._jugaad_quote")
    @patch("data.free_market_data._nsetools_quote")
    def test_returns_none_when_all_fail(self, mock_nsetools, mock_jugaad, mock_nse):
        mock_nse.get_quote.return_value = None
        mock_jugaad.return_value        = None
        mock_nsetools.return_value      = None

        from data.free_market_data import get_live_quote
        result = get_live_quote("UNKNOWN")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL OHLCV
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoricalOHLCV:

    @patch("data.free_market_data._openchart_fetch")
    def test_openchart_primary(self, mock_oc):
        import numpy as np
        df = pd.DataFrame(
            {"open": [100], "high": [105], "low": [98], "close": [103], "volume": [500_000]},
            index=pd.date_range("2025-03-10 09:20", periods=1, freq="5min")
        )
        mock_oc.return_value = df

        from data.free_market_data import get_intraday_ohlcv_free
        result = get_intraday_ohlcv_free("RELIANCE", "5m", days=1)

        assert result is not None
        assert len(result) == 1
        mock_oc.assert_called_once()

    @patch("data.free_market_data._openchart_fetch")
    @patch("data.free_market_data._yfinance_intraday")
    def test_yfinance_fallback_on_openchart_fail(self, mock_yf, mock_oc):
        mock_oc.return_value = None   # OpenChart fails

        df = pd.DataFrame(
            {"open": [200], "high": [210], "low": [195], "close": [205], "volume": [300_000]},
            index=pd.date_range("2025-03-10 09:20", periods=1, freq="5min")
        )
        mock_yf.return_value = df

        from data.free_market_data import get_intraday_ohlcv_free
        result = get_intraday_ohlcv_free("TCS", "5m", days=1)

        assert result is not None
        mock_yf.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TICK STREAMER BACKWARDS COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────

class TestTickStreamerCompat:

    def test_imports_work(self):
        """tick_streamer.py must expose same interface as before."""
        from data.tick_streamer import (
            Tick, TickStreamer, CandleBuilder,
            get_live_quote, get_intraday_ohlcv, get_daily_ohlcv,
            free_streamer, INSTRUMENT_TOKENS,
        )
        assert Tick is not None
        assert TickStreamer is not None
        assert CandleBuilder is not None
        assert get_live_quote is not None
        assert isinstance(INSTRUMENT_TOKENS, dict)

    def test_tick_streamer_is_free_implementation(self):
        from data.tick_streamer import TickStreamer
        from data.free_market_data import FreeTickStreamer
        assert TickStreamer is FreeTickStreamer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
