"""
Tick Streamer — Free Edition (₹0/month)
Delegates entirely to FreeTickStreamer in free_market_data.py.
Keeps the same external interface so no other files need to change.

OLD: Zerodha Kite WebSocket (₹500/month for market data)
NEW: NSE India public REST API + jugaad-data + nsetools (₹0/month)

Polling interval: 3 seconds — sufficient for 5-minute candle trading.
No API key, no subscription, no broker dependency for price data.
"""

# ── Re-export everything from the free implementation ─────────────────────────
from data.free_market_data import (
    Tick,
    FreeTickStreamer as TickStreamer,
    CandleBuilder,
    get_live_quote,
    get_intraday_ohlcv_free as get_intraday_ohlcv,
    get_daily_ohlcv_free    as get_daily_ohlcv,
    free_streamer,
)

# ── Backwards-compatible singleton (old code used this name) ──────────────────
# Any code that imported tick_streamer.free_streamer or tick_streamer.TickStreamer
# works without modification.

__all__ = [
    "Tick",
    "TickStreamer",
    "CandleBuilder",
    "get_live_quote",
    "get_intraday_ohlcv",
    "get_daily_ohlcv",
    "free_streamer",
]

# Instrument token map — populated from nifty50_universe
try:
    from data.nifty50_universe import NIFTY50_TOKENS as INSTRUMENT_TOKENS
except Exception:
    INSTRUMENT_TOKENS: dict = {}
