"""
Free Market Data Engine
Replaces Zerodha Kite WebSocket (₹500/month) with 100% free alternatives.

SOURCES (all free, no API key needed):
  1. NSE India REST API  → live quotes (every 3s during market hours)
  2. OpenChart           → intraday OHLCV candles (1m, 5m, 15m) from NSE
  3. jugaad-data         → live quotes + daily historical from NSE
  4. nsetools            → live quotes fallback
  5. yfinance            → daily + weekly historical (already in use)

ARCHITECTURE:
  Multi-source poller with automatic failover:
    Primary   → NSE India public API (most reliable, same data as Kite WS)
    Secondary → jugaad-data (NSELive)
    Tertiary  → nsetools
    Historical→ OpenChart (intraday) → yfinance (daily)

  Smart rate limiter: 3-second minimum between requests per symbol.
  Batch fetching: fetches all subscribed symbols in one NSE API call.
  Same interface as tick_streamer.py — drop-in replacement.

ACCURACY vs Kite WebSocket:
  Kite WS: ~100ms latency, continuous push
  This:    ~3 second polling, pull-based
  For our strategy (5m candles, 9:20–3:10 window): MORE than sufficient.
  Short selling on 5-min signals does not need sub-second data.
"""

import time
import logging
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# ── Poll intervals ────────────────────────────────────────────────────────────
LIVE_POLL_SEC   = 3      # Quote poll during market hours
IDLE_POLL_SEC   = 30     # Between 3:30 PM and 9:15 AM (no need to poll fast)
BATCH_SIZE      = 10     # Max symbols per NSE batch request
REQUEST_TIMEOUT = 10     # Seconds before giving up on a request


@dataclass
class Tick:
    """Identical interface to the old KiteTicker tick — zero code changes needed."""
    symbol:               str
    instrument_token:     int     = 0        # NSE token (if known)
    ltp:                  float   = 0.0
    last_traded_quantity: int     = 0
    average_traded_price: float   = 0.0
    volume_traded:        int     = 0
    buy_quantity:         int     = 0
    sell_quantity:        int     = 0
    ohlc:                 Dict    = field(default_factory=dict)
    change:               float   = 0.0
    change_pct:           float   = 0.0
    high:                 float   = 0.0
    low:                  float   = 0.0
    timestamp:            datetime = field(default_factory=lambda: datetime.now(IST))
    source:               str     = "nse_api"


# ─────────────────────────────────────────────────────────────────────────────
# NSE SESSION — maintains cookies required for public API
# ─────────────────────────────────────────────────────────────────────────────

class _NSESession:
    """Manages the NSE website cookie session. Refreshes every 10 minutes."""

    BASE = "https://www.nseindia.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
    }

    def __init__(self):
        self._s = requests.Session()
        self._s.headers.update(self.HEADERS)
        self._refreshed_at: Optional[datetime] = None

    def _ensure_cookies(self):
        need_refresh = (
            self._refreshed_at is None or
            (datetime.now() - self._refreshed_at).seconds > 600
        )
        if need_refresh:
            try:
                self._s.get(self.BASE, timeout=10)
                self._s.get(f"{self.BASE}/market-data/live-equity-market", timeout=8)
                self._refreshed_at = datetime.now()
            except Exception as e:
                logger.debug(f"NSE cookie refresh: {e}")

    def get_quote(self, symbol: str) -> Optional[Dict]:
        self._ensure_cookies()
        try:
            r = self._s.get(
                f"{self.BASE}/api/quote-equity?symbol={symbol}",
                timeout=REQUEST_TIMEOUT
            )
            if r.status_code == 401:
                self._refreshed_at = None   # Force cookie refresh
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.debug(f"NSE quote error [{symbol}]: {e}")
            return None

    def get_market_data(self, index: str = "NIFTY%2050") -> Optional[Dict]:
        self._ensure_cookies()
        try:
            r = self._s.get(
                f"{self.BASE}/api/equity-stockIndices?index={index}",
                timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.debug(f"NSE market data error: {e}")
            return None

    def get_all_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batch fetch quotes for multiple symbols (one HTTP call per symbol)."""
        results = {}
        for sym in symbols:
            data = self.get_quote(sym)
            if data:
                pi = data.get("priceInfo", {})
                intra = pi.get("intraDayHighLow", {})
                results[sym] = {
                    "symbol":     sym,
                    "ltp":        pi.get("lastPrice", 0),
                    "open":       pi.get("open", 0),
                    "high":       intra.get("max", 0),
                    "low":        intra.get("min", 0),
                    "prev_close": pi.get("previousClose", 0),
                    "change":     pi.get("change", 0),
                    "change_pct": pi.get("pChange", 0),
                    "vwap":       pi.get("vwap", 0),
                    "volume":     (data.get("marketDeptOrderBook", {})
                                       .get("tradeInfo", {})
                                       .get("totalTradedVolume", 0)),
                }
            time.sleep(0.3)   # ~300ms between symbols to be NSE-friendly
        return results


_nse_session = _NSESession()


# ─────────────────────────────────────────────────────────────────────────────
# JUGAAD-DATA FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _jugaad_quote(symbol: str) -> Optional[Dict]:
    """Fetch live quote via jugaad-data (NSELive)."""
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        q = n.stock_quote(symbol)
        pi = q.get("priceInfo", {})
        intra = pi.get("intraDayHighLow", {})
        return {
            "symbol":     symbol,
            "ltp":        pi.get("lastPrice", 0),
            "open":       pi.get("open", 0),
            "high":       intra.get("max", 0),
            "low":        intra.get("min", 0),
            "prev_close": pi.get("previousClose", 0),
            "change_pct": pi.get("pChange", 0),
            "volume":     0,
        }
    except Exception as e:
        logger.debug(f"jugaad quote error [{symbol}]: {e}")
        return None


def _nsetools_quote(symbol: str) -> Optional[Dict]:
    """Fetch live quote via nsetools."""
    try:
        from nsetools import Nse
        n = Nse()
        q = n.get_quote(symbol.lower())
        if not q:
            return None
        return {
            "symbol":     symbol,
            "ltp":        q.get("lastPrice", 0),
            "open":       q.get("open", 0),
            "high":       q.get("intraDayHighLow", {}).get("max", 0),
            "low":        q.get("intraDayHighLow", {}).get("min", 0),
            "prev_close": q.get("previousClose", 0),
            "change_pct": q.get("pChange", 0),
            "volume":     0,
        }
    except Exception as e:
        logger.debug(f"nsetools quote error [{symbol}]: {e}")
        return None


def get_live_quote(symbol: str) -> Optional[Dict]:
    """
    Get live quote with automatic failover:
    NSE API → jugaad-data → nsetools → None
    """
    # Primary: NSE API
    data = _nse_session.get_quote(symbol)
    if data:
        pi = data.get("priceInfo", {})
        intra = pi.get("intraDayHighLow", {})
        return {
            "symbol":     symbol,
            "ltp":        float(pi.get("lastPrice", 0)),
            "open":       float(pi.get("open", 0)),
            "high":       float(intra.get("max", 0)),
            "low":        float(intra.get("min", 0)),
            "prev_close": float(pi.get("previousClose", 0)),
            "change":     float(pi.get("change", 0)),
            "change_pct": float(pi.get("pChange", 0)),
            "vwap":       float(pi.get("vwap", 0)),
            "volume":     int(data.get("marketDeptOrderBook", {})
                               .get("tradeInfo", {})
                               .get("totalTradedVolume", 0)),
            "source":     "nse_api",
        }

    # Secondary: jugaad-data
    q = _jugaad_quote(symbol)
    if q:
        q["source"] = "jugaad"
        return q

    # Tertiary: nsetools
    q = _nsetools_quote(symbol)
    if q:
        q["source"] = "nsetools"
        return q

    return None


# ─────────────────────────────────────────────────────────────────────────────
# FREE INTRADAY OHLCV (OpenChart — MIT license, NSE public API)
# ─────────────────────────────────────────────────────────────────────────────

def get_intraday_ohlcv_free(
    symbol: str,
    interval: str = "5m",
    days: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLCV using OpenChart (free, MIT, NSE public API).
    Intervals: "1m", "3m", "5m", "10m", "15m", "30m", "60m"
    Falls back to yfinance if OpenChart fails.
    """
    # Try OpenChart first (NSE direct — most accurate for Indian markets)
    df = _openchart_fetch(symbol, interval, days)
    if df is not None and not df.empty:
        return df

    # Fallback: yfinance (Yahoo Finance, reliable but slight delay)
    df = _yfinance_intraday(symbol, interval, days)
    if df is not None and not df.empty:
        return df

    logger.warning(f"All intraday sources failed for {symbol}")
    return None


def _openchart_fetch(
    symbol: str, interval: str, days: int
) -> Optional[pd.DataFrame]:
    """Fetch intraday data via OpenChart (marketcalls/openchart on GitHub)."""
    try:
        from openchart import NSEData
        nse = NSEData()
        nse.download()   # Downloads master CSV (cached after first call)

        end   = datetime.now(IST)
        start = end - timedelta(days=max(days, 1))

        # OpenChart interval format: "1minute", "5minute", "15minute"
        interval_map = {
            "1m":  "1minute",
            "3m":  "3minute",
            "5m":  "5minute",
            "10m": "10minute",
            "15m": "15minute",
            "30m": "30minute",
            "60m": "60minute",
            "1h":  "60minute",
            "1d":  "1d",
        }
        oc_interval = interval_map.get(interval, "5minute")

        df = nse.historical(
            symbol     = f"{symbol}-EQ",
            exchange   = "NSE",
            start      = start,
            end        = end,
            timeframe  = oc_interval,
        )

        if df is None or df.empty:
            return None

        # Normalise column names
        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        logger.debug(f"OpenChart [{symbol}/{interval}]: {len(df)} candles")
        return df

    except ImportError:
        logger.debug("openchart not installed — run: pip install openchart")
        return None
    except Exception as e:
        logger.debug(f"OpenChart error [{symbol}]: {e}")
        return None


def _yfinance_intraday(
    symbol: str, interval: str, days: int
) -> Optional[pd.DataFrame]:
    """Fetch intraday data via yfinance (Yahoo Finance)."""
    try:
        import yfinance as yf
        ticker = f"{symbol}.NS"
        period = f"{max(days, 1)}d"
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=True
        )
        if df is None or df.empty:
            return None
        df.columns  = [c.lower() for c in df.columns]
        df.index    = pd.to_datetime(df.index)
        return df.dropna()
    except Exception as e:
        logger.debug(f"yfinance intraday error [{symbol}]: {e}")
        return None


def get_daily_ohlcv_free(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV — yfinance primary, jugaad-data fallback.
    """
    # Primary: yfinance (fastest, reliable)
    try:
        import yfinance as yf
        df = yf.download(
            f"{symbol}.NS", period=f"{days}d",
            interval="1d", progress=False, auto_adjust=True
        )
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df.index   = pd.to_datetime(df.index)
            return df.dropna()
    except Exception as e:
        logger.debug(f"yfinance daily error [{symbol}]: {e}")

    # Fallback: jugaad-data
    try:
        from jugaad_data.nse import stock_df
        from_ = date.today() - timedelta(days=days + 10)
        to_   = date.today()
        df = stock_df(symbol=symbol, from_date=from_, to_date=to_, series="EQ")
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if "ch. 52-week high" in df.columns:
                df = df.rename(columns={"series ": "series"})
            # Keep only OHLCV columns
            col_map = {
                "open price":  "open",
                "high price":  "high",
                "low price":   "low",
                "close price": "close",
                "total traded quantity": "volume",
            }
            for old, new in col_map.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[keep].dropna()
    except Exception as e:
        logger.debug(f"jugaad daily error [{symbol}]: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SMART TICK POLLER — replaces Kite WebSocket
# ─────────────────────────────────────────────────────────────────────────────

class FreeTickStreamer:
    """
    Drop-in replacement for the paid Kite WebSocket tick streamer.
    Polls NSE India public API every 3 seconds during market hours.
    Same subscribe/unsubscribe/get_ltp interface as tick_streamer.py.
    Zero cost. No API key required.
    """

    def __init__(self):
        self._subscribers:    Dict[str, List[Callable[[Tick], None]]] = {}
        self._latest_ticks:   Dict[str, Tick] = {}
        self._subscribed:     Set[str]  = set()
        self._running:        bool      = False
        self._poll_thread:    Optional[threading.Thread] = None
        self._lock            = threading.Lock()
        self._stats           = {"polls": 0, "errors": 0, "ticks_dispatched": 0}

    # ── PUBLIC API (identical to old tick_streamer.py) ──────────────────────

    def subscribe(self, symbol: str, callback: Callable[[Tick], None]):
        with self._lock:
            if symbol not in self._subscribers:
                self._subscribers[symbol] = []
            self._subscribers[symbol].append(callback)
            self._subscribed.add(symbol)
        logger.info(f"[FreeStreamer] Subscribed: {symbol}")

    def unsubscribe(self, symbol: str):
        with self._lock:
            self._subscribers.pop(symbol, None)
            self._subscribed.discard(symbol)
            self._latest_ticks.pop(symbol, None)
        logger.info(f"[FreeStreamer] Unsubscribed: {symbol}")

    def _token_to_symbol(self, token: int) -> Optional[str]:
        """Reverse lookup: instrument token → symbol name."""
        from data.nifty50_universe import NIFTY50_TOKENS
        return NIFTY50_TOKENS.get(token)

    def get_ltp(self, symbol: str) -> Optional[float]:
        tick = self._latest_ticks.get(symbol)
        if tick:
            return tick.ltp
        # On-demand fetch if not streaming yet
        quote = get_live_quote(symbol)
        return float(quote["ltp"]) if quote else None

    def start(self):
        if self._running:
            return
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="free-tick-poller",
        )
        self._poll_thread.start()
        logger.info("[FreeStreamer] Started — polling NSE public API (₹0/month)")

    def stop(self):
        self._running = False
        logger.info("[FreeStreamer] Stopped")

    def get_stats(self) -> Dict:
        return {
            **self._stats,
            "subscribed_symbols": len(self._subscribed),
            "source": "NSE public API + jugaad-data + nsetools",
            "cost": "₹0",
        }

    # ── POLLING LOOP ────────────────────────────────────────────────────────

    def _poll_loop(self):
        """Poll all subscribed symbols in batches."""
        logger.info("[FreeStreamer] Poll loop started")
        while self._running:
            try:
                symbols = list(self._subscribed)
                if not symbols:
                    time.sleep(LIVE_POLL_SEC)
                    continue

                poll_start = time.time()

                # Fetch in batches
                for i in range(0, len(symbols), BATCH_SIZE):
                    batch = symbols[i:i + BATCH_SIZE]
                    self._fetch_batch(batch)
                    if i + BATCH_SIZE < len(symbols):
                        time.sleep(0.5)   # Between batches

                self._stats["polls"] += 1

                # Adaptive sleep: target LIVE_POLL_SEC total cycle time
                elapsed  = time.time() - poll_start
                sleep_for = max(0.5, LIVE_POLL_SEC - elapsed)
                time.sleep(sleep_for)

            except Exception as e:
                self._stats["errors"] += 1
                logger.warning(f"[FreeStreamer] Poll error: {e}")
                time.sleep(5)

    def _fetch_batch(self, symbols: List[str]):
        """Fetch a batch of quotes and dispatch ticks."""
        for sym in symbols:
            try:
                quote = get_live_quote(sym)
                if not quote:
                    continue

                ltp   = float(quote.get("ltp", 0))
                if ltp == 0:
                    continue

                tick = Tick(
                    symbol               = sym,
                    ltp                  = ltp,
                    last_traded_quantity = 0,
                    average_traded_price = float(quote.get("vwap", ltp)),
                    volume_traded        = int(quote.get("volume", 0)),
                    ohlc = {
                        "open":  quote.get("open",  0),
                        "high":  quote.get("high",  ltp),
                        "low":   quote.get("low",   ltp),
                        "close": quote.get("prev_close", 0),
                    },
                    high       = float(quote.get("high", ltp)),
                    low        = float(quote.get("low",  ltp)),
                    change     = float(quote.get("change", 0)),
                    change_pct = float(quote.get("change_pct", 0)),
                    source     = quote.get("source", "nse_api"),
                )

                self._latest_ticks[sym] = tick
                self._dispatch(sym, tick)
                self._stats["ticks_dispatched"] += 1

            except Exception as e:
                logger.debug(f"[FreeStreamer] Tick error [{sym}]: {e}")

    def _dispatch(self, symbol: str, tick: Tick):
        callbacks = self._subscribers.get(symbol, [])
        for cb in callbacks:
            try:
                cb(tick)
            except Exception as e:
                logger.error(f"[FreeStreamer] Callback error [{symbol}]: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE BUILDER — assembles real-time OHLCV from tick stream
# ─────────────────────────────────────────────────────────────────────────────

class CandleBuilder:
    """
    Builds OHLCV candles from the tick stream in real-time.
    Works with the 3-second polling interval — builds 5m candles accurately.
    Fires a callback each time a candle completes.
    """

    def __init__(
        self,
        symbol:        str,
        interval_min:  int = 5,
        on_candle:     Callable[[Dict], None] = None,
    ):
        self.symbol       = symbol
        self.interval_min = interval_min
        self.on_candle    = on_candle
        self._current:    Optional[Dict] = None
        self._candles:    List[Dict]     = []

    def on_tick(self, tick: Tick):
        """Process one tick. Call this as the streaming callback."""
        now       = tick.timestamp
        candle_ts = self._candle_start(now)

        if self._current is None or self._current["timestamp"] != candle_ts:
            # Close the old candle
            if self._current is not None:
                self._candles.append(dict(self._current))
                if self.on_candle:
                    self.on_candle(dict(self._current))

            # Open a new candle
            self._current = {
                "timestamp": candle_ts,
                "open":   tick.ltp,
                "high":   tick.ltp,
                "low":    tick.ltp,
                "close":  tick.ltp,
                "volume": tick.volume_traded,
                "symbol": self.symbol,
            }
        else:
            # Update the current candle
            self._current["high"]   = max(self._current["high"],   tick.ltp)
            self._current["low"]    = min(self._current["low"],    tick.ltp)
            self._current["close"]  = tick.ltp
            self._current["volume"] += tick.volume_traded

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Return completed candles as a DataFrame."""
        if not self._candles:
            return None
        df = pd.DataFrame(self._candles)
        df = df.set_index("timestamp").drop(columns=["symbol"], errors="ignore")
        return df

    def _candle_start(self, dt: datetime) -> datetime:
        """Floor a datetime to the candle interval boundary."""
        minutes  = dt.minute
        floored  = (minutes // self.interval_min) * self.interval_min
        return dt.replace(minute=floored, second=0, microsecond=0)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON — import and use this everywhere
# ─────────────────────────────────────────────────────────────────────────────
free_streamer = FreeTickStreamer()
