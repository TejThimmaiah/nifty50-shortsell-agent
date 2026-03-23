"""
NSE Data Fetcher
Pulls live prices, OHLCV, FII/DII data, and F&O OI from NSE India
— completely free using public endpoints and yfinance.
"""

import time
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

class NSESession:
    """Manages NSE cookie session (required for API calls)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._cookie_refreshed_at = None

    def _refresh_cookies(self):
        """Hit NSE homepage to get fresh cookies."""
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
            self.session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=10)
            self._cookie_refreshed_at = datetime.now()
            logger.debug("NSE cookies refreshed")
        except Exception as e:
            logger.warning(f"Cookie refresh failed: {e}")

    def get(self, url: str, **kwargs) -> Optional[dict]:
        # Refresh cookies if older than 10 minutes
        if not self._cookie_refreshed_at or \
           (datetime.now() - self._cookie_refreshed_at).seconds > 600:
            self._refresh_cookies()
            time.sleep(1)

        try:
            resp = self.session.get(url, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"NSE fetch failed [{url}]: {e}")
            return None


_nse = NSESession()


# ──────────────────────────────────────────────────────────────
# LIVE QUOTE
# ──────────────────────────────────────────────────────────────

def get_quote(symbol: str) -> Optional[Dict]:
    """Fetch live quote from NSE for a given symbol."""
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    data = _nse.get(url)
    if not data:
        return None
    try:
        pd_data = data.get("priceInfo", {})
        return {
            "symbol": symbol,
            "ltp": pd_data.get("lastPrice", 0),
            "open": pd_data.get("open", 0),
            "high": pd_data.get("intraDayHighLow", {}).get("max", 0),
            "low":  pd_data.get("intraDayHighLow", {}).get("min", 0),
            "prev_close": pd_data.get("previousClose", 0),
            "change_pct": pd_data.get("pChange", 0),
            "volume": data.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("totalTradedVolume", 0),
            "avg_price": pd_data.get("vwap", 0),
            "52w_high": pd_data.get("weekHighLow", {}).get("max", 0),
            "52w_low": pd_data.get("weekHighLow", {}).get("min", 0),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Quote parse error [{symbol}]: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# INTRADAY OHLCV via yfinance (most reliable free source)
# ──────────────────────────────────────────────────────────────

def get_intraday_ohlcv(symbol: str, interval: str = "5m", period: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLCV data — free stack (OpenChart → yfinance).
    interval: '1m', '3m', '5m', '10m', '15m', '30m', '60m'
    No Zerodha WebSocket subscription needed. Cost: ₹0.
    """
    try:
        from data.free_market_data import get_intraday_ohlcv_free
        # Convert period string to days integer
        days = int(period.replace("d", "").replace("wk", "7")) if period else 1
        df = get_intraday_ohlcv_free(symbol, interval=interval, days=days)
        if df is not None and not df.empty:
            logger.debug(f"Fetched {len(df)} candles for {symbol} [{interval}]")
            return df
    except Exception as e:
        logger.debug(f"free_market_data intraday error: {e}")

    # Final fallback: yfinance direct
    try:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if not df.empty:
            df.index   = pd.to_datetime(df.index)
            df.columns = [c.lower() for c in df.columns]
            return df.dropna()
    except Exception as e:
        logger.error(f"yfinance intraday fallback error [{symbol}]: {e}")

    return None


def get_historical_ohlcv(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for technical indicator calculation."""
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except Exception as e:
        logger.error(f"Historical fetch error [{symbol}]: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# NSE MARKET-WIDE DATA
# ──────────────────────────────────────────────────────────────

def get_top_losers(count: int = 20) -> List[Dict]:
    """Fetch today's top losers from NSE (overbought reversal candidates)."""
    url = "https://www.nseindia.com/api/live-analysis-variations?index=loseers"
    data = _nse.get(url)
    if not data:
        return []
    try:
        records = data.get("NIFTY", {}).get("data", [])
        return [
            {
                "symbol": r.get("symbol"),
                "ltp": r.get("ltp", 0),
                "change_pct": r.get("perChange", 0),
                "volume": r.get("tradedQuantity", 0),
            }
            for r in records[:count]
        ]
    except Exception as e:
        logger.error(f"Top losers fetch error: {e}")
        return []


def get_top_gainers(count: int = 20) -> List[Dict]:
    """Fetch top gainers — short candidates after gap-up exhaustion."""
    url = "https://www.nseindia.com/api/live-analysis-variations?index=gainers"
    data = _nse.get(url)
    if not data:
        return []
    try:
        records = data.get("NIFTY", {}).get("data", [])
        return [
            {
                "symbol": r.get("symbol"),
                "ltp": r.get("ltp", 0),
                "change_pct": r.get("perChange", 0),
                "volume": r.get("tradedQuantity", 0),
            }
            for r in records[:count]
        ]
    except Exception as e:
        logger.error(f"Top gainers fetch error: {e}")
        return []


def get_fo_stocks() -> List[str]:
    """Fetch complete list of F&O eligible stocks (can be shorted)."""
    url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
    data = _nse.get(url)
    if not data:
        # Return a safe default list
        from config import TRADING
        return TRADING.priority_watchlist
    try:
        records = data.get("data", [])
        return [r["symbol"] for r in records if r.get("symbol")]
    except Exception as e:
        logger.error(f"F&O list error: {e}")
        return []


def get_fii_dii_data() -> Dict:
    """Fetch FII/DII provisional data for the day."""
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    data = _nse.get(url)
    if not data:
        return {}
    try:
        # Returns list of records by category
        fii_row = next((r for r in data if r.get("category") == "FII/FPI *"), {})
        dii_row = next((r for r in data if r.get("category") == "DII"), {})
        return {
            "fii_buy_value":  fii_row.get("buyValue", 0),
            "fii_sell_value": fii_row.get("sellValue", 0),
            "fii_net":        fii_row.get("netValue", 0),
            "dii_buy_value":  dii_row.get("buyValue", 0),
            "dii_sell_value": dii_row.get("sellValue", 0),
            "dii_net":        dii_row.get("netValue", 0),
            "market_sentiment": "BEARISH" if float(fii_row.get("netValue", 0)) < 0 else "BULLISH",
        }
    except Exception as e:
        logger.error(f"FII/DII data error: {e}")
        return {}


def get_oi_data(symbol: str) -> Dict:
    """Fetch Open Interest data from F&O for directional bias."""
    url = f"https://www.nseindia.com/api/quote-derivative?symbol={symbol}"
    data = _nse.get(url)
    if not data:
        return {}
    try:
        records = data.get("stocks", [])
        futures = [r for r in records if r.get("metadata", {}).get("instrumentType") == "Stock Futures"]
        if not futures:
            return {}
        fut = futures[0]
        return {
            "oi": fut.get("marketDeptOrderBook", {}).get("otherInfo", {}).get("openInterest", 0),
            "oi_change_pct": fut.get("metadata", {}).get("pchangeInOpenInterest", 0),
            "iv": fut.get("marketDeptOrderBook", {}).get("otherInfo", {}).get("impliedVolatility", 0),
        }
    except Exception as e:
        logger.error(f"OI data error [{symbol}]: {e}")
        return {}


def get_nifty_index() -> Dict:
    """Fetch NIFTY 50 index data for market breadth."""
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    data = _nse.get(url)
    if not data:
        return {}
    try:
        meta = data.get("metadata", {})
        return {
            "index_value": meta.get("indexSymbol", ""),
            "last":   meta.get("last", 0),
            "change_pct": meta.get("percChange", 0),
            "open":   meta.get("open", 0),
            "high":   meta.get("high", 0),
            "low":    meta.get("low", 0),
            "advances": data.get("advance", {}).get("advances", 0),
            "declines": data.get("advance", {}).get("declines", 0),
            "unchanged": data.get("advance", {}).get("unchanged", 0),
        }
    except Exception as e:
        logger.error(f"NIFTY data error: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# VOLUME ANALYSIS
# ──────────────────────────────────────────────────────────────

def get_volume_surge_stocks(threshold: float = 2.0) -> List[Dict]:
    """Find stocks with unusual volume (>2x 20-day average)."""
    url = "https://www.nseindia.com/api/live-analysis-volume-shockers"
    data = _nse.get(url)
    if not data:
        return []
    try:
        records = data.get("data", [])
        return [
            {
                "symbol": r.get("symbol"),
                "volume_ratio": r.get("quantityRatio", 0),
                "change_pct": r.get("perChange", 0),
                "ltp": r.get("ltp", 0),
            }
            for r in records
            if float(r.get("quantityRatio", 0)) >= threshold
        ]
    except Exception as e:
        logger.error(f"Volume surge error: {e}")
        return []
