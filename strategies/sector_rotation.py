"""
Sector Rotation Analyser
Tracks NSE sector indices to identify sectors under distribution (selling pressure).
Short selling works best when you trade WITH the sector trend — short weak stocks
in weak sectors, not strong stocks in strong sectors.

Sector indices tracked (NSE Indices):
  NIFTY BANK, NIFTY IT, NIFTY PHARMA, NIFTY AUTO, NIFTY METAL,
  NIFTY REALTY, NIFTY FMCG, NIFTY ENERGY, NIFTY INFRA, NIFTY MEDIA
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# NSE sector index → Yahoo Finance ticker mapping
SECTOR_TICKERS = {
    "BANK":     "^NSEBANK",
    "IT":       "^CNXIT",
    "PHARMA":   "^CNXPHARMA",
    "AUTO":     "^CNXAUTO",
    "METAL":    "^CNXMETAL",
    "REALTY":   "^CNXREALTY",
    "FMCG":     "^CNXFMCG",
    "ENERGY":   "^CNXENERGY",
    "INFRA":    "^CNXINFRA",
    "MEDIA":    "^CNXMEDIA",
    "MIDCAP":   "^NSMIDCP",
    "SMALLCAP": "^CNXSC",
}

# Which stocks belong to which sector (for candidate routing)
SECTOR_STOCKS = {
    "BANK":    ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
                "INDUSINDBK", "BANDHANBNK", "IDFCFIRSTB", "FEDERALBNK", "AUBANK"],
    "IT":      ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
                "MPHASIS", "PERSISTENT", "COFORGE", "LTTS"],
    "PHARMA":  ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON",
                "TORNTPHARM", "LUPIN", "AUROPHARMA", "ALKEM", "IPCALAB"],
    "AUTO":    ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "EICHERMOT",
                "HEROMOTOCO", "M&M", "ASHOKLEY", "TVSMOTORS", "TVSMOTOR"],
    "METAL":   ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "NMDC",
                "VEDL", "SAIL", "JINDALSTEL", "NATIONALUM"],
    "FMCG":   ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
                "GODREJCP", "MARICO", "COLPAL", "EMAMILTD"],
    "ENERGY":  ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "POWERGRID",
                "NTPC", "TATAPOWER", "ADANIGREEN"],
}


@dataclass
class SectorReading:
    sector: str
    ticker: str
    change_1d_pct: float
    change_5d_pct: float
    change_20d_pct: float
    rsi_14: float
    above_ema50: bool
    trend: str              # "UPTREND" | "DOWNTREND" | "SIDEWAYS"
    distribution: bool      # True = sector is being sold off
    short_favorable: bool   # True = good for shorting stocks in this sector
    relative_strength: float  # vs Nifty 50 (negative = underperforming)


@dataclass
class SectorSnapshot:
    timestamp: str
    readings: Dict[str, SectorReading]
    weakest_sectors: List[str]       # best for shorting
    strongest_sectors: List[str]     # avoid for shorting
    market_phase: str                # "RISK_OFF" | "ROTATION" | "RISK_ON" | "MIXED"
    short_favorable_sectors: List[str]
    recommended_avoid: List[str]


class SectorRotationAnalyser:
    """
    Analyses sector performance to guide stock selection.
    Short candidates should come from the weakest sectors.
    """

    def __init__(self):
        self._cache: Optional[SectorSnapshot] = None
        self._cache_time: Optional[datetime]  = None
        self._nifty_returns: Dict[str, float] = {}

    def get_snapshot(self, force_refresh: bool = False) -> SectorSnapshot:
        """Get current sector rotation snapshot (cached for 30 min)."""
        if (not force_refresh and self._cache and self._cache_time and
                (datetime.now(IST) - self._cache_time).seconds < 1800):
            return self._cache

        snapshot = self._build_snapshot()
        self._cache = snapshot
        self._cache_time = datetime.now(IST)
        return snapshot

    def is_sector_shortable(self, symbol: str) -> Tuple[bool, str]:
        """
        Given a stock symbol, check if its sector is in distribution.
        Returns (shortable: bool, reason: str).
        """
        sector = self._symbol_to_sector(symbol)
        if not sector:
            return True, "Unknown sector — no restriction"

        snapshot = self.get_snapshot()
        reading  = snapshot.readings.get(sector)

        if not reading:
            return True, "No sector data"

        if not reading.short_favorable:
            return False, f"{sector} sector is strong — avoid shorting {symbol}"

        strength = "very weak" if reading.change_5d_pct < -3 else "weak"
        return True, f"{sector} sector {strength} ({reading.change_5d_pct:.1f}% in 5d)"

    def get_best_sectors_for_shorts(self, top_n: int = 3) -> List[str]:
        """Return the top N sectors most favorable for short selling."""
        snapshot = self.get_snapshot()
        return snapshot.weakest_sectors[:top_n]

    def get_sector_stocks(self, sector: str) -> List[str]:
        """Get stocks belonging to a sector."""
        return SECTOR_STOCKS.get(sector.upper(), [])

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _build_snapshot(self) -> SectorSnapshot:
        """Build a fresh sector snapshot."""
        logger.info("Building sector rotation snapshot...")

        # Get Nifty 50 returns as benchmark
        nifty_df = self._fetch_returns("^NSEI")
        nifty_1d  = nifty_df.get("1d", 0)
        nifty_5d  = nifty_df.get("5d", 0)
        nifty_20d = nifty_df.get("20d", 0)

        readings: Dict[str, SectorReading] = {}

        for sector, ticker in SECTOR_TICKERS.items():
            try:
                ret   = self._fetch_returns(ticker)
                rsi   = self._fetch_rsi(ticker)
                above = self._is_above_ema50(ticker)

                ch1d  = ret.get("1d",  0)
                ch5d  = ret.get("5d",  0)
                ch20d = ret.get("20d", 0)

                # Relative strength vs Nifty
                rel_str = ch5d - nifty_5d

                # Trend classification
                if ch20d > 2 and above:
                    trend = "UPTREND"
                elif ch20d < -2 or not above:
                    trend = "DOWNTREND"
                else:
                    trend = "SIDEWAYS"

                # Distribution: price falling + volume pattern (simplified: returns negative)
                distribution = ch5d < -2 or (ch1d < -1 and ch20d < 0)

                # Short favorable: downtrend or distribution, underperforming Nifty
                short_favorable = (
                    trend == "DOWNTREND" or
                    distribution or
                    (rel_str < -1.5 and ch5d < 0)
                )

                readings[sector] = SectorReading(
                    sector=sector,
                    ticker=ticker,
                    change_1d_pct=round(ch1d, 2),
                    change_5d_pct=round(ch5d, 2),
                    change_20d_pct=round(ch20d, 2),
                    rsi_14=round(rsi, 1),
                    above_ema50=above,
                    trend=trend,
                    distribution=distribution,
                    short_favorable=short_favorable,
                    relative_strength=round(rel_str, 2),
                )
                logger.debug(f"  {sector}: 5d={ch5d:.1f}% | trend={trend} | shorts={'✓' if short_favorable else '✗'}")
                time.sleep(0.5)  # yfinance rate limit

            except Exception as e:
                logger.warning(f"Sector data error [{sector}]: {e}")

        # Rank sectors
        sorted_by_weakness = sorted(
            readings.values(),
            key=lambda r: r.change_5d_pct
        )
        weakest   = [r.sector for r in sorted_by_weakness[:4]]
        strongest = [r.sector for r in sorted_by_weakness[-4:]]
        shortable = [r.sector for r in readings.values() if r.short_favorable]

        # Market phase assessment
        weak_count   = sum(1 for r in readings.values() if r.trend == "DOWNTREND")
        strong_count = sum(1 for r in readings.values() if r.trend == "UPTREND")
        total = len(readings) or 1

        if weak_count / total > 0.6:
            phase = "RISK_OFF"
        elif strong_count / total > 0.6:
            phase = "RISK_ON"
        elif weak_count > 2 and strong_count > 2:
            phase = "ROTATION"
        else:
            phase = "MIXED"

        logger.info(
            f"Sector snapshot done: phase={phase} | "
            f"weakest={weakest[:2]} | shortable={shortable[:3]}"
        )

        return SectorSnapshot(
            timestamp=datetime.now(IST).isoformat(),
            readings=readings,
            weakest_sectors=weakest,
            strongest_sectors=strongest,
            market_phase=phase,
            short_favorable_sectors=shortable,
            recommended_avoid=strongest,
        )

    def _fetch_returns(self, ticker: str) -> Dict[str, float]:
        """Fetch 1d, 5d, 20d returns for a ticker."""
        try:
            df = yf.download(ticker, period="30d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 2:
                return {"1d": 0, "5d": 0, "20d": 0}
            closes = df["Close"].dropna()
            c  = float(closes.iloc[-1])
            p1 = float(closes.iloc[-2])  if len(closes) > 1  else c
            p5 = float(closes.iloc[-6])  if len(closes) > 5  else closes.iloc[0]
            p20= float(closes.iloc[-21]) if len(closes) > 20 else closes.iloc[0]
            return {
                "1d":  (c - p1)  / p1  * 100,
                "5d":  (c - p5)  / p5  * 100,
                "20d": (c - p20) / p20 * 100,
            }
        except Exception:
            return {"1d": 0, "5d": 0, "20d": 0}

    def _fetch_rsi(self, ticker: str, period: int = 14) -> float:
        """Calculate RSI for a sector index."""
        try:
            df = yf.download(ticker, period="60d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < period + 1:
                return 50.0
            closes = df["Close"].dropna()
            delta  = closes.diff()
            gain   = delta.clip(lower=0).rolling(period).mean()
            loss   = (-delta.clip(upper=0)).rolling(period).mean()
            rs     = gain / loss.replace(0, 1e-10)
            rsi    = 100 - 100 / (1 + rs)
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0

    def _is_above_ema50(self, ticker: str) -> bool:
        """Check if sector index is trading above 50-day EMA."""
        try:
            df = yf.download(ticker, period="90d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 50:
                return True
            closes = df["Close"].dropna()
            ema50  = closes.ewm(span=50, adjust=False).mean()
            return float(closes.iloc[-1]) > float(ema50.iloc[-1])
        except Exception:
            return True

    def _symbol_to_sector(self, symbol: str) -> Optional[str]:
        """Look up which sector a stock belongs to."""
        for sector, stocks in SECTOR_STOCKS.items():
            if symbol in stocks:
                return sector
        return None
