"""
Tej Options Flow Analyzer
===========================
Reads Nifty 50 options chain to detect where big money is positioned.

"HDFCBANK: Max pain at 1600. Put/Call ratio 1.8 (bearish). 
 OI buildup at 1620 resistance. Strong short signal."
"""

import logging
import requests
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("options_flow")
IST = ZoneInfo("Asia/Kolkata")

NSE_BASE = "https://www.nseindia.com"
HEADERS  = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}


@dataclass
class OptionsAnalysis:
    symbol:          str
    expiry:          str
    spot_price:      float
    max_pain:        float
    pcr:             float    # Put/Call Ratio
    pcr_signal:      str      # "BEARISH" / "NEUTRAL" / "BULLISH"
    call_oi_wall:    float    # Strongest call resistance
    put_oi_wall:     float    # Strongest put support
    iv_rank:         float    # Implied volatility rank (0-100)
    smart_money:     str      # "BUYING_PUTS" / "SELLING_CALLS" / "NEUTRAL"
    signal:          str      # "STRONG_SHORT" / "SHORT" / "NEUTRAL" / "AVOID"
    confidence:      float


class OptionsFlowAnalyzer:
    """
    Analyzes NSE options chain for institutional positioning.
    NSE provides free options data via public APIs.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._init_session()

    def _init_session(self):
        """Initialize NSE session with cookies."""
        try:
            self.session.get(f"{NSE_BASE}/", timeout=10)
            self.session.get(f"{NSE_BASE}/market-data/equity-derivatives-watch", timeout=10)
        except Exception:
            pass

    def get_option_chain(self, symbol: str) -> Optional[dict]:
        """Fetch options chain from NSE."""
        try:
            if symbol.upper() == "NIFTY":
                url = f"{NSE_BASE}/api/option-chain-indices?symbol=NIFTY"
            else:
                url = f"{NSE_BASE}/api/option-chain-equities?symbol={symbol.upper()}"

            r = self.session.get(url, timeout=15)
            if r.ok:
                return r.json()
        except Exception as e:
            logger.warning(f"Options chain fetch failed for {symbol}: {e}")
        return None

    def calculate_max_pain(self, chain_data: dict) -> float:
        """
        Calculate max pain strike — where most option holders lose money.
        This is often where the stock tends to gravitate on expiry.
        """
        try:
            records = chain_data.get("records", {}).get("data", [])
            strikes = {}
            for row in records:
                strike = row.get("strikePrice", 0)
                ce_oi  = row.get("CE", {}).get("openInterest", 0)
                pe_oi  = row.get("PE", {}).get("openInterest", 0)
                strikes[strike] = {"ce_oi": ce_oi, "pe_oi": pe_oi}

            if not strikes:
                return 0.0

            min_pain   = float("inf")
            max_pain_s = 0

            for test_strike in strikes:
                total_pain = 0
                for strike, data in strikes.items():
                    # Call pain: calls ITM at test_strike
                    if strike < test_strike:
                        total_pain += data["ce_oi"] * (test_strike - strike)
                    # Put pain: puts ITM at test_strike
                    if strike > test_strike:
                        total_pain += data["pe_oi"] * (strike - test_strike)

                if total_pain < min_pain:
                    min_pain   = total_pain
                    max_pain_s = test_strike

            return float(max_pain_s)
        except Exception:
            return 0.0

    def calculate_pcr(self, chain_data: dict) -> float:
        """Put/Call Ratio by Open Interest."""
        try:
            records = chain_data.get("records", {}).get("data", [])
            total_put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in records)
            total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records)
            if total_call_oi == 0:
                return 1.0
            return round(total_put_oi / total_call_oi, 2)
        except Exception:
            return 1.0

    def find_oi_walls(self, chain_data: dict, spot: float) -> tuple:
        """Find strongest call wall (resistance) and put wall (support)."""
        try:
            records = chain_data.get("records", {}).get("data", [])
            calls = {}
            puts  = {}

            for row in records:
                strike = row.get("strikePrice", 0)
                if strike > spot:
                    calls[strike] = row.get("CE", {}).get("openInterest", 0)
                elif strike < spot:
                    puts[strike]  = row.get("PE", {}).get("openInterest", 0)

            call_wall = max(calls, key=calls.get) if calls else spot * 1.02
            put_wall  = max(puts,  key=puts.get)  if puts  else spot * 0.98
            return float(call_wall), float(put_wall)
        except Exception:
            return spot * 1.02, spot * 0.98

    def analyze(self, symbol: str) -> OptionsAnalysis:
        """Full options flow analysis for a symbol."""
        chain = self.get_option_chain(symbol)

        if not chain:
            return OptionsAnalysis(
                symbol=symbol, expiry="", spot_price=0, max_pain=0,
                pcr=1.0, pcr_signal="NEUTRAL", call_oi_wall=0, put_oi_wall=0,
                iv_rank=50, smart_money="NEUTRAL", signal="NO_DATA", confidence=0.0
            )

        try:
            spot = float(chain.get("records", {}).get("underlyingValue", 0))
            expiry = chain.get("records", {}).get("expiryDates", [""])[0]

            max_pain   = self.calculate_max_pain(chain)
            pcr        = self.calculate_pcr(chain)
            call_wall, put_wall = self.find_oi_walls(chain, spot)

            # PCR interpretation
            if pcr > 1.5:
                pcr_signal = "BEARISH"   # High puts = hedging = fear
            elif pcr < 0.7:
                pcr_signal = "BULLISH"   # High calls = optimism
            else:
                pcr_signal = "NEUTRAL"

            # Smart money detection
            # If spot > max_pain → likely to fall toward max pain → bearish
            # If heavy call writing above → institutions capping upside
            if spot > max_pain * 1.02 and pcr > 1.2:
                smart_money = "BUYING_PUTS"
            elif pcr > 1.5:
                smart_money = "SELLING_CALLS"
            else:
                smart_money = "NEUTRAL"

            # Distance from call wall (resistance)
            call_wall_dist = (call_wall - spot) / spot * 100 if spot else 5
            max_pain_dist  = (spot - max_pain) / spot * 100 if spot else 0

            # Generate signal
            bearish_factors = 0
            if pcr_signal == "BEARISH":
                bearish_factors += 2
            if smart_money in ("BUYING_PUTS", "SELLING_CALLS"):
                bearish_factors += 2
            if max_pain_dist > 2:   # spot well above max pain
                bearish_factors += 1
            if call_wall_dist < 2:  # close to call resistance
                bearish_factors += 1

            if bearish_factors >= 5:
                signal     = "STRONG_SHORT"
                confidence = 0.85
            elif bearish_factors >= 3:
                signal     = "SHORT"
                confidence = 0.65
            elif bearish_factors == 0 and pcr_signal == "BULLISH":
                signal     = "AVOID"
                confidence = 0.70
            else:
                signal     = "NEUTRAL"
                confidence = 0.50

            return OptionsAnalysis(
                symbol=symbol, expiry=expiry, spot_price=spot,
                max_pain=max_pain, pcr=pcr, pcr_signal=pcr_signal,
                call_oi_wall=call_wall, put_oi_wall=put_wall,
                iv_rank=50, smart_money=smart_money,
                signal=signal, confidence=confidence,
            )
        except Exception as e:
            logger.error(f"Options analysis failed for {symbol}: {e}")
            return OptionsAnalysis(
                symbol=symbol, expiry="", spot_price=0, max_pain=0,
                pcr=1.0, pcr_signal="NEUTRAL", call_oi_wall=0, put_oi_wall=0,
                iv_rank=50, smart_money="NEUTRAL", signal="ERROR", confidence=0.0
            )

    def format_for_telegram(self, symbol: str) -> str:
        """Format options analysis as Telegram message."""
        a = self.analyze(symbol)
        sig_emoji = {"STRONG_SHORT": "🔴🔴", "SHORT": "🔴", "NEUTRAL": "🟡",
                     "AVOID": "🟢", "NO_DATA": "⚪", "ERROR": "⚠️"}.get(a.signal, "⚪")
        return (
            f"<b>Options Flow: {symbol}</b>\n\n"
            f"{sig_emoji} Signal: {a.signal} ({a.confidence:.0%})\n"
            f"Spot: {a.spot_price:.2f} | Max Pain: {a.max_pain:.2f}\n"
            f"PCR: {a.pcr:.2f} ({a.pcr_signal})\n"
            f"Call Wall: {a.call_oi_wall:.2f} | Put Wall: {a.put_oi_wall:.2f}\n"
            f"Smart Money: {a.smart_money}"
        )


options_analyzer = OptionsFlowAnalyzer()
