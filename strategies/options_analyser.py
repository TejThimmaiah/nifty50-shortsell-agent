"""
Options Chain Analyser
Reads NSE F&O data to extract put/call ratio, max pain, and OI build-up.
These are powerful directional bias indicators for short selling.
Completely free — NSE public API.
"""

import logging
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer":    "https://www.nseindia.com/",
    "Accept":     "application/json",
}


@dataclass
class OptionsSignal:
    symbol: str
    put_call_ratio: float           # PCR > 1 = bearish sentiment for stock
    max_pain: float                 # price where options writers profit most
    current_price: float
    max_pain_distance_pct: float    # % distance from max pain to current price
    call_oi_buildup: bool           # heavy call OI = resistance above
    put_oi_buildup: bool            # heavy put OI = support below
    bearish_bias: bool              # overall options market is bearish on this stock
    conviction: float               # 0–1
    key_resistance: float           # nearest strike with heavy call OI
    key_support: float              # nearest strike with heavy put OI
    summary: str


class OptionsChainAnalyser:
    """
    Fetches and analyses options chain data from NSE.
    Used to add options-based conviction to short trades.
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(NSE_HEADERS)
        self._cookie_ok = False

    def _refresh_cookies(self):
        try:
            self._session.get("https://www.nseindia.com", timeout=10)
            self._cookie_ok = True
        except Exception as e:
            logger.warning(f"Cookie refresh failed: {e}")

    def analyse(self, symbol: str, expiry: str = None) -> Optional[OptionsSignal]:
        """
        Full options chain analysis for a stock.
        expiry: "YYYY-MM-DD" string or None for nearest expiry.
        """
        if not self._cookie_ok:
            self._refresh_cookies()

        chain = self._fetch_chain(symbol)
        if not chain:
            return None

        try:
            return self._parse_chain(symbol, chain, expiry)
        except Exception as e:
            logger.error(f"Options chain parse error [{symbol}]: {e}")
            return None

    def get_nifty_pcr(self) -> Optional[float]:
        """Fetch Nifty 50 overall put/call ratio — market-wide sentiment gauge."""
        chain = self._fetch_chain("NIFTY", is_index=True)
        if not chain:
            return None
        try:
            data = chain.get("filtered", {}).get("data", [])
            total_put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in data)
            total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in data)
            if total_call_oi == 0:
                return None
            pcr = total_put_oi / total_call_oi
            logger.info(f"Nifty PCR: {pcr:.3f}")
            return round(pcr, 3)
        except Exception as e:
            logger.error(f"Nifty PCR error: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _fetch_chain(self, symbol: str, is_index: bool = False) -> Optional[Dict]:
        if is_index:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Options chain fetch error [{symbol}]: {e}")
            return None

    def _parse_chain(self, symbol: str, chain: Dict, target_expiry: str) -> OptionsSignal:
        records     = chain.get("filtered", {}).get("data", [])
        current_price = float(chain.get("filtered", {}).get("underlyingValue", 0))

        # Pick expiry
        expiries = chain.get("records", {}).get("expiryDates", [])
        if not expiries:
            raise ValueError("No expiry dates found")
        expiry = target_expiry if target_expiry and target_expiry in expiries else expiries[0]

        # Filter to chosen expiry
        exp_records = [r for r in records if r.get("expiryDate") == expiry]

        # Aggregate by strike
        strikes      = {}
        total_put_oi  = 0
        total_call_oi = 0

        for r in exp_records:
            strike = float(r.get("strikePrice", 0))
            ce = r.get("CE", {})
            pe = r.get("PE", {})
            call_oi = int(ce.get("openInterest", 0))
            put_oi  = int(pe.get("openInterest", 0))
            strikes[strike] = {"call_oi": call_oi, "put_oi": put_oi}
            total_call_oi += call_oi
            total_put_oi  += put_oi

        if total_call_oi == 0:
            raise ValueError("No call OI data")

        pcr = total_put_oi / total_call_oi

        # Max pain: strike where total option loss for buyers is maximum
        max_pain = self._calculate_max_pain(strikes)

        # Key levels: strike with highest call OI (resistance) and put OI (support)
        if strikes:
            key_resistance = max(strikes, key=lambda s: strikes[s]["call_oi"])
            key_support    = max(strikes, key=lambda s: strikes[s]["put_oi"])
        else:
            key_resistance = key_support = current_price

        # Distance from max pain
        max_pain_dist_pct = ((current_price - max_pain) / max_pain * 100
                             if max_pain > 0 else 0)

        # OI build-up patterns
        call_oi_above = sum(
            v["call_oi"] for s, v in strikes.items() if s > current_price
        )
        put_oi_below  = sum(
            v["put_oi"] for s, v in strikes.items() if s < current_price
        )
        call_oi_buildup = call_oi_above > total_call_oi * 0.60
        put_oi_buildup  = put_oi_below  > total_put_oi  * 0.60

        # Bearish bias: PCR < 0.8 (more calls than puts) AND price > max pain
        # Means call writers (bears) have advantage; stock likely to fall to max pain
        bearish_bias = (pcr < 0.8 and max_pain_dist_pct > 1.0) or call_oi_buildup

        # Conviction score
        conviction = 0.0
        if pcr < 0.7:   conviction += 0.30
        elif pcr < 0.9: conviction += 0.15
        if call_oi_buildup:       conviction += 0.25
        if max_pain_dist_pct > 2: conviction += 0.20
        if bearish_bias:          conviction += 0.15
        conviction = min(1.0, conviction)

        summary_parts = []
        if pcr < 0.8:
            summary_parts.append(f"PCR {pcr:.2f} (bearish — heavy call writing)")
        if max_pain_dist_pct > 1:
            summary_parts.append(f"₹{max_pain:.0f} max pain ({max_pain_dist_pct:.1f}% below spot)")
        if call_oi_buildup:
            summary_parts.append(f"Call OI wall at ₹{key_resistance:.0f}")

        return OptionsSignal(
            symbol=symbol,
            put_call_ratio=round(pcr, 3),
            max_pain=round(max_pain, 2),
            current_price=current_price,
            max_pain_distance_pct=round(max_pain_dist_pct, 2),
            call_oi_buildup=call_oi_buildup,
            put_oi_buildup=put_oi_buildup,
            bearish_bias=bearish_bias,
            conviction=round(conviction, 3),
            key_resistance=key_resistance,
            key_support=key_support,
            summary="; ".join(summary_parts) if summary_parts else "No strong options signal",
        )

    @staticmethod
    def _calculate_max_pain(strikes: Dict[float, Dict]) -> float:
        """
        Max pain: the strike at which the total $ value of expiring options
        (for buyers) is minimised — i.e. where option writers profit most.
        """
        if not strikes:
            return 0.0

        strike_list = sorted(strikes.keys())
        min_loss = float("inf")
        max_pain_strike = strike_list[0]

        for test_strike in strike_list:
            total_loss = 0
            for s, oi in strikes.items():
                # Call option loss (if expires worthless): strike < test_strike
                if s < test_strike:
                    total_loss += oi["call_oi"] * (test_strike - s)
                # Put option loss (if expires worthless): strike > test_strike
                if s > test_strike:
                    total_loss += oi["put_oi"] * (s - test_strike)

            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = test_strike

        return max_pain_strike
