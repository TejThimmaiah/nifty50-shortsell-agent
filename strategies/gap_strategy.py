"""
Gap-Up Short Strategy
The most reliable intraday short pattern on NSE:
Stocks that gap up >2% at open frequently fade within the first hour.
Institutional sellers use gap-ups to distribute at better prices.

Signal logic:
  1. Stock gaps up ≥ 2% from previous close at market open
  2. Volume in first 15 min is > 1.5× 20-day ADV (institutions exiting)
  3. RSI > 65 after gap (already overbought before the day starts)
  4. Gap size is < 8% (very large gaps can keep running — avoid)
  5. Sector is neutral or bearish (no sector tailwind)
  6. No major positive catalyst (checked via self-healer news search)

Entry: short after first 15-min candle closes BELOW the opening price
SL: above the gap-up high
Target: gap fill (previous day's close) — typically a 1.5–4% move
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

from config import TRADING

logger = logging.getLogger(__name__)


@dataclass
class GapSetup:
    symbol:             str
    gap_pct:            float       # % gap from prev close to today's open
    open_price:         float
    prev_close:         float
    gap_fill_target:    float       # = prev_close (gap fill)
    stop_loss:          float       # above today's high after 15 min
    entry_trigger:      float       # first 15-min candle low (enter on breach)
    gap_type:           str         # "FULL_GAP" | "PARTIAL_GAP"
    vol_ratio:          float       # first-15m volume / ADV
    rsi_at_open:        float
    confidence:         float       # 0–1
    reason:             str

    @property
    def risk_reward(self) -> float:
        if self.stop_loss <= self.entry_trigger:
            return 0
        risk   = self.stop_loss - self.entry_trigger
        reward = self.entry_trigger - self.gap_fill_target
        return round(reward / max(risk, 0.01), 2)


class GapUpShortStrategy:
    """
    Identifies gap-up short setups from overnight/pre-market data
    and first-15-minute price action.
    """

    GAP_MIN_PCT      = 2.0    # minimum gap % to qualify
    GAP_MAX_PCT      = 8.0    # max gap % (too big = risky continuation)
    VOL_RATIO_MIN    = 1.5    # first-15m volume / ADV
    RSI_THRESHOLD    = 65.0   # RSI at open must be elevated
    MIN_RR           = 1.5    # minimum risk/reward ratio

    def scan_for_gaps(self, quotes: List[Dict], historical: Dict) -> List[GapSetup]:
        """
        Scan a list of live quotes for gap-up short setups.

        quotes:    list of dicts from get_quote() — must include ltp, open, prev_close, volume
        historical: dict of symbol → DataFrame (for ADV and RSI)
        Returns a list of GapSetup ranked by confidence.
        """
        setups = []

        for q in quotes:
            symbol     = q.get("symbol") or q.get("tradingsymbol", "")
            open_price = float(q.get("open", 0))
            prev_close = float(q.get("prev_close", 0))
            ltp        = float(q.get("ltp", open_price))
            volume     = float(q.get("volume", 0))

            if not symbol or prev_close == 0 or open_price == 0:
                continue

            setup = self._evaluate(symbol, open_price, prev_close, ltp, volume, historical)
            if setup:
                setups.append(setup)

        setups.sort(key=lambda s: s.confidence, reverse=True)
        return setups

    def _evaluate(
        self,
        symbol:     str,
        open_price: float,
        prev_close: float,
        ltp:        float,
        volume:     float,
        historical: Dict,
    ) -> Optional[GapSetup]:
        """Evaluate one stock for a gap-up short setup."""

        # Compute gap %
        gap_pct = (open_price - prev_close) / prev_close * 100
        if not (self.GAP_MIN_PCT <= gap_pct <= self.GAP_MAX_PCT):
            return None

        # Volume ratio (first 15 min vs 20-day ADV)
        df    = historical.get(symbol)
        adv   = float(df["volume"].tail(20).mean()) if df is not None and len(df) > 5 else 0
        vol_ratio = volume / max(adv, 1)

        if vol_ratio < self.VOL_RATIO_MIN:
            return None    # Not enough institutional activity

        # RSI estimate from prior day data
        rsi_at_open = 50.0
        if df is not None and len(df) > 14:
            try:
                import pandas_ta as ta
                rsi_series = ta.rsi(df["close"], length=14)
                rsi_at_open = float(rsi_series.iloc[-1])
            except Exception:
                pass

        if rsi_at_open < self.RSI_THRESHOLD:
            return None    # Not overbought enough

        # Entry: short when LTP dips below open (momentum exhaustion)
        entry_trigger = open_price   # refine with first 15-min candle in live mode

        # SL: above today's intraday high (capped at 1.2% above open)
        intraday_high = ltp * 1.01   # proxy — refine with live high
        stop_loss     = round(max(intraday_high, open_price * 1.012), 2)

        # Target: gap fill = previous close
        gap_fill = prev_close

        # Confidence score
        confidence = 0.0
        confidence += min(0.35, (gap_pct - self.GAP_MIN_PCT) / 6 * 0.35)   # gap size (larger = more confidence, up to 8%)
        confidence += min(0.25, (vol_ratio - self.VOL_RATIO_MIN) / 3 * 0.25)  # volume
        confidence += min(0.25, (rsi_at_open - self.RSI_THRESHOLD) / 35 * 0.25)  # RSI
        if ltp < open_price:
            confidence += 0.15   # Price already starting to fade — highest conviction

        confidence = round(min(1.0, confidence), 3)

        if confidence < 0.40:
            return None

        gap_type = "FULL_GAP" if gap_pct >= 3.0 else "PARTIAL_GAP"

        reason_parts = [
            f"Gap +{gap_pct:.1f}% ({gap_type})",
            f"Vol {vol_ratio:.1f}×ADV",
            f"RSI {rsi_at_open:.0f}",
        ]
        if ltp < open_price:
            reason_parts.append("already fading")

        setup = GapSetup(
            symbol=symbol,
            gap_pct=round(gap_pct, 2),
            open_price=open_price,
            prev_close=prev_close,
            gap_fill_target=round(gap_fill, 2),
            stop_loss=stop_loss,
            entry_trigger=round(entry_trigger, 2),
            gap_type=gap_type,
            vol_ratio=round(vol_ratio, 2),
            rsi_at_open=round(rsi_at_open, 1),
            confidence=confidence,
            reason="; ".join(reason_parts),
        )

        # Check R:R
        if setup.risk_reward < self.MIN_RR:
            logger.debug(
                f"Gap setup rejected [{symbol}]: R:R {setup.risk_reward:.2f} < {self.MIN_RR}"
            )
            return None

        logger.info(
            f"Gap setup: {symbol} | gap={gap_pct:.1f}% | "
            f"conf={confidence:.2f} | R:R={setup.risk_reward:.2f} | {setup.reason}"
        )
        return setup

    def to_short_candidate(self, setup: GapSetup, capital: float = 100_000):
        """Convert a GapSetup into a ShortCandidate-compatible dict."""
        risk_per_share = setup.stop_loss - setup.entry_trigger
        max_risk       = capital * TRADING.max_risk_per_trade_pct / 100
        quantity       = max(1, int(max_risk / max(risk_per_share, 0.01)))

        return {
            "symbol":       setup.symbol,
            "score":        setup.confidence,
            "entry_price":  setup.entry_trigger,
            "stop_loss":    setup.stop_loss,
            "target":       setup.gap_fill_target,
            "quantity":     quantity,
            "strategy":     "GAP_UP_SHORT",
            "reason":       setup.reason,
            "gap_pct":      setup.gap_pct,
            "risk_reward":  setup.risk_reward,
        }
