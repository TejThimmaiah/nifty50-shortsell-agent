"""
Market Regime Detector
Classifies the current market regime every morning.
The agent behaves differently in each regime.

CRITICAL — SHORT-SELLING PERSPECTIVE:
  A falling Nifty is NOT a crisis — it is the PRIMARY OPPORTUNITY.
  This agent PROFITS when markets fall.

Regimes for a short-selling agent:
  TRENDING_DOWN  → BEST regime — ideal for shorts, be aggressive
  RANGING        → GOOD regime — short near resistance, quick exits
  VOLATILE       → CAUTION — smaller positions, tighter SL (choppy)
  TRENDING_UP    → AVOID — momentum against shorts, minimal trades
  CRISIS (extreme) → Only when VIX > 30 AND Nifty > -5%
                     (individual stocks hit lower circuits = can't cover shorts)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    label: str              # TRENDING_DOWN | RANGING | VOLATILE | TRENDING_UP | CRISIS
    confidence: float
    nifty_trend_5d: float
    nifty_trend_20d: float
    vix: float
    advance_decline: float
    fii_net_cr: float
    is_good_for_shorts: bool
    short_aggressiveness: str   # AGGRESSIVE | NORMAL | CAUTIOUS | AVOID
    position_size_multiplier: float
    confidence_threshold: float
    max_positions: int
    description: str


class MarketRegimeDetector:
    """
    Classifies market regime from the SHORT-SELLER's perspective.
    A falling market = TRENDING_DOWN = best possible environment.
    """

    def detect(
        self,
        nifty_df:          Optional[pd.DataFrame] = None,
        vix:               float = 15.0,
        advance_decline:   float = 0.5,
        fii_net_cr:        float = 0.0,
        nifty_change_1d:   float = 0.0,
    ) -> MarketRegime:

        trend_5d  = self._get_nifty_trend(nifty_df, days=5)
        trend_20d = self._get_nifty_trend(nifty_df, days=20)
        atr_pct   = self._get_atr_pct(nifty_df)

        # ── CRISIS: Only when VIX is extreme AND crash is so bad stocks hit
        # lower circuit breakers (making it impossible to COVER short positions)
        # This is NOT triggered by a normal 1–3% fall — that's ideal for us
        if vix > 30 and nifty_change_1d < -4.0:
            return MarketRegime(
                label="CRISIS",
                confidence=0.95,
                nifty_trend_5d=trend_5d,
                nifty_trend_20d=trend_20d,
                vix=vix,
                advance_decline=advance_decline,
                fii_net_cr=fii_net_cr,
                is_good_for_shorts=False,
                short_aggressiveness="AVOID",
                position_size_multiplier=0.0,
                confidence_threshold=0.99,
                max_positions=0,
                description=(
                    f"CRISIS: VIX={vix:.1f}, Nifty={nifty_change_1d:.2f}%. "
                    f"Stocks may hit lower circuit — cannot cover shorts. Halt."
                ),
            )

        # ── TRENDING DOWN: Nifty falling — this is the PRIMARY OPPORTUNITY
        # The more it falls, the better the short-selling environment
        if nifty_change_1d < -0.5 or (trend_5d < -1.5 and trend_20d < -2):
            # Quality of downtrend determines aggressiveness
            strong_down = nifty_change_1d < -1.5 or trend_5d < -2.5
            fii_selling = fii_net_cr < -200
            breadth_bearish = advance_decline < 0.40

            if strong_down and fii_selling:
                mult = 1.3; thresh = 0.42; aggressiveness = "AGGRESSIVE"
                desc = (f"Strong downtrend: Nifty {nifty_change_1d:+.2f}%, "
                        f"FII sold ₹{abs(fii_net_cr):.0f}Cr. Prime shorts environment.")
            elif strong_down or fii_selling or breadth_bearish:
                mult = 1.15; thresh = 0.48; aggressiveness = "AGGRESSIVE"
                desc = (f"Downtrend: Nifty {nifty_change_1d:+.2f}%. "
                        f"Good shorts environment.")
            else:
                mult = 1.0; thresh = 0.50; aggressiveness = "NORMAL"
                desc = f"Mild downtrend: Nifty {nifty_change_1d:+.2f}%. Normal short sizing."

            return MarketRegime(
                label="TRENDING_DOWN",
                confidence=min(0.95, 0.60 + abs(nifty_change_1d) * 0.08),
                nifty_trend_5d=trend_5d,
                nifty_trend_20d=trend_20d,
                vix=vix,
                advance_decline=advance_decline,
                fii_net_cr=fii_net_cr,
                is_good_for_shorts=True,
                short_aggressiveness=aggressiveness,
                position_size_multiplier=mult,
                confidence_threshold=thresh,
                max_positions=4 if strong_down else 3,
                description=desc,
            )

        # ── VOLATILE: High VIX, choppy market — trade with tighter stops
        if vix > 18 or atr_pct > 1.5:
            return MarketRegime(
                label="VOLATILE",
                confidence=0.75,
                nifty_trend_5d=trend_5d,
                nifty_trend_20d=trend_20d,
                vix=vix,
                advance_decline=advance_decline,
                fii_net_cr=fii_net_cr,
                is_good_for_shorts=trend_5d < 0,
                short_aggressiveness="CAUTIOUS",
                position_size_multiplier=0.65,
                confidence_threshold=0.60,
                max_positions=2,
                description=(
                    f"Volatile: VIX={vix:.1f}. Choppy price action. "
                    f"Smaller positions, tighter stops, only high-conviction shorts."
                ),
            )

        # ── TRENDING UP: Market rallying — shorts face headwind
        if nifty_change_1d > 1.5 or (trend_5d > 2 and trend_20d > 3):
            return MarketRegime(
                label="TRENDING_UP",
                confidence=min(0.90, 0.55 + trend_5d * 0.05),
                nifty_trend_5d=trend_5d,
                nifty_trend_20d=trend_20d,
                vix=vix,
                advance_decline=advance_decline,
                fii_net_cr=fii_net_cr,
                is_good_for_shorts=False,
                short_aggressiveness="AVOID",
                position_size_multiplier=0.3,
                confidence_threshold=0.80,
                max_positions=1,
                description=(
                    f"Uptrend: Nifty +{nifty_change_1d:.2f}%, 5d={trend_5d:.1f}%. "
                    f"Wait for pullback or overbought reversal. Minimal shorts."
                ),
            )

        # ── RANGING: Sideways — short at resistance, quick exits
        return MarketRegime(
            label="RANGING",
            confidence=0.65,
            nifty_trend_5d=trend_5d,
            nifty_trend_20d=trend_20d,
            vix=vix,
            advance_decline=advance_decline,
            fii_net_cr=fii_net_cr,
            is_good_for_shorts=True,
            short_aggressiveness="NORMAL",
            position_size_multiplier=0.85,
            confidence_threshold=0.52,
            max_positions=3,
            description=(
                f"Ranging: Nifty {nifty_change_1d:+.2f}%. "
                f"Short near resistance, target quick 0.8–1.5% moves."
            ),
        )

    def _get_nifty_trend(self, df, days):
        if df is None or len(df) < days + 1:
            return 0.0
        try:
            closes = df["close"].dropna()
            if len(closes) < days + 1:
                return 0.0
            return round((float(closes.iloc[-1]) - float(closes.iloc[-days-1])) / float(closes.iloc[-days-1]) * 100, 3)
        except Exception:
            return 0.0

    def _get_atr_pct(self, df, period=5):
        if df is None or len(df) < period + 1:
            return 1.0
        try:
            df2 = df.tail(period + 1).copy()
            hl  = df2["high"] - df2["low"]
            hc  = (df2["high"] - df2["close"].shift()).abs()
            lc  = (df2["low"]  - df2["close"].shift()).abs()
            tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr = float(tr.rolling(period).mean().iloc[-1])
            return round(atr / float(df2["close"].iloc[-1]) * 100, 3)
        except Exception:
            return 1.0


regime_detector = MarketRegimeDetector()
