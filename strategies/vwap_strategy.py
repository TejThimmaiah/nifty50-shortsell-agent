"""
VWAP Strategy
VWAP (Volume Weighted Average Price) is the most important institutional
reference level for intraday trading. Institutions benchmark to VWAP.

VWAP Short Setups:
1. VWAP Rejection      — price rallies to VWAP from below, gets rejected → short
2. VWAP Breakdown      — price breaks below VWAP on high volume → short
3. VWAP Band Extension — price extends 2 std-devs above VWAP → mean-reversion short
4. VWAP Re-test Fail   — price tests VWAP from above, fails → continuation short

Anchored VWAP:
- We compute VWAP anchored from today's open (intraday)
- Also from weekly open (for context)

Why VWAP works for shorts:
  Institutions sell above VWAP (better than average price)
  They are comfortable being short if price is above VWAP
  Retail algorithms target VWAP for execution — creates self-fulfilling levels
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VWAPSetup:
    symbol:             str
    setup_type:         str         # REJECTION | BREAKDOWN | EXTENSION | RETEST_FAIL
    vwap:               float
    upper_band_1:       float
    upper_band_2:       float
    lower_band_1:       float
    current_price:      float
    price_vs_vwap_pct:  float       # how far above VWAP (positive = above)
    entry_trigger:      float       # price level that confirms setup
    stop_loss:          float
    target:             float
    confidence:         float       # 0–1
    volume_confirms:    bool
    description:        str


class VWAPStrategy:
    """
    Identifies VWAP-based short setups from intraday data.
    Returns high-quality entry signals with precise levels.
    """

    # Standard deviations for bands
    STD_MULTIPLIERS = [1.0, 2.0, 3.0]

    def analyse(self, df: pd.DataFrame, symbol: str = "") -> Optional[VWAPSetup]:
        """
        Find the best VWAP setup in the current bar.
        Returns None if no clean setup.
        """
        if df is None or len(df) < 10:
            return None

        bands = self._compute_bands(df)
        price = float(df["close"].iloc[-1])
        vol   = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].tail(20).mean())

        setups = [
            self._check_rejection(df, bands, price),
            self._check_breakdown(df, bands, price, vol, avg_vol),
            self._check_extension(df, bands, price, vol, avg_vol),
            self._check_retest_fail(df, bands, price),
        ]
        setups = [s for s in setups if s is not None]
        if not setups:
            return None

        # Return highest-confidence setup
        best = max(setups, key=lambda s: s.confidence)
        best.symbol = symbol
        logger.info(
            f"VWAP setup [{symbol}]: {best.setup_type} | "
            f"price={price:.2f} VWAP={best.vwap:.2f} conf={best.confidence:.2f}"
        )
        return best

    # ──────────────────────────────────────────────────────────────
    # VWAP BANDS
    # ──────────────────────────────────────────────────────────────

    def _compute_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute VWAP and standard deviation bands."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol     = df["volume"].replace(0, 1)
        vwap    = float((typical * vol).sum() / vol.sum())

        # Standard deviation of typical price from VWAP
        deviation = float(np.sqrt(((typical - vwap) ** 2 * vol).sum() / vol.sum()))

        return {
            "vwap":    vwap,
            "u1":      vwap + deviation,
            "u2":      vwap + deviation * 2,
            "u3":      vwap + deviation * 3,
            "l1":      vwap - deviation,
            "l2":      vwap - deviation * 2,
            "l3":      vwap - deviation * 3,
            "dev":     deviation,
        }

    # ──────────────────────────────────────────────────────────────
    # SETUP 1: VWAP REJECTION
    # ──────────────────────────────────────────────────────────────

    def _check_rejection(self, df, bands, price) -> Optional[VWAPSetup]:
        """
        VWAP Rejection: price came up to VWAP level and failed.
        Required: price was below VWAP, spiked to/above VWAP, then closed back below.
        """
        vwap = bands["vwap"]
        recent = df.tail(4)
        if len(recent) < 3:
            return None

        # Check if recent candles touched VWAP
        touched_vwap = any(
            float(row["high"]) >= vwap * 0.998 and float(row["low"]) < vwap
            for _, row in recent.iterrows()
        )
        # Current price is below VWAP
        below_vwap = price < vwap

        if touched_vwap and below_vwap:
            pct_below = (vwap - price) / vwap * 100
            conf = min(0.85, 0.55 + pct_below * 0.05)

            return VWAPSetup(
                symbol="",
                setup_type="REJECTION",
                vwap=round(vwap, 2),
                upper_band_1=round(bands["u1"], 2),
                upper_band_2=round(bands["u2"], 2),
                lower_band_1=round(bands["l1"], 2),
                current_price=price,
                price_vs_vwap_pct=round((price - vwap) / vwap * 100, 3),
                entry_trigger=round(price * 0.998, 2),        # slight momentum confirmation
                stop_loss=round(vwap * 1.003, 2),             # just above VWAP
                target=round(bands["l1"], 2),                  # lower band
                confidence=round(conf, 3),
                volume_confirms=True,
                description=f"VWAP rejected at {vwap:.2f}. Price now {pct_below:.2f}% below.",
            )
        return None

    # ──────────────────────────────────────────────────────────────
    # SETUP 2: VWAP BREAKDOWN
    # ──────────────────────────────────────────────────────────────

    def _check_breakdown(self, df, bands, price, vol, avg_vol) -> Optional[VWAPSetup]:
        """
        VWAP Breakdown: price breaks below VWAP on high volume.
        Previous candles were above VWAP.
        """
        vwap = bands["vwap"]
        if len(df) < 5:
            return None

        prev_price = float(df["close"].iloc[-2])
        was_above  = prev_price > vwap
        now_below  = price < vwap * 0.999
        high_vol   = vol >= avg_vol * 1.5

        if was_above and now_below:
            conf = 0.65
            if high_vol:
                conf += 0.15     # volume confirms the breakdown
            return VWAPSetup(
                symbol="",
                setup_type="BREAKDOWN",
                vwap=round(vwap, 2),
                upper_band_1=round(bands["u1"], 2),
                upper_band_2=round(bands["u2"], 2),
                lower_band_1=round(bands["l1"], 2),
                current_price=price,
                price_vs_vwap_pct=round((price - vwap) / vwap * 100, 3),
                entry_trigger=round(price, 2),
                stop_loss=round(vwap * 1.005, 2),
                target=round(bands["l1"], 2),
                confidence=round(conf, 3),
                volume_confirms=high_vol,
                description=f"VWAP breakdown {'on high volume' if high_vol else ''}. Prev: {prev_price:.2f} above {vwap:.2f}.",
            )
        return None

    # ──────────────────────────────────────────────────────────────
    # SETUP 3: UPPER BAND EXTENSION (mean reversion short)
    # ──────────────────────────────────────────────────────────────

    def _check_extension(self, df, bands, price, vol, avg_vol) -> Optional[VWAPSetup]:
        """
        Extension short: price is 2+ std deviations above VWAP.
        Mean reversion is statistically likely.
        Works best in ranging markets.
        """
        vwap = bands["vwap"]
        u2   = bands["u2"]

        if price < u2:
            return None

        extension_pct = (price - vwap) / bands["dev"] if bands["dev"] > 0 else 0
        high_vol      = vol >= avg_vol * 2.0   # institutions distributing at extension

        conf = 0.55 + min(0.25, (extension_pct - 2) * 0.1)
        if high_vol:
            conf += 0.10

        return VWAPSetup(
            symbol="",
            setup_type="EXTENSION",
            vwap=round(vwap, 2),
            upper_band_1=round(bands["u1"], 2),
            upper_band_2=round(u2, 2),
            lower_band_1=round(bands["l1"], 2),
            current_price=price,
            price_vs_vwap_pct=round((price - vwap) / vwap * 100, 3),
            entry_trigger=round(price, 2),
            stop_loss=round(price * 1.006, 2),
            target=round(bands["u1"], 2),           # mean-revert to U1
            confidence=round(min(0.82, conf), 3),
            volume_confirms=high_vol,
            description=f"Price {extension_pct:.1f}σ above VWAP. Mean reversion short. Target VWAP+1σ.",
        )

    # ──────────────────────────────────────────────────────────────
    # SETUP 4: VWAP RE-TEST FAIL
    # ──────────────────────────────────────────────────────────────

    def _check_retest_fail(self, df, bands, price) -> Optional[VWAPSetup]:
        """
        Re-test fail: price broke below VWAP earlier, came back up to test it,
        failed to reclaim, and is now heading back down.
        Classic distribution pattern.
        """
        vwap = bands["vwap"]
        if len(df) < 8:
            return None

        closes = df["close"].values
        highs  = df["high"].values

        # Look for: was above VWAP → dropped below → came back near VWAP → failed
        last_8 = closes[-8:]
        peak_idx = np.argmax(highs[-8:])

        was_above    = any(c > vwap for c in last_8[:4])
        came_back    = any(h >= vwap * 0.998 for h in highs[-4:])
        still_below  = price < vwap * 0.999
        peak_was_mid = 2 <= peak_idx <= 6

        if was_above and came_back and still_below and peak_was_mid:
            return VWAPSetup(
                symbol="",
                setup_type="RETEST_FAIL",
                vwap=round(vwap, 2),
                upper_band_1=round(bands["u1"], 2),
                upper_band_2=round(bands["u2"], 2),
                lower_band_1=round(bands["l1"], 2),
                current_price=price,
                price_vs_vwap_pct=round((price - vwap) / vwap * 100, 3),
                entry_trigger=round(price * 0.997, 2),
                stop_loss=round(vwap * 1.004, 2),
                target=round(bands["l2"], 2),           # deeper target after failed re-test
                confidence=0.78,
                volume_confirms=False,
                description=f"Failed VWAP re-test. Distribution pattern. Target L2={bands['l2']:.2f}.",
            )
        return None

    # ──────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────

    def get_vwap_only(self, df: pd.DataFrame) -> float:
        """Fast VWAP calculation."""
        if df is None or df.empty:
            return 0.0
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol     = df["volume"].replace(0, 1)
        return float((typical * vol).sum() / vol.sum())

    def price_position_vs_vwap(self, price: float, vwap: float) -> str:
        """Classify price position relative to VWAP."""
        pct = (price - vwap) / vwap * 100
        if   pct >  1.0: return "FAR_ABOVE"
        elif pct >  0.2: return "ABOVE"
        elif pct > -0.2: return "AT"
        elif pct > -1.0: return "BELOW"
        else:            return "FAR_BELOW"


# Singleton
vwap_strategy = VWAPStrategy()
