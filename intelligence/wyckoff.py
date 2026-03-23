"""
Wyckoff Distribution Analyser
Richard Wyckoff's methodology identifies how institutional traders
(the "composite operator") distribute (sell) shares to retail buyers.

Distribution phases (the short seller's setup):
  Phase A: Preliminary Supply (PSY) — first heavy selling
  Phase B: Distribution range forms — institutions sell into rallies
  Phase C: Upthrust (UT/UTAD) — false breakout to trap longs
  Phase D: Sign of Weakness (SOW) — price breaks support
  Phase E: Markdown — the actual downtrend begins

For intraday short selling, we focus on:
  1. Upthrust (UT) — false breakout above range high on high volume,
                     then reversal → best short entry
  2. Sign of Weakness (SOW) — break below range support on volume
  3. Distribution Range — wide trading range after a rally = smart money selling

These patterns appear on all timeframes including 5m and 15m.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WyckoffSignal:
    pattern:            str         # UPTHRUST | SIGN_OF_WEAKNESS | DISTRIBUTION_RANGE | SPRING_FAILED
    phase:              str         # A | B | C | D | E
    confidence:         float       # 0–1
    range_high:         float       # upper bound of trading range
    range_low:          float       # lower bound of trading range
    current_price:      float
    entry_price:        float
    stop_loss:          float
    target:             float
    volume_confirms:    bool
    institutional_tells: List[str]   # specific clues found
    description:        str


class WyckoffAnalyser:
    """
    Detects Wyckoff distribution patterns from OHLCV data.
    Works best on 15m and daily charts.
    Provides the highest conviction short signals in the system.
    """

    def analyse(self, df: pd.DataFrame, symbol: str = "") -> Optional[WyckoffSignal]:
        """
        Identify the most significant Wyckoff pattern in the data.
        Returns the highest-confidence pattern found, or None.
        """
        if df is None or len(df) < 30:
            return None

        patterns = []

        ut = self._detect_upthrust(df)
        if ut:
            patterns.append(ut)

        sow = self._detect_sign_of_weakness(df)
        if sow:
            patterns.append(sow)

        dr = self._detect_distribution_range(df)
        if dr:
            patterns.append(dr)

        if not patterns:
            return None

        best = max(patterns, key=lambda p: p.confidence)
        logger.info(
            f"Wyckoff [{symbol}]: {best.pattern} (phase {best.phase}) "
            f"conf={best.confidence:.2f} | {best.description[:60]}"
        )
        return best

    # ──────────────────────────────────────────────────────────────
    # UPTHRUST (UT) — Phase C: the key short entry
    # ──────────────────────────────────────────────────────────────

    def _detect_upthrust(self, df: pd.DataFrame) -> Optional[WyckoffSignal]:
        """
        Upthrust: price briefly breaks above the recent trading range high
        (trapping breakout buyers), then reverses sharply back into range.
        This is the highest-probability Wyckoff short entry.

        Conditions:
          1. Establish a range high from last 20 bars
          2. Current or recent bar breaks above range high (by 0.2%+)
          3. Bar closes back INSIDE the range (tail above, body inside)
          4. Volume is elevated (institutions selling into the breakout)
        """
        if len(df) < 25:
            return None

        # Define range from bars 25–6 bars ago (give context)
        range_data  = df.iloc[-25:-5]
        range_high  = float(range_data["high"].max())
        range_low   = float(range_data["low"].min())
        range_size  = range_high - range_low

        if range_size < 0.5:    # Degenerate range
            return None

        recent      = df.tail(5)
        vol_avg     = float(df["volume"].tail(20).mean())

        for _, row in recent.iterrows():
            bar_high  = float(row["high"])
            bar_close = float(row["close"])
            bar_vol   = float(row["volume"])

            # Broke above range AND closed back inside
            broke_above = bar_high > range_high * 1.002
            closed_inside = bar_close < range_high * 1.001
            high_vol    = bar_vol >= vol_avg * 1.5

            if broke_above and closed_inside:
                tells = ["False breakout above range high", "Close back inside range"]
                conf  = 0.70
                if high_vol:
                    conf  += 0.12
                    tells.append("High volume confirms distribution")
                if bar_close < (range_high + range_low) / 2:
                    conf  += 0.08
                    tells.append("Closed below range midpoint — strong rejection")

                return WyckoffSignal(
                    pattern="UPTHRUST",
                    phase="C",
                    confidence=round(min(0.90, conf), 3),
                    range_high=round(range_high, 2),
                    range_low=round(range_low, 2),
                    current_price=float(df["close"].iloc[-1]),
                    entry_price=round(bar_close * 0.999, 2),
                    stop_loss=round(bar_high * 1.003, 2),    # just above the UT high
                    target=round(range_low, 2),              # target range low
                    volume_confirms=high_vol,
                    institutional_tells=tells,
                    description=f"Upthrust above {range_high:.2f}, closed at {bar_close:.2f}. Bearish trap.",
                )
        return None

    # ──────────────────────────────────────────────────────────────
    # SIGN OF WEAKNESS (SOW) — Phase D: confirmation of markdown
    # ──────────────────────────────────────────────────────────────

    def _detect_sign_of_weakness(self, df: pd.DataFrame) -> Optional[WyckoffSignal]:
        """
        Sign of Weakness: price breaks below the trading range support
        on increased volume (institutions no longer defending the level).
        This confirms the distribution is complete; markdown begins.

        Best entered on re-test of the broken support from below.
        """
        if len(df) < 20:
            return None

        range_data  = df.iloc[-25:-3]
        support     = float(range_data["low"].quantile(0.15))   # lower 15% = support zone
        vol_avg     = float(df["volume"].tail(20).mean())

        recent      = df.tail(3)
        breaks      = []
        for _, row in recent.iterrows():
            if float(row["close"]) < support * 0.999:
                breaks.append(float(row["volume"]))

        if not breaks:
            return None

        avg_break_vol = sum(breaks) / len(breaks)
        high_vol      = avg_break_vol >= vol_avg * 1.4
        current_price = float(df["close"].iloc[-1])

        tells = [f"Support {support:.2f} broken"]
        conf  = 0.65
        if high_vol:
            conf += 0.12
            tells.append("Volume surge confirms smart money exiting")
        if current_price < support:
            conf += 0.08
            tells.append("Holding below former support (now resistance)")

        return WyckoffSignal(
            pattern="SIGN_OF_WEAKNESS",
            phase="D",
            confidence=round(min(0.85, conf), 3),
            range_high=float(range_data["high"].max()),
            range_low=support,
            current_price=current_price,
            entry_price=round(current_price, 2),
            stop_loss=round(support * 1.005, 2),    # above broken support
            target=round(support - (float(range_data["high"].max()) - support) * 0.5, 2),
            volume_confirms=high_vol,
            institutional_tells=tells,
            description=f"SOW: support {support:.2f} broken. Markdown phase starting.",
        )

    # ──────────────────────────────────────────────────────────────
    # DISTRIBUTION RANGE DETECTION
    # ──────────────────────────────────────────────────────────────

    def _detect_distribution_range(self, df: pd.DataFrame) -> Optional[WyckoffSignal]:
        """
        Identifies a tight trading range after an uptrend.
        In Wyckoff terms: Phase B — institutions distributing while price moves sideways.
        Shallow range after a strong rally = accumulation of short positions by smart money.
        """
        if len(df) < 30:
            return None

        # Check for prior uptrend
        pre_range   = df.iloc[-35:-20]
        in_range    = df.iloc[-20:]
        prior_trend = float(pre_range["close"].iloc[-1]) - float(pre_range["close"].iloc[0])
        if prior_trend <= 0:
            return None    # No prior uptrend

        # Range: high - low / midpoint should be < 3%
        r_high    = float(in_range["high"].max())
        r_low     = float(in_range["low"].min())
        r_mid     = (r_high + r_low) / 2
        range_pct = (r_high - r_low) / r_mid * 100

        if range_pct > 4.0:
            return None    # Too wide to be a clean range

        current_price = float(df["close"].iloc[-1])

        # Classify by position in range
        pos_in_range = (current_price - r_low) / max(r_high - r_low, 1e-6)
        at_top       = pos_in_range > 0.75

        tells = [f"Tight range {range_pct:.1f}% after uptrend"]
        conf  = 0.55
        if at_top:
            conf  += 0.12
            tells.append("Price at upper range — best short zone")

        # Declining volume in range = distribution exhaustion
        vol_in_range = float(in_range["volume"].mean())
        vol_prior    = float(pre_range["volume"].mean())
        if vol_in_range < vol_prior * 0.75:
            conf  += 0.08
            tells.append("Declining volume in range (supply absorbed)")

        return WyckoffSignal(
            pattern="DISTRIBUTION_RANGE",
            phase="B",
            confidence=round(min(0.78, conf), 3),
            range_high=round(r_high, 2),
            range_low=round(r_low, 2),
            current_price=current_price,
            entry_price=round(r_high * 0.998, 2),   # near range top
            stop_loss=round(r_high * 1.005, 2),
            target=round(r_low, 2),
            volume_confirms=vol_in_range < vol_prior * 0.75,
            institutional_tells=tells,
            description=f"Distribution range {r_low:.2f}–{r_high:.2f} ({range_pct:.1f}%). Short near top.",
        )

    # ──────────────────────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────────────────────

    def is_wyckoff_short(self, df: pd.DataFrame) -> bool:
        """Quick boolean check — is there any Wyckoff short signal?"""
        signal = self.analyse(df)
        return signal is not None and signal.confidence >= 0.65


# Singleton
wyckoff_analyser = WyckoffAnalyser()
