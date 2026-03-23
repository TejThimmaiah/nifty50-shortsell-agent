"""
Volume Profile Analyser
Volume Profile shows WHERE trading activity is concentrated — not just WHEN.
It's the vertical distribution of volume at each price level.

Key levels derived from Volume Profile:
  POC  (Point of Control)    — price with highest traded volume
  VAH  (Value Area High)     — top of 70% of volume (institutional value area)
  VAL  (Value Area Low)      — bottom of 70% of volume
  HVN  (High Volume Node)    — price clusters with heavy volume = strong S/R
  LVN  (Low Volume Node)     — price gaps with thin volume = price moves fast through here

Short selling uses:
  1. Short at POC from below (price rejected at POC = strong resistance)
  2. Short at VAH (institutional value area ceiling)
  3. Short when price in LVN heading down (no support — accelerates)
  4. Price above POC = currently in premium zone = short opportunity

This is the professional trader's preferred method for identifying support/resistance
over arbitrary chart-based S/R lines.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileLevel:
    price:    float
    volume:   float
    pct_of_total: float    # 0–1


@dataclass
class VolumeProfile:
    symbol:       str
    poc:          float      # Point of Control (highest volume price)
    vah:          float      # Value Area High
    val:          float      # Value Area Low
    hvn_levels:   List[float]   # High Volume Nodes
    lvn_levels:   List[float]   # Low Volume Nodes
    current_price: float
    price_zone:   str        # "ABOVE_VAH" | "IN_VALUE" | "BELOW_VAL"
    poc_distance_pct: float  # % distance from current price to POC
    short_signals: List[str]
    short_score:  float      # 0–1 from volume profile perspective


class VolumeProfileAnalyser:
    """
    Builds and analyses volume profiles from OHLCV data.
    Uses price-range bucketing to estimate volume at each price level.
    """

    # Number of price buckets for volume profile
    N_BUCKETS = 40

    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        lookback: int = 20,
    ) -> Optional[VolumeProfile]:
        """
        Build volume profile from last `lookback` bars.
        Returns key price levels and short signals.
        """
        if df is None or len(df) < 5:
            return None

        try:
            data = df.tail(lookback).copy()
            profile = self._build_profile(data)
            if not profile:
                return None

            poc    = self._find_poc(profile)
            va     = self._find_value_area(profile, poc, target_pct=0.70)
            hvns   = self._find_hvn(profile, poc)
            lvns   = self._find_lvn(profile)

            current = float(data["close"].iloc[-1])
            poc_dist = (current - poc) / poc * 100

            # Classify price zone
            if   current > va["vah"]: zone = "ABOVE_VAH"
            elif current < va["val"]: zone = "BELOW_VAL"
            else:                     zone = "IN_VALUE"

            # Short signals from volume profile
            signals = []
            score   = 0.0

            if zone == "ABOVE_VAH":
                signals.append(f"Price above Value Area High {va['vah']:.2f} — premium zone")
                score += 0.30
            if current > poc and poc_dist < 2.0:
                signals.append(f"Near POC {poc:.2f} from above — strong resistance")
                score += 0.25
            if any(abs(current - h) / current < 0.005 for h in hvns if h > current):
                signals.append("HVN resistance cluster above")
                score += 0.20
            if current > poc:
                signals.append(f"Above POC {poc:.2f} — in distribution zone")
                score += 0.15
            # LVN below = price will accelerate if it breaks down
            if any(lvn < current and abs(lvn - current) / current < 0.015 for lvn in lvns):
                signals.append("LVN below — minimal support, sharp move likely")
                score += 0.10

            return VolumeProfile(
                symbol=symbol,
                poc=round(poc, 2),
                vah=round(va["vah"], 2),
                val=round(va["val"], 2),
                hvn_levels=[round(h, 2) for h in hvns[:3]],
                lvn_levels=[round(l, 2) for l in lvns[:3]],
                current_price=current,
                price_zone=zone,
                poc_distance_pct=round(poc_dist, 3),
                short_signals=signals,
                short_score=round(min(1.0, score), 3),
            )

        except Exception as e:
            logger.error(f"VolumeProfile error [{symbol}]: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # BUILD PROFILE
    # ──────────────────────────────────────────────────────────────

    def _build_profile(self, df: pd.DataFrame) -> Optional[Dict[float, float]]:
        """Build {price_level: volume} dict from OHLCV data."""
        all_low  = float(df["low"].min())
        all_high = float(df["high"].max())
        if all_high <= all_low:
            return None

        # Create price buckets
        bucket_size = (all_high - all_low) / self.N_BUCKETS
        if bucket_size < 1e-6:
            return None

        profile: Dict[float, float] = {}

        for _, row in df.iterrows():
            bar_low  = float(row["low"])
            bar_high = float(row["high"])
            bar_vol  = float(row["volume"])
            bar_range = bar_high - bar_low

            if bar_range < 1e-6:
                # Doji — all volume at close
                bucket = self._price_to_bucket(float(row["close"]), all_low, bucket_size)
                profile[bucket] = profile.get(bucket, 0) + bar_vol
                continue

            # Distribute bar volume across price range
            # Assume triangular distribution: most volume near close
            close_position = (float(row["close"]) - bar_low) / bar_range

            for i in range(self.N_BUCKETS):
                bucket_price = all_low + i * bucket_size
                if bucket_price < bar_low or bucket_price > bar_high:
                    continue
                # Weight: more volume near close (triangular distribution)
                distance_from_close = abs((bucket_price - bar_low) / bar_range - close_position)
                weight = max(0.1, 1.0 - distance_from_close * 1.5)
                profile[round(bucket_price, 2)] = profile.get(round(bucket_price, 2), 0) + bar_vol * weight

        return profile

    def _price_to_bucket(self, price: float, low: float, bucket_size: float) -> float:
        bucket_idx = int((price - low) / bucket_size)
        return round(low + bucket_idx * bucket_size, 2)

    # ──────────────────────────────────────────────────────────────
    # KEY LEVELS
    # ──────────────────────────────────────────────────────────────

    def _find_poc(self, profile: Dict[float, float]) -> float:
        """Price with the highest volume = Point of Control."""
        return max(profile, key=profile.get)

    def _find_value_area(
        self, profile: Dict[float, float], poc: float, target_pct: float = 0.70
    ) -> Dict[str, float]:
        """
        Value Area: the range containing target_pct% of total volume.
        Starts at POC and expands outward.
        """
        total_vol    = sum(profile.values())
        target_vol   = total_vol * target_pct
        sorted_prices = sorted(profile.keys())

        poc_idx  = sorted_prices.index(poc) if poc in sorted_prices else len(sorted_prices) // 2
        va_low   = poc
        va_high  = poc
        included = profile.get(poc, 0)

        lo_idx = poc_idx - 1
        hi_idx = poc_idx + 1

        while included < target_vol:
            lo_vol = profile.get(sorted_prices[lo_idx], 0) if lo_idx >= 0 else 0
            hi_vol = profile.get(sorted_prices[hi_idx], 0) if hi_idx < len(sorted_prices) else 0

            if lo_vol == 0 and hi_vol == 0:
                break
            if lo_vol >= hi_vol:
                if lo_idx >= 0:
                    va_low = sorted_prices[lo_idx]
                    included += lo_vol
                    lo_idx -= 1
            else:
                if hi_idx < len(sorted_prices):
                    va_high = sorted_prices[hi_idx]
                    included += hi_vol
                    hi_idx += 1

        return {"vah": va_high, "val": va_low, "pct_included": included / total_vol}

    def _find_hvn(self, profile: Dict[float, float], poc: float, top_n: int = 5) -> List[float]:
        """High Volume Nodes: price clusters with significant volume concentration."""
        avg_vol = sum(profile.values()) / max(len(profile), 1)
        hvns = [
            price for price, vol in profile.items()
            if vol >= avg_vol * 1.8 and abs(price - poc) / max(poc, 1) > 0.003
        ]
        return sorted(hvns, key=lambda p: profile[p], reverse=True)[:top_n]

    def _find_lvn(self, profile: Dict[float, float], top_n: int = 5) -> List[float]:
        """Low Volume Nodes: price areas with thin volume (fast price movement zones)."""
        avg_vol = sum(profile.values()) / max(len(profile), 1)
        lvns = [
            price for price, vol in profile.items()
            if vol <= avg_vol * 0.3
        ]
        return sorted(lvns)[:top_n]


# Singleton
volume_profile_analyser = VolumeProfileAnalyser()
