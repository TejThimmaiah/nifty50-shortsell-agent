"""
Candlestick Pattern Recognition
Identifies bearish reversal patterns that confirm short selling setups.
Patterns are validated against volume and price context.
Uses pure pandas/numpy — no paid libraries.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CandlePattern:
    name: str
    pattern_type: str           # "BEARISH_REVERSAL" | "BEARISH_CONTINUATION" | "NEUTRAL"
    confidence: float           # 0.0–1.0
    candle_index: int           # which bar triggered the pattern (0 = latest)
    description: str


def detect_all_bearish_patterns(df: pd.DataFrame) -> List[CandlePattern]:
    """
    Scan the last 5 bars for all bearish candlestick patterns.
    Returns a list sorted by confidence (highest first).
    df must have: open, high, low, close, volume (lowercase).
    """
    if df is None or len(df) < 5:
        return []

    patterns: List[CandlePattern] = []

    # Single-candle patterns
    p = _bearish_engulfing(df)
    if p: patterns.append(p)

    p = _shooting_star(df)
    if p: patterns.append(p)

    p = _hanging_man(df)
    if p: patterns.append(p)

    p = _gravestone_doji(df)
    if p: patterns.append(p)

    p = _dark_cloud_cover(df)
    if p: patterns.append(p)

    # Two-candle patterns
    p = _evening_star(df)
    if p: patterns.append(p)

    p = _tweezer_top(df)
    if p: patterns.append(p)

    # Three-candle patterns
    p = _three_black_crows(df)
    if p: patterns.append(p)

    p = _three_inside_down(df)
    if p: patterns.append(p)

    patterns.sort(key=lambda x: x.confidence, reverse=True)
    return patterns


def get_best_pattern(df: pd.DataFrame) -> Optional[CandlePattern]:
    """Return the highest-confidence bearish pattern, or None."""
    patterns = detect_all_bearish_patterns(df)
    return patterns[0] if patterns else None


def pattern_confidence_score(df: pd.DataFrame) -> float:
    """Return a 0–1 score based on the best pattern found."""
    patterns = detect_all_bearish_patterns(df)
    if not patterns:
        return 0.0
    # Weight by top 2 patterns
    top2 = patterns[:2]
    score = top2[0].confidence
    if len(top2) > 1:
        score = min(1.0, score + top2[1].confidence * 0.3)
    return round(score, 3)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE CANDLE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def _shooting_star(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Shooting Star:
    - Small real body near the LOW of the candle
    - Upper shadow at least 2x the body size
    - Little or no lower shadow
    - Appears after an uptrend
    """
    c = df.iloc[-1]
    o, h, l, close = c["open"], c["high"], c["low"], c["close"]

    body   = abs(close - o)
    upper  = h - max(o, close)
    lower  = min(o, close) - l
    total  = h - l

    if total == 0:
        return None

    is_small_body  = body <= total * 0.30
    is_long_upper  = upper >= body * 2.0
    is_small_lower = lower <= body * 0.5
    is_after_uptrend = float(df["close"].iloc[-4]) < float(df["close"].iloc[-2])

    if is_small_body and is_long_upper and is_small_lower:
        # Volume confirmation
        vol_ratio = float(c["volume"]) / float(df["volume"].iloc[-6:-1].mean()) if len(df) > 5 else 1
        base_conf = 0.70
        conf = base_conf + (0.15 if vol_ratio > 1.5 else 0) + (0.10 if is_after_uptrend else 0)
        return CandlePattern(
            name="Shooting Star",
            pattern_type="BEARISH_REVERSAL",
            confidence=min(1.0, conf),
            candle_index=0,
            description=f"Long upper shadow ({upper:.2f}), small body ({body:.2f}). Reversal signal.",
        )
    return None


def _bearish_engulfing(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Bearish Engulfing:
    - Previous candle is bullish (close > open)
    - Current candle is bearish and completely engulfs the previous body
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_bullish  = float(prev["close"]) > float(prev["open"])
    curr_bearish  = float(curr["close"]) < float(curr["open"])
    engulfs_body  = (float(curr["open"]) >= float(prev["close"]) and
                     float(curr["close"]) <= float(prev["open"]))

    if prev_bullish and curr_bearish and engulfs_body:
        body_ratio = abs(float(curr["close"]) - float(curr["open"])) / max(
            abs(float(prev["close"]) - float(prev["open"])), 0.01
        )
        conf = 0.75 + min(0.20, (body_ratio - 1) * 0.1)
        return CandlePattern(
            name="Bearish Engulfing",
            pattern_type="BEARISH_REVERSAL",
            confidence=min(1.0, conf),
            candle_index=0,
            description=f"Bearish candle engulfs prior bullish candle. Body ratio: {body_ratio:.2f}x.",
        )
    return None


def _hanging_man(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Hanging Man:
    - Small real body near the TOP of the candle
    - Long lower shadow (at least 2x body)
    - Little or no upper shadow
    - Appears after an uptrend (warning: less reliable alone)
    """
    c = df.iloc[-1]
    o, h, l, close = c["open"], c["high"], c["low"], c["close"]

    body  = abs(close - o)
    upper = h - max(o, close)
    lower = min(o, close) - l
    total = h - l

    if total == 0 or body == 0:
        return None

    is_small_body  = body <= total * 0.30
    is_long_lower  = lower >= body * 2.0
    is_small_upper = upper <= body * 0.5
    is_after_uptrend = len(df) > 5 and float(df["close"].iloc[-5]) < float(df["close"].iloc[-2])

    if is_small_body and is_long_lower and is_small_upper and is_after_uptrend:
        return CandlePattern(
            name="Hanging Man",
            pattern_type="BEARISH_REVERSAL",
            confidence=0.60,
            candle_index=0,
            description=f"Long lower shadow after uptrend. Potential reversal (confirm with volume).",
        )
    return None


def _gravestone_doji(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Gravestone Doji:
    - Open ≈ Close ≈ Low
    - Long upper shadow
    - Strong bearish signal after uptrend
    """
    c = df.iloc[-1]
    o, h, l, close = c["open"], c["high"], c["low"], c["close"]

    total  = h - l
    body   = abs(close - o)
    upper  = h - max(o, close)

    if total == 0:
        return None

    is_doji        = body <= total * 0.10
    is_long_upper  = upper >= total * 0.70
    near_low       = min(o, close) <= l + total * 0.10

    if is_doji and is_long_upper and near_low:
        return CandlePattern(
            name="Gravestone Doji",
            pattern_type="BEARISH_REVERSAL",
            confidence=0.78,
            candle_index=0,
            description=f"Open=Close≈Low, long upper shadow {upper:.2f}. Strong reversal signal.",
        )
    return None


def _dark_cloud_cover(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Dark Cloud Cover:
    - Previous candle: strong bullish
    - Current candle: opens above previous high, closes below midpoint of previous body
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_body   = float(prev["close"]) - float(prev["open"])
    prev_mid    = float(prev["open"]) + prev_body / 2
    curr_opens_above = float(curr["open"]) > float(prev["high"])
    curr_closes_below_mid = float(curr["close"]) < prev_mid
    prev_is_bullish = prev_body > 0

    if prev_is_bullish and curr_opens_above and curr_closes_below_mid:
        penetration = (float(curr["open"]) - float(curr["close"])) / max(prev_body, 0.01)
        conf = 0.70 + min(0.20, penetration * 0.1)
        return CandlePattern(
            name="Dark Cloud Cover",
            pattern_type="BEARISH_REVERSAL",
            confidence=min(1.0, conf),
            candle_index=0,
            description=f"Opens above prior high, closes below prior midpoint. Penetration: {penetration:.2f}x.",
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TWO-CANDLE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def _evening_star(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Evening Star (3-candle):
    1. Strong bullish candle
    2. Small body (star) — gaps up
    3. Strong bearish candle closing into first candle's body
    """
    if len(df) < 3:
        return None

    c1 = df.iloc[-3]   # bullish
    c2 = df.iloc[-2]   # star
    c3 = df.iloc[-1]   # bearish

    c1_body    = float(c1["close"]) - float(c1["open"])
    c2_body    = abs(float(c2["close"]) - float(c2["open"]))
    c3_body    = float(c3["open"]) - float(c3["close"])

    c1_bullish = c1_body > 0
    c3_bearish = c3_body > 0
    c2_is_star = c2_body <= c1_body * 0.30
    c3_closes_into_c1 = float(c3["close"]) < float(c1["open"]) + c1_body * 0.50

    if c1_bullish and c2_is_star and c3_bearish and c3_closes_into_c1:
        return CandlePattern(
            name="Evening Star",
            pattern_type="BEARISH_REVERSAL",
            confidence=0.82,
            candle_index=0,
            description="Classic 3-candle reversal: bullish → star → bearish. High reliability.",
        )
    return None


def _tweezer_top(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Tweezer Top:
    Two consecutive candles with equal (or near-equal) highs after an uptrend.
    Second candle is bearish.
    """
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    high_diff = abs(float(prev["high"]) - float(curr["high"]))
    avg_price = (float(prev["high"]) + float(curr["high"])) / 2
    highs_equal = high_diff / avg_price < 0.002   # within 0.2%

    curr_bearish = float(curr["close"]) < float(curr["open"])

    if highs_equal and curr_bearish:
        return CandlePattern(
            name="Tweezer Top",
            pattern_type="BEARISH_REVERSAL",
            confidence=0.65,
            candle_index=0,
            description=f"Equal highs at ₹{avg_price:.2f} — resistance confirmed. Bearish follow-through.",
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# THREE-CANDLE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def _three_black_crows(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Three Black Crows:
    Three consecutive long bearish candles, each opening within the prior body
    and closing near the low. Confirms strong downtrend.
    """
    if len(df) < 3:
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    def is_long_bearish(c):
        body  = float(c["open"]) - float(c["close"])
        total = float(c["high"]) - float(c["low"])
        return body > 0 and total > 0 and body / total > 0.60

    all_bearish = is_long_bearish(c1) and is_long_bearish(c2) and is_long_bearish(c3)
    descending  = (float(c1["close"]) > float(c2["close"]) > float(c3["close"]))

    # Each opens within prior body
    c2_opens_in_c1 = float(c1["open"]) >= float(c2["open"]) >= float(c1["close"])
    c3_opens_in_c2 = float(c2["open"]) >= float(c3["open"]) >= float(c2["close"])

    if all_bearish and descending and c2_opens_in_c1 and c3_opens_in_c2:
        return CandlePattern(
            name="Three Black Crows",
            pattern_type="BEARISH_CONTINUATION",
            confidence=0.85,
            candle_index=0,
            description="3 consecutive bearish candles — strong downtrend confirmation. High conviction.",
        )
    return None


def _three_inside_down(df: pd.DataFrame) -> Optional[CandlePattern]:
    """
    Three Inside Down:
    1. Large bullish candle
    2. Small bearish candle inside candle 1 (bearish harami)
    3. Bearish candle closing below candle 1's open
    """
    if len(df) < 3:
        return None

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    c1_bullish   = float(c1["close"]) > float(c1["open"])
    c2_inside_c1 = (float(c2["open"]) < float(c1["close"]) and
                    float(c2["close"]) > float(c1["open"]))
    c2_bearish   = float(c2["close"]) < float(c2["open"])
    c3_bearish   = float(c3["close"]) < float(c3["open"])
    c3_confirms  = float(c3["close"]) < float(c1["open"])

    if c1_bullish and c2_inside_c1 and c2_bearish and c3_bearish and c3_confirms:
        return CandlePattern(
            name="Three Inside Down",
            pattern_type="BEARISH_REVERSAL",
            confidence=0.80,
            candle_index=0,
            description="Bearish harami confirmed by strong bearish candle. Reliable reversal.",
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME PATTERN CHECK
# ─────────────────────────────────────────────────────────────────────────────

def multi_timeframe_bearish_score(
    df_5m:  Optional[pd.DataFrame],
    df_15m: Optional[pd.DataFrame],
    df_1d:  Optional[pd.DataFrame],
) -> Tuple[float, List[str]]:
    """
    Check patterns across 3 timeframes.
    Score is highest when all timeframes agree (bearish alignment).
    Returns (score: float, pattern_names: list).
    """
    signals = []
    score   = 0.0

    for label, df, weight in [("5m", df_5m, 0.35), ("15m", df_15m, 0.35), ("1d", df_1d, 0.30)]:
        if df is None:
            continue
        tf_score = pattern_confidence_score(df)
        if tf_score > 0:
            best = get_best_pattern(df)
            if best:
                signals.append(f"{label}: {best.name} ({tf_score:.0%})")
        score += tf_score * weight

    return round(min(1.0, score), 3), signals
