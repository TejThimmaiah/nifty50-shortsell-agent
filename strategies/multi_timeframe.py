"""
Multi-Timeframe Analysis (MTF)
Confirms short signals across three timeframes before entry.
A signal must align on at least 2 of 3 timeframes to qualify.
Dramatically reduces false positives.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import pandas as pd

from agents.technical_analyst import calculate_all, TechnicalSignal
from strategies.candlestick_patterns import pattern_confidence_score, get_best_pattern
from data.nse_fetcher import get_intraday_ohlcv, get_historical_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class MTFSignal:
    symbol: str
    aligned: bool                        # True if ≥2 timeframes agree on SHORT
    alignment_count: int                 # 0, 1, 2, or 3 timeframes aligned
    composite_confidence: float          # weighted confidence 0–1
    signal_5m:  Optional[TechnicalSignal]
    signal_15m: Optional[TechnicalSignal]
    signal_1d:  Optional[TechnicalSignal]
    pattern_5m:  str
    pattern_15m: str
    pattern_1d:  str
    entry_price: float
    stop_loss: float
    target: float
    summary: str


class MultiTimeframeAnalyser:
    """
    Validates short selling signals across 5m, 15m, and daily charts.
    Uses the technical analyst on each timeframe independently.
    """

    # Timeframe weights (daily has highest weight for trend direction)
    WEIGHTS = {"5m": 0.25, "15m": 0.35, "1d": 0.40}

    def analyse(self, symbol: str) -> Optional[MTFSignal]:
        """
        Full MTF analysis for a symbol.
        Returns None if no short signal on any timeframe.
        """
        logger.debug(f"MTF analysis: {symbol}")

        # Fetch all timeframes
        df_5m  = get_intraday_ohlcv(symbol, interval="5m",  period="1d")
        df_15m = get_intraday_ohlcv(symbol, interval="15m", period="5d")
        df_1d  = get_historical_ohlcv(symbol, days=60)

        # Run technical analysis on each
        sig_5m  = calculate_all(df_5m,  symbol) if df_5m  is not None and len(df_5m)  > 20 else None
        sig_15m = calculate_all(df_15m, symbol) if df_15m is not None and len(df_15m) > 20 else None
        sig_1d  = calculate_all(df_1d,  symbol) if df_1d  is not None and len(df_1d)  > 30 else None

        if not any([sig_5m, sig_15m, sig_1d]):
            return None

        # Check which timeframes show SHORT signals
        alignments = []
        confidences = []

        for label, sig, weight in [
            ("5m", sig_5m, self.WEIGHTS["5m"]),
            ("15m", sig_15m, self.WEIGHTS["15m"]),
            ("1d", sig_1d, self.WEIGHTS["1d"]),
        ]:
            if sig and sig.signal in ("STRONG_SHORT", "SHORT"):
                alignments.append(label)
                confidences.append(sig.confidence * weight)
            else:
                confidences.append(0)

        # Candlestick patterns on each timeframe
        pat_5m  = _get_pattern_name(df_5m)
        pat_15m = _get_pattern_name(df_15m)
        pat_1d  = _get_pattern_name(df_1d)

        # Composite confidence
        base_conf = sum(confidences)
        # Bonus for multi-timeframe alignment
        alignment_bonus = {0: 0.0, 1: 0.0, 2: 0.10, 3: 0.20}
        composite = min(1.0, base_conf + alignment_bonus.get(len(alignments), 0))

        # Candlestick pattern bonus
        pattern_score = (
            pattern_confidence_score(df_5m or pd.DataFrame()) * 0.25 +
            pattern_confidence_score(df_15m or pd.DataFrame()) * 0.35 +
            pattern_confidence_score(df_1d or pd.DataFrame()) * 0.40
        )
        composite = min(1.0, composite + pattern_score * 0.15)

        # Best entry parameters — prefer 5m for precise entry, 1d for context
        primary_sig = sig_5m or sig_15m or sig_1d
        entry  = primary_sig.entry_price  if primary_sig else 0
        sl     = primary_sig.stop_loss    if primary_sig else 0
        target = primary_sig.target       if primary_sig else 0

        aligned = len(alignments) >= 2

        # Build summary
        tf_status = []
        for label, sig in [("5m", sig_5m), ("15m", sig_15m), ("1d", sig_1d)]:
            if sig:
                icon = "🔴" if sig.signal in ("STRONG_SHORT", "SHORT") else "⚪"
                tf_status.append(f"{icon}{label}:{sig.signal}")
            else:
                tf_status.append(f"⚫{label}:NO_DATA")

        summary = (
            f"{symbol} MTF [{' | '.join(tf_status)}] | "
            f"Aligned: {len(alignments)}/3 | "
            f"Confidence: {composite:.2f}"
        )
        logger.info(summary)

        return MTFSignal(
            symbol=symbol,
            aligned=aligned,
            alignment_count=len(alignments),
            composite_confidence=round(composite, 3),
            signal_5m=sig_5m,
            signal_15m=sig_15m,
            signal_1d=sig_1d,
            pattern_5m=pat_5m,
            pattern_15m=pat_15m,
            pattern_1d=pat_1d,
            entry_price=entry,
            stop_loss=sl,
            target=target,
            summary=summary,
        )

    def batch_analyse(self, symbols: list) -> Dict[str, Optional[MTFSignal]]:
        """Run MTF analysis on multiple symbols."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.analyse(sym)
            except Exception as e:
                logger.error(f"MTF error [{sym}]: {e}")
                results[sym] = None
        return results

    def is_high_conviction(self, mtf: MTFSignal, min_alignment: int = 2) -> bool:
        """
        Returns True if the MTF signal meets high-conviction criteria.
        min_alignment: minimum number of timeframes that must agree.
        """
        if not mtf or not mtf.aligned:
            return False
        if mtf.alignment_count < min_alignment:
            return False
        if mtf.composite_confidence < 0.50:
            return False
        # Daily must be SHORT or STRONG_SHORT — INSUFFICIENT means skip
        if mtf.signal_1d and mtf.signal_1d.signal not in ("SHORT", "STRONG_SHORT"):
            return False
        return True


def _get_pattern_name(df: Optional[pd.DataFrame]) -> str:
    """Get the best candlestick pattern name for display."""
    if df is None or len(df) < 5:
        return "—"
    pattern = get_best_pattern(df)
    return pattern.name if pattern else "—"
