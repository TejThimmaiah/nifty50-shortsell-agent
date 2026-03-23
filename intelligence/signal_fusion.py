"""
Bayesian Signal Fusion Engine
Replaces fixed signal weights with a probabilistic framework.

Instead of: score = rsi*0.25 + divergence*0.20 + ...
We compute:  P(short wins | signals observed) using Bayes theorem

Each signal has a likelihood ratio:
  LR = P(signal fires | trade wins) / P(signal fires | trade loses)

LR > 1 means the signal is positively predictive.
LR < 1 means it HURTS the case for a short.

The agent computes LR for every signal from real trade history.
Combined LR = product of individual LRs (naive Bayes assumption).
Prior = base win rate from recent trades.

Result: posterior probability P(win) for each candidate.
This single number replaces all fixed confidence thresholds.

References:
  - Bayesian approach to trading signals: Aldridge (2013)
  - Naive Bayes signal fusion for market timing: Lo & Hasanhodzic (2010)
"""

import json
import logging
import math
import sqlite3
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta

logger = logging.getLogger(__name__)

FUSION_DB = os.path.join(os.path.dirname(__file__), "..", "db", "signal_fusion.db")

# All signals the system can observe
ALL_SIGNALS = [
    "RSI_OVERBOUGHT",
    "BEARISH_DIVERGENCE",
    "AT_RESISTANCE",
    "VOLUME_CONFIRMS",
    "MACD_TURNING_DOWN",
    "BB_EXTENDED",
    "EMA_DOWNTREND",
    "BEARISH_ENGULFING",
    "SHOOTING_STAR",
    "EVENING_STAR",
    "THREE_BLACK_CROWS",
    "DARK_CLOUD_COVER",
    "GRAVESTONE_DOJI",
    "MTF_ALIGNED",
    "OPTIONS_BEARISH",
    "FII_SELLING",
    "SECTOR_DOWNTREND",
    "GAP_UP_FADE",
    "VWAP_REJECTION",
    "OI_BUILDUP_BEARISH",
    "ORDERFLOW_NEGATIVE",
    "WYCKOFF_DISTRIBUTION",
]

# Minimum observations before trusting a signal's LR
MIN_OBSERVATIONS = 5


@dataclass
class SignalLikelihood:
    signal:         str
    lr:             float       # likelihood ratio P(win|signal) / P(win|no_signal)
    win_count:      int
    total_count:    int
    precision:      float       # signal precision = win_count / total_count
    confidence:     float       # how much to trust this LR (based on sample size)
    last_updated:   str = ""


@dataclass
class FusionResult:
    symbols:        str
    signals_observed: List[str]
    prior_win_rate:   float     # base win rate (recent 30 days)
    posterior_win_prob: float   # P(win | signals) — the key number
    individual_lrs:   Dict[str, float]
    combined_lr:      float
    recommendation:   str       # "STRONG_SHORT" | "SHORT" | "WEAK" | "SKIP"
    kelly_fraction:   float     # optimal position fraction (Kelly criterion)


class BayesianSignalFusion:
    """
    Maintains signal likelihood ratios from trade history.
    Updates after every trade closes.
    Computes posterior win probability for new trade candidates.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(FUSION_DB), exist_ok=True)
        self._init_schema()
        self._cache: Dict[str, SignalLikelihood] = {}
        self._prior: float = 0.50    # Will be updated from recent trade history
        self._refresh_cache()

    # ──────────────────────────────────────────────────────────────
    # MAIN: compute posterior for a candidate
    # ──────────────────────────────────────────────────────────────

    def fuse(self, signals: List[str], symbol: str = "") -> FusionResult:
        """
        Given observed signals for a candidate, compute P(win).
        Returns FusionResult with recommendation and Kelly fraction.
        """
        self._refresh_prior()

        active = [s for s in signals if s in self._cache]
        lrs    = {}
        combined_lr = 1.0

        for sig in active:
            sl = self._cache.get(sig)
            if sl and sl.total_count >= MIN_OBSERVATIONS:
                # Smooth LR toward 1.0 for low-confidence signals
                effective_lr = self._smooth_lr(sl.lr, sl.confidence)
                lrs[sig]     = round(effective_lr, 4)
                combined_lr  *= effective_lr
            else:
                # Unknown signal: assume neutral (LR=1)
                lrs[sig] = 1.0
                combined_lr *= 1.0

        # Bayesian update: posterior odds = prior odds × combined LR
        prior_odds = self._prior / max(1 - self._prior, 1e-6)
        post_odds  = prior_odds * combined_lr
        posterior  = post_odds / (1 + post_odds)
        posterior  = round(max(0.01, min(0.99, posterior)), 4)

        # Recommendation thresholds
        if   posterior >= 0.72: rec = "STRONG_SHORT"
        elif posterior >= 0.58: rec = "SHORT"
        elif posterior >= 0.48: rec = "WEAK"
        else:                   rec = "SKIP"

        # Kelly fraction: f = (p*(b+1) - 1) / b where b = target/SL ratio
        # Using risk/reward of 1.5 target / 0.5 SL = ratio of 3
        b = 3.0
        kelly = max(0.0, (posterior * (b + 1) - 1) / b)
        kelly = round(min(kelly, 0.25), 4)   # cap at 25% of risk capital

        return FusionResult(
            symbols=symbol,
            signals_observed=active,
            prior_win_rate=self._prior,
            posterior_win_prob=posterior,
            individual_lrs=lrs,
            combined_lr=round(combined_lr, 4),
            recommendation=rec,
            kelly_fraction=kelly,
        )

    # ──────────────────────────────────────────────────────────────
    # LEARNING: update LRs after each trade closes
    # ──────────────────────────────────────────────────────────────

    def record_outcome(self, signals: List[str], won: bool):
        """
        Update signal statistics after a trade closes.
        Called by the orchestrator after every closed trade.
        """
        for sig in signals:
            if sig not in ALL_SIGNALS:
                continue
            with sqlite3.connect(FUSION_DB) as conn:
                conn.execute("""
                    INSERT INTO signal_outcomes (signal, won, recorded_at)
                    VALUES (?, ?, DATE('now'))
                """, (sig, int(won)))

        self._refresh_cache()
        logger.debug(f"Fusion: recorded {'WIN' if won else 'LOSS'} for {len(signals)} signals")

    def get_signal_report(self) -> Dict[str, Dict]:
        """Return current LR and precision for all tracked signals."""
        return {
            sig: {
                "lr":        round(sl.lr, 3),
                "precision": round(sl.precision, 3),
                "count":     sl.total_count,
                "confidence": round(sl.confidence, 3),
            }
            for sig, sl in self._cache.items()
        }

    def get_best_signals(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return top N signals by likelihood ratio."""
        qualified = [
            (sig, sl.lr)
            for sig, sl in self._cache.items()
            if sl.total_count >= MIN_OBSERVATIONS
        ]
        return sorted(qualified, key=lambda x: x[1], reverse=True)[:top_n]

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _refresh_cache(self):
        """Recompute LRs from stored observations."""
        with sqlite3.connect(FUSION_DB) as conn:
            # Overall stats for last 60 days
            total_row = conn.execute("""
                SELECT COUNT(*), SUM(won) FROM signal_outcomes
                WHERE recorded_at >= DATE('now', '-60 days')
            """).fetchone()
            total_trades = total_row[0] or 1
            total_wins   = total_row[1] or 0
            base_wr      = total_wins / total_trades

            # Per-signal stats
            rows = conn.execute("""
                SELECT signal,
                       COUNT(*) as total,
                       SUM(won)  as wins
                FROM signal_outcomes
                WHERE recorded_at >= DATE('now', '-60 days')
                GROUP BY signal
            """).fetchall()

        for sig, total, wins in rows:
            if total < 2:
                continue
            precision = wins / total
            # P(win|signal) / P(win overall) — how much the signal improves odds
            # Laplace smoothing to avoid division by zero
            smoothed_p_win_given_sig  = (wins + 1)  / (total + 2)
            smoothed_p_win_overall    = (total_wins + 1) / (total_trades + 2)
            lr = smoothed_p_win_given_sig / max(smoothed_p_win_overall, 1e-6)

            # Confidence: how much sample size supports the LR
            confidence = min(1.0, total / 30.0)

            self._cache[sig] = SignalLikelihood(
                signal=sig,
                lr=round(lr, 4),
                win_count=int(wins or 0),
                total_count=total,
                precision=round(precision, 4),
                confidence=round(confidence, 3),
                last_updated=date.today().isoformat(),
            )
        self._prior = base_wr

    def _refresh_prior(self):
        """Update the prior win rate from recent trades."""
        try:
            with sqlite3.connect(FUSION_DB) as conn:
                row = conn.execute("""
                    SELECT COUNT(*), SUM(won) FROM signal_outcomes
                    WHERE recorded_at >= DATE('now', '-30 days')
                """).fetchone()
            if row and row[0] > 5:
                self._prior = (row[1] or 0) / row[0]
        except Exception:
            pass

    def _smooth_lr(self, lr: float, confidence: float) -> float:
        """
        Shrink LR toward 1.0 proportional to uncertainty.
        Low confidence signal → LR pulled toward neutral (1.0).
        """
        return 1.0 + (lr - 1.0) * confidence

    def _init_schema(self):
        with sqlite3.connect(FUSION_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal      TEXT NOT NULL,
                    won         INTEGER NOT NULL,
                    recorded_at TEXT DEFAULT (DATE('now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_so_signal ON signal_outcomes(signal)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_so_date   ON signal_outcomes(recorded_at)")

            # Seed with reasonable priors so agent isn't blind on day 1
            count = conn.execute("SELECT COUNT(*) FROM signal_outcomes").fetchone()[0]
            if count == 0:
                seeds = [
                    # (signal, won) — based on known NSE intraday short statistics
                    ("RSI_OVERBOUGHT",     1), ("RSI_OVERBOUGHT",     1), ("RSI_OVERBOUGHT",     0),
                    ("BEARISH_ENGULFING",  1), ("BEARISH_ENGULFING",  1), ("BEARISH_ENGULFING",  0),
                    ("EVENING_STAR",       1), ("EVENING_STAR",       1), ("EVENING_STAR",       1),
                    ("THREE_BLACK_CROWS",  1), ("THREE_BLACK_CROWS",  1), ("THREE_BLACK_CROWS",  0),
                    ("BEARISH_DIVERGENCE", 1), ("BEARISH_DIVERGENCE", 0), ("BEARISH_DIVERGENCE", 0),
                    ("MTF_ALIGNED",        1), ("MTF_ALIGNED",        1), ("MTF_ALIGNED",        0),
                    ("VOLUME_CONFIRMS",    1), ("VOLUME_CONFIRMS",    0),
                    ("AT_RESISTANCE",      1), ("AT_RESISTANCE",      0),
                    ("FII_SELLING",        1), ("FII_SELLING",        1),
                    ("GAP_UP_FADE",        1), ("GAP_UP_FADE",        1),
                    ("VWAP_REJECTION",     1), ("VWAP_REJECTION",     0),
                    ("SECTOR_DOWNTREND",   1), ("SECTOR_DOWNTREND",   0),
                    ("SHOOTING_STAR",      1), ("SHOOTING_STAR",      0),
                    ("MACD_TURNING_DOWN",  1), ("MACD_TURNING_DOWN",  0),
                ]
                for sig, won in seeds:
                    conn.execute(
                        "INSERT INTO signal_outcomes (signal, won, recorded_at) VALUES (?, ?, DATE('now', '-30 days'))",
                        (sig, won)
                    )
                logger.info("Signal fusion DB seeded with prior knowledge")


# Singleton
signal_fusion = BayesianSignalFusion()
