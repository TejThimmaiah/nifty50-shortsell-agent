"""
Adaptive Configuration
Parameters that self-update every night based on recent trade performance.
The agent doesn't use fixed RSI=70, SL=0.5% forever.
It adjusts to what's actually working in current market conditions.

Bounds are enforced so the agent can't go dangerously out of range.
Every change is logged with a reason. Changes are small and gradual.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

ADAPTIVE_DB = os.path.join(os.path.dirname(__file__), "..", "db", "adaptive_config.db")


@dataclass
class AdaptiveParams:
    """Live trading parameters. Updated nightly by the learning system."""
    date_updated:          str   = ""

    # RSI thresholds — may shift if higher RSI gives better results
    rsi_overbought:        float = 70.0          # bounds: 65–80
    rsi_period:            int   = 14            # bounds: 10–21

    # Risk parameters — adapt based on recent win rate
    stop_loss_pct:         float = 0.5           # bounds: 0.3–1.2
    target_pct:            float = 1.5           # bounds: 1.0–3.0
    trailing_sl_pct:       float = 0.3           # bounds: 0.2–0.6
    trailing_activate_pct: float = 0.8           # bounds: 0.5–1.5

    # Position sizing — adapts to recent drawdown
    max_risk_per_trade_pct: float = 2.0          # bounds: 0.5–3.0
    max_positions:         int   = 3             # bounds: 1–5
    position_size_multiplier: float = 1.0        # regime-based multiplier

    # Confidence gate — raises when agent is losing, lowers when winning
    min_confidence:        float = 0.50          # bounds: 0.35–0.80
    require_pattern:       bool  = False         # True when struggling
    require_mtf_alignment: bool  = True
    min_mtf_timeframes:    int   = 2             # 2 of 3 timeframes

    # Volume filter
    volume_multiplier:     float = 1.5           # bounds: 1.0–2.5

    # Time filter — avoid losing time windows
    skip_early_morning:    bool  = False         # skip 9:20–9:45 if not working
    skip_late_afternoon:   bool  = False         # skip after 12:30 if not working

    # Signal weights (0–1, sum doesn't need to equal 1)
    weight_rsi:            float = 0.25
    weight_divergence:     float = 0.20
    weight_resistance:     float = 0.20
    weight_volume:         float = 0.15
    weight_macd:           float = 0.10
    weight_bb:             float = 0.05
    weight_ema:            float = 0.05

    # Pattern weights (adjusted based on recent pattern performance)
    pattern_weights:       Dict  = None

    # Regime-specific overrides
    regime_overrides:      Dict  = None

    # Metadata
    update_reason:         str   = "default"
    days_since_update:     int   = 0
    performance_window:    int   = 14            # days of data used for adaptation

    def __post_init__(self):
        if self.pattern_weights is None:
            self.pattern_weights = {
                "Bearish Engulfing":    1.0,
                "Evening Star":         1.0,
                "Three Black Crows":    1.0,
                "Shooting Star":        0.85,
                "Dark Cloud Cover":     0.85,
                "Gravestone Doji":      0.80,
                "Three Inside Down":    0.80,
                "Tweezer Top":          0.70,
                "Hanging Man":          0.60,
            }
        if self.regime_overrides is None:
            self.regime_overrides = {}


# ── Bounds enforcement ────────────────────────────────────────────────────────
BOUNDS = {
    "rsi_overbought":           (65.0, 80.0),
    "stop_loss_pct":            (0.3,  1.2),
    "target_pct":               (1.0,  3.0),
    "trailing_sl_pct":          (0.2,  0.6),
    "trailing_activate_pct":    (0.5,  1.5),
    "max_risk_per_trade_pct":   (0.5,  3.0),
    "volume_multiplier":        (1.0,  2.5),
    "min_confidence":           (0.35, 0.80),
    "position_size_multiplier": (0.3,  1.5),
}


class AdaptiveConfigManager:
    """
    Manages the adaptive parameter store.
    The self-improver writes here nightly.
    The orchestrator reads here every morning.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(ADAPTIVE_DB), exist_ok=True)
        self._init_schema()
        self._current: Optional[AdaptiveParams] = None

    def load(self) -> AdaptiveParams:
        """Load current adaptive parameters (or return defaults)."""
        with sqlite3.connect(ADAPTIVE_DB) as conn:
            row = conn.execute(
                "SELECT params_json FROM adaptive_config ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row:
            try:
                data = json.loads(row[0])
                params = AdaptiveParams(**{k: v for k, v in data.items()
                                          if k in AdaptiveParams.__dataclass_fields__})
                self._current = params
                logger.info(f"Adaptive config loaded (updated {params.date_updated}, reason: {params.update_reason})")
                return params
            except Exception as e:
                logger.warning(f"Failed to load adaptive config: {e} — using defaults")

        defaults = AdaptiveParams(date_updated=date.today().isoformat())
        self._current = defaults
        return defaults

    def save(self, params: AdaptiveParams, reason: str = "manual"):
        """Persist updated parameters with a reason for the change."""
        params.date_updated = date.today().isoformat()
        params.update_reason = reason
        self._enforce_bounds(params)

        params_json = json.dumps(asdict(params), default=str)
        with sqlite3.connect(ADAPTIVE_DB) as conn:
            conn.execute(
                "INSERT INTO adaptive_config (params_json, reason) VALUES (?, ?)",
                (params_json, reason)
            )
        self._current = params
        logger.info(f"Adaptive config saved: {reason}")

    def get_current(self) -> AdaptiveParams:
        """Get in-memory current params (or load from DB)."""
        if self._current is None:
            return self.load()
        return self._current

    def get_history(self, days: int = 30) -> List[Dict]:
        """Get config change history for analysis."""
        with sqlite3.connect(ADAPTIVE_DB) as conn:
            from datetime import timedelta
            since = (date.today() - timedelta(days=days)).isoformat()
            rows = conn.execute(
                "SELECT params_json, reason, created_at FROM adaptive_config WHERE created_at >= ? ORDER BY created_at DESC",
                (since,)
            ).fetchall()
        return [{"params": json.loads(r[0]), "reason": r[1], "at": r[2]} for r in rows]

    def apply_regime_override(self, params: AdaptiveParams, regime_label: str) -> AdaptiveParams:
        """Apply regime-specific parameter overrides on top of base params."""
        import copy
        p = copy.deepcopy(params)
        overrides = params.regime_overrides.get(regime_label, {})

        if regime_label == "CRISIS":
            p.max_positions = 0
            p.position_size_multiplier = 0.0
        elif regime_label == "VOLATILE":
            p.position_size_multiplier = min(p.position_size_multiplier, 0.6)
            p.stop_loss_pct = min(p.stop_loss_pct * 1.3, BOUNDS["stop_loss_pct"][1])
            p.min_confidence = min(p.min_confidence + 0.10, BOUNDS["min_confidence"][1])
        elif regime_label == "TRENDING_DOWN":
            p.position_size_multiplier = min(p.position_size_multiplier * 1.2, 1.3)
            p.min_confidence = max(p.min_confidence - 0.05, BOUNDS["min_confidence"][0])
        elif regime_label == "TRENDING_UP":
            p.position_size_multiplier = min(p.position_size_multiplier, 0.4)
            p.min_confidence = min(p.min_confidence + 0.20, BOUNDS["min_confidence"][1])
            p.max_positions = max(1, p.max_positions - 1)

        # Apply any custom overrides stored in the DB
        for key, val in overrides.items():
            if hasattr(p, key):
                setattr(p, key, val)

        self._enforce_bounds(p)
        return p

    def _enforce_bounds(self, params: AdaptiveParams):
        """Clamp all bounded parameters to safe ranges."""
        for attr, (lo, hi) in BOUNDS.items():
            val = getattr(params, attr, None)
            if val is not None:
                setattr(params, attr, round(max(lo, min(hi, val)), 4))
        # Keep risk/reward rational
        if params.target_pct < params.stop_loss_pct * 1.3:
            params.target_pct = round(params.stop_loss_pct * 1.5, 2)

    def _init_schema(self):
        with sqlite3.connect(ADAPTIVE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    params_json TEXT NOT NULL,
                    reason TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)


# Singleton
adaptive_config = AdaptiveConfigManager()
