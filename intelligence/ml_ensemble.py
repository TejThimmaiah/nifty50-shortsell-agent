"""
ML Ensemble Predictor
A lightweight machine learning ensemble that predicts short trade direction
using the same features the agent already computes.

Models used (all from scikit-learn, no GPU needed):
  1. Random Forest       — handles non-linear relationships between signals
  2. Gradient Boosting   — captures feature interactions
  3. Logistic Regression — linear baseline with regularisation
  4. Naive Bayes         — probabilistic, fast, complements Bayesian fusion

Ensemble output = average predicted probability from all 4 models.

Training data = all closed trades from trade_memory.db.
Re-trains every week or after 50 new trades.
Uses walk-forward validation (no lookahead bias).

Features:
  RSI, MACD histogram, BB position, volume ratio, gap pct, VIX,
  FII net, advance/decline ratio, hour of day, day of week,
  sector downtrend (binary), MTF alignment count, confidence score,
  Nifty 5d change, number of bearish signals fired

Output: P(short wins) — blended with Bayesian fusion posterior.
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "ml_ensemble.pkl")
MIN_TRAINING_SAMPLES = 30   # Need at least 30 trades to train


@dataclass
class MLPrediction:
    probability:      float      # P(win) from ensemble
    model_agreement:  float      # 0–1: how much models agree with each other
    feature_importance: Dict[str, float]  # top features driving prediction
    trained_on:       int        # number of training samples
    is_reliable:      bool       # True if trained on MIN_TRAINING_SAMPLES+
    models_used:      List[str]


# ── Feature extraction ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rsi", "macd_histogram", "bb_position", "volume_ratio",
    "gap_pct", "confidence_score", "mtf_alignment",
    "fii_net_norm", "vix_norm", "hour", "day_of_week",
    "nifty_5d_change", "is_early", "is_late",
    "sector_downtrend", "n_signals_fired",
]


def _extract_features(trade: Dict) -> Optional[np.ndarray]:
    """Extract ML features from a trade memory record."""
    try:
        hour = 0
        if trade.get("entry_time"):
            try:
                hour = int(trade["entry_time"].split(":")[0])
            except Exception:
                hour = 10

        features = [
            float(trade.get("rsi_at_entry", 65)),
            float(trade.get("macd_histogram", 0)),
            float(trade.get("bb_position", 0.7)),
            float(trade.get("volume_ratio", 1.0)),
            float(trade.get("gap_pct", 0)),
            float(trade.get("confidence_score", 0.5)),
            float(trade.get("mtf_alignment_count", 0)),
            float(trade.get("fii_net_cr", 0)) / 1000.0,   # normalise to ~[-1,1]
            float(trade.get("india_vix", 15)) / 25.0,
            float(hour),
            float(date.fromisoformat(trade["trade_date"]).weekday()) if trade.get("trade_date") else 2,
            float(trade.get("nifty_change_pct", 0)),
            1.0 if trade.get("time_of_day") == "EARLY" else 0.0,
            1.0 if trade.get("time_of_day") == "LATE" else 0.0,
            1.0 if trade.get("sector_trend") == "DOWNTREND" else 0.0,
            float(len(trade.get("signals_fired", []))),
        ]
        return np.array(features, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None


# ── Ensemble model ────────────────────────────────────────────────────────────

class MLEnsemblePredictor:
    """
    Trains and serves an ML ensemble on trade history.
    Completely free — uses scikit-learn only.
    """

    def __init__(self):
        self._models: Optional[List] = None
        self._trained_on: int = 0
        self._load_models()

    # ──────────────────────────────────────────────────────────────
    # PREDICT
    # ──────────────────────────────────────────────────────────────

    def predict(self, trade_features: Dict) -> Optional[MLPrediction]:
        """
        Predict P(win) for a candidate trade.
        Returns None if model isn't trained yet.
        """
        if not self._models or self._trained_on < MIN_TRAINING_SAMPLES:
            return None

        x = _extract_features(trade_features)
        if x is None:
            return None

        x = x.reshape(1, -1)
        probs = []
        names = []

        for name, model in self._models:
            try:
                p = model.predict_proba(x)[0][1]  # P(class=1=win)
                probs.append(p)
                names.append(name)
            except Exception as e:
                logger.debug(f"Model {name} predict error: {e}")

        if not probs:
            return None

        ensemble_prob = float(np.mean(probs))
        agreement     = 1.0 - float(np.std(probs)) * 2   # high std = disagreement
        agreement     = max(0.0, min(1.0, agreement))

        # Feature importance from Random Forest
        importance = {}
        try:
            rf = next(m for name, m in self._models if name == "RandomForest")
            imp = rf.feature_importances_
            importance = dict(sorted(
                zip(FEATURE_NAMES, imp),
                key=lambda x: x[1], reverse=True
            )[:5])
        except Exception:
            pass

        return MLPrediction(
            probability=round(ensemble_prob, 4),
            model_agreement=round(agreement, 3),
            feature_importance={k: round(v, 4) for k, v in importance.items()},
            trained_on=self._trained_on,
            is_reliable=self._trained_on >= MIN_TRAINING_SAMPLES,
            models_used=names,
        )

    # ──────────────────────────────────────────────────────────────
    # TRAIN
    # ──────────────────────────────────────────────────────────────

    def train(self, force: bool = False) -> bool:
        """
        Train the ensemble on all available trade history.
        Returns True if training was performed.
        """
        from intelligence.trade_memory import memory_store

        trades = memory_store.get_recent(days=180)
        closed = [t for t in trades if t.get("exit_reason") and t.get("rsi_at_entry", 0) > 0]

        if len(closed) < MIN_TRAINING_SAMPLES and not force:
            logger.info(f"ML ensemble: only {len(closed)} trades — need {MIN_TRAINING_SAMPLES} to train")
            return False

        X, y = [], []
        for trade in closed:
            features = _extract_features(trade)
            if features is not None:
                X.append(features)
                y.append(int(trade.get("won", False)))

        if len(X) < MIN_TRAINING_SAMPLES:
            return False

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Training ML ensemble on {len(X)} trades ({int(y.mean()*100)}% win rate)")

        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            logger.warning("scikit-learn not installed — skipping ML training. Run: pip install scikit-learn")
            return False

        # Walk-forward split: train on first 80%, validate on last 20%
        split  = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        models = [
            ("RandomForest", RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=3,
                class_weight="balanced", random_state=42
            )),
            ("GradientBoosting", GradientBoostingClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )),
            ("LogisticRegression", Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=0.5, class_weight="balanced", random_state=42)),
            ])),
            ("NaiveBayes", GaussianNB()),
        ]

        trained = []
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                if len(X_val) > 0:
                    val_acc = (model.predict(X_val) == y_val).mean()
                    logger.info(f"  {name}: val_acc={val_acc:.2%}")
                trained.append((name, model))
            except Exception as e:
                logger.warning(f"  {name} training failed: {e}")

        self._models = trained
        self._trained_on = len(X)
        self._save_models()
        logger.info(f"ML ensemble trained: {len(trained)} models, {len(X)} samples")
        return True

    def should_retrain(self) -> bool:
        """Check if retraining is needed (weekly or after 50 new trades)."""
        from intelligence.trade_memory import memory_store
        recent = memory_store.get_recent(days=7)
        new_trades = len([t for t in recent if t.get("exit_reason")])
        return new_trades >= 50 or self._models is None

    # ──────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────

    def _save_models(self):
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({
                    "models": self._models,
                    "trained_on": self._trained_on,
                    "date": date.today().isoformat(),
                }, f)
            logger.info(f"ML ensemble saved: {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Model save error: {e}")

    def _load_models(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self._models    = data.get("models", [])
            self._trained_on = data.get("trained_on", 0)
            logger.info(f"ML ensemble loaded: {self._trained_on} training samples, {len(self._models or [])} models")
        except Exception as e:
            logger.warning(f"Model load error: {e}")


# Singleton
ml_predictor = MLEnsemblePredictor()
