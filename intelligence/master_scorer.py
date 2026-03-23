"""
Master Scoring Engine
The unified intelligence hub. Takes a candidate and runs it through
every technical intelligence layer, returning a single fused score
with full attribution — exactly which factors drove the decision.

Layers (in order of reliability):
  1. Bayesian Signal Fusion       → P(win | observed signals)
  2. ML Ensemble Prediction       → P(win | feature vector)
  3. Wyckoff Analysis             → Distribution pattern score
  4. VWAP Setup                   → VWAP-based confirmation
  5. Volume Profile               → POC/VAH/VAL zone analysis
  6. Order Flow                   → Institutional buying/selling pressure
  7. Statistical Edge             → Proven E[P&L] for this signal combo
  8. Intermarket Bias             → Pre-market directional context
  9. Kelly Fraction               → Optimal position size

Final score = weighted combination, regime-adjusted.
Each layer can VETO the trade if strongly contradictory.

This replaces the old single-number "confidence" with a rich,
multi-dimensional assessment that explains exactly why a trade
is being taken or rejected.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MasterScore:
    symbol: str
    final_score: float          # 0–1: fused intelligence score
    recommendation: str         # STRONG_SHORT | SHORT | WEAK | SKIP | VETO
    confidence: float           # calibrated confidence (not same as score)

    # Individual layer scores
    bayesian_posterior:  Optional[float] = None
    ml_probability:      Optional[float] = None
    wyckoff_score:       Optional[float] = None
    vwap_score:          Optional[float] = None
    volume_profile_score: Optional[float] = None
    orderflow_score:     Optional[float] = None
    edge_expected_value: Optional[float] = None
    intermarket_score:   Optional[float] = None

    # Position sizing
    kelly_fraction:      float  = 0.0
    recommended_qty:     int    = 0
    max_risk_amount:     float  = 0.0

    # Attribution
    supporting_factors:  List[str] = field(default_factory=list)
    opposing_factors:    List[str] = field(default_factory=list)
    vetoed:              bool = False
    veto_reason:         str = ""

    # Layer count (transparency)
    layers_used:         int = 0
    layers_available:    int = 9


class MasterScoringEngine:
    """
    Orchestrates all intelligence layers into a final trade decision.
    """

    # Layer weights
    WEIGHTS = {
        "bayesian":       0.20,
        "ml":             0.15,
        "wyckoff":        0.12,
        "vwap":           0.10,
        "volume_profile": 0.08,
        "orderflow":      0.08,
        "edge":           0.06,
        "intermarket":    0.05,
        "divergence":     0.08,
        "fibo_pivot":     0.05,
        "market_char":    0.03,
    }

    # Veto conditions
    VETO_CONDITIONS = {
        "regime_crisis":       "Market in CRISIS regime",
        "intermarket_bullish": "Intermarket strongly bullish — avoid shorts",
        "orderflow_buying":    "Strong institutional buying — contradicts short thesis",
        "wyckoff_accumulation":"Wyckoff accumulation pattern — wrong side of trade",
        "no_edge":             "No proven statistical edge for this signal combination",
    }

    def __init__(self):
        # Import all intelligence layers
        from intelligence.signal_fusion    import signal_fusion
        from intelligence.ml_ensemble      import ml_predictor
        from intelligence.wyckoff          import wyckoff_analyser
        from intelligence.orderflow        import orderflow_analyser
        from intelligence.statistical_edge import edge_calculator
        from intelligence.kelly_sizer      import kelly_sizer
        from intelligence.volume_profile   import volume_profile_analyser
        from intelligence.divergence       import divergence_detector
        from intelligence.market_character import character_classifier
        from intelligence.fibonacci_pivots import fibo_pivot
        from intelligence.atr_stops        import atr_stop_engine
        from intelligence.walk_forward     import wfo_engine
        from strategies.vwap_strategy      import vwap_strategy

        self.fusion      = signal_fusion
        self.ml          = ml_predictor
        self.wyckoff     = wyckoff_analyser
        self.orderflow   = orderflow_analyser
        self.edge        = edge_calculator
        self.kelly       = kelly_sizer
        self.vp          = volume_profile_analyser
        self.divergence  = divergence_detector
        self.character   = character_classifier
        self.fibo        = fibo_pivot
        self.atr_stops   = atr_stop_engine
        self.wfo         = wfo_engine
        self.vwap        = vwap_strategy

    def score(
        self,
        symbol:          str,
        signals:         List[str],
        df_5m:           Optional[object] = None,
        df_15m:          Optional[object] = None,
        df_1d:           Optional[object] = None,
        trade_features:  Dict = None,
        entry_price:     float = 0,
        stop_loss:       float = 0,
        target:          float = 0,
        capital:         float = 100_000,
        regime_label:    str = "RANGING",
        regime_mult:     float = 1.0,
        intermarket_bias: float = 0.0,
        open_positions:  int = 0,
    ) -> MasterScore:
        """
        Run all intelligence layers and return a MasterScore.
        """
        supporting = []
        opposing   = []
        layer_scores = {}
        layers_used  = 0

        # ── LAYER 1: Bayesian Signal Fusion ──────────────────────────────
        fusion_result = self.fusion.fuse(signals, symbol)
        posterior     = fusion_result.posterior_win_prob
        layer_scores["bayesian"] = posterior
        layers_used += 1

        if posterior >= 0.65:
            supporting.append(f"Bayesian P(win)={posterior:.0%}")
        elif posterior <= 0.40:
            opposing.append(f"Bayesian P(win) low ({posterior:.0%})")

        # ── LAYER 2: ML Ensemble ────────────────────────────────────────
        ml_prob = None
        if trade_features:
            ml_pred = self.ml.predict(trade_features)
            if ml_pred and ml_pred.is_reliable:
                ml_prob = ml_pred.probability
                layer_scores["ml"] = ml_prob
                layers_used += 1
                if ml_prob >= 0.60:
                    top_feat = list(ml_pred.feature_importance.items())[:2]
                    feat_str = ", ".join(f"{k}:{v:.2f}" for k, v in top_feat)
                    supporting.append(f"ML P(win)={ml_prob:.0%} ({feat_str})")
                elif ml_prob <= 0.42:
                    opposing.append(f"ML P(win) low ({ml_prob:.0%})")

        # ── LAYER 3: Wyckoff ─────────────────────────────────────────────
        wyckoff_sc = None
        df_primary = df_15m or df_1d or df_5m
        if df_primary is not None:
            wy = self.wyckoff.analyse(df_primary, symbol)
            if wy:
                wyckoff_sc = wy.confidence
                layer_scores["wyckoff"] = wyckoff_sc
                layers_used += 1
                supporting.append(
                    f"Wyckoff {wy.pattern} (phase {wy.phase}, conf={wy.confidence:.0%})"
                )
                signals = list(signals) + ["WYCKOFF_DISTRIBUTION"]

        # ── LAYER 4: VWAP ────────────────────────────────────────────────
        vwap_sc = None
        if df_5m is not None:
            vwap_setup = self.vwap.analyse(df_5m, symbol)
            if vwap_setup:
                vwap_sc = vwap_setup.confidence
                layer_scores["vwap"] = vwap_sc
                layers_used += 1
                supporting.append(
                    f"VWAP {vwap_setup.setup_type} (conf={vwap_sc:.0%})"
                )
                signals = list(signals) + ["VWAP_REJECTION"]

        # ── LAYER 5: Volume Profile ──────────────────────────────────────
        vp_sc = None
        df_vol = df_15m or df_5m
        if df_vol is not None:
            vp = self.vp.analyse(df_vol, symbol, lookback=20)
            if vp:
                vp_sc = vp.short_score
                layer_scores["volume_profile"] = vp_sc
                layers_used += 1
                if vp_sc >= 0.3:
                    supporting.extend(vp.short_signals[:2])
                elif vp_sc <= 0:
                    opposing.append(f"Volume Profile: {vp.price_zone} zone (neutral/bullish)")

        # ── LAYER 6: Order Flow ──────────────────────────────────────────
        of_sc = None
        if df_5m is not None:
            of = self.orderflow.analyse(df_5m, symbol)
            if of:
                # orderflow_score: -1=buying, +1=selling
                of_sc = -of.orderflow_score   # negate: negative OF score = selling = good for shorts
                layer_scores["orderflow"] = of_sc
                layers_used += 1
                if of.bearish_signal:
                    supporting.append(f"Order flow bearish: {of.summary}")
                elif of.orderflow_score > 0.3:
                    opposing.append(f"Order flow: institutional buying ({of.orderflow_score:.2f})")
                    # VETO if very strong institutional buying
                    if of.orderflow_score > 0.6:
                        return self._veto(
                            symbol, "orderflow_buying",
                            layer_scores, supporting, opposing, layers_used
                        )

        # ── LAYER 7: Statistical Edge ────────────────────────────────────
        edge_ev = None
        if signals and entry_price and stop_loss and target:
            edge_ev = self.edge.expected_value_for_candidate(signals, entry_price, stop_loss, target)
            layer_scores["edge"] = max(0, min(1, (edge_ev + 500) / 1000))   # normalise EV to 0–1
            layers_used += 1
            if edge_ev > 0:
                supporting.append(f"Statistical EV=₹{edge_ev:.0f}/trade")
            elif edge_ev < -200:
                opposing.append(f"Negative edge (EV=₹{edge_ev:.0f})")

        # ── LAYER 8: Intermarket ─────────────────────────────────────────
        if intermarket_bias != 0:
            im_sc = -intermarket_bias
            layer_scores["intermarket"] = max(0, min(1, (im_sc + 1) / 2))
            layers_used += 1
            if intermarket_bias <= -0.3:
                supporting.append(f"Intermarket bearish (score={intermarket_bias:.2f})")
            elif intermarket_bias >= 0.4:
                opposing.append(f"Intermarket bullish — shorts face headwind")
                if intermarket_bias >= 0.7:
                    return self._veto(
                        symbol, "intermarket_bullish",
                        layer_scores, supporting, opposing, layers_used
                    )

        # ── LAYER 9: Divergence ──────────────────────────────────────────────
        try:
            from intelligence.divergence import divergence_detector
            div_result = divergence_detector.analyse(
                df_5m=df_5m, df_15m=df_15m, df_1d=df_1d, symbol=symbol
            )
            if div_result.composite_score > 0.20:
                layer_scores["divergence"] = div_result.composite_score
                layers_used += 1
                supporting.append(f"Divergence: {div_result.summary[:60]}")
                if div_result.multi_tf_confirmed:
                    supporting.append("Multi-TF divergence confirmed!")
        except Exception:
            pass

        # ── LAYER 10: Fibonacci / Pivot Confluence ───────────────────────────
        try:
            from intelligence.fibonacci_pivots import fibo_pivot
            df_fib = df_1d or df_15m
            if df_fib is not None and entry_price > 0:
                key_lvls = fibo_pivot.compute(df_fib, symbol)
                if key_lvls.at_fib_resistance or key_lvls.at_pivot_resistance:
                    fib_sc = 0.65 + key_lvls.short_confluence * 0.30
                    layer_scores["fibo_pivot"] = fib_sc
                    layers_used += 1
                    detail = []
                    if key_lvls.at_fib_resistance: detail.append("Fib resistance")
                    if key_lvls.at_pivot_resistance: detail.append("Pivot resistance")
                    if key_lvls.short_confluence > 0.3: detail.append(f"confluence {key_lvls.short_confluence:.0%}")
                    supporting.append(f"Fib/Pivot: {', '.join(detail)}")
        except Exception:
            pass

        # ── LAYER 11: Market Character ───────────────────────────────────────
        try:
            from intelligence.market_character import character_classifier
            df_char = df_1d or df_15m
            if df_char is not None:
                char = character_classifier.classify(df_char, symbol)
                if char.regime == "MEAN_REVERTING" and char.confidence >= 0.60:
                    layer_scores["market_char"] = char.confidence
                    layers_used += 1
                    supporting.append(f"Mean-reverting market (H={char.hurst_exponent:.2f}) — reversion favoured")
                elif char.regime == "TRENDING" and char.confidence >= 0.70:
                    opposing.append(f"Strong trend (H={char.hurst_exponent:.2f}) — momentum against shorts")
        except Exception:
            pass

        # ── ATR-ADAPTIVE STOP LOSS ───────────────────────────────────────────
        try:
            from intelligence.atr_stops import atr_stop_engine
            df_atr = df_5m or df_15m or df_1d
            if df_atr is not None and entry_price > 0:
                atr_result = atr_stop_engine.compute(df_atr, entry_price, "SHORT")
                if atr_result and stop_loss > 0:
                    stop_loss = min(atr_result.final_sl, stop_loss)
                    target    = max(atr_result.target, target)
                    supporting.append(
                        f"ATR stop ₹{atr_result.final_sl:.2f} ({atr_result.vol_regime} vol)"
                    )
        except Exception:
            pass

        # ── WFO ROBUSTNESS ───────────────────────────────────────────────────
        try:
            from intelligence.walk_forward import wfo_engine
            wfo_mult = wfo_engine.get_robustness_multiplier()
            if wfo_mult < 0.75:
                opposing.append(f"WFO robustness {wfo_mult:.0%} — fragile strategy")
        except Exception:
            wfo_mult = 1.0

        # ── COMPUTE FINAL SCORE ──────────────────────────────────────────
        if not layer_scores:
            final_score = posterior
        else:
            weighted_sum = 0.0
            total_weight = 0.0
            for layer, sc in layer_scores.items():
                if sc is not None:
                    w = self.WEIGHTS.get(layer, 0.05)
                    weighted_sum += sc * w
                    total_weight += w
            final_score = weighted_sum / max(total_weight, 1e-6)

        # Regime adjustment
        if regime_label == "TRENDING_DOWN":
            final_score = min(1.0, final_score * 1.08)
        elif regime_label == "TRENDING_UP":
            final_score *= 0.70
        elif regime_label == "VOLATILE":
            final_score *= 0.85

        final_score = round(max(0, min(1, final_score)), 4)

        # Recommendation
        if   final_score >= 0.72: rec = "STRONG_SHORT"
        elif final_score >= 0.58: rec = "SHORT"
        elif final_score >= 0.46: rec = "WEAK"
        else:                     rec = "SKIP"

        # Kelly position sizing
        win_rate = float(np.mean([v for v in [posterior, ml_prob] if v is not None] or [0.55]))
        kelly_ps = self.kelly.compute(
            symbol=symbol,
            entry_price=entry_price or 1000,
            stop_loss=stop_loss or 995,
            capital=capital,
            posterior_win_p=win_rate,
            open_positions=open_positions,
            regime_mult=regime_mult,
        )

        return MasterScore(
            symbol=symbol,
            final_score=final_score,
            recommendation=rec,
            confidence=round(final_score, 3),
            bayesian_posterior=posterior,
            ml_probability=ml_prob,
            wyckoff_score=wyckoff_sc,
            vwap_score=vwap_sc,
            volume_profile_score=vp_sc,
            orderflow_score=of_sc,
            edge_expected_value=edge_ev,
            intermarket_score=intermarket_bias,
            kelly_fraction=kelly_ps.kelly_fraction,
            recommended_qty=kelly_ps.quantity,
            max_risk_amount=kelly_ps.max_loss,
            supporting_factors=supporting,
            opposing_factors=opposing,
            vetoed=False,
            layers_used=layers_used,
        )

    def _veto(
        self, symbol: str, reason_key: str,
        layer_scores: Dict, supporting: List, opposing: List, layers_used: int,
    ) -> MasterScore:
        """Return a vetoed MasterScore."""
        reason = self.VETO_CONDITIONS.get(reason_key, reason_key)
        logger.info(f"Trade VETOED [{symbol}]: {reason}")
        return MasterScore(
            symbol=symbol,
            final_score=0.0,
            recommendation="VETO",
            confidence=0.0,
            supporting_factors=supporting,
            opposing_factors=opposing + [f"VETO: {reason}"],
            vetoed=True,
            veto_reason=reason,
            layers_used=layers_used,
        )


# Lazy import for numpy
try:
    import numpy as np
except ImportError:
    class np:
        @staticmethod
        def mean(x): return sum(x)/len(x) if x else 0

# Singleton
master_scorer = MasterScoringEngine()
