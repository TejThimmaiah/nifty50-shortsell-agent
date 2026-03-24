"""
Tej Upgrade Loader
===================
Safely loads all Phase 2+3 upgrade modules.
If ANY module fails to import, Tej continues trading with the existing system.
Nothing here can break the core trading loop.

Usage in brain/orchestrator.py:
    from utils.upgrade_loader import upgrades
    if upgrades.sentiment:
        sentiment = upgrades.sentiment.analyze_symbol(symbol)
"""

import logging
logger = logging.getLogger("upgrade_loader")


class UpgradeLoader:
    """Loads all upgrade modules safely. Failed imports = None (not crash)."""

    def __init__(self):
        self._loaded = {}
        self._load_all()

    def _safe_import(self, name: str, module_path: str, attr: str = None):
        """Import a module safely. Returns None if fails."""
        try:
            import importlib
            mod = importlib.import_module(module_path)
            obj = getattr(mod, attr) if attr else mod
            self._loaded[name] = True
            logger.info(f"✅ Upgrade loaded: {name}")
            return obj
        except Exception as e:
            self._loaded[name] = False
            logger.debug(f"⚠️  Upgrade skipped: {name} ({e})")
            return None

    def _load_all(self):
        # ── Phase 2 ───────────────────────────────────────────────────────
        self.vector_memory     = self._safe_import("vector_memory",     "intelligence.vector_memory",       "vector_memory")
        self.langraph          = self._safe_import("langraph",          "brain.langraph_orchestrator",      "trading_graph")

        # ── Phase 3 Intelligence ──────────────────────────────────────────
        self.sentiment         = self._safe_import("sentiment",         "intelligence.sentiment_engine",    "sentiment_engine")
        self.social_sentiment  = self._safe_import("social_sentiment",  "intelligence.social_sentiment",    "social_radar")
        self.options_flow      = self._safe_import("options_flow",      "intelligence.options_flow",        "options_analyzer")
        self.insider_tracker   = self._safe_import("insider_tracker",   "intelligence.insider_tracker",     "insider_tracker")
        self.macro_radar       = self._safe_import("macro_radar",       "intelligence.macro_radar",         "macro_radar")
        self.contagion         = self._safe_import("contagion",         "intelligence.contagion_detector",  "contagion_detector")
        self.causal            = self._safe_import("causal",            "intelligence.causal_reasoning",    "causal_reasoner")
        self.multi_debate      = self._safe_import("multi_debate",      "intelligence.multi_agent_debate",  None)
        self.genetic           = self._safe_import("genetic",           "intelligence.genetic_evolution",   "genetic_evolution")
        self.tail_risk         = self._safe_import("tail_risk",         "intelligence.tail_risk",           "tail_risk_engine")
        self.meta_learning     = self._safe_import("meta_learning",     "intelligence.meta_learning",       "maml_agent")
        self.hypothesis        = self._safe_import("hypothesis",        "intelligence.hypothesis_memory",   "hypothesis_generator")
        self.faiss_memory      = self._safe_import("faiss_memory",      "intelligence.hypothesis_memory",   "faiss_memory")
        self.chart_vision      = self._safe_import("chart_vision",      "intelligence.chart_vision",        "chart_vision")
        self.rl_agent          = self._safe_import("rl_agent",          "intelligence.rl_agent",            "rl_agent")
        self.earnings          = self._safe_import("earnings",          "intelligence.earnings_altdata",    "earnings_analyzer")
        self.alt_data          = self._safe_import("alt_data",          "intelligence.earnings_altdata",    "alt_data_engine")

        # ── Phase 3 Execution ─────────────────────────────────────────────
        self.tick_executor     = self._safe_import("tick_executor",     "agents.tick_executor",             "tick_executor")
        self.dynamic_hedge     = self._safe_import("dynamic_hedge",     "agents.dynamic_hedge",             "dynamic_hedge")
        self.smart_router      = self._safe_import("smart_router",      "agents.smart_execution",           "smart_router")
        self.corr_detector     = self._safe_import("corr_detector",     "agents.smart_execution",           "correlation_detector")

        # ── Phase 3 Utils ─────────────────────────────────────────────────
        self.explainer         = self._safe_import("explainer",         "utils.explainable_ai",             "explainer")
        self.predictive_alerts = self._safe_import("predictive_alerts", "utils.predictive_alerts",          "predictive_alerts")
        self.working_memory    = self._safe_import("working_memory",    "utils.working_memory",             "working_memory")
        self.nemo_guardrails   = self._safe_import("nemo_guardrails",   "utils.nemo_ollama",                "nemo_guardrails")
        self.ollama            = self._safe_import("ollama",            "utils.nemo_ollama",                "ollama_llm")

        # Summary
        loaded  = sum(1 for v in self._loaded.values() if v)
        total   = len(self._loaded)
        logger.info(f"Upgrades: {loaded}/{total} loaded successfully")

    def status(self) -> str:
        loaded  = [(k, v) for k, v in self._loaded.items()]
        active  = [k for k, v in loaded if v]
        skipped = [k for k, v in loaded if not v]
        lines   = [f"Tej Upgrades: {len(active)}/{len(loaded)} active"]
        if active:
            lines.append(f"Active: {', '.join(active)}")
        if skipped:
            lines.append(f"Skipped (pkg not installed): {', '.join(skipped)}")
        return "\n".join(lines)

    def get_enhanced_signals(self, symbol: str, df, context: dict) -> dict:
        """
        Run all available upgrade intelligence on a symbol.
        Returns dict of enhancement signals — all optional.
        Falls back gracefully if any module unavailable.
        """
        enhancements = {}

        # Sentiment
        if self.sentiment:
            try:
                result = self.sentiment.analyze_symbol(symbol)
                enhancements["sentiment_score"] = result.get("score", 0)
                enhancements["sentiment_signal"] = result.get("signal", "NEUTRAL")
            except Exception:
                pass

        # Options flow
        if self.options_flow:
            try:
                result = self.options_flow.analyze(symbol)
                enhancements["options_signal"] = result.signal
                enhancements["pcr"] = result.pcr
                enhancements["max_pain"] = result.max_pain
            except Exception:
                pass

        # Insider activity
        if self.insider_tracker:
            try:
                result = self.insider_tracker.analyze(symbol)
                enhancements["insider_signal"] = result.signal
                enhancements["insider_net_flow"] = result.net_flow
            except Exception:
                pass

        # Tail risk / position size multiplier
        if self.tail_risk:
            try:
                vix = context.get("vix", 15)
                nifty_5d = context.get("nifty_5d", -1)
                report = self.tail_risk.generate_report(
                    vix=vix, nifty_5d=nifty_5d
                )
                enhancements["risk_regime"] = report.regime
                enhancements["size_multiplier"] = self.tail_risk.get_size_multiplier(
                    report.regime, vix
                )
            except Exception:
                pass

        # Causal reasoning
        if self.causal:
            try:
                factors = self.causal.analyze_causality(context)
                enhancements["causal_factors"] = len(factors)
                enhancements["causal_summary"] = self.causal.get_causal_summary(context)
            except Exception:
                pass

        # Hypothesis matching
        if self.hypothesis:
            try:
                applicable = self.hypothesis.get_applicable(context)
                if applicable:
                    enhancements["hypothesis_match"] = applicable[0].description
                    enhancements["hypothesis_wr"] = applicable[0].win_rate
            except Exception:
                pass

        # FAISS similar past trades
        if self.faiss_memory:
            try:
                enhancements["similar_trades"] = self.faiss_memory.associative_insight(
                    context, symbol
                )
            except Exception:
                pass

        # Vector memory insight
        if self.vector_memory:
            try:
                enhancements["memory_insight"] = self.vector_memory.get_insight(
                    context, symbol
                )
            except Exception:
                pass

        # RL agent signal
        if self.rl_agent:
            try:
                import numpy as np
                obs = np.array([context.get(k, 0) for k in [
                    "rsi","macd","volume_ratio","vwap_dist","atr","bb_pos",
                    "mom_5","mom_10","nifty_ret","vix","fii","sector",
                    "support_dist","resist_dist","obv","wyckoff","regime",
                    "hour","dow","prev_ret","gap","vol_spike","rr","kelly","score"
                ]], dtype=np.float32)
                rl_result = self.rl_agent.get_action(obs)
                enhancements["rl_signal"] = rl_result.get("action", "SKIP")
            except Exception:
                pass

        return enhancements

    def store_trade_result(self, symbol: str, context: dict, outcome: dict):
        """Store completed trade in all memory systems."""
        trade_id = f"{symbol}_{outcome.get('date','')}"

        if self.vector_memory:
            try:
                self.vector_memory.store_trade(trade_id, symbol, context, outcome)
            except Exception:
                pass

        if self.faiss_memory:
            try:
                self.faiss_memory.store(symbol, context, outcome)
            except Exception:
                pass

        if self.working_memory:
            try:
                self.working_memory.log_trade({
                    "symbol": symbol, **outcome
                })
            except Exception:
                pass

        if self.meta_learning:
            try:
                pass  # MAML updates happen in weekly training
            except Exception:
                pass


# Singleton — import once, use everywhere
upgrades = UpgradeLoader()
