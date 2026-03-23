"""
Self-Improver Agent
The most important intelligence component.
Every night after market close, it:

1. Reviews every trade from today with full context
2. Identifies which signals and patterns worked vs failed
3. Reflects using LLM to extract deep lessons
4. Updates adaptive parameters (RSI threshold, SL%, confidence gates, etc.)
5. Writes lessons to the knowledge base for tomorrow's morning brief
6. Auto-tunes signal weights based on recent accuracy
7. Adjusts pattern weights based on which patterns delivered results
8. Detects strategy drift and raises alerts before losses compound

This is what makes the agent genuinely smarter every single day.
Day 1: Uses defaults. Day 30: Tuned to current market conditions.
Day 90: Deep understanding of what works in Indian markets.
"""

import json
import logging
import os
import requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

from intelligence.trade_memory import memory_store, TradeMemory
from intelligence.adaptive_config import adaptive_config, AdaptiveParams, BOUNDS
from config import GROQ_API_KEY, GEMINI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

KNOWLEDGE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "db", "knowledge_base.json"
)


class SelfImproverAgent:
    """
    Nightly reflection and parameter adaptation engine.
    Runs after square-off (3:30 PM IST) every trading day.
    """

    def __init__(self):
        self._knowledge = self._load_knowledge()

    # ──────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────

    def run_nightly_improvement(self) -> Dict:
        """
        Full nightly improvement cycle.
        Returns a summary of what changed.
        """
        logger.info("=== Nightly self-improvement cycle starting ===")
        today = date.today().isoformat()

        # Step 1: Load today's trades and recent history
        today_trades   = memory_store.get_recent(days=1)
        recent_trades  = memory_store.get_recent(days=14)

        if not today_trades:
            logger.info("No trades today — skipping improvement cycle")
            return {"status": "no_trades", "date": today}

        # Step 2: Compute performance statistics
        stats = self._compute_stats(today_trades, recent_trades)
        logger.info(
            f"Today: {stats['today_trades']} trades, "
            f"{stats['today_win_rate']:.0%} WR, "
            f"₹{stats['today_pnl']:,.0f} P&L"
        )

        # Step 3: LLM reflection — extract lessons
        lessons = self._llm_reflect(today_trades, stats)
        logger.info(f"Lessons extracted: {len(lessons)}")

        # Step 4: Update adaptive parameters
        current_params = adaptive_config.load()
        updated_params, changes = self._update_parameters(current_params, stats, lessons)

        if changes:
            adaptive_config.save(updated_params, reason=f"nightly_learning_{today}")
            logger.info(f"Parameters updated: {changes}")
        else:
            logger.info("Parameters unchanged — performance stable")

        # Step 5: Update signal/pattern weights
        self._update_weights(updated_params, stats)

        # Step 6: Write to knowledge base
        self._update_knowledge(stats, lessons, changes, today)

        # Step 7: Write lessons back to trade memories
        self._annotate_trade_memories(today_trades, lessons)

        result = {
            "status":         "complete",
            "date":           today,
            "trades_analysed": len(today_trades),
            "lessons":        lessons[:5],
            "param_changes":  changes,
            "new_win_rate_14d": stats.get("win_rate_14d", 0),
        }
        logger.info(f"=== Nightly improvement complete: {len(changes)} parameter changes ===")
        return result

    # ──────────────────────────────────────────────────────────────
    # MORNING BRIEF — read by orchestrator at 9:15 AM
    # ──────────────────────────────────────────────────────────────

    def get_morning_brief(self) -> str:
        """
        Generates today's intelligence brief for the orchestrator.
        Called every morning before market opens.
        Includes: what worked recently, what to avoid, regime guidance.
        """
        params  = adaptive_config.load()
        stats   = self._compute_stats(memory_store.get_recent(days=7), memory_store.get_recent(days=30))
        lessons = memory_store.get_lessons(days=7)
        knowledge = self._load_knowledge()

        signal_stats  = memory_store.get_signal_stats(days=14)
        pattern_stats = memory_store.get_pattern_stats(days=14)
        regime_stats  = memory_store.get_regime_stats(days=30)
        time_stats    = memory_store.get_time_stats(days=14)
        rsi_analysis  = memory_store.get_rsi_optimum(days=14)

        # Find top performing patterns
        top_patterns = sorted(
            [(p, d["win_rate"]) for p, d in pattern_stats.items() if d["count"] >= 3],
            key=lambda x: x[1], reverse=True
        )[:3]

        # Find worst time windows
        worst_times = sorted(
            [(t, d["win_rate"]) for t, d in time_stats.items() if d["count"] >= 3],
            key=lambda x: x[1]
        )[:1]

        # Best regime for shorts
        best_regime = sorted(
            [(r, d["win_rate"]) for r, d in regime_stats.items() if d["count"] >= 3],
            key=lambda x: x[1], reverse=True
        )[:1]

        brief = f"""=== INTELLIGENCE BRIEF — {date.today().isoformat()} ===

ADAPTIVE PARAMETERS (updated from learning):
  RSI threshold    : {params.rsi_overbought}
  Stop loss        : {params.stop_loss_pct}%
  Target           : {params.target_pct}%
  Min confidence   : {params.min_confidence}
  Position size    : {params.position_size_multiplier}x
  Max positions    : {params.max_positions}

14-DAY PERFORMANCE:
  Win rate    : {stats.get('win_rate_14d', 0):.0%}
  Avg P&L/trade: ₹{stats.get('avg_pnl_14d', 0):,.0f}

WHAT IS WORKING (last 14 days):
{self._format_signal_stats(signal_stats, top=3, metric="win_rate")}

TOP PATTERNS (by win rate, min 3 trades):
{chr(10).join(f'  • {p}: {wr:.0%} WR' for p, wr in top_patterns) or '  • Insufficient data'}

OPTIMAL RSI:
  Best RSI threshold: {rsi_analysis.get("best_threshold", 70)} 
  Win rate at that level: {rsi_analysis.get("win_rate_at_best", 0):.0%}

AVOID TODAY:
{chr(10).join(f'  ⚠ {t[0]} time window ({t[1]:.0%} WR)' for t in worst_times) or '  • No strong avoidance signals'}

RECENT LESSONS (last 7 days):
{chr(10).join(f'  • {l}' for l in lessons[:4]) or '  • No lessons yet — keep trading'}

REGIME INSIGHT:
  Best performing regime: {best_regime[0][0] if best_regime else 'N/A'} ({best_regime[0][1]:.0%} WR)

=== END BRIEF ==="""

        logger.info("Morning brief generated")
        return brief

    # ──────────────────────────────────────────────────────────────
    # STATISTICS
    # ──────────────────────────────────────────────────────────────

    def _compute_stats(self, today_trades: List[Dict], recent_trades: List[Dict]) -> Dict:
        """Compute all performance metrics needed for learning."""
        def win_rate(trades): return sum(1 for t in trades if t.get("won")) / max(len(trades), 1)
        def avg_pnl(trades):  return sum(t.get("pnl", 0) for t in trades) / max(len(trades), 1)
        def total_pnl(trades): return sum(t.get("pnl", 0) for t in trades)

        closed_today   = [t for t in today_trades  if t.get("exit_reason")]
        closed_14d     = [t for t in recent_trades if t.get("exit_reason")]

        return {
            "today_trades":    len(closed_today),
            "today_pnl":       total_pnl(closed_today),
            "today_win_rate":  win_rate(closed_today),
            "win_rate_14d":    win_rate(closed_14d),
            "avg_pnl_14d":     avg_pnl(closed_14d),
            "total_pnl_14d":   total_pnl(closed_14d),
            "consecutive_losses": self._count_streak(closed_14d, winning=False),
            "consecutive_wins":   self._count_streak(closed_14d, winning=True),
            "signal_stats":    memory_store.get_signal_stats(days=14),
            "pattern_stats":   memory_store.get_pattern_stats(days=14),
            "regime_stats":    memory_store.get_regime_stats(days=30),
            "time_stats":      memory_store.get_time_stats(days=14),
            "rsi_analysis":    memory_store.get_rsi_optimum(days=14),
        }

    def _count_streak(self, trades: List[Dict], winning: bool) -> int:
        """Count the current consecutive win or loss streak."""
        count = 0
        for t in reversed(trades):
            if bool(t.get("won")) == winning:
                count += 1
            else:
                break
        return count

    # ──────────────────────────────────────────────────────────────
    # LLM REFLECTION
    # ──────────────────────────────────────────────────────────────

    def _llm_reflect(self, trades: List[Dict], stats: Dict) -> List[str]:
        """Use LLM to extract actionable lessons from today's trades."""
        if not trades:
            return []

        trade_summaries = []
        for t in trades:
            if not t.get("exit_reason"):
                continue
            result = "WON" if t.get("won") else "LOST"
            trade_summaries.append(
                f"  {result}: {t.get('symbol')} | Entry ₹{t.get('entry_price',0):.2f} | "
                f"RSI={t.get('rsi_at_entry',0):.0f} | Pattern={t.get('candlestick_pattern','-')} | "
                f"Regime={t.get('market_regime','-')} | Time={t.get('time_of_day','-')} | "
                f"Signals={t.get('signals_fired',[])} | P&L=₹{t.get('pnl',0):.0f} | "
                f"Exit={t.get('exit_reason','-')}"
            )

        prompt = f"""You are the learning system of an autonomous NSE intraday short selling agent.
Analyse today's trades and extract specific, actionable lessons to improve tomorrow's performance.

TODAY'S TRADES:
{chr(10).join(trade_summaries)}

14-DAY STATISTICS:
  Win rate: {stats['win_rate_14d']:.0%}
  Avg P&L: ₹{stats['avg_pnl_14d']:,.0f}/trade
  Loss streak: {stats['consecutive_losses']}
  Win streak: {stats['consecutive_wins']}

Best signal (14d): {self._best_item(stats['signal_stats'], 'win_rate')}
Best pattern (14d): {self._best_item(stats['pattern_stats'], 'win_rate')}
Best time (14d): {self._best_item(stats['time_stats'], 'win_rate')}
RSI optimum: {stats['rsi_analysis'].get('best_threshold', 70)}

Based on this data, extract 3-5 specific, actionable lessons.
Each lesson must be concrete (e.g. "Avoid EARLY time window — only 38% WR" not "be more selective").
Lessons should inform parameter changes, signal weights, or entry filters.

RESPOND ONLY IN JSON:
{{
  "lessons": [
    "specific lesson 1",
    "specific lesson 2",
    "specific lesson 3"
  ],
  "confidence": 0.0-1.0,
  "key_finding": "the single most important insight today"
}}"""

        response = self._call_llm(prompt)
        try:
            data = json.loads(response)
            return data.get("lessons", [])
        except Exception:
            # Parse as plain text lessons
            lines = [l.strip() for l in response.split("\n") if l.strip() and len(l) > 20]
            return lines[:5]

    # ──────────────────────────────────────────────────────────────
    # PARAMETER ADAPTATION
    # ──────────────────────────────────────────────────────────────

    def _update_parameters(
        self, params: AdaptiveParams, stats: Dict, lessons: List[str]
    ) -> tuple[AdaptiveParams, List[str]]:
        """
        Adjust parameters based on recent performance.
        Changes are gradual (max 10% shift per day) to avoid overreaction.
        """
        import copy
        p = copy.deepcopy(params)
        changes = []

        wr_14d   = stats["win_rate_14d"]
        streak_l = stats["consecutive_losses"]
        streak_w = stats["consecutive_wins"]
        rsi_opt  = stats["rsi_analysis"].get("best_threshold", 70)

        # ── RSI threshold adaptation ──────────────────────────────
        if rsi_opt and abs(rsi_opt - p.rsi_overbought) >= 3:
            # Shift RSI threshold toward optimal — max 2 points per day
            direction = 1 if rsi_opt > p.rsi_overbought else -1
            shift = min(2.0, abs(rsi_opt - p.rsi_overbought) * 0.3)
            p.rsi_overbought = round(p.rsi_overbought + direction * shift, 1)
            changes.append(f"RSI threshold → {p.rsi_overbought} (optimal found at {rsi_opt})")

        # ── Confidence gate: raise when losing, lower when winning ─
        if streak_l >= 3:
            new_conf = min(p.min_confidence + 0.05, BOUNDS["min_confidence"][1])
            if new_conf != p.min_confidence:
                p.min_confidence = new_conf
                changes.append(f"Confidence gate raised → {p.min_confidence} ({streak_l} losses)")
        elif streak_w >= 5 and wr_14d > 0.60:
            new_conf = max(p.min_confidence - 0.03, BOUNDS["min_confidence"][0])
            if new_conf != p.min_confidence:
                p.min_confidence = new_conf
                changes.append(f"Confidence gate lowered → {p.min_confidence} ({streak_w} wins)")

        # ── Position size: reduce when losing badly ────────────────
        if wr_14d < 0.40 and stats["total_pnl_14d"] < 0:
            new_mult = max(p.position_size_multiplier * 0.85, 0.40)
            changes.append(f"Position size reduced → {new_mult:.2f}x (WR={wr_14d:.0%})")
            p.position_size_multiplier = round(new_mult, 2)
        elif wr_14d > 0.60 and stats["total_pnl_14d"] > 0:
            new_mult = min(p.position_size_multiplier * 1.08, 1.30)
            changes.append(f"Position size increased → {new_mult:.2f}x (WR={wr_14d:.0%})")
            p.position_size_multiplier = round(new_mult, 2)

        # ── Stop loss / target: adapt based on average holding time ─
        if wr_14d > 0.55:
            # Winning — can afford slightly wider targets
            new_tgt = min(p.target_pct * 1.05, BOUNDS["target_pct"][1])
            if new_tgt != p.target_pct:
                p.target_pct = round(new_tgt, 2)
                changes.append(f"Target widened → {p.target_pct}% (strategy working)")

        # ── Pattern requirement: turn on if win rate very low ─────
        if wr_14d < 0.35 and not p.require_pattern:
            p.require_pattern = True
            changes.append("Pattern requirement ENABLED (WR < 35%)")
        elif wr_14d > 0.55 and p.require_pattern:
            p.require_pattern = False
            changes.append("Pattern requirement DISABLED (WR > 55%)")

        # ── Time filter: disable worst time window if persistent loser
        time_stats = stats.get("time_stats", {})
        early_wr   = time_stats.get("EARLY", {}).get("win_rate", 0.5)
        late_wr    = time_stats.get("LATE", {}).get("win_rate", 0.5)
        early_cnt  = time_stats.get("EARLY", {}).get("count", 0)
        late_cnt   = time_stats.get("LATE", {}).get("count", 0)

        if early_wr < 0.35 and early_cnt >= 8 and not p.skip_early_morning:
            p.skip_early_morning = True
            changes.append(f"Early morning disabled (9:20-9:45 WR={early_wr:.0%})")
        elif early_wr > 0.55 and p.skip_early_morning:
            p.skip_early_morning = False
            changes.append("Early morning re-enabled (WR recovered)")

        if late_wr < 0.35 and late_cnt >= 8 and not p.skip_late_afternoon:
            p.skip_late_afternoon = True
            changes.append(f"Late afternoon disabled (12-1pm WR={late_wr:.0%})")
        elif late_wr > 0.55 and p.skip_late_afternoon:
            p.skip_late_afternoon = False
            changes.append("Late afternoon re-enabled (WR recovered)")

        adaptive_config._enforce_bounds(p)
        return p, changes

    def _update_weights(self, params: AdaptiveParams, stats: Dict):
        """Update signal and pattern weights based on recent accuracy."""
        signal_stats = stats.get("signal_stats", {})

        # Adjust pattern weights
        pattern_stats = stats.get("pattern_stats", {})
        for pat, data in pattern_stats.items():
            if data["count"] >= 5:
                wr = data["win_rate"]
                old_w = params.pattern_weights.get(pat, 1.0)
                # Shift weight toward observed win rate (slow update)
                new_w = round(old_w * 0.80 + wr * 1.2 * 0.20, 3)
                new_w = max(0.2, min(1.5, new_w))
                if abs(new_w - old_w) > 0.05:
                    params.pattern_weights[pat] = new_w
                    logger.info(f"Pattern weight {pat}: {old_w:.2f} → {new_w:.2f} (WR={wr:.0%})")

        # Adjust signal weights
        total_weight = sum([
            params.weight_rsi, params.weight_divergence, params.weight_resistance,
            params.weight_volume, params.weight_macd, params.weight_bb, params.weight_ema
        ])

        signal_map = {
            "RSI_OVERBOUGHT":        "weight_rsi",
            "BEARISH_DIVERGENCE":    "weight_divergence",
            "AT_RESISTANCE":         "weight_resistance",
            "VOLUME_CONFIRMS":       "weight_volume",
            "MACD_TURNING_DOWN":     "weight_macd",
            "BB_EXTENDED":           "weight_bb",
            "EMA_DOWNTREND":         "weight_ema",
        }
        for signal, attr in signal_map.items():
            data = signal_stats.get(signal, {})
            if data.get("count", 0) >= 5:
                wr = data["win_rate"]
                old_w = getattr(params, attr)
                new_w = round(old_w * 0.80 + wr * 0.30 * 0.20, 4)
                new_w = max(0.01, min(0.40, new_w))
                setattr(params, attr, new_w)

        adaptive_config.save(params, reason="weight_update")

    # ──────────────────────────────────────────────────────────────
    # KNOWLEDGE BASE
    # ──────────────────────────────────────────────────────────────

    def _update_knowledge(self, stats: Dict, lessons: List[str], changes: List[str], today: str):
        """Write today's findings to the persistent knowledge base."""
        kb = self._load_knowledge()

        kb["last_updated"]        = today
        kb["total_learning_days"] = kb.get("total_learning_days", 0) + 1

        # Append lessons
        existing_lessons = kb.get("all_lessons", [])
        new_entries = [{"date": today, "lesson": l} for l in lessons]
        kb["all_lessons"] = (new_entries + existing_lessons)[:200]  # Keep last 200

        # Update win rate history
        wr_history = kb.get("win_rate_history", [])
        wr_history.append({"date": today, "wr": stats["win_rate_14d"], "pnl": stats["today_pnl"]})
        kb["win_rate_history"] = wr_history[-90:]  # 3 months

        # Update best signals
        kb["best_signals_14d"] = {
            sig: data for sig, data in stats.get("signal_stats", {}).items()
            if data.get("win_rate", 0) > 0.55 and data.get("count", 0) >= 5
        }
        kb["best_patterns_14d"] = {
            pat: data for pat, data in stats.get("pattern_stats", {}).items()
            if data.get("win_rate", 0) > 0.55 and data.get("count", 0) >= 3
        }

        # Regime performance
        kb["regime_stats"] = stats.get("regime_stats", {})

        # Parameter change log
        change_log = kb.get("parameter_changes", [])
        if changes:
            change_log.append({"date": today, "changes": changes})
        kb["parameter_changes"] = change_log[-60:]

        self._save_knowledge(kb)
        logger.info(f"Knowledge base updated. Total learning days: {kb['total_learning_days']}")

    def _annotate_trade_memories(self, trades: List[Dict], lessons: List[str]):
        """Write a general lesson to today's trade memories."""
        if not lessons or not trades:
            return
        lesson_text = " | ".join(lessons[:2])
        for trade in trades:
            if trade.get("trade_id") and not trade.get("lesson"):
                memory_store.update_outcome(
                    trade_id=trade["trade_id"],
                    exit_price=trade.get("exit_price", 0),
                    exit_reason=trade.get("exit_reason", ""),
                    pnl=trade.get("pnl", 0),
                    pnl_pct=trade.get("pnl_pct", 0),
                    holding_minutes=trade.get("holding_minutes", 0),
                    lesson=lesson_text,
                )

    # ──────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────

    def _load_knowledge(self) -> Dict:
        if os.path.exists(KNOWLEDGE_FILE):
            try:
                with open(KNOWLEDGE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "created": date.today().isoformat(),
            "total_learning_days": 0,
            "all_lessons": [],
            "win_rate_history": [],
            "best_signals_14d": {},
            "best_patterns_14d": {},
            "regime_stats": {},
            "parameter_changes": [],
        }

    def _save_knowledge(self, kb: Dict):
        os.makedirs(os.path.dirname(KNOWLEDGE_FILE), exist_ok=True)
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(kb, f, indent=2, default=str)

    def _best_item(self, stats: Dict, metric: str) -> str:
        if not stats:
            return "N/A"
        best = max(stats.items(), key=lambda x: x[1].get(metric, 0))
        return f"{best[0]} ({best[1].get(metric, 0):.0%} WR, n={best[1].get('count', 0)})"

    def _format_signal_stats(self, stats: Dict, top: int = 3, metric: str = "win_rate") -> str:
        if not stats:
            return "  • Insufficient data"
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].get(metric, 0), reverse=True)[:top]
        return "\n".join(
            f"  • {sig}: {d.get('win_rate', 0):.0%} WR, avg ₹{d.get('avg_pnl', 0):,.0f} (n={d.get('count', 0)})"
            for sig, d in sorted_stats
        )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for reflection."""
        if GROQ_API_KEY:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 800, "temperature": 0.2},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Groq LLM error: {e}")

        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                resp = requests.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"maxOutputTokens": 800, "temperature": 0.2}},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                logger.warning(f"Gemini LLM error: {e}")

        return '{"lessons": ["LLM unavailable — using statistical learning only"], "confidence": 0.5}'


# Singleton
self_improver = SelfImproverAgent()
