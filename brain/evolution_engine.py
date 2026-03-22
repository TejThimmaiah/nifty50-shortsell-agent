"""
Autonomous Evolution Engine
============================
The agent rewrites its own rules.

Every Sunday night, the Evolution Engine:
  1. Reads 90 days of trade memory
  2. Uses the Neural Brain to discover patterns
  3. Proposes new trading rules (not just parameter changes)
  4. Tests proposed rules on historical data (walk-forward)
  5. If a rule improves OOS performance → adds it to the live strategy
  6. If an existing rule degrades → removes or weakens it
  7. Writes evolution log for transparency

This is autonomous strategy development.
The agent literally writes new rules for itself.

Evolution types:
  PARAMETER     — adjust a numeric threshold (RSI=70 → 72)
  FILTER        — add a new entry filter (skip IT stocks before budget)
  SIGNAL        — discover a new leading indicator
  TIMING        — find better entry/exit windows
  SECTOR_RULE   — stock-specific behaviour (HDFCBANK at pivot = stronger short)
  ANTI_PATTERN  — learn what NOT to do

The evolution engine is honest — it only adopts rules that improve OOS metrics.
It rejects changes that only improve IS (overfit).
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

EVOLUTION_DB = os.path.join(os.path.dirname(__file__), "..", "db", "evolution_log.db")
STRATEGY_FILE = os.path.join(os.path.dirname(__file__), "..", "db", "evolved_strategy.json")


@dataclass
class EvolutionProposal:
    rule_id:        str
    rule_type:      str      # PARAMETER | FILTER | SIGNAL | TIMING | SECTOR_RULE | ANTI_PATTERN
    description:    str
    code_change:    str      # human-readable description of the change
    is_perf:        float    # in-sample improvement
    oos_perf:       float    # out-of-sample improvement (what matters)
    confidence:     float    # how confident we are in this improvement
    adopted:        bool = False
    rejected:       bool = False
    reject_reason:  str = ""
    proposed_at:    str = ""

    def __post_init__(self):
        if not self.proposed_at:
            self.proposed_at = date.today().isoformat()


@dataclass
class EvolvedStrategy:
    """The strategy as it has evolved over time — a living document."""
    version:             int   = 1
    created_at:          str   = ""
    last_evolved:        str   = ""
    total_evolutions:    int   = 0
    active_rules:        List  = field(default_factory=list)
    retired_rules:       List  = field(default_factory=list)
    rsi_threshold:       float = 70.0
    confidence_gate:     float = 0.50
    min_rr:              float = 2.0
    max_positions:       int   = 3
    sector_restrictions: Dict  = field(default_factory=dict)
    time_restrictions:   Dict  = field(default_factory=dict)
    stock_overrides:     Dict  = field(default_factory=dict)
    signal_weights:      Dict  = field(default_factory=dict)
    notes:               List  = field(default_factory=list)


class AutonomousEvolutionEngine:
    """
    Self-improving strategy engine.
    Tests new rules rigorously and only adopts those that survive validation.
    """

    # Minimum improvement required to adopt a new rule
    MIN_OOS_IMPROVEMENT = 0.05    # 5% improvement in profit factor
    MIN_SAMPLE_SIZE     = 20      # minimum trades to evaluate a rule

    def __init__(self):
        self._init_db()
        self._strategy = self._load_strategy()

    # ──────────────────────────────────────────────────────────────
    # MAIN EVOLUTION CYCLE (Sunday 8 PM)
    # ──────────────────────────────────────────────────────────────

    def run_evolution_cycle(self) -> Dict:
        """
        Full weekly evolution cycle.
        Returns summary of what changed.
        """
        from brain.neural_core import brain
        from intelligence.trade_memory import memory_store

        logger.info("=== Autonomous Evolution Cycle Starting ===")
        results = {
            "proposals":  [],
            "adopted":    [],
            "rejected":   [],
            "discoveries": [],
        }

        # Step 1: Get recent trade data
        recent_trades = memory_store.get_recent(days=90)
        if len(recent_trades) < self.MIN_SAMPLE_SIZE:
            logger.info(f"Only {len(recent_trades)} trades — need {self.MIN_SAMPLE_SIZE} to evolve")
            return results

        # Step 2: Brain discovers patterns
        new_patterns = brain.discover_patterns(recent_trades)
        results["discoveries"] = new_patterns
        logger.info(f"Brain discovered {len(new_patterns)} new patterns")

        # Step 3: Brain proposes strategic evolutions
        perf_data = self._compute_performance_summary(recent_trades)
        evolution_proposals = brain.evolve_strategy(perf_data)

        # Step 4: Convert proposals to testable rules
        for ep in evolution_proposals.get("proposed_evolutions", []):
            proposal = self._create_proposal(ep)
            if proposal:
                results["proposals"].append(proposal.description)

                # Step 5: Test each proposal on historical data
                validated = self._validate_proposal(proposal, recent_trades)
                if validated:
                    self._adopt_rule(proposal)
                    results["adopted"].append(proposal.description)
                    logger.info(f"  ✅ ADOPTED: {proposal.description[:80]}")
                else:
                    proposal.rejected = True
                    proposal.reject_reason = f"OOS improvement {proposal.oos_perf:.2f} < threshold {self.MIN_OOS_IMPROVEMENT}"
                    self._log_proposal(proposal)
                    results["rejected"].append(proposal.description)
                    logger.info(f"  ❌ REJECTED: {proposal.description[:80]}")

        # Step 6: Prune underperforming rules
        pruned = self._prune_weak_rules(recent_trades)
        if pruned:
            logger.info(f"  🗑 Pruned {len(pruned)} underperforming rules")

        # Step 7: Write evolution summary
        self._strategy.last_evolved    = date.today().isoformat()
        self._strategy.total_evolutions += 1
        self._save_strategy()

        summary = (
            f"Evolution cycle v{self._strategy.version}: "
            f"{len(results['adopted'])} adopted, "
            f"{len(results['rejected'])} rejected, "
            f"{len(new_patterns)} patterns discovered"
        )
        logger.info(f"=== {summary} ===")
        results["summary"] = summary

        return results

    # ──────────────────────────────────────────────────────────────
    # RULE APPLICATION (used by scanner every morning)
    # ──────────────────────────────────────────────────────────────

    def apply_rules(self, symbol: str, context: Dict) -> Tuple[bool, str]:
        """
        Apply all evolved rules to a candidate.
        Returns (allowed: bool, reason: str).
        """
        strategy = self._strategy

        # Check sector restrictions
        sector = context.get("sector", "")
        if sector in strategy.sector_restrictions:
            restriction = strategy.sector_restrictions[sector]
            if restriction.get("avoid"):
                return False, f"Evolved rule: avoid {sector} sector ({restriction.get('reason', '')})"

        # Check time restrictions
        hour = datetime.now(IST).hour
        minute = datetime.now(IST).minute
        time_key = f"{hour:02d}:{minute // 30 * 30:02d}"   # 30-min buckets
        if time_key in strategy.time_restrictions:
            rest = strategy.time_restrictions[time_key]
            if rest.get("avoid"):
                return False, f"Evolved rule: avoid {time_key} window ({rest.get('reason', '')})"

        # Check stock-specific overrides
        if symbol in strategy.stock_overrides:
            override = strategy.stock_overrides[symbol]
            if override.get("condition_required"):
                condition_met = context.get(override["condition_required"], False)
                if not condition_met:
                    return False, f"Evolved rule: {symbol} requires {override['condition_required']}"

        # Apply active rules
        for rule in strategy.active_rules:
            rule_type  = rule.get("type", "")
            conditions = rule.get("conditions", {})

            if rule_type == "ANTI_PATTERN":
                # Check if this anti-pattern is present
                if self._check_anti_pattern(rule, symbol, context):
                    return False, f"Evolved anti-pattern: {rule.get('description', '')[:60]}"

        return True, "All evolved rules passed"

    def get_adapted_params(self) -> Dict:
        """Return current evolved strategy parameters."""
        return {
            "rsi_threshold":    self._strategy.rsi_threshold,
            "confidence_gate":  self._strategy.confidence_gate,
            "min_rr":           self._strategy.min_rr,
            "max_positions":    self._strategy.max_positions,
            "signal_weights":   self._strategy.signal_weights,
            "active_rules":     len(self._strategy.active_rules),
            "total_evolutions": self._strategy.total_evolutions,
        }

    # ──────────────────────────────────────────────────────────────
    # PROPOSAL VALIDATION
    # ──────────────────────────────────────────────────────────────

    def _validate_proposal(self, proposal: EvolutionProposal, trades: List[Dict]) -> bool:
        """
        Test a proposed rule on historical data.
        Only adopt if it improves OOS metrics.
        Walk-forward: test on most recent 20% of trades (OOS).
        """
        if not trades:
            return False

        split    = int(len(trades) * 0.8)
        oos_data = trades[split:]

        if len(oos_data) < 5:
            return False

        # Compute baseline OOS performance
        baseline_pf = self._profit_factor(oos_data)

        # Simulate applying the rule
        filtered = self._simulate_rule(proposal, oos_data)
        if len(filtered) < 3:
            return False   # rule filters too aggressively

        rule_pf = self._profit_factor(filtered)

        # Does it improve profit factor by minimum threshold?
        improvement = rule_pf - baseline_pf
        proposal.is_perf  = self._profit_factor(trades[:split])
        proposal.oos_perf = improvement
        proposal.confidence = min(0.95, len(oos_data) / 30 * 0.5 + improvement * 2)

        logger.debug(
            f"  Validating '{proposal.description[:50]}': "
            f"baseline_pf={baseline_pf:.2f} → rule_pf={rule_pf:.2f} "
            f"(improvement={improvement:.2f})"
        )

        return improvement >= self.MIN_OOS_IMPROVEMENT

    def _simulate_rule(self, proposal: EvolutionProposal, trades: List[Dict]) -> List[Dict]:
        """Simulate applying a rule to filter the trade list."""
        rule_type = proposal.rule_type

        if rule_type == "TIMING":
            # Extract time window from description
            hour_str = [t for t in proposal.code_change.split() if ":" in t]
            if hour_str:
                try:
                    hour = int(hour_str[0].split(":")[0])
                    return [t for t in trades
                            if t.get("entry_time", "10:00").split(":")[0] != str(hour)]
                except Exception:
                    pass

        elif rule_type == "PARAMETER":
            # Apply RSI threshold change
            if "rsi" in proposal.code_change.lower():
                try:
                    new_thresh = float([w for w in proposal.code_change.split() if w.replace(".","").isdigit()][-1])
                    return [t for t in trades if t.get("rsi_at_entry", 0) >= new_thresh]
                except Exception:
                    pass

        elif rule_type == "SECTOR_RULE":
            # Filter trades where the rule applies
            if "avoid" in proposal.code_change.lower():
                sector = proposal.code_change.split("avoid")[-1].strip().split()[0].upper()
                return [t for t in trades if t.get("sector", "").upper() != sector[:6]]

        elif rule_type == "FILTER":
            # Add minimum MTF alignment requirement
            if "mtf" in proposal.code_change.lower():
                return [t for t in trades if t.get("mtf_alignment_count", 0) >= 2]

        # Default: no filtering
        return trades

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _create_proposal(self, brain_proposal: Dict) -> Optional[EvolutionProposal]:
        """Convert a brain proposal into a testable EvolutionProposal."""
        try:
            change      = brain_proposal.get("change", "")
            rationale   = brain_proposal.get("rationale", "")
            confidence  = float(brain_proposal.get("confidence", 0.5))

            # Classify the rule type
            rule_type = "FILTER"
            if any(w in change.lower() for w in ["rsi", "threshold", "volume"]):
                rule_type = "PARAMETER"
            elif any(w in change.lower() for w in ["morning", "afternoon", "pm", "am", "hour"]):
                rule_type = "TIMING"
            elif any(w in change.lower() for w in ["sector", "bank", "it", "pharma", "auto"]):
                rule_type = "SECTOR_RULE"
            elif any(w in change.lower() for w in ["avoid", "never", "skip", "don't"]):
                rule_type = "ANTI_PATTERN"

            import hashlib
            rule_id = hashlib.md5(change.encode()).hexdigest()[:8]

            return EvolutionProposal(
                rule_id=rule_id,
                rule_type=rule_type,
                description=change[:200],
                code_change=change,
                is_perf=0.0,
                oos_perf=0.0,
                confidence=confidence,
            )
        except Exception as e:
            logger.debug(f"Proposal creation error: {e}")
            return None

    def _adopt_rule(self, proposal: EvolutionProposal):
        """Add a validated rule to the live strategy."""
        proposal.adopted = True
        rule_entry = {
            "id":          proposal.rule_id,
            "type":        proposal.rule_type,
            "description": proposal.description,
            "conditions":  {},
            "adopted_at":  date.today().isoformat(),
            "oos_improvement": proposal.oos_perf,
        }

        # Update specific strategy parameters
        if proposal.rule_type == "PARAMETER":
            if "rsi" in proposal.code_change.lower():
                try:
                    new_val = float([w for w in proposal.code_change.split()
                                     if w.replace(".", "").isdigit()][-1])
                    if 65 <= new_val <= 80:
                        self._strategy.rsi_threshold = new_val
                        logger.info(f"  RSI threshold updated to {new_val}")
                except Exception:
                    pass

        elif proposal.rule_type == "TIMING":
            self._strategy.time_restrictions[proposal.code_change[:20]] = {
                "avoid":  True,
                "reason": proposal.description[:100],
            }

        elif proposal.rule_type == "SECTOR_RULE":
            sector = proposal.code_change.upper().split("SECTOR")[-1].strip()[:10]
            if sector:
                self._strategy.sector_restrictions[sector] = {
                    "avoid":  True,
                    "reason": proposal.description[:100],
                }

        # Add to active rules
        self._strategy.active_rules.append(rule_entry)
        self._strategy.version += 1
        self._log_proposal(proposal)
        self._save_strategy()

    def _prune_weak_rules(self, recent_trades: List[Dict]) -> List[str]:
        """Remove rules that are no longer improving performance."""
        pruned = []
        rules_to_keep = []

        for rule in self._strategy.active_rules:
            adopted = datetime.fromisoformat(rule.get("adopted_at", "2025-01-01"))
            days_active = (date.today() - adopted.date()).days

            if days_active < 14:  # Give new rules 2 weeks to prove themselves
                rules_to_keep.append(rule)
                continue

            # Check if rule is still helping
            # (Simplified: if OOS improvement was > 0.10 when adopted, keep it for 60 days)
            if rule.get("oos_improvement", 0) < 0.03 and days_active > 30:
                self._strategy.retired_rules.append({
                    **rule,
                    "retired_at": date.today().isoformat(),
                    "reason": "Insufficient sustained improvement",
                })
                pruned.append(rule["description"][:60])
            else:
                rules_to_keep.append(rule)

        if pruned:
            self._strategy.active_rules = rules_to_keep
            self._save_strategy()

        return pruned

    def _check_anti_pattern(self, rule: Dict, symbol: str, context: Dict) -> bool:
        """Check if an anti-pattern condition is present."""
        conditions = rule.get("conditions", {})
        for key, val in conditions.items():
            ctx_val = context.get(key)
            if isinstance(val, str) and ctx_val == val:
                return True
            elif isinstance(val, (int, float)) and ctx_val is not None:
                if ctx_val > val:
                    return True
        return False

    def _profit_factor(self, trades: List[Dict]) -> float:
        """Compute profit factor from a list of trades."""
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss   = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
        return gross_profit / max(gross_loss, 1e-6)

    def _compute_performance_summary(self, trades: List[Dict]) -> Dict:
        """Build performance summary for brain to analyse."""
        closed = [t for t in trades if t.get("exit_reason")]
        if not closed:
            return {}
        wins = [t for t in closed if t.get("won")]
        return {
            "total_trades":     len(closed),
            "win_rate":         len(wins) / len(closed),
            "avg_pnl":          sum(t.get("pnl", 0) for t in closed) / len(closed),
            "profit_factor":    self._profit_factor(closed),
            "total_pnl":        sum(t.get("pnl", 0) for t in closed),
            "period_days":      90,
            "active_rules":     len(self._strategy.active_rules),
            "total_evolutions": self._strategy.total_evolutions,
        }

    def _log_proposal(self, proposal: EvolutionProposal):
        with sqlite3.connect(EVOLUTION_DB) as conn:
            conn.execute("""
                INSERT INTO evolution_log
                (rule_id, rule_type, description, oos_perf, adopted, rejected, proposed_at)
                VALUES (?,?,?,?,?,?,?)
            """, (
                proposal.rule_id, proposal.rule_type, proposal.description,
                proposal.oos_perf, int(proposal.adopted), int(proposal.rejected),
                proposal.proposed_at,
            ))

    def _load_strategy(self) -> EvolvedStrategy:
        if os.path.exists(STRATEGY_FILE):
            try:
                with open(STRATEGY_FILE) as f:
                    data = json.load(f)
                return EvolvedStrategy(**{k: v for k, v in data.items()
                                          if k in EvolvedStrategy.__dataclass_fields__})
            except Exception:
                pass
        s = EvolvedStrategy(created_at=date.today().isoformat())
        self._save_strategy(s)
        return s

    def _save_strategy(self, strategy: EvolvedStrategy = None):
        os.makedirs(os.path.dirname(STRATEGY_FILE), exist_ok=True)
        s = strategy or self._strategy
        with open(STRATEGY_FILE, "w") as f:
            json.dump(asdict(s), f, indent=2, default=str)

    def _init_db(self):
        os.makedirs(os.path.dirname(EVOLUTION_DB), exist_ok=True)
        with sqlite3.connect(EVOLUTION_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id     TEXT,
                    rule_type   TEXT,
                    description TEXT,
                    oos_perf    REAL,
                    adopted     INTEGER,
                    rejected    INTEGER,
                    proposed_at TEXT,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)


# Singleton
evolution_engine = AutonomousEvolutionEngine()
