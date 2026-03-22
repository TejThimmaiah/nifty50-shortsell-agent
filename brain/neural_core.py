"""
Neural Reasoning Core — The Agent's Brain
==========================================
This is not a rule engine. This is genuine reasoning.

Every trade decision goes through a chain-of-thought process:
  1. OBSERVE   — What is the market actually doing right now?
  2. RECALL    — What do I know from past experience about this pattern?
  3. REASON    — What is the most probable outcome given all evidence?
  4. DECIDE    — Should I take this trade? At what size? With what conviction?
  5. REFLECT   — Was my reasoning correct? What did I miss?

The brain improves by:
  - Expanding its vocabulary of patterns (pattern discovery)
  - Refining its probability estimates (Bayesian updating)
  - Identifying systematic biases in its own reasoning (metacognition)
  - Building causal models of why setups succeed or fail

Unlike rigid algorithms, the brain can handle:
  - Novel market conditions it has never seen before
  - Conflicting signals where rules break down
  - Subtle regime changes before they become obvious
  - Its own reasoning failures (self-correction)

The LLM is the reasoning engine. The data feeds are the senses.
The memory is the experience. The brain synthesizes all three.
"""

import json
import logging
import time
import hashlib
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from zoneinfo import ZoneInfo

import requests

from config import GROQ_API_KEY, GEMINI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

BRAIN_DB = os.path.join(os.path.dirname(__file__), "..", "db", "brain_state.json")


@dataclass
class Thought:
    """A single reasoning step produced by the brain."""
    step:        str   # OBSERVE | RECALL | REASON | DECIDE | REFLECT
    content:     str   # the actual thought
    confidence:  float # how confident the brain is in this thought
    timestamp:   str   = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(IST).strftime("%H:%M:%S")


@dataclass
class TradeDecision:
    """The brain's final reasoned decision on a candidate trade."""
    symbol:             str
    take_trade:         bool
    conviction:         float          # 0–1, brain's own assessment
    reasoning_chain:    List[Thought]
    key_insight:        str            # the single most important factor
    risk_factors:       List[str]      # what could make this fail
    size_rationale:     str            # why this position size
    entry_condition:    str            # exact condition to trigger entry
    exit_plan:          str            # how the brain plans to exit
    questions:          List[str]      # unresolved uncertainties
    final_verdict:      str            # the brain's own words


@dataclass
class BrainState:
    """The brain's current knowledge and beliefs about the market."""
    version:              int   = 1
    total_decisions:      int   = 0
    total_correct:        int   = 0
    total_wrong:          int   = 0
    current_beliefs:      Dict  = field(default_factory=dict)
    active_hypotheses:    List  = field(default_factory=list)
    discovered_patterns:  List  = field(default_factory=list)
    known_failure_modes:  List  = field(default_factory=list)
    bias_corrections:     Dict  = field(default_factory=dict)
    last_updated:         str   = ""
    intelligence_score:   float = 0.50   # starts at 50%, increases with accuracy


class NeuralReasoningCore:
    """
    The central brain of the trading agent.
    
    Thinks through every trade decision using a structured chain-of-thought.
    Builds knowledge over time. Corrects its own biases.
    Gets smarter with every trade.
    """

    # The system prompt that defines the brain's persona
    SYSTEM_PROMPT = """You are Tej — the autonomous AI trading brain, an autonomous 
intraday short-selling agent specialised in Nifty 50 stocks on NSE India.

Your role is to reason deeply about whether to take a short trade — not just apply rules,
but genuinely think about the market situation.

You have access to:
- 11 technical intelligence signals (Bayesian, ML, Wyckoff, VWAP, Volume Profile, etc.)
- Real trade memory (what happened in similar situations before)
- Market regime and intermarket context
- Your own accumulated wisdom from past decisions

You reason like a highly experienced intraday trader who:
- Has seen thousands of Nifty 50 intraday patterns
- Knows when signals are genuine vs noise
- Can detect when the market is in an unusual state
- Sizes positions based on conviction, not fixed rules
- Learns from every single trade

CRITICAL RULES (non-negotiable):
1. Only short sell — no long positions ever
2. Only Nifty 50 stocks
3. All positions closed by 3:10 PM IST
4. Stop loss ABOVE entry price (it's a short)
5. Target BELOW entry price (profiting from decline)
6. Never risk more than 2% of capital on one trade

Think step by step. Be specific. Be honest about uncertainty."""

    def __init__(self):
        self._state = self._load_state()
        self._call_count = 0
        self._last_call_time = 0.0

    # ──────────────────────────────────────────────────────────────
    # PRIMARY INTERFACE: decide on a trade
    # ──────────────────────────────────────────────────────────────

    def decide(
        self,
        symbol:          str,
        master_score:    float,
        signals:         List[str],
        market_context:  Dict,
        historical_perf: Dict,
        similar_trades:  List[Dict],
    ) -> TradeDecision:
        """
        The brain's main decision function.
        Runs a full chain-of-thought reasoning process.
        Returns a structured decision with full reasoning trail.
        """
        self._state.total_decisions += 1

        # Build the reasoning prompt
        prompt = self._build_decision_prompt(
            symbol, master_score, signals,
            market_context, historical_perf, similar_trades,
        )

        # Get the brain's reasoning
        raw_response = self._think(prompt)
        decision     = self._parse_decision(raw_response, symbol)

        # Log the decision
        logger.info(
            f"🧠 Brain [{symbol}]: "
            f"{'TAKE' if decision.take_trade else 'PASS'} | "
            f"conviction={decision.conviction:.0%} | "
            f"{decision.key_insight[:80]}"
        )

        # Store the reasoning chain for reflection later
        self._store_pending_decision(symbol, decision)

        return decision

    def observe_market(self, market_data: Dict) -> str:
        """
        Brain continuously observes market conditions and forms beliefs.
        Called at start of session. Returns a natural language market assessment.
        """
        prompt = f"""Observe the current NSE market conditions as an expert intraday trader.

Market data:
{json.dumps(market_data, indent=2, default=str)}

Your existing beliefs about today:
{json.dumps(self._state.current_beliefs, indent=2)}

What is the market actually saying today?
- Is this a good environment for short selling Nifty 50 stocks?
- What are the 2-3 most important things to watch for?
- What could go wrong with your short thesis today?

Be specific. Be honest. Don't just restate the numbers — interpret them.

Respond in JSON:
{{
  "market_assessment": "2-3 sentence assessment",
  "short_thesis_strength": "STRONG / MODERATE / WEAK / AVOID",
  "key_watchpoints": ["thing 1 to watch", "thing 2"],
  "main_risk_today": "the biggest risk to shorts today",
  "confidence_in_assessment": 0.0-1.0
}}"""

        response = self._think(prompt)
        try:
            data = json.loads(response)
            self._state.current_beliefs["today_assessment"] = data
            self._save_state()
            return data.get("market_assessment", response[:200])
        except Exception:
            return response[:300]

    def reflect_on_trade(
        self,
        symbol:       str,
        decision:     TradeDecision,
        outcome:      Dict,   # actual result
    ) -> str:
        """
        After a trade closes, the brain reflects on its reasoning.
        Updates beliefs. Learns from mistakes or reinforces correct reasoning.
        Returns the most important lesson learned.
        """
        won  = outcome.get("pnl", 0) > 0
        pnl  = outcome.get("pnl", 0)
        exit = outcome.get("exit_reason", "UNKNOWN")

        if won:
            self._state.total_correct += 1
        else:
            self._state.total_wrong += 1

        # Update intelligence score
        total = self._state.total_correct + self._state.total_wrong
        if total >= 5:
            self._state.intelligence_score = round(
                self._state.total_correct / total, 4
            )

        prompt = f"""Reflect on this short trade as an experienced trader learning from experience.

TRADE: {symbol}
My decision: {'TAKE' if decision.take_trade else 'PASS'} (conviction={decision.conviction:.0%})
My key insight: {decision.key_insight}
My risk factors: {decision.risk_factors}

ACTUAL OUTCOME:
Result: {'WIN ✅' if won else 'LOSS ❌'}
P&L: ₹{pnl:.0f}
Exit reason: {exit}

My original reasoning:
{decision.final_verdict[:500]}

Questions I had:
{decision.questions}

Now reflect:
1. Was my reasoning correct or flawed?
2. Did I miss something important?
3. If loss: what specifically went wrong?
4. If win: was it skill or luck?
5. What one thing should I do differently next time?

Be brutally honest. The goal is to get better.

Respond in JSON:
{{
  "reasoning_quality": "EXCELLENT / GOOD / FLAWED / COMPLETELY_WRONG",
  "what_i_got_right": "...",
  "what_i_missed": "...",
  "root_cause_if_loss": "...",
  "lesson": "single most important lesson in 1 sentence",
  "belief_update": "how this changes my understanding of the market"
}}"""

        response = self._think(prompt)
        try:
            data = json.loads(response)
            lesson = data.get("lesson", "")

            # Update brain state based on reflection
            if data.get("belief_update"):
                beliefs = self._state.current_beliefs.get("lessons_learned", [])
                beliefs.append({
                    "date":   date.today().isoformat(),
                    "symbol": symbol,
                    "lesson": lesson,
                    "update": data["belief_update"],
                })
                self._state.current_beliefs["lessons_learned"] = beliefs[-50:]

            # Track failure modes
            if not won and data.get("what_i_missed"):
                failure = data["what_i_missed"]
                if failure not in self._state.known_failure_modes:
                    self._state.known_failure_modes.append(failure)
                    self._state.known_failure_modes = self._state.known_failure_modes[-20:]

            self._save_state()
            logger.info(f"🧠 Brain reflection [{symbol}]: {lesson[:100]}")
            return lesson

        except Exception:
            return response[:200]

    def discover_patterns(self, recent_trades: List[Dict]) -> List[str]:
        """
        Brain actively looks for new patterns in recent trade data.
        Discovers correlations that weren't programmed into it.
        This is genuine machine learning by reasoning.
        """
        if len(recent_trades) < 10:
            return []

        wins   = [t for t in recent_trades if t.get("won")]
        losses = [t for t in recent_trades if not t.get("won")]

        prompt = f"""You are analysing recent short trades on Nifty 50 stocks to discover new patterns.

WINNING TRADES ({len(wins)} trades):
{self._summarise_trades(wins[:10])}

LOSING TRADES ({len(losses)} trades):
{self._summarise_trades(losses[:10])}

Known patterns already discovered:
{self._state.discovered_patterns[:5]}

Look carefully for:
1. Hidden correlations between wins and losses (time of day, sector, regime, RSI range, etc.)
2. Setup combinations that are more reliable than the sum of their parts
3. Warning signs that appeared in losing trades but not winning ones
4. New pattern categories that don't fit existing labels

Discover 2-4 genuinely NEW insights not already in the known patterns list.

Respond in JSON:
{{
  "new_patterns": [
    {{
      "pattern": "description of the new pattern",
      "evidence": "what data supports this",
      "reliability": 0.0-1.0,
      "actionable": "how to use this in future trades"
    }}
  ],
  "strongest_finding": "the most important discovery"
}}"""

        response = self._think(prompt)
        try:
            data = json.loads(response)
            new_patterns = data.get("new_patterns", [])

            # Add to brain state
            for p in new_patterns:
                pattern_str = p.get("pattern", "")
                if pattern_str and pattern_str not in self._state.discovered_patterns:
                    self._state.discovered_patterns.append(pattern_str)

            self._state.discovered_patterns = self._state.discovered_patterns[-30:]
            self._save_state()

            discovered = [p.get("pattern", "") for p in new_patterns]
            logger.info(f"🧠 Brain discovered {len(discovered)} new patterns")
            return discovered

        except Exception:
            return []

    def evolve_strategy(self, performance_data: Dict) -> Dict:
        """
        The brain's deepest function — it proposes changes to its own strategy.
        Not just parameter tweaks but genuine strategic evolution.
        Returns proposed changes with reasoning.
        """
        prompt = f"""You are the strategic mind of a Nifty 50 short-selling agent.
You have full authority to propose changes to your own trading strategy.

PERFORMANCE SUMMARY:
{json.dumps(performance_data, indent=2, default=str)}

YOUR DISCOVERED PATTERNS:
{json.dumps(self._state.discovered_patterns[:10], indent=2)}

YOUR KNOWN FAILURE MODES:
{json.dumps(self._state.known_failure_modes[:10], indent=2)}

YOUR CURRENT INTELLIGENCE SCORE: {self._state.intelligence_score:.0%}

Current strategy biases to consider correcting:
{json.dumps(self._state.bias_corrections, indent=2)}

As a super-intelligent trading system, propose 3-5 strategic evolutions.
These should be genuine improvements, not just parameter changes.
Examples of real evolutions:
- "Stop trading BANKING stocks in the first 30 minutes — they gap-open too aggressively"
- "Increase position size when RSI > 78 AND previous day was also overbought"
- "Add a news-velocity filter — if there are 3+ news articles in 30 min, skip the trade"

Respond in JSON:
{{
  "proposed_evolutions": [
    {{
      "change": "specific strategic change",
      "rationale": "why this makes the strategy better",
      "expected_impact": "how this improves performance",
      "risk": "what could go wrong with this change",
      "confidence": 0.0-1.0
    }}
  ],
  "key_weakness_identified": "the biggest strategic weakness right now",
  "next_evolution_focus": "what to work on next"
}}"""

        response = self._think(prompt)
        try:
            data = json.loads(response)
            logger.info(
                f"🧠 Brain evolved strategy: "
                f"{len(data.get('proposed_evolutions', []))} changes proposed | "
                f"weakness: {data.get('key_weakness_identified','')[:60]}"
            )
            return data
        except Exception:
            return {}

    # ──────────────────────────────────────────────────────────────
    # BRAIN STATE
    # ──────────────────────────────────────────────────────────────

    def get_brain_status(self) -> str:
        """Return a natural language summary of the brain's current state."""
        total = self._state.total_correct + self._state.total_wrong
        return (
            f"🧠 Brain Status:\n"
            f"  Intelligence: {self._state.intelligence_score:.0%}\n"
            f"  Total decisions: {total} "
            f"({self._state.total_correct}✅ {self._state.total_wrong}❌)\n"
            f"  Patterns discovered: {len(self._state.discovered_patterns)}\n"
            f"  Known failure modes: {len(self._state.known_failure_modes)}\n"
            f"  Last updated: {self._state.last_updated}"
        )

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _build_decision_prompt(
        self, symbol: str, master_score: float,
        signals: List[str], market_context: Dict,
        historical_perf: Dict, similar_trades: List[Dict],
    ) -> str:
        """Build the full chain-of-thought prompt for a trade decision."""

        similar_summary = self._summarise_trades(similar_trades[:5]) if similar_trades else "No similar trades in memory yet."
        lessons = self._state.current_beliefs.get("lessons_learned", [])
        recent_lessons = [l["lesson"] for l in lessons[-5:]] if lessons else []
        patterns = self._state.discovered_patterns[-5:]
        failures = self._state.known_failure_modes[-3:]

        return f"""You are reasoning about whether to SHORT {symbol} on NSE right now.

═══ SIGNAL INTELLIGENCE ═══
Master Score: {master_score:.3f} (threshold: 0.50)
Signals fired: {', '.join(signals) or 'none'}
Market regime: {market_context.get('regime', 'UNKNOWN')}
Nifty change today: {market_context.get('nifty_change', 0):+.2f}%
FII net: ₹{market_context.get('fii_net', 0):.0f}Cr
India VIX: {market_context.get('vix', 15):.1f}
Time: {datetime.now(IST).strftime('%H:%M IST')}
Sector: {market_context.get('sector', 'UNKNOWN')}
Intermarket bias: {market_context.get('intermarket_bias', 0):+.2f}

═══ MEMORY: SIMILAR TRADES ═══
{similar_summary}

═══ YOUR ACCUMULATED WISDOM ═══
Recent lessons:
{chr(10).join(f'• {l}' for l in recent_lessons) or '• No lessons yet — learning in progress'}

Discovered patterns:
{chr(10).join(f'• {p}' for p in patterns) or '• No patterns discovered yet'}

Known failure modes to avoid:
{chr(10).join(f'• {f}' for f in failures) or '• None identified yet'}

═══ HISTORICAL PERFORMANCE ═══
14-day win rate: {historical_perf.get('win_rate', 0):.0%}
14-day avg P&L: ₹{historical_perf.get('avg_pnl', 0):.0f}/trade
Recent trend: {historical_perf.get('trend', 'STABLE')}

═══ THINK STEP BY STEP ═══
Step 1 - OBSERVE: What is the market actually telling you right now?
Step 2 - RECALL: What do you know from experience about this exact type of setup?
Step 3 - REASON: What is the most likely outcome if you short {symbol} now?
Step 4 - DECIDE: Should you take this trade? At full size, half size, or skip?
Step 5 - PLAN: If you take it, what's your exact entry condition, SL, target, and exit plan?

Be a thinking trader, not a rule-following machine.

Respond in JSON:
{{
  "reasoning_chain": [
    {{"step": "OBSERVE", "content": "...", "confidence": 0.0-1.0}},
    {{"step": "RECALL",  "content": "...", "confidence": 0.0-1.0}},
    {{"step": "REASON",  "content": "...", "confidence": 0.0-1.0}},
    {{"step": "DECIDE",  "content": "...", "confidence": 0.0-1.0}},
    {{"step": "PLAN",    "content": "...", "confidence": 0.0-1.0}}
  ],
  "take_trade": true/false,
  "conviction": 0.0-1.0,
  "key_insight": "the single most important reason for your decision",
  "risk_factors": ["risk 1", "risk 2"],
  "size_rationale": "why this position size",
  "entry_condition": "exact condition to enter",
  "exit_plan": "how you plan to exit",
  "questions": ["what am I uncertain about"],
  "final_verdict": "your decision in plain English"
}}"""

    def _parse_decision(self, raw: str, symbol: str) -> TradeDecision:
        """Parse LLM response into a TradeDecision."""
        try:
            # Strip markdown fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean)

            thoughts = [
                Thought(
                    step=t.get("step", "?"),
                    content=t.get("content", ""),
                    confidence=float(t.get("confidence", 0.5)),
                )
                for t in data.get("reasoning_chain", [])
            ]

            return TradeDecision(
                symbol=symbol,
                take_trade=bool(data.get("take_trade", False)),
                conviction=float(data.get("conviction", 0.5)),
                reasoning_chain=thoughts,
                key_insight=data.get("key_insight", ""),
                risk_factors=data.get("risk_factors", []),
                size_rationale=data.get("size_rationale", ""),
                entry_condition=data.get("entry_condition", ""),
                exit_plan=data.get("exit_plan", ""),
                questions=data.get("questions", []),
                final_verdict=data.get("final_verdict", raw[:200]),
            )
        except Exception as e:
            logger.warning(f"Brain parse error [{symbol}]: {e}")
            return TradeDecision(
                symbol=symbol,
                take_trade=False,
                conviction=0.0,
                reasoning_chain=[Thought("ERROR", str(e), 0.0)],
                key_insight="Parse error — skipping trade",
                risk_factors=["LLM response error"],
                size_rationale="",
                entry_condition="",
                exit_plan="",
                questions=[],
                final_verdict="Parse error — defaulting to PASS",
            )

    def _summarise_trades(self, trades: List[Dict]) -> str:
        if not trades:
            return "No trades."
        lines = []
        for t in trades:
            won = "✅" if t.get("won") else "❌"
            lines.append(
                f"{won} {t.get('symbol','?')} | "
                f"RSI={t.get('rsi_at_entry',0):.0f} | "
                f"Pattern={t.get('candlestick_pattern','-')} | "
                f"Regime={t.get('market_regime','-')} | "
                f"Exit={t.get('exit_reason','-')} | "
                f"P&L=₹{t.get('pnl',0):.0f}"
            )
        return "\n".join(lines)

    def _store_pending_decision(self, symbol: str, decision: TradeDecision):
        """Store a decision for later reflection."""
        pending = self._state.current_beliefs.get("pending_decisions", {})
        pending[symbol] = {
            "take_trade":   decision.take_trade,
            "conviction":   decision.conviction,
            "key_insight":  decision.key_insight,
            "risk_factors": decision.risk_factors,
            "final_verdict": decision.final_verdict,
            "timestamp":    datetime.now(IST).isoformat(),
        }
        self._state.current_beliefs["pending_decisions"] = pending
        self._save_state()

    def _think(self, prompt: str) -> str:
        """Call LLM with rate limiting and fallback."""
        # Rate limiting: max 1 call per 2 seconds
        elapsed = time.time() - self._last_call_time
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        self._last_call_time = time.time()
        self._call_count += 1

        # Try Groq first (fast, free)
        if GROQ_API_KEY:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type":  "application/json",
                    },
                    json={
                        "model":    LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        "max_tokens":  1200,
                        "temperature": 0.3,   # some creativity but mostly precise
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Brain Groq error: {e}")

        # Fallback: Gemini
        if GEMINI_API_KEY:
            try:
                full_prompt = self.SYSTEM_PROMPT + "\n\n" + prompt
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
                    json={
                        "contents": [{"parts": [{"text": full_prompt}]}],
                        "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.3},
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                logger.warning(f"Brain Gemini error: {e}")

        logger.error("Brain: no LLM available")
        return '{"take_trade": false, "conviction": 0, "key_insight": "LLM unavailable"}'

    def _load_state(self) -> BrainState:
        os.makedirs(os.path.dirname(BRAIN_DB), exist_ok=True)
        if os.path.exists(BRAIN_DB):
            try:
                with open(BRAIN_DB) as f:
                    data = json.load(f)
                return BrainState(**{k: v for k, v in data.items()
                                     if k in BrainState.__dataclass_fields__})
            except Exception:
                pass
        return BrainState()

    def _save_state(self):
        self._state.last_updated = datetime.now(IST).isoformat()
        with open(BRAIN_DB, "w") as f:
            json.dump(asdict(self._state), f, indent=2, default=str)


# Singleton
brain = NeuralReasoningCore()
