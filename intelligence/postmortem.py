"""
Trade Post-Mortem AI Reviewer
After every trade closes, the LLM writes a detailed post-mortem:
  - Was this trade set up correctly?
  - What did the entry miss?
  - Was the stop too tight / too loose?
  - What would a perfect version of this trade look like?
  - What's the pattern in today's winners vs losers?

These post-mortems are stored in trade_memory and feed the nightly
self-improvement cycle. Over time they build a rich knowledge base
of what the agent has learned about specific setups, stocks, and conditions.

The agent literally reviews its own work like a professional trader would —
journaling each trade and extracting lessons.
"""

import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from config import GROQ_API_KEY, GEMINI_API_KEY, LLM_MODEL
from intelligence.trade_memory import memory_store, TradeMemory

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class TradePostMortemReviewer:
    """
    Writes AI post-mortems for every closed trade.
    Runs asynchronously — does not block trade execution.
    """

    def review_trade(
        self,
        trade: Dict,
        market_context: Dict = None,
    ) -> Optional[Dict]:
        """
        Write a detailed post-mortem for a single trade.
        Returns dict with lesson, what_worked, what_failed.
        """
        if not trade.get("exit_reason"):
            return None  # Trade not yet closed

        prompt = self._build_prompt(trade, market_context or {})
        response = self._call_llm(prompt)

        if not response:
            return None

        try:
            data = json.loads(response)
        except Exception:
            # Try to extract from plain text
            data = {
                "lesson":      response[:200],
                "what_worked": "",
                "what_failed": "",
                "grade":       "B",
                "improvement": "",
            }

        # Write back to trade memory
        if trade.get("trade_id"):
            memory_store.update_outcome(
                trade_id=trade["trade_id"],
                exit_price=trade.get("exit_price", 0),
                exit_reason=trade.get("exit_reason", ""),
                pnl=trade.get("pnl", 0),
                pnl_pct=trade.get("pnl_pct", 0),
                holding_minutes=trade.get("holding_minutes", 0),
                lesson=data.get("lesson", ""),
                what_worked=data.get("what_worked", ""),
                what_failed=data.get("what_failed", ""),
            )

        logger.info(
            f"Post-mortem [{trade.get('symbol')}]: "
            f"Grade={data.get('grade','?')} | {data.get('lesson','')[:80]}"
        )
        return data

    def review_day(self, trades: List[Dict], day_context: Dict = None) -> Optional[Dict]:
        """
        End-of-day review: patterns across ALL trades today.
        Identifies systematic issues and improvements.
        """
        closed = [t for t in trades if t.get("exit_reason")]
        if len(closed) < 2:
            return None

        winners = [t for t in closed if t.get("pnl", 0) > 0]
        losers  = [t for t in closed if t.get("pnl", 0) <= 0]

        prompt = self._build_day_review_prompt(closed, winners, losers, day_context or {})
        response = self._call_llm(prompt)

        if not response:
            return None

        try:
            return json.loads(response)
        except Exception:
            return {
                "daily_lesson": response[:300],
                "pattern_in_winners": "",
                "pattern_in_losers": "",
                "tomorrow_action": "",
            }

    def identify_bad_habits(self, days: int = 30) -> List[str]:
        """
        Analyse trade memories to find recurring mistakes.
        Returns a list of bad habits the agent should fix.
        """
        recent = memory_store.get_recent(days=days)
        losses = [t for t in recent if not t.get("won") and t.get("lesson")]
        if len(losses) < 5:
            return []

        lessons = [t.get("lesson", "") for t in losses[:20]]
        prompt = f"""You are reviewing {len(losses)} losing trades from an automated short-selling agent.

Losing trade lessons:
{chr(10).join(f'- {l}' for l in lessons if l)}

Identify 3-5 RECURRING bad habits or systematic mistakes in these losses.
Be specific and actionable.

Respond in JSON:
{{
  "bad_habits": [
    "specific recurring mistake 1",
    "specific recurring mistake 2"
  ]
}}"""

        response = self._call_llm(prompt)
        try:
            return json.loads(response).get("bad_habits", [])
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────
    # PROMPT BUILDERS
    # ──────────────────────────────────────────────────────────────

    def _build_prompt(self, trade: Dict, ctx: Dict) -> str:
        won = trade.get("pnl", 0) > 0
        return f"""You are reviewing a short trade taken by an automated NSE intraday agent.

TRADE DETAILS:
Symbol:         {trade.get('symbol')}
Result:         {'WIN' if won else 'LOSS'} (P&L: ₹{trade.get('pnl', 0):.0f})
Entry:          ₹{trade.get('entry_price', 0):.2f} @ {trade.get('entry_time', '?')} ({trade.get('time_of_day','?')})
Exit:           ₹{trade.get('exit_price', 0):.2f} ({trade.get('exit_reason', '?')})
Stop Loss:      ₹{trade.get('stop_loss', 0):.2f}
Target:         ₹{trade.get('target', 0):.2f}
RSI at entry:   {trade.get('rsi_at_entry', 0):.1f}
Signals fired:  {trade.get('signals_fired', [])}
Pattern:        {trade.get('candlestick_pattern', 'none')}
Market regime:  {trade.get('market_regime', '?')}
Sector trend:   {trade.get('sector_trend', '?')}
MTF aligned:    {trade.get('mtf_aligned', False)} ({trade.get('mtf_alignment_count', 0)}/3)
Confidence:     {trade.get('confidence_score', 0):.2f}

MARKET CONTEXT:
Nifty change:   {ctx.get('nifty_change', 0):.2f}%
FII net:        ₹{ctx.get('fii_net', 0):.0f}Cr
VIX:            {ctx.get('vix', 15):.1f}

Analyse this trade professionally. Was it a good setup? What can be improved?

Respond ONLY in JSON:
{{
  "grade": "A/B/C/D (A=perfect execution, D=should not have been taken)",
  "lesson": "single most important lesson from this trade (1 sentence)",
  "what_worked": "what the setup did correctly (if anything)",
  "what_failed": "what went wrong or could be improved",
  "improvement": "one specific change to make this trade better next time",
  "entry_quality": "EXCELLENT/GOOD/POOR",
  "exit_quality": "EXCELLENT/GOOD/POOR/PREMATURE"
}}"""

    def _build_day_review_prompt(
        self, closed: List[Dict], winners: List[Dict], losers: List[Dict], ctx: Dict
    ) -> str:
        def summarise(trades):
            return [
                f"{t.get('symbol')}: {'WIN' if t.get('pnl',0)>0 else 'LOSS'} "
                f"₹{t.get('pnl',0):.0f} | RSI={t.get('rsi_at_entry',0):.0f} | "
                f"regime={t.get('market_regime','?')} | exit={t.get('exit_reason','?')}"
                for t in trades
            ]

        return f"""End-of-day review for NSE short-selling agent.

WINNERS ({len(winners)} trades):
{chr(10).join(summarise(winners)) or 'None'}

LOSERS ({len(losers)} trades):
{chr(10).join(summarise(losers)) or 'None'}

MARKET TODAY:
{json.dumps(ctx, indent=2)}

Find patterns that separate winners from losers today.
What should change for tomorrow?

Respond in JSON:
{{
  "daily_lesson": "most important learning from today (2-3 sentences)",
  "pattern_in_winners": "what today's winning trades had in common",
  "pattern_in_losers": "what today's losing trades had in common",
  "tomorrow_action": "one specific action to take tomorrow based on today"
}}"""

    # ──────────────────────────────────────────────────────────────
    # LLM CALL
    # ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        if GROQ_API_KEY:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": LLM_MODEL,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 600, "temperature": 0.2},
                    timeout=25,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Groq postmortem error: {e}")

        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                resp = requests.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"maxOutputTokens": 600, "temperature": 0.2}},
                    timeout=25,
                )
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                logger.warning(f"Gemini postmortem error: {e}")

        return None


# Singleton
postmortem_reviewer = TradePostMortemReviewer()
