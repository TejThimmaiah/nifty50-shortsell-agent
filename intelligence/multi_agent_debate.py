"""
Tej Multi-Agent Debate System
================================
3 AI agents argue every trade before execution.

BEAR AGENT  — wants to short, finds bearish evidence
BULL AGENT  — argues against shorting, finds bullish evidence
RISK AGENT  — focuses purely on risk/reward math

Best argument wins. Tej only trades when Bear wins convincingly.
This eliminates low-conviction trades and improves win rate.
"""

import os
import logging
import requests
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("multi_agent_debate")
IST = ZoneInfo("Asia/Kolkata")

GROQ_API = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "llama-3.3-70b-versatile"


@dataclass
class DebateResult:
    verdict:        str    # "SHORT" or "SKIP"
    confidence:     float  # 0-1
    bear_argument:  str
    bull_argument:  str
    risk_argument:  str
    final_reasoning: str
    conviction:     str   # "HIGH" / "MEDIUM" / "LOW"


def _call_groq(system: str, user: str, max_tokens: int = 200) -> str:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return ""
    try:
        r = requests.post(
            GROQ_API,
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=20,
        )
        if r.ok:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
    return ""


def bear_agent(symbol: str, signals: dict, sentiment: dict, memory: str) -> str:
    """Bear agent — argues FOR shorting."""
    system = (
        "You are the BEAR AGENT. Your job is to find every reason to short this stock. "
        "Focus on: technical weakness, bearish signals, negative sentiment, resistance levels, "
        "sector weakness. Be concise — 2-3 sentences maximum. End with: BEAR VERDICT: SHORT or BEAR VERDICT: WEAK"
    )
    user = (
        f"Stock: {symbol}\n"
        f"Signal: {signals.get('signal')} | RSI: {signals.get('rsi', 50):.0f} | "
        f"Score: {signals.get('master_score', 0.5):.2f}\n"
        f"Sentiment: {sentiment.get('label', 'neutral')} ({sentiment.get('score', 0):+.2f})\n"
        f"Memory: {memory}\n"
        f"Entry: {signals.get('entry', 0):.2f} | Stop: {signals.get('stop', 0):.2f} | "
        f"Target: {signals.get('target', 0):.2f}"
    )
    return _call_groq(system, user)


def bull_agent(symbol: str, signals: dict, sentiment: dict) -> str:
    """Bull agent — argues AGAINST shorting."""
    system = (
        "You are the BULL AGENT. Your job is to find every reason NOT to short this stock. "
        "Focus on: support levels, bullish divergence, positive momentum, sector strength, "
        "institutional buying, any reason the short could fail. "
        "Be concise — 2-3 sentences. End with: BULL VERDICT: AVOID or BULL VERDICT: WEAK_RESISTANCE"
    )
    user = (
        f"Stock: {symbol}\n"
        f"Signal: {signals.get('signal')} | RSI: {signals.get('rsi', 50):.0f}\n"
        f"Sentiment: {sentiment.get('label', 'neutral')}\n"
        f"Entry: {signals.get('entry', 0):.2f} | Stop: {signals.get('stop', 0):.2f}"
    )
    return _call_groq(system, user)


def risk_agent(symbol: str, signals: dict, capital: float = 100000) -> str:
    """Risk agent — pure math focus."""
    system = (
        "You are the RISK AGENT. Focus only on risk/reward math. "
        "Calculate: R:R ratio, max loss in Rs, win rate needed to break even, "
        "Kelly fraction. Be purely mathematical. "
        "End with: RISK VERDICT: ACCEPTABLE or RISK VERDICT: POOR"
    )
    entry  = signals.get("entry", 0)
    stop   = signals.get("stop", 0)
    target = signals.get("target", 0)
    risk_per_share   = abs(stop - entry) if stop and entry else 0
    reward_per_share = abs(entry - target) if target and entry else 0
    rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    max_loss = risk_per_share * signals.get("quantity", 10)

    user = (
        f"Stock: {symbol}\n"
        f"Entry: {entry:.2f} | Stop: {stop:.2f} | Target: {target:.2f}\n"
        f"R:R ratio: {rr:.2f}:1\n"
        f"Max loss: Rs {max_loss:.0f} | Capital: Rs {capital:.0f}\n"
        f"Risk %: {(max_loss/capital*100):.2f}%\n"
        f"Win rate needed to break even: {1/(1+rr)*100:.0f}%"
    )
    return _call_groq(system, user)


def judge_agent(bear: str, bull: str, risk: str, symbol: str, score: float) -> DebateResult:
    """Judge agent — weighs all arguments and delivers final verdict."""
    system = (
        "You are the JUDGE. You've heard from Bear, Bull, and Risk agents. "
        "Weigh the arguments and deliver a final verdict. "
        "Only approve SHORT if: Bear makes strong case, Risk is acceptable, Bull's resistance is weak. "
        "Be decisive. Respond in this exact format:\n"
        "VERDICT: SHORT or SKIP\n"
        "CONFIDENCE: 0.0-1.0\n"
        "CONVICTION: HIGH or MEDIUM or LOW\n"
        "REASON: one sentence"
    )
    user = (
        f"Stock: {symbol} | Master Score: {score:.2f}\n\n"
        f"BEAR said: {bear}\n\n"
        f"BULL said: {bull}\n\n"
        f"RISK said: {risk}"
    )
    response = _call_groq(system, user, max_tokens=150)

    # Parse response
    verdict    = "SHORT" if "VERDICT: SHORT" in response else "SKIP"
    confidence = 0.7 if verdict == "SHORT" else 0.3
    conviction = "HIGH"
    reason     = "Debate complete"

    for line in response.split("\n"):
        if "CONFIDENCE:" in line:
            try:
                confidence = float(line.split(":")[1].strip())
            except Exception:
                pass
        if "CONVICTION:" in line:
            conviction = line.split(":")[1].strip()
        if "REASON:" in line:
            reason = line.split(":", 1)[1].strip() if ":" in line else reason

    return DebateResult(
        verdict=verdict,
        confidence=confidence,
        bear_argument=bear,
        bull_argument=bull,
        risk_argument=risk,
        final_reasoning=reason,
        conviction=conviction,
    )


def run_debate(symbol: str, signals: dict, sentiment: dict = None,
               memory: str = "", capital: float = 100000) -> DebateResult:
    """
    Run full 3-agent debate for a trade decision.
    Returns DebateResult with verdict and reasoning.
    """
    if sentiment is None:
        sentiment = {"label": "neutral", "score": 0.0}

    logger.info(f"Starting debate for {symbol}...")

    bear = bear_agent(symbol, signals, sentiment, memory)
    bull = bull_agent(symbol, signals, sentiment)
    risk = risk_agent(symbol, signals, capital)

    score = signals.get("master_score", 0.5)
    result = judge_agent(bear, bull, risk, symbol, score)

    logger.info(
        f"Debate {symbol}: {result.verdict} | "
        f"Confidence: {result.confidence:.2f} | Conviction: {result.conviction}"
    )
    return result


def format_debate_for_telegram(symbol: str, result: DebateResult) -> str:
    """Format debate result as Telegram message."""
    emoji = "🔴" if result.verdict == "SHORT" else "⏭️"
    conv_emoji = {"HIGH": "💪", "MEDIUM": "👍", "LOW": "😐"}.get(result.conviction, "")

    return (
        f"<b>Agent Debate: {symbol}</b>\n\n"
        f"{emoji} <b>Verdict: {result.verdict}</b> {conv_emoji}\n"
        f"Confidence: {result.confidence:.0%} | Conviction: {result.conviction}\n\n"
        f"🐻 Bear: {result.bear_argument[:120]}...\n\n"
        f"🐂 Bull: {result.bull_argument[:120]}...\n\n"
        f"⚖️ Risk: {result.risk_argument[:120]}...\n\n"
        f"⚖️ Judge: {result.final_reasoning}"
    )
