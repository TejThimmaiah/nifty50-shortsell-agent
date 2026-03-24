"""
Tej LangGraph Orchestrator
============================
Replaces custom BrainOrchestrator with production-grade LangGraph agent graph.

Graph nodes:
  observe → recall → analyze → debate → decide → execute → reflect

Each node is an autonomous agent step with full state passing.
"""

import os
import logging
from typing import TypedDict, Annotated, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("langraph_orchestrator")
IST = ZoneInfo("Asia/Kolkata")

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not installed — run: pip install langgraph")


class TradingState(TypedDict):
    """Full state passed between agent nodes."""
    timestamp:       str
    symbol:          str
    market_data:     dict
    memory_insight:  str
    signals:         list
    debate_result:   dict
    decision:        dict
    execution:       dict
    reflection:      str
    errors:          list


def observe_node(state: TradingState) -> TradingState:
    """Node 1: Observe market — fetch live data for symbol."""
    try:
        from data.free_market_data import get_ohlcv
        symbol = state["symbol"]
        df = get_ohlcv(symbol, days=30)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            state["market_data"] = {
                "close":  float(latest["close"]),
                "volume": float(latest["volume"]),
                "high":   float(latest["high"]),
                "low":    float(latest["low"]),
                "open":   float(latest["open"]),
                "df":     df,
            }
        logger.info(f"[OBSERVE] {symbol} — close: {state['market_data'].get('close', 0):.2f}")
    except Exception as e:
        state["errors"].append(f"observe: {e}")
    return state


def recall_node(state: TradingState) -> TradingState:
    """Node 2: Recall — find similar past trades in vector memory."""
    try:
        from intelligence.vector_memory import vector_memory
        from agents.technical_analyst import calculate_all

        df = state["market_data"].get("df")
        if df is not None:
            sig = calculate_all(df, state["symbol"])
            context = {
                "rsi":          sig.rsi if hasattr(sig, "rsi") else 50,
                "master_score": getattr(sig, "confidence", 0.5),
            }
            state["memory_insight"] = vector_memory.get_insight(context, state["symbol"])
        logger.info(f"[RECALL] {state['memory_insight'][:80]}")
    except Exception as e:
        state["errors"].append(f"recall: {e}")
        state["memory_insight"] = "Memory unavailable"
    return state


def analyze_node(state: TradingState) -> TradingState:
    """Node 3: Analyze — run all 11 intelligence layers."""
    try:
        from agents.technical_analyst import calculate_all
        from intelligence.master_scorer import master_scorer

        df = state["market_data"].get("df")
        symbol = state["symbol"]
        if df is not None:
            sig = calculate_all(df, symbol)
            ms = master_scorer.score(
                symbol,
                [sig.signal] if sig.signal else [],
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                target=sig.target,
                capital=100000
            )
            state["signals"] = [{
                "signal":       sig.signal,
                "entry":        sig.entry_price,
                "stop":         sig.stop_loss,
                "target":       sig.target,
                "confidence":   sig.confidence,
                "master_score": ms.final_score,
            }]
        logger.info(f"[ANALYZE] {symbol} — score: {state['signals'][0].get('master_score', 0):.2f}")
    except Exception as e:
        state["errors"].append(f"analyze: {e}")
    return state


def debate_node(state: TradingState) -> TradingState:
    """
    Node 4: Multi-agent debate — Bull vs Bear vs Neutral.
    Three perspectives argue. Best argument wins.
    """
    try:
        import requests
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key or not state["signals"]:
            state["debate_result"] = {"verdict": "SKIP", "reason": "No signal"}
            return state

        sig = state["signals"][0]
        symbol = state["symbol"]
        score = sig.get("master_score", 0.5)

        prompt = f"""You are 3 trading agents debating whether to short {symbol}.

Signal: {sig.get('signal')} | Score: {score:.2f} | Entry: {sig.get('entry')} | Stop: {sig.get('stop')} | Target: {sig.get('target')}
Memory: {state.get('memory_insight', 'none')}

BEAR AGENT (wants to short): Make the case for shorting.
BULL AGENT (against shorting): Make the case against.
NEUTRAL AGENT (risk focused): Assess risk/reward objectively.
VERDICT: Based on debate, final decision: SHORT or SKIP, and why in one sentence.

Keep each argument to 1 sentence. End with VERDICT: SHORT or VERDICT: SKIP"""

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 300},
            timeout=20,
        )
        if r.ok:
            text = r.json()["choices"][0]["message"]["content"]
            verdict = "SHORT" if "VERDICT: SHORT" in text else "SKIP"
            state["debate_result"] = {"verdict": verdict, "debate": text, "raw_score": score}
            logger.info(f"[DEBATE] {symbol} — verdict: {verdict}")
        else:
            state["debate_result"] = {"verdict": "SHORT" if score > 0.65 else "SKIP", "reason": "Groq unavailable"}
    except Exception as e:
        state["errors"].append(f"debate: {e}")
        state["debate_result"] = {"verdict": "SKIP"}
    return state


def decide_node(state: TradingState) -> TradingState:
    """Node 5: Decide — final go/no-go with risk check."""
    try:
        from agents.risk_manager import RiskManagerAgent
        verdict = state["debate_result"].get("verdict", "SKIP")
        sig = state["signals"][0] if state["signals"] else {}

        if verdict == "SHORT" and sig:
            rm = RiskManagerAgent(capital=100000)
            approval = rm.approve_trade(
                state["symbol"],
                sig.get("entry", 0),
                sig.get("stop", 0),
                10,
                "SHORT"
            )
            state["decision"] = {
                "action":   "SHORT" if approval.approved else "SKIP",
                "reason":   approval.reason if not approval.approved else "All checks passed",
                "quantity": approval.quantity if approval.approved else 0,
                "entry":    sig.get("entry", 0),
                "stop":     sig.get("stop", 0),
                "target":   sig.get("target", 0),
            }
        else:
            state["decision"] = {"action": "SKIP", "reason": "Debate verdict: skip"}
        logger.info(f"[DECIDE] {state['symbol']} — {state['decision']['action']}")
    except Exception as e:
        state["errors"].append(f"decide: {e}")
        state["decision"] = {"action": "SKIP"}
    return state


def execute_node(state: TradingState) -> TradingState:
    """Node 6: Execute — place live order if approved."""
    try:
        decision = state["decision"]
        if decision.get("action") == "SHORT":
            from agents.trade_executor import TradeExecutorAgent
            executor = TradeExecutorAgent()
            result = executor.short_sell(
                state["symbol"],
                decision.get("quantity", 1),
                decision.get("entry", 0),
                decision.get("stop", 0),
                decision.get("target", 0),
            )
            state["execution"] = result
            logger.info(f"[EXECUTE] {state['symbol']} — {result.get('status')}")
        else:
            state["execution"] = {"status": "SKIPPED"}
    except Exception as e:
        state["errors"].append(f"execute: {e}")
        state["execution"] = {"status": "ERROR", "error": str(e)}
    return state


def reflect_node(state: TradingState) -> TradingState:
    """Node 7: Reflect — brain learns from this decision."""
    try:
        from brain.neural_core import brain
        reflection = brain.reflect_on_decision(
            symbol=state["symbol"],
            decision=state["decision"],
            execution=state["execution"],
            signals=state["signals"],
            debate=state.get("debate_result", {}),
        )
        state["reflection"] = reflection
        logger.info(f"[REFLECT] {state['symbol']} — reflection stored")
    except Exception as e:
        state["errors"].append(f"reflect: {e}")
        state["reflection"] = "Reflection unavailable"
    return state


def build_trading_graph():
    """Build the LangGraph trading agent graph."""
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available — using fallback")
        return None

    workflow = StateGraph(TradingState)
    workflow.add_node("observe",  observe_node)
    workflow.add_node("recall",   recall_node)
    workflow.add_node("analyze",  analyze_node)
    workflow.add_node("debate",   debate_node)
    workflow.add_node("decide",   decide_node)
    workflow.add_node("execute",  execute_node)
    workflow.add_node("reflect",  reflect_node)

    workflow.set_entry_point("observe")
    workflow.add_edge("observe",  "recall")
    workflow.add_edge("recall",   "analyze")
    workflow.add_edge("analyze",  "debate")
    workflow.add_edge("debate",   "decide")
    workflow.add_edge("decide",   "execute")
    workflow.add_edge("execute",  "reflect")
    workflow.add_edge("reflect",  END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_symbol(symbol: str) -> dict:
    """Run the full agent graph for one symbol."""
    graph = build_trading_graph()
    if not graph:
        return {"status": "unavailable"}

    initial_state = TradingState(
        timestamp=datetime.now(IST).isoformat(),
        symbol=symbol,
        market_data={},
        memory_insight="",
        signals=[],
        debate_result={},
        decision={},
        execution={},
        reflection="",
        errors=[],
    )

    config = {"configurable": {"thread_id": f"{symbol}_{datetime.now(IST).strftime('%Y%m%d')}"}}
    final_state = graph.invoke(initial_state, config=config)
    return final_state


trading_graph = build_trading_graph()
