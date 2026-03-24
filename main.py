"""
Tej — Autonomous AI Trading Agent
====================================
Built by Tej Thimmaiah. Named after him. Built for him.
Mission: First billionaire in the Thimmaiah family.

Built on open-source technology:
  Python           — trading engine, intelligence layers
  Groq Llama 3.3   — reasoning brain (free, open)
  NSE public APIs  — real-time Nifty 50 data (free)
  Zerodha Kite     — order execution
  GitHub Actions   — autonomous compute (free)
  scikit-learn     — ML ensemble (open-source)

Tej trades Nifty 50 stocks intraday — short selling only.
Every day he scans all 50 stocks, reasons through setups,
executes live trades, reflects after every close, and evolves
his strategy weekly. He gets smarter every single day.

You can talk to Tej freely on Telegram — just message him.

Usage:
  python main.py               — Start Tej
  python main.py --healthcheck — Run diagnostics
  python main.py --brain-status — Tej's current intelligence state
  python main.py --chat "message" — Talk to Tej directly
  python main.py --evolve      — Run evolution cycle manually
"""

import sys
import argparse
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

os.makedirs("logs",    exist_ok=True)
os.makedirs("db",      exist_ok=True)
os.makedirs("reports", exist_ok=True)

from utils.logger import setup_logging
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("main")
IST = ZoneInfo("Asia/Kolkata")


def _check_env():
    from config import KITE_API_KEY, GROQ_API_KEY, TELEGRAM_BOT_TOKEN, TRADING
    errors   = []   # blocking — healthcheck fails
    warnings = []   # non-blocking — just inform

    if not KITE_API_KEY:        errors.append("KITE_API_KEY not set")
    if not GROQ_API_KEY:        errors.append("GROQ_API_KEY not set — Tej needs his reasoning engine")
    if not TELEGRAM_BOT_TOKEN:  warnings.append("TELEGRAM_BOT_TOKEN not set — can't send Telegram messages")

    # KITE_ACCESS_TOKEN is generated fresh each morning by kite_login.py
    # It won't exist at healthcheck time — that's normal and expected
    if not os.getenv("KITE_ACCESS_TOKEN"):
        warnings.append("KITE_ACCESS_TOKEN not set — will be generated at 8:45 AM IST by morning prep")

    try:
        TRADING.validate()
    except AssertionError as e:
        errors.append(f"Config violation: {e}")

    return errors, warnings


def run_healthcheck():
    errors, warnings = _check_env()
    from config import TRADING, PAPER_TRADE
    from brain.neural_core import brain
    from data.nifty50_universe import NIFTY50

    print("\n" + "═" * 52)
    print("  Tej — Autonomous AI Trading Agent")
    print("  Nifty 50 Intraday Short Selling")
    print("═" * 52)
    print(f"  Mode:          {'⚡ LIVE' if not PAPER_TRADE else '📄 PAPER'}")
    print(f"  Universe:      {len(NIFTY50)} Nifty 50 stocks | SHORT ONLY")
    print(f"  Capital:       ₹{TRADING.total_capital:,.0f}")
    print(f"  Intelligence:  {brain._state.intelligence_score:.0%}")
    print(f"  Decisions:     {brain._state.total_decisions} total")
    print(f"  Patterns:      {len(brain._state.discovered_patterns)} discovered")
    print(f"  Built on:      Groq Llama 3.3 70B (open-source)")
    print()
    for w in warnings:
        print(f"  ⚠  {w}")
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("  ✅ All critical checks passed — Tej is ready")
    print("═" * 52 + "\n")
    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Tej — Autonomous AI Trading Agent")
    parser.add_argument("--healthcheck",  action="store_true", help="Run diagnostics")
    parser.add_argument("--brain-status", action="store_true", help="Tej's intelligence state")
    parser.add_argument("--chat",  type=str, default=None,     help="Talk to Tej directly")
    parser.add_argument("--evolve",       action="store_true", help="Run evolution cycle")
    parser.add_argument("--backtest",     action="store_true", help="Run 90-day backtest")
    args = parser.parse_args()

    if args.healthcheck:
        ok = run_healthcheck()
        sys.exit(0 if ok else 1)

    if args.brain_status:
        from brain.neural_core import brain
        print(brain.get_brain_status())
        sys.exit(0)

    if args.chat:
        # Talk to Tej directly from the command line
        from brain.tej_persona import tej
        print(f"\nYou: {args.chat}\n")
        response = tej.respond(args.chat)
        print(f"Tej: {response}\n")
        sys.exit(0)

    if args.evolve:
        from brain.evolution_engine import evolution_engine
        result = evolution_engine.run_evolution_cycle()
        print(f"Evolution: {result.get('summary', 'complete')}")
        sys.exit(0)

    if args.backtest:
        from backtest.backtester import Backtester
        from config import TRADING
        backtester = Backtester()
        for sym in TRADING.priority_watchlist[:5]:
            try:
                result = backtester.run(sym, days=90)
                print(f"{sym}: WR={result.get('win_rate',0):.0%} PF={result.get('profit_factor',0):.2f}")
            except Exception as e:
                print(f"{sym}: error — {e}")
        sys.exit(0)

    # ── Start Tej ──────────────────────────────────────────────────
    errors, warnings = _check_env()
    for w in warnings:
        logger.warning(f"Config: {w}")
    for e in errors:
        logger.error(f"Config: {e}")

    print("\n" + "═" * 52)
    print("  Starting Tej — Autonomous AI Trading Agent")
    print("  Nifty 50 Short Selling | Built on Open Source")
    print("  Mission: First billionaire in the Thimmaiah family")
    print("═" * 52 + "\n")

    from brain.orchestrator import BrainOrchestrator
    orchestrator = BrainOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
