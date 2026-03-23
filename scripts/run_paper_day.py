"""
Paper Day Simulator
Simulates a full trading day using historical data.
Useful for validating the agent logic without waiting for market hours.

Usage:
  python scripts/run_paper_day.py --date 2025-03-10
  python scripts/run_paper_day.py --fast  (uses today's data)
  python scripts/run_paper_day.py --symbol RELIANCE --date 2025-03-10
"""

import sys
import os
import argparse
import logging
import time
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import setup_logging
setup_logging(level="INFO")
logger = logging.getLogger("paper_day")

os.environ["PAPER_TRADE"] = "true"


def run_simulated_day(target_date: str, symbols: list = None, fast: bool = False):
    """
    Replay a trading day using historical data.

    In fast mode: uses recent market data and runs analysis immediately.
    In date mode: uses historical data for the specified date.
    """
    from data.nse_fetcher import get_historical_ohlcv, get_fo_stocks
    from agents.technical_analyst import calculate_all
    from agents.risk_manager import RiskManagerAgent
    from agents.trade_executor import TradeExecutorAgent
    from utils.circuit_breaker import CircuitBreaker
    from utils.liquidity_filter import LiquidityFilter
    from strategies.gap_strategy import GapUpShortStrategy
    from strategies.candlestick_patterns import detect_all_bearish_patterns
    from config import TRADING
    import tempfile
    import config

    # Isolated DB for simulation
    tmp_db = tempfile.mktemp(suffix="_papersim.db")
    config.DB_PATH = tmp_db

    print(f"\n{'═'*60}")
    print(f"  Tej Paper Day Simulator")
    print(f"  Target date : {target_date}")
    print(f"  Mode        : {'Fast (live data)' if fast else 'Historical replay'}")
    print(f"  Capital     : ₹{TRADING.total_capital:,.0f}")
    print(f"{'═'*60}\n")

    risk_mgr  = RiskManagerAgent(capital=TRADING.total_capital)
    executor  = TradeExecutorAgent()
    cb        = CircuitBreaker(
        capital=TRADING.total_capital,
        notify_fn=lambda m: print(f"  🚨 CB: {m}"),
    )
    liq       = LiquidityFilter()
    gap_strat = GapUpShortStrategy()

    # ── Step 1: Build candidate list ─────────────────────────────
    scan_symbols = symbols or TRADING.priority_watchlist[:15]
    print(f"[09:20] Scanning {len(scan_symbols)} symbols...\n")

    candidates = []
    historical_data = {}

    for sym in scan_symbols:
        df = get_historical_ohlcv(sym, days=70)
        if df is None or len(df) < 40:
            continue
        historical_data[sym] = df

        signal = calculate_all(df, sym)
        if signal and signal.signal in ("SHORT", "STRONG_SHORT"):
            # Liquidity check
            liq_result = liq.check(sym)
            if not liq_result.passed:
                print(f"  ❌ {sym}: Liquidity rejected — {liq_result.reason}")
                continue

            # Candlestick pattern
            patterns = detect_all_bearish_patterns(df)
            pat_name = patterns[0].name if patterns else "—"

            candidates.append({
                "symbol":     sym,
                "signal":     signal.signal,
                "confidence": signal.confidence,
                "rsi":        signal.rsi,
                "entry":      signal.entry_price,
                "sl":         signal.stop_loss,
                "target":     signal.target,
                "pattern":    pat_name,
                "reason":     signal.reason[:60],
            })
            print(
                f"  ✓ {sym:12s} | {signal.signal:13s} | conf={signal.confidence:.2f} "
                f"| RSI={signal.rsi:.0f} | pattern={pat_name}"
            )

    # ── Step 2: Gap setups ────────────────────────────────────────
    print(f"\n[09:20] Checking gap-up setups...")
    gap_candidates = []
    for sym, df in historical_data.items():
        if len(df) < 2:
            continue
        prev_close  = float(df["close"].iloc[-2])
        today_open  = float(df["open"].iloc[-1])
        today_vol   = float(df["volume"].iloc[-1])
        gap_pct     = (today_open - prev_close) / prev_close * 100
        if gap_pct >= 2.0:
            quote_sim = {
                "symbol":     sym,
                "open":       today_open,
                "prev_close": prev_close,
                "ltp":        today_open,
                "volume":     today_vol,
            }
            gap_setups = gap_strat.scan_for_gaps([quote_sim], historical_data)
            if gap_setups:
                g = gap_setups[0]
                gap_candidates.append(g)
                print(f"  🔺 {sym}: gap +{g.gap_pct:.1f}% | R:R={g.risk_reward:.2f} | conf={g.confidence:.2f}")

    # ── Step 3: Execute top candidates ───────────────────────────
    all_candidates = sorted(
        candidates, key=lambda c: c["confidence"], reverse=True
    )[:TRADING.max_open_positions]

    print(f"\n[09:25] Executing {len(all_candidates)} trades...\n")

    executed = []
    for c in all_candidates:
        allowed, reason = cb.allow_trade()
        if not allowed:
            print(f"  ⛔ {c['symbol']}: Circuit breaker — {reason}")
            break

        decision = risk_mgr.approve_trade(
            c["symbol"], c["entry"], c["sl"], 100, "SHORT"
        )
        if not decision.approved:
            print(f"  ❌ {c['symbol']}: Risk rejected — {decision.reason}")
            continue

        result = executor.short_sell(
            c["symbol"], decision.adjusted_quantity,
            c["entry"], c["sl"], c["target"]
        )
        risk_mgr.record_trade(c["symbol"], "SHORT", c["entry"],
                               decision.adjusted_quantity, "SIM_ORD")

        print(
            f"  📉 SHORT {c['symbol']:12s} | "
            f"Entry ₹{c['entry']:>8.2f} | "
            f"SL ₹{c['sl']:>8.2f} | "
            f"Target ₹{c['target']:>8.2f} | "
            f"Qty {decision.adjusted_quantity:>4} | "
            f"Risk ₹{decision.max_loss_this_trade:>6.0f}"
        )
        executed.append(c)

        if not fast:
            time.sleep(0.2)

    # ── Step 4: Simulate outcomes ─────────────────────────────────
    print(f"\n[15:10] Simulating trade outcomes...\n")
    total_pnl = 0

    for c in executed:
        sym   = c["symbol"]
        df    = historical_data.get(sym)
        if df is None:
            continue

        entry  = c["entry"]
        sl     = c["sl"]
        target = c["target"]

        # Use next day's OHLC to simulate intraday outcome
        day_high = float(df["high"].iloc[-1])
        day_low  = float(df["low"].iloc[-1])

        if day_high >= sl:
            exit_price = sl
            exit_reason = "STOP_LOSS"
        elif day_low <= target:
            exit_price = target
            exit_reason = "TARGET"
        else:
            exit_price = float(df["close"].iloc[-1])
            exit_reason = "EOD_SQUAREOFF"

        pnl  = (entry - exit_price) * decision.adjusted_quantity
        pnl -= 20   # commission
        total_pnl += pnl
        cb.record_trade_result(pnl)

        risk_mgr.close_trade(sym, exit_price)

        emoji = "✅" if pnl > 0 else "🔴"
        print(
            f"  {emoji} {sym:12s} → {exit_reason:14s} "
            f"@ ₹{exit_price:>8.2f} | "
            f"P&L {'+'if pnl>=0 else ''}₹{pnl:>7.0f}"
        )

    # ── Step 5: Summary ───────────────────────────────────────────
    summary = risk_mgr.get_daily_summary()
    print(f"\n{'─'*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Total P&L   : {'+'if total_pnl>=0 else ''}₹{total_pnl:,.0f}")
    print(f"  Win rate    : {summary['win_rate']:.0f}%")
    print(f"  Trades      : {summary['win_count']}W / {summary['loss_count']}L")
    print(f"  CB Status   : {'TRIGGERED' if cb.state.triggered else 'OK'}")
    print(f"{'═'*60}\n")

    # Cleanup
    os.unlink(tmp_db)
    return total_pnl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tej Paper Day Simulator")
    parser.add_argument("--date",   default=date.today().isoformat(), help="Target date YYYY-MM-DD")
    parser.add_argument("--symbol", nargs="*", help="Specific symbols to test")
    parser.add_argument("--fast",   action="store_true", help="Skip delays")
    args = parser.parse_args()

    run_simulated_day(
        target_date=args.date,
        symbols=args.symbol,
        fast=args.fast,
    )
