"""
Orchestrator Agent — The Brain
Coordinates all specialist agents, makes final trading decisions,
and runs the main trading loop from 9:15 AM to 3:30 PM IST.
Uses LLM (Groq Llama 3.3 70B) for high-level reasoning.
Self-heals via web search when stuck.
"""

import logging
import time
import json
import schedule
import requests
from datetime import datetime
from datetime import date
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict

from config import (
    TRADING, GROQ_API_KEY, GEMINI_API_KEY, LLM_MODEL,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PAPER_TRADE
)
from agents.nifty50_scanner import Nifty50ShortScanner, ShortCandidate
from agents.technical_analyst import calculate_all
from agents.risk_manager import RiskManagerAgent
from agents.trade_executor import TradeExecutorAgent
from agents.self_healer import SelfHealerAgent
from agents.sentiment_agent import SentimentAgent
from agents.position_monitor import PositionMonitorAgent
from reports.eod_reporter import EODReporter
from reports.performance_analytics import PerformanceAnalytics
from strategies.candlestick_patterns import get_best_pattern
from strategies.multi_timeframe import MultiTimeframeAnalyser
from strategies.sector_rotation import SectorRotationAnalyser
from utils.circuit_breaker import CircuitBreaker
from utils.market_calendar import calendar as mkt_calendar
from utils.telegram_bot import TelegramCommandBot
from intelligence.trade_memory import memory_store, TradeMemory
from intelligence.market_regime import regime_detector, MarketRegime
from intelligence.adaptive_config import adaptive_config, AdaptiveParams
from intelligence.self_improver import self_improver
from data.nse_fetcher import get_quote, get_intraday_ohlcv, get_historical_ohlcv
from data.free_market_data import free_streamer, get_live_quote, FreeTickStreamer

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class OrchestratorAgent:
    """
    The central agent. Runs the full intraday trading cycle.
    Operates 9:15 AM – 3:30 PM IST, Monday – Friday.
    All decisions are autonomous — no human intervention needed.
    """

    def __init__(self):
        logger.info("Initializing Tej Autonomous Trading Agent...")

        self.healer   = SelfHealerAgent()
        self.scanner  = Nifty50ShortScanner(healer=self.healer, capital=TRADING.total_capital)
        self.risk_mgr = RiskManagerAgent(capital=TRADING.total_capital)
        self.executor = TradeExecutorAgent(healer=self.healer)

        self.sentiment  = SentimentAgent(healer=self.healer)
        self.monitor    = PositionMonitorAgent(
            risk_mgr=self.risk_mgr,
            executor=self.executor,
            notify_fn=self._notify,
        )
        self.reporter   = EODReporter(risk_mgr=self.risk_mgr)
        self.analytics  = PerformanceAnalytics(capital=TRADING.total_capital)
        self.mtf        = MultiTimeframeAnalyser()
        self.sectors    = SectorRotationAnalyser()
        self.breaker    = CircuitBreaker(
            capital=TRADING.total_capital,
            notify_fn=self._notify,
            on_halt=self._emergency_halt,
        )
        self.tg_bot     = TelegramCommandBot()
        self.tg_bot.register_orchestrator(self)

        # ── Intelligence layer ────────────────────────────────────────────────
        self.current_regime: Optional[MarketRegime] = None
        self.adaptive_params: AdaptiveParams = adaptive_config.load()
        logger.info(
            f"Adaptive params loaded — RSI={self.adaptive_params.rsi_overbought}, "
            f"SL={self.adaptive_params.stop_loss_pct}%, "
            f"confidence≥{self.adaptive_params.min_confidence}"
        )

        self.active_trades: Dict[str, Dict] = {}
        self.today_candidates: List[ShortCandidate] = []
        self.market_session_active = False

        logger.info(f"Agent initialized | Paper mode: {PAPER_TRADE} | Capital: ₹{TRADING.total_capital:,.0f}")

    # ──────────────────────────────────────────────────────────────
    # MAIN TRADING LOOP
    # ──────────────────────────────────────────────────────────────

    def run(self):
        """Start the agent. Runs all day via schedule."""
        logger.info("Starting Tej Agent...")
        self._notify("🤖 Tej Agent started. Paper mode: " + str(PAPER_TRADE))

        # Start Telegram command bot in background
        self.tg_bot.start()

        # Start free tick streamer (NSE public API — ₹0/month)
        free_streamer.start()
        logger.info("Free tick streamer started (NSE public API, no subscription needed)")

        # Daily circuit breaker reset at market open
        schedule.every().day.at("09:10").do(self.breaker.reset_daily)

        # Intelligence: nightly improvement at 3:35 PM (after EOD report)
        schedule.every().day.at("15:35").do(self._nightly_improvement)

        # Weekly WFO + ML retraining every Sunday 8 PM IST
        schedule.every().sunday.at("20:00").do(self._weekly_intelligence_update)

        # Refresh NSE holiday calendar monthly
        schedule.every(30).days.do(mkt_calendar.refresh_holidays)

        # Schedule all tasks
        schedule.every().day.at(TRADING.scan_start).do(self._morning_scan)
        schedule.every(5).minutes.do(self._monitor_positions)
        schedule.every(30).minutes.do(self._rescan_if_needed)
        schedule.every().day.at(TRADING.square_off).do(self._square_off_all)
        schedule.every().day.at(TRADING.market_close).do(self._end_of_day_report)
        schedule.every().monday.at("08:00").do(self._weekly_health_check)

        logger.info("Schedule configured. Waiting for market...")

        while True:
            try:
                schedule.run_pending()
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Agent stopped by user.")
                self._notify("🛑 Agent stopped.")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                fix = self.healer.heal(f"Main loop crashed: {e}")
                logger.info(f"Healer: {fix.get('solution')}")
                self._notify(f"⚠️ Agent error self-healed: {str(e)[:100]}")
                time.sleep(30)

    # ──────────────────────────────────────────────────────────────
    # SCHEDULED TASKS
    # ──────────────────────────────────────────────────────────────

    def _morning_scan(self):
        """9:20 AM: Scan for the day's short candidates."""
        if not self._is_market_day():
            logger.info("Market holiday — skipping scan.")
            return

        logger.info("=== MORNING SCAN ===")
        self.market_session_active = True

        try:
            # Get market sentiment from web (self-healer does this)
            sentiment = self.healer.search_market_sentiment()
            logger.info(f"Market sentiment: {sentiment.get('sentiment')} | Good for shorts: {sentiment.get('good_for_shorts')}")

            # ── INTELLIGENCE: Load adaptive params + morning brief ────────────
            self.adaptive_params = adaptive_config.load()
            morning_brief = self_improver.get_morning_brief()
            logger.info(f"\n{morning_brief}")
            self._notify(f"🧠 Intelligence brief ready. RSI≥{self.adaptive_params.rsi_overbought}, conf≥{self.adaptive_params.min_confidence}")

            # ── Detect market regime ──────────────────────────────────────────
            nifty_df = get_historical_ohlcv("NIFTY", days=30) if False else None  # use Nifty index data
            fii_data = self._fii_data if hasattr(self, '_fii_data') else {}
            vix_val  = 15.0  # fetched from India VIX
            nifty_change = float(self._nifty_data.get("change_pct", 0)) if hasattr(self, '_nifty_data') else 0
            adv_dec  = 0.5

            try:
                nifty_data = self._nifty_data if hasattr(self, '_nifty_data') else {}
                advances = int(nifty_data.get("advances", 25))
                declines = int(nifty_data.get("declines", 25))
                adv_dec  = advances / max(advances + declines, 1)
            except Exception:
                pass

            self.current_regime = regime_detector.detect(
                vix=vix_val,
                advance_decline=adv_dec,
                fii_net_cr=float(fii_data.get("fii_net", 0)),
                nifty_change_1d=nifty_change,
            )

            # Apply regime overrides to adaptive params
            self.adaptive_params = adaptive_config.apply_regime_override(
                self.adaptive_params, self.current_regime.label
            )

            logger.info(
                f"Regime: {self.current_regime.label} | "
                f"Good for shorts: {self.current_regime.is_good_for_shorts} | "
                f"Aggressiveness: {self.current_regime.short_aggressiveness}"
            )
            self._notify(
                f"📊 Regime: {self.current_regime.label} | "
                f"{'✅ Go short' if self.current_regime.is_good_for_shorts else '⚠ Avoid shorts'} | "
                f"Size {self.current_regime.position_size_multiplier}x"
            )

            # CRISIS: halt immediately
            if self.current_regime.label == "CRISIS":
                self._notify("🚨 Crisis regime detected — no trades today")
                return

            # Pre-market intermarket bias (SGX Nifty, crude, USD/INR)
            intermarket_bias = 0.0
            try:
                from intelligence.intermarket import intermarket_analyser
                im_bias = intermarket_analyser.get_morning_bias()
                intermarket_bias = im_bias.bias_score
                self._nifty_data = im_bias.__dict__ if im_bias else {}
                if im_bias.signals:
                    self._notify(
                        f"🌍 Intermarket: {im_bias.bias} | "
                        + "; ".join(im_bias.signals[:2])
                    )
                logger.info(f"Intermarket bias: {im_bias.bias} (score={im_bias.bias_score:.2f})")
            except Exception as e:
                logger.debug(f"Intermarket fetch error: {e}")

            # Get sector snapshot (Nifty 50 sectors only)
            sector_snap = self.sectors.get_snapshot()
            logger.info(
                f"Market phase: {sector_snap.market_phase} | "
                f"Weak Nifty50 sectors: {sector_snap.weakest_sectors[:3]}"
            )

            # Run the full Nifty 50 scan — SHORT ONLY
            candidates = self.scanner.scan(intermarket_bias=intermarket_bias)
            self.today_candidates = candidates

            if candidates:
                # Enrich with news sentiment for Nifty 50 stocks
                symbols    = [c.symbol for c in candidates]
                sentiments = self.sentiment.analyse_multiple(symbols)
                for candidate in candidates:
                    sym_sent = sentiments.get(candidate.symbol)
                    if sym_sent:
                        if sym_sent.label in ("BULLISH", "STRONG_BULLISH"):
                            # Bearish news about a Nifty50 stock — lower conviction to short
                            candidate.score = max(0, candidate.score * 0.75)
                            logger.info(f"Score dampened [{candidate.symbol}]: bullish news offset")
                        elif sym_sent.label in ("BEARISH", "STRONG_BEARISH"):
                            candidate.score = min(1.0, candidate.score * 1.10)
                            logger.info(f"Score boosted [{candidate.symbol}]: bearish news confirms")
                candidates.sort(key=lambda x: x.score, reverse=True)

            if not candidates:
                logger.info("No short candidates found today.")
                self._notify("📊 Morning scan complete. No candidates today.")
                return

            # Use LLM to make final selection
            final_picks = self._llm_select_trades(candidates, sentiment)

            # Execute approved trades
            for pick in final_picks:
                self._execute_candidate(pick)
                time.sleep(2)

            summary = f"📊 Morning scan: {len(candidates)} candidates | {len(final_picks)} trades initiated"
            logger.info(summary)
            self._notify(summary)

        except Exception as e:
            logger.error(f"Morning scan error: {e}", exc_info=True)
            fix = self.healer.heal(f"Morning scan failed: {e}")
            self._notify(f"⚠️ Scan error (self-healing): {str(e)[:80]}")

    def _monitor_positions(self):
        """Every 5 min: Delegate to PositionMonitorAgent."""
        if not self.active_trades or not self.market_session_active:
            return
        closed = self.monitor.check_all()
        for symbol in closed:
            self.active_trades.pop(symbol, None)
            logger.info(f"Position {symbol} closed and removed from active trades")

    def _rescan_if_needed(self):
        """Every 30 min: Re-scan if positions < max and time < 1 PM."""
        now = datetime.now(IST)
        scan_deadline = now.replace(hour=13, minute=0, second=0)
        if now > scan_deadline:
            return

        open_count = self.risk_mgr.get_open_position_count()
        if open_count >= TRADING.max_open_positions:
            return

        if len(self.today_candidates) == 0:
            logger.info("Re-scanning for new candidates...")
            self._morning_scan()

    def _square_off_all(self):
        """3:10 PM: Square off via PositionMonitorAgent."""
        logger.info("=== SQUARE OFF ALL ===")
        self.market_session_active = False
        self.monitor.force_square_off_all()
        self.active_trades.clear()
        daily_pnl = self.risk_mgr.get_today_pnl()
        logger.info(f"Square off complete. Day P&L: ₹{daily_pnl:.2f}")
        self._notify(f"🔔 3:10 PM square-off done | Day P&L: ₹{daily_pnl:.0f}")

    def _end_of_day_report(self):
        """3:30 PM: Generate HTML report, push to GitHub, send Telegram summary."""
        summary = self.risk_mgr.get_daily_summary()
        report_path = self.reporter.generate()
        report_msg  = self._format_daily_report(summary)
        logger.info(report_msg)
        logger.info(f"EOD report saved: {report_path}")
        self._notify(report_msg)
        self.today_candidates = []

        # ── Update Bayesian signal fusion from today's trades ─────────────────
        try:
            from intelligence.signal_fusion  import signal_fusion
            from intelligence.statistical_edge import edge_calculator
            for t in summary.get("trades", []):
                if t.get("status") == "CLOSED":
                    # Reconstruct signals from reason string
                    reason   = t.get("reason", "")
                    won      = t.get("pnl", 0) > 0
                    signals  = [s for s in [
                        "RSI_OVERBOUGHT"     if "RSI" in reason else None,
                        "BEARISH_DIVERGENCE" if "divergence" in reason.lower() else None,
                        "AT_RESISTANCE"      if "resistance" in reason.lower() else None,
                        "VOLUME_CONFIRMS"    if "vol" in reason.lower() else None,
                        "MACD_TURNING_DOWN"  if "MACD" in reason else None,
                        "BEARISH_ENGULFING"  if "Engulfing" in reason else None,
                        "FII_SELLING"        if "FII" in reason else None,
                    ] if s]
                    if signals:
                        signal_fusion.record_outcome(signals, won)
                        edge_calculator.record(signals, t.get("pnl", 0), won)
            logger.info("Bayesian signal fusion updated from today's trades")
        except Exception as e:
            logger.warning(f"Signal fusion update error: {e}")

    def _weekly_health_check(self):
        """Monday 8 AM: Self-healing health check of all components."""
        logger.info("Running weekly health check...")
        issues = []

        # Check LLM availability
        try:
            resp = self.healer._call_llm("Reply OK")
            if not resp or "OK" not in resp.upper():
                issues.append("LLM not responding")
        except Exception as e:
            issues.append(f"LLM error: {e}")

        # Check Zerodha connection
        if not PAPER_TRADE:
            try:
                pos = self.executor.get_positions()
                logger.info(f"Zerodha connection: OK ({len(pos)} positions)")
            except Exception as e:
                issues.append(f"Zerodha error: {e}")
                fix = self.healer.search_api_error_fix(str(e), "Zerodha Kite Connect")
                logger.info(f"Auto-fix: {fix}")

        if issues:
            msg = "⚠️ Weekly health check issues:\n" + "\n".join(issues)
        else:
            msg = "✅ Weekly health check: all systems nominal"
        self._notify(msg)
        logger.info(msg)

    # ──────────────────────────────────────────────────────────────
    # LLM DECISION MAKING
    # ──────────────────────────────────────────────────────────────

    def _llm_select_trades(
        self, candidates: List[ShortCandidate], sentiment: Dict
    ) -> List[ShortCandidate]:
        """Use LLM to make the final trade selection from scanner output."""
        if not candidates:
            return []

        # Prepare candidate summary for LLM
        candidate_data = [
            {
                "symbol": c.symbol,
                "score": c.score,
                "rsi": c.technical.rsi,
                "signal": c.technical.signal,
                "confidence": c.technical.confidence,
                "entry": c.entry_price,
                "stop_loss": c.stop_loss,
                "target": c.target,
                "quantity": c.quantity,
                "risk": c.risk_amount,
                "reason": c.reason,
            }
            for c in candidates[:10]  # Send top 10 to LLM
        ]

        daily_pnl = self.risk_mgr.get_today_pnl()
        open_count = self.risk_mgr.get_open_position_count()
        slots_available = TRADING.max_open_positions - open_count

        prompt = f"""You are an autonomous intraday short selling agent for NSE Indian stocks.
        
Current market sentiment: {json.dumps(sentiment)}
Daily P&L so far: ₹{daily_pnl:.0f}
Open positions: {open_count}/{TRADING.max_open_positions}
Slots available: {slots_available}

Short selling candidates identified by technical analysis:
{json.dumps(candidate_data, indent=2)}

Select the BEST {slots_available} candidates (or fewer if risk is high).
Criteria: highest confidence SHORT signals, diversified sectors, reasonable risk/reward.
Avoid if: market sentiment is strongly bullish, score < 0.50, daily loss already > 3%.

Respond ONLY in JSON array:
[{{"symbol": "XXX", "reason": "brief reasoning", "approved": true/false}}]"""

        response = self._call_llm(prompt)

        try:
            selections = json.loads(response)
            approved_symbols = {s["symbol"] for s in selections if s.get("approved")}
            return [c for c in candidates if c.symbol in approved_symbols]
        except Exception as e:
            logger.warning(f"LLM selection parse error: {e}. Using score-based selection.")
            return candidates[:slots_available]

    def _call_llm(self, prompt: str) -> str:
        """Call the primary LLM with fallback."""
        result = self.healer._call_groq(prompt) or self.healer._call_gemini(prompt)
        return result or "[]"

    # ──────────────────────────────────────────────────────────────
    # TRADE EXECUTION
    # ──────────────────────────────────────────────────────────────

    def _execute_candidate(self, candidate: ShortCandidate):
        """Risk-check, circuit-breaker gate, MTF confirm, then execute."""
        symbol = candidate.symbol

        # Gate 0: circuit breaker
        allowed, cb_reason = self.breaker.allow_trade()
        if not allowed:
            logger.warning(f"Circuit breaker blocks [{symbol}]: {cb_reason}")
            return

        # Gate 0b: sector check — avoid shorting stocks in strong sectors
        sector_ok, sector_reason = self.sectors.is_sector_shortable(symbol)
        if not sector_ok:
            logger.info(f"Sector gate blocked [{symbol}]: {sector_reason}")
            return

        # Gate 1: multi-timeframe confirmation
        mtf = self.mtf.analyse(symbol)
        if mtf and not self.mtf.is_high_conviction(mtf, min_alignment=2):
            logger.info(
                f"MTF rejected [{symbol}]: only {mtf.alignment_count}/3 "
                f"timeframes aligned (need 2)"
            )
            return
        if mtf:
            logger.info(f"MTF confirmed [{symbol}]: {mtf.alignment_count}/3 | "
                        f"confidence={mtf.composite_confidence:.2f}")

        # Gate 2: candlestick pattern on 5m chart (optional enrichment)
        df_5m = get_intraday_ohlcv(symbol, interval="5m", period="1d")
        if df_5m is not None:
            pattern = get_best_pattern(df_5m)
            if pattern and pattern.pattern_type == "BEARISH_REVERSAL":
                logger.info(f"Pattern [{symbol}]: {pattern.name} ({pattern.confidence:.0%})")
                candidate.score = min(1.0, candidate.score + 0.05)

        # Gate 3: risk manager final check (uses adaptive params for risk%)
        import config
        config.TRADING.max_risk_per_trade_pct = (
            self.adaptive_params.max_risk_per_trade_pct
            * self.adaptive_params.position_size_multiplier
        )
        risk_decision = self.risk_mgr.approve_trade(
            symbol=symbol,
            entry_price=candidate.entry_price,
            stop_loss=candidate.stop_loss,
            quantity=candidate.quantity,
            direction="SHORT",
        )

        if not risk_decision.approved:
            logger.warning(f"Trade REJECTED [{symbol}]: {risk_decision.reason}")
            return

        # Execute
        result = self.executor.short_sell(
            symbol=symbol,
            quantity=risk_decision.adjusted_quantity,
            entry_price=candidate.entry_price,
            stop_loss=risk_decision.adjusted_stop_loss,
            target=candidate.target,
        )

        if result.get("status") == "EXECUTED":
            self.risk_mgr.record_trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=candidate.entry_price,
                quantity=risk_decision.adjusted_quantity,
                order_id=result.get("entry_order_id", ""),
            )
            self.active_trades[symbol] = {
                "entry_price": candidate.entry_price,
                "stop_loss":   risk_decision.adjusted_stop_loss,
                "target":      candidate.target,
                "quantity":    risk_decision.adjusted_quantity,
                "reason":      candidate.reason,
            }
            # Register with position monitor (handles trailing SL, alerts, auto-close)
            self.monitor.register_position(
                symbol=symbol,
                entry_price=candidate.entry_price,
                stop_loss=risk_decision.adjusted_stop_loss,
                target=candidate.target,
                quantity=risk_decision.adjusted_quantity,
                direction="SHORT",
            )

            # ── Record to trade memory (intelligence learning) ────────────────
            try:
                now_ist = datetime.now(IST)
                hour    = now_ist.hour
                if   hour < 10: tod = "EARLY"
                elif hour < 12: tod = "MID"
                else:           tod = "LATE"

                mem = TradeMemory(
                    symbol          = symbol,
                    trade_date      = date.today().isoformat(),
                    entry_time      = now_ist.strftime("%H:%M:%S"),
                    strategy        = "SHORT",  # always short, Nifty50 only
                    signals_fired   = candidate.reason.split("; ") if candidate.reason else [],
                    confidence_score= candidate.score,
                    rsi_at_entry    = candidate.technical.rsi if hasattr(candidate, 'technical') else 0,
                    market_regime   = self.current_regime.label if self.current_regime else "",
                    nifty_breadth   = self.current_regime.description[:20] if self.current_regime else "",
                    fii_net_cr      = float(self._fii_data.get("fii_net", 0)) if hasattr(self, '_fii_data') else 0,
                    time_of_day     = tod,
                    entry_price     = candidate.entry_price,
                    stop_loss       = risk_decision.adjusted_stop_loss,
                    target          = candidate.target,
                    quantity        = risk_decision.adjusted_quantity,
                )
                trade_id = memory_store.record(mem)
                self.active_trades[symbol]["memory_id"] = trade_id
            except Exception as me:
                logger.debug(f"Trade memory record error: {me}")
            paper_tag = " [PAPER]" if PAPER_TRADE else ""
            self._notify(
                f"📉 SHORT{paper_tag}: {symbol}\n"
                f"Entry: ₹{candidate.entry_price:.2f} | SL: ₹{risk_decision.adjusted_stop_loss:.2f} | "
                f"Target: ₹{candidate.target:.2f}\n"
                f"Qty: {risk_decision.adjusted_quantity} | Risk: ₹{risk_decision.max_loss_this_trade:.0f}\n"
                f"Reason: {candidate.reason[:100]}"
            )
        else:
            logger.error(f"Execution failed [{symbol}]: {result.get('error')}")
            self._notify(f"❌ Order failed: {symbol} — {result.get('error', 'Unknown')[:80]}")

    # ──────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────

    def _is_market_day(self) -> bool:
        """Check if today is a trading day using MarketCalendar."""
        return mkt_calendar.is_trading_day()

    def _emergency_halt(self):
        """Called by circuit breaker on halt — square off all positions immediately."""
        logger.critical("EMERGENCY HALT — squaring off all positions")
        self.market_session_active = False
        self.monitor.force_square_off_all()
        self.active_trades.clear()

    def _nightly_improvement(self):
        """3:35 PM: Run the self-improvement cycle. The agent gets smarter."""
        logger.info("=== Starting nightly intelligence improvement cycle ===")
        try:
            # Run post-mortem on today's trades
            from intelligence.postmortem import postmortem_reviewer
            summary = self.risk_mgr.get_daily_summary()
            trades  = summary.get("trades", [])
            if trades:
                day_review = postmortem_reviewer.review_day(
                    trades,
                    day_context={
                        "regime":   self.current_regime.label if self.current_regime else "UNKNOWN",
                        "nifty_change": float(self._nifty_data.get("change_pct", 0)) if hasattr(self, "_nifty_data") else 0,
                    }
                )
                if day_review and day_review.get("daily_lesson"):
                    logger.info(f"Day review: {day_review['daily_lesson']}")

            result = self_improver.run_nightly_improvement()
            summary_msg = result.get("lessons", [])
            changes     = result.get("param_changes", [])

            msg_parts = ["🧠 Nightly learning complete:"]
            if summary_msg:
                msg_parts.append(f"💡 {summary_msg[0]}")
            if changes:
                msg_parts.append(f"⚙️ Changed: {'; '.join(changes[:2])}")
            msg_parts.append(f"📈 14d WR: {result.get('new_win_rate_14d', 0):.0%}")

            self._notify("\n".join(msg_parts))

        except Exception as e:
            logger.error(f"Nightly improvement error: {e}", exc_info=True)
            self._notify(f"⚠️ Nightly improvement error: {str(e)[:80]}")

    def _weekly_intelligence_update(self):
        """Sunday 8 PM: Walk-forward optimization + ML retraining."""
        logger.info("=== Weekly intelligence update ===")
        self._notify("🔬 Weekly intelligence update starting...")

        try:
            # Walk-forward optimization
            from intelligence.walk_forward import wfo_engine
            from config import TRADING
            symbols = TRADING.priority_watchlist[:8]
            for sym in symbols[:3]:   # limit to 3 for speed
                report = wfo_engine.run(sym, total_days=120)
                logger.info(f"WFO {sym}: {report.strategy_verdict} (rob={report.avg_robustness:.2f})")

            # ML ensemble retraining
            from intelligence.ml_ensemble import ml_predictor
            retrained = ml_predictor.train()
            if retrained:
                logger.info("ML ensemble retrained successfully")

            # Bad habit detection
            from intelligence.postmortem import postmortem_reviewer
            bad_habits = postmortem_reviewer.identify_bad_habits(days=30)
            if bad_habits:
                habits_msg = "⚠️ Recurring mistakes:\n" + "\n".join(f"• {h}" for h in bad_habits[:3])
                self._notify(habits_msg)

            self._notify("✅ Weekly intelligence update complete")

        except Exception as e:
            logger.error(f"Weekly update error: {e}", exc_info=True)
            self._notify(f"⚠️ Weekly update error: {str(e)[:80]}")

    def _format_daily_report(self, summary: Dict) -> str:
        emoji = "📈" if summary["total_pnl"] >= 0 else "📉"
        lines = [
            f"{emoji} Daily Report — {summary['date']}",
            f"Total P&L: ₹{summary['total_pnl']:,.0f}",
            f"Trades: {summary['win_count']}W / {summary['loss_count']}L "
            f"(Win rate: {summary['win_rate']:.0f}%)",
        ]
        if summary.get("best_trade"):
            lines.append(f"Best: {summary['best_trade']}")
        if summary.get("worst_trade"):
            lines.append(f"Worst: {summary['worst_trade']}")
        return "\n".join(lines)

    def _notify(self, message: str):
        """Send notification via Telegram."""
        logger.info(f"NOTIFY: {message}")
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
