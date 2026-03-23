"""
Orchestrator patch: wire sentiment_agent and position_monitor into the existing orchestrator.
This file shows the additions needed in agents/orchestrator.py.
Apply by updating the orchestrator's __init__ and relevant methods.
"""

# ── ADD to OrchestratorAgent.__init__ ─────────────────────────────────────────

PATCH_INIT = """
from agents.sentiment_agent import SentimentAgent
from agents.position_monitor import PositionMonitorAgent
from reports.eod_reporter import EODReporter

# In __init__, after self.executor = ...:
self.sentiment  = SentimentAgent(healer=self.healer)
self.monitor    = PositionMonitorAgent(
    risk_mgr=self.risk_mgr,
    executor=self.executor,
    notify_fn=self._notify,
)
self.reporter   = EODReporter(risk_mgr=self.risk_mgr)
"""

# ── ADD to _morning_scan, after candidates are confirmed ──────────────────────

PATCH_SCAN_SENTIMENT = """
# After self.today_candidates = candidates:

# Enrich candidates with sentiment scores
if candidates:
    symbols = [c.symbol for c in candidates]
    sentiments = self.sentiment.analyse_multiple(symbols)
    for candidate in candidates:
        sym_sentiment = sentiments.get(candidate.symbol)
        if sym_sentiment:
            candidate.news_sentiment = sym_sentiment.label
            # Adjust score downward if news is strongly bullish (bad for shorts)
            if sym_sentiment.label in ("BULLISH", "STRONG_BULLISH"):
                candidate.score *= 0.7
                logger.info(
                    f"Score reduced for {candidate.symbol}: "
                    f"bullish news sentiment ({sym_sentiment.label})"
                )
            elif sym_sentiment.label in ("BEARISH", "STRONG_BEARISH"):
                candidate.score = min(1.0, candidate.score * 1.1)
    # Re-sort after sentiment adjustment
    candidates.sort(key=lambda x: x.score, reverse=True)
"""

# ── ADD to _execute_candidate, after result["status"] == "EXECUTED" ───────────

PATCH_EXECUTE_MONITOR = """
# After self.active_trades[symbol] = {...}:
self.monitor.register_position(
    symbol=symbol,
    entry_price=candidate.entry_price,
    stop_loss=risk_decision.adjusted_stop_loss,
    target=candidate.target,
    quantity=risk_decision.adjusted_quantity,
    direction="SHORT",
)
"""

# ── REPLACE _monitor_positions with monitor-agent call ────────────────────────

PATCH_MONITOR = """
def _monitor_positions(self):
    if not self.active_trades or not self.market_session_active:
        return
    closed = self.monitor.check_all()
    for symbol in closed:
        self.active_trades.pop(symbol, None)
"""

# ── REPLACE _square_off_all ───────────────────────────────────────────────────

PATCH_SQUAREOFF = """
def _square_off_all(self):
    logger.info("=== SQUARE OFF ALL ===")
    self.market_session_active = False
    results = self.monitor.force_square_off_all()
    self.active_trades.clear()
    daily_pnl = self.risk_mgr.get_today_pnl()
    logger.info(f"Square off complete. Day P&L: ₹{daily_pnl:.2f}")
    self._notify(f"🔔 3:10 PM square-off done | Day P&L: ₹{daily_pnl:.0f}")
"""

# ── REPLACE _end_of_day_report ────────────────────────────────────────────────

PATCH_EOD = """
def _end_of_day_report(self):
    summary = self.risk_mgr.get_daily_summary()
    report_path = self.reporter.generate()   # generates HTML + pushes to GitHub
    report_msg = self._format_daily_report(summary)
    logger.info(report_msg)
    self._notify(report_msg)
    self.today_candidates = []
    logger.info(f"EOD report saved: {report_path}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# These are applied as targeted str_replace patches to orchestrator.py.
# The orchestrator_patch.py file itself is only for documentation.
# See the updated orchestrator below for the fully integrated version.
# ─────────────────────────────────────────────────────────────────────────────
print("This file documents the orchestrator patches. See orchestrator.py for the full implementation.")
