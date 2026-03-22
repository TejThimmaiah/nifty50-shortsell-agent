"""
Tej — Autonomous AI Trading Agent
Named after his owner: Tej Thimmaiah.
Mission: First billionaire in the Thimmaiah family.
Built on: Python + Groq Llama 3.3 70B + NSE APIs + Zerodha Kite (all open-source/free)

MANDATE:
  This agent has ONE job: short sell individual Nifty 50 stocks intraday.
  - ONLY Nifty 50 constituents (50 stocks, India's benchmark index)
  - ONLY short selling (no long positions, ever)
  - ONLY intraday MIS (all positions squared off by 3:10 PM IST)
  - ONLY NSE (National Stock Exchange of India)

COST: Rs 0/month (Kite Connect Personal 2026 — order API free)
"""

import os
from dataclasses import dataclass, field
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# BROKER: Zerodha Kite Connect Personal (FREE in 2026)
# ─────────────────────────────────────────────────────────────────────────────
KITE_API_KEY      = os.getenv("KITE_API_KEY",      "")
KITE_API_SECRET   = os.getenv("KITE_API_SECRET",   "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# ─────────────────────────────────────────────────────────────────────────────
# FREE LLM
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL      = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

# ─────────────────────────────────────────────────────────────────────────────
# CLOUDFLARE
# ─────────────────────────────────────────────────────────────────────────────
CLOUDFLARE_WEBHOOK_URL    = os.getenv("CLOUDFLARE_WEBHOOK_URL",    "")
CLOUDFLARE_WEBHOOK_SECRET = os.getenv("CLOUDFLARE_WEBHOOK_SECRET", "")

# ─────────────────────────────────────────────────────────────────────────────
# GCP / GITHUB
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO  = os.getenv("GITHUB_REPO",  "")

# ─────────────────────────────────────────────────────────────────────────────
# MODE
# ─────────────────────────────────────────────────────────────────────────────
PAPER_TRADE = os.getenv("PAPER_TRADE", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# TRADING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TradingConfig:
    # MANDATE — never change
    direction:    str = "SHORT_ONLY"
    universe:     str = "NIFTY50"
    exchange:     str = "NSE"
    order_type:   str = "MIS"

    # Capital & risk
    total_capital:          float = 100_000.0
    max_risk_per_trade_pct: float = 2.0
    max_open_positions:     int   = 3
    max_daily_loss_pct:     float = 5.0
    max_daily_loss_abs:     float = 5_000.0

    # Short execution
    stop_loss_pct:          float = 0.5
    target_pct:             float = 1.5
    trailing_sl_pct:        float = 0.3
    trailing_activate_pct:  float = 0.8
    min_risk_reward:        float = 2.0

    # Timing (IST)
    market_open:     str = "09:15"
    scan_start:      str = "09:20"
    scan_end:        str = "13:00"
    no_entry_after:  str = "13:00"
    square_off_warn: str = "15:00"
    square_off:      str = "15:10"
    market_close:    str = "15:30"

    # Entry filters
    rsi_overbought:       float = 70.0
    rsi_period:           int   = 14
    min_confidence:       float = 0.50
    volume_multiplier:    float = 1.5
    min_price:            float = 100.0
    max_price:            float = 10_000.0
    require_fo_eligible:  bool  = True

    # All 50 Nifty 50 constituents — sorted by short-selling priority
    priority_watchlist: List[str] = field(default_factory=lambda: [
        # Tier 1: Highest ADV, best short-selling behavior
        "RELIANCE",   "HDFCBANK",   "ICICIBANK",  "TCS",        "INFY",
        "SBIN",       "AXISBANK",   "BAJFINANCE", "KOTAKBANK",  "HCLTECH",
        # Tier 2
        "TATAMOTORS", "MARUTI",     "SUNPHARMA",  "DRREDDY",    "ITC",
        "WIPRO",      "LTIM",       "BHARTIARTL", "ONGC",       "NTPC",
        # Tier 3
        "HINDUNILVR", "TATASTEEL",  "JSWSTEEL",   "HINDALCO",   "COALINDIA",
        "LARSEN",     "ADANIENT",   "TITAN",      "BAJAJ-AUTO", "EICHERMOT",
        # Tier 4
        "HEROMOTOCO", "M&M",        "CIPLA",      "DIVISLAB",   "APOLLOHOSP",
        "ADANIPORTS", "ULTRACEMCO", "GRASIM",     "POWERGRID",  "NESTLEIND",
        # Tier 5
        "BRITANNIA",  "GODREJCP",   "INDUSINDBK", "BAJAJFINSV", "TRENT",
        "TECHM",      "ASIANPAINT", "SHRIRAMFIN", "BPCL",       "JSWENERGY",
    ])

    # Circuit breaker — SHORT-SELLING SPECIFIC
    max_consecutive_losses: int   = 3      # pause after 3 wrong-direction trades
    # NOTE: Nifty FALLING is NOT a halt condition — it's our primary opportunity
    # We only halt when:
    extreme_crash_threshold: float = 4.0  # Nifty down >4% → lower circuit risk on stocks
    vix_spike_threshold:     float = 30.0 # VIX > 30 → spreads too wide
    single_position_loss:    float = 2.0  # one position loses >2% capital → reassess

    # Kelly sizing
    kelly_fraction:          float = 0.5
    max_single_position_pct: float = 30.0

    def is_nifty50(self, symbol: str) -> bool:
        return symbol in self.priority_watchlist

    def validate(self):
        assert self.direction  == "SHORT_ONLY", "Only short selling permitted"
        assert self.universe   == "NIFTY50",    "Only Nifty 50 stocks permitted"
        assert self.order_type == "MIS",         "Only MIS intraday orders permitted"
        assert len(self.priority_watchlist) == 50


TRADING = TradingConfig()

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH    = os.path.join(os.path.dirname(__file__), "db",     "trades.db")
LOG_DIR    = os.path.join(os.path.dirname(__file__), "logs")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# ─────────────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────────────
MAX_SEARCH_ATTEMPTS = 3
SEARCH_COOLDOWN_SEC = 30
AGENT_TIMEOUT_SEC   = 60
DATA_FETCH_TIMEOUT  = 30
ORDER_TIMEOUT_SEC   = 10
