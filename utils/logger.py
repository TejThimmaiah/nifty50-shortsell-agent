"""
Logging Utilities
Sets up structured, colour-coded logging for the trading agent.
Rotates logs daily. Separate files for trades, errors, and general output.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


class ColorFormatter(logging.Formatter):
    """Colour-coded console output."""
    COLORS = {
        logging.DEBUG:    "\033[37m",    # White
        logging.INFO:     "\033[32m",    # Green
        logging.WARNING:  "\033[33m",    # Yellow
        logging.ERROR:    "\033[31m",    # Red
        logging.CRITICAL: "\033[35m",    # Magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    TRADE_KEYWORDS = ("SHORT", "COVER", "TARGET", "STOP_LOSS", "EXECUTED", "CLOSED")

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")

        # Extra bold for trade events
        if any(kw in msg for kw in self.TRADE_KEYWORDS):
            return f"{self.BOLD}{color}{msg}{self.RESET}"
        return f"{color}{msg}{self.RESET}"


class TradeFilter(logging.Filter):
    """Only passes records that contain trade-related keywords."""
    KEYWORDS = ("SHORT", "COVER", "TARGET HIT", "STOP_LOSS", "EXECUTED",
                "CLOSED", "P&L", "P&l", "entry", "Entry", "NOTIFY")

    def filter(self, record: logging.LogRecord) -> bool:
        return any(kw in record.getMessage() for kw in self.KEYWORDS)


def setup_logging(
    level: str = "INFO",
    log_dir: str = None,
    agent_name: str = "nifty50-shortsell-agent",
) -> logging.Logger:
    """
    Configure all loggers for the trading agent.
    Creates three log files:
      - agent_YYYYMMDD.log     — Everything (INFO+)
      - trades_YYYYMMDD.log    — Trade events only
      - errors_YYYYMMDD.log    — Warnings and above

    Returns the root logger.
    """
    log_dir = log_dir or LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    today      = datetime.now().strftime("%Y%m%d")
    log_level  = getattr(logging, level.upper(), logging.INFO)

    # ── Root logger ───────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # Handlers filter level individually

    fmt_full  = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-25s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_trade = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Console handler (coloured) ────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(ColorFormatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # ── General log file (rotating, keeps 7 days) ─────────────────
    general_path = os.path.join(log_dir, f"agent_{today}.log")
    general_file = logging.handlers.TimedRotatingFileHandler(
        general_path, when="midnight", backupCount=7, encoding="utf-8"
    )
    general_file.setLevel(log_level)
    general_file.setFormatter(fmt_full)
    root.addHandler(general_file)

    # ── Trade log (trades only, keeps 30 days) ────────────────────
    trade_path = os.path.join(log_dir, f"trades_{today}.log")
    trade_file = logging.handlers.TimedRotatingFileHandler(
        trade_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    trade_file.setLevel(logging.INFO)
    trade_file.setFormatter(fmt_trade)
    trade_file.addFilter(TradeFilter())
    root.addHandler(trade_file)

    # ── Error log (WARNING+, keeps 30 days) ───────────────────────
    error_path = os.path.join(log_dir, f"errors_{today}.log")
    error_file = logging.handlers.TimedRotatingFileHandler(
        error_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    error_file.setLevel(logging.WARNING)
    error_file.setFormatter(fmt_full)
    root.addHandler(error_file)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "requests", "yfinance", "peewee",
                  "schedule", "kiteconnect", "websocket"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(
        f"Logging initialised — level={level} "
        f"| general={general_path} | trades={trade_path}"
    )
    return root


def get_trade_logger() -> logging.Logger:
    """Get a dedicated logger for trade events."""
    return logging.getLogger("trades")


def log_trade_event(
    event: str,
    symbol: str,
    details: dict,
    logger: logging.Logger = None,
):
    """Structured trade event logging."""
    lg = logger or logging.getLogger("trades")
    detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
    lg.info(f"[{event}] {symbol} | {detail_str}")
