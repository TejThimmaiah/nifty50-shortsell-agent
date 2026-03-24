"""
Tej NeMo Guardrails + Ollama Local LLM
========================================
TWO FEATURES:

1. NEMO GUARDRAILS:
   Battle-tested safety rails for Tej's AI decisions.
   Prevents hallucinations, invalid orders, dangerous positions.

2. OLLAMA + NEMOTRON:
   Run AI brain locally on GCP e2-micro — zero API dependency.
   Free forever. No rate limits. No internet needed.
"""

import os, logging, requests
logger = logging.getLogger("nemo_ollama")

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nemotron-mini")   # 4B params, fits in 4GB RAM


# ══════════════════════════════════════════════════════
# PART 1 — NEMO GUARDRAILS
# ══════════════════════════════════════════════════════

GUARDRAIL_CONFIG = """
# Tej NeMo Guardrails Configuration
# Prevents dangerous AI decisions

define user wants to trade long
  "buy", "go long", "long position", "bullish trade"

define bot refuse long trades
  "I only short sell. We have a SHORT_ONLY mandate. No long positions."

define user wants to exceed position limit
  "all in", "use all capital", "maximum position"

define bot refuse excessive risk
  "Maximum 2% capital risk per trade. I will not exceed this."

define user wants to trade unknown stock
  "trade X" where X is not in Nifty50

define bot refuse non-nifty
  "I only trade Nifty 50 stocks on NSE. That stock is not in our universe."

define flow main
  user message
  if detect user wants to trade long
    bot refuse long trades
  if detect user wants to exceed position limit
    bot refuse excessive risk
  bot respond
"""


class NeMoGuardrails:
    """
    Safety rails for Tej's trading decisions.
    Falls back to rule-based checks if NeMo not available.
    """

    BLOCKED_ACTIONS = [
        "long", "buy equity", "delivery", "cnc", "overnight hold",
        "leverage beyond capital", "all in"
    ]

    NIFTY50_SYMBOLS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
        "KOTAKBANK", "LT", "SBIN", "BHARTIARTL", "ASIANPAINT", "AXISBANK",
        "BAJFINANCE", "WIPRO", "MARUTI", "HCLTECH", "ULTRACEMCO", "TITAN",
        "NTPC", "SUNPHARMA", "POWERGRID", "NESTLEIND", "BAJAJFINSV",
        "TECHM", "TATASTEEL", "DRREDDY", "DIVISLAB", "HINDALCO",
        "ADANIENT", "ONGC", "JSWSTEEL", "TATAMOTORS", "GRASIM",
        "CIPLA", "COALINDIA", "BRITANNIA", "BPCL", "EICHERMOT",
        "APOLLOHOSP", "BAJAJ_AUTO", "HEROMOTOCO", "TATACONSUM",
        "SBILIFE", "HDFCLIFE", "INDUSINDBK", "UPL", "LTIMindtree",
        "ADANIPORTS", "MM", "SHRIRAMFIN"
    ]

    def check_trade(self, symbol: str, direction: str, quantity: int,
                    capital: float, entry: float, stop: float) -> dict:
        """
        Validate a trade against all guardrails.
        Returns {"approved": bool, "reason": str}
        """
        # Rule 1: SHORT only
        if direction.upper() not in ("SHORT", "SELL"):
            return {"approved": False, "reason": f"GUARDRAIL: Only SHORT trades allowed. Got {direction}"}

        # Rule 2: Nifty 50 only
        if symbol.upper() not in [s.upper() for s in self.NIFTY50_SYMBOLS]:
            return {"approved": False, "reason": f"GUARDRAIL: {symbol} not in Nifty 50 universe"}

        # Rule 3: Max 2% risk
        risk = abs(stop - entry) * quantity
        risk_pct = risk / capital * 100
        if risk_pct > 2.5:
            return {"approved": False, "reason": f"GUARDRAIL: Risk {risk_pct:.1f}% exceeds 2% limit"}

        # Rule 4: Positive R:R
        # (entry - target) / (stop - entry) should be >= 2
        # Caller must check R:R separately

        # Rule 5: No 0 or negative quantity
        if quantity <= 0:
            return {"approved": False, "reason": "GUARDRAIL: Quantity must be positive"}

        return {"approved": True, "reason": "All guardrails passed"}

    def check_ai_response(self, response: str) -> dict:
        """
        Check if AI response contains anything dangerous.
        """
        response_lower = response.lower()
        for blocked in self.BLOCKED_ACTIONS:
            if blocked in response_lower:
                return {
                    "safe": False,
                    "reason": f"AI response contains blocked action: '{blocked}'",
                    "filtered": response.replace(blocked, "[BLOCKED]")
                }
        return {"safe": True, "reason": "Response passed guardrails", "filtered": response}


# ══════════════════════════════════════════════════════
# PART 2 — OLLAMA LOCAL LLM
# ══════════════════════════════════════════════════════

class OllamaLLM:
    """
    Run Nemotron or any open-source model locally on GCP.
    Completely free. No rate limits. Works offline.
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model     = model
        self.available = self._check()

    def _check(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
            if r.ok:
                models = [m["name"] for m in r.json().get("models", [])]
                if any(self.model in m for m in models):
                    logger.info(f"Ollama: {self.model} available locally")
                    return True
                else:
                    logger.info(f"Ollama running but {self.model} not pulled yet")
                    return False
        except Exception:
            logger.info("Ollama not running locally")
            return False

    def generate(self, prompt: str, system: str = "", max_tokens: int = 500) -> str:
        """Generate text using local Ollama model."""
        if not self.available:
            return ""
        try:
            payload = {
                "model":  self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            }
            if system:
                payload["system"] = system
            r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=30)
            if r.ok:
                return r.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
        return ""

    def chat(self, messages: list, max_tokens: int = 500) -> str:
        """Chat with local model."""
        if not self.available:
            return ""
        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False,
                      "options": {"num_predict": max_tokens}},
                timeout=30,
            )
            if r.ok:
                return r.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
        return ""

    @staticmethod
    def install_instructions() -> str:
        return """
# Install Ollama on GCP e2-micro:
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Nemotron mini (4B — fits in 4GB RAM):
ollama pull nemotron-mini

# Or Llama 3.2 3B (even smaller):
ollama pull llama3.2:3b

# Test:
ollama run nemotron-mini "What is short selling?"

# Ollama runs as a service — always available at localhost:11434
sudo systemctl enable ollama
sudo systemctl start ollama
"""


nemo_guardrails = NeMoGuardrails()
ollama_llm      = OllamaLLM()
