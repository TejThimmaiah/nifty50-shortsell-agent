"""
Tej Chart Vision — LLaVA
==========================
Tej sees candlestick charts as images.
Finds patterns humans can't see.

"I can see a classic bearish engulfing on the weekly chart,
 combined with a hanging man candle on the daily.
 This is a textbook distribution top."
"""

import os, io, logging, base64, requests
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
logger = logging.getLogger("chart_vision")
IST = ZoneInfo("Asia/Kolkata")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
GROQ_KEY    = os.getenv("GROQ_API_KEY", "")


def render_candlestick_chart(df, symbol: str, days: int = 30) -> bytes:
    """Render OHLCV DataFrame as candlestick PNG image."""
    if not MATPLOTLIB_AVAILABLE:
        return b""
    try:
        recent = df.tail(days).copy()
        recent.reset_index(inplace=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#0a0f1e")
        ax, vol_ax = axes

        for ax_ in axes:
            ax_.set_facecolor("#0a0f1e")
            ax_.tick_params(colors="#7090c0")
            for spine in ax_.spines.values():
                spine.set_color("#1a3a6a")

        # Draw candles
        for i, row in recent.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = "#e74c3c" if c < o else "#2ecc71"
            ax.plot([i, i], [l, h], color=color, linewidth=0.8)
            ax.add_patch(mpatches.Rectangle(
                (i - 0.3, min(o, c)), 0.6, abs(c - o),
                color=color, alpha=0.9
            ))

        # Volume bars
        for i, row in recent.iterrows():
            color = "#e74c3c" if row["close"] < row["open"] else "#2ecc71"
            vol_ax.bar(i, row["volume"], color=color, alpha=0.6, width=0.8)

        ax.set_title(f"{symbol} — {days}D Chart", color="#90d0ff", fontsize=14)
        ax.set_ylabel("Price", color="#5a8fc0")
        vol_ax.set_ylabel("Volume", color="#5a8fc0")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="#0a0f1e")
        plt.close()
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Chart render failed: {e}")
        return b""


def analyze_chart_with_llava(img_bytes: bytes, symbol: str) -> str:
    """
    Send chart image to LLaVA (via Ollama) for visual pattern analysis.
    LLaVA is a multimodal open-source model that understands images.
    """
    if not img_bytes:
        return "No chart image available."

    img_b64 = base64.b64encode(img_bytes).decode()

    # Try Ollama first (runs locally on GCP)
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model":  "llava",
                "prompt": (
                    f"You are analyzing a candlestick chart for {symbol} on NSE India. "
                    "Focus on: 1) Overall trend direction, 2) Key patterns (head & shoulders, "
                    "double top, bearish engulfing, etc), 3) Support/resistance levels visible, "
                    "4) Volume analysis, 5) Short-selling opportunity assessment. "
                    "Be specific. End with: SIGNAL: STRONG_SHORT / SHORT / NEUTRAL / AVOID"
                ),
                "images": [img_b64],
                "stream": False,
            },
            timeout=30,
        )
        if r.ok:
            return r.json().get("response", "No analysis")
    except Exception:
        pass

    # Fallback: describe with heuristics
    return f"Chart analysis for {symbol}: Visual pattern analysis requires Ollama+LLaVA on GCP."


class ChartVisionAnalyzer:
    """Analyzes stock charts using computer vision."""

    def analyze(self, df, symbol: str) -> dict:
        img = render_candlestick_chart(df, symbol)
        analysis = analyze_chart_with_llava(img, symbol)

        # Extract signal
        signal = "NEUTRAL"
        if "STRONG_SHORT" in analysis:
            signal = "STRONG_SHORT"
        elif "SHORT" in analysis and "AVOID" not in analysis:
            signal = "SHORT"
        elif "AVOID" in analysis:
            signal = "AVOID"

        return {
            "symbol":   symbol,
            "analysis": analysis,
            "signal":   signal,
            "has_image": len(img) > 0,
        }

    def format_for_telegram(self, df, symbol: str) -> tuple:
        """Returns (text_message, image_bytes)."""
        result = self.analyze(df, symbol)
        msg = (
            f"<b>Chart Vision: {symbol}</b>\n\n"
            f"Signal: {result['signal']}\n\n"
            f"{result['analysis'][:600]}"
        )
        img = render_candlestick_chart(df, symbol)
        return msg, img


chart_vision = ChartVisionAnalyzer()
