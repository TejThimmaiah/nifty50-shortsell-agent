"""
Tej Sentiment Engine — FinBERT + DuckDuckGo
=============================================
Real-time financial sentiment analysis.
Reads news headlines for every Nifty 50 stock before market opens.

"HDFCBANK: 8 bearish articles, 2 bullish — sentiment score: -0.72 → SHORT bias"
"""

import os
import logging
import requests
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("sentiment_engine")
IST = ZoneInfo("Asia/Kolkata")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False


class SentimentEngine:
    """
    Financial sentiment analysis using FinBERT.
    Falls back to keyword-based scoring if FinBERT unavailable.
    """

    BEARISH_WORDS = [
        "decline", "fall", "drop", "loss", "weak", "bearish", "sell",
        "downgrade", "miss", "disappoint", "concern", "risk", "debt",
        "fraud", "probe", "investigation", "penalty", "fine", "lawsuit",
        "resign", "layoff", "cut", "reduced", "below", "warning"
    ]
    BULLISH_WORDS = [
        "rise", "gain", "profit", "strong", "bullish", "buy", "upgrade",
        "beat", "exceed", "growth", "record", "expand", "launch", "win",
        "acquire", "dividend", "buyback", "positive", "above", "target"
    ]

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if not FINBERT_AVAILABLE:
            logger.info("FinBERT not available — using keyword scoring")
            return
        try:
            logger.info("Loading FinBERT model...")
            self.model = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
                device=-1
            )
            logger.info("FinBERT loaded")
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e} — using keyword scoring")

    def _keyword_score(self, text: str) -> float:
        """Simple keyword-based sentiment: -1 (very bearish) to +1 (very bullish)."""
        text_lower = text.lower()
        bearish = sum(1 for w in self.BEARISH_WORDS if w in text_lower)
        bullish = sum(1 for w in self.BULLISH_WORDS if w in text_lower)
        total = bearish + bullish
        if total == 0:
            return 0.0
        return (bullish - bearish) / total

    def score_text(self, text: str) -> dict:
        """
        Score a single text snippet.
        Returns: {"score": -1 to +1, "label": "bearish/neutral/bullish", "confidence": 0-1}
        """
        if self.model:
            try:
                results = self.model(text[:512])[0]
                scores = {r["label"].lower(): r["score"] for r in results}
                net = scores.get("positive", 0) - scores.get("negative", 0)
                label = "bullish" if net > 0.1 else ("bearish" if net < -0.1 else "neutral")
                return {"score": round(net, 3), "label": label,
                        "confidence": max(scores.values())}
            except Exception:
                pass

        raw = self._keyword_score(text)
        label = "bullish" if raw > 0.1 else ("bearish" if raw < -0.1 else "neutral")
        return {"score": round(raw, 3), "label": label, "confidence": 0.6}

    def fetch_news(self, symbol: str, max_results: int = 10) -> list:
        """Fetch latest news headlines for a stock using DuckDuckGo."""
        if not DDGS_AVAILABLE:
            return []
        try:
            query = f"{symbol} NSE India stock news today"
            results = []
            with DDGS() as ddgs:
                for r in ddgs.news(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "body":  r.get("body", ""),
                        "date":  r.get("date", ""),
                        "url":   r.get("url", ""),
                    })
            return results
        except Exception as e:
            logger.warning(f"News fetch failed for {symbol}: {e}")
            return []

    def analyze_symbol(self, symbol: str) -> dict:
        """
        Full sentiment analysis for a stock.
        Returns aggregate score, breakdown, and trading signal.
        """
        news = self.fetch_news(symbol)
        if not news:
            return {
                "symbol":    symbol,
                "score":     0.0,
                "label":     "neutral",
                "signal":    "NO_DATA",
                "articles":  0,
                "breakdown": {"bearish": 0, "neutral": 0, "bullish": 0},
                "headlines": [],
            }

        scores = []
        breakdown = {"bearish": 0, "neutral": 0, "bullish": 0}
        headlines = []

        for article in news:
            text = f"{article['title']} {article['body'][:200]}"
            result = self.score_text(text)
            scores.append(result["score"])
            breakdown[result["label"]] += 1
            headlines.append({
                "title":     article["title"],
                "sentiment": result["label"],
                "score":     result["score"],
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        label = "bullish" if avg_score > 0.15 else ("bearish" if avg_score < -0.15 else "neutral")

        # Trading signal — we only short, so bearish news = opportunity
        if label == "bearish" and avg_score < -0.3:
            signal = "STRONG_SHORT_BIAS"
        elif label == "bearish":
            signal = "SHORT_BIAS"
        elif label == "bullish" and avg_score > 0.3:
            signal = "AVOID_SHORT"
        else:
            signal = "NEUTRAL"

        return {
            "symbol":    symbol,
            "score":     round(avg_score, 3),
            "label":     label,
            "signal":    signal,
            "articles":  len(news),
            "breakdown": breakdown,
            "headlines": headlines[:5],
            "timestamp": datetime.now(IST).isoformat(),
        }

    def scan_watchlist(self, symbols: list) -> dict:
        """
        Scan entire watchlist for sentiment.
        Returns ranked list — most bearish first (best short candidates).
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze_symbol(symbol)
                logger.info(f"Sentiment {symbol}: {results[symbol]['label']} ({results[symbol]['score']:+.2f})")
            except Exception as e:
                logger.error(f"Sentiment failed for {symbol}: {e}")

        # Sort by score ascending (most bearish first = best for shorts)
        ranked = sorted(results.items(), key=lambda x: x[1]["score"])
        return {
            "ranked":    [(s, r["score"], r["signal"]) for s, r in ranked],
            "results":   results,
            "timestamp": datetime.now(IST).isoformat(),
        }

    def format_for_telegram(self, symbol: str) -> str:
        """Format sentiment analysis as Telegram message."""
        result = self.analyze_symbol(symbol)
        emoji = {"bearish": "🔴", "neutral": "🟡", "bullish": "🟢"}.get(result["label"], "⚪")
        msg = (
            f"<b>Sentiment: {symbol}</b>\n"
            f"{emoji} {result['label'].upper()} (score: {result['score']:+.2f})\n"
            f"Signal: {result['signal']}\n"
            f"Articles: {result['articles']} | "
            f"🔴{result['breakdown']['bearish']} 🟡{result['breakdown']['neutral']} 🟢{result['breakdown']['bullish']}\n\n"
        )
        for h in result["headlines"][:3]:
            e = {"bearish": "🔴", "neutral": "🟡", "bullish": "🟢"}.get(h["sentiment"], "⚪")
            msg += f"{e} {h['title'][:80]}\n"
        return msg


sentiment_engine = SentimentEngine()
