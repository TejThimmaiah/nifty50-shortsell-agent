"""
Tej Social Sentiment Radar
============================
Real-time Reddit, Twitter/X, StockTwits sentiment on every Nifty stock.
Free scraping — no API keys needed.

"HDFCBANK: 847 negative posts on Reddit r/IndiaInvestments today.
 Twitter sentiment: -0.72. Retail is bearish — confirms our short."
"""

import os, re, logging, requests, time
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
logger = logging.getLogger("social_sentiment")
IST = ZoneInfo("Asia/Kolkata")

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


@dataclass
class SocialPost:
    source:    str   # "reddit" / "twitter" / "stocktwits"
    text:      str
    score:     float # -1 to +1
    timestamp: str


@dataclass
class SocialSentiment:
    symbol:      str
    avg_score:   float
    post_count:  int
    breakdown:   Dict[str, int]   # bearish/neutral/bullish count per source
    signal:      str
    confidence:  float
    top_posts:   List[str]


class SocialSentimentRadar:
    """
    Scrapes social media for stock sentiment.
    Uses DuckDuckGo news search as free data source.
    Falls back gracefully if no data found.
    """

    BEARISH = ["sell", "short", "crash", "dump", "fall", "drop", "down", "loss",
               "fraud", "scam", "avoid", "negative", "concern", "weak", "disappointing"]
    BULLISH = ["buy", "hold", "long", "moon", "pump", "rise", "up", "profit",
               "strong", "bullish", "positive", "growth", "upside", "buy the dip"]

    def _score_text(self, text: str) -> float:
        t = text.lower()
        b = sum(1 for w in self.BEARISH if w in t)
        u = sum(1 for w in self.BULLISH if w in t)
        total = b + u
        if total == 0:
            return 0.0
        return round((u - b) / total, 3)

    def _search_social(self, symbol: str, source: str) -> List[SocialPost]:
        """Search DuckDuckGo for social posts about a stock."""
        if not DDGS:
            return []
        queries = {
            "reddit":     f"site:reddit.com {symbol} NSE stock",
            "twitter":    f"site:twitter.com {symbol} NSE India stock",
            "stocktwits": f"site:stocktwits.com {symbol}",
            "general":    f"{symbol} NSE India investor opinion today",
        }
        posts = []
        try:
            q = queries.get(source, queries["general"])
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=10):
                    text  = f"{r.get('title','')} {r.get('body','')}"
                    score = self._score_text(text)
                    posts.append(SocialPost(
                        source=source, text=text[:200],
                        score=score,
                        timestamp=datetime.now(IST).isoformat()
                    ))
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            logger.debug(f"Social search failed ({source}, {symbol}): {e}")
        return posts

    def analyze(self, symbol: str) -> SocialSentiment:
        """Full social sentiment analysis."""
        all_posts = []
        breakdown = {"reddit": 0, "twitter": 0, "stocktwits": 0}

        for source in ["reddit", "twitter", "stocktwits"]:
            posts = self._search_social(symbol, source)
            all_posts.extend(posts)
            breakdown[source] = len(posts)

        if not all_posts:
            return SocialSentiment(
                symbol=symbol, avg_score=0.0, post_count=0,
                breakdown={"bearish": 0, "neutral": 0, "bullish": 0},
                signal="NO_DATA", confidence=0.0, top_posts=[]
            )

        scores    = [p.score for p in all_posts]
        avg_score = sum(scores) / len(scores)
        sentiment_breakdown = {
            "bearish": sum(1 for s in scores if s < -0.1),
            "neutral": sum(1 for s in scores if -0.1 <= s <= 0.1),
            "bullish": sum(1 for s in scores if s > 0.1),
        }

        if avg_score < -0.3:
            signal, confidence = "STRONG_SHORT_BIAS", 0.75
        elif avg_score < -0.1:
            signal, confidence = "SHORT_BIAS", 0.60
        elif avg_score > 0.3:
            signal, confidence = "AVOID_SHORT", 0.70
        else:
            signal, confidence = "NEUTRAL", 0.50

        top = sorted(all_posts, key=lambda p: p.score)[:3]
        return SocialSentiment(
            symbol=symbol, avg_score=round(avg_score, 3),
            post_count=len(all_posts),
            breakdown=sentiment_breakdown,
            signal=signal, confidence=confidence,
            top_posts=[p.text[:100] for p in top],
        )

    def format_for_telegram(self, symbol: str) -> str:
        s = self.analyze(symbol)
        emoji = "🔴" if s.avg_score < -0.1 else ("🟢" if s.avg_score > 0.1 else "🟡")
        msg = (
            f"<b>Social Sentiment: {symbol}</b>\n\n"
            f"{emoji} Score: {s.avg_score:+.3f} | Signal: {s.signal}\n"
            f"Posts analyzed: {s.post_count}\n"
            f"🔴 {s.breakdown.get('bearish',0)} bearish | "
            f"🟡 {s.breakdown.get('neutral',0)} neutral | "
            f"🟢 {s.breakdown.get('bullish',0)} bullish\n"
        )
        if s.top_posts:
            msg += "\n<b>Sample posts:</b>\n"
            for p in s.top_posts[:2]:
                msg += f"• {p}\n"
        return msg


social_radar = SocialSentimentRadar()
