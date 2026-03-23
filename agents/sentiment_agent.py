"""
Sentiment Agent
Scrapes financial news, MoneyControl headlines, Economic Times,
and Reddit/Twitter signals to score sentiment for each stock.
Fully free — no paid APIs needed.
"""

import re
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from agents.self_healer import SelfHealerAgent

logger = logging.getLogger(__name__)

# ── Free RSS feeds for Indian financial news ─────────────────────────────────
RSS_FEEDS = {
    "economic_times": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "moneycontrol":   "https://www.moneycontrol.com/rss/marketoutlook.xml",
    "livemint":       "https://www.livemint.com/rss/markets",
    "business_std":   "https://www.business-standard.com/rss/markets-106.rss",
    "ndtv_profit":    "https://feeds.feedburner.com/NdtvProfit-Markets",
}

# Bearish trigger words for sentiment scoring
BEARISH_WORDS = {
    "strong": [
        "fraud", "scam", "investigation", "sebi action", "bankruptcy", "default",
        "promoter selling", "block deal", "debt trap", "rating downgrade",
        "profit warning", "guidance cut", "write-off", "npa", "insolvency",
        "regulatory action", "penalty", "ban", "suspension",
    ],
    "moderate": [
        "miss", "below estimate", "disappointing", "slowdown", "margin pressure",
        "loss", "decline", "fall", "drop", "weak", "concern", "risk",
        "headwind", "challenge", "competition", "expensive", "overvalued",
        "sell", "underperform", "reduce", "target cut", "downgrade",
    ],
    "mild": [
        "caution", "uncertainty", "volatile", "pressure", "difficult",
        "moderate", "mixed", "flat", "sluggish",
    ],
}

BULLISH_WORDS = {
    "strong": [
        "record profit", "beat estimate", "upgrade", "outperform", "buy",
        "strong results", "acquisition", "buyback", "dividend", "bonus",
        "new order", "contract win", "expansion", "recovery",
    ],
    "moderate": [
        "growth", "positive", "above", "strong", "profit", "revenue up",
        "margin expansion", "guidance raised", "optimistic",
    ],
}


@dataclass
class StockSentiment:
    symbol: str
    score: float                  # -1.0 (very bearish) to +1.0 (very bullish)
    label: str                    # "STRONG_BEARISH", "BEARISH", "NEUTRAL", "BULLISH"
    articles_found: int
    key_headlines: List[str] = field(default_factory=list)
    bearish_signals: List[str] = field(default_factory=list)
    bullish_signals: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MarketSentiment:
    overall_score: float
    label: str
    fii_sentiment: str
    news_sentiment: str
    vix_level: float
    good_for_shorts: bool
    key_factors: List[str] = field(default_factory=list)


class SentimentAgent:
    """
    Analyses market and stock-level sentiment using free data sources.
    Uses RSS feeds, BeautifulSoup scraping, and the self-healer for web search.
    """

    def __init__(self, healer: SelfHealerAgent = None):
        self.healer = healer or SelfHealerAgent()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; TejBot/1.0)",
        })
        self._news_cache: Dict[str, List[Dict]] = {}
        self._cache_time: Optional[datetime] = None

    # ──────────────────────────────────────────────────────────────
    # STOCK SENTIMENT
    # ──────────────────────────────────────────────────────────────

    def analyse_stock(self, symbol: str) -> StockSentiment:
        """
        Full sentiment analysis for a single stock.
        Checks news headlines and recent articles.
        """
        articles = self._fetch_stock_news(symbol)
        score, bearish_signals, bullish_signals = self._score_articles(articles, symbol)
        headlines = [a.get("title", "") for a in articles[:5]]
        label = self._score_to_label(score)
        confidence = min(1.0, len(articles) * 0.15)

        logger.info(
            f"Sentiment [{symbol}]: {label} (score={score:.2f}, "
            f"articles={len(articles)}, confidence={confidence:.2f})"
        )

        return StockSentiment(
            symbol=symbol,
            score=score,
            label=label,
            articles_found=len(articles),
            key_headlines=headlines,
            bearish_signals=bearish_signals[:5],
            bullish_signals=bullish_signals[:5],
            confidence=confidence,
        )

    def analyse_multiple(self, symbols: List[str]) -> Dict[str, StockSentiment]:
        """Analyse sentiment for a list of stocks. Uses cached news to avoid re-fetching."""
        self._refresh_news_cache()
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyse_stock(symbol)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed [{symbol}]: {e}")
                results[symbol] = StockSentiment(
                    symbol=symbol, score=0.0, label="NEUTRAL",
                    articles_found=0, confidence=0.0,
                )
        return results

    # ──────────────────────────────────────────────────────────────
    # MARKET-WIDE SENTIMENT
    # ──────────────────────────────────────────────────────────────

    def get_market_sentiment(self, fii_net: float = 0) -> MarketSentiment:
        """
        Assess overall market sentiment for the trading day.
        Combines: news headlines + FII flow + India VIX.
        """
        # Refresh news cache
        self._refresh_news_cache()

        # Score general market news
        market_articles = self._get_cached_articles("market")
        market_score, _, _ = self._score_articles(market_articles, "NIFTY")

        # FII sentiment
        if fii_net < -1000:
            fii_label = "STRONGLY_BEARISH"
            market_score -= 0.3
        elif fii_net < -300:
            fii_label = "BEARISH"
            market_score -= 0.15
        elif fii_net > 1000:
            fii_label = "STRONGLY_BULLISH"
            market_score += 0.3
        elif fii_net > 300:
            fii_label = "BULLISH"
            market_score += 0.15
        else:
            fii_label = "NEUTRAL"

        # India VIX (fetch from NSE)
        vix = self._fetch_india_vix()
        if vix > 20:
            market_score -= 0.1   # High VIX = fear = caution for shorts too
        elif vix < 14:
            market_score += 0.1   # Low VIX = complacency = good for shorts

        market_score = max(-1.0, min(1.0, market_score))
        label = self._score_to_label(market_score)

        # Shorts work best in bearish/neutral market
        good_for_shorts = market_score <= 0.1 and fii_label in (
            "BEARISH", "STRONGLY_BEARISH", "NEUTRAL"
        )

        key_factors = []
        if fii_net < -300:
            key_factors.append(f"FII selling ₹{abs(fii_net):.0f}Cr")
        if vix > 18:
            key_factors.append(f"High VIX {vix:.1f}")
        if market_score < -0.3:
            key_factors.append("Negative news flow")

        return MarketSentiment(
            overall_score=round(market_score, 3),
            label=label,
            fii_sentiment=fii_label,
            news_sentiment="BEARISH" if market_score < -0.1 else (
                "BULLISH" if market_score > 0.1 else "NEUTRAL"
            ),
            vix_level=vix,
            good_for_shorts=good_for_shorts,
            key_factors=key_factors,
        )

    # ──────────────────────────────────────────────────────────────
    # NEWS FETCHING
    # ──────────────────────────────────────────────────────────────

    def _refresh_news_cache(self, force: bool = False):
        """Refresh RSS news cache if older than 30 minutes."""
        if (not force and self._cache_time and
                (datetime.now() - self._cache_time).seconds < 1800):
            return

        logger.info("Refreshing news cache from RSS feeds...")
        self._news_cache = {}

        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                articles = []
                for entry in feed.entries[:20]:
                    articles.append({
                        "title":   entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "link":    entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source":  source,
                    })
                self._news_cache[source] = articles
                logger.debug(f"Cached {len(articles)} articles from {source}")
            except Exception as e:
                logger.warning(f"RSS fetch failed [{source}]: {e}")

        self._cache_time = datetime.now()

    def _fetch_stock_news(self, symbol: str) -> List[Dict]:
        """
        Find news articles mentioning this stock symbol.
        Searches across all cached RSS sources + a targeted DDG search.
        """
        symbol_lower = symbol.lower()
        matches = []

        # Search cached RSS articles
        for source, articles in self._news_cache.items():
            for article in articles:
                text = (article.get("title", "") + " " + article.get("summary", "")).lower()
                if symbol_lower in text or symbol_lower.replace("ltd", "") in text:
                    matches.append(article)

        # If sparse, fall back to self-healer web search
        if len(matches) < 2:
            search_results = self.healer.search_stock_news(symbol)
            for r in search_results:
                matches.append({
                    "title":   r.get("title", ""),
                    "summary": r.get("body", ""),
                    "source":  "duckduckgo",
                    "link":    r.get("href", ""),
                })

        return matches

    def _get_cached_articles(self, category: str = "market") -> List[Dict]:
        """Get all cached articles for general market sentiment."""
        all_articles = []
        for articles in self._news_cache.values():
            all_articles.extend(articles)
        return all_articles

    def _fetch_india_vix(self) -> float:
        """Fetch India VIX from NSE."""
        try:
            resp = self._session.get(
                "https://www.nseindia.com/api/equity-stockIndices?index=INDIA+VIX",
                timeout=10,
                headers={
                    "Referer": "https://www.nseindia.com/",
                    "User-Agent": "Mozilla/5.0",
                }
            )
            data = resp.json()
            vix = float(data.get("metadata", {}).get("last", 15.0))
            logger.debug(f"India VIX: {vix}")
            return vix
        except Exception:
            return 15.0   # Neutral default

    # ──────────────────────────────────────────────────────────────
    # SCORING ENGINE
    # ──────────────────────────────────────────────────────────────

    def _score_articles(
        self, articles: List[Dict], symbol: str
    ) -> Tuple[float, List[str], List[str]]:
        """
        Score a list of articles for bearish/bullish sentiment.
        Returns (score, bearish_signals, bullish_signals).
        score: -1.0 = very bearish, 0 = neutral, +1.0 = very bullish
        """
        if not articles:
            return 0.0, [], []

        total_score = 0.0
        bearish_signals = []
        bullish_signals = []

        for article in articles:
            text = (
                article.get("title", "") + " " + article.get("summary", "")
            ).lower()

            # Check bearish words
            for word in BEARISH_WORDS["strong"]:
                if word in text:
                    total_score -= 0.4
                    bearish_signals.append(f"[strong] {word}")
            for word in BEARISH_WORDS["moderate"]:
                if word in text:
                    total_score -= 0.15
                    bearish_signals.append(f"[mod] {word}")
            for word in BEARISH_WORDS["mild"]:
                if word in text:
                    total_score -= 0.05

            # Check bullish words
            for word in BULLISH_WORDS["strong"]:
                if word in text:
                    total_score += 0.4
                    bullish_signals.append(f"[strong] {word}")
            for word in BULLISH_WORDS["moderate"]:
                if word in text:
                    total_score += 0.15
                    bullish_signals.append(f"[mod] {word}")

        # Normalize by number of articles
        normalized = total_score / max(len(articles), 1)
        clamped = max(-1.0, min(1.0, normalized))

        return round(clamped, 3), list(set(bearish_signals)), list(set(bullish_signals))

    def _score_to_label(self, score: float) -> str:
        if score <= -0.5:   return "STRONG_BEARISH"
        if score <= -0.2:   return "BEARISH"
        if score >= 0.5:    return "STRONG_BULLISH"
        if score >= 0.2:    return "BULLISH"
        return "NEUTRAL"

    # ──────────────────────────────────────────────────────────────
    # CONVICTION BOOST FOR SHORT TRADES
    # ──────────────────────────────────────────────────────────────

    def get_short_conviction(self, symbol: str, technical_confidence: float) -> float:
        """
        Combine technical confidence with sentiment to get final short conviction.
        Returns adjusted confidence score (0–1).
        """
        sentiment = self.analyse_stock(symbol)
        s = sentiment.score

        # Bearish news boosts short conviction; bullish news reduces it
        if s <= -0.5:    adjustment = +0.15
        elif s <= -0.2:  adjustment = +0.08
        elif s >= 0.5:   adjustment = -0.20  # Strong bullish news = avoid short
        elif s >= 0.2:   adjustment = -0.10
        else:            adjustment = 0.0

        final = max(0.0, min(1.0, technical_confidence + adjustment))
        logger.info(
            f"Conviction [{symbol}]: tech={technical_confidence:.2f} + "
            f"sentiment={s:.2f} ({sentiment.label}) → final={final:.2f}"
        )
        return final
