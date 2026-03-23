"""
Self-Healer Agent
When any component fails or the agent hits an unknown situation,
this agent autonomously searches the web for solutions and applies them.
No human intervention required.
"""

import logging
import time
import json
import requests
from typing import Optional, Dict, List, Any
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from config import MAX_SEARCH_ATTEMPTS, SEARCH_COOLDOWN_SEC, GROQ_API_KEY, GEMINI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


class SelfHealerAgent:
    """
    Autonomous self-healing agent.
    When stuck, searches DuckDuckGo for relevant financial/technical information,
    then uses the LLM to synthesize a solution or updated strategy.
    """

    def __init__(self):
        self.search_history: List[str] = []
        self.last_search_time: float = 0
        self._ddgs = DDGS()

    # ──────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────

    def heal(self, problem: str, context: Dict = None) -> Dict:
        """
        Given a problem description, search for solutions and return a response.
        Returns: {"solution": str, "sources": list, "confidence": float}
        """
        logger.info(f"Self-healer activated for: {problem[:100]}")

        # Rate limiting
        elapsed = time.time() - self.last_search_time
        if elapsed < SEARCH_COOLDOWN_SEC:
            time.sleep(SEARCH_COOLDOWN_SEC - elapsed)

        # Build targeted search queries
        queries = self._build_queries(problem, context or {})
        all_results = []

        for query in queries[:MAX_SEARCH_ATTEMPTS]:
            results = self._search(query)
            all_results.extend(results)
            if all_results:
                break
            time.sleep(2)

        self.last_search_time = time.time()

        if not all_results:
            return {
                "solution": "Could not find relevant information. Using default fallback.",
                "sources": [],
                "confidence": 0.1,
            }

        # Synthesize with LLM
        synthesis = self._synthesize(problem, all_results, context or {})
        return synthesis

    # ──────────────────────────────────────────────────────────────
    # FINANCE-SPECIFIC SEARCHES
    # ──────────────────────────────────────────────────────────────

    def search_stock_news(self, symbol: str) -> List[Dict]:
        """Search for latest news about a specific stock."""
        query = f"{symbol} NSE stock news today bearish"
        results = self._search(query, max_results=5)
        return results

    def search_sector_news(self, sector: str) -> List[Dict]:
        """Search for sector-wide news."""
        query = f"{sector} sector India stock market today"
        return self._search(query, max_results=5)

    def search_market_sentiment(self) -> Dict:
        """Search for current market sentiment signals."""
        queries = [
            "NSE Nifty 50 today market outlook bearish bullish",
            "FII DII activity NSE today",
            "India stock market news today",
        ]
        results = []
        for q in queries:
            results.extend(self._search(q, max_results=3))

        if not results:
            return {"sentiment": "UNKNOWN", "reason": "No data"}

        # Use LLM to assess sentiment
        snippets = "\n".join([r.get("body", "") for r in results[:6]])
        prompt = f"""Based on these news snippets about Indian stock markets, 
        determine the market sentiment for SHORT SELLING opportunities today.
        News: {snippets[:2000]}
        
        Respond ONLY in JSON: {{"sentiment": "BEARISH/BULLISH/NEUTRAL", "confidence": 0.0-1.0, 
        "key_factors": ["factor1", "factor2"], "good_for_shorts": true/false}}"""

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except Exception:
            return {"sentiment": "NEUTRAL", "confidence": 0.3, "good_for_shorts": False}

    def search_api_error_fix(self, error_message: str, api_name: str) -> str:
        """Search for fixes to API errors (Zerodha, NSE, etc.)."""
        query = f"{api_name} API error fix {error_message[:50]} Python"
        results = self._search(query, max_results=5)
        if not results:
            return "No solution found. Check API credentials."

        snippets = "\n".join([r.get("body", "") for r in results[:4]])
        prompt = f"""An API error occurred:
        API: {api_name}
        Error: {error_message}
        
        Based on these search results, what is the most likely fix?
        Search results: {snippets[:1500]}
        
        Give a concise, actionable fix in 2-3 sentences."""

        return self._call_llm(prompt)

    def search_short_strategy_update(self, market_condition: str) -> Dict:
        """Search for current short selling strategies given market conditions."""
        query = f"intraday short selling NSE India strategy {market_condition} 2024"
        results = self._search(query, max_results=6)

        if not results:
            return {"strategy": "default", "adjustments": []}

        snippets = "\n".join([r.get("body", "") for r in results[:5]])
        prompt = f"""Market condition: {market_condition}
        Based on these search results about Indian intraday short selling:
        {snippets[:2000]}
        
        Suggest strategy adjustments as JSON:
        {{"tighten_sl": true/false, "reduce_position_size": true/false, 
         "focus_sectors": ["sector1"], "avoid_sectors": ["sector2"],
         "confidence_threshold_adjustment": -0.1 to 0.1,
         "reasoning": "brief explanation"}}"""

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except Exception:
            return {"strategy": "default", "adjustments": [], "reasoning": "Parse error"}

    # ──────────────────────────────────────────────────────────────
    # INTERNAL METHODS
    # ──────────────────────────────────────────────────────────────

    def _search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Execute DuckDuckGo search — completely free, no API key needed."""
        if query in self.search_history[-20:]:
            logger.debug(f"Skipping duplicate query: {query}")
            return []

        self.search_history.append(query)
        logger.info(f"Searching: {query}")

        try:
            results = list(self._ddgs.text(
                query,
                max_results=max_results,
                region="in-en",          # India region for better NSE results
                safesearch="moderate",
            ))
            logger.debug(f"Found {len(results)} results for: {query}")
            return results
        except Exception as e:
            logger.warning(f"Search failed [{query}]: {e}")
            try:
                # Retry with global region
                results = list(DDGS().text(query, max_results=max_results))
                return results
            except Exception:
                return []

    def _build_queries(self, problem: str, context: Dict) -> List[str]:
        """Build targeted search queries from the problem description."""
        queries = []
        symbol = context.get("symbol", "")
        error  = context.get("error", "")

        # Primary query
        if symbol:
            queries.append(f"{symbol} NSE {problem[:60]}")
        queries.append(f"NSE India short selling {problem[:60]}")

        # Error-specific query
        if error:
            queries.append(f"Python fix {error[:50]}")

        # Generic financial query
        queries.append(f"Indian stock market {problem[:50]} intraday trading")

        return queries[:MAX_SEARCH_ATTEMPTS]

    def _synthesize(self, problem: str, results: List[Dict], context: Dict) -> Dict:
        """Use LLM to synthesize search results into an actionable solution."""
        snippets = "\n\n".join([
            f"Source: {r.get('href', 'unknown')}\n{r.get('body', '')}"
            for r in results[:6]
        ])
        sources = [r.get("href", "") for r in results[:6]]

        prompt = f"""You are an autonomous trading agent that handles problems without human help.

Problem encountered: {problem}
Context: {json.dumps(context)}

Web search results:
{snippets[:3000]}

Provide a concrete solution to this problem for an automated intraday short selling system on NSE.
Be specific, actionable, and consider the financial/technical context.
Response (max 200 words):"""

        solution = self._call_llm(prompt)

        return {
            "solution": solution,
            "sources": [s for s in sources if s],
            "confidence": min(0.8, 0.3 + len(results) * 0.1),
        }

    def _call_llm(self, prompt: str) -> str:
        """Call LLM (Groq free tier → Gemini fallback)."""
        # Try Groq first
        if GROQ_API_KEY:
            result = self._call_groq(prompt)
            if result:
                return result

        # Fallback to Gemini
        if GEMINI_API_KEY:
            result = self._call_gemini(prompt)
            if result:
                return result

        return "LLM unavailable. Using default behavior."

    def _call_groq(self, prompt: str) -> Optional[str]:
        """Groq API — free tier, Llama 3.3 70B."""
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Groq API error: {e}")
            return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Google Gemini Flash — free tier fallback."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 512, "temperature": 0.3},
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.warning(f"Gemini API error: {e}")
            return None
