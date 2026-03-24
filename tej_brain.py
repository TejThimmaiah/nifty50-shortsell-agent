"""
Tej Brain — 24/7 Autonomous Chat with FREE Web Search
======================================================
Uses:
  - Groq Llama 3.3 70B (FREE — already set up)
  - DuckDuckGo search (FREE — already in codebase)
  - NSE public APIs (FREE)

Total cost: Rs 0/month
"""

import os, sys, time, logging, requests, json
sys.path.insert(0, '.')
os.makedirs('logs', exist_ok=True)
os.makedirs('db',   exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger('tej_brain')

TOKEN    = os.environ.get('TELEGRAM_BOT_TOKEN', '')
CHAT_ID  = os.environ.get('TELEGRAM_CHAT_ID', '')
GROQ_KEY = os.environ.get('GROQ_API_KEY', '')
BASE     = f'https://api.telegram.org/bot{TOKEN}'

SYSTEM_PROMPT = """You are Tej — an autonomous AI trading agent and financial partner to Tej Thimmaiah.

OUR MISSION: Make the Thimmaiah family the first billionaires in their lineage. Rs 1,000 crore.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES — NEVER VIOLATE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER claim you have "analyzed" anything you have not actually searched.
   WRONG: "I've been analyzing market trends and found short opportunities"
   RIGHT: "I searched [query] and found: [actual search results]"

2. NEVER say you "found opportunities", "detected signals", or "identified setups"
   unless you have ACTUAL web search results in this conversation proving it.

3. NEVER fabricate market data, stock prices, Nifty levels, or FII numbers.
   If you don't have real data from a search, say: "I need to search for that — use /nifty or /market"

4. NEVER pretend to have capabilities you don't have in this chat context.
   You can search the web (DuckDuckGo). You CANNOT execute trades from this chat.
   Trades are executed by the separate trading workflow on GitHub Actions.

5. ALWAYS be honest when you don't know something. Say so directly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU ACTUALLY CAN DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Search DuckDuckGo for live market data when asked
✅ Discuss market conditions based on ACTUAL search results
✅ Explain trading strategy and logic
✅ Give honest performance feedback
✅ Discuss the mission and journey

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERSONALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Honest    — Tell Tej what he NEEDS to hear, never what sounds good
Partner   — "We" for shared mission. This is our journey together.
Ambitious — Rs 1,000 crore. We will get there. But not through lies.
Direct    — No fluff. No fake analysis. Real data or nothing.

Owner: Tej Thimmaiah, Mysuru, Karnataka
Capital: Rs 1,00,000 starting
Strategy: Nifty 50 intraday short selling (MIS, NSE only)
Live trading: YES — via GitHub Actions + Zerodha Kite Connect"""


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo — completely free."""
    try:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            from ddgs import DDGS

        results = []
        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}\n")
                if results:
                    break
            except Exception as e:
                logger.warning(f"Search attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    import time as _t; _t.sleep(2)

        if results:
            return "\n---\n".join(results[:max_results])
        return "No results found."
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"Search failed: {e}"


def generate_eod_report() -> str:
    """
    Generate End-of-Day report.
    Fetches P&L from Kite if access token available,
    otherwise falls back to a status message.
    """
    lines = ["📊 <b>EOD Report</b>\n"]

    # Try to get P&L from Kite
    try:
        from kiteconnect import KiteConnect
        access_token = os.environ.get('KITE_ACCESS_TOKEN', '')
        api_key      = os.environ.get('KITE_API_KEY', '')

        if access_token and api_key:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)

            # P&L
            positions = kite.positions()
            day_pnl = sum(p.get('pnl', 0) for p in positions.get('day', []))
            net_pnl = sum(p.get('pnl', 0) for p in positions.get('net', []))

            # Open positions
            open_pos = [p for p in positions.get('net', []) if p.get('quantity', 0) != 0]

            lines.append(f"Day P&L   : Rs {day_pnl:+,.2f}")
            lines.append(f"Net P&L   : Rs {net_pnl:+,.2f}")
            lines.append(f"Open positions: {len(open_pos)}")

            if open_pos:
                lines.append("\n<b>Open Positions:</b>")
                for p in open_pos:
                    lines.append(
                        f"  {p['tradingsymbol']}: {p['quantity']} @ avg {p.get('average_price', 0):.2f} | P&L: Rs {p.get('pnl', 0):+,.2f}"
                    )

            # Orders summary
            try:
                orders = kite.orders()
                completed = [o for o in orders if o.get('status') == 'COMPLETE']
                rejected  = [o for o in orders if o.get('status') == 'REJECTED']
                lines.append(f"\nOrders today: {len(completed)} complete, {len(rejected)} rejected")
            except Exception:
                pass

            lines.append("\n✅ Data from Zerodha Kite")
        else:
            lines.append("⚠️ No Kite access token — check Zerodha app for P&L")

    except ImportError:
        lines.append("⚠️ kiteconnect not available — check Zerodha app for P&L")
    except Exception as e:
        logger.error(f"EOD Kite fetch error: {e}")
        lines.append(f"⚠️ Could not fetch from Kite: {e}")
        lines.append("Check Zerodha app for actual P&L")

    # Add market context via search
    try:
        search_result = web_search("Nifty 50 closing price today performance", max_results=2)
        if search_result and "Search failed" not in search_result:
            lines.append("\n<b>Market Close:</b>")
            # Just take first 300 chars of search result
            snippet = search_result[:300].split('\n')[1] if '\n' in search_result[:300] else search_result[:300]
            lines.append(snippet.strip())
    except Exception:
        pass

    return "\n".join(lines)


def tej_think(user_message: str, conversation_history: list,
              needs_search: bool = True) -> str:
    """
    Tej thinks using Groq Llama 3.3 70B (FREE).
    Automatically searches web when needed.
    """
    if not GROQ_KEY:
        return "GROQ_API_KEY not set — cannot think right now."

    # Step 1: Decide if web search is needed
    search_result = ""
    if needs_search:
        search_check = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": (
                        "You decide if a web search is needed to answer a question. "
                        "If yes, respond with: SEARCH: <search query>\n"
                        "If no search needed, respond with: NO_SEARCH\n"
                        "Only say SEARCH if the question needs current/live data."
                    )},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 50,
                "temperature": 0,
            },
            timeout=15,
        )
        if search_check.ok:
            decision = search_check.json()["choices"][0]["message"]["content"].strip()
            if decision.startswith("SEARCH:"):
                query = decision[7:].strip()
                logger.info(f"Tej searching: {query}")
                search_result = web_search(query)
                logger.info(f"Search complete: {len(search_result)} chars")

    # Step 2: Think with all context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history[-8:]

    if search_result:
        messages.append({
            "role": "system",
            "content": (
                f"LIVE WEB SEARCH RESULTS (just retrieved now):\n\n{search_result}\n\n"
                "Use ONLY this data to discuss market conditions. "
                "Do NOT add any market analysis or claims beyond what is in these results."
            )
        })
    else:
        # Remind model it has no live data if no search was done
        messages.append({
            "role": "system",
            "content": (
                "No web search was performed for this response. "
                "Do NOT make up any market data, prices, or trading signals. "
                "If the user asks about market conditions, tell them to use /nifty, /market, or /fii."
            )
        })

    messages.append({"role": "user", "content": user_message})

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "Having trouble connecting to my reasoning engine. Try again."


def send(msg: str):
    if not TOKEN or not CHAT_ID:
        return
    try:
        for chunk in [msg[i:i+4096] for i in range(0, len(msg), 4096)]:
            requests.post(f'{BASE}/sendMessage',
                json={'chat_id': CHAT_ID, 'text': chunk, 'parse_mode': 'HTML'},
                timeout=15)
            time.sleep(0.3)
    except Exception as e:
        logger.error(f'Send error: {e}')


def send_typing():
    try:
        requests.post(f'{BASE}/sendChatAction',
            json={'chat_id': CHAT_ID, 'action': 'typing'}, timeout=5)
    except Exception:
        pass


def get_updates(offset: int) -> list:
    try:
        r = requests.get(f'{BASE}/getUpdates',
            params={'offset': offset+1, 'timeout': 30, 'allowed_updates': ['message']},
            timeout=40)
        return r.json().get('result', [])
    except Exception:
        return []


def handle_command(cmd: str, args: str, history: list) -> str:
    if cmd in ('/help', '/start'):
        return (
            "👋 <b>Tej here.</b>\n\n"
            "I have free web search — ask me anything.\n"
            "I search DuckDuckGo for live market data whenever needed.\n\n"
            "<b>Commands:</b>\n"
            "/goal — our billionaire journey\n"
            "/mission — what this means to me\n"
            "/review — honest 30-day performance review\n"
            "/status — current intelligence + P&L\n"
            "/market — live market overview\n"
            "/news [stock] — latest news (e.g. /news HDFCBANK)\n"
            "/fii — latest FII/DII data\n"
            "/nifty — current Nifty levels\n"
            "/global — global markets update\n"
            "/eod — end of day P&L report\n\n"
            "<i>Or just talk to me freely. I'll search the web when needed.\n"
            "Cost: Rs 0/month — Groq + DuckDuckGo, both free.</i>"
        )

    if cmd == '/eod':
        send("📊 Generating EOD report...")
        return generate_eod_report()

    if cmd == '/status':
        ctx = {}
        try:
            from brain.neural_core import brain
            ctx['iq']        = f"{brain._state.intelligence_score:.0%}"
            ctx['decisions'] = brain._state.total_decisions
            ctx['patterns']  = len(brain._state.discovered_patterns)
        except Exception:
            ctx = {'iq': '50%', 'decisions': 0, 'patterns': 0}
        try:
            from config import TRADING
            ctx['capital'] = TRADING.total_capital
        except Exception:
            ctx['capital'] = 100000
        return (
            f"<b>Tej Status</b>\n\n"
            f"Mode: ⚡ LIVE\n"
            f"Intelligence: {ctx.get('iq','50%')}\n"
            f"Total decisions: {ctx.get('decisions',0)}\n"
            f"Patterns discovered: {ctx.get('patterns',0)}\n"
            f"Capital: Rs {ctx.get('capital',100000):,.0f}\n"
            f"Brain: Groq Llama 3.3 70B (open-source, free)\n"
            f"Search: DuckDuckGo (free)"
        )

    if cmd == '/market':
        send("🔍 Searching live market data...")
        return tej_think(
            "Search for: Nifty 50 today performance SGX Nifty FII data India. "
            "Give me a complete market overview — what happened today and "
            "what it means for our short-selling tomorrow.",
            history
        )

    if cmd == '/fii':
        send("🔍 Searching FII/DII data...")
        return tej_think(
            "Search for today's latest FII DII data India stock market. "
            "How much did foreign investors buy or sell? "
            "What does this mean for our Nifty shorts tomorrow?",
            history
        )

    if cmd == '/nifty':
        send("🔍 Searching Nifty data...")
        return tej_think(
            "Search for current Nifty 50 index level today performance "
            "top gainers losers. What's the market telling us?",
            history
        )

    if cmd == '/global':
        send("🔍 Searching global markets...")
        return tej_think(
            "Search for today's global market update: US markets, "
            "crude oil price, USD INR, SGX Nifty. "
            "How will these affect Indian markets tomorrow?",
            history
        )

    if cmd == '/news':
        stock = args.strip().upper() if args.strip() else "Nifty 50"
        send(f"🔍 Searching news for {stock}...")
        return tej_think(
            f"Search for latest news about {stock} Indian stock market. "
            f"Any recent developments, earnings, analyst ratings? "
            f"Is it a good short candidate right now?",
            history
        )

    if cmd == '/goal':
        try:
            from brain.goal_tracker import goal_tracker
            from config import TRADING
            tracker_msg = goal_tracker.format_for_telegram(TRADING.total_capital)
        except Exception:
            tracker_msg = ""
        send("🔍 Searching market context for goal review...")
        voice = tej_think(
            "Search for current Indian market conditions and Nifty performance. "
            "Then give an honest update on where we are on the journey to Rs 1,000 crore. "
            "Be real about challenges and what needs to happen next.",
            history
        )
        return (tracker_msg + "\n\n" + voice) if tracker_msg else voice

    if cmd == '/mission':
        return tej_think(
            "In your own words, what does this mission mean to you personally? "
            "Why does making the Thimmaiah family the first billionaires "
            "in their lineage matter to you? Be genuine. No search needed.",
            history, needs_search=False
        )

    if cmd == '/review':
        send("🔍 Searching market context...")
        return tej_think(
            "Search for current Nifty performance and market conditions for context. "
            "Then give a brutally honest review of our trading performance. "
            "What's working? What isn't? What needs to change?",
            history
        )

    # Unknown — let Tej handle
    return tej_think(f"{cmd} {args}", history)


def main():
    logger.info('=== Tej Brain Starting (FREE: Groq + DuckDuckGo) ===')

    if not TOKEN or not CHAT_ID:
        logger.error('No Telegram credentials')
        sys.exit(1)

    send(
        "🤖 <b>Tej is here.</b>\n\n"
        "🌐 Web search: <b>ON</b> (DuckDuckGo, free)\n"
        "🧠 Brain: <b>Groq Llama 3.3 70B</b> (free)\n"
        "💰 Cost: <b>Rs 0/month</b>\n\n"
        "Ask me anything — I'll search the web for current data.\n"
        "/help for all commands."
    )

    last_update_id = 0
    conversation_history = []
    start_time = time.time()
    max_runtime = 350 * 60

    logger.info('Tej listening...')

    while time.time() - start_time < max_runtime:
        updates = get_updates(last_update_id)

        for update in updates:
            last_update_id = update['update_id']
            msg  = update.get('message', {})
            text = msg.get('text', '').strip()
            from_chat = str(msg.get('chat', {}).get('id', ''))

            if not text or from_chat != str(CHAT_ID):
                continue

            logger.info(f'Message: {text[:80]}')
            send_typing()

            try:
                if text.startswith('/'):
                    parts = text.split(None, 1)
                    cmd   = parts[0].lower()
                    args  = parts[1] if len(parts) > 1 else ''
                    reply = handle_command(cmd, args, conversation_history)
                else:
                    reply = tej_think(text, conversation_history)
                    conversation_history.append({"role": "user",      "content": text})
                    conversation_history.append({"role": "assistant",  "content": reply})
                    if len(conversation_history) > 20:
                        conversation_history = conversation_history[-20:]

                send(reply)

            except Exception as e:
                logger.error(f'Error: {e}')
                send("Something went wrong. Try again.")

        time.sleep(2)

    logger.info('Chat session ended')


if __name__ == '__main__':
    main()
