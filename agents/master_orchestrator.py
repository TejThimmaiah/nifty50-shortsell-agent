import os, requests, threading, time, logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)
IST = pytz.timezone('Asia/Kolkata')

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def handle_command(text):
    text = text.strip().lower()
    if text == '/status':
        now = datetime.now(IST).strftime('%H:%M:%S IST')
        send_telegram(f"🤖 <b>Tej Status</b>\n⏰ {now}\n✅ Agent running on GCP\n📍 IP: 35.222.173.227\n💹 Mode: LIVE SHORT-ONLY")
    elif text == '/pnl':
        try:
            import sys; sys.path.insert(0, "."); from intelligence.trade_memory import TradeMemory
            tm = TradeMemory()
            recent = tm.get_recent_trades(5) if hasattr(tm, 'get_recent_trades') else []
            msg = "📊 <b>Recent Trades</b>\n"
            for t in recent:
                msg += f"• {t.get('symbol','?')} {t.get('pnl',0):+.0f}\n"
            send_telegram(msg or "No recent trades found.")
        except Exception as e:
            send_telegram(f"⚠️ PnL fetch error: {e}")
    elif text.startswith('/research '):
        query = text[10:]
        send_telegram(f"🔍 Researching: {query}...")
        threading.Thread(target=research_and_reply, args=(query,), daemon=True).start()
    elif text == '/help':
        send_telegram("🤖 <b>Tej Commands</b>\n/status - Agent status\n/pnl - Recent trades\n/research <query> - Web research\n/help - This menu")
    elif text.startswith('/code'):
        handle_code_command(text)
    else:
        send_telegram(f"❓ Unknown command. Type /help for options.")

def research_and_reply(query):
    try:
        from duckduckgo_search import DDGS
        from groq import Groq
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"{r['title']}: {r['body'][:200]}")
        context = "\n".join(results)
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Summarise this research in 3 bullet points for an Indian stock trader:\nQuery: {query}\nResults:\n{context}"}],
            max_tokens=300
        )
        summary = resp.choices[0].message.content
        send_telegram(f"🔍 <b>Research: {query}</b>\n\n{summary}")
    except Exception as e:
        send_telegram(f"⚠️ Research error: {e}")

def morning_briefing():
    while True:
        now = datetime.now(IST)
        if now.weekday() < 5 and now.hour == 8 and now.minute == 30:
            send_telegram("🌅 <b>Good Morning! Trading Day Briefing</b>\n\n⏰ Market opens in 50 mins\n📋 Tej will scan Nifty50 at 9:20 AM\n🎯 Strategy: SHORT-ONLY MIS\n💰 Static IP: 35.222.173.227 ✅ SEBI Compliant\n\nType /help for commands")
            time.sleep(60)
        time.sleep(30)

def poll_telegram():
    offset = None
    while True:
        try:
            params = {"timeout": 30, "offset": offset}
            resp = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates", params=params, timeout=35)
            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if text and chat_id == str(TELEGRAM_CHAT_ID):
                    handle_command(text)
        except Exception as e:
            logger.error(f"Poll error: {e}")
            time.sleep(5)

def start():
    send_telegram("🚀 <b>Master Orchestrator started!</b>\nType /help for commands.")
    threading.Thread(target=morning_briefing, daemon=True).start()
    poll_telegram()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start()

from agents.code_agent import handle_code_command
