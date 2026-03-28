import os, sys, time, logging, threading, requests
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Fix working directory so imports resolve
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
load_dotenv()

IST = pytz.timezone("Asia/Kolkata")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
logger = logging.getLogger(__name__)

def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def get_kite():
    """Return authenticated Kite instance using saved access token."""
    try:
        from kiteconnect import KiteConnect
        api_key = os.environ.get("KITE_API_KEY", "")
        token_file = os.path.join(os.getcwd(), "kite_access_token.txt")
        if not os.path.exists(token_file):
            return None, "❌ No access token file found. Run morning prep first."
        access_token = open(token_file).read().strip()
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return kite, None
    except Exception as e:
        return None, f"❌ Kite init error: {e}"

def fetch_capital():
    kite, err = get_kite()
    if err:
        send_telegram(err)
        return
    try:
        margins = kite.margins(segment="equity")
        available = margins.get("net", margins.get("available", {}).get("live_balance", 0))
        used = margins.get("utilised", {}).get("debits", 0)
        total = margins.get("available", {}).get("opening_balance", available)
        send_telegram(
            f"💰 <b>Zerodha Account Balance</b>\n\n"
            f"✅ Available: ₹{available:,.2f}\n"
            f"📊 Used Margin: ₹{used:,.2f}\n"
            f"🏦 Opening Balance: ₹{total:,.2f}\n\n"
            f"🎯 Mode: LIVE SHORT-ONLY MIS"
        )
    except Exception as e:
        send_telegram(f"⚠️ Could not fetch capital: {e}")

def handle_code_command(text):
    try:
        from agents.code_agent import handle_code_command as _hcc
        _hcc(text)
    except ImportError:
        send_telegram("⚠️ Code agent not available. Check agents/code_agent.py exists.")
    except Exception as e:
        send_telegram(f"⚠️ Code agent error: {e}")

def research_and_reply(query):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        context = "\n".join([r.get("body","") for r in results])
        import groq
        client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY",""))
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":f"Summarise in 3 bullet points for Indian stock trader:\nQuery: {query}\nResults:\n{context}"}],
            max_tokens=300
        )
        send_telegram(f"🔍 <b>Research: {query}</b>\n\n{resp.choices[0].message.content}")
    except Exception as e:
        send_telegram(f"⚠️ Research error: {e}")

def handle_command(text):
    text = text.strip()
    tl = text.lower()
    now = datetime.now(IST).strftime('%H:%M:%S IST')
    if tl == '/status':
        send_telegram(
            f"🤖 <b>Tej Status</b>\n"
            f"🕐 {now}\n"
            f"✅ Agent running on GCP\n"
            f"📍 IP: 35.222.173.227\n"
            f"🟢 Mode: LIVE SHORT-ONLY"
        )
    elif tl == '/pnl':
        send_telegram("📊 No trades yet — first trade day is Monday 9:20 AM IST!")
    elif tl in ('/capital', '/funds', '/balance'):
        send_telegram("🔄 Fetching live balance from Zerodha...")
        threading.Thread(target=fetch_capital, daemon=True).start()
    elif tl == '/help':
        send_telegram(
            "📋 <b>Tej Commands:</b>\n\n"
            "/status — Agent status & IP\n"
            "/capital — Live Zerodha balance\n"
            "/funds — Same as /capital\n"
            "/pnl — Today's P&L\n"
            "/research [query] — Market research\n"
            "/code [cmd] — Code agent\n"
            "/help — This menu"
        )
    elif tl.startswith('/research '):
        query = text[10:]
        send_telegram(f"🔵 Researching: {query}...")
        threading.Thread(target=research_and_reply, args=(query,), daemon=True).start()
    elif tl.startswith('/code'):
        handle_code_command(text)
    else:
        send_telegram(f"❓ Unknown command: <code>{text}</code>\nType /help for available commands.")

def morning_briefing():
    while True:
        now = datetime.now(IST)
        if now.weekday() < 5 and now.hour == 8 and now.minute == 30:
            send_telegram(
                "🌅 <b>Good Morning! Trading Day Briefing</b>\n\n"
                "⏰ Market opens in 50 mins\n"
                "📋 Tej scans Nifty50 at 9:20 AM\n"
                "🎯 Strategy: SHORT-ONLY MIS\n"
                "💰 IP: 35.222.173.227 ✅ SEBI Compliant\n\n"
                "Type /capital to check your balance\nType /help for commands"
            )
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
