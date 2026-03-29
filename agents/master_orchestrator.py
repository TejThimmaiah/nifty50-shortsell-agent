import os, sys, time, logging, threading, requests, json
from datetime import datetime
import pytz
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
load_dotenv()

IST = pytz.timezone("Asia/Kolkata")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
logger = logging.getLogger(__name__)

def send_telegram(msg):
    try:
        for i in range(0, len(msg), 4000):
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": msg[i:i+4000], "parse_mode": "HTML"}, timeout=10)
    except Exception as e: logger.error(f"TG error: {e}")

# ── SEARCH: Google → Tavily → DuckDuckGo ────────────────────
def web_search(query, max_results=5):
    # Try Tavily first (best for AI)
    try:
        tk = os.environ.get("TAVILY_API_KEY","")
        if tk:
            from tavily import TavilyClient
            tv = TavilyClient(api_key=tk)
            r = tv.search(query, max_results=max_results)
            results = [{"title": x.get("title",""), "body": x.get("content",""), "href": x.get("url","")} for x in r.get("results",[])]
            if results: return results
    except Exception as e: logger.warning(f"Tavily failed: {e}")
    # Try Google
    try:
        from googlesearch import search as gsearch
        import re
        results = []
        for url in list(gsearch(query, num_results=max_results, lang="en"))[:max_results]:
            try:
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                desc = re.search(r'<meta[^>]*name="description"[^>]*content="([^"]*)"', resp.text[:3000])
                body = desc.group(1) if desc else url
                results.append({"title": url.split("/")[2], "body": body, "href": url})
            except: results.append({"title": url, "body": url, "href": url})
        if results: return results
    except: pass
    # Fallback DuckDuckGo
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d: return list(d.text(query, max_results=max_results))
    except: pass
    return []

# ── AI: Groq → Gemini → OpenRouter → HuggingFace ────────────
def ai_think(prompt, system="You are Tej, a helpful AI assistant. Be concise.", max_tokens=500):
    # Try Groq (fastest)
    try:
        gk = os.environ.get("GROQ_API_KEY","")
        if gk:
            import groq
            r = groq.Groq(api_key=gk).chat.completions.create(model="llama-3.3-70b-versatile",
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}], max_tokens=max_tokens)
            return r.choices[0].message.content
    except Exception as e: logger.warning(f"Groq failed: {e}")
    # Try Gemini
    try:
        gk = os.environ.get("GEMINI_API_KEY","")
        if gk:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gk}",
                json={"contents":[{"parts":[{"text":f"{system}\n\n{prompt}"}]}]}, timeout=30)
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e: logger.warning(f"Gemini failed: {e}")
    # Try OpenRouter
    try:
        ok = os.environ.get("OPENROUTER_API_KEY","")
        if ok:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {ok}"},
                json={"model":"meta-llama/llama-3.3-70b-instruct:free","messages":[{"role":"system","content":system},{"role":"user","content":prompt}],"max_tokens":max_tokens}, timeout=30)
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e: logger.warning(f"OpenRouter failed: {e}")
    # Try HuggingFace
    try:
        hk = os.environ.get("HF_API_KEY","")
        if hk:
            r = requests.post("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
                headers={"Authorization": f"Bearer {hk}"},
                json={"inputs": f"<s>[INST] {system}\n{prompt} [/INST]", "parameters":{"max_new_tokens":max_tokens}}, timeout=30)
            return r.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except Exception as e: logger.warning(f"HF failed: {e}")
    return "All AI engines are unavailable right now. Try again later."

# ── MARKET DATA ──────────────────────────────────────────────
def get_stock_yahoo(symbol):
    try:
        import yfinance as yf
        tk = yf.Ticker(f"{symbol}.NS")
        info = tk.info
        return {"name": info.get("shortName", symbol), "price": info.get("currentPrice", info.get("regularMarketPrice",0)),
            "change": info.get("regularMarketChangePercent",0), "open": info.get("regularMarketOpen",0),
            "high": info.get("regularMarketDayHigh",0), "low": info.get("regularMarketDayLow",0),
            "volume": info.get("regularMarketVolume",0), "pe": info.get("trailingPE",0),
            "52wh": info.get("fiftyTwoWeekHigh",0), "52wl": info.get("fiftyTwoWeekLow",0),
            "div_yield": info.get("dividendYield",0), "market_cap": info.get("marketCap",0),
            "sector": info.get("sector",""), "prev_close": info.get("regularMarketPreviousClose",0)}
    except Exception as e: logger.warning(f"Yahoo error: {e}"); return None

def get_finnhub_news(symbol="", category="general"):
    try:
        fk = os.environ.get("FINNHUB_API_KEY","")
        if not fk: return []
        if symbol:
            r = requests.get(f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2026-03-22&to=2026-03-29&token={fk}", timeout=10)
        else:
            r = requests.get(f"https://finnhub.io/api/v1/news?category={category}&token={fk}", timeout=10)
        news = r.json()[:5]
        return [{"title": n.get("headline",""), "summary": n.get("summary","")[:200], "source": n.get("source",""), "url": n.get("url","")} for n in news]
    except: return []

def get_tv_nifty50():
    try:
        from tradingview_screener import Query
        _, df = (Query().select('name','close','change','volume','RSI','EMA20','SMA50','market_cap_basic','ADX','MACD.macd','BB.upper','BB.lower')
            .where(Query.Column('exchange')=='NSE').where(Query.Column('is_primary')==True)
            .order_by('market_cap_basic',ascending=False).limit(50).get_scanner_data())
        return df
    except Exception as e: logger.warning(f"TV error: {e}"); return None

def get_tv_nifty_index():
    try:
        from tvDatafeed import TvDatafeed, Interval
        data = TvDatafeed().get_hist(symbol='NIFTY',exchange='NSE',interval=Interval.in_daily,n_bars=2)
        if data is not None and len(data)>0:
            l,p = data.iloc[-1], data.iloc[-2] if len(data)>1 else data.iloc[-1]
            ch = l['close']-p['close']
            return {"close":l['close'],"open":l['open'],"high":l['high'],"low":l['low'],"change":ch,"pct":(ch/p['close'])*100}
    except: pass
    return None

# ── COMMAND HANDLERS ─────────────────────────────────────────
def fetch_price(symbol):
    data = get_stock_yahoo(symbol)
    if not data or not data['price']:
        send_telegram(f"❌ {symbol} not found. Try exact NSE symbol like RELIANCE, HDFCBANK, TCS")
        return
    d = data
    sign = "🟢" if d['change']>=0 else "🔴"
    mc = f"₹{d['market_cap']/1e7:,.0f} Cr" if d['market_cap'] else "N/A"
    lines = [f"📊 <b>{d['name']}</b> ({symbol})\n",
        f"{sign} <b>₹{d['price']:,.2f}</b> ({d['change']:+.2f}%)",
        f"O: ₹{d['open']:,.2f} | H: ₹{d['high']:,.2f} | L: ₹{d['low']:,.2f}",
        f"Vol: {d['volume']:,.0f} | Prev: ₹{d['prev_close']:,.2f}",
        f"\n📈 52W: ₹{d['52wl']:,.2f} — ₹{d['52wh']:,.2f}",
        f"PE: {d['pe']:.2f} | Div: {(d['div_yield'] or 0)*100:.2f}%",
        f"MCap: {mc} | Sector: {d['sector']}"]
    send_telegram("\n".join(lines))

def fetch_nifty():
    idx = get_tv_nifty_index()
    lines = ["📈 <b>Nifty 50 — Live</b>\n"]
    if idx:
        s = "🟢" if idx['change']>=0 else "🔴"
        lines.append(f"{s} <b>{idx['close']:,.2f}</b> ({idx['change']:+,.2f} | {idx['pct']:+.2f}%)")
        lines.append(f"O:{idx['open']:,.2f} H:{idx['high']:,.2f} L:{idx['low']:,.2f}")
    df = get_tv_nifty50()
    if df is not None and len(df)>0:
        lines.append("\n<b>Top Gainers:</b>")
        for _,r in df.nlargest(3,'change').iterrows(): lines.append(f"  🟢 {r['name']}: ₹{r['close']:,.2f} ({r['change']:+.2f}%)")
        lines.append("\n<b>Top Losers:</b>")
        for _,r in df.nsmallest(3,'change').iterrows(): lines.append(f"  🔴 {r['name']}: ₹{r['close']:,.2f} ({r['change']:+.2f}%)")
        if 'RSI' in df.columns:
            ob = df[df['RSI']>70]
            if len(ob)>0:
                lines.append(f"\n⚡ <b>Overbought RSI&gt;70:</b>")
                for _,r in ob.head(5).iterrows(): lines.append(f"  🎯 {r['name']}: RSI={r['RSI']:.1f} ₹{r['close']:,.2f}")
    send_telegram("\n".join(lines))

def fetch_scan():
    df = get_tv_nifty50()
    if df is None or len(df)==0: send_telegram("⚠️ Screener unavailable"); return
    lines = ["🔍 <b>Short Screener — TradingView</b>\n"]
    if 'RSI' in df.columns:
        ob = df[df['RSI']>65].sort_values('RSI',ascending=False)
        if len(ob)>0:
            lines.append("<b>⚡ Overbought RSI&gt;65:</b>")
            for _,r in ob.head(8).iterrows(): lines.append(f"  🎯 {r['name']}: RSI={r['RSI']:.1f} ₹{r['close']:,.2f} ({r['change']:+.2f}%)")
    lines.append("\n<b>📉 Biggest Losers:</b>")
    for _,r in df.nsmallest(5,'change').iterrows():
        rs = f"RSI={r['RSI']:.1f}" if 'RSI' in df.columns else ""
        lines.append(f"  🔴 {r['name']}: ₹{r['close']:,.2f} ({r['change']:+.2f}%) {rs}")
    send_telegram("\n".join(lines))

def fetch_news(topic):
    # Try Finnhub first
    news = get_finnhub_news(category="general")
    if news:
        lines = [f"📰 <b>Market News — Finnhub</b>\n"]
        for n in news: lines.append(f"• <b>{n['title']}</b>\n  {n['summary']}\n  <i>{n['source']}</i>")
        send_telegram("\n".join(lines))
        return
    # Fallback to web search
    research_and_reply(f"latest news {topic} India stock market")

def fetch_capital():
    kite, err = get_kite()
    if err: send_telegram(err); return
    try:
        m = kite.margins(segment="equity")
        a = m.get("net", m.get("available",{}).get("live_balance",0))
        u = m.get("utilised",{}).get("debits",0)
        t = m.get("available",{}).get("opening_balance",a)
        send_telegram(f"💰 <b>Zerodha</b>\n\n✅ Available: ₹{a:,.2f}\n📊 Used: ₹{u:,.2f}\n🏦 Opening: ₹{t:,.2f}\n\n🎯 LIVE SHORT-ONLY MIS")
    except Exception as e: send_telegram(f"⚠️ {e}")

def fetch_pnl():
    kite, err = get_kite()
    if err: send_telegram(err); return
    try:
        orders = kite.orders(); today = __import__('datetime').date.today().isoformat()
        pos = kite.positions().get("day",[])
        pnl = sum(p.get("pnl",0) for p in pos)
        trades = len([o for o in orders if o.get("status")=="COMPLETE" and o.get("order_timestamp","").startswith(today)])
        op = [p for p in pos if p.get("quantity",0)!=0]
        lines = [f"📊 <b>P&L</b>\n📅 {today}\n💰 ₹{pnl:+,.2f}\n🔢 {trades} trades"]
        for p in op: lines.append(f"  {p['tradingsymbol']}: {p['quantity']} ₹{p.get('pnl',0):+,.2f}")
        if not op: lines.append("✅ No open positions")
        send_telegram("\n".join(lines))
    except Exception as e: send_telegram(f"⚠️ {e}")

def get_kite():
    try:
        from kiteconnect import KiteConnect
        api_key = os.environ.get("KITE_API_KEY","")
        if not api_key: return None,"❌ KITE_API_KEY not set"
        tf = os.path.join(os.getcwd(),"kite_access_token.txt")
        at = None
        if os.path.exists(tf): at = open(tf).read().strip() or None
        if not at: at = os.environ.get("KITE_ACCESS_TOKEN","").strip() or None
        if not at:
            at = live_kite_login(api_key)
            if at: open(tf,"w").write(at)
        if not at: return None,"❌ No Kite token"
        kite = KiteConnect(api_key=api_key); kite.set_access_token(at); kite.profile()
        return kite, None
    except Exception as e: return None, f"❌ Kite: {e}"

def live_kite_login(api_key):
    try:
        import pyotp, time as _t
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from kiteconnect import KiteConnect
        uid,pw,ts,aps = [os.environ.get(k,"") for k in ["KITE_USER_ID","KITE_PASSWORD","KITE_TOTP_SECRET","KITE_API_SECRET"]]
        if not all([uid,pw,ts,aps]): return None
        opts = Options()
        for a in ["--headless=new","--no-sandbox","--disable-dev-shm-usage","--disable-gpu","--window-size=1280,800"]: opts.add_argument(a)
        opts.binary_location = "/usr/bin/chromium-browser"
        d = webdriver.Chrome(service=Service("/usr/bin/chromedriver"),options=opts); w = WebDriverWait(d,15)
        kite = KiteConnect(api_key=api_key); d.get(kite.login_url())
        w.until(EC.presence_of_element_located((By.ID,"userid"))).send_keys(uid)
        d.find_element(By.ID,"password").send_keys(pw)
        d.find_element(By.XPATH,"//button[@type='submit']").click(); _t.sleep(2)
        tf = w.until(EC.presence_of_element_located((By.XPATH,"//input[@type='number' or @inputmode='numeric' or contains(@placeholder,'TOTP')]")))
        tf.clear(); tf.send_keys(pyotp.TOTP(ts).now()); _t.sleep(3)
        url = d.current_url; d.quit()
        from urllib.parse import urlparse, parse_qs
        rt = parse_qs(urlparse(url).query).get("request_token",[None])[0]
        return kite.generate_session(rt, api_secret=aps)["access_token"] if rt else None
    except Exception as e: logger.error(f"Login failed: {e}"); return None

def research_and_reply(query):
    results = web_search(query, 5)
    if not results: send_telegram(f"⚠️ No results: {query}"); return
    ctx = "\n".join([r.get("body","") for r in results[:5]])
    src = "\n".join([f"• {r.get('href','')}" for r in results[:3]])
    reply = ai_think(f"Summarise for Indian trader:\nQuery: {query}\n{ctx}", max_tokens=400)
    send_telegram(f"🔍 <b>{query}</b>\n\n{reply}\n\n<i>Sources:</i>\n{src}")

def ai_respond(text):
    search_ctx = ""
    try:
        decision = ai_think(text, system="If web search needed reply SEARCH: <query>. Otherwise reply NO_SEARCH. Nothing else.", max_tokens=50)
        if decision.strip().startswith("SEARCH:"):
            results = web_search(decision[7:].strip(), 3)
            search_ctx = "\n".join([r.get("body","") for r in results])
    except: pass
    prompt = text + (f"\n\n[Search results]\n{search_ctx}" if search_ctx else "")
    system = "You are Tej, an AI assistant that can do ANYTHING: answer questions, write code, build websites, plan businesses, analyze markets. Be direct and helpful. Under 300 words. NEVER claim to execute trades."
    reply = ai_think(prompt, system=system, max_tokens=600)
    send_telegram(reply)

def handle_code_command(text):
    try:
        from agents.code_agent import handle_code_command as _h; _h(text)
    except Exception as e: send_telegram(f"⚠️ {e}")

# ── COMMAND ROUTER ───────────────────────────────────────────
def handle_command(text):
    text = text.strip(); tl = text.lower()
    now = datetime.now(IST).strftime('%H:%M IST')
    if tl=='/status':
        send_telegram(f"🤖 <b>Tej v3.0</b>\n🕐 {now}\n✅ GCP Live\n\n🧠 AI: Groq→Gemini→OpenRouter→HF\n🔍 Search: Tavily→Google→DDG\n📊 Data: TradingView+Yahoo+Finnhub\n💰 Trading: Zerodha Kite")
    elif tl=='/pnl': send_telegram("🔄..."); threading.Thread(target=fetch_pnl,daemon=True).start()
    elif tl in('/capital','/funds','/balance'): send_telegram("🔄..."); threading.Thread(target=fetch_capital,daemon=True).start()
    elif tl=='/help':
        send_telegram("📋 <b>Tej v3.0</b>\n\n<b>💰 Trading</b>\n/capital /pnl /status\n\n<b>📊 Live Data</b>\n/nifty — Index + movers\n/price SYMBOL — Full stock data\n/scan — Short screener\n\n<b>📰 News &amp; Research</b>\n/news — Market news (Finnhub)\n/market — Market overview\n/fii — FII/DII data\n/global — Global markets\n/research [query]\n\n<b>🛠️ Tools</b>\n/code — Code agent\n/help — Commands\n\n<i>Or just chat! I can write code, build websites, plan businesses, answer anything.</i>")
    elif tl=='/nifty': send_telegram("📊..."); threading.Thread(target=fetch_nifty,daemon=True).start()
    elif tl.startswith('/price'):
        sym = text[6:].strip().upper()
        if not sym: send_telegram("Usage: /price RELIANCE"); return
        send_telegram(f"📊 {sym}..."); threading.Thread(target=fetch_price,args=(sym,),daemon=True).start()
    elif tl=='/scan': send_telegram("🔍..."); threading.Thread(target=fetch_scan,daemon=True).start()
    elif tl.startswith('/news'):
        topic = text[5:].strip() or "market"
        send_telegram(f"📰..."); threading.Thread(target=fetch_news,args=(topic,),daemon=True).start()
    elif tl=='/market': send_telegram("🔍..."); threading.Thread(target=research_and_reply,args=("Nifty 50 today India market FII DII",),daemon=True).start()
    elif tl=='/fii': send_telegram("🔍..."); threading.Thread(target=research_and_reply,args=("FII DII today India stock market",),daemon=True).start()
    elif tl=='/global': send_telegram("🔍..."); threading.Thread(target=research_and_reply,args=("US markets crude oil USD INR SGX Nifty today",),daemon=True).start()
    elif tl.startswith('/research '): send_telegram("🔵..."); threading.Thread(target=research_and_reply,args=(text[10:],),daemon=True).start()
    elif tl.startswith('/code'): handle_code_command(text)
    elif text.startswith('/'): send_telegram(f"❓ <code>{text}</code>\n/help for commands")
    else: threading.Thread(target=ai_respond,args=(text,),daemon=True).start()

# ── MAIN ─────────────────────────────────────────────────────
def morning_briefing():
    while True:
        now = datetime.now(IST)
        if now.weekday()<5 and now.hour==8 and now.minute==30:
            send_telegram("🌅 <b>Good Morning!</b>\n/nifty for live data\n/scan for short candidates\n/help for commands")
            time.sleep(60)
        time.sleep(30)

def poll_telegram():
    offset = None
    while True:
        try:
            resp = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",params={"timeout":30,"offset":offset},timeout=35)
            for u in resp.json().get("result",[]):
                offset = u["update_id"]+1
                m = u.get("message",{}); t = m.get("text","")
                if t and str(m.get("chat",{}).get("id",""))==str(TELEGRAM_CHAT_ID): handle_command(t)
        except Exception as e: logger.error(f"Poll: {e}"); time.sleep(5)

def start():
    send_telegram("🚀 <b>Tej v3.0 — Ultimate Free AI</b>\n\n🧠 4 AI Engines (Groq+Gemini+OpenRouter+HF)\n🔍 3 Search (Tavily+Google+DDG)\n📊 6 Data (TV+Yahoo+Finnhub+AV+NSE+Kite)\n\n/help for commands\nOr just chat with me!")
    threading.Thread(target=morning_briefing,daemon=True).start()
    poll_telegram()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); start()
