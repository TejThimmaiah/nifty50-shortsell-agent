import os, sys, time, logging, threading, requests
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
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def web_search(query, max_results=5):
    results = []
    try:
        from googlesearch import search as gsearch
        import re
        for url in list(gsearch(query, num_results=max_results, lang="en"))[:max_results]:
            try:
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                text = resp.text[:3000]
                desc = re.search(r'<meta[^>]*name="description"[^>]*content="([^"]*)"', text)
                body = desc.group(1) if desc else ""
                if not body:
                    p = re.search(r'<p[^>]*>(.{50,300})</p>', text)
                    body = re.sub(r'<[^>]+>', '', p.group(1)) if p else url
                results.append({"title": url.split("/")[2], "body": body, "href": url})
            except Exception:
                results.append({"title": url, "body": url, "href": url})
        if results: return results
    except ImportError: pass
    except Exception as e: logger.warning(f"Google failed: {e}")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs: results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e: logger.warning(f"DDG failed: {e}")
    return results

def get_tv_nifty50_data():
    try:
        from tradingview_screener import Query
        _, df = (Query().select('name','close','change','change_abs','volume','RSI','EMA20','SMA50','market_cap_basic','Perf.W','Perf.1M','ADX','MACD.macd','BB.upper','BB.lower')
            .where(Query.Column('exchange')=='NSE').where(Query.Column('is_primary')==True)
            .order_by('market_cap_basic',ascending=False).limit(50).get_scanner_data())
        return df
    except Exception as e: logger.warning(f"TV error: {e}"); return None

def get_tv_stock_price(symbol):
    try:
        from tradingview_screener import Query
        _, df = (Query().select('name','close','change','change_abs','volume','RSI','high','low','open','Perf.W','Perf.1M','ADX','MACD.macd','BB.upper','BB.lower','ATR','EMA20','SMA50')
            .where(Query.Column('exchange')=='NSE').where(Query.Column('name')==symbol.upper()).get_scanner_data())
        return df.iloc[0].to_dict() if len(df)>0 else None
    except Exception as e: logger.warning(f"TV stock error: {e}"); return None

def get_tv_nifty_index():
    try:
        from tvDatafeed import TvDatafeed, Interval
        data = TvDatafeed().get_hist(symbol='NIFTY',exchange='NSE',interval=Interval.in_daily,n_bars=2)
        if data is not None and len(data)>0:
            l,p = data.iloc[-1], data.iloc[-2] if len(data)>1 else data.iloc[-1]
            ch = l['close']-p['close']
            return {"close":l['close'],"open":l['open'],"high":l['high'],"low":l['low'],"change":ch,"change_pct":(ch/p['close'])*100}
    except Exception as e: logger.warning(f"TvDatafeed error: {e}")
    return None

def fetch_tv_nifty():
    try:
        idx = get_tv_nifty_index()
        lines = ["📈 <b>Nifty 50 — TradingView</b>\n"]
        if idx:
            s = "🟢" if idx['change']>=0 else "🔴"
            lines.append(f"{s} <b>{idx['close']:,.2f}</b> ({idx['change']:+,.2f} | {idx['change_pct']:+.2f}%)")
            lines.append(f"O:{idx['open']:,.2f} H:{idx['high']:,.2f} L:{idx['low']:,.2f}")
        df = get_tv_nifty50_data()
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
    except Exception as e: send_telegram(f"⚠️ TV error: {e}")

def fetch_tv_stock(symbol):
    try:
        d = get_tv_stock_price(symbol)
        if not d: send_telegram(f"❌ {symbol} not found on NSE"); return
        c,ch,rsi = d.get('close',0),d.get('change',0),d.get('RSI',0)
        s = "🟢" if ch>=0 else "🔴"
        lines = [f"📊 <b>{d.get('name',symbol)}</b>\n",f"{s} <b>₹{c:,.2f}</b> ({ch:+.2f}%)",
            f"O:{d.get('open',0):,.2f} H:{d.get('high',0):,.2f} L:{d.get('low',0):,.2f}",f"Vol:{d.get('volume',0):,.0f}",
            f"\nRSI:{rsi:.1f}{'⚠️ OVERBOUGHT' if rsi>70 else ''}", f"ADX:{d.get('ADX',0):.1f} MACD:{d.get('MACD.macd',0):.2f}",
            f"EMA20:₹{d.get('EMA20',0):,.2f} SMA50:₹{d.get('SMA50',0):,.2f}"]
        sigs = []
        if rsi>70: sigs.append("RSI overbought")
        if c>d.get('BB.upper',c+1): sigs.append("Above BB")
        if d.get('MACD.macd',0)<0: sigs.append("MACD bearish")
        lines.append(f"\n🎯 Shorts: {', '.join(sigs)}" if sigs else "\n📋 No short signals")
        send_telegram("\n".join(lines))
    except Exception as e: send_telegram(f"⚠️ Error: {e}")

def fetch_tv_screener():
    try:
        df = get_tv_nifty50_data()
        if df is None or len(df)==0: send_telegram("⚠️ Screener unavailable"); return
        lines = ["🔍 <b>Short Screener</b>\n"]
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
    except Exception as e: send_telegram(f"⚠️ Screener error: {e}")

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
        if not at: return None,"❌ No Kite token. Check .env"
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
        uid,pw,totp_s,api_s = [os.environ.get(k,"") for k in ["KITE_USER_ID","KITE_PASSWORD","KITE_TOTP_SECRET","KITE_API_SECRET"]]
        if not all([uid,pw,totp_s,api_s]): return None
        opts = Options()
        for a in ["--headless=new","--no-sandbox","--disable-dev-shm-usage","--disable-gpu","--window-size=1280,800"]: opts.add_argument(a)
        opts.binary_location = "/usr/bin/chromium-browser"
        d = webdriver.Chrome(service=Service("/usr/bin/chromedriver"),options=opts)
        w = WebDriverWait(d,15)
        kite = KiteConnect(api_key=api_key); d.get(kite.login_url())
        w.until(EC.presence_of_element_located((By.ID,"userid"))).send_keys(uid)
        d.find_element(By.ID,"password").send_keys(pw)
        d.find_element(By.XPATH,"//button[@type='submit']").click()
        _t.sleep(2)
        tf = w.until(EC.presence_of_element_located((By.XPATH,"//input[@type='number' or @inputmode='numeric' or contains(@placeholder,'TOTP')]")))
        tf.clear(); tf.send_keys(pyotp.TOTP(totp_s).now()); _t.sleep(3)
        url = d.current_url; d.quit()
        from urllib.parse import urlparse,parse_qs
        rt = parse_qs(urlparse(url).query).get("request_token",[None])[0]
        return kite.generate_session(rt,api_secret=api_s)["access_token"] if rt else None
    except Exception as e: logger.error(f"Login failed: {e}"); return None

def fetch_capital():
    kite,err = get_kite()
    if err: send_telegram(err); return
    try:
        m = kite.margins(segment="equity")
        a = m.get("net",m.get("available",{}).get("live_balance",0))
        send_telegram(f"💰 <b>Zerodha</b>\n✅ ₹{a:,.2f}\n🎯 LIVE SHORT MIS")
    except Exception as e: send_telegram(f"⚠️ {e}")

def fetch_pnl():
    kite,err = get_kite()
    if err: send_telegram(err); return
    try:
        orders = kite.orders(); today = __import__('datetime').date.today().isoformat()
        pos = kite.positions().get("day",[])
        pnl = sum(p.get("pnl",0) for p in pos)
        trades = len([o for o in orders if o.get("status")=="COMPLETE" and o.get("order_timestamp","").startswith(today)])
        op = [p for p in pos if p.get("quantity",0)!=0]
        lines = [f"📊 <b>P&L</b>\n📅 {today}\n💰 ₹{pnl:+,.2f}\n🔢 {trades} trades"]
        if op:
            for p in op: lines.append(f"  {p['tradingsymbol']}: {p['quantity']} ₹{p.get('pnl',0):+,.2f}")
        else: lines.append("✅ No open positions")
        send_telegram("\n".join(lines))
    except Exception as e: send_telegram(f"⚠️ {e}")

def handle_code_command(text):
    try:
        from agents.code_agent import handle_code_command as _h; _h(text)
    except Exception as e: send_telegram(f"⚠️ {e}")

def research_and_reply(query):
    try:
        results = web_search(query,5)
        if not results: send_telegram(f"⚠️ No results: {query}"); return
        ctx = "\n".join([r.get("body","") for r in results[:5]])
        src = "\n".join([f"• {r.get('href','')}" for r in results[:3]])
        gk = os.environ.get("GROQ_API_KEY","")
        if not gk: send_telegram(f"🔍 <b>{query}</b>\n\n{ctx[:2000]}"); return
        import groq
        r = groq.Groq(api_key=gk).chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":f"Summarise for Indian trader:\nQuery: {query}\n{ctx}"}],max_tokens=400)
        send_telegram(f"🔍 <b>{query}</b>\n\n{r.choices[0].message.content}\n\n<i>Sources:</i>\n{src}")
    except Exception as e: send_telegram(f"⚠️ Research: {e}")

def ai_respond(text):
    try:
        gk = os.environ.get("GROQ_API_KEY","")
        if not gk: send_telegram("⚠️ No GROQ key"); return
        sc = ""
        try:
            import groq
            d = groq.Groq(api_key=gk).chat.completions.create(model="llama-3.3-70b-versatile",
                messages=[{"role":"system","content":"If search needed: SEARCH: <q>. Else: NO_SEARCH"},{"role":"user","content":text}],max_tokens=50).choices[0].message.content.strip()
            if d.startswith("SEARCH:"):
                r = web_search(d[7:].strip(),3)
                sc = "\n".join([x.get("body","") for x in r])
        except: pass
        import groq
        uc = text+(f"\n\n[Search]\n{sc}" if sc else "")
        r = groq.Groq(api_key=gk).chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":"You are Tej, AI trading agent. Be direct, honest, <200 words. NEVER claim to trade from chat."},
                      {"role":"user","content":uc}],max_tokens=400)
        send_telegram(r.choices[0].message.content)
    except Exception as e: send_telegram(f"⚠️ AI: {e}")

def handle_command(text):
    text = text.strip(); tl = text.lower(); now = datetime.now(IST).strftime('%H:%M:%S IST')
    if tl=='/status': send_telegram(f"🤖 <b>Tej</b>\n🕐 {now}\n✅ GCP\n🔍 Google+DDG\n📊 TradingView\n🧠 Groq Llama")
    elif tl=='/pnl': send_telegram("🔄 Fetching P&L..."); threading.Thread(target=fetch_pnl,daemon=True).start()
    elif tl in('/capital','/funds','/balance'): send_telegram("🔄 Fetching balance..."); threading.Thread(target=fetch_capital,daemon=True).start()
    elif tl=='/help': send_telegram("📋 <b>Tej</b>\n\n<b>💰</b> /capital /pnl /status\n<b>📊 TradingView</b>\n/nifty /price SYMBOL /scan\n<b>🔍 Research</b>\n/market /news /fii /global\n/research [q]\n<b>🛠️</b> /code /help\n\n<i>Or chat freely!</i>")
    elif tl=='/nifty': send_telegram("📊 Fetching Nifty..."); threading.Thread(target=fetch_tv_nifty,daemon=True).start()
    elif tl.startswith('/price'):
        sym = text[6:].strip().upper()
        if not sym: send_telegram("/price RELIANCE"); return
        send_telegram(f"📊 {sym}..."); threading.Thread(target=fetch_tv_stock,args=(sym,),daemon=True).start()
    elif tl=='/scan': send_telegram("🔍 Scanning..."); threading.Thread(target=fetch_tv_screener,daemon=True).start()
    elif tl=='/market': send_telegram("🔍 Market..."); threading.Thread(target=research_and_reply,args=("Nifty 50 today India market FII DII",),daemon=True).start()
    elif tl=='/fii': send_telegram("🔍 FII/DII..."); threading.Thread(target=research_and_reply,args=("FII DII today India",),daemon=True).start()
    elif tl=='/global': send_telegram("🔍 Global..."); threading.Thread(target=research_and_reply,args=("US markets crude USD INR SGX Nifty",),daemon=True).start()
    elif tl.startswith('/news'):
        st = text[5:].strip().upper() or "Nifty 50"
        send_telegram(f"🔍 {st}..."); threading.Thread(target=research_and_reply,args=(f"news {st} India stock",),daemon=True).start()
    elif tl.startswith('/research '): send_telegram("🔵 Researching..."); threading.Thread(target=research_and_reply,args=(text[10:],),daemon=True).start()
    elif tl.startswith('/code'): handle_code_command(text)
    elif text.startswith('/'): send_telegram(f"❓ <code>{text}</code>\n/help")
    else: threading.Thread(target=ai_respond,args=(text,),daemon=True).start()

def morning_briefing():
    while True:
        now = datetime.now(IST)
        if now.weekday()<5 and now.hour==8 and now.minute==30:
            send_telegram("🌅 <b>Good Morning!</b>\n/nifty /scan /help")
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
    send_telegram("🚀 <b>Tej v2.0</b>\n🔍 Google+DDG\n📊 TradingView\n🧠 Groq\n💰 Kite\n/help")
    threading.Thread(target=morning_briefing,daemon=True).start()
    poll_telegram()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); start()
