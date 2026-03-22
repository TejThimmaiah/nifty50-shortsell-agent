# nifty50-shortsell-agent

Autonomous intraday short-selling agent for Nifty 50 stocks on NSE India.

**ONE MANDATE. ONE DIRECTION. ONE UNIVERSE.**

| Rule | Value |
|---|---|
| Direction | **SHORT ONLY** — no long positions, ever |
| Universe | **Nifty 50 only** — 50 stocks, India's benchmark index |
| Order type | **MIS (intraday)** — all positions closed by 3:10 PM IST |
| Exchange | **NSE only** |

**Fully autonomous. Brain-powered. Gets smarter every day.**

---

## This is NOT connected to Finance Meridian website

This is a completely separate, standalone Python trading agent.  
It has its own GitHub repo, its own database, its own server.  
**Your Finance Meridian website is unaffected.**

---

## Cost: ₹0/month

| Component | Cost |
|---|---|
| Zerodha Kite Connect Personal (order API) | **FREE** (2026) |
| Market data (NSE public API + yfinance) | **FREE** |
| Compute (GitHub Actions, public repo) | **FREE** |
| Off-hours controller (GCP e2-micro) | **FREE** |
| LLM (Groq Llama 3.3 70B) | **FREE** |
| Dashboard (Cloudflare Pages) | **FREE** |
| **TOTAL** | **₹0/month** |

---

## Setup (one-time, 30 minutes)

### 1. Create a new GitHub repo

Go to github.com → New repository → name it **`nifty50-shortsell-agent`**  
Set it to **Public** (required for free unlimited GitHub Actions minutes)

### 2. Push this code

```bash
cd nifty50-shortsell-agent
git init
git remote add origin https://github.com/YOUR_USERNAME/nifty50-shortsell-agent.git
git add .
git commit -m "Initial: Nifty50 intraday short-selling agent"
git push -u origin main
```

### 3. Set GitHub Secrets

Go to: `github.com/YOUR_USERNAME/nifty50-shortsell-agent` → Settings → Secrets → Actions

```
KITE_API_KEY          ← from kite.trade
KITE_API_SECRET       ← from kite.trade
KITE_USER_ID          ← your Zerodha login ID
KITE_PASSWORD         ← your Zerodha password
KITE_TOTP_SECRET      ← base32 key from Zerodha 2FA setup
GROQ_API_KEY          ← from console.groq.com (free)
TELEGRAM_BOT_TOKEN    ← from @BotFather on Telegram
TELEGRAM_CHAT_ID      ← your Telegram chat ID
CLOUDFLARE_WEBHOOK_URL
CLOUDFLARE_WEBHOOK_SECRET
```

### 4. Set GitHub Variables

Settings → Variables → Actions:
```
PAPER_TRADE = false
```

### 5. Configure GCP e2-micro (off-hours tasks)

```bash
# SSH into your GCP e2-micro instance
git clone https://github.com/YOUR_USERNAME/nifty50-shortsell-agent.git
cd nifty50-shortsell-agent
bash deploy/setup_gcp_emicro.sh
nano .env   # fill in KITE credentials + TELEGRAM tokens
```

### 6. Health check

```bash
pip install -r requirements.txt
python main.py --healthcheck
```

### 7. Launch

GitHub Actions automatically runs the agent every weekday at 9:20 AM IST.  
Or manually: `python main.py`

---

## How the brain works

```
8:50 AM  Brain observes: SGX Nifty, crude oil, USD/INR, FII data
         Forms beliefs about today's market before NSE opens

9:20 AM  Scans all 50 Nifty 50 stocks
         Each candidate: 11 intelligence layers + brain chain-of-thought
         Brain reasons: OBSERVE → RECALL → REASON → DECIDE → PLAN
         Only brain-approved setups execute

9:20–1PM Live short positions via Zerodha MIS
         Trailing stops active, brain monitors

3:10 PM  ALL positions force-closed (MIS auto-squareoff)

3:35 PM  Brain reflects on every trade — what worked, what failed
         Nightly self-improvement: parameters adapt

Sunday   Weekly evolution cycle
         Brain discovers new patterns
         New rules adopted if they survive walk-forward validation
```

---

## Telegram commands

| Command | Action |
|---|---|
| `/status` | Regime, day P&L, open positions |
| `/positions` | Live P&L per short |
| `/scan` | Force immediate scan |
| `/pause` / `/resume` | Control trading |
| `/report` | Today's full report |
| `/weekly` | 7-day summary |
| `/squareoff` | Emergency close all |

---

## Risk controls (hard-coded)

- Max 3 simultaneous short positions
- Max 2% capital risk per trade
- Max 5% daily loss → halt for the day
- 3 consecutive losses → 30-min circuit breaker
- Nifty falls 0.5–4% → 📈 GREEN SIGNAL — ideal environment, increase size
- Nifty falls >4% → reduce size (lower circuit risk on stocks)
- Nifty rises >1.5% → reduce size (momentum against shorts)
- ALL positions closed by 3:10 PM IST — no exceptions

---

## Disclaimer

Educational purposes only. Not SEBI registered advice.  
Intraday short selling carries substantial risk of loss.
