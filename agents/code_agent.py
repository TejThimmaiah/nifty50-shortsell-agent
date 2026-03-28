import os, subprocess, requests, logging, threading
from groq import Groq

logger = logging.getLogger(__name__)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
REPO_DIR = '/home/thimmaiah18tej/nifty50-shortsell-agent'

def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO_DIR, timeout=120)
    return result.stdout + result.stderr

def run_tests():
    send_telegram("🧪 Running test suite...")
    out = run_cmd("source venv/bin/activate && python -m pytest tests/ -v --tb=short 2>&1 | tail -40")
    passed = out.count(' PASSED')
    failed = out.count(' FAILED')
    send_telegram(f"🧪 <b>Test Results</b>\n✅ Passed: {passed}\n❌ Failed: {failed}\n\n<code>{out[-600:]}</code>")
    return failed, out

def fix_tests():
    send_telegram("🔧 Analysing failing tests with AI...")
    failed_count, test_output = run_tests()
    if failed_count == 0:
        send_telegram("✅ All tests passing!")
        return
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Fix these Python test failures briefly:\n{test_output[-2000:]}"}],
        max_tokens=800
    )
    send_telegram(f"🤖 <b>AI Fix:</b>\n{resp.choices[0].message.content[:1500]}")

def git_status():
    out = run_cmd("git status --short && git log --oneline -5")
    send_telegram(f"📁 <b>Git Status</b>\n<code>{out}</code>")

def git_push(msg="Auto-commit from Code Agent"):
    send_telegram("📤 Pushing to GitHub...")
    out = run_cmd(f'git add -A && git commit -m "{msg}" && git push')
    send_telegram(f"✅ <b>Push result:</b>\n<code>{out[-300:]}</code>")

def handle_code_command(text):
    t = text.strip().lower()
    if t == '/code status': git_status()
    elif t == '/code test': threading.Thread(target=run_tests, daemon=True).start()
    elif t == '/code fix': threading.Thread(target=fix_tests, daemon=True).start()
    elif t.startswith('/code push'):
        msg = text[11:].strip() or "Auto-commit from Code Agent"
        threading.Thread(target=git_push, args=(msg,), daemon=True).start()
    elif t == '/code help':
        send_telegram("💻 <b>Code Agent Commands</b>\n/code status — git status\n/code test — run tests\n/code fix — AI fix suggestions\n/code push <msg> — push to GitHub\n/code help — this menu")
