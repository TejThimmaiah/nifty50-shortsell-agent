"""
Zerodha Kite Daily Login & Token Refresh
Uses Selenium headless Chrome to handle JavaScript redirects.
Fixed: clicks the Authorize button on connect/authorize page.
"""

import os, sys, time, logging, requests, pyotp, re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("kite_login")

KITE_API_KEY     = os.getenv("KITE_API_KEY")
KITE_API_SECRET  = os.getenv("KITE_API_SECRET")
KITE_USER_ID     = os.getenv("KITE_USER_ID")
KITE_PASSWORD    = os.getenv("KITE_PASSWORD")
KITE_TOTP_SECRET = os.getenv("KITE_TOTP_SECRET")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT    = os.getenv("TELEGRAM_CHAT_ID")
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN")
GITHUB_REPO      = os.getenv("GITHUB_REPO")


def notify(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=10)
    except Exception:
        pass


def get_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,800")
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    except Exception as e:
        logger.warning(f"webdriver-manager failed: {e}, trying system Chrome")
        for binary in ["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"]:
            if os.path.exists(binary):
                opts.binary_location = binary
                break
        return webdriver.Chrome(options=opts)


def login() -> str:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager", "-q"])

    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = get_driver()
    wait = WebDriverWait(driver, 20)

    try:
        # ── Open Kite Connect login ────────────────────────────────
        url = f"https://kite.zerodha.com/connect/login?api_key={KITE_API_KEY}&v=3"
        logger.info(f"Opening: {url}")
        driver.get(url)
        time.sleep(3)

        # ── Enter credentials ──────────────────────────────────────
        wait.until(EC.presence_of_element_located((By.ID, "userid"))).send_keys(KITE_USER_ID)
        driver.find_element(By.ID, "password").send_keys(KITE_PASSWORD)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        logger.info("Credentials submitted")
        time.sleep(3)

        # ── Enter TOTP ─────────────────────────────────────────────
        totp_code = pyotp.TOTP(KITE_TOTP_SECRET).now()
        logger.info(f"TOTP: {totp_code}")

        for selector in [
            (By.ID, "pin"),
            (By.XPATH, "//input[@type='number']"),
            (By.CSS_SELECTOR, "input[maxlength='6']"),
            (By.XPATH, "//input[contains(@placeholder,'TOTP') or contains(@placeholder,'PIN') or contains(@placeholder,'code')]"),
        ]:
            try:
                field = wait.until(EC.presence_of_element_located(selector))
                field.clear()
                field.send_keys(totp_code)
                logger.info(f"TOTP entered via {selector}")
                break
            except Exception:
                continue

        time.sleep(1)
        try:
            driver.find_element(By.XPATH, "//button[@type='submit']").click()
        except Exception:
            pass
        time.sleep(4)

        # ── Click Authorize button if on connect/authorize page ────
        current = driver.current_url
        logger.info(f"URL after TOTP: {current[:120]}")

        if "connect/authorize" in current or "authorize" in current.lower():
            logger.info("On authorize page — clicking Authorize button...")
            # Try multiple possible button selectors
            authorized = False
            for selector in [
                (By.XPATH, "//button[contains(text(),'Authorise') or contains(text(),'Authorize') or contains(text(),'Allow')]"),
                (By.XPATH, "//button[@type='submit']"),
                (By.CSS_SELECTOR, "button.button-blue"),
                (By.CSS_SELECTOR, "button.btn-primary"),
                (By.XPATH, "//input[@type='submit']"),
                (By.XPATH, "//a[contains(text(),'Authorise') or contains(text(),'Authorize')]"),
            ]:
                try:
                    btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(selector))
                    logger.info(f"Clicking authorize button: {selector}")
                    btn.click()
                    authorized = True
                    time.sleep(4)
                    break
                except Exception:
                    continue

            if not authorized:
                # Log page source for debugging
                logger.warning(f"Could not find authorize button. Page title: {driver.title}")
                logger.warning(f"Page source snippet: {driver.page_source[:500]}")

        # ── Wait for request_token in URL ──────────────────────────
        logger.info("Waiting for request_token...")
        request_token = None

        for i in range(25):
            current = driver.current_url
            logger.info(f"[{i+1}] URL: {current[:120]}")

            # Check URL for request_token
            if "request_token=" in current:
                params = parse_qs(urlparse(current).query)
                request_token = params.get("request_token", [None])[0]
                if request_token:
                    logger.info(f"✅ request_token from URL: {request_token[:12]}...")
                    break

            # Check if still on authorize page — try clicking again
            if "connect/authorize" in current and i > 0 and i % 5 == 0:
                logger.info("Still on authorize page, trying to click authorize again...")
                for selector in [
                    (By.XPATH, "//button[@type='submit']"),
                    (By.XPATH, "//button[contains(text(),'uthoris')]"),
                    (By.CSS_SELECTOR, "button"),
                ]:
                    try:
                        driver.find_element(*selector).click()
                        time.sleep(3)
                        break
                    except Exception:
                        continue

            time.sleep(1)

        if not request_token:
            # Save screenshot for debugging
            try:
                driver.save_screenshot("/tmp/kite_debug.png")
                logger.info("Screenshot: /tmp/kite_debug.png")
            except Exception:
                pass
            raise ValueError(f"request_token not found. Last URL: {driver.current_url[:200]}")

        # ── Exchange for access_token ──────────────────────────────
        logger.info("Generating access token...")
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = data["access_token"]
        logger.info(f"✅ access_token obtained: {access_token[:12]}...")
        return access_token

    finally:
        try:
            driver.quit()
        except Exception:
            pass


def save_to_github(token):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        import base64
        from nacl import encoding, public
        hdrs = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        kd   = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/public-key",
                            headers=hdrs, timeout=15).json()
        pk   = public.PublicKey(kd["key"].encode(), encoding.Base64Encoder())
        enc  = base64.b64encode(public.SealedBox(pk).encrypt(token.encode())).decode()
        r    = requests.put(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/KITE_ACCESS_TOKEN",
            headers=hdrs, json={"encrypted_value": enc, "key_id": kd["key_id"]}, timeout=15)
        logger.info(f"GitHub Secret saved: {r.status_code}")
    except Exception as e:
        logger.error(f"GitHub Secret save failed: {e}")


def main():
    logger.info("=== Zerodha Daily Token Refresh ===")
    missing = [k for k in ["KITE_API_KEY","KITE_API_SECRET","KITE_USER_ID","KITE_PASSWORD","KITE_TOTP_SECRET"]
               if not os.getenv(k)]
    if missing:
        msg = f"❌ Missing: {missing}"
        logger.error(msg); notify(msg); sys.exit(1)

    try:
        token = login()
        save_to_github(token)
        notify("🔑 Kite token refreshed ✅ — Tej is ready to trade today")
        logger.info("✅ Done")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FAILED: {e}")
        notify(f"❌ Kite token refresh FAILED: {str(e)[:200]}\n⚠️ Tej will NOT trade today.")
        sys.exit(1)


if __name__ == "__main__":
    main()
