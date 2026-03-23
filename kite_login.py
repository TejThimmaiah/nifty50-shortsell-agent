"""
Zerodha Kite Daily Login & Token Refresh
=========================================
Uses Selenium headless Chrome because Zerodha's connect/finish page
uses JavaScript to redirect to 127.0.0.1?request_token=XXX.
A plain HTTP requests library cannot execute JavaScript.
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


def setup_chrome():
    import subprocess
    logger.info("Installing Chrome and Selenium...")
    subprocess.run([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager", "-q"], check=True)
    subprocess.run(["sudo", "apt-get", "update", "-qq"], capture_output=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "google-chrome-stable"], capture_output=True)
    # Fallback to chromium
    result = subprocess.run(["which", "google-chrome"], capture_output=True, text=True)
    if not result.stdout.strip():
        subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "chromium-browser"], capture_output=True)


def get_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,800")

    # Try webdriver-manager first (auto-downloads matching chromedriver)
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=opts)
    except Exception as e:
        logger.warning(f"webdriver-manager failed: {e}")

    # Fallback to system chromedriver
    for binary in ["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"]:
        if os.path.exists(binary):
            opts.binary_location = binary
            break

    return webdriver.Chrome(options=opts)


def login() -> str:
    setup_chrome()
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = get_driver()
    try:
        login_url = f"https://kite.zerodha.com/connect/login?api_key={KITE_API_KEY}&v=3"
        logger.info(f"Opening: {login_url}")
        driver.get(login_url)
        time.sleep(3)

        wait = WebDriverWait(driver, 20)

        # Enter user ID
        uid = wait.until(EC.presence_of_element_located((By.ID, "userid")))
        uid.clear(); uid.send_keys(KITE_USER_ID)
        logger.info("User ID entered")

        # Enter password
        pwd = driver.find_element(By.ID, "password")
        pwd.clear(); pwd.send_keys(KITE_PASSWORD)
        logger.info("Password entered")

        # Click login
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(3)

        # Enter TOTP
        logger.info("Entering TOTP...")
        totp_code = pyotp.TOTP(KITE_TOTP_SECRET).now()

        # Try multiple selectors for TOTP field
        totp_field = None
        for selector in [
            (By.ID, "pin"),
            (By.XPATH, "//input[@type='number']"),
            (By.XPATH, "//input[contains(@placeholder,'TOTP')]"),
            (By.XPATH, "//input[contains(@placeholder,'PIN')]"),
            (By.XPATH, "//input[contains(@placeholder,'code')]"),
            (By.CSS_SELECTOR, "input[maxlength='6']"),
        ]:
            try:
                totp_field = wait.until(EC.presence_of_element_located(selector))
                if totp_field:
                    break
            except Exception:
                continue

        if not totp_field:
            raise ValueError("Could not find TOTP input field")

        totp_field.clear()
        totp_field.send_keys(totp_code)
        time.sleep(1)

        # Submit
        try:
            driver.find_element(By.XPATH, "//button[@type='submit']").click()
        except Exception:
            pass
        time.sleep(4)

        # Wait for redirect to 127.0.0.1 with request_token
        logger.info("Waiting for request_token redirect...")
        request_token = None

        for _ in range(20):
            url = driver.current_url
            logger.info(f"URL: {url[:120]}")

            if "request_token=" in url:
                params = parse_qs(urlparse(url).query)
                request_token = params.get("request_token", [None])[0]
                if request_token:
                    logger.info(f"✅ Got request_token: {request_token[:12]}...")
                    break

            # Also check page source
            try:
                src = driver.page_source
                m = re.search(r'request_token["\s:=]+([A-Za-z0-9]{10,})', src)
                if m:
                    request_token = m.group(1)
                    logger.info(f"✅ Got request_token from page: {request_token[:12]}...")
                    break
            except Exception:
                pass

            time.sleep(1)

        if not request_token:
            screenshot_path = "/tmp/kite_debug.png"
            try:
                driver.save_screenshot(screenshot_path)
                logger.info(f"Screenshot saved: {screenshot_path}")
            except Exception:
                pass
            raise ValueError(f"request_token not found. Last URL: {driver.current_url[:200]}")

        # Exchange for access_token
        logger.info("Exchanging for access_token...")
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = data["access_token"]
        logger.info(f"✅ access_token: {access_token[:12]}...")
        return access_token

    finally:
        try:
            driver.quit()
        except Exception:
            pass


def save_to_github(access_token):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning("No GITHUB_TOKEN — skipping GitHub Secrets save")
        return
    try:
        import base64
        from nacl import encoding, public
        hdrs = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        kd = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/public-key",
            headers=hdrs, timeout=15).json()
        pk  = public.PublicKey(kd["key"].encode(), encoding.Base64Encoder())
        enc = base64.b64encode(public.SealedBox(pk).encrypt(access_token.encode())).decode()
        r   = requests.put(
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
