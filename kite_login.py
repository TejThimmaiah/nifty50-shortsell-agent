"""
Zerodha Kite Daily Login & Token Refresh
Runs at 8:45 AM IST every weekday via GitHub Actions morning_prep job.

Correct login flow:
  1. POST credentials to kite.zerodha.com/api/login
  2. POST TOTP to kite.zerodha.com/api/twofa
  3. GET connect/login?api_key=... → redirects to 127.0.0.1?request_token=...
  4. Exchange request_token for access_token via KiteConnect
  5. Store access_token as GitHub Secret for trading session

The access_token is valid for the rest of the trading day.
"""

import os
import sys
import time
import logging
import requests
import pyotp
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
GITHUB_REPO      = os.getenv("GITHUB_REPO")   # e.g. TejThimmaiah/nifty50-shortsell-agent


def notify(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg},
            timeout=10,
        )
    except Exception:
        pass


def get_totp() -> str:
    return pyotp.TOTP(KITE_TOTP_SECRET).now()


def login() -> str:
    """
    Full Zerodha login flow. Returns access_token.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "X-Kite-Version": "3",
    })

    # ── Step 1: Login with user_id + password ─────────────────────
    logger.info("Step 1: Password login...")
    r1 = session.post(
        "https://kite.zerodha.com/api/login",
        data={"user_id": KITE_USER_ID, "password": KITE_PASSWORD},
        timeout=20,
    )
    r1.raise_for_status()
    d1 = r1.json()
    if d1.get("status") != "success":
        raise ValueError(f"Login failed: {d1.get('message', d1)}")
    request_id = d1["data"]["request_id"]
    logger.info(f"Step 1 OK — request_id: {request_id}")

    # ── Step 2: TOTP verification ──────────────────────────────────
    logger.info("Step 2: TOTP verification...")
    time.sleep(1)  # Small delay to avoid rate limiting
    totp_code = get_totp()
    r2 = session.post(
        "https://kite.zerodha.com/api/twofa",
        data={
            "user_id":     KITE_USER_ID,
            "request_id":  request_id,
            "twofa_value": totp_code,
            "twofa_type":  "totp",
        },
        timeout=20,
    )
    r2.raise_for_status()
    d2 = r2.json()
    if d2.get("status") != "success":
        raise ValueError(f"TOTP failed: {d2.get('message', d2)}")
    logger.info("Step 2 OK — TOTP verified")

    # ── Step 3: Get request_token via connect/login redirect ───────
    logger.info("Step 3: Getting request_token...")
    time.sleep(1)

    # This will redirect to https://127.0.0.1?request_token=XXX&...
    # We catch the redirect and extract the token from the URL
    connect_url = f"https://kite.zerodha.com/connect/login?api_key={KITE_API_KEY}&v=3"

    try:
        r3 = session.get(connect_url, allow_redirects=False, timeout=20)
        redirect_url = r3.headers.get("Location", "")
    except requests.exceptions.ConnectionError as e:
        # Sometimes the redirect to 127.0.0.1 raises a connection error
        # The URL is still in the exception
        redirect_url = str(e)

    logger.info(f"Redirect URL: {redirect_url[:80]}...")

    # Extract request_token from URL
    request_token = None

    if "request_token=" in redirect_url:
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        request_token = params.get("request_token", [None])[0]

    if not request_token:
        # Try following redirects fully
        try:
            r3b = session.get(connect_url, allow_redirects=True, timeout=20)
            final_url = r3b.url
            if "request_token=" in final_url:
                parsed = urlparse(final_url)
                params = parse_qs(parsed.query)
                request_token = params.get("request_token", [None])[0]
        except Exception as e:
            url_str = str(e)
            if "request_token=" in url_str:
                import re
                match = re.search(r'request_token=([^&\s"\']+)', url_str)
                if match:
                    request_token = match.group(1)

    if not request_token:
        raise ValueError(f"Could not extract request_token from: {redirect_url[:200]}")

    logger.info(f"Step 3 OK — request_token: {request_token[:10]}...")

    # ── Step 4: Exchange request_token for access_token ───────────
    logger.info("Step 4: Generating session...")
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=KITE_API_KEY)
    data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    access_token = data["access_token"]
    logger.info(f"Step 4 OK — access_token: {access_token[:10]}...")

    return access_token


def save_token(access_token: str):
    """
    Save access_token to GitHub Secrets so trading_session job can use it.
    Also saves to .env for local use.
    """
    # Save to GitHub Secret (for GitHub Actions)
    if GITHUB_TOKEN and GITHUB_REPO:
        try:
            import base64
            from nacl import encoding, public

            # Get repo public key
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            }
            key_resp = requests.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/public-key",
                headers=headers,
                timeout=15,
            )
            key_resp.raise_for_status()
            key_data = key_resp.json()
            public_key = key_data["key"]
            key_id     = key_data["key_id"]

            # Encrypt the token
            pk = public.PublicKey(public_key.encode(), encoding.Base64Encoder())
            box = public.SealedBox(pk)
            encrypted = base64.b64encode(
                box.encrypt(access_token.encode())
            ).decode()

            # Update the secret
            secret_resp = requests.put(
                f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/KITE_ACCESS_TOKEN",
                headers=headers,
                json={"encrypted_value": encrypted, "key_id": key_id},
                timeout=15,
            )
            if secret_resp.status_code in (201, 204):
                logger.info("✅ KITE_ACCESS_TOKEN saved to GitHub Secrets")
            else:
                logger.warning(f"GitHub Secret update: {secret_resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to save to GitHub Secrets: {e}")

    # Also save to local .env if it exists
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r") as f:
                lines = f.readlines()
            updated = False
            new_lines = []
            for line in lines:
                if line.startswith("KITE_ACCESS_TOKEN="):
                    new_lines.append(f"KITE_ACCESS_TOKEN={access_token}\n")
                    updated = True
                else:
                    new_lines.append(line)
            if not updated:
                new_lines.append(f"KITE_ACCESS_TOKEN={access_token}\n")
            with open(env_file, "w") as f:
                f.writelines(new_lines)
            logger.info(".env updated with new access token")
        except Exception as e:
            logger.error(f"Failed to update .env: {e}")


def main():
    logger.info("=== Zerodha Daily Token Refresh ===")

    missing = [k for k in ["KITE_API_KEY","KITE_API_SECRET","KITE_USER_ID","KITE_PASSWORD","KITE_TOTP_SECRET"]
               if not os.getenv(k)]
    if missing:
        msg = f"❌ Missing credentials: {missing}"
        logger.error(msg)
        notify(msg)
        sys.exit(1)

    try:
        access_token = login()
        save_token(access_token)
        notify("🔑 Kite token refreshed ✅ — Tej is ready to trade today")
        logger.info("✅ Token refresh complete")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        notify(
            f"❌ Kite token refresh FAILED: {str(e)[:150]}\n"
            f"Trading will NOT execute today without a valid token."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
