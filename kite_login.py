"""
Zerodha Kite Daily Login & Token Refresh
=========================================
Uses the enctoken approach — the most reliable headless login method.

After /api/login + /api/twofa, Zerodha returns an enc_token.
We set this as a cookie and use it to get the request_token
from the connect/login redirect.
"""

import os
import sys
import re
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
GITHUB_REPO      = os.getenv("GITHUB_REPO")


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


def extract_token_from_url(url: str):
    """Extract request_token from a URL string."""
    if "request_token=" in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        token = params.get("request_token", [None])[0]
        if token:
            return token
    match = re.search(r'request_token=([A-Za-z0-9]+)', url)
    if match:
        return match.group(1)
    return None


def login() -> str:
    """
    Full Zerodha login. Returns access_token.

    Flow:
    1. POST /api/login → request_id
    2. POST /api/twofa → enc_token (in response + cookies)
    3. Use enc_token cookie to GET connect/login → redirects to 127.0.0.1?request_token=XXX
    4. Exchange request_token for access_token
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent":     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120",
        "Accept":         "*/*",
        "X-Kite-Version": "3",
    })

    # ── Step 1: Password Login ────────────────────────────────────
    logger.info("Step 1: Password login...")
    r1 = session.post(
        "https://kite.zerodha.com/api/login",
        data={"user_id": KITE_USER_ID, "password": KITE_PASSWORD},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    r1.raise_for_status()
    d1 = r1.json()
    if d1.get("status") != "success":
        raise ValueError(f"Login failed: {d1.get('message', d1)}")
    request_id = d1["data"]["request_id"]
    logger.info(f"Step 1 OK — request_id: {request_id}")

    # ── Step 2: TOTP ──────────────────────────────────────────────
    logger.info("Step 2: TOTP verification...")
    time.sleep(2)
    totp_code = pyotp.TOTP(KITE_TOTP_SECRET).now()

    r2 = session.post(
        "https://kite.zerodha.com/api/twofa",
        data={
            "user_id":     KITE_USER_ID,
            "request_id":  request_id,
            "twofa_value": totp_code,
            "twofa_type":  "totp",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    r2.raise_for_status()
    d2 = r2.json()
    if d2.get("status") != "success":
        raise ValueError(f"TOTP failed: {d2.get('message', d2)}")
    logger.info("Step 2 OK — TOTP verified")

    # Extract enc_token — it comes in response data or cookies
    enc_token = None
    if isinstance(d2.get("data"), dict):
        enc_token = d2["data"].get("enc_token")
    if not enc_token:
        for cookie in session.cookies:
            if cookie.name == "enctoken":
                enc_token = cookie.value
                break
    if enc_token:
        logger.info(f"enc_token obtained: {enc_token[:10]}...")
        session.cookies.set("enctoken", enc_token, domain=".zerodha.com")

    # ── Step 3: Get request_token ─────────────────────────────────
    logger.info("Step 3: Getting request_token...")
    time.sleep(1)

    connect_url = f"https://kite.zerodha.com/connect/login?api_key={KITE_API_KEY}&v=3"
    request_token = None

    # Try A: No redirect following — check Location header
    try:
        r3a = session.get(connect_url, allow_redirects=False, timeout=20)
        location = r3a.headers.get("Location", "")
        logger.info(f"A - Location: {location[:120]}")
        request_token = extract_token_from_url(location)
    except Exception as e:
        logger.warning(f"A failed: {e}")

    # Try B: Follow redirects — 127.0.0.1 causes ConnectionError containing the URL
    if not request_token:
        try:
            r3b = session.get(connect_url, allow_redirects=True, timeout=10)
            final_url = r3b.url
            logger.info(f"B - Final URL: {final_url[:120]}")
            request_token = extract_token_from_url(final_url)
        except requests.exceptions.ConnectionError as e:
            err_str = str(e)
            logger.info(f"B - ConnectionError (expected for 127.0.0.1): {err_str[:120]}")
            request_token = extract_token_from_url(err_str)
        except Exception as e:
            logger.warning(f"B failed: {e}")

    # Try C: Use enc_token as Authorization header
    if not request_token and enc_token:
        try:
            r3c = requests.get(
                connect_url,
                headers={
                    "User-Agent":      "Mozilla/5.0",
                    "Authorization":   f"enctoken {enc_token}",
                    "X-Kite-Version":  "3",
                },
                cookies={"enctoken": enc_token},
                allow_redirects=False,
                timeout=20,
            )
            location = r3c.headers.get("Location", "")
            logger.info(f"C - Location: {location[:120]}")
            request_token = extract_token_from_url(location)
        except Exception as e:
            logger.warning(f"C failed: {e}")

    if not request_token:
        raise ValueError(
            "Could not extract request_token. "
            "Please verify redirect URL in Kite Connect app is set to: https://127.0.0.1"
        )

    logger.info(f"Step 3 OK — request_token: {request_token[:10]}...")

    # ── Step 4: Exchange for access_token ─────────────────────────
    logger.info("Step 4: Generating access token...")
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=KITE_API_KEY)
    data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    access_token = data["access_token"]
    logger.info(f"Step 4 OK — access_token: {access_token[:10]}...")

    return access_token


def save_to_github_secrets(access_token: str):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning("No GITHUB_TOKEN/GITHUB_REPO — skipping GitHub Secrets update")
        return
    try:
        import base64
        from nacl import encoding, public

        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        key_resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/public-key",
            headers=headers, timeout=15,
        )
        key_resp.raise_for_status()
        key_data   = key_resp.json()
        pk         = public.PublicKey(key_data["key"].encode(), encoding.Base64Encoder())
        box        = public.SealedBox(pk)
        encrypted  = base64.b64encode(box.encrypt(access_token.encode())).decode()
        r = requests.put(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/KITE_ACCESS_TOKEN",
            headers=headers,
            json={"encrypted_value": encrypted, "key_id": key_data["key_id"]},
            timeout=15,
        )
        if r.status_code in (201, 204):
            logger.info("✅ KITE_ACCESS_TOKEN saved to GitHub Secrets")
        else:
            logger.warning(f"GitHub Secret update: {r.status_code}")
    except Exception as e:
        logger.error(f"Failed to save to GitHub Secrets: {e}")


def save_to_env(access_token: str):
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_file):
        return
    try:
        with open(env_file) as f:
            lines = f.readlines()
        new_lines = []
        updated = False
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
        logger.info(".env updated")
    except Exception as e:
        logger.error(f"Failed to update .env: {e}")


def main():
    logger.info("=== Zerodha Daily Token Refresh ===")

    missing = [k for k in [
        "KITE_API_KEY", "KITE_API_SECRET",
        "KITE_USER_ID", "KITE_PASSWORD", "KITE_TOTP_SECRET"
    ] if not os.getenv(k)]

    if missing:
        msg = f"❌ Missing Kite credentials: {missing}"
        logger.error(msg)
        notify(msg)
        sys.exit(1)

    try:
        access_token = login()
        save_to_github_secrets(access_token)
        save_to_env(access_token)
        notify("🔑 Kite token refreshed ✅ — Tej is ready to trade today")
        logger.info("✅ Token refresh complete")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        notify(
            f"❌ Kite token refresh FAILED: {str(e)[:200]}\n"
            f"⚠️ Tej will NOT trade today without a valid token."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
