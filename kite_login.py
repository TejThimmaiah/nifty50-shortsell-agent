"""
Zerodha Kite Daily Login & Token Refresh
Runs at 8:00 AM IST every weekday (via cron).
Refreshes the access token required for API trading.

Usage: python kite_login.py
Cron:  0 8 * * 1-5 cd /home/ubuntu/finance-agent && python kite_login.py

If you have TOTP (Authenticator app) on your Zerodha account,
set KITE_TOTP_SECRET in .env for fully automated login.
"""

import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("kite_login")

KITE_API_KEY    = os.getenv("KITE_API_KEY")
KITE_API_SECRET = os.getenv("KITE_API_SECRET")
KITE_USER_ID    = os.getenv("KITE_USER_ID")
KITE_PASSWORD   = os.getenv("KITE_PASSWORD")
KITE_TOTP_SECRET = os.getenv("KITE_TOTP_SECRET")   # Base32 TOTP secret for automation
ENV_FILE        = os.path.join(os.path.dirname(__file__), ".env")


def get_totp_code():
    """Generate current TOTP code from secret."""
    try:
        import pyotp
        totp = pyotp.TOTP(KITE_TOTP_SECRET)
        return totp.now()
    except ImportError:
        logger.error("pyotp not installed. Run: pip install pyotp")
        return None


def automated_login() -> str:
    """
    Perform automated Zerodha login using credentials + TOTP.
    Returns access_token if successful.
    """
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=KITE_API_KEY)

    # Step 1: Get request token via Kite login URL
    # For headless automation, we use the API login directly
    session = requests.Session()

    # Kite login step 1
    resp = session.post(
        "https://kite.zerodha.com/api/login",
        data={
            "user_id": KITE_USER_ID,
            "password": KITE_PASSWORD,
        },
        timeout=15,
    )
    resp.raise_for_status()
    login_data = resp.json()
    request_id = login_data["data"]["request_id"]
    logger.info(f"Login step 1 OK, request_id: {request_id}")

    # Step 2: TOTP verification
    totp_code = get_totp_code()
    if not totp_code:
        raise ValueError("Could not generate TOTP code")

    resp2 = session.post(
        "https://kite.zerodha.com/api/twofa",
        data={
            "user_id": KITE_USER_ID,
            "request_id": request_id,
            "twofa_value": totp_code,
            "twofa_type": "totp",
        },
        timeout=15,
    )
    resp2.raise_for_status()
    twofa_data = resp2.json()
    logger.info("Login step 2 (TOTP) OK")

    # Step 3: Get request token from the redirect
    # (Kite sends this via redirect, we parse it from the session)
    resp3 = session.get(
        f"https://kite.zerodha.com/connect/login?api_key={KITE_API_KEY}&v=3",
        allow_redirects=True,
        timeout=15,
    )

    # Extract request_token from redirect URL
    final_url = resp3.url
    if "request_token" in final_url:
        request_token = final_url.split("request_token=")[1].split("&")[0]
        logger.info(f"Got request_token: {request_token[:10]}...")
    else:
        raise ValueError(f"No request_token in redirect: {final_url}")

    # Step 4: Exchange for access token
    data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    access_token = data["access_token"]
    logger.info(f"Access token obtained: {access_token[:10]}...")

    return access_token


def update_env_file(access_token: str):
    """Update KITE_ACCESS_TOKEN in .env file."""
    try:
        with open(ENV_FILE, "r") as f:
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

        with open(ENV_FILE, "w") as f:
            f.writelines(new_lines)

        logger.info(".env updated with new access token")

        # Also restart the trading agent service to pick up new token
        os.system("sudo systemctl restart finance-agent.service 2>/dev/null || true")

    except Exception as e:
        logger.error(f"Failed to update .env: {e}")


def notify_telegram(message: str):
    """Send notification to Telegram."""
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
    except Exception:
        pass


def main():
    logger.info("Starting Zerodha daily token refresh...")

    if not all([KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET]):
        logger.error("Missing required Kite credentials in .env")
        logger.info("Required: KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET")
        notify_telegram("❌ Kite token refresh failed: missing credentials")
        return

    try:
        access_token = automated_login()
        update_env_file(access_token)
        notify_telegram("🔑 Zerodha token refreshed successfully. Agent ready.")
        logger.info("✅ Token refresh complete.")

    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        notify_telegram(f"❌ Kite token refresh FAILED: {str(e)[:100]}\nManual login required.")


if __name__ == "__main__":
    main()
