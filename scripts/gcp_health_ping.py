"""
GCP Health Ping (runs on e2-micro at 9:30 AM IST)
Checks that today's GitHub Actions trading workflow started successfully.
If the workflow appears to have failed, sends a Telegram alert.
"""

import os
import sys
import requests
import logging
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("gcp_ping")

GH_TOKEN        = os.getenv("GITHUB_TOKEN")        # Personal Access Token (repo scope)
GH_REPO         = os.getenv("GITHUB_REPO")         # e.g. "yourusername/finance-agent"
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")
CF_WEBHOOK      = os.getenv("CLOUDFLARE_WEBHOOK_URL")
CF_SECRET       = os.getenv("CLOUDFLARE_WEBHOOK_SECRET")


def check_workflow_status() -> dict:
    """Check if today's trading session workflow is running on GitHub Actions."""
    if not GH_TOKEN or not GH_REPO:
        return {"status": "unknown", "reason": "GITHUB_TOKEN or GITHUB_REPO not set"}

    url = f"https://api.github.com/repos/{GH_REPO}/actions/runs"
    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        params={"per_page": 5},
        timeout=15,
    )
    resp.raise_for_status()
    runs = resp.json().get("workflow_runs", [])

    today = date.today().isoformat()
    trading_runs = [
        r for r in runs
        if "trading" in r.get("name", "").lower()
        and r.get("created_at", "")[:10] == today
    ]

    if not trading_runs:
        return {"status": "NOT_STARTED", "reason": "No trading workflow found for today"}

    latest = trading_runs[0]
    return {
        "status": latest.get("status"),      # queued, in_progress, completed
        "conclusion": latest.get("conclusion"),  # success, failure, cancelled
        "name": latest.get("name"),
        "started_at": latest.get("run_started_at"),
        "url": latest.get("html_url"),
    }


def notify(message: str):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": message},
                timeout=8,
            )
        except Exception as e:
            logger.warning(f"Telegram notify failed: {e}")

    if CF_WEBHOOK:
        try:
            requests.post(
                f"{CF_WEBHOOK.rstrip('/')}/webhook/alert",
                json={"message": message},
                headers={"X-Webhook-Secret": CF_SECRET or ""},
                timeout=5,
            )
        except Exception:
            pass


def main():
    logger.info("GCP health ping starting...")
    status = check_workflow_status()
    logger.info(f"Workflow status: {status}")

    ws = status.get("status", "unknown")
    conc = status.get("conclusion")

    if ws == "NOT_STARTED":
        msg = ("⚠️ ALERT: Trading workflow did NOT start today!\n"
               "Check GitHub Actions and start manually if needed:\n"
               "https://github.com/" + (GH_REPO or "YOUR_REPO") + "/actions")
        notify(msg)
        logger.error(msg)

    elif ws == "in_progress":
        logger.info(f"✅ Trading session running normally — {status.get('name')}")
        # Ping Cloudflare dashboard to show agent is alive
        if CF_WEBHOOK:
            try:
                requests.post(
                    f"{CF_WEBHOOK.rstrip('/')}/webhook/alert",
                    json={"message": f"✅ Agent running | {datetime.now().strftime('%H:%M IST')}",
                          "category": "system"},
                    headers={"X-Webhook-Secret": CF_SECRET or ""},
                    timeout=5,
                )
            except Exception:
                pass

    elif ws == "completed" and conc in ("failure", "cancelled"):
        msg = (f"🚨 Trading workflow FAILED today!\n"
               f"Status: {conc}\n"
               f"URL: {status.get('url', 'N/A')}")
        notify(msg)
        logger.error(msg)

    elif ws == "queued":
        logger.info("Workflow queued — will start shortly")

    else:
        logger.info(f"Status: {ws} / {conc}")


if __name__ == "__main__":
    main()
