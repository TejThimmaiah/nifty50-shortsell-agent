"""
GCP Backup Script (runs on e2-micro at 4 PM IST)
Backs up today's trade database to Google Cloud Storage (5GB free tier).
Also downloads the day's GitHub Actions logs for local archiving.
"""

import os
import sys
import shutil
import subprocess
import logging
from datetime import date, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("gcp_backup")

INSTALL_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH     = os.path.join(INSTALL_DIR, "db", "trades.db")
BACKUP_DIR  = os.path.join(INSTALL_DIR, "db", "backups")
GCS_BUCKET  = os.getenv("GCS_BUCKET")   # e.g. "gs://nifty50-shortsell-agent-backups"


def backup_locally():
    """Create a local timestamped backup of the DB."""
    if not os.path.exists(DB_PATH):
        logger.info("No trades.db found — nothing to back up")
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)
    today    = date.today().isoformat()
    dest     = os.path.join(BACKUP_DIR, f"trades_{today}.db")
    shutil.copy2(DB_PATH, dest)
    size_kb  = os.path.getsize(dest) / 1024
    logger.info(f"Local backup: {dest} ({size_kb:.1f} KB)")
    return dest


def upload_to_gcs(local_path: str):
    """Upload backup to Google Cloud Storage (5GB always free tier)."""
    if not GCS_BUCKET:
        logger.info("GCS_BUCKET not set — skipping cloud backup")
        return

    try:
        result = subprocess.run(
            ["gsutil", "cp", local_path, f"{GCS_BUCKET}/"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.info(f"✅ Uploaded to GCS: {GCS_BUCKET}/")
        else:
            logger.warning(f"GCS upload failed: {result.stderr}")
    except FileNotFoundError:
        logger.warning("gsutil not found — install Google Cloud SDK for cloud backup")
    except Exception as e:
        logger.error(f"GCS upload error: {e}")


def cleanup_old_local_backups(keep_days: int = 30):
    """Remove local backups older than keep_days."""
    if not os.path.exists(BACKUP_DIR):
        return
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=keep_days)
    removed = 0
    for fname in os.listdir(BACKUP_DIR):
        fpath = os.path.join(BACKUP_DIR, fname)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
        if mtime < cutoff:
            os.remove(fpath)
            removed += 1
    if removed:
        logger.info(f"Removed {removed} old backups")


def main():
    logger.info(f"=== Backup started {date.today().isoformat()} ===")
    local = backup_locally()
    if local:
        upload_to_gcs(local)
    cleanup_old_local_backups()
    logger.info("=== Backup complete ===")


if __name__ == "__main__":
    main()
