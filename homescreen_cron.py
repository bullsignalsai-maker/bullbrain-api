# homescreen_cron.py
# ---------------------------------------------------------
# BullSignalsAI — HomeScreen Snapshot Cron (LIGHTWEIGHT)
# ---------------------------------------------------------

import datetime
import requests
import firebase_admin
import main as backend


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
INTERNAL_ENDPOINT = os.getenv(
    "HOMESCREEN_INTERNAL_URL",
    "http://localhost:10000/internal/homescreen/compute"
)


# ---------------------------------------------------------
# Logging helper
# ---------------------------------------------------------
def log(msg: str) -> None:
    backend.log(f"[homescreen_cron] {msg}")


# ---------------------------------------------------------
# Firestore handle
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


# ---------------------------------------------------------
# Fetch snapshot from API (NO ML HERE)
# ---------------------------------------------------------
def fetch_homescreen_snapshot():
    log("Calling internal HomeScreen compute endpoint")

    resp = requests.get(INTERNAL_ENDPOINT, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Internal endpoint failed: {resp.status_code} {resp.text}"
        )

    return resp.json()


# ---------------------------------------------------------
# Save to Firestore
# ---------------------------------------------------------
def save_homescreen_to_firestore(homescreen_doc):
    db = get_db()
    db.collection("bullsignals_ai").document("homescreen_snapshot").set(
        homescreen_doc, merge=True
    )
    log("💾 Saved bullsignals_ai/homescreen_snapshot")


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
def main():
    started = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    log(f"HomeScreen cron started at {started}")

    try:
        homescreen_doc = fetch_homescreen_snapshot()
        save_homescreen_to_firestore(homescreen_doc)

        finished = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        log(f"HomeScreen cron completed at {finished}")

    except Exception as e:
        log(f"❌ HomeScreen cron failed: {e}")
        raise


if __name__ == "__main__":
    main()
