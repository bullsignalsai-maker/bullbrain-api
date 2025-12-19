# homescreen_cron.py
# ---------------------------------------------------------
# BullSignalsAI — HomeScreen Snapshot Cron
# ---------------------------------------------------------

import datetime
import firebase_admin
import main as backend

import backend.bullbrain as bullbrain
from backend.homescreen_logic import build_homescreen_mag7_block
from backend.homescreen_macro_logic import build_homescreen_macro_snapshot


def log(msg: str) -> None:
    backend.log(f"[homescreen_cron] {msg}")


def get_db():
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


def compute_homescreen_snapshot():
    log("Loading BullBrain model for HomeScreen cron…")

    model = bullbrain.load_bullbrain_model()
    if model is None:
        raise RuntimeError("BullBrain model failed to load")

    log("BullBrain model loaded successfully")

    log("Computing HomeScreen MAG7 snapshot")
    mag7_block = build_homescreen_mag7_block()

    log("Computing HomeScreen macro snapshot")
    macro_snapshot = build_homescreen_macro_snapshot()

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return {
        "schema_version": "homescreen_v1",
        "updated_at": now,
        "market": macro_snapshot.get("market"),
        "macro": {
            "carousel": macro_snapshot.get("carousel", [])
        },
        "mag7": mag7_block,
        "meta": {
            "computed_by": "homescreen_cron",
            "refresh_minutes": 15,
            "bullbrain_version": "v2-48f",
        },
    }


def save_homescreen_to_firestore(homescreen_doc):
    db = get_db()
    db.collection("bullsignals_ai").document("homescreen_snapshot").set(
        homescreen_doc, merge=True
    )
    log("💾 Saved bullsignals_ai/homescreen_snapshot")


def main():
    log("HomeScreen cron started")

    homescreen_doc = compute_homescreen_snapshot()
    save_homescreen_to_firestore(homescreen_doc)

    log("HomeScreen cron completed")


if __name__ == "__main__":
    main()
