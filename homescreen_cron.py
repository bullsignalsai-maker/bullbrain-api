# homescreen_cron.py
# ---------------------------------------------------------
# BullSignalsAI — HomeScreen Snapshot Cron
#
# Render Cron:
#   Command : python homescreen_cron.py
#   Schedule: */15 * * * 1-5
# ---------------------------------------------------------

import datetime

import firebase_admin
from firebase_admin import firestore  # type: ignore

import main as backend

from backend.bullbrain import load_bullbrain_model
from backend.homescreen_logic import (
    build_homescreen_mag7_block,
)
from backend.homescreen_macro_logic import (
    build_homescreen_macro_snapshot,
)

# ---------------------------------------------------------
# Logging helper (same style as market_cron)
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
# Build HomeScreen snapshot (MAG7 + Macro)
# ---------------------------------------------------------
def compute_homescreen_snapshot():
    # -----------------------------------------------------
    # Ensure BullBrain model is loaded ONCE
    # -----------------------------------------------------
    if backend.bullbrain_model is None:
        log("Loading BullBrain model for HomeScreen cron…")
        backend.bullbrain_model = load_bullbrain_model()
        log("BullBrain model loaded successfully for HomeScreen")

    # -----------------------------------------------------
    # MAG7 BullBrain snapshot
    # -----------------------------------------------------
    log("Computing HomeScreen MAG7 snapshot")
    mag7_block = build_homescreen_mag7_block()

    # -----------------------------------------------------
    # Macro snapshot (carousel + market row)
    # -----------------------------------------------------
    log("Computing HomeScreen macro snapshot")
    macro_snapshot = build_homescreen_macro_snapshot()

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # -----------------------------------------------------
    # Final Firestore document
    # -----------------------------------------------------
    homescreen_doc = {
        "schema_version": "homescreen_v1",
        "updated_at": now,

        # 🔹 Market status row
        "market": macro_snapshot.get("market"),

        # 🔹 Carousel cards
        "macro": {
            "carousel": macro_snapshot.get("carousel", [])
        },

        # 🔹 MAG7 BullBrain snapshot
        "mag7": mag7_block,

        "meta": {
            "computed_by": "homescreen_cron",
            "refresh_minutes": 15,
            "bullbrain_version": "v2-48f",
        },
    }

    return homescreen_doc


# ---------------------------------------------------------
# Save to Firestore
# ---------------------------------------------------------
def save_homescreen_to_firestore(homescreen_doc):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("homescreen_snapshot").set(homescreen_doc, merge=True)
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
        homescreen_doc = compute_homescreen_snapshot()
        save_homescreen_to_firestore(homescreen_doc)

        finished = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        log(f"HomeScreen cron completed at {finished}")

    except Exception as e:
        log(f"❌ Fatal error in homescreen_cron: {e}")


if __name__ == "__main__":
    main()
