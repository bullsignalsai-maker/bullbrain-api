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

from backend.homescreen_logic import (
    build_homescreen_raw,
    DEFAULT_MAG7,
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
# Firestore handle (same pattern as market_cron)
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


# ---------------------------------------------------------
# Ensure BullBrain model is loaded (once per cron)
# ---------------------------------------------------------
def ensure_bullbrain_loaded():
    if backend.bullbrain_model is not None:
        return

    log("Loading BullBrain model for HomeScreen cron…")
    backend.bullbrain_model = backend.load_bullbrain_model()
    log("BullBrain model loaded successfully for HomeScreen")


# ---------------------------------------------------------
# Build HomeScreen snapshot (MAG7 + Macro)
# ---------------------------------------------------------
def compute_homescreen_snapshot():
    ensure_bullbrain_loaded()

    log("Computing HomeScreen MAG7 snapshot")
    homescreen_raw = build_homescreen_raw(
        universe=DEFAULT_MAG7,
        include_grok=True,
        include_carousel=False,  # macro handled separately
    )

    log("Computing HomeScreen macro snapshot (carousel + market row)")
    macro_snapshot = build_homescreen_macro_snapshot()

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    homescreen_doc = {
        "schema_version": "homescreen_v1",
        "updated_at": now,

        # 🔹 Live market row (authoritative)
        "live_market": macro_snapshot.get("live_market"),

        # 🔹 Carousel cards
        "carousel": macro_snapshot.get("carousel"),

        # 🔹 MAG7 BullBrain snapshot
        "mag7": homescreen_raw.get("mag7"),

        # 🔹 Market mood (derived from MAG7 BullBrain)
        "market_mood": homescreen_raw.get("market_mood"),

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
