# backend/active_symbols.py
# ---------------------------------------------------------
from typing import Dict
import datetime

import firebase_admin
from firebase_admin import firestore  # type: ignore

THROTTLE_MINUTES = 30


# ---------------------------------------------------------
# Firestore (MATCH quote_repo EXACTLY)
# ---------------------------------------------------------
def _db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _minutes_between(now_iso: str, past_iso: str) -> float:
    now = datetime.datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    past = datetime.datetime.fromisoformat(past_iso.replace("Z", "+00:00"))
    return (now - past).total_seconds() / 60.0


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def touch_active_symbol(symbol: str) -> None:
    print("TOUCH ACTIVE SYMBOL:", symbol)

    db = _db()
    print("🔥 FIRESTORE PROJECT:", db._client.project)
    ref = db.collection("bullsignals_ai").document("active_symbols")
    now = _now_iso()

    def txn(tx):
        snap = ref.get(transaction=tx)
        data = snap.to_dict() or {}
        symbols: Dict[str, Dict] = data.get("symbols", {})

        entry = symbols.get(symbol, {})
        last_seen = entry.get("last_seen")
        count = entry.get("count", 0)

        entry["last_seen"] = now

        if not last_seen or _minutes_between(now, last_seen) >= THROTTLE_MINUTES:
            entry["count"] = count + 1
        else:
            entry["count"] = count

        symbols[symbol] = entry

        tx.set(
            ref,
            {
                "symbols": symbols,
                "updated_at": now,
            },
            merge=True,
        )

    db.transaction()(txn)
