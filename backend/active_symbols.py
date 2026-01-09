# backend/active_symbols.py
# ---------------------------------------------------------
from typing import Dict
import datetime
import json
import os

import firebase_admin
from firebase_admin import credentials, firestore  # type: ignore

THROTTLE_MINUTES = 30


# ---------------------------------------------------------
# Firestore (EXACT, SAFE, EXPLICIT)
# ---------------------------------------------------------
def _init_firebase_if_needed():
    """
    Initialize Firebase Admin exactly once.
    MUST use FIREBASE_ADMIN_JSON on Render.
    """
    if firebase_admin._apps:
        return

    raw = os.getenv("FIREBASE_ADMIN_JSON")
    if not raw:
        raise RuntimeError("FIREBASE_ADMIN_JSON missing for active_symbols")

    data = json.loads(raw)

    # Fix escaped private key newlines (Render requirement)
    pk = data.get("private_key")
    if isinstance(pk, str):
        data["private_key"] = pk.replace("\\n", "\n")

    cred = credentials.Certificate(data)
    firebase_admin.initialize_app(cred)


def _db():
    _init_firebase_if_needed()
    return firestore.client()


# ---------------------------------------------------------
# Time helpers
# ---------------------------------------------------------
def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _now_iso() -> str:
    return _now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _minutes_between(now_iso: str, past_iso: str) -> float:
    now = datetime.datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    past = datetime.datetime.fromisoformat(past_iso.replace("Z", "+00:00"))
    return (now - past).total_seconds() / 60.0


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def touch_active_symbol(symbol: str) -> None:
    """
    Records user interest in a symbol.
    Creates bullsignals_ai/active_symbols on first write.
    """
    symbol = symbol.upper().strip()
    if not symbol:
        return

    print("TOUCH ACTIVE SYMBOL:", symbol, flush=True)

    db = _db()
    print("🔥 FIRESTORE PROJECT:", db._client.project, flush=True)

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
