# backend/active_symbols.py

from typing import Optional
from backend.firestore_utils import get_db, iso_now

import datetime

THROTTLE_MINUTES = 30

def _minutes_between(now_iso: str, past_iso: str) -> float:
    now = datetime.datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    past = datetime.datetime.fromisoformat(past_iso.replace("Z", "+00:00"))
    return (now - past).total_seconds() / 60.0


def touch_active_symbol(symbol: str) -> None:
    db = get_db()
    ref = db.collection("bullsignals_ai").document("active_symbols")

    now = iso_now()

    def txn(tx):
        snap = ref.get(transaction=tx)
        data = snap.to_dict() or {}
        symbols = data.get("symbols", {})

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
