# backend/stock_repo.py
# ---------------------------------------------------------
# Global Stock Repository (Firestore)
# - Read / write cached BullBrain intelligence
# - TTL-based freshness check
# ---------------------------------------------------------

import datetime
from typing import Optional, Dict, Any

import firebase_admin
from firebase_admin import firestore  # type: ignore


# ---------------------------------------------------------
# Firestore handle (safe, shared)
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


# ---------------------------------------------------------
# Time helpers
# ---------------------------------------------------------
def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def get_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch stocks/{SYMBOL} doc if exists.
    """
    db = get_db()
    doc = db.collection("stocks").document(symbol.upper()).get()
    if not doc.exists:
        return None
    return doc.to_dict()


def is_stock_fresh(doc: Dict[str, Any]) -> bool:
    """
    Checks TTL freshness using compute_ttl_minutes.
    """
    try:
        computed_at = doc.get("computed_at")
        ttl = int(doc.get("compute_ttl_minutes", 60))

        if not computed_at:
            return False

        ts = datetime.datetime.fromisoformat(
            computed_at.replace("Z", "")
        ).replace(tzinfo=datetime.timezone.utc)

        age_min = (utc_now() - ts).total_seconds() / 60.0
        return age_min <= ttl

    except Exception:
        return False


def save_stock(symbol: str, payload: Dict[str, Any]) -> None:
    """
    Upserts stocks/{SYMBOL}.
    """
    db = get_db()
    db.collection("stocks").document(symbol.upper()).set(payload, merge=True)
