# backend/quote_repo.py
# ---------------------------------------------------------
# Quote Repository (Firestore)
# - Global quote cache
# - TTL-based freshness
# - Worker signaling via needs_refresh
# ---------------------------------------------------------

from typing import Optional, Dict, Any
import datetime

import firebase_admin
from firebase_admin import firestore  # type: ignore


# ---------------------------------------------------------
# Firestore init (safe)
# ---------------------------------------------------------
def _db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_ts(ts: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(
            ts.replace("Z", "")
        ).replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return None


def _quote_doc(symbol: str):
    return (
        _db()
        .collection("bullsignals_ai")
        .document("quotes")
        .collection("symbols")
        .document(symbol.upper())
    )


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
    doc = _quote_doc(symbol).get()
    if not doc.exists:
        return None
    return doc.to_dict()


def is_quote_fresh(quote: Dict[str, Any]) -> bool:
    try:
        updated_at = _parse_ts(quote.get("updated_at"))
        ttl = int(quote.get("ttl_seconds", 30))

        if not updated_at:
            return False

        age = (_now_utc() - updated_at).total_seconds()
        return age <= ttl
    except Exception:
        return False


def mark_needs_refresh(symbol: str) -> None:
    _quote_doc(symbol).set(
        {
            "needs_refresh": True,
        },
        merge=True,
    )


def save_quote(symbol: str, payload: Dict[str, Any]) -> None:
    """
    Used ONLY by quote_worker.
    """
    payload = {
        **payload,
        "symbol": symbol.upper(),
        "updated_at": _now_utc().isoformat().replace("+00:00", "Z"),
        "needs_refresh": False,
    }

    _quote_doc(symbol).set(payload, merge=True)


def get_quote_safe(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Safe read:
    - returns quote if exists
    - never throws
    - never marks refresh
    """
    try:
        return get_quote(symbol)
    except Exception:
        return None


# ---------------------------------------------------------
# Pending quotes (on-demand support)
# ---------------------------------------------------------

def get_pending_quotes(limit: int = 50):
    """
    Returns symbols that requested quotes but are not fresh yet.
    Used by quote_worker background job.
    """
    db = get_db()
    now = utc_now()

    pending = []

    docs = (
        db.collection("bullsignals_ai")
        .document("quotes")
        .collection("symbols")
        .stream()
    )

    for doc in docs:
        d = doc.to_dict() or {}
        last = d.get("last_fetch")

        # never fetched → pending
        if not last:
            pending.append(doc.id)
            continue

        try:
            last_dt = datetime.datetime.fromisoformat(
                last.replace("Z", "")
            ).replace(tzinfo=datetime.timezone.utc)

            age = (now - last_dt).total_seconds()
            if age > QUOTE_TTL_SECONDS:
                pending.append(doc.id)
        except Exception:
            pending.append(doc.id)

        if len(pending) >= limit:
            break

    return pending
