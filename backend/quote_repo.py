# backend/quote_repo.py
# ---------------------------------------------------------
# Quote Repository (Firestore)
# ---------------------------------------------------------

from typing import Optional, Dict, Any, List
import datetime

import firebase_admin
from firebase_admin import firestore  # type: ignore

QUOTE_TTL_SECONDS = 30


# ---------------------------------------------------------
# Firestore
# ---------------------------------------------------------
def _db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


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
    ts = _parse_ts(quote.get("updated_at"))
    if not ts:
        return False
    age = (_now_utc() - ts).total_seconds()
    return age <= int(quote.get("ttl_seconds", QUOTE_TTL_SECONDS))


def mark_needs_refresh(symbol: str) -> None:
    _quote_doc(symbol).set(
        {"needs_refresh": True},
        merge=True,
    )


def clear_needs_refresh(symbol: str) -> None:
    _quote_doc(symbol).set(
        {"needs_refresh": False},
        merge=True,
    )


def save_quote(symbol: str, payload: Dict[str, Any]) -> None:
    payload = {
        **payload,
        "symbol": symbol.upper(),
        "updated_at": _now_iso(),
        "ttl_seconds": QUOTE_TTL_SECONDS,
        "needs_refresh": False,
    }
    _quote_doc(symbol).set(payload, merge=True)


def get_quote_safe(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        return get_quote(symbol)
    except Exception:
        return None


# ---------------------------------------------------------
# Pending quotes for worker
# ---------------------------------------------------------
def get_pending_quotes(limit: int = 50) -> List[str]:
    """
    Quotes needing refresh:
      - needs_refresh == True
      - OR stale based on updated_at
    """
    out: List[str] = []

    docs = (
        _db()
        .collection("bullsignals_ai")
        .document("quotes")
        .collection("symbols")
        .stream()
    )

    now = _now_utc()

    for doc in docs:
        d = doc.to_dict() or {}

        if d.get("needs_refresh") is True:
            out.append(doc.id)
        else:
            ts = _parse_ts(d.get("updated_at"))
            if not ts or (now - ts).total_seconds() > QUOTE_TTL_SECONDS:
                out.append(doc.id)

        if len(out) >= limit:
            break

    return out
