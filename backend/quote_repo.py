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
    """
    Returns the full quote document.
    Backward-compatible: existing callers expecting
    price/changePct still work.
    """
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
    """
    Saves quote payload safely.

    HARD RULES:
    - ❌ Never overwrite with empty / null quotes
    - ❌ Never write price=None + changePct=None
    - ✅ Preserve last known good values
    """

    if not isinstance(payload, dict):
        return

    price = payload.get("price")
    chg = payload.get("changePct")

    # -------------------------------------------------
    # GUARD 1: reject fully empty quotes
    # -------------------------------------------------
    if price is None and chg is None:
        return

    # -------------------------------------------------
    # GUARD 2: numeric validation
    # -------------------------------------------------
    if price is not None:
        try:
            price = float(price)
        except Exception:
            price = None

    if chg is not None:
        try:
            chg = float(chg)
        except Exception:
            chg = None

    # If both invalid → do nothing
    if price is None and chg is None:
        return

    final_doc = {
        "symbol": symbol.upper(),
        "updated_at": _now_iso(),
        "ttl_seconds": QUOTE_TTL_SECONDS,
        "needs_refresh": False,
    }

    # Only write what is valid
    if price is not None:
        final_doc["price"] = price

    if chg is not None:
        final_doc["changePct"] = chg

    # Preserve optional provider fields
    for k in ["open", "high", "low", "prevClose", "timestamp", "source"]:
        if k in payload:
            final_doc[k] = payload[k]

    _quote_doc(symbol).set(final_doc, merge=True)


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
