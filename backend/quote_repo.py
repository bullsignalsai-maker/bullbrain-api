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


def is_quote_fresh(quote: Dict[str, Any], max_age_seconds: Optional[int] = None) -> bool:
    """
    max_age_seconds overrides the doc's own ttl_seconds (and the 30s
    default) when the caller has a legitimately looser freshness bar
    than quote_worker's real-time refresh contract — e.g. market_cron
    treating "quote_worker is actively cycling this symbol" as fresh.
    """
    ts = _parse_ts(quote.get("updated_at"))
    if not ts:
        return False
    age = (_now_utc() - ts).total_seconds()
    ttl = max_age_seconds if max_age_seconds is not None else int(quote.get("ttl_seconds", QUOTE_TTL_SECONDS))
    return age <= ttl


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

    SCALABLE RULES:
    ✅ Always allow control flags to persist (needs_refresh)
    ✅ Never overwrite last good price/changePct with None
    ✅ Only write numeric fields when they are valid numbers
    ✅ Merge-friendly (preserves other stored fields)
    """

    if not isinstance(payload, dict):
        return

    sym = symbol.upper().strip()
    doc_ref = _quote_doc(sym)

    # Read existing ONLY when the caller needs the needs_refresh fallback below --
    # this is the only place `existing` is used in this function. Every current
    # caller that already knows its own needs_refresh value (market_cron.py,
    # main.py, quote_worker.py's crypto seed/main equity loop) skips this read
    # entirely; only the crypto-refresh path (quote_worker.py) still relies on it.
    existing = {}
    if payload.get("needs_refresh") is None:
        try:
            doc = doc_ref.get()
            if doc.exists:
                existing = doc.to_dict() or {}
        except Exception:
            existing = {}

    # Incoming fields
    incoming_price = payload.get("price")
    incoming_chg = payload.get("changePct")

    # Validate numeric inputs
    def as_num(v):
        try:
            if v is None:
                return None
            if isinstance(v, bool):
                return None
            return float(v)
        except Exception:
            return None

    price = as_num(incoming_price)
    chg = as_num(incoming_chg)

    # Control flags (must persist even when quote data is missing)
    incoming_needs = payload.get("needs_refresh")
    if incoming_needs is None:
        incoming_needs = existing.get("needs_refresh", False)
    needs_refresh = bool(incoming_needs)

    # Build final doc (merge)
    final_doc: Dict[str, Any] = {
        "symbol": sym,
        "updated_at": _now_iso(),
        "ttl_seconds": QUOTE_TTL_SECONDS,
        "needs_refresh": needs_refresh,
    }

    # ✅ Only write price/changePct when we have valid numbers
    # Otherwise preserve last good values by NOT writing them at all.
    if price is not None:
        final_doc["price"] = price
    if chg is not None:
        final_doc["changePct"] = chg

    # Preserve optional provider fields ONLY if present (and don't null-poison)
    for k in ["change", "open", "high", "low", "prevClose", "timestamp", "source"]:
        if k in payload and payload[k] is not None:
            final_doc[k] = payload[k]

    # If payload explicitly says source, keep it (even if quote missing)
    if "source" in payload and payload.get("source"):
        final_doc["source"] = payload["source"]

    doc_ref.set(final_doc, merge=True)


def get_quote_safe(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        return get_quote(symbol)
    except Exception:
        return None


# ---------------------------------------------------------
# Pending quotes for worker
# ---------------------------------------------------------
def get_pending_quotes(limit: int = 3) -> List[str]:
    """
    Quotes needing refresh:
      - needs_refresh == True
      - OR stale based on updated_at

    IMPORTANT:
      - Skip crypto quote docs (source == 'crypto')
        because they should NOT be fetched via Finnhub equity endpoint.
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

        # ✅ Skip crypto docs from equity refresh loop
        if str(d.get("source") or "").lower() == "crypto":
            continue

        if d.get("needs_refresh") is True:
            out.append(doc.id)
        else:
            ts = _parse_ts(d.get("updated_at"))
            if not ts or (now - ts).total_seconds() > QUOTE_TTL_SECONDS:
                out.append(doc.id)

        if len(out) >= limit:
            break

    return out
