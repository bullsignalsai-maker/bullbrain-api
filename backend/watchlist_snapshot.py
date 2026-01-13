# backend/watchlist_snapshot.py

from typing import List, Dict, Any
from firebase_admin import firestore
from backend.quote_repo import get_quote_safe
from backend.stock_repo import get_stock
from backend.firestore_utils import utc_now_iso


COL_ROOT = "bullsignals_ai"
COL_SNAPSHOTS = "watchlist_snapshots"


def build_watchlist_snapshot(user_id: str, symbols: list[str]):
    db = firestore.client()
    items = []

    for sym in symbols:
        sym = sym.upper()

        quote = get_quote_safe(sym) or {}
        stock = get_stock(sym) or {}

        bullbrain = stock.get("bullbrain") or {}
        insights = stock.get("insights") or {}
        stock_quote = stock.get("quote") or {}

        price = quote.get("price")

        items.append({
            "symbol": sym,
            "companyName": stock.get("company_name"),

            "price": price,
            "changePct": quote.get("changePct"),
            "quote_updated_at": quote.get("updated_at"),

            "hybridSignal": bullbrain.get("signal", "HOLD"),
            "hybridScore": bullbrain.get("confidence", 0),

            "features": {
                "open": stock_quote.get("open") or price,
                "high": stock_quote.get("high") or price,
                "low": stock_quote.get("low") or price,
                "close": price,
            },

            "grokSummary": insights.get("oneLiner")
                or insights.get("summaryLine")
                or "Market signal based on trend and momentum.",

            "computed_at": stock.get("computed_at"),
        })

    snapshot = {
        "user_id": user_id,
        "symbols": symbols,
        "items": items,
        "generated_at": utc_now_iso(),
        "ttl_seconds": 30,
        "version": "v1",
    }

    db.collection("watchlist_snapshots") \
      .document(user_id) \
      .set(snapshot, merge=True)

    return snapshot

def is_snapshot_fresh(snapshot: Dict[str, Any], max_age_seconds: int = 30) -> bool:
    try:
        ts = snapshot.get("generated_at")
        if not ts:
            return False
        from datetime import datetime, timezone
        gen = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - gen).total_seconds()
        return age <= max_age_seconds
    except Exception:
        return False


def get_watchlist_snapshot(user_id: str):
    db = firestore.client()
    snap = (
        db.collection("watchlist_snapshots")
          .document(user_id)
          .get()
    )
    return snap.to_dict() if snap.exists else None

