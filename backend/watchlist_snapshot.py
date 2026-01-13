# backend/watchlist_snapshot.py

from typing import Dict, Any, List
from datetime import datetime, timezone

from backend.quote_repo import get_quote_safe
from backend.stock_repo import get_stock
from backend.firestore_utils import utc_now_iso, get_db


COL_SNAPSHOTS = "watchlist_snapshots"


# =========================================================
# SNAPSHOT BUILDER (SINGLE SOURCE OF TRUTH)
# =========================================================
def build_watchlist_snapshot(user_id: str) -> Dict[str, Any]:
    """
    Builds and stores a watchlist snapshot for a user.

    Source of truth for symbols:
      users/{user_id}/watchlist/*
    """
    db = get_db()

    # -----------------------------------------------------
    # 1️⃣ Read user's watchlist symbols
    # -----------------------------------------------------
    docs = (
        db.collection("users")
          .document(user_id)
          .collection("watchlist")
          .stream()
    )

    symbols: List[str] = sorted({d.id.upper() for d in docs if d.id})
    items: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 2️⃣ Build snapshot items
    # -----------------------------------------------------
    for sym in symbols:
        quote = get_quote_safe(sym) or {}
        stock = get_stock(sym) or {}

        bullbrain = stock.get("bullbrain") or {}
        insights = stock.get("insights") or {}
        stock_quote = stock.get("quote") or {}

        price = quote.get("price")
        sparkline = stock.get("sparkline")
        items.append({
            "symbol": sym,
            "companyName": stock.get("company_name"),

            # Quote (always from quote system)
            "price": price,
            "changePct": quote.get("changePct"),
            "quote_updated_at": quote.get("updated_at"),

            # Signal (hybrid/global intelligence)
            "hybridSignal": bullbrain.get("signal", "HOLD"),
            "hybridScore": bullbrain.get("confidence", 0),

            # Minimal OHLC for UI
            "features": {
                "open": stock_quote.get("open") or price,
                "high": stock_quote.get("high") or price,
                "low": stock_quote.get("low") or price,
                "close": price,
            },
            "sparkline": sparkline if isinstance(sparkline, list) else None,
            # One-liner insight
            "grokSummary": (
                insights.get("oneLiner")
                or insights.get("summaryLine")
                or "Market signal based on trend and momentum."
            ),

            "computed_at": stock.get("computed_at"),
        })

    # -----------------------------------------------------
    # 3️⃣ Persist snapshot
    # -----------------------------------------------------
    snapshot = {
        "user_id": user_id,
        "symbols": symbols,
        "items": items,
        "generated_at": utc_now_iso(),
        "ttl_seconds": 30,
        "version": "v1",
    }

    db.collection(COL_SNAPSHOTS) \
      .document(user_id) \
      .set(snapshot, merge=True)

    return snapshot


# =========================================================
# SNAPSHOT FRESHNESS CHECK
# =========================================================
def is_snapshot_fresh(snapshot: Dict[str, Any], max_age_seconds: int = 30) -> bool:
    try:
        ts = snapshot.get("generated_at")
        if not ts:
            return False

        gen = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - gen).total_seconds()
        return age <= max_age_seconds
    except Exception:
        return False


# =========================================================
# SNAPSHOT READER
# =========================================================
def get_watchlist_snapshot(user_id: str) -> Dict[str, Any] | None:
    db = get_db()
    snap = (
        db.collection(COL_SNAPSHOTS)
          .document(user_id)
          .get()
    )
    return snap.to_dict() if snap.exists else None
