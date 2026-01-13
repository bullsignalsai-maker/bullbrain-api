# backend/watchlist_snapshot.py
# ------------------------------------------------------------
# Watchlist Snapshot Builder
# UI reads ONLY this snapshot
# ------------------------------------------------------------

from typing import Dict, Any, List
from firebase_admin import firestore

from backend.quote_repo import get_quote_safe
from backend.stock_repo import get_stock
from backend.time_utils import utc_now_iso


COL_ROOT = "bullsignals_ai"
DOC_SNAPSHOTS = "watchlist_snapshots"


# ------------------------------------------------------------
# Read user intent (source of truth)
# ------------------------------------------------------------
def _read_user_watchlist_symbols(user_id: str) -> List[str]:
    db = firestore.client()
    docs = (
        db.collection("users")
          .document(user_id)
          .collection("watchlist")
          .stream()
    )
    return sorted({d.id.upper() for d in docs if d.id})


# ------------------------------------------------------------
# Build snapshot (CORE)
# ------------------------------------------------------------
def build_watchlist_snapshot(user_id: str) -> Dict[str, Any]:
    db = firestore.client()
    symbols = _read_user_watchlist_symbols(user_id)

    items: List[Dict[str, Any]] = []

    for sym in symbols:
        # -----------------------------
        # Quote (live, refreshed by worker)
        # -----------------------------
        quote = get_quote_safe(sym) or {}
        price = quote.get("price")

        # -----------------------------
        # Stock intelligence
        # -----------------------------
        stock = get_stock(sym) or {}
        bullbrain = stock.get("bullbrain") or {}
        insights = stock.get("insights") or {}
        stock_quote = stock.get("quote") or {}

        items.append({
            # identity
            "symbol": sym,
            "companyName": stock.get("company_name", sym),

            # 🔥 live price
            "price": price,
            "changePct": quote.get("changePct"),
            "quote_updated_at": quote.get("updated_at"),

            # 🧠 AI signal (UI contract)
            "hybridSignal": bullbrain.get("signal", "HOLD"),
            "hybridScore": bullbrain.get("confidence", 0),

            # 📊 OHLC (safe fallback)
            "features": {
                "open":  stock_quote.get("open")  or price,
                "high":  stock_quote.get("high")  or price,
                "low":   stock_quote.get("low")   or price,
                "close": price,
            },

            # 💬 explanation
            "grokSummary": (
                insights.get("oneLiner")
                or insights.get("summaryLine")
                or "Market signal based on trend and momentum."
            ),

            # timestamps
            "computed_at": stock.get("computed_at"),
        })

    snapshot = {
        "user_id": user_id,
        "symbols": symbols,
        "count": len(items),
        "items": items,
        "snapshot_version": "v1",
        "generated_at": utc_now_iso(),
        "ttl_seconds": 300,  # 5 minutes (NOT quote TTL)
    }

    # --------------------------------------------------------
    # Write snapshot (correct Firestore path)
    # --------------------------------------------------------
    db.collection(COL_ROOT) \
      .document(DOC_SNAPSHOTS) \
      .collection("users") \
      .document(user_id) \
      .set(snapshot, merge=True)

    return snapshot


# ------------------------------------------------------------
# Snapshot freshness helper
# ------------------------------------------------------------
def is_snapshot_fresh(snapshot: Dict[str, Any], max_age_seconds: int = 300) -> bool:
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


# ------------------------------------------------------------
# Read snapshot (used by API if needed)
# ------------------------------------------------------------
def get_watchlist_snapshot(user_id: str) -> Dict[str, Any] | None:
    db = firestore.client()
    snap = (
        db.collection(COL_ROOT)
          .document(DOC_SNAPSHOTS)
          .collection("users")
          .document(user_id)
          .get()
    )
    return snap.to_dict() if snap.exists else None
