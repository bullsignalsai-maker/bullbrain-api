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
        smart_pattern = stock.get("smartPattern") or {}

        price = quote.get("price")
        change_pct = quote.get("changePct")

        change = None
        try:
            if price is not None and change_pct is not None:
                change = round(price * (float(change_pct) / 100.0), 4)
        except Exception:
            change = None

        sparkline = stock.get("sparkline")

        items.append({
            "symbol": sym,
            "companyName": stock.get("company_name"),

            # ── QUOTE (MATCH homescreen-mag7) ──
            "quote": {
                "price": price,
                "change": change,
                "changePct": change_pct,
            },
            "quote_updated_at": quote.get("updated_at"),

            # ── BULLBRAIN (MATCH homescreen-mag7) ──
            "bullbrain": {
                "signal": bullbrain.get("signal", "HOLD"),
                "confidence": bullbrain.get("confidence", 0),
            },

            # ── SMART PATTERN (MATCH homescreen-mag7) ──
            "pattern": {
                "name": smart_pattern.get("pattern"),
                "winRate": smart_pattern.get("winRate"),
            },

            # ── OPTIONAL UI DATA ──
            "sparkline": sparkline if isinstance(sparkline, list) else [],

            "grokSummary": (
                insights.get("oneLiner")
                or insights.get("summaryLine")
                or "Market signal based on trend and momentum."
            ),

            "updated_at": stock.get("computed_at"),
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
