from typing import Dict, Any, List
from datetime import datetime, timezone

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
    # 2️⃣ Build snapshot items (Firestore is truth)
    # -----------------------------------------------------
    for sym in symbols:
        stock = get_stock(sym) or {}

        # ── NARRATIVES (Firestore-first) ──
        narratives = stock.get("narratives") or {}
        watchlist_summary = (
            narratives.get("signal")
            or narratives.get("summary")
            or narratives.get("probability")
        )

        bullbrain = stock.get("bullbrain") or {}
        raw = bullbrain.get("raw") or {}

        pattern = stock.get("pattern") or {}
        pattern_hist = stock.get("patternHistory") or {}
        fwd = pattern_hist.get("forwardReturns") or {}
        days5 = fwd.get("days5") or {}

        stock_quote = stock.get("quote") or {}
        price = stock_quote.get("price")
        change_pct = stock_quote.get("changePct")

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

            # ── QUOTE ──
            "quote": {
                "price": price,
                "change": change,
                "changePct": change_pct,
            },
            "quote_updated_at": stock_quote.get("updated_at"),

            # ── BULLBRAIN ──
            "bullbrain": {
                "signal": bullbrain.get("signal", "HOLD"),
                "confidence": bullbrain.get("confidence", 0),
                "prob_up": raw.get("prob_up"),
                "prob_down": raw.get("prob_down"),
            },

            # ── PATTERN ──
            "pattern": {
                "name": pattern.get("pattern") or pattern.get("patternLabel"),
                "bias": pattern.get("bias") or pattern.get("patternBias"),
                "winRate": days5.get("winRate"),
            },

            # ── OPTIONAL UI DATA ──
            "sparkline": sparkline if isinstance(sparkline, list) else [],

            # ── HUMAN SUMMARY ──
            "grokSummary": watchlist_summary,

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
        "version": "v2",
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
