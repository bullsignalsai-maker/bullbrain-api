# backend/watchlist_snapshot.py

from typing import List, Dict, Any
from firebase_admin import firestore
from backend.quote_repo import get_quote_safe
from backend.stock_repo import get_stock
from backend.firestore_utils import utc_now_iso


COL_ROOT = "bullsignals_ai"
COL_SNAPSHOTS = "watchlist_snapshots"


def build_watchlist_snapshot(user_id: str, symbols: List[str]) -> Dict[str, Any]:
    db = firestore.client()
    items = []

    for sym in symbols:
        sym = sym.upper()

        # -------------------------------------------------
        # 1️⃣ FULL INTELLIGENCE (PRIMARY SOURCE)
        # -------------------------------------------------
        stock = get_stock(sym)
        if not stock:
            # symbol exists in watchlist but not yet computed
            items.append({
                "symbol": sym,
                "companyName": sym,
                "price": None,
                "changePct": None,
                "hybridSignal": "HOLD",
                "hybridScore": 0,
                "features": {},
                "grokSummary": "Intelligence is being prepared.",
            })
            continue

        bullbrain = stock.get("bullbrain", {})
        insights = stock.get("insights", {})
        stock_quote = stock.get("quote", {}) or {}

        # -------------------------------------------------
        # 2️⃣ LIVE QUOTE OVERLAY (SOURCE OF TRUTH FOR PRICE)
        # -------------------------------------------------
        live_quote = get_quote_safe(sym) or {}

        price = (
            live_quote.get("price")
            or stock_quote.get("price")
        )

        change_pct = (
            live_quote.get("changePct")
            if live_quote.get("changePct") is not None
            else stock_quote.get("changePct")
        )

        open_px = (
            live_quote.get("open")
            or stock_quote.get("open")
            or price
        )

        high_px = (
            live_quote.get("high")
            or stock_quote.get("high")
            or price
        )

        low_px = (
            live_quote.get("low")
            or stock_quote.get("low")
            or price
        )

        # -------------------------------------------------
        # 3️⃣ UI-CONTRACT-COMPATIBLE ITEM
        # -------------------------------------------------
        items.append({
            "symbol": sym,
            "companyName": stock.get("company_name", sym),

            # 🔥 LIVE PRICE
            "price": price,
            "changePct": change_pct,
            "timestamp": (
                live_quote.get("updated_at")
                or stock_quote.get("updated_at")
            ),

            # 🧠 AI SIGNAL (MATCH UI)
            "hybridSignal": bullbrain.get("signal", "HOLD"),
            "hybridScore": bullbrain.get("confidence", 0),

            # 📊 OHLC
            "features": {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": price,
            },

            # 💬 WHY
            "grokSummary": (
                insights.get("oneLiner")
                or insights.get("summaryLine")
                or "Signal based on trend, momentum, and volatility."
            ),

            "computed_at": stock.get("computed_at"),
        })

    snapshot = {
        "user_id": user_id,
        "symbols": symbols,
        "items": items,
        "snapshot_version": "v1",
        "generated_at": utc_now_iso(),
        "ttl_seconds": 30,
    }

    db.collection(COL_ROOT).collection(COL_SNAPSHOTS) \
        .document(user_id).set(snapshot, merge=True)

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


def get_watchlist_snapshot(user_id: str) -> Dict[str, Any] | None:
    db = firestore.client()
    snap = (
        db.collection(COL_ROOT)
          .collection(COL_SNAPSHOTS)
          .document(user_id)
          .get()
    )
    return snap.to_dict() if snap.exists else None
