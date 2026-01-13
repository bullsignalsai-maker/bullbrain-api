# backend/stock_repo.py
# ---------------------------------------------------------
# Global Stock Repository (Firestore)
# Canonical source for BullBrain stock intelligence
#
# READ / WRITE PATH:
#   /bullsignals_ai/stocks/symbols/{SYMBOL}
# ---------------------------------------------------------

import datetime
from typing import Optional, Dict, Any

from backend.firestore_utils import get_db, utc_now_iso

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
COL_ROOT = "bullsignals_ai"
DOC_STOCKS = "stocks"
COL_SYMBOLS = "symbols"


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def get_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch stock intelligence for a symbol.

    Path:
      /bullsignals_ai/stocks/symbols/{SYMBOL}
    """
    if not symbol:
        return None

    symbol = symbol.upper()

    doc = (
        get_db()
        .collection(COL_ROOT)
        .document(DOC_STOCKS)
        .collection(COL_SYMBOLS)
        .document(symbol)
        .get()
    )

    return doc.to_dict() if doc.exists else None


def save_stock(symbol: str, payload: Dict[str, Any]) -> None:
    """
    Upsert stock intelligence for a symbol.

    Path:
      /bullsignals_ai/stocks/symbols/{SYMBOL}
    """
    if not symbol or not isinstance(payload, dict):
        return

    symbol = symbol.upper()

    payload.setdefault("symbol", symbol)
    payload.setdefault("computed_at", utc_now_iso())

    get_db() \
        .collection(COL_ROOT) \
        .document(DOC_STOCKS) \
        .collection(COL_SYMBOLS) \
        .document(symbol) \
        .set(payload, merge=True)


def is_stock_fresh(doc: Dict[str, Any], max_age_minutes: int = 60) -> bool:
    """
    TTL freshness check using computed_at.
    """
    try:
        computed_at = doc.get("computed_at")
        if not computed_at:
            return False

        ts = datetime.datetime.fromisoformat(
            computed_at.replace("Z", "+00:00")
        )
        age_min = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds() / 60
        return age_min <= max_age_minutes

    except Exception:
        return False
