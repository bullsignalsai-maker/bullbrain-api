# backend/quote_demand.py
# ---------------------------------------------------------
# Quote Demand Layer
# - Read-safe
# - Non-blocking
# - Signals quote_worker when refresh is needed
# ---------------------------------------------------------

from typing import Dict, Any

from backend.quote_repo import (
    get_quote,
    is_quote_fresh,
    mark_needs_refresh,
    save_quote,
)
from backend.quote_repo import _now_utc  # internal helper is fine here


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def ensure_quote(symbol: str) -> Dict[str, Any]:
    """
    Ensures a quote exists for the symbol.

    Behavior:
    - If fresh → return quote
    - If stale → signal refresh, return stale
    - If missing → create placeholder, signal refresh, return placeholder

    NEVER blocks.
    NEVER calls external APIs.
    """

    symbol = symbol.upper().strip()

    # 1️⃣ Try existing quote
    quote = get_quote(symbol)

    if quote:
        if is_quote_fresh(quote):
            return quote

        # stale → signal worker
        mark_needs_refresh(symbol)
        return quote

    # 2️⃣ No quote exists → create placeholder
    placeholder = {
        "symbol": symbol,
        "price": None,
        "change": None,
        "changePct": None,
        "open": None,
        "high": None,
        "low": None,
        "prevClose": None,

        "source": "pending",
        "ttl_seconds": 30,
        "needs_refresh": True,
        "updated_at": _now_utc().isoformat().replace("+00:00", "Z"),
    }

    save_quote(symbol, placeholder)
    return placeholder
