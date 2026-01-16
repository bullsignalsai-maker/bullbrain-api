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
from typing import Dict, Any
from backend.quote_repo import (
    get_quote,
    is_quote_fresh,
    mark_needs_refresh,
)
from backend.quote_repo import _quote_doc, _now_iso, QUOTE_TTL_SECONDS  # internal ok

def ensure_quote(symbol: str) -> Dict[str, Any]:
    """
    Ensures a quote exists for the symbol.

    Behavior:
    - If fresh → return quote
    - If stale → signal refresh, return stale
    - If missing → create MINIMAL placeholder (no poisoning), signal refresh, return placeholder

    NEVER blocks. NEVER calls external APIs.
    """

    symbol = symbol.upper().strip()

    # 1) Existing quote
    quote = get_quote(symbol)
    if quote:
        if is_quote_fresh(quote):
            return quote
        mark_needs_refresh(symbol)
        return quote

    # 2) Missing quote → create minimal placeholder doc (NO price/change fields)
    _quote_doc(symbol).set(
        {
            "symbol": symbol,
            "needs_refresh": True,
            "updated_at": _now_iso(),
            "ttl_seconds": QUOTE_TTL_SECONDS,
            "source": "pending",
        },
        merge=True,
    )

    return {
        "symbol": symbol,
        "needs_refresh": True,
        "updated_at": _now_iso(),
        "ttl_seconds": QUOTE_TTL_SECONDS,
        "source": "pending",
    }
