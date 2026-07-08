# backend/watchlist_symbols.py
# ---------------------------------------------------------
# Global Watchlist Symbol Aggregate
#
# Tracks how many users currently have each symbol on their
# watchlist, so the quote worker can give watchlisted symbols
# a guaranteed refresh path (see quote_worker.py's WATCHLIST tier).
#
# Path: /bullsignals_ai/watchlist_symbols
# Shape: { "symbols": { "AAPL": {"count": 2}, ... }, "updated_at": ... }
#
# Counts are maintained via atomic Firestore Increment field
# transforms (not read-modify-write), so concurrent add/remove
# calls for different symbols never clobber each other.
# ---------------------------------------------------------
from firebase_admin import firestore

from backend.firestore_utils import get_db, iso_now


def _ref():
    return get_db().collection("bullsignals_ai").document("watchlist_symbols")


def increment_watchlist_symbol(symbol: str) -> None:
    sym = (symbol or "").upper().strip()
    if not sym:
        return

    _ref().set(
        {
            "symbols": {sym: {"count": firestore.Increment(1)}},
            "updated_at": iso_now(),
        },
        merge=True,
    )


def decrement_watchlist_symbol(symbol: str) -> None:
    sym = (symbol or "").upper().strip()
    if not sym:
        return

    _ref().set(
        {
            "symbols": {sym: {"count": firestore.Increment(-1)}},
            "updated_at": iso_now(),
        },
        merge=True,
    )
