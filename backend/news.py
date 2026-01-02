# ---------------------------------------------------------
# News Fetcher (Thin Wrapper)
# ---------------------------------------------------------

from typing import List, Dict, Any
from backend.market_data import get_symbol_news  # wherever it currently lives


def fetch_symbol_news(symbol: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Safe wrapper for ticker-specific news.
    Returns a list (never throws).
    """
    try:
        n = get_symbol_news(symbol, limit=limit)
        if isinstance(n, dict):
            return n.get("data", []) or []
        if isinstance(n, list):
            return n
    except Exception:
        pass

    return []
