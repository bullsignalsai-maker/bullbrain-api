# ---------------------------------------------------------
# News Fetcher (Circular-import safe)
# ---------------------------------------------------------

from typing import List, Dict, Any


def fetch_symbol_news(symbol: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Safe wrapper for ticker-specific news.

    IMPORTANT:
    - Import is done inside the function to avoid circular imports:
      main → ui_stock_builder → news → market_data → main
    """
    try:
        # Lazy import to avoid circular dependency
        from backend.market_data import get_symbol_news

        n = get_symbol_news(symbol, limit=limit)
        if isinstance(n, dict):
            return n.get("data", []) or []
        if isinstance(n, list):
            return n
    except Exception:
        pass

    return []
