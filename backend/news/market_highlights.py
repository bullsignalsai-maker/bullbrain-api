# backend/news/market_highlights.py
from typing import Dict, Any, List


def build_market_highlights(
    movers: List[Dict[str, Any]],
    news: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Placeholder for future AI/logic.
    """
    return {
        "headline": "Market Movers Driving Today’s Action",
        "summary": "Semiconductors and AI-linked stocks show strong momentum.",
        "relatedTickers": list({m["symbol"] for m in movers[:5]}),
    }
