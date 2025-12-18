# backend/market_data.py

import os
import time
import requests
from typing import Dict, Any, List, Optional


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

FINNHUB_KEY = os.getenv("FINNHUB_KEY")
POLYGON_KEY = os.getenv("POLYGON_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY") or os.getenv("NEWSDATA_API_KEY")


# ------------------------------------------------------------
# Quote (Finnhub)
# ------------------------------------------------------------

def fetch_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch real-time quote from Finnhub.
    """
    if not FINNHUB_KEY:
        return None

    try:
        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": symbol, "token": FINNHUB_KEY}
        r = requests.get(url, params=params, timeout=6)
        j = r.json()

        if not j or "c" not in j:
            return None

        return {
            "current": j.get("c"),
            "change": j.get("d"),
            "changePct": j.get("dp"),
            "high": j.get("h"),
            "low": j.get("l"),
            "open": j.get("o"),
            "prevClose": j.get("pc"),
            "timestamp": int(time.time()),
            "source": "finnhub",
        }
    except Exception:
        return None


# ------------------------------------------------------------
# Daily Candles (Polygon-style payload)
# ------------------------------------------------------------

def fetch_daily_candles(symbol: str, days: int = 250) -> Optional[Dict[str, Any]]:
    """
    Fetch daily OHLCV candles.
    Output format matches existing BullBrain expectations.
    """
    if not POLYGON_KEY:
        return None

    try:
        end = int(time.time() * 1000)
        start = end - (days * 86400000)

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start}/{end}"
        )

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": days,
            "apiKey": POLYGON_KEY,
        }

        r = requests.get(url, params=params, timeout=8)
        j = r.json()

        results = j.get("results") or []
        if not results:
            return None

        return {
            "open": [c["o"] for c in results],
            "high": [c["h"] for c in results],
            "low": [c["l"] for c in results],
            "close": [c["c"] for c in results],
            "volume": [c["v"] for c in results],
            "timestamp": [c["t"] for c in results],
            "source": "polygon",
        }
    except Exception:
        return None


# ------------------------------------------------------------
# News (Lightweight, Headlines Only)
# ------------------------------------------------------------

def fetch_symbol_news(symbol: str, limit: int = 8) -> List[str]:
    """
    Fetch recent headlines for symbol.
    Returned as clean string list (UI + Grok friendly).
    """
    if not NEWS_API_KEY:
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY,
        }

        r = requests.get(url, params=params, timeout=6)
        j = r.json()

        articles = j.get("articles") or []
        headlines = []

        for a in articles:
            title = a.get("title")
            if title:
                headlines.append(title.strip())
            if len(headlines) >= limit:
                break

        return headlines
    except Exception:
        return []
