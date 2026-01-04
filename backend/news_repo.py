# backend/news_repo.py
# ---------------------------------------------------------
# Standalone News Fetcher (NO imports from main.py)
# ---------------------------------------------------------
from typing import Any, Dict, List, Optional
import os
import time
import requests

FINNHUB_KEY = os.getenv("FINNHUB_KEY")


def fetch_symbol_news(symbol: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch recent company news from Finnhub.
    Returns a lightweight list safe for UI.
    No Firestore read/write here. No main.py imports.

    Output item shape:
      {
        "headline": str,
        "summary": str,
        "url": str,
        "source": str,
        "datetime": int,
        "image": str|None
      }
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return []

    if not FINNHUB_KEY:
        return []

    # Finnhub needs from/to dates (YYYY-MM-DD)
    # We'll take ~14 days window; UI only needs a few.
    now = int(time.time())
    days = 14 * 86400
    frm = time.strftime("%Y-%m-%d", time.gmtime(now - days))
    to = time.strftime("%Y-%m-%d", time.gmtime(now))

    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": frm, "to": to, "token": FINNHUB_KEY}

    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
        data = r.json()
        if not isinstance(data, list):
            return []

        items: List[Dict[str, Any]] = []
        for it in data[: max(1, limit)]:
            if not isinstance(it, dict):
                continue
            items.append(
                {
                    "headline": it.get("headline") or "",
                    "summary": it.get("summary") or "",
                    "url": it.get("url") or "",
                    "source": it.get("source") or "",
                    "datetime": it.get("datetime"),
                    "image": it.get("image") or None,
                }
            )
        return items
    except Exception:
        return []
