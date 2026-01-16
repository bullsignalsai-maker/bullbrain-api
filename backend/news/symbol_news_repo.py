# backend/news/symbol_news_repo.py
from typing import Any, Dict, List
import os
import time
import requests
import re

FINNHUB_KEY = os.getenv("FINNHUB_KEY")


def fetch_symbol_news(
    symbol: str,
    company_name: str | None = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    STRICT company-only news from Finnhub.
    Used only for Stock Detail screen.
    """
    symbol = (symbol or "").upper().strip()
    if not symbol or not FINNHUB_KEY:
        return []

    name_tokens = []
    if company_name:
        name_tokens = [
            t.lower()
            for t in re.split(r"[ ,.&]", company_name)
            if len(t) > 3
        ]

    now = int(time.time())
    frm = time.strftime("%Y-%m-%d", time.gmtime(now - 14 * 86400))
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

        results = []

        for it in data:
            headline = (it.get("headline") or "").lower()
            summary = (it.get("summary") or "").lower()

            if symbol.lower() not in headline and symbol.lower() not in summary:
                if not any(t in headline or t in summary for t in name_tokens):
                    continue

            results.append({
                "headline": it.get("headline") or "",
                "summary": it.get("summary") or "",
                "url": it.get("url") or "",
                "source": it.get("source") or "",
                "datetime": it.get("datetime"),
                "image": it.get("image") or None,
            })

            if len(results) >= limit:
                break

        return results

    except Exception:
        return []
