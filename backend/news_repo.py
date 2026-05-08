# backend/news_repo.py
# ---------------------------------------------------------
from typing import Any, Dict, List
import os
import time
import requests
import re

FINNHUB_KEY = os.getenv("FINNHUB_KEY")

BLOCK_KEYWORDS = {
    "mortgage",
    "retirement",
    "medicare",
    "medicaid",
    "social security",
    "student loan",
    "credit card",
    "travel tips",
    "cruise",
    "virus",
    "hantavirus",
    "dementia",
    "cancer",
    "recipe",
    "weight loss",
}

BLOCK_PHRASES = {
    "should you",
    "what you need to know",
    "how to",
    "is it time",
    "can i",
    "do i",
    "analyst report:",
}

TRUSTED_SOURCES = {
    "CNBC",
    "Reuters",
    "Bloomberg",
    "Yahoo",
    "MarketWatch",
    "Benzinga",
    "SeekingAlpha",
    "The Fly",
    "Zacks",
    "Investing.com",
}

def is_relevant_company_news(
    text: str,
    symbol: str,
    company_name: str | None = None,
) -> bool:
    """
    Only allow news clearly related to the requested stock.

    Examples:
    TSLA page should allow:
      - TSLA
      - Tesla
      - Tesla Inc

    TSLA page should block:
      - generic EV market news without Tesla
      - SpaceX/X/Elon-only stories without Tesla
      - broad market news
    """
    t = f" {text.lower()} "
    sym = (symbol or "").lower().strip()

    # 1) Direct ticker mention as a standalone word
    if sym and re.search(rf"\b{re.escape(sym)}\b", t):
        return True

    if not company_name:
        return False

    company = company_name.lower().strip()

    # 2) Full company name match
    if company and company in t:
        return True

    # 3) Strong company/brand token match
    tokens = [
        x.lower()
        for x in re.split(r"[ ,.&()\-]", company_name)
        if len(x) > 3
    ]

    ignore_tokens = {
        "inc", "corp", "corporation", "company", "class",
        "holdings", "holding", "limited", "ltd", "plc",
        "common", "stock"
    }

    strong_tokens = [tok for tok in tokens if tok not in ignore_tokens]

    # For names like Tesla Inc, Tesla alone is strong enough.
    return any(re.search(rf"\b{re.escape(tok)}\b", t) for tok in strong_tokens)

def fetch_symbol_news(
    symbol: str,
    company_name: str | None = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch STRICT company-only news from Finnhub.

    Filters out:
      - ETFs
      - Market-wide articles
      - Other tickers (QQQ, AMD, AMZN, etc.)
    """

    symbol = (symbol or "").upper().strip()
    if not symbol or not FINNHUB_KEY:
        return []

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

        results: List[Dict[str, Any]] = []

        for it in data:
            headline = (it.get("headline") or "").lower()
            summary = (it.get("summary") or "").lower()

            combined = f"{headline} {summary}"

            # -------------------------------------------------
            # Block junk / personal / lifestyle
            # -------------------------------------------------

            if any(k in combined for k in BLOCK_KEYWORDS):
                continue

            if any(p in combined for p in BLOCK_PHRASES):
                continue

            # -------------------------------------------------
            # Trusted source filtering
            # -------------------------------------------------

            source = (it.get("source") or "").strip()

            if source and source not in TRUSTED_SOURCES:
                continue

            # -------------------------------------------------
            # Strong relevance filtering
            # -------------------------------------------------

            if not is_relevant_company_news(
                combined,
                symbol,
                company_name,
            ):
                continue
            existing_urls = {x.get("url") for x in results}

            if (it.get("url") or "") in existing_urls:
                continue
            results.append(
                {
                    "headline": it.get("headline") or "",
                    "summary": (it.get("summary") or "")[:280],
                    "url": it.get("url") or "",
                    "source": it.get("source") or "",
                    "datetime": it.get("datetime"),
                    "image": it.get("image") or None,
                }
            )

            if len(results) >= limit:
                break

        return results

    except Exception:
        return []
