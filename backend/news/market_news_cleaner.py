# backend/news/market_news_cleaner.py
import re
from typing import List, Dict, Any
from urllib.parse import urlparse

NOISY_TICKERS = {
    "A", "I", "U", "T", "ON", "UP", "DAY", "IT", "ARE", "HAS",
    "FAST", "COST", "TECH"
}

SOURCE_MAP = {
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "finance.yahoo.com": "Yahoo Finance",
    "investing.com": "Investing.com",
    "zacks.com": "Zacks",
}


def normalize_source(link: str) -> str:
    try:
        domain = urlparse(link).netloc.lower().replace("www.", "")
        return SOURCE_MAP.get(domain, domain.title())
    except Exception:
        return "News"


def extract_ticker(title: str) -> str | None:
    m = re.search(r"\(([A-Z]{1,5})\)", title)
    if not m:
        return None
    t = m.group(1)
    if t.isalpha() and 2 <= len(t) <= 5 and t not in NOISY_TICKERS:
        return t
    return None


def clean_summary(summary: str, title: str) -> str:
    if not summary or summary.strip() in {"...", ""}:
        return title
    return summary.strip()[:240]


def clean_market_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    cleaned = []

    for it in items:
        title = it.get("title", "").strip()
        link = it.get("link", "").strip()
        if not title or not link:
            continue

        key = (title.lower(), link)
        if key in seen:
            continue
        seen.add(key)

        ticker = extract_ticker(title)

        cleaned.append({
            "title": title,
            "summary": clean_summary(it.get("summary", ""), title),
            "link": link,
            "pubDate": it.get("pubDate"),
            "source": normalize_source(link),
            "ticker": ticker,
            "category": "General",  # optional future upgrade
        })

    cleaned.sort(key=lambda x: x.get("pubDate") or "", reverse=True)
    return cleaned[:80]
