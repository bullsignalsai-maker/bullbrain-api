# backend/news/market_news_cleaner.py
import re
from typing import List, Dict, Any
from urllib.parse import urlparse

BLOCK_KEYWORDS = {
    "mom", "dad", "family", "dementia", "health",
    "mortgage", "house", "retirement", "lifestyle",
    "cancer", "illness", "benefits", "social security",
}

BLOCK_PHRASES = {
    "my ", " i ", " we ", " you ",
    "should you", "what you need", "here's why",
    "is it time", "how to",
}

MARKET_KEYWORDS = {
    "earnings", "revenue", "profit", "loss",
    "shares", "stock", "stocks",
    "guidance", "forecast",
    "merger", "acquisition",
    "sec", "filing", "insider",
    "downgrade", "upgrade",
}

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

def is_strictly_market_news(title: str, summary: str, ticker: str | None) -> bool:
    text = f"{title} {summary}".lower()

    # ❌ Block personal / lifestyle / advice
    if any(k in text for k in BLOCK_KEYWORDS):
        return False

    if any(p in text for p in BLOCK_PHRASES):
        return False

    # ✅ Allow if valid ticker exists
    if ticker:
        return True

    # ✅ Allow only if clear market action language exists
    if any(k in text for k in MARKET_KEYWORDS):
        return True

    return False


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
        summary = clean_summary(it.get("summary", ""), title)

        # 🔒 STRICT MARKET FILTER (THIS IS THE GUARDRAIL)
        if not is_strictly_market_news(title, summary, ticker):
            continue

        cleaned.append({
            "title": title,
            "summary": summary,
            "link": link,
            "pubDate": it.get("pubDate"),
            "source": normalize_source(link),
            "ticker": ticker,
            "category": "Market",
        })

    cleaned.sort(key=lambda x: x.get("pubDate") or "", reverse=True)
    return cleaned[:80]
