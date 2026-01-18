# backend/news/market_news_cleaner.py

import re
from typing import List, Dict, Any
from urllib.parse import urlparse

# ---------------------------------------------------------
# 🚫 HARD BLOCK — Personal / Lifestyle / Advice
# ---------------------------------------------------------

BLOCK_KEYWORDS = {
    "mom", "dad", "family", "dementia", "health",
    "mortgage", "house", "retirement", "retiree",
    "lifestyle", "cancer", "illness",
    "benefits", "social security", "ssi",
    "medicaid", "medicare",
    "student loan", "credit card", "credit score",
    "budget", "debt",
}

BLOCK_PHRASES = {
    "my ", " i ", " we ", " you ",
    "should you", "what you need",
    "here's why", "is it time",
    "how to", "am i", "can i",
    "do i", "should i",
}

# ---------------------------------------------------------
# ✅ STRONG MARKET LANGUAGE (REQUIRED)
# ---------------------------------------------------------

MARKET_KEYWORDS = {
    "earnings", "revenue", "profit", "loss",
    "beats", "misses", "guidance", "forecast",
    "shares", "stock", "stocks",
    "rises", "falls", "jumps", "drops",
    "downgrade", "upgrade",
    "ipo", "filing", "sec",
    "merger", "acquisition",
    "buyback", "dividend",
    "nasdaq", "s&p", "dow",
}

# ---------------------------------------------------------
# ❌ JUNK / ENGLISH-WORD TICKERS
# ---------------------------------------------------------

NOISY_TICKERS = {
    "A", "I", "U", "T", "ON", "UP", "DAY", "IT", "ARE", "HAS",
    "FAST", "COST", "TECH",
    "YOU", "YOUR", "IS", "AS", "IN", "TO", "OF",
}

# ---------------------------------------------------------
# Source normalization
# ---------------------------------------------------------

SOURCE_MAP = {
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "finance.yahoo.com": "Yahoo Finance",
    "investing.com": "Investing.com",
    "zacks.com": "Zacks",
}

# ---------------------------------------------------------
# Core Filters
# ---------------------------------------------------------

def is_strictly_market_news(title: str, summary: str, ticker: str | None) -> bool:
    text = f"{title} {summary}".lower()

    # ❌ Block personal / lifestyle / advice immediately
    if any(k in text for k in BLOCK_KEYWORDS):
        return False

    if any(p in text for p in BLOCK_PHRASES):
        return False

    # ❌ Reject junk English-word tickers
    if ticker and ticker in NOISY_TICKERS:
        return False

    # ✅ Allow if VALID ticker exists
    if ticker:
        return True

    # ✅ Otherwise require STRONG market language
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
    if (
        t.isalpha()
        and 2 <= len(t) <= 5
        and t not in NOISY_TICKERS
    ):
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
        title = (it.get("title") or "").strip()
        link = (it.get("link") or "").strip()
        if not title or not link:
            continue

        key = (title.lower(), link)
        if key in seen:
            continue
        seen.add(key)

        ticker = extract_ticker(title)
        summary = clean_summary(it.get("summary", ""), title)

        # 🔒 STRICT MARKET FILTER (FINAL GATE)
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
