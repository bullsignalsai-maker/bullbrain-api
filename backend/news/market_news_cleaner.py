# backend/news/market_news_cleaner.py

import re
from typing import List, Dict, Any
from urllib.parse import urlparse

# ---------------------------------------------------------
# 🚫 HARD BLOCK — Personal / Lifestyle / Advice
# ---------------------------------------------------------

BLOCK_KEYWORDS = {
    # Personal / life / consumer advice
    "mom", "dad", "family", "marriage", "divorce", "dating",
    "retirement", "retiree", "retirees", "social security",
    "medicare", "medicaid", "ssi", "benefits",
    "mortgage", "house", "homebuyer", "rent", "landlord",
    "student loan", "credit card", "credit score", "budget", "debt",

    # Health / lifestyle / non-market
    "dementia", "cancer", "illness", "virus", "hantavirus",
    "treatment", "treatments", "outbreak", "cruise", "hospital",
    "weight loss", "diet", "doctor", "medicine",

    # Politics / general policy unless clearly market-moving
    "election", "senate", "congress", "lawmakers",

    # Generic lifestyle
    "vacation", "travel tips", "recipe", "school", "college",
}

BLOCK_PHRASES = {
    "should you", "what you need", "here's why", "heres why",
    "is it time", "how to", "am i", "can i", "do i", "should i",
    "what to know", "things to know", "need to know",
    "my ", "i'm ", "i am ", "we ", "you ","analyst report:",
}

MARKET_KEYWORDS = {
    # Market/index/macro
    "stocks", "stock market", "shares", "wall street", "nasdaq",
    "s&p", "dow", "futures", "yields", "treasury", "fed",
    "inflation", "jobs report", "cpi", "ppi", "gdp","reports", "reported", "results", "estimates", "estimate",
    "target", "raises", "lowers", "initiates", "maintains",
    "overweight", "underweight", "neutral rating",
    "buy rating", "sell rating", "hold rating",
    "premarket", "after hours", "trading", "options",
    "etf", "bitcoin", "crypto", "oil", "energy",

    # Company financials
    "earnings", "revenue", "profit", "loss", "eps",
    "beats", "misses", "guidance", "forecast", "outlook",
    "quarter", "q1", "q2", "q3", "q4",

    # Stock moves / analyst
    "rises", "falls", "jumps", "drops", "surges", "slides",
    "rallies", "plunges", "upgrade", "downgrade", "price target",
    "analyst", "bullish", "bearish",

    # Corporate actions
    "ipo", "filing", "sec", "merger", "acquisition", "deal",
    "buyback", "dividend", "split",
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
    "seekingalpha.com": "Seeking Alpha",
    "barrons.com": "Barron's",
    "wsj.com": "WSJ",
}

# ---------------------------------------------------------
# Core Filters
# ---------------------------------------------------------

def is_strictly_market_news(title: str, summary: str, ticker: str | None) -> bool:
    text = f" {title} {summary} ".lower()

    # ❌ Block question/advice/personal-title style
    if has_blocked_personal_style(title):
        return False

    # ❌ Block personal / lifestyle / health / non-market content
    if any(k in text for k in BLOCK_KEYWORDS):
        return False

    if any(p in text for p in BLOCK_PHRASES):
        return False

    # ❌ Reject junk English-word tickers
    if ticker and ticker in NOISY_TICKERS:
        return False

    # ✅ Strong market language required
    has_market_language = any(k in text for k in MARKET_KEYWORDS)

    # ✅ Ticker alone is not enough anymore.
    # Require ticker + market language.
    if ticker and has_market_language:
        return True

    if has_market_language:
        return True

    return False

def normalize_source(link: str) -> str:
    try:
        domain = urlparse(link).netloc.lower().replace("www.", "")
        return SOURCE_MAP.get(domain, domain.title())
    except Exception:
        return "News"


def has_blocked_personal_style(title: str) -> bool:
    t = f" {title.lower().strip()} "

    if "?" in title:
        return True

    starts = (
        "should ", "can ", "could ", "will ", "would ",
        "is ", "are ", "how ", "why ", "what ",
        "when ", "where ", "who "
    )

    return title.lower().strip().startswith(starts)

def clean_summary(summary: str, title: str) -> str:
    s = re.sub("<[^<]+?>", "", summary or "").strip()
    if not s or s in {"...", ""}:
        return title
    return s[:240]

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
