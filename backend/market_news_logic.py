# backend/market_news_logic.py
# ============================================================
# MARKET NEWS — FULL RSS AGGREGATION + CLEANUP (UNCHANGED LOGIC)
# ============================================================

import re
import datetime
from typing import List, Dict, Any
from urllib.parse import urlparse

import feedparser

from sp500_list_optimized import extract_ticker, detect_category


# ------------------------------------------------------------
# RSS FEEDS (UNCHANGED)
# ------------------------------------------------------------
FEEDS = [
    "https://seekingalpha.com/api/sa/combined/global_news.rss",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.investing.com/rss/news.rss",
    "https://www.zacks.com/rss/news.xml",
    "https://finance.yahoo.com/rss/topstories",
    "https://finance.yahoo.com/topic/earnings/rss",
    "https://finance.yahoo.com/rss/tech",
    "https://finance.yahoo.com/rss/pharma",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
]

# ------------------------------------------------------------
# FILTERING KEYWORDS (UNCHANGED)
# ------------------------------------------------------------
BLOCK_KEYWORDS = [
    "why ", "how ", "what ", "should ", "could ",
    "wife", "husband", "family", "children",
    "tv", "celebrity", "gossip",
    "crime", "murder", "scam",
    "recipe", "diet", "health",
    "war", "ukraine", "russia",
]

HARD_KEYWORDS = [
    "earnings", "revenue", "profit", "loss", "guidance", "forecast",
    "ipo", "merger", "acquisition", "m&a",
    "stocks", "market", "dow", "nasdaq", "s&p", "fed",
]

# ------------------------------------------------------------
# SOURCE NORMALIZATION
# ------------------------------------------------------------
SOURCE_MAP = {
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "finance.yahoo.com": "Yahoo Finance",
    "investing.com": "Investing.com",
    "investors.com": "Investor's Business Daily",
    "barrons.com": "Barron's",
}

# Garbage tickers from English words
NOISY_TICKERS = {
    "A", "I", "U", "T", "ON", "UP", "DAY", "IT", "ARE", "HAS",
    "FAST", "COST", "TECH",
}


# ------------------------------------------------------------
# HELPERS (UNCHANGED BEHAVIOR)
# ------------------------------------------------------------
def clean_summary(summary: str | None, title: str) -> str:
    if not summary:
        return title

    s = summary.strip()
    if not s or s == "..." or len(s) < 10:
        return title

    if len(s) > 240:
        return s[:240].rstrip() + "..."

    return s


def normalize_source(source: str | None, link: str | None) -> str:
    if source:
        s = source.strip()
        if s:
            return s

    if not link:
        return "Unknown"

    try:
        domain = urlparse(link).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return SOURCE_MAP.get(domain, domain.split(":")[0].title())
    except Exception:
        return source or "Unknown"


def is_valid_ticker(t: str | None) -> bool:
    if not t:
        return False
    t = t.strip().upper()
    if not (2 <= len(t) <= 5):
        return False
    if not t.isalpha():
        return False
    if t in NOISY_TICKERS:
        return False
    return True


def extract_ticker_from_title(title: str) -> str | None:
    if not title:
        return None
    m = re.search(r"\(([A-Z]{1,5})\)", title)
    if m:
        return m.group(1)
    return None


def extract_ticker_from_url(link: str) -> str | None:
    if not link:
        return None
    try:
        path = urlparse(link).path
        m = re.search(r"-([A-Z]{1,5})-", path)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


def clean_ticker(raw_ticker: str | None, title: str, link: str) -> str | None:
    if raw_ticker and is_valid_ticker(raw_ticker):
        return raw_ticker.strip().upper()

    t = extract_ticker_from_title(title)
    if is_valid_ticker(t):
        return t

    t = extract_ticker_from_url(link)
    if is_valid_ticker(t):
        return t

    return None


def normalize_category(raw_category: str | None, title: str) -> str:
    allowed = {"Earnings", "Fed / Macro", "Tech / AI", "M&A", "Crypto", "General"}
    if raw_category in allowed:
        return raw_category

    txt = f"{raw_category or ''} {title}".lower()

    if any(k in txt for k in ["earnings", "profit", "loss", "guidance"]):
        return "Earnings"
    if any(k in txt for k in ["fed", "rates", "inflation", "jobs", "gdp"]):
        return "Fed / Macro"
    if any(k in txt for k in ["ai", "chip", "semiconductor", "cloud"]):
        return "Tech / AI"
    if any(k in txt for k in ["merger", "acquisition", "ipo"]):
        return "M&A"
    if any(k in txt for k in ["bitcoin", "crypto", "ethereum"]):
        return "Crypto"

    return "General"


# ------------------------------------------------------------
# MAIN BUILDER — USED BY CRON & API
# ------------------------------------------------------------
def build_market_news() -> List[Dict[str, Any]]:
    all_news: List[Dict[str, Any]] = []

    for url in FEEDS:
        try:
            feed = feedparser.parse(url)

            for e in feed.entries[:25]:
                title = getattr(e, "title", "") or ""
                summary_raw = getattr(e, "summary", "") or ""

                if any(b in title.lower() for b in BLOCK_KEYWORDS):
                    continue

                combined = (title + " " + summary_raw).lower()
                allowed = (
                    any(k in combined for k in HARD_KEYWORDS)
                    or extract_ticker(combined.upper())
                )
                if not allowed:
                    continue

                pub_date = getattr(e, "published", None)
                try:
                    pub_dt = datetime.datetime(*e.published_parsed[:6])
                except Exception:
                    pub_dt = datetime.datetime.utcnow()

                link = getattr(e, "link", "") or ""

                raw_ticker = extract_ticker((title + " " + summary_raw).upper())
                ticker = clean_ticker(raw_ticker, title, link)

                category = detect_category((title + summary_raw).upper())
                source = normalize_source(None, link)
                summary = clean_summary(summary_raw, title)

                all_news.append(
                    {
                        "title": title.strip(),
                        "summary": summary,
                        "link": link,
                        "pubDate": pub_dt.isoformat(),
                        "source": source,
                        "ticker": ticker,
                        "category": normalize_category(category, title),
                    }
                )

        except Exception as e:
            print("RSS error:", e)

    # Deduplicate by title
    seen = set()
    result = []
    for n in all_news:
        key = n["title"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(n)

    result.sort(key=lambda x: x["pubDate"], reverse=True)
    return result[:80]
