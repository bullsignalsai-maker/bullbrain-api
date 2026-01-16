# backend/news/market_news_sources.py
import feedparser
import datetime
from typing import List, Dict, Any

FEEDS = [
    "https://seekingalpha.com/api/sa/combined/global_news.rss",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://finance.yahoo.com/rss/topstories",
    "https://finance.yahoo.com/topic/earnings/rss",
    "https://finance.yahoo.com/rss/tech",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
]


def fetch_market_news_raw(limit_per_feed: int = 25) -> List[Dict[str, Any]]:
    """
    Fetch raw market-wide news.
    NO filtering, NO cleaning, NO Firestore.
    """
    items: List[Dict[str, Any]] = []

    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:limit_per_feed]:
                try:
                    pub_dt = datetime.datetime(*e.published_parsed[:6])
                except Exception:
                    pub_dt = datetime.datetime.utcnow()

                items.append({
                    "title": getattr(e, "title", "") or "",
                    "summary": getattr(e, "summary", "") or "",
                    "link": getattr(e, "link", "") or "",
                    "pubDate": pub_dt.isoformat(),
                })
        except Exception:
            continue

    return items
