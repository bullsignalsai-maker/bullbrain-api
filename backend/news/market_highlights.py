# backend/news/market_highlights.py

from typing import Dict, Any, List
from collections import Counter

# ---------------------------------------------------------
# Keyword maps (STRICTLY market-related)
# ---------------------------------------------------------

BULLISH_KEYWORDS = {
    "beats", "beat estimates", "strong earnings", "revenue growth",
    "raises guidance", "upgraded", "record profit",
    "surge", "rally", "gain", "jump", "outperform",
}

BEARISH_KEYWORDS = {
    "misses", "missed estimates", "cuts guidance", "downgraded",
    "loss", "decline", "falls", "plunge",
    "layoffs", "warning", "weak demand", "slump",
}

SECTOR_KEYWORDS = {
    "ai": "AI",
    "artificial intelligence": "AI",
    "semiconductor": "Semiconductors",
    "chip": "Semiconductors",
    "bank": "Financials",
    "financial": "Financials",
    "energy": "Energy",
    "oil": "Energy",
    "gas": "Energy",
    "health": "Healthcare",
    "pharma": "Healthcare",
    "biotech": "Healthcare",
    "retail": "Consumer",
    "consumer": "Consumer",
    "auto": "Automotive",
    "ev": "Automotive",
    "crypto": "Crypto",
    "bitcoin": "Crypto",
    "ethereum": "Crypto",
    "xrp":'"Crypto",
    "binance":'"Crypto",
    "solana":'"Crypto",

}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def classify_news_sentiment(title: str, summary: str) -> str:
    """
    Deterministic sentiment classification.
    No opinions, no probabilities.
    """
    text = f"{title} {summary}".lower()

    if any(k in text for k in BULLISH_KEYWORDS):
        return "bullish"

    if any(k in text for k in BEARISH_KEYWORDS):
        return "bearish"

    return "neutral"


def detect_sector(title: str, summary: str) -> str | None:
    """
    Light sector/theme detection.
    """
    text = f"{title} {summary}".lower()
    for key, sector in SECTOR_KEYWORDS.items():
        if key in text:
            return sector
    return None


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def build_market_highlights(
    news: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Auto-generate market highlights from cleaned market news.

    Output:
    {
      "bullish": [...],
      "neutral": [...],
      "bearish": [...]
    }
    """

    buckets: Dict[str, List[str]] = {
        "bullish": [],
        "neutral": [],
        "bearish": [],
    }

    # Count (sentiment, sector) occurrences
    counts: Counter = Counter()

    for item in news:
        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()

        if not title:
            continue

        sentiment = classify_news_sentiment(title, summary)
        sector = detect_sector(title, summary)

        if sector:
            counts[(sentiment, sector)] += 1

    # Build concise, professional highlights
    for (sentiment, sector), count in counts.items():
        if count < 2:
            continue  # avoid noise from single headlines

        if sentiment == "bullish":
            buckets["bullish"].append(
                f"{sector} stocks showed strength following positive developments."
            )

        elif sentiment == "bearish":
            buckets["bearish"].append(
                f"{sector} stocks faced pressure amid negative news flow."
            )

        else:
            buckets["neutral"].append(
                f"{sector} stocks traded mixed as markets digested fresh data."
            )

    # Fallback: ensure UI always has something meaningful
    if not any(buckets.values()):
        buckets["neutral"].append(
            "Markets traded mixed as investors assessed ongoing developments."
        )

    return buckets
