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
    "solana": "Crypto",
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def classify_news_sentiment(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()

    if any(k in text for k in BULLISH_KEYWORDS):
        return "bullish"

    if any(k in text for k in BEARISH_KEYWORDS):
        return "bearish"

    return "neutral"


def detect_sector(title: str, summary: str) -> str | None:
    text = f"{title} {summary}".lower()
    for key, sector in SECTOR_KEYWORDS.items():
        if key in text:
            return sector
    return None


# ---------------------------------------------------------
# Headline Generator (NEW)
# ---------------------------------------------------------

def build_market_headline(
    sentiment_counts: Counter,
    sector_counts: Counter,
) -> str:
    """
    Generate ONE professional market summary headline.
    Deterministic, neutral, market-only.
    """

    if not sentiment_counts:
        return "Markets traded mixed as investors assessed incoming data."

    dominant_sentiment, s_count = sentiment_counts.most_common(1)[0]
    dominant_sector = (
        sector_counts.most_common(1)[0][0]
        if sector_counts else None
    )

    if dominant_sentiment == "bullish":
        if dominant_sector:
            return f"Markets advanced as strength in {dominant_sector} stocks lifted sentiment."
        return "Markets moved higher amid broadly positive developments."

    if dominant_sentiment == "bearish":
        if dominant_sector:
            return f"Markets faced pressure as weakness in {dominant_sector} weighed on sentiment."
        return "Markets declined as negative news flow dampened risk appetite."

    # Neutral / mixed
    if dominant_sector:
        return f"Markets traded mixed as investors evaluated developments in {dominant_sector}."
    return "Markets traded mixed as investors digested fresh data."


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def build_market_highlights(
    news: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Auto-generate market headline + highlights from cleaned market news.

    Output:
    {
      "headline": "Markets traded mixed as AI optimism offset macro uncertainty.",
      "highlights": {
          "bullish": [...],
          "neutral": [...],
          "bearish": [...]
      }
    }
    """

    highlight_buckets: Dict[str, List[str]] = {
        "bullish": [],
        "neutral": [],
        "bearish": [],
    }

    sentiment_counts: Counter = Counter()
    sector_counts: Counter = Counter()

    for item in news:
        title = (item.get("title") or "").strip()
        summary = (item.get("summary") or "").strip()
        if not title:
            continue

        sentiment = classify_news_sentiment(title, summary)
        sector = detect_sector(title, summary)

        sentiment_counts[sentiment] += 1
        if sector:
            sector_counts[sector] += 1

    # Build highlights
    for (sentiment, sector), count in Counter(
        [(classify_news_sentiment(n["title"], n.get("summary", "")),
          detect_sector(n["title"], n.get("summary", "")))
         for n in news if detect_sector(n["title"], n.get("summary", ""))]
    ).items():

        if count < 2 or not sector:
            continue

        if sentiment == "bullish":
            highlight_buckets["bullish"].append(
                f"{sector} stocks showed strength following positive developments."
            )
        elif sentiment == "bearish":
            highlight_buckets["bearish"].append(
                f"{sector} stocks faced pressure amid negative news flow."
            )
        else:
            highlight_buckets["neutral"].append(
                f"{sector} stocks traded mixed as markets digested fresh data."
            )

    # Fallback highlight
    if not any(highlight_buckets.values()):
        highlight_buckets["neutral"].append(
            "Markets traded mixed as investors assessed ongoing developments."
        )

    headline = build_market_headline(sentiment_counts, sector_counts)

    return {
        "headline": headline,
        "highlights": highlight_buckets,
    }
