# backend/market_highlights_logic.py
# ============================================================
# MARKET HIGHLIGHTS — SENTIMENT + FILTERING (UNCHANGED LOGIC)
# ============================================================

import datetime
from typing import List, Dict, Any

import pytz


# ------------------------------------------------------------
# KEYWORDS & SOURCE FILTERS (UNCHANGED)
# ------------------------------------------------------------
MARKET_KEYWORDS = [
    "stock", "stocks", "market", "markets", "futures",
    "s&p", "dow", "nasdaq", "indexes",
    "fed", "rates", "yields", "inflation", "cpi", "jobs",
    "earnings", "guidance", "sectors",
    "tech stocks", "financial stocks", "banks",
]

EXCLUDE_KEYWORDS = [
    "death", "killed", "homicide", "crime",
    "relationship", "couples", "psychologist",
    "celebrity", "actor", "actress",
    "obamacare", "health insurance",
    "marijuana", "cannabis",
    "weather", "earthquake",
]

ALLOWED_SOURCES = {
    "CNBC",
    "MarketWatch",
    "Bloomberg",
    "Reuters",
    "WSJ",
    "Investing.com",
    "Yahoo",
}


# ------------------------------------------------------------
# MARKET HEADLINE FILTER (UNCHANGED)
# ------------------------------------------------------------
def is_market_highlight(item: Dict[str, Any]) -> bool:
    title = (item.get("title") or "").lower()
    source = item.get("source")

    if not title or source not in ALLOWED_SOURCES:
        return False

    if any(bad in title for bad in EXCLUDE_KEYWORDS):
        return False

    if any(good in title for good in MARKET_KEYWORDS):
        return True

    return False


# ------------------------------------------------------------
# BUILD MARKET HIGHLIGHTS
# ------------------------------------------------------------
def build_market_highlights(
    cleaned_news: List[Dict[str, Any]],
    analyze_headline_sentiment,
    ensure_five_func,
) -> Dict[str, Any]:
    """
    Builds bullish / neutral / bearish market highlights.

    Parameters:
      cleaned_news: output of build_market_news()
      analyze_headline_sentiment: existing sentiment analyzer
      ensure_five_func: your existing fallback filler

    Returns:
      highlights_grouped + highlights_numeric (UNCHANGED SHAPE)
    """

    eastern = pytz.timezone("America/New_York")
    utc = pytz.utc

    # -----------------------------------------------------
    # Normalize timestamps (UNCHANGED)
    # -----------------------------------------------------
    normalized = []

    for n in cleaned_news:
        try:
            dt_utc = datetime.datetime.fromisoformat(
                n["pubDate"].replace("Z", "")
            ).replace(tzinfo=utc)

            dt_et = dt_utc.astimezone(eastern)
            n["pubDateET"] = dt_et.isoformat()
            n["pubDateObj"] = dt_et

            normalized.append(n)
        except Exception:
            continue

    normalized.sort(key=lambda x: x["pubDateObj"], reverse=True)

    # -----------------------------------------------------
    # FILTER: ONLY REAL MARKET HEADLINES
    # -----------------------------------------------------
    market_news = [
        n for n in normalized
        if is_market_highlight(n)
    ]

    # -----------------------------------------------------
    # SENTIMENT ANALYSIS (UNCHANGED)
    # -----------------------------------------------------
    titles = [
        n.get("title", "")
        for n in market_news[:80]
        if n.get("title")
    ]

    analyzed = analyze_headline_sentiment(titles)

    bullish = [a["title"] for a in analyzed if a["tag"] == "📈"]
    bearish = [a["title"] for a in analyzed if a["tag"] == "📉"]
    neutral = [a["title"] for a in analyzed if a["tag"] == "⚖️"]

    # -----------------------------------------------------
    # ENSURE EXACTLY 5 EACH (UNCHANGED FALLBACK)
    # -----------------------------------------------------
    bullish = ensure_five_func(bullish, "bullish")
    neutral = ensure_five_func(neutral, "neutral")
    bearish = ensure_five_func(bearish, "bearish")

    # -----------------------------------------------------
    # FINAL STRUCTURE (UNCHANGED)
    # -----------------------------------------------------
    return {
        "highlights_grouped": {
            "bullish": bullish,
            "neutral": neutral,
            "bearish": bearish,
        },
        "highlights_numeric": {
            "bull": len(bullish),
            "neutral": len(neutral),
            "bear": len(bearish),
        },
    }
