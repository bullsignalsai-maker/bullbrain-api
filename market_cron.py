# market_cron.py
# ---------------------------------------------------------
# BullSignalsAI — 15-minute BullBrain scan + Market Pulse
#
# Render Cron:
#   Command : python market_cron.py
#   Schedule: */15 * * * 1-5
# ---------------------------------------------------------
# ADD near the top

from main import (
    _get_market_overview_quick,
    _analyze_headline_sentiment_py,
    _clean_text_py,
    market_news,
)
import pytz

import datetime
import math

import firebase_admin
from firebase_admin import firestore  # type: ignore

import main as backend
from symbols_clean import REAL_TICKERS, COMPANY_NAMES


MARKET_KEYWORDS = [
    "stock", "stocks", "market", "markets", "futures",
    "s&p", "dow", "nasdaq", "indexes",
    "fed", "rates", "yields", "inflation", "cpi", "jobs",
    "earnings", "guidance", "sectors",
    "tech stocks", "financial stocks", "banks"
]

EXCLUDE_KEYWORDS = [
    "death", "killed", "homicide", "crime",
    "relationship", "couples", "psychologist",
    "celebrity", "actor", "actress",
    "obamacare", "health insurance",
    "marijuana", "cannabis",
    "weather", "earthquake"
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


# ---------------------------------------------------------
# Logging helper
# ---------------------------------------------------------
@@ -98,6 +126,27 @@
    return "HOLD"


# ---------------------------------------------------------
# Market Pulse – highlight filter
# ---------------------------------------------------------
def is_market_highlight(item: dict) -> bool:
    title = (item.get("title") or "").lower()
    source = item.get("source")

    if not title or source not in ALLOWED_SOURCES:
        return False

    # Hard exclusions
    if any(bad in title for bad in EXCLUDE_KEYWORDS):
        return False

    # Must contain at least one market keyword
    if any(good in title for good in MARKET_KEYWORDS):
        return True

    return False


# ---------------------------------------------------------
# BUY explanations (with light technical hints)
#   - NO ticker name in short line
@@ -455,6 +504,7 @@
        }



# =========================================================
# 🆕 MARKET HIGHLIGHTS + NEWS (FIRESTORE)
# =========================================================
@@ -467,43 +517,76 @@
    eastern = pytz.timezone("America/New_York")
    utc = pytz.utc

    # -----------------------------------------------------
    # 1) Fetch raw news (unchanged)
    # -----------------------------------------------------
    news_resp = backend.market_news()
    raw_news = news_resp.get("data", []) if isinstance(news_resp, dict) else []

    cleaned = []

    # -----------------------------------------------------
    # 2) Normalize timestamps (unchanged)
    # -----------------------------------------------------
    for n in raw_news:
        try:
            dt_utc = datetime.datetime.fromisoformat(
                n["pubDate"].replace("Z", "")
            ).replace(tzinfo=utc)

            dt_et = dt_utc.astimezone(eastern)

            n["pubDateET"] = dt_et.isoformat()
            n["pubDateObj"] = dt_et
            cleaned.append(n)

        except Exception:
            continue

    cleaned.sort(key=lambda x: x["pubDateObj"], reverse=True)

    titles = [n.get("title", "") for n in cleaned[:80] if n.get("title")]
    # -----------------------------------------------------
    # 3) 🔒 FILTER: ONLY REAL US MARKET HEADLINES (NEW)
    # -----------------------------------------------------
    market_news = [
        n for n in cleaned
        if is_market_highlight(n)
    ]

    # -----------------------------------------------------
    # 4) SENTIMENT (ONLY ON MARKET NEWS)
    # -----------------------------------------------------
    titles = [
        n.get("title", "")
        for n in market_news[:80]
        if n.get("title")
    ]

    analyzed = backend._analyze_headline_sentiment_py(titles)

    bullish = [a["title"] for a in analyzed if a["tag"] == "📈"]
    bearish = [a["title"] for a in analyzed if a["tag"] == "📉"]
    neutral = [a["title"] for a in analyzed if a["tag"] == "⚖️"]

    # Ensure exactly 5 each (existing fallback logic)
    bullish = backend._ensure_five(bullish, "bullish")
    neutral = backend._ensure_five(neutral, "neutral")
    bearish = backend._ensure_five(bearish, "bearish")

    # -----------------------------------------------------
    # 5) NEWS GROUPING (UNCHANGED — uses FULL cleaned list)
    # -----------------------------------------------------
    now_et = datetime.datetime.now(eastern)
    today = now_et.date()
    yesterday = today - datetime.timedelta(days=1)
    week_ago = today - datetime.timedelta(days=7)

    grouped = {"today": [], "yesterday": [], "week": [], "older": []}
    grouped = {
        "today": [],
        "yesterday": [],
        "week": [],
        "older": [],
    }

    for n in cleaned:
        d = n["pubDateObj"].date()
@@ -519,185 +602,188 @@
    for k in grouped:
        grouped[k].sort(key=lambda x: x["pubDateObj"], reverse=True)

    # -----------------------------------------------------
    # 6) FINAL DOCUMENT (SCHEMA UNCHANGED)
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
        "news_grouped": grouped,
        "updated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
    }


# =========================================================
# 🆕 SAVE MARKET PULSE DOCS
# =========================================================
def save_market_pulse_docs(overview_doc, pulse_doc):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("market_overview_live").set(overview_doc, merge=True)
    log("💾 Saved bullsignals_ai/market_overview_live")

    col.document("market_pulse").set(pulse_doc, merge=True)
    log("💾 Saved bullsignals_ai/market_pulse")


def build_market_overview_live():
    overview = _get_market_overview_quick()

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return {
        **overview,
        "updated_at": now,
    }

def build_market_pulse():
    eastern = pytz.timezone("America/New_York")
    utc = pytz.utc

    # ----------------------------------------------------
    # 1) Fetch news (same source as before)
    # ----------------------------------------------------
    news_resp = market_news()
    raw_news = news_resp.get("data", []) if isinstance(news_resp, dict) else []

    cleaned = []

    for n in raw_news:
        try:
            dt_utc = datetime.datetime.fromisoformat(
                n["pubDate"].replace("Z", "")
            ).replace(tzinfo=utc)

            dt_et = dt_utc.astimezone(eastern)
            n["pubDateET"] = dt_et.isoformat()
            n["pubDateObj"] = dt_et

            cleaned.append(n)
        except:
            continue

    # Latest first
    cleaned.sort(key=lambda x: x["pubDateObj"], reverse=True)

    # ----------------------------------------------------
    # 2) Sentiment analysis (top ~80)
    # ----------------------------------------------------
    titles = [_clean_text_py(n.get("title", "")) for n in cleaned[:80]]
    analyzed = _analyze_headline_sentiment_py(titles)

    bullish_raw = [a["title"] for a in analyzed if a["tag"] == "📈"]
    bearish_raw = [a["title"] for a in analyzed if a["tag"] == "📉"]
    neutral_raw = [a["title"] for a in analyzed if a["tag"] == "⚖️"]

    # Deduplicate
    bullish_raw = list(dict.fromkeys(bullish_raw))
    bearish_raw = list(dict.fromkeys(bearish_raw))
    neutral_raw = list(dict.fromkeys(neutral_raw))

    bullish = bullish_raw[:5]
    bearish = bearish_raw[:5]
    neutral = neutral_raw[:5]

    highlights_numeric = {
        "bull": len(bullish_raw),
        "bear": len(bearish_raw),
        "neutral": len(neutral_raw),
    }

    # ----------------------------------------------------
    # 3) Group news by date (ET)
    # ----------------------------------------------------
    grouped = {"today": [], "yesterday": [], "week": [], "older": []}

    now_et = datetime.datetime.now(eastern)
    today = now_et.date()
    yesterday = today - datetime.timedelta(days=1)
    week_ago = today - datetime.timedelta(days=7)

    for n in cleaned:
        d = n["pubDateObj"].date()
        if d == today:
            grouped["today"].append(n)
        elif d == yesterday:
            grouped["yesterday"].append(n)
        elif d >= week_ago:
            grouped["week"].append(n)
        else:
            grouped["older"].append(n)

    for k in grouped:
        grouped[k].sort(key=lambda x: x["pubDateObj"], reverse=True)

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return {
        "highlights_grouped": {
            "bullish": bullish,
            "neutral": neutral,
            "bearish": bearish,
        },
        "highlights_numeric": highlights_numeric,
        "news_grouped": grouped,
        "updated_at": now,
    }


def save_market_pulse_docs(overview_doc, pulse_doc):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("market_overview_live").set(overview_doc, merge=True)
    log("💾 Saved bullsignals_ai/market_overview_live")

    col.document("market_pulse").set(pulse_doc, merge=True)
    log("💾 Saved bullsignals_ai/market_pulse")


# =========================================================
# ENTRYPOINT
# =========================================================
def main():
    started = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat().replace("+00:00", "Z")
    log(f"Market cron started at {started}")

    try:
        # 🔒 Existing
        hotlist_doc, bearwatch_doc = compute_hotlist_and_bearwatch()
        save_docs_to_firestore(hotlist_doc, bearwatch_doc)

        # 🆕 New
        overview_doc = compute_market_overview()
        pulse_doc = compute_market_pulse()
        save_market_pulse_docs(overview_doc, pulse_doc)

        finished = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        log(f"Market cron completed at {finished}")

    except Exception as e:
        log(f"Fatal error in market_cron: {e}")


if __name__ == "__main__":