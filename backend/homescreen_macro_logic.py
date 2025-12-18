# backend/homescreen_macro_logic.py
# ------------------------------------------------------------
# HomeScreen — Global Macro Snapshot (PURE LOGIC)
#
# Owns:
#   - Live market status
#   - Carousel macro insights
#
# NO Firestore
# NO Cron
# NO FastAPI
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, List
import datetime
import requests


# ------------------------------------------------------------
# Market hours (authoritative)
# ------------------------------------------------------------
def is_us_market_open() -> bool:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_et = now_utc.astimezone(
        datetime.timezone(datetime.timedelta(hours=-5))
    )

    # Monday–Friday
    if now_et.weekday() >= 5:
        return False

    # 9:30 AM – 4:00 PM ET
    if now_et.hour < 9 or now_et.hour > 15:
        return False
    if now_et.hour == 9 and now_et.minute < 30:
        return False

    return True


# ------------------------------------------------------------
# 1️⃣ US Market Indices (Yahoo)
# ------------------------------------------------------------
def build_us_market_insights() -> str:
    indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq",
        "^DJI": "Dow",
    }

    changes = []

    for symbol in indices:
        try:
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{symbol}?range=1d&interval=1d"
            )
            r = requests.get(url, timeout=6)
            data = r.json()

            meta = data["chart"]["result"][0]["meta"]
            close = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose")

            if close and prev:
                pct = ((close - prev) / prev) * 100
                changes.append(pct)

        except Exception:
            continue

    if not changes:
        return "Market data unavailable"

    avg = sum(changes) / len(changes)

    if abs(avg) < 0.05:
        msg = "Flat today"
    elif avg > 0:
        msg = f"U.S. Market up +{avg:.2f}%"
    else:
        msg = f"U.S. Market down {avg:.2f}%"

    if not is_us_market_open():
        msg = f"Market Closed · Last data {msg.replace('U.S. Market ', '')}"

    return msg


# ------------------------------------------------------------
# 2️⃣ Crypto Movers (CoinGecko)
# ------------------------------------------------------------
def build_crypto_movers() -> str:
    try:
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd&order=market_cap_desc&per_page=10&page=1"
        )
        r = requests.get(url, timeout=6)
        coins = r.json()

        movers = sorted(
            coins,
            key=lambda c: c.get("price_change_percentage_24h", 0),
            reverse=True,
        )[:3]

        parts = [
            f"{c['symbol'].upper()} ({c['price_change_percentage_24h']:+.1f}%)"
            for c in movers
            if c.get("price_change_percentage_24h") is not None
        ]

        return ", ".join(parts) if parts else "No major movers"

    except Exception:
        return "Crypto data unavailable"


# ------------------------------------------------------------
# 3️⃣ Fear & Greed Index
# ------------------------------------------------------------
def build_market_sentiment() -> str:
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=6)
        data = r.json()["data"][0]
        return f"{data['value_classification']} ({data['value']})"
    except Exception:
        return "Sentiment unavailable"


# ------------------------------------------------------------
# 4️⃣ Commodities Snapshot (Yahoo)
# ------------------------------------------------------------
def build_commodities_snapshot() -> str:
    commodities = {
        "GC=F": "Gold",
        "CL=F": "Oil",
        "SI=F": "Silver",
    }

    results = []

    for symbol, name in commodities.items():
        try:
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{symbol}?range=1d&interval=1d"
            )
            r = requests.get(url, timeout=6)
            data = r.json()

            meta = data["chart"]["result"][0]["meta"]
            close = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose")

            if close and prev:
                pct = ((close - prev) / prev) * 100
                results.append(f"{name} ({pct:+.1f}%)")

        except Exception:
            continue

    return ", ".join(results) if results else "Commodities unavailable"


# ------------------------------------------------------------
# 5️⃣ Trending Sectors (static for now)
# ------------------------------------------------------------
def build_trending_sectors() -> str:
    return "AI, Tech, Energy"


# ------------------------------------------------------------
# Final Macro Snapshot (UI-ready)
# ------------------------------------------------------------
def build_homescreen_macro_snapshot() -> Dict[str, Any]:
    market_open = is_us_market_open()

    return {
        "live_market": {
            "status": "OPEN" if market_open else "CLOSED",
            "label": "U.S. Market Open" if market_open else "U.S. Market Closed",
        },
        "carousel": [
            {
                "key": "market_insights",
                "title": "AI Market Insights",
                "value": build_us_market_insights(),
            },
            {
                "key": "crypto",
                "title": "Crypto Movers",
                "value": build_crypto_movers(),
            },
            {
                "key": "sentiment",
                "title": "Market Sentiment",
                "value": build_market_sentiment(),
            },
            {
                "key": "commodities",
                "title": "Commodities Snapshot",
                "value": build_commodities_snapshot(),
            },
            {
                "key": "sectors",
                "title": "Trending Sectors",
                "value": build_trending_sectors(),
            },
        ],
    }
