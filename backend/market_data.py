# backend/market_data.py
# ------------------------------------------------------------
# Market Data Fetching Layer (Quotes, Candles, News)
# ------------------------------------------------------------

import os
import time
import requests
from typing import Dict, Any, List, Optional

import feedparser


# ------------------------------------------------------------
# Environment / API Keys
# ------------------------------------------------------------

FINNHUB_KEY = os.getenv("FINNHUB_KEY")
POLYGON_KEY = os.getenv("POLYGON_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# ------------------------------------------------------------
# Low-level HTTP helper
# ------------------------------------------------------------

def _safe_json(url: str, timeout: int = 8) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ============================================================
# QUOTES
# ============================================================

def fetch_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Unified quote fetcher with fallback chain:
    Finnhub → FMP → Yahoo (limited)
    """

    symbol = symbol.upper()

    # -----------------------------
    # Finnhub
    # -----------------------------
    if FINNHUB_KEY:
        url = (
            f"https://finnhub.io/api/v1/quote"
            f"?symbol={symbol}&token={FINNHUB_KEY}"
        )
        q = _safe_json(url)
        if q and q.get("c") is not None:
            return {
                "symbol": symbol,
                "price": q.get("c"),
                "change": q.get("d"),
                "changePct": q.get("dp"),
                "high": q.get("h"),
                "low": q.get("l"),
                "open": q.get("o"),
                "prevClose": q.get("pc"),
                "timestamp": int(time.time()),
                "source": "finnhub",
            }

    # -----------------------------
    # FMP fallback
    # -----------------------------
    if FMP_API_KEY:
        url = (
            f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
            f"?apikey={FMP_API_KEY}"
        )
        data = _safe_json(url)
        if isinstance(data, list) and data:
            q = data[0]
            return {
                "symbol": symbol,
                "price": q.get("price"),
                "change": q.get("change"),
                "changePct": q.get("changesPercentage"),
                "high": q.get("dayHigh"),
                "low": q.get("dayLow"),
                "open": q.get("open"),
                "prevClose": q.get("previousClose"),
                "timestamp": int(time.time()),
                "source": "fmp",
            }

    # -----------------------------
    # Yahoo fallback (minimal)
    # -----------------------------
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        data = _safe_json(url)
        result = data["quoteResponse"]["result"][0]
        return {
            "symbol": symbol,
            "price": result.get("regularMarketPrice"),
            "change": result.get("regularMarketChange"),
            "changePct": result.get("regularMarketChangePercent"),
            "high": result.get("regularMarketDayHigh"),
            "low": result.get("regularMarketDayLow"),
            "open": result.get("regularMarketOpen"),
            "prevClose": result.get("regularMarketPreviousClose"),
            "timestamp": int(time.time()),
            "source": "yahoo",
        }
    except Exception:
        return None


# ============================================================
# DAILY CANDLES
# ============================================================

def fetch_daily_candles(
    symbol: str,
    limit: int = 180,
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch daily OHLCV candles.
    ALWAYS returns a list of candle dicts:
      [{open, high, low, close, volume, timestamp}, ...]
    """

    symbol = symbol.upper()

    # -----------------------------
    # Polygon
    # -----------------------------
    if POLYGON_KEY:
        try:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
                f"2020-01-01/{time.strftime('%Y-%m-%d')}"
                f"?adjusted=true&sort=asc&limit={limit}&apiKey={POLYGON_KEY}"
            )
            data = _safe_json(url)
            if data and data.get("results"):
                res = data["results"][-limit:]
                return [
                    {
                        "open": r["o"],
                        "high": r["h"],
                        "low": r["l"],
                        "close": r["c"],
                        "volume": r["v"],
                        "timestamp": r["t"],
                    }
                    for r in res
                ]
        except Exception as e:
            print(f"[market_data] Polygon error {symbol}: {e}")

    # -----------------------------
    # FMP fallback
    # -----------------------------
    if FMP_API_KEY:
        try:
            url = (
                f"https://financialmodelingprep.com/api/v3/historical-price-full/"
                f"{symbol}?apikey={FMP_API_KEY}"
            )
            data = _safe_json(url)
            hist = data.get("historical", [])
            hist = list(reversed(hist))[-limit:]

            return [
                {
                    "open": h["open"],
                    "high": h["high"],
                    "low": h["low"],
                    "close": h["close"],
                    "volume": h["volume"],
                    "timestamp": int(
                        time.mktime(time.strptime(h["date"], "%Y-%m-%d"))
                    ) * 1000,
                }
                for h in hist
            ]
        except Exception as e:
            print(f"[market_data] FMP error {symbol}: {e}")

    return None


# ============================================================
# NEWS
# ============================================================

def fetch_market_news(
    query: str = "stock market",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch market or stock-specific news.
    Used by Market screen & Astra context.
    """

    items: List[Dict[str, Any]] = []

    # -----------------------------
    # NewsAPI
    # -----------------------------
    if NEWS_API_KEY:
        try:
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={query}&language=en&pageSize={limit}"
                f"&apiKey={NEWS_API_KEY}"
            )
            data = _safe_json(url)
            for a in data.get("articles", []):
                items.append(
                    {
                        "title": a.get("title"),
                        "summary": a.get("description"),
                        "source": a.get("source", {}).get("name"),
                        "url": a.get("url"),
                        "publishedAt": a.get("publishedAt"),
                    }
                )
        except Exception:
            pass

    # -----------------------------
    # Google News RSS fallback
    # -----------------------------
    if not items:
        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
        )
        for e in feed.entries[:limit]:
            items.append(
                {
                    "title": e.title,
                    "summary": e.get("summary"),
                    "source": "Google News",
                    "url": e.link,
                    "publishedAt": e.get("published"),
                }
            )

    return items
