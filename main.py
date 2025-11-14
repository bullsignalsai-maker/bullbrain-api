# main.py — BullSignalsAI Backend (Production)

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import datetime

app = FastAPI()

# Allow Expo mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # You can restrict later to your app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# Load keys from environment (Render dashboard)
# ----------------------------------------------------------
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

MODEL = "grok-4-fast-reasoning"
GROK_STOCK_CACHE_HOURS = 6
WATCH_GROK_CACHE_HOURS = 24

# Simple in-memory cache for Grok and watchlist summaries
cache = {}

# ----------------------------------------------------------
@app.get("/")
def root():
    return {"status": "BullSignalsAI Backend Running"}

# ----------------------------------------------------------
# Helper: backend quote fetch (Finnhub + Yahoo fallback)
# ----------------------------------------------------------
def backend_fetch_quote(symbol: str):
    symbol = symbol.upper()
    try:
        quote = None
        profile = {}

        if FINNHUB_KEY:
            # Finnhub quote
            q_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
            quote = requests.get(q_url, timeout=8).json()

            # Finnhub profile for name (non-critical)
            try:
                p_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_KEY}"
                profile = requests.get(p_url, timeout=8).json()
            except Exception:
                profile = {}

        # If Finnhub failed or price bad → Yahoo fallback
        if not quote or "c" not in quote or quote["c"] in [None, 0]:
            y_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d"
            y = requests.get(y_url, timeout=8).json()
            meta = (
                y.get("chart", {})
                .get("result", [{}])[0]
                .get("meta", {})
            )
            close = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            if close is None:
                return None

            change = (close - prev) if prev else 0.0
            change_pct = ((close - prev) / prev * 100) if prev else 0.0
            now_ts = int(datetime.datetime.utcnow().timestamp())
            return {
                "symbol": symbol,
                "name": profile.get("name") or symbol,
                "current": float(close),
                "change": float(change),
                "changePct": float(change_pct),
                "high": float(close),
                "low": float(close),
                "open": float(prev) if prev else float(close),
                "prevClose": float(prev) if prev else float(close),
                "timestamp": now_ts,
            }

        # Normal Finnhub case
        price = float(quote["c"])
        prev = float(quote.get("pc") or 0) or price
        change = float(quote.get("d") or (price - prev))
        change_pct = float(quote.get("dp") or ((price - prev) / prev * 100 if prev else 0))

        return {
            "symbol": symbol,
            "name": profile.get("name") or symbol,
            "current": price,
            "change": change,
            "changePct": change_pct,
            "high": float(quote.get("h") or price),
            "low": float(quote.get("l") or price),
            "open": float(quote.get("o") or prev),
            "prevClose": float(prev),
            "timestamp": int(quote.get("t") or datetime.datetime.utcnow().timestamp()),
        }

    except Exception as e:
        print("backend_fetch_quote error:", e)
        return None

# ----------------------------------------------------------
# 1. Quote (Primary Finnhub, fallback Yahoo)
# ----------------------------------------------------------
@app.get("/quote/{symbol}")
def quote(symbol: str):
    try:
        q = backend_fetch_quote(symbol)
        if not q:
            return {"error": "Quote unavailable"}

        return {
            "price": q["current"],
            "change": q["change"],
            "changePct": q["changePct"],
            "high": q["high"],
            "low": q["low"],
            "open": q["open"],
            "prevClose": q["prevClose"],
            "timestamp": q["timestamp"],
        }

    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 2. Analyst recommendations
# ----------------------------------------------------------
@app.get("/recommendations/{symbol}")
def recommendations(symbol: str):
    try:
        if not FINNHUB_KEY:
            return {"data": []}
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_KEY}"
        data = requests.get(url, timeout=8).json()
        return {"data": data}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 3. Generic Grok/XAI summary (kept if used elsewhere)
# ----------------------------------------------------------
@app.post("/grok-summary")
def grok_summary(payload: dict):
    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        url = "https://api.x.ai/v1/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 4. Full ticker data (combines /quote + /recommendations)
# ----------------------------------------------------------
@app.get("/ticker-full/{symbol}")
def ticker_full(symbol: str):
    try:
        q = backend_fetch_quote(symbol)
        rec_data = recommendations(symbol)
        return {
            "symbol": symbol.upper(),
            "quote": q,
            "recommendations": rec_data,
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 5. Multiple quotes
# ----------------------------------------------------------
@app.get("/quotes")
def quotes(symbols: str):
    try:
        out = {}
        for s in symbols.split(","):
            s = s.strip().upper()
            if not s:
                continue
            q = backend_fetch_quote(s)
            out[s] = q
        return out
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 6. Macro Watch (FMP)
# ----------------------------------------------------------
@app.get("/macro-watch")
def macro_watch():
    try:
        today = datetime.date.today()
        to_date = today + datetime.timedelta(days=10)

        url = (
            "https://financialmodelingprep.com/api/v3/economic_calendar"
            f"?from={today}&to={to_date}&apikey={FMP_API_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        return {"data": data[:20] if isinstance(data, list) else []}
    except Exception as e:
        return {"data": [], "error": str(e)}

# ----------------------------------------------------------
# 7. Earnings
# ----------------------------------------------------------
@app.get("/earnings")
def earnings():
    try:
        today = datetime.date.today()
        next_week = today + datetime.timedelta(days=7)

        url = (
            "https://financialmodelingprep.com/api/v3/earning_calendar"
            f"?from={today}&to={next_week}&apikey={FMP_API_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        return {"data": data[:20] if isinstance(data, list) else []}
    except Exception as e:
        return {"data": [], "error": str(e)}

# ----------------------------------------------------------
# 8. Live stats (fear & greed + VIX + S&P)
# ----------------------------------------------------------
@app.get("/stats/live")
def live_stats():
    try:
        # Fear & Greed Index (placeholder – can wire RapidAPI later)
        fearGreed = {"value": 50, "label": "Neutral"}

        # VIX
        vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
        vix_data = requests.get(vix_url, timeout=10).json()
        vix = (
            vix_data.get("chart", {})
            .get("result", [{}])[0]
            .get("meta", {})
            .get("regularMarketPrice", 15)
        )

        # S&P Change %
        sp_url = "https://query1.finance.yahoo.com/v8/finance/chart/^GSPC"
        sp_data = requests.get(sp_url, timeout=10).json()
        sp_meta = (
            sp_data.get("chart", {})
            .get("result", [{}])[0]
            .get("meta", {})
        )
        prev = sp_meta.get("previousClose")
        sp_change = (
            (sp_meta.get("regularMarketPrice") - prev) / prev * 100
            if prev
            else 0
        )

        return {
            "fearGreed": fearGreed,
            "vix": round(vix, 2),
            "sp500_change": round(sp_change, 2),
        }
    except Exception as e:
        return {
            "fearGreed": {"value": 50, "label": "Neutral"},
            "vix": 14.5,
            "sp500_change": 0.2,
            "error": str(e),
        }

# ----------------------------------------------------------
# 9. Market Mood (Fear & Greed + VIX) — for MoodService
# ----------------------------------------------------------
@app.get("/market-mood")
def market_mood():
    try:
        # Fear & Greed Index
        fng = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=5,
        ).json()

        fear_value = int(fng.get("data", [{}])[0].get("value", 50))
        fear_label = fng.get("data", [{}])[0].get("value_classification", "Neutral")

        # VIX Index
        vix_json = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
            timeout=5,
        ).json()

        vix_price = (
            vix_json.get("chart", {})
            .get("result", [{}])[0]
            .get("meta", {})
            .get("regularMarketPrice", 15.0)
        )

        return {
            "data": {
                "fearGreed": {"value": fear_value, "label": fear_label},
                "vix": round(float(vix_price), 2),
            }
        }

    except Exception as e:
        return {
            "data": {
                "fearGreed": {"value": 50, "label": "Neutral"},
                "vix": 15.0,
            },
            "error": str(e),
        }

# ----------------------------------------------------------
# 10. Grok Stock Analysis (for StockDetailScreen)
# ----------------------------------------------------------
@app.get("/grok-stock/{symbol}")
def grok_stock(symbol: str, force: bool = False):
    """Full structured Grok analysis for StockDetailScreen."""
    now = datetime.datetime.utcnow()
    key = f"grok_stock_{symbol.upper()}"

    # Cache
    if not force:
        item = cache.get(key)
        if item:
            age_hours = (now - item["time"]).total_seconds() / 3600
            if age_hours < GROK_STOCK_CACHE_HOURS:
                return {"text": item["text"], "updatedAt": item["time"].isoformat()}

    quote = backend_fetch_quote(symbol)

    price_context = (
        f"Current Price: {quote['current']}\n"
        f"Change: {quote['change']} ({quote['changePct']:.2f}%)\n"
        f"Day Range: {quote['low']} – {quote['high']}\n"
        f"Open: {quote['open']}\n"
        f"Prev Close: {quote['prevClose']}\n"
        f"Company: {quote['name']}\n"
        if quote
        else f"Symbol: {symbol.upper()}"
    )

    prompt = f"""
Analyze {symbol.upper()} using this structure:

AI Signal
Predictions
Executive Summary
Key Statistics
Technical Outlook
News & Market Sentiment
Risks & Opportunities
Trade Idea
Recommendation

Market Context:
{price_context}

Keep each section concise but meaningful. Include NFA disclaimer at end.
"""

    try:
        if not XAI_API_KEY:
            raise RuntimeError("Missing XAI_API_KEY")

        res = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.45,
                "max_tokens": 1500,
            },
            timeout=20,
        )
        j = res.json()
        text = (
            j.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not text:
            text = "⚠️ AI analysis unavailable."

        cache[key] = {"text": text, "time": now}
        return {"text": text, "updatedAt": now.isoformat()}

    except Exception as e:
        print("GROK STOCK ERROR:", e)
        return {"text": "⚠️ AI analysis unavailable.", "updatedAt": None}

# ----------------------------------------------------------
# 11. Market News (for AIPulse / NewsFeedService)
# ----------------------------------------------------------
@app.get("/market-news")
def market_news():
    import feedparser

    FEEDS = [
        "https://www.benzinga.com/rss/stock-news.xml",
        "https://seekingalpha.com/api/sa/combined/global_news.rss",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://www.investing.com/rss/news.rss",
        "https://www.zacks.com/rss/news.xml",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,TSLA,MSFT,NVDA,META,AMZN,GOOGL,AMD,INTC,JPM,BAC,GS&region=US&lang=en-US",
    ]

    KEYWORDS = [
        "dow", "nasdaq", "s&p", "fed", "inflation", "cpi", "ppi",
        "earnings", "guidance", "profit", "loss", "upgrade", "downgrade",
        "ipo", "merger", "acquisition", "forecast", "ai", "chip",
        "market", "stock", "recession", "treasury", "jobs", "rate", "futures",
    ]

    news = []

    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:20]:
                title = getattr(e, "title", "")
                summary = getattr(e, "summary", "")
                link = getattr(e, "link", "")
                pub_date = getattr(e, "published", datetime.datetime.utcnow().isoformat())

                text = (title + summary).lower()
                if any(k in text for k in KEYWORDS):
                    news.append({
                        "title": title,
                        "summary": (summary or "")[:220] + "...",
                        "link": link,
                        "pubDate": pub_date,
                        "source": getattr(e, "source", {}).get("title", "News"),
                    })
        except Exception as ex:
            print("RSS error:", ex)

    # Deduplicate by first 40 chars
    seen = set()
    uniq = []
    for n in news:
        key = (n["title"] or "")[:40].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(n)

    # Sort by pubDate where possible
    try:
        uniq.sort(
            key=lambda x: x["pubDate"],
            reverse=True,
        )
    except Exception:
        pass

    return {"data": uniq[:50]}

# ----------------------------------------------------------
# 12. SEARCH + WATCHLIST endpoints (for WatchlistScreen)
# ----------------------------------------------------------

def compute_signal_and_conf(change_pct: float):
    try:
        cp = float(change_pct or 0.0)
    except Exception:
        cp = 0.0

    if cp > 0.8:
        signal = "BUY"
    elif cp < -0.8:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(95, max(70, abs(cp) * 10 + 70))
    return signal, int(round(confidence))


def build_watchlist_item(symbol: str):
    symbol = symbol.upper()
    q = backend_fetch_quote(symbol)
    if not q:
        return {
            "symbol": symbol,
            "price": 0.0,
            "changePct": 0.0,
            "signal": "HOLD",
            "confidence": 75,
            "sentimentSummary": "Live data temporarily unavailable; showing neutral placeholder.",
        }

    price = q.get("current") or q.get("price") or 0.0
    change_pct = q.get("changePct") or 0.0
    signal, confidence = compute_signal_and_conf(change_pct)

    # Grok-style short sentiment line with backend cache
    now = datetime.datetime.utcnow()
    cache_key = f"watch_grok_{symbol}"
    summary = None

    # Try cache
    item = cache.get(cache_key)
    if item:
        age_hours = (now - item["time"]).total_seconds() / 3600
        if age_hours < WATCH_GROK_CACHE_HOURS:
            summary = item["text"]

    # If no cached summary and Grok key is present, call Grok
    if not summary and XAI_API_KEY:
        try:
            prompt = (
                f"In one concise line (max 15 words), describe {symbol}'s "
                f"market trend given a daily move of {change_pct:.2f}%."
            )
            res = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}"},
                json={
                    "model": "grok-beta",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 40,
                    "temperature": 0.6,
                },
                timeout=12,
            )
            j = res.json()
            text = (
                j.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if text:
                summary = text
                cache[cache_key] = {"text": summary, "time": now}
        except Exception as e:
            print("Watchlist Grok error:", e)

    # Fallback summaries if Grok not available / empty
    if not summary:
        if signal == "BUY":
            summary = "Strong bullish activity detected."
        elif signal == "SELL":
            summary = "Bearish pressure observed."
        else:
            summary = "Market appears neutral."

    try:
        price_val = float(price)
    except Exception:
        price_val = 0.0

    try:
        cp_val = float(change_pct)
    except Exception:
        cp_val = 0.0

    return {
        "symbol": symbol,
        "price": round(price_val, 2),
        "changePct": round(cp_val, 2),
        "signal": signal,
        "confidence": confidence,
        "sentimentSummary": summary,
    }


@app.get("/search")
def search(q: str, limit: int = 5):
    """Autocomplete for tickers (used by WatchlistScreen)."""
    try:
        if not FINNHUB_KEY:
            return {"data": []}

        url = f"https://finnhub.io/api/v1/search?q={q}&token={FINNHUB_KEY}"
        data = requests.get(url, timeout=8).json()
        out = []
        for item in data.get("result", [])[:limit]:
            sym = item.get("symbol")
            desc = item.get("description")
            if sym and desc:
                out.append({"symbol": sym, "description": desc})
        return {"data": out}
    except Exception as e:
        print("SEARCH error:", e)
        return {"data": []}


@app.get("/watchlist-item/{symbol}")
def watchlist_item(symbol: str):
    """Single watchlist item data for a ticker."""
    try:
        return build_watchlist_item(symbol)
    except Exception as e:
        return {"error": str(e)}


@app.get("/watchlist-batch")
def watchlist_batch(symbols: str = Query(..., description="Comma-separated tickers")):
    """Optional batch endpoint if needed later."""
    try:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        data = [build_watchlist_item(s) for s in sym_list]
        return {"data": data}
    except Exception as e:
        return {"error": str(e)}
