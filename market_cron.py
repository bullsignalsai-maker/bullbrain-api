# market_cron.py
# ---------------------------------------------------------
# BullSignalsAI — 15-minute BullBrain scan + Market Pulse
#
# Render Cron:
#   Command : python market_cron.py
#   Schedule: */15 * * * 1-5
# ---------------------------------------------------------
# ADD near the top
import main as backend
from main import (
    backend._get_market_overview_quick()
    backend._analyze_headline_sentiment_py,
    backend._clean_text_py,
    backend.market_news,
)
import pytz
import datetime
import math
import requests
import firebase_admin
from firebase_admin import firestore  # type: ignore
from symbols_clean import REAL_TICKERS, COMPANY_NAMES
from typing import Optional, Dict, Any, List
import time
import random
from backend.candle_store import get_candles


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
# MAG-7 (mandatory on every cron run)
# ---------------------------------------------------------
MAG7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

print("✅ cron get_candles loaded from:", get_candles.__module__)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
# ---------------------------------------------------------
# Logging helper
# ---------------------------------------------------------
def log(msg: str) -> None:
    try:
        backend.log(f"[cron] {msg}")
    except Exception:
        pass
    print(f"[cron] {msg}", flush=True)

# ---------------------------------------------------------
# Firestore handle (SAFE, standalone)
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


# ---------------------------------------------------------
# Fetch previous MAG-7 snapshot from Firestore
# ---------------------------------------------------------
def get_previous_mag7_map() -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "AAPL": { ... previous mag7 object ... },
        "MSFT": { ... },
        ...
      }
    """
    try:
        db = get_db()
        doc = (
            db.collection("bullsignals_ai")
              .document("homescreen_snapshot")
              .get()
        )

        if not doc.exists:
            return {}

        data = doc.to_dict() or {}
        prev_list = data.get("mag7", [])

        return {
            item.get("symbol"): item
            for item in prev_list
            if isinstance(item, dict) and item.get("symbol")
        }

    except Exception:
        return {}


# ---------------------------------------------------------
# Ensure BullBrain model is loaded
# ---------------------------------------------------------
def ensure_bullbrain_loaded():
    if backend.bullbrain_model is not None:
        return

    log("Loading BullBrain model for cron process…")
    backend.bullbrain_model = backend.load_bullbrain_model()
    log("BullBrain model loaded successfully in cron")


# ---------------------------------------------------------
# Safe feature getter
# ---------------------------------------------------------
def safe_feat(feat_dict, key):
    try:
        v = float(feat_dict.get(key, float("nan")))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# ---------------------------------------------------------
# Signal classification (Option C - mixed, a bit more lenient)
# ---------------------------------------------------------
def classify_signal(prob_up: float, prob_down: float) -> str:
    """
    STRONG_BUY / BUY / STRONG_SELL / SELL / HOLD

    Slightly more relaxed than earlier so we actually see some BUYs,
    but still requires a real edge.
    """
    edge_up = prob_up - prob_down
    edge_down = prob_down - prob_up

    # Strong BUY if upside clearly dominates
    if prob_up >= 0.58 and edge_up >= 0.08:
        return "STRONG_BUY"

    # Normal BUY with mild edge
    if prob_up >= 0.52 and edge_up >= 0.02:
        return "BUY"

    # Strong SELL if downside clearly dominates
    if prob_down >= 0.58 and edge_down >= 0.08:
        return "STRONG_SELL"

    # Normal SELL with mild edge
    if prob_down >= 0.52 and edge_down >= 0.02:
        return "SELL"

    return "HOLD"

# ---------------------------------------------------------
# MAG-7 Trend Arrow Calculation
# ---------------------------------------------------------
def compute_trend_arrow(
    current_confidence: float,
    previous_confidence: Optional[float],
) -> str:
    """
    Returns:
      "UP"   -> ⬆️ Bullish momentum increasing
      "DOWN" -> ⬇️ Momentum weakening
      "FLAT" -> ➡️ No meaningful change
    """
    if previous_confidence is None:
        return "FLAT"

    try:
        delta = float(current_confidence) - float(previous_confidence)
    except Exception:
        return "FLAT"

    # Stable thresholds (avoid flip-flop)
    if delta >= 3.0:
        return "UP"
    if delta <= -3.0:
        return "DOWN"

    return "FLAT"


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
# ---------------------------------------------------------
def build_buy_explanations(symbol, prob_up, prob_down, kind, feat_dict):
    up = prob_up * 100.0
    down = prob_down * 100.0

    rsi = safe_feat(feat_dict, "rsi14")
    ret10 = safe_feat(feat_dict, "return_10d")
    price_vs_20 = safe_feat(feat_dict, "price_vs_sma20_pct")
    vol_20 = safe_feat(feat_dict, "volatility_20d")
    vol_vs_ma20 = safe_feat(feat_dict, "volume_vs_ma20_pct")
    trend = safe_feat(feat_dict, "trend_strength_20")

    # Label phrase (Option D style)
    if kind == "STRONG_BUY":
        label = "strong BUY"
    elif kind == "BUY":
        label = "BUY"
    else:
        label = "watchlist BUY"

    # 🔹 Ticker removed from the sentence (only probabilities + label)
    short = (
        f"BullBrain flags a {label} setup — about "
        f"{up:.1f}% chance of upside vs {down:.1f}% downside."
    )

    # Build simple, human + technical hint
    parts = []

    if ret10 is not None:
        parts.append(f"~{ret10:+.1f}% move over the last 10 sessions")

    if price_vs_20 is not None:
        side = "above" if price_vs_20 >= 0 else "below"
        parts.append(f"price is {abs(price_vs_20):.1f}% {side} its 20-day average")

    if rsi is not None:
        if rsi >= 60:
            parts.append(f"RSI ≈ {rsi:.0f} (strong momentum zone)")
        elif rsi >= 50:
            parts.append(f"RSI ≈ {rsi:.0f} (slightly bullish, just above 50)")
        else:
            parts.append(f"RSI ≈ {rsi:.0f} (still not overbought)")

    if vol_vs_ma20 is not None and abs(vol_vs_ma20) >= 15:
        side = "higher" if vol_vs_ma20 > 0 else "lighter"
        parts.append(f"volume is {abs(vol_vs_ma20):.0f}% {side} than 20-day normal")

    if trend is not None:
        if trend >= 0.6:
            parts.append("trend strength looks solid on the 1-month window")
        elif trend <= -0.6:
            parts.append("trend is still shaky, so be extra careful")

    tech_sentence = ""
    if parts:
        tech_sentence = " | ".join(parts[:3]) + ". "

    risk_sentence = (
        "This is not financial advice — consider position sizing, a clear stop-loss, "
        "and your own research before acting."
    )

    explanation_risk = tech_sentence + risk_sentence
    return short, explanation_risk


# ---------------------------------------------------------
# SELL / HOLD explanations (with light technical hints)
#   - NO ticker name in short line
# ---------------------------------------------------------
def build_bear_explanations(symbol, prob_up, prob_down, kind, feat_dict):
    up = prob_up * 100.0
    down = prob_down * 100.0

    rsi = safe_feat(feat_dict, "rsi14")
    ret10 = safe_feat(feat_dict, "return_10d")
    price_vs_20 = safe_feat(feat_dict, "price_vs_sma20_pct")
    vol_20 = safe_feat(feat_dict, "volatility_20d")
    vol_vs_ma20 = safe_feat(feat_dict, "volume_vs_ma20_pct")
    trend = safe_feat(feat_dict, "trend_strength_20")

    if kind == "STRONG_SELL":
        label = "strong SELL"
        short = (
            "Bearish pressure dominates — about "
            f"{down:.1f}% chance of downside vs {up:.1f}% upside."
        )
    elif kind == "SELL":
        label = "SELL"
        short = (
            "Bearish bias — about "
            f"{down:.1f}% chance of downside vs {up:.1f}% upside."
        )
    else:  # HOLD
        label = "HOLD"
        short = (
            "No clear edge — model sees roughly "
            f"{up:.1f}% up vs {down:.1f}% down."
        )

    parts = []

    if ret10 is not None:
        parts.append(f"~{ret10:+.1f}% move over the last 10 sessions")

    if price_vs_20 is not None:
        side = "below" if price_vs_20 <= 0 else "above"
        parts.append(f"price is {abs(price_vs_20):.1f}% {side} its 20-day average")

    if rsi is not None:
        if rsi <= 40:
            parts.append(f"RSI ≈ {rsi:.0f} (weak momentum, below 40)")
        elif rsi <= 50:
            parts.append(f"RSI ≈ {rsi:.0f} (neutral / slightly weak)")
        else:
            parts.append(f"RSI ≈ {rsi:.0f} (still not deeply oversold)")

    if vol_vs_ma20 is not None and abs(vol_vs_ma20) >= 15:
        side = "higher" if vol_vs_ma20 > 0 else "lighter"
        parts.append(f"volume is {abs(vol_vs_ma20):.0f}% {side} than 20-day normal")

    if trend is not None:
        if trend <= -0.6:
            parts.append("downtrend looks strong on the 1-month window")
        elif trend >= 0.6:
            parts.append("trend is mixed — some strength despite this short-term risk")

    tech_sentence = ""
    if parts:
        tech_sentence = " | ".join(parts[:3]) + ". "

    if label in ("strong SELL", "SELL"):
        risk_sentence = (
            "Signal: SELL zone. Many traders use this type of setup to reduce exposure "
            "or tighten stops instead of adding fresh risk."
        )
    else:
        risk_sentence = (
            "Signal: HOLD. Price action looks more sideways/uncertain — waiting for a "
            "clearer edge can often be safer."
        )

    explanation_risk = tech_sentence + risk_sentence
    return short, explanation_risk

# ---------------------------------------------------------
# Main scan: build Hotlist + BearWatch docs
# ---------------------------------------------------------
def compute_hotlist_and_bearwatch():

    ensure_bullbrain_loaded()

    buy_candidates = []
    bear_candidates = []
    all_symbols = []

    total = len(REAL_TICKERS)
    log(f"Scanning {total} tickers with BullBrain…")

    for i, sym in enumerate(REAL_TICKERS, start=1):
        if i % 25 == 0:
            log(f"...processed {i}/{total}")

        # -------------------------------------------------
        # 🔒 SAFE candle fetch with 429 protection
        # -------------------------------------------------
        try:
            candles = get_candles(sym, min_points=120)

            if not candles:
                log(f"No candles for {sym}, skipping…")
                continue

        except Exception as e:
            # 🚨 HARD STOP on Polygon rate limit
            if "429" in str(e):
                log("Polygon rate limit hit — aborting SP500 scan early")
                break

            log(f"Candle fetch error for {sym}: {type(e).__name__}: {e}")
            continue

        # -------------------------------------------------
        # BullBrain feature generation + inference
        # -------------------------------------------------
        try:
            feats_vec, feat_dict, last_close = backend.compute_bullbrain_features(
                candles
            )

            if feats_vec is None:
                log(f"Feature generation failed for {sym}, skipping…")
                continue

            infer = backend.bullbrain_infer(feats_vec)
            if infer is None:
                log(f"bullbrain_infer returned None for {sym}, skipping…")
                continue

        except Exception as e:
            log(f"bullbrain_infer_single error for {sym}: {e}")
            continue

        # -------------------------------------------------
        # Signal interpretation
        # -------------------------------------------------
        prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
        prob_down = float(infer.get("probability_down") or (1.0 - prob_up))

        kind = classify_signal(prob_up, prob_down)
        confidence = max(prob_up, prob_down) * 100.0

        company_name = COMPANY_NAMES.get(sym, sym)

        base = {
            "symbol": sym,
            "company_name": company_name,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "confidence": round(confidence, 2),
        }

        all_symbols.append(
            {
                **base,
                "kind": kind,
                "prob_up_raw": prob_up,
                "prob_down_raw": prob_down,
            }
        )

        if kind in ("STRONG_BUY", "BUY"):
            short, risk = build_buy_explanations(
                sym, prob_up, prob_down, kind, feat_dict
            )
            buy_candidates.append(
                {
                    **base,
                    "signal": "BUY",
                    "kind": kind,
                    "explanation_short": short,
                    "explanation_risk": risk,
                }
            )
        else:
            short, risk = build_bear_explanations(
                sym, prob_up, prob_down, kind, feat_dict
            )
            signal_label = "SELL" if kind in ("STRONG_SELL", "SELL") else "HOLD"
            bear_candidates.append(
                {
                    **base,
                    "signal": signal_label,
                    "kind": kind,
                    "explanation_short": short,
                    "explanation_risk": risk,
                }
            )

        # -------------------------------------------------
        # 🕒 THROTTLE (VERY IMPORTANT)
        # -------------------------------------------------
        time.sleep(random.uniform(0.15, 0.25))

    # ----------------------------
    # Build Hotlist (Top 5 BUY)
    # ----------------------------
    buy_candidates.sort(key=lambda x: x["prob_up"], reverse=True)
    hotlist = buy_candidates[:5]

    if len(hotlist) < 5:
        already = {item["symbol"] for item in hotlist}
        extras_pool = [c for c in all_symbols if c["symbol"] not in already]
        extras_pool.sort(key=lambda x: x["prob_up_raw"], reverse=True)

        needed = 5 - len(hotlist)
        for extra in extras_pool[:needed]:
            sym = extra["symbol"]
            prob_up = extra["prob_up_raw"]
            prob_down = extra["prob_down_raw"]
            feat_dict = extra["feat_dict"]

            try:
                candles_extra = get_candles(sym, min_points=120)
                if not candles_extra:
                    continue

                _, feat_dict, _ = backend.compute_bullbrain_features(candles_extra)

                short, risk = build_buy_explanations(
                    sym, prob_up, prob_down, "WATCHLIST_BUY", feat_dict
                )

            except Exception:
                continue


            hotlist.append(
                {
                    "symbol": sym,
                    "company_name": extra["company_name"],
                    "prob_up": round(prob_up, 4),
                    "prob_down": round(prob_down, 4),
                    "confidence": extra["confidence"],
                    "signal": "BUY",
                    "kind": "WATCHLIST_BUY",
                    "explanation_short": short,
                    "explanation_risk": risk,
                }
            )

    hotlist = hotlist[:5]

    # ----------------------------
    # Build BearWatch (Top 5 SELL / HOLD)
    # ----------------------------
    bear_candidates.sort(key=lambda x: x["prob_down"], reverse=True)
    bearwatch = bear_candidates[:5]

    log(f"Built Hotlist: {len(hotlist)} | BearWatch: {len(bearwatch)}")

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return (
        {
            "count": len(hotlist),
            "hotlist": hotlist,
            "updated_at": now,
        },
        {
            "count": len(bearwatch),
            "bearwatch": bearwatch,
            "updated_at": now,
        },
    )


# ---------------------------------------------------------
# Save to Firestore
# ---------------------------------------------------------
def save_docs_to_firestore(hotlist_doc, bearwatch_doc):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("market_hotlist").set(hotlist_doc, merge=True)
    log("💾 Saved bullsignals_ai/market_hotlist")

    col.document("market_bearwatch").set(bearwatch_doc, merge=True)
    log("💾 Saved bullsignals_ai/market_bearwatch")



# =========================================================
# 🆕 MARKET OVERVIEW (FIRESTORE)
# =========================================================
def compute_market_overview():
    """
    Lightweight market overview.
    Uses existing backend helpers.
    """
    try:
        overview = backend._get_market_overview_quick()
        return {
            **overview,
            "updated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat().replace("+00:00", "Z"),
        }
    except Exception as e:
        log(f"Market overview error: {e}")
        return {
            "sp500_change": 0.0,
            "vix": 15.0,
            "fearGreed": {"value": 50, "label": "Neutral"},
            "risk_level": "Moderate Risk",
            "updated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat().replace("+00:00", "Z"),
        }



# =========================================================
# 🆕 MARKET HIGHLIGHTS + NEWS (FIRESTORE)
# =========================================================
def compute_market_pulse():
    """
    Builds:
      - Highlights (bullish / neutral / bearish) — 5 each
      - News grouped by time buckets
    """
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

    grouped = {
        "today": [],
        "yesterday": [],
        "week": [],
        "older": [],
    }

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



# ---------------------------------------------------------
# Safe quote fetcher (cron)
# ---------------------------------------------------------
def fetch_quote_safe(symbol: str) -> dict:
    """
    Uses backend_fetch_quote from main.py.
    Never throws inside cron.
    """
    try:
        q = backend.backend_fetch_quote(symbol)
        if isinstance(q, dict):
            return q
    except Exception as e:
        log(f"Quote fetch failed for {symbol}: {e}")

    return {}



# =========================================================
# HOME SCREEN SNAPSHOT (Firestore)
# =========================================================
def percent_str(x: Optional[float], digits: int = 2) -> str:
    try:
        if x is None:
            return "--"
        return f"{x:+.{digits}f}%"
    except Exception:
        return "--"


def build_us_market_card() -> Dict[str, Any]:
    # Use SPY + QQQ as "AI Market Insights" proxy
    spy = fetch_quote_safe("SPY")
    qqq = fetch_quote_safe("QQQ")

    spy_chg = spy.get("changePct")
    qqq_chg = qqq.get("changePct")

    # Some providers return decimals; normalize if needed
    def normalize_pct(v):
        try:
            v = float(v)
            # if it's like 0.0086 => 0.86%
            if abs(v) <= 1.5:
                return v * 100.0
            return v
        except Exception:
            return None

    spy_chg = normalize_pct(spy_chg)
    qqq_chg = normalize_pct(qqq_chg)

    return {
        "id": "us_market",
        "title": "AI Market Insights",
        "subtitle": "US Market snapshot",
        "items": [
            {"label": "S&P 500 (SPY)", "value": percent_str(spy_chg)},
            {"label": "Nasdaq (QQQ)", "value": percent_str(qqq_chg)},
        ],
        "updated_at": utc_now_iso(),
    }


def build_crypto_movers_card() -> Dict[str, Any]:
    # Free CoinGecko
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=dogecoin,ripple,solana"
        "&vs_currencies=usd"
        "&include_24hr_change=true"
    )
    data = requests.get(url, timeout=10).json()

    def cg_change(id_):
        try:
            return float(data[id_]["usd_24h_change"])
        except Exception:
            return None

    return {
        "id": "crypto",
        "title": "Crypto Movers",
        "subtitle": "24h change",
        "items": [
            {"label": "DOGE", "value": percent_str(cg_change("dogecoin"))},
            {"label": "XRP", "value": percent_str(cg_change("ripple"))},
            {"label": "SOL", "value": percent_str(cg_change("solana"))},
        ],
        "updated_at": utc_now_iso(),
    }


def build_sentiment_card() -> Dict[str, Any]:
    # Free alternative.me
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    data = requests.get(url, timeout=10).json()

    value = None
    label = "Neutral"
    try:
        row = (data.get("data") or [])[0]
        value = int(row.get("value"))
        label = row.get("value_classification") or "Neutral"
    except Exception:
        value = 50
        label = "Neutral"

    return {
        "id": "sentiment",
        "title": "Market Sentiment",
        "subtitle": "Fear & Greed (crypto proxy)",
        "items": [
            {"label": "Mood", "value": f"{label} ({value})"},
        ],
        "updated_at": utc_now_iso(),
    }


def build_commodities_card() -> Dict[str, Any]:
    # Use ETFs as proxies (free quote source already in your backend)
    gld = fetch_quote_safe("GLD")
    slv = fetch_quote_safe("SLV")
    uso = fetch_quote_safe("USO")

    def norm(v):
        try:
            v = float(v)
            if abs(v) <= 1.5:
                return v * 100.0
            return v
        except Exception:
            return None

    return {
        "id": "commodities",
        "title": "Commodities Snapshot",
        "subtitle": "ETF proxies",
        "items": [
            {"label": "Gold (GLD)", "value": percent_str(norm(gld.get("changePct")) )},
            {"label": "Oil (USO)", "value": percent_str(norm(uso.get("changePct")) )},
            {"label": "Silver (SLV)", "value": percent_str(norm(slv.get("changePct")) )},
        ],
        "updated_at": utc_now_iso(),
    }


def compute_homescreen_carousel() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []

    for builder in [build_us_market_card, build_crypto_movers_card, build_sentiment_card, build_commodities_card]:
        try:
            cards.append(builder())
        except Exception as e:
            log(f"Home carousel card failed ({builder.__name__}): {e}")

    # Ensure exactly 4 cards (if any failed, add a static fallback)
    while len(cards) < 4:
        cards.append(
            {
                "id": f"fallback_{len(cards)+1}",
                "title": "Trending Sectors",
                "subtitle": "Quick view",
                "items": [
                    {"label": "AI", "value": "Watch"},
                    {"label": "Tech", "value": "Watch"},
                    {"label": "Energy", "value": "Watch"},
                ],
                "updated_at": utc_now_iso(),
            }
        )

    return cards[:4]


def build_mag7_summary(signal: str) -> str:
    if signal == "BUY":
        return "BullBrain detects favorable upside conditions with improving momentum."
    if signal == "SELL":
        return "BullBrain flags downside risk as selling pressure increases."
    return "BullBrain sees mixed signals with no strong directional edge."


def build_mag7_fallback(symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "company_name": COMPANY_NAMES.get(symbol, symbol),
        "price": None,
        "changePct": None,
        "signal": "HOLD",
        "confidence": 50.0,
        "prob_up": 0.5,
        "prob_down": 0.5,
        "summary": "Data temporarily unavailable. Model defaults to neutral.",
        "updated_at": utc_now_iso(),
    }


def compute_single_mag7(symbol: str) -> Dict[str, Any]:
    ensure_bullbrain_loaded()

    candles = get_candles(symbol, min_points=120)

    if not candles:
        raise RuntimeError("No candles")

    feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles)
    if feats_vec is None:
        raise RuntimeError("Feature generation failed")

    infer = backend.bullbrain_infer(feats_vec)
    if infer is None:
        raise RuntimeError("Inference failed")

    prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
    prob_down = float(infer.get("probability_down") or (1.0 - prob_up))

    kind = classify_signal(prob_up, prob_down)

    if kind in ("BUY", "STRONG_BUY"):
        signal = "BUY"
    elif kind in ("SELL", "STRONG_SELL"):
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(max(prob_up, prob_down) * 100.0, 1)

    q = fetch_quote_safe(symbol)
    price = q.get("price") or q.get("close")
    change_pct = q.get("changePct")

    # normalize if needed (0.0086 => 0.86)
    try:
        if change_pct is not None:
            change_pct = float(change_pct)
            if abs(change_pct) <= 1.5:
                change_pct = change_pct * 100.0
    except Exception:
        pass

    return {
        "symbol": symbol,
        "company_name": COMPANY_NAMES.get(symbol, symbol),
        "price": price,
        "changePct": change_pct,
        "signal": signal,
        "confidence": confidence,
        "prob_up": round(prob_up, 4),
        "prob_down": round(prob_down, 4),
        "summary": build_mag7_summary(signal),
        "updated_at": utc_now_iso(),
    }


def compute_mag7_snapshot() -> List[Dict[str, Any]]:
    previous_map = get_previous_mag7_map()
    results: List[Dict[str, Any]] = []

    for sym in MAG7:
        try:
            current = compute_single_mag7(sym)

            prev_conf = (
                previous_map.get(sym, {})
                .get("confidence")
            )

            trend = compute_trend_arrow(
                current_confidence=current["confidence"],
                previous_confidence=prev_conf,
            )

            current["trend"] = trend
            results.append(current)

        except Exception as e:
            log(f"MAG7 fallback for {sym}: {e}")
            fallback = build_mag7_fallback(sym)
            fallback["trend"] = "FLAT"
            results.append(fallback)

    return results


def build_homescreen_snapshot() -> Dict[str, Any]:
    # reuse same market overview helper for consistency
    overview = compute_market_overview()

    return {
        "market_overview": overview,
        "carousel": compute_homescreen_carousel(),
        "mag7": compute_mag7_snapshot(),
        "updated_at": utc_now_iso(),
        "version": "v1",
    }


def save_homescreen_snapshot(snapshot: Dict[str, Any]) -> None:
    db = get_db()
    db.collection("bullsignals_ai").document("homescreen_snapshot").set(snapshot, merge=True)
    log("💾 Saved bullsignals_ai/homescreen_snapshot")


# =========================================================
# ENTRYPOINT
# =========================================================
def main():
    started = utc_now_iso()
    log(f"Market cron started at {started}")

    try:
        # 🔥 1) MAG7 FIRST — mandatory
        try:
            hs = build_homescreen_snapshot()
            save_homescreen_snapshot(hs)
        except Exception as e:
            log(f"MAG7/Home snapshot failed (non-fatal): {e}")

        # ⏳ Small pause (rate-limit safety)
        time.sleep(2)

        # 2) SP500 scan (Hotlist + BearWatch)
        hotlist_doc, bearwatch_doc = compute_hotlist_and_bearwatch()
        save_docs_to_firestore(hotlist_doc, bearwatch_doc)

        # 3) Market Pulse
        overview_doc = compute_market_overview()
        pulse_doc = compute_market_pulse()
        save_market_pulse_docs(overview_doc, pulse_doc)

        log(f"Market cron completed at {utc_now_iso()}")

    except Exception as e:
        log(f"Fatal error in market_cron: {e}")

if __name__ == "__main__":
    main()