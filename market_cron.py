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


# ---------------------------------------------------------
# Logging helper
# ---------------------------------------------------------
def log(msg: str) -> None:
    backend.log(f"[cron] {msg}")


# ---------------------------------------------------------
# Firestore handle
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


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
        if i % 50 == 0:
            log(f"...processed {i}/{total}")

        try:
            candles = backend.fetch_daily_candles(sym)
            if not candles:
                log(f"No candles for {sym}, skipping…")
                continue

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
                "feat_dict": feat_dict,
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

    # ----------------------------
    # Build Hotlist (Top 5 BUY)
    # ----------------------------
    buy_candidates.sort(key=lambda x: x["prob_up"], reverse=True)
    hotlist = buy_candidates[:5]

    # Fallback: if still <5, fill with top bullish names (WATCHLIST_BUY),
    # even if the model is slightly bearish overall.
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

            short, risk = build_buy_explanations(
                sym, prob_up, prob_down, "WATCHLIST_BUY", feat_dict
            )

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

    # Cap at 5
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

    hotlist_doc = {
        "count": len(hotlist),
        "hotlist": hotlist,
        "updated_at": now,
    }

    bearwatch_doc = {
        "count": len(bearwatch),
        "bearwatch": bearwatch,
        "updated_at": now,
    }

    return hotlist_doc, bearwatch_doc


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
    eastern = backend.pytz.timezone("America/New_York")
    utc = backend.pytz.utc

    news_resp = backend.market_news()
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
        except Exception:
            continue

    cleaned.sort(key=lambda x: x["pubDateObj"], reverse=True)

    titles = [n.get("title", "") for n in cleaned[:80] if n.get("title")]
    analyzed = backend._analyze_headline_sentiment_py(titles)

    bullish = [a["title"] for a in analyzed if a["tag"] == "📈"]
    bearish = [a["title"] for a in analyzed if a["tag"] == "📉"]
    neutral = [a["title"] for a in analyzed if a["tag"] == "⚖️"]

    bullish = backend._ensure_five(bullish, "bullish")
    neutral = backend._ensure_five(neutral, "neutral")
    bearish = backend._ensure_five(bearish, "bearish")

    now_et = datetime.datetime.now(eastern)
    today = now_et.date()
    yesterday = today - datetime.timedelta(days=1)
    week_ago = today - datetime.timedelta(days=7)

    grouped = {"today": [], "yesterday": [], "week": [], "older": []}

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
    main()