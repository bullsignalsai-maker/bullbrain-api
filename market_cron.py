# market_cron.py
# ---------------------------------------------------------
# BullSignalsAI — 15-minute BullBrain scan + HomeScreen build
#
# Render Cron:
#   Command : python market_cron.py
#   Schedule: */15 * * * 1-5
# ---------------------------------------------------------

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


# =========================================================
# CONSTANTS
# =========================================================

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]

MARKET_KEYWORDS = [
    "stock", "stocks", "market", "markets", "futures",
    "s&p", "dow", "nasdaq", "indexes",
    "fed", "rates", "yields", "inflation", "cpi", "jobs",
    "earnings", "guidance", "sectors",
]

EXCLUDE_KEYWORDS = [
    "death", "killed", "crime", "celebrity",
    "weather", "earthquake",
]

ALLOWED_SOURCES = {
    "CNBC", "MarketWatch", "Bloomberg",
    "Reuters", "WSJ", "Investing.com", "Yahoo",
}


# =========================================================
# HELPERS
# =========================================================

def log(msg: str) -> None:
    backend.log(f"[cron] {msg}")


def get_db():
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


def ensure_bullbrain_loaded():
    if backend.bullbrain_model is not None:
        return
    log("Loading BullBrain model…")
    backend.bullbrain_model = backend.load_bullbrain_model()
    log("BullBrain model loaded")


def safe_feat(feat_dict, key):
    try:
        v = float(feat_dict.get(key, float("nan")))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# =========================================================
# MAG7 SNAPSHOT (NEW)
# =========================================================

def compute_mag7_snapshot():
    ensure_bullbrain_loaded()

    items = []

    for sym in MAG7:
        try:
            candles = backend.fetch_daily_candles(sym)
            if not candles:
                log(f"[MAG7] No candles for {sym}")
                continue

            feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles)
            if feats_vec is None:
                continue

            infer = backend.bullbrain_infer(feats_vec)

            prob_up = float(infer.get("probability_up", 0.5))
            prob_down = float(infer.get("probability_down", 0.5))
            confidence = max(prob_up, prob_down) * 100.0

            items.append({
                "symbol": sym,
                "company_name": COMPANY_NAMES.get(sym, sym),
                "signal": infer.get("signal", "HOLD"),
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "confidence": round(confidence, 2),
                "version": infer.get("version"),
            })

        except Exception as e:
            log(f"[MAG7] Error for {sym}: {e}")

    return {
        "count": len(items),
        "items": items,
    }


# =========================================================
# MACRO / CAROUSEL
# =========================================================

def build_macro_snapshot():
    from backend.homescreen_macro_logic import build_homescreen_macro_snapshot
    return build_homescreen_macro_snapshot()


# =========================================================
# HOME SCREEN DOC (FINAL)
# =========================================================

def build_homescreen_snapshot():
    mag7 = compute_mag7_snapshot()
    macro = build_macro_snapshot()

    now = datetime.datetime.now(datetime.timezone.utc)\
        .isoformat().replace("+00:00", "Z")

    return {
        "schema_version": "homescreen_v1",
        "updated_at": now,
        "market": macro.get("live_market"),
        "macro": {
            "carousel": macro.get("carousel", [])
        },
        "mag7": mag7,
        "meta": {
            "computed_by": "market_cron",
            "refresh_minutes": 15,
        },
    }


def save_homescreen_snapshot(doc):
    db = get_db()
    db.collection("bullsignals_ai")\
      .document("homescreen_snapshot")\
      .set(doc, merge=True)

    log("💾 Saved bullsignals_ai/homescreen_snapshot")


# =========================================================
# EXISTING: HOTLIST + BEARWATCH (UNCHANGED)
# =========================================================

def classify_signal(prob_up: float, prob_down: float) -> str:
    edge = prob_up - prob_down
    if prob_up >= 0.58 and edge >= 0.08:
        return "STRONG_BUY"
    if prob_up >= 0.52 and edge >= 0.02:
        return "BUY"
    if prob_down >= 0.58 and -edge >= 0.08:
        return "STRONG_SELL"
    if prob_down >= 0.52 and -edge >= 0.02:
        return "SELL"
    return "HOLD"


def compute_hotlist_and_bearwatch():
    ensure_bullbrain_loaded()

    buy, bear = [], []

    for sym in REAL_TICKERS:
        try:
            candles = backend.fetch_daily_candles(sym)
            if not candles:
                continue

            feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles)
            if feats_vec is None:
                continue

            infer = backend.bullbrain_infer(feats_vec)
            prob_up = float(infer.get("probability_up", 0.5))
            prob_down = 1.0 - prob_up

            kind = classify_signal(prob_up, prob_down)
            confidence = max(prob_up, prob_down) * 100.0

            base = {
                "symbol": sym,
                "company_name": COMPANY_NAMES.get(sym, sym),
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "confidence": round(confidence, 2),
                "kind": kind,
            }

            if kind in ("BUY", "STRONG_BUY"):
                buy.append({**base, "signal": "BUY"})
            else:
                bear.append({**base, "signal": "SELL" if kind != "HOLD" else "HOLD"})

        except Exception:
            continue

    buy.sort(key=lambda x: x["prob_up"], reverse=True)
    bear.sort(key=lambda x: x["prob_down"], reverse=True)

    now = datetime.datetime.now(datetime.timezone.utc)\
        .isoformat().replace("+00:00", "Z")

    return (
        {"count": len(buy[:5]), "hotlist": buy[:5], "updated_at": now},
        {"count": len(bear[:5]), "bearwatch": bear[:5], "updated_at": now},
    )


def save_docs_to_firestore(hotlist_doc, bearwatch_doc):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("market_hotlist").set(hotlist_doc, merge=True)
    col.document("market_bearwatch").set(bearwatch_doc, merge=True)

    log("💾 Saved market_hotlist & market_bearwatch")


# =========================================================
# ENTRYPOINT
# =========================================================

def main():
    started = datetime.datetime.now(datetime.timezone.utc)\
        .isoformat().replace("+00:00", "Z")

    log(f"Market cron started at {started}")

    try:
        hotlist_doc, bearwatch_doc = compute_hotlist_and_bearwatch()
        save_docs_to_firestore(hotlist_doc, bearwatch_doc)

        homescreen_doc = build_homescreen_snapshot()
        save_homescreen_snapshot(homescreen_doc)

        finished = datetime.datetime.now(datetime.timezone.utc)\
            .isoformat().replace("+00:00", "Z")

        log(f"Market cron completed at {finished}")

    except Exception as e:
        log(f"❌ Fatal error in market_cron: {e}")


if __name__ == "__main__":
    main()
