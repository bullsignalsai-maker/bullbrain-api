# market_cron.py
# ---------------------------------------------------------
# BullSignalsAI — 15-minute BullBrain scan for Hotlist & BearWatch
#
# This script is meant to be called by Render Cron:
#   Command:  python market_cron.py
#   Schedule: */15 * * * *
# ---------------------------------------------------------

import os
import json
import datetime
import time

import firebase_admin
from firebase_admin import credentials, firestore

from symbols_clean import REAL_TICKERS
from main import fetch_daily_candles, compute_bullbrain_features, bullbrain_infer


# ---------------------------------------------------------
# Firebase Admin init (standalone for cron process)
# ---------------------------------------------------------
def init_firebase_admin():
    if firebase_admin._apps:
        return firebase_admin._apps[0]

    firebase_json = os.getenv("FIREBASE_ADMIN_JSON")
    if not firebase_json:
        raise RuntimeError("FIREBASE_ADMIN_JSON is missing in environment")

    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    app = firebase_admin.initialize_app(cred)
    print("🔥 [cron] Firebase Admin initialized")
    return app


def get_db():
    if not firebase_admin._apps:
        init_firebase_admin()
    return firestore.client()


# ---------------------------------------------------------
# BullBrain single-symbol helper (reuses your main logic)
# ---------------------------------------------------------
def bullbrain_infer_single(symbol: str):
    try:
        candles = fetch_daily_candles(symbol)
        if not candles:
            return None

        features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
        infer = bullbrain_infer(features_vec)
        return infer
    except Exception as e:
        print("[cron] bullbrain_infer_single error:", symbol, e)
        return None


# ---------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------
def classify_signal(prob_up: float, prob_down: float):
    """
    Classify into STRONG_BUY / BUY / STRONG_SELL / SELL / HOLD.

    Option B + weak buy/sell rule:
      - BUY if prob_up >= 0.55 and prob_up > prob_down + 0.05
      - STRONG_BUY if prob_up >= 0.60 and prob_up > prob_down + 0.08
      - SELL / STRONG_SELL with symmetric rules for prob_down
      - else HOLD
    """
    edge_up = prob_up - prob_down
    edge_down = prob_down - prob_up

    # Strong & normal BUY
    if prob_up >= 0.60 and edge_up >= 0.08:
        return "STRONG_BUY"
    if prob_up >= 0.55 and edge_up >= 0.05:
        return "BUY"

    # Strong & normal SELL
    if prob_down >= 0.60 and edge_down >= 0.08:
        return "STRONG_SELL"
    if prob_down >= 0.55 and edge_down >= 0.05:
        return "SELL"

    return "HOLD"


def build_buy_explanations(symbol, prob_up, prob_down, kind: str):
    up_pct = prob_up * 100.0
    down_pct = prob_down * 100.0

    if kind == "STRONG_BUY":
        short = (
            f"{symbol}: BullBrain sees a strong bullish edge — "
            f"~{up_pct:.1f}% chance up vs {down_pct:.1f}% down in the short term."
        )
        risk = (
            "Signal: STRONG BUY. The model favors upside, but always use position sizing, "
            "stops, and your own risk tolerance."
        )
    elif kind == "BUY":
        short = (
            f"{symbol}: BullBrain leans bullish with ~{up_pct:.1f}% up vs "
            f"{down_pct:.1f}% down."
        )
        risk = (
            "Signal: BUY. Edge is noticeable but not extreme — good for scaling in, "
            "not all-in moves."
        )
    else:  # WATCHLIST_BUY
        short = (
            f"{symbol}: Mild bullish tilt (~{up_pct:.1f}% up vs {down_pct:.1f}% down)."
        )
        risk = (
            "Signal: WATCHLIST BUY. Treat this as higher-risk and confirm with price "
            "action or your own indicators before acting."
        )

    return short, risk


def build_bear_explanations(symbol, prob_up, prob_down, kind: str):
    up_pct = prob_up * 100.0
    down_pct = prob_down * 100.0

    if kind == "STRONG_SELL":
        short = (
            f"{symbol}: Bearish pressure dominates — ~{down_pct:.1f}% down vs "
            f"{up_pct:.1f}% up."
        )
        risk = (
            "Signal: STRONG SELL. The model expects weakness; consider trimming, "
            "tightening stops, or avoiding new entries."
        )
    elif kind == "SELL":
        short = (
            f"{symbol}: Downside is more likely (~{down_pct:.1f}% down vs "
            f"{up_pct:.1f}% up)."
        )
        risk = (
            "Signal: SELL. This looks like a weaker tape; better for risk reduction "
            "than fresh buys."
        )
    else:  # HOLD
        short = (
            f"{symbol}: No clear edge — model sees ~{up_pct:.1f}% up vs "
            f"{down_pct:.1f}% down."
        )
        risk = (
            "Signal: HOLD. Price action looks more sideways/uncertain; waiting for a "
            "stronger signal may be safer."
        )

    return short, risk


# ---------------------------------------------------------
# Main scan: build top 5 Hotlist + top 5 BearWatch
# ---------------------------------------------------------
def compute_hotlist_and_bearwatch():
    buy_candidates = []
    bear_candidates = []

    total = len(REAL_TICKERS)
    print(f"🔍 [cron] Scanning {total} tickers with BullBrain...")

    for i, sym in enumerate(REAL_TICKERS, start=1):
        sym = sym.upper()
        if i % 50 == 0:
            print(f"  ...processed {i}/{total}")

        infer = bullbrain_infer_single(sym)
        if not infer:
            continue

        # Pull probabilities safely
        prob_up = float(
            infer.get("probability_up")
            or infer.get("raw_output")
            or 0.5
        )
        # If backend didn't store probability_down, compute mirror
        prob_down = float(
            infer.get("probability_down")
            or (1.0 - prob_up)
        )

        kind = classify_signal(prob_up, prob_down)

        # Confidence as the dominant side in %
        confidence = max(prob_up, prob_down) * 100.0

        base_item = {
            "symbol": sym,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "confidence": round(confidence, 2),
        }

        if kind in ("STRONG_BUY", "BUY"):
            short, risk = build_buy_explanations(sym, prob_up, prob_down, kind)
            item = {
                **base_item,
                "signal": "BUY",
                "kind": kind,
                "explanation_short": short,
                "explanation_risk": risk,
            }
            buy_candidates.append(item)
        else:
            # SELL or HOLD bucket for BearWatch
            short, risk = build_bear_explanations(sym, prob_up, prob_down, kind)
            signal_label = "SELL" if kind in ("STRONG_SELL", "SELL") else "HOLD"
            item = {
                **base_item,
                "signal": signal_label,
                "kind": kind,
                "explanation_short": short,
                "explanation_risk": risk,
            }
            bear_candidates.append(item)

    # ----------------------------
    # Build final Hotlist (Top 5)
    # ----------------------------
    # Primary: strong/normal buys sorted by prob_up desc
    buy_candidates.sort(key=lambda x: x["prob_up"], reverse=True)
    hotlist = buy_candidates[:5]

    # Fallback: if fewer than 5 BUYS, top "mild edge" HOLDs (>0.52 up) become WATCHLIST_BUY
    if len(hotlist) < 5:
        # Find HOLDs that are slightly bullish
        mild_candidates = [
            b for b in bear_candidates
            if b["kind"] == "HOLD" and b["prob_up"] > 0.52
        ]
        mild_candidates.sort(key=lambda x: x["prob_up"], reverse=True)

        needed = 5 - len(hotlist)
        for extra in mild_candidates[:needed]:
            sym = extra["symbol"]
            prob_up = extra["prob_up"]
            prob_down = extra["prob_down"]
            short, risk = build_buy_explanations(sym, prob_up, prob_down, "WATCHLIST_BUY")
            item = {
                "symbol": sym,
                "prob_up": prob_up,
                "prob_down": prob_down,
                "confidence": extra["confidence"],
                "signal": "BUY",
                "kind": "WATCHLIST_BUY",
                "explanation_short": short,
                "explanation_risk": risk,
            }
            hotlist.append(item)

    # Ensure max 5
    hotlist = hotlist[:5]

    # ----------------------------
    # Build final BearWatch (Top 5)
    # ----------------------------
    # Sort by highest downside probability
    bear_candidates.sort(key=lambda x: x["prob_down"], reverse=True)
    bearwatch = bear_candidates[:5]

    print(f"✅ [cron] Built Hotlist: {len(hotlist)} tickers")
    print(f"✅ [cron] Built BearWatch: {len(bearwatch)} tickers")

    now_iso = datetime.datetime.utcnow().isoformat() + "Z"

    hotlist_doc = {
        "count": len(hotlist),
        "hotlist": hotlist,
        "updated_at": now_iso,
    }

    bearwatch_doc = {
        "count": len(bearwatch),
        "bearwatch": bearwatch,
        "updated_at": now_iso,
    }

    return hotlist_doc, bearwatch_doc


# ---------------------------------------------------------
# Save to Firestore
# ---------------------------------------------------------
def save_docs_to_firestore(hotlist_doc, bearwatch_doc):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("market_hotlist").set(hotlist_doc, merge=True)
    print("💾 [cron] Saved bullsignals_ai/market_hotlist")

    col.document("market_bearwatch").set(bearwatch_doc, merge=True)
    print("💾 [cron] Saved bullsignals_ai/market_bearwatch")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
def main():
    started = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"\n⏱️ [cron] BullBrain market scan started at {started}")

    try:
        hotlist_doc, bearwatch_doc = compute_hotlist_and_bearwatch()
        save_docs_to_firestore(hotlist_doc, bearwatch_doc)
        finished = datetime.datetime.utcnow().isoformat() + "Z"
        print(f"🎉 [cron] Completed BullBrain market scan at {finished}")
    except Exception as e:
        print("❌ [cron] Fatal error in market_cron:", e)


if __name__ == "__main__":
    main()
