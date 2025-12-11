# market_cron.py
# ---------------------------------------------------------
# BullSignalsAI — 15-minute BullBrain scan for Hotlist & BearWatch
#
# This script is meant to be called by Render Cron:
#   Command:  python market_cron.py
#   Schedule: */15 * * * 1-5   (weekdays, every 15 mins)
# ---------------------------------------------------------

import datetime
import time

import firebase_admin
from firebase_admin import firestore

# IMPORTANT: reuse backend logic from main.py
import main as backend
from symbols_clean import REAL_TICKERS  # your full SP500 (or 492) ticker list


# ---------------------------------------------------------
# Helpers to reuse backend's logging + model + firestore
# ---------------------------------------------------------
def log(msg: str) -> None:
    backend.log(f"[cron] {msg}")


def ensure_bullbrain_loaded():
    """
    Make sure BullBrain model is loaded for this cron process.
    Reuses backend.load_bullbrain_model and backend.bullbrain_model.
    """
    if backend.bullbrain_model is not None:
        return

    log("Loading BullBrain model for cron process…")
    try:
        backend.bullbrain_model = backend.load_bullbrain_model()
        log("BullBrain model loaded successfully in cron")
    except Exception as e:
        log(f"Failed to load BullBrain model in cron: {e}")
        raise


def get_db():
    """
    Reuse backend's Firebase Admin initialization and Firestore client.
    """
    # Ensure Firebase is initialized (backend.init_firebase_admin already
    # knows how to read FIREBASE_ADMIN_JSON from environment)
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


# ---------------------------------------------------------
# BullBrain single-symbol helper (reuses backend logic)
# ---------------------------------------------------------
def bullbrain_infer_single(symbol: str):
    """
    Use the same candles + feature pipeline and BullBrain inference
    used in your main FastAPI backend.
    """
    try:
        candles = backend.fetch_daily_candles(symbol)
        if not candles:
            return None

        features_vec, feature_dict, last_close = backend.compute_bullbrain_features(
            candles
        )
        infer = backend.bullbrain_infer(features_vec)
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

      - STRONG_BUY: prob_up   >= 0.60 AND prob_up   > prob_down + 0.08
      - BUY:        prob_up   >= 0.55 AND prob_up   > prob_down + 0.05

      - STRONG_SELL: prob_down >= 0.60 AND prob_down > prob_up + 0.08
      - SELL:        prob_down >= 0.55 AND prob_down > prob_up + 0.05

      - Else: HOLD
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

    # Otherwise sideways / uncertain
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
    # Make sure model is loaded once per cron run
    ensure_bullbrain_loaded()

    buy_candidates = []
    bear_candidates = []

    total = len(REAL_TICKERS)
    log(f"Scanning {total} tickers with BullBrain...")

    for i, sym in enumerate(REAL_TICKERS, start=1):
        sym = sym.upper()
        if i % 50 == 0:
            log(f"...processed {i}/{total}")

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
            # ✅ BUYS go only into Hotlist
            short, risk = build_buy_explanations(sym, prob_up, prob_down, kind)
            item = {
                **base_item,
                "signal": "BUY",         # label is always BUY in Hotlist
                "kind": kind,            # STRONG_BUY or BUY
                "explanation_short": short,
                "explanation_risk": risk,
            }
            buy_candidates.append(item)
        else:
            # ✅ SELL / STRONG_SELL / HOLD go into BearWatch bucket
            short, risk = build_bear_explanations(sym, prob_up, prob_down, kind)
            signal_label = "SELL" if kind in ("STRONG_SELL", "SELL") else "HOLD"
            item = {
                **base_item,
                "signal": signal_label,  # SELL or HOLD
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

    # Fallback: if fewer than 5 BUYS, add mildly bullish HOLDs as WATCHLIST_BUY
    if len(hotlist) < 5:
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
                "signal": "BUY",              # still labeled BUY for UI
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

    log(f"Built Hotlist: {len(hotlist)} tickers")
    log(f"Built BearWatch: {len(bearwatch)} tickers")

    # Use timezone-aware UTC now
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

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
    log("Saved bullsignals_ai/market_hotlist")

    col.document("market_bearwatch").set(bearwatch_doc, merge=True)
    log("Saved bullsignals_ai/market_bearwatch")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    log(f"BullBrain market scan started at {started}")

    try:
        hotlist_doc, bearwatch_doc = compute_hotlist_and_bearwatch()
        save_docs_to_firestore(hotlist_doc, bearwatch_doc)
        finished = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        log(f"Completed BullBrain market scan at {finished}")
    except Exception as e:
        log(f"Fatal error in market_cron: {e}")


if __name__ == "__main__":
    main()
