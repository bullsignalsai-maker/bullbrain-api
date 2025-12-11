# market_cron.py
# ---------------------------------------------------------
# BullSignalsAI — 15-minute BullBrain scan for Hotlist & BearWatch
# Render Cron Command:
#     python market_cron.py
# Schedule (weekdays every 15 min):
#     */15 * * * 1-5
# ---------------------------------------------------------

import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import main as backend
from symbols_clean_test import REAL_TICKERS


# ---------------------------------------------------------
# Logging wrapper
# ---------------------------------------------------------
def log(msg: str):
    backend.log(f"[cron] {msg}")


# ---------------------------------------------------------
# Firebase Admin
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


# ---------------------------------------------------------
# Ensure BullBrain Model is Loaded
# ---------------------------------------------------------
def ensure_bullbrain_loaded():
    if backend.bullbrain_model is not None:
        return

    log("Loading BullBrain model for cron…")
    try:
        backend.bullbrain_model = backend.load_bullbrain_model()
        if backend.bullbrain_model is None:
            raise RuntimeError("Model returned None")

        log("BullBrain model loaded successfully.")

    except Exception as e:
        log(f"❌ Failed loading model: {e}")
        raise


# ---------------------------------------------------------
# Single-symbol inference
# ---------------------------------------------------------
def bullbrain_infer_single(symbol: str):
    try:
        candles = backend.fetch_daily_candles(symbol)
        if not candles:
            return None

        feats, feat_dict, last_close = backend.compute_bullbrain_features(candles)

        # IMPORTANT: ensure model exists
        if backend.bullbrain_model is None:
            raise RuntimeError("BullBrain model not loaded")

        return backend.bullbrain_infer(feats)

    except Exception as e:
        log(f"bullbrain_infer_single error for {symbol}: {e}")
        return None


# ---------------------------------------------------------
# Signal classification
# ---------------------------------------------------------
def classify_signal(prob_up, prob_down):
    edge_up = prob_up - prob_down
    edge_down = prob_down - prob_up

    # BUY family
    if prob_up >= 0.60 and edge_up >= 0.08:
        return "STRONG_BUY"
    if prob_up >= 0.55 and edge_up >= 0.05:
        return "BUY"

    # SELL family
    if prob_down >= 0.60 and edge_down >= 0.08:
        return "STRONG_SELL"
    if prob_down >= 0.55 and edge_down >= 0.05:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------
# Explanations (BUY)
# ---------------------------------------------------------
def build_buy_explanations(symbol, prob_up, prob_down, kind):
    up = prob_up * 100
    down = prob_down * 100

    if kind == "STRONG_BUY":
        short = f"{symbol}: Strong bullish momentum — {up:.1f}% up vs {down:.1f}% down."
        risk = "Signal: STRONG BUY. Trend is solid, but still manage risk and avoid oversized entries."
    elif kind == "BUY":
        short = f"{symbol}: Bullish tilt — {up:.1f}% up vs {down:.1f}% down."
        risk = "Signal: BUY. Decent edge; suitable for scaling in gradually."
    else:  # WATCHLIST BUY
        short = f"{symbol}: Mild bullish bias — {up:.1f}% up vs {down:.1f}% down."
        risk = "Signal: WATCHLIST BUY. Higher-risk; wait for confirmation or tighter price action."

    return short, risk


# ---------------------------------------------------------
# Explanations (SELL / HOLD)
# ---------------------------------------------------------
def build_bear_explanations(symbol, prob_up, prob_down, kind):
    up = prob_up * 100
    down = prob_down * 100

    if kind == "STRONG_SELL":
        short = f"{symbol}: Strong downside pressure — {down:.1f}% down vs {up:.1f}% up."
        risk = "Signal: STRONG SELL. Weak structure; consider trimming or avoiding new positions."
    elif kind == "SELL":
        short = f"{symbol}: Bearish bias — {down:.1f}% down vs {up:.1f}% up."
        risk = "Signal: SELL. Consider reducing exposure or tightening stops."
    else:
        short = f"{symbol}: Sideways bias — {up:.1f}% up vs {down:.1f}% down."
        risk = "Signal: HOLD. No clear trend; wait for stronger signals before acting."

    return short, risk


# ---------------------------------------------------------
# Market Scan
# ---------------------------------------------------------
def compute_hotlist_and_bearwatch():
    ensure_bullbrain_loaded()

    buys = []
    bears = []

    total = len(REAL_TICKERS)
    log(f"Scanning {total} tickers...")

    for i, sym in enumerate(REAL_TICKERS, start=1):
        if i % 50 == 0:
            log(f"...processed {i}/{total}")

        infer = bullbrain_infer_single(sym)
        if not infer:
            continue

        prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
        prob_down = float(infer.get("probability_down") or (1 - prob_up))

        kind = classify_signal(prob_up, prob_down)
        confidence = max(prob_up, prob_down) * 100

        base = {
            "symbol": sym,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "confidence": round(confidence, 2),
        }

        if kind in ("STRONG_BUY", "BUY"):
            short, risk = build_buy_explanations(sym, prob_up, prob_down, kind)
            buys.append({**base, "signal": "BUY", "kind": kind,
                         "explanation_short": short, "explanation_risk": risk})
        else:
            short, risk = build_bear_explanations(sym, prob_up, prob_down, kind)
            signal_label = "SELL" if kind in ("STRONG_SELL", "SELL") else "HOLD"
            bears.append({**base, "signal": signal_label, "kind": kind,
                          "explanation_short": short, "explanation_risk": risk})

    # ------- Hotlist (Top 5 BUYs) -------
    buys.sort(key=lambda x: x["prob_up"], reverse=True)
    hotlist = buys[:5]

    # Fallback (WATCHLIST_BUY)
    if len(hotlist) < 5:
        mild = [b for b in bears if b["kind"] == "HOLD" and b["prob_up"] > 0.52]
        mild.sort(key=lambda x: x["prob_up"], reverse=True)

        needed = 5 - len(hotlist)
        for m in mild[:needed]:
            short, risk = build_buy_explanations(m["symbol"], m["prob_up"], m["prob_down"], "WATCHLIST_BUY")
            hotlist.append({
                **m,
                "signal": "BUY",
                "kind": "WATCHLIST_BUY",
                "explanation_short": short,
                "explanation_risk": risk,
            })

    hotlist = hotlist[:5]

    # ------- BearWatch (Top 5 SELL/HOLD) -------
    bears.sort(key=lambda x: x["prob_down"], reverse=True)
    bearwatch = bears[:5]

    log(f"Built Hotlist {len(hotlist)} | BearWatch {len(bearwatch)}")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    return (
        {"count": len(hotlist), "hotlist": hotlist, "updated_at": now},
        {"count": len(bearwatch), "bearwatch": bearwatch, "updated_at": now},
    )


# ---------------------------------------------------------
# Save results to Firestore
# ---------------------------------------------------------
def save_docs(hotlist, bearwatch):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("market_hotlist").set(hotlist, merge=True)
    log("Saved market_hotlist")

    col.document("market_bearwatch").set(bearwatch, merge=True)
    log("Saved market_bearwatch")


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def main():
    start = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    log(f"Cron started at {start}")

    try:
        hot, bear = compute_hotlist_and_bearwatch()
        save_docs(hot, bear)

        end = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        log(f"Cron completed at {end}")

    except Exception as e:
        log(f"❌ Fatal cron error: {e}")


if __name__ == "__main__":
    main()
