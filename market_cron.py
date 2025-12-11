# market_cron.py
# ---------------------------------------------------------
# BullSignalsAI — 15-minute BullBrain Hotlist & BearWatch Scan
#
# Render Cron:
#   Command : python market_cron.py
#   Schedule: */15 * * * 1-5   (Weekdays only)
#
# This script:
#   ✓ Loads BullBrain model (once)
#   ✓ Scans ALL S&P500 tickers from symbols_clean.py
#   ✓ Builds:
#        HOTLIST (Top 5 BUY / WATCHLIST BUY)
#        BEARWATCH (Top 5 SELL / STRONG SELL / HOLD)
#   ✓ Adds company_name
#   ✓ Generates explanations with technical hints
#   ✓ Removes ticker name from explanation_short
#   ✓ Saves results into Firestore:
#         bullsignals_ai/market_hotlist
#         bullsignals_ai/market_bearwatch
# ---------------------------------------------------------

import datetime
import math

import firebase_admin
from firebase_admin import firestore

import main as backend
from symbols_clean import REAL_TICKERS, COMPANY_NAMES


def log(msg: str):
    backend.log(f"[cron] {msg}")


# ---------------------------------------------------------
# Firestore connection
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        backend.init_firebase_admin()
    return backend.db


# ---------------------------------------------------------
# Ensure BullBrain model is loaded once
# ---------------------------------------------------------
def ensure_bullbrain_loaded():
    if backend.bullbrain_model is not None:
        return

    log("Loading BullBrain model for cron…")
    backend.bullbrain_model = backend.load_bullbrain_model()
    log("Model loaded successfully.")


# ---------------------------------------------------------
# Safe getter for indicators
# ---------------------------------------------------------
def safe_feat(feat_dict, key):
    try:
        v = float(feat_dict.get(key, float("nan")))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except:
        return None


# ---------------------------------------------------------
# Signal classification (Balanced Option C)
# ---------------------------------------------------------
def classify_signal(prob_up, prob_down):
    edge_up = prob_up - prob_down
    edge_down = prob_down - prob_up

    # BUY conditions
    if prob_up >= 0.58 and edge_up >= 0.08:
        return "STRONG_BUY"
    if prob_up >= 0.52 and edge_up >= 0.02:
        return "BUY"

    # SELL conditions
    if prob_down >= 0.58 and edge_down >= 0.08:
        return "STRONG_SELL"
    if prob_down >= 0.52 and edge_down >= 0.02:
        return "SELL"

    return "HOLD"


# ---------------------------------------------------------
# BUY explanations (ticker removed from short sentence)
# ---------------------------------------------------------
def build_buy_explanations(prob_up, prob_down, kind, feat_dict):
    up = prob_up * 100
    down = prob_down * 100

    # Remove ticker from short explanation
    label = "strong BUY" if kind == "STRONG_BUY" else (
        "BUY" if kind == "BUY" else "watchlist BUY"
    )

    short = (
        f"BullBrain flags a {label} setup — about "
        f"{up:.1f}% upside probability vs {down:.1f}% downside."
    )

    # Technical hints
    rsi = safe_feat(feat_dict, "rsi14")
    ret10 = safe_feat(feat_dict, "return_10d")
    price_vs_20 = safe_feat(feat_dict, "price_vs_sma20_pct")
    vol_vs_ma20 = safe_feat(feat_dict, "volume_vs_ma20_pct")
    trend = safe_feat(feat_dict, "trend_strength_20")

    hints = []

    if ret10 is not None:
        hints.append(f"~{ret10:+.1f}% move over last 10 sessions")

    if price_vs_20 is not None:
        side = "above" if price_vs_20 >= 0 else "below"
        hints.append(f"price is {abs(price_vs_20):.1f}% {side} its 20-day average")

    if rsi is not None:
        if rsi >= 60:
            hints.append(f"RSI ≈ {rsi:.0f} (strong momentum)")
        elif rsi >= 50:
            hints.append(f"RSI ≈ {rsi:.0f} (slightly bullish)")
        else:
            hints.append(f"RSI ≈ {rsi:.0f} (not overbought)")

    if vol_vs_ma20 is not None and abs(vol_vs_ma20) >= 15:
        side = "higher" if vol_vs_ma20 > 0 else "lighter"
        hints.append(f"volume {abs(vol_vs_ma20):.0f}% {side} than 20-day normal")

    if trend is not None:
        if trend >= 0.6:
            hints.append("trend strength looks solid")
        elif trend <= -0.6:
            hints.append("trend still shaky — be cautious")

    tech_line = " | ".join(hints[:3]) + ". " if hints else ""

    risk_line = (
        "This is not financial advice — consider position sizing, a clear stop-loss, "
        "and your own research before acting."
    )

    return short, tech_line + risk_line


# ---------------------------------------------------------
# SELL / HOLD explanations (ticker removed from short sentence)
# ---------------------------------------------------------
def build_bear_explanations(prob_up, prob_down, kind, feat_dict):
    up = prob_up * 100
    down = prob_down * 100

    if kind == "STRONG_SELL":
        short = f"Bearish pressure dominates — about {down:.1f}% downside vs {up:.1f}% upside."
    elif kind == "SELL":
        short = f"Bearish bias — about {down:.1f}% downside vs {up:.1f}% upside."
    else:
        short = f"No clear edge — model sees {up:.1f}% up vs {down:.1f}% down."

    # Technicals
    rsi = safe_feat(feat_dict, "rsi14")
    ret10 = safe_feat(feat_dict, "return_10d")
    price_vs_20 = safe_feat(feat_dict, "price_vs_sma20_pct")
    vol_vs_ma20 = safe_feat(feat_dict, "volume_vs_ma20_pct")
    trend = safe_feat(feat_dict, "trend_strength_20")

    hints = []

    if ret10 is not None:
        hints.append(f"~{ret10:+.1f}% move over last 10 sessions")

    if price_vs_20 is not None:
        side = "below" if price_vs_20 <= 0 else "above"
        hints.append(f"price is {abs(price_vs_20):.1f}% {side} 20-day avg")

    if rsi is not None:
        if rsi <= 40:
            hints.append(f"RSI ≈ {rsi:.0f} (weak momentum)")
        elif rsi <= 50:
            hints.append(f"RSI ≈ {rsi:.0f} (neutral / weak)")
        else:
            hints.append(f"RSI ≈ {rsi:.0f} (not deeply oversold)")

    if vol_vs_ma20 is not None and abs(vol_vs_ma20) >= 15:
        side = "higher" if vol_vs_ma20 > 0 else "lighter"
        hints.append(f"volume {abs(vol_vs_ma20):.0f}% {side} than normal")

    if trend is not None:
        if trend <= -0.6:
            hints.append("downtrend looks strong")
        elif trend >= 0.6:
            hints.append("trend mixed — some hidden strength")

    tech_line = " | ".join(hints[:3]) + ". " if hints else ""

    risk_line = (
        "Signal: SELL zone — many traders reduce exposure or tighten stops "
        "instead of adding new risk."
        if kind in ("STRONG_SELL", "SELL")
        else "Signal: HOLD — waiting for a clearer edge may be safer."
    )

    return short, tech_line + risk_line


# ---------------------------------------------------------
# MAIN SCAN LOGIC
# ---------------------------------------------------------
def compute_hotlist_and_bearwatch():
    ensure_bullbrain_loaded()

    buys = []
    bears = []
    everything = []

    log(f"Scanning {len(REAL_TICKERS)} tickers…")

    for idx, sym in enumerate(REAL_TICKERS, start=1):
        if idx % 50 == 0:
            log(f"...processed {idx}/{len(REAL_TICKERS)}")

        try:
            candles = backend.fetch_daily_candles(sym)
            feats_vec, feat_dict, last_close = backend.compute_bullbrain_features(candles)
            infer = backend.bullbrain_infer(feats_vec)
        except Exception as e:
            log(f"Inference error {sym}: {e}")
            continue

        prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
        prob_down = float(infer.get("probability_down") or (1 - prob_up))

        kind = classify_signal(prob_up, prob_down)
        confidence = max(prob_up, prob_down) * 100

        company_name = COMPANY_NAMES.get(sym, "")

        everything.append(
            {
                "symbol": sym,
                "company_name": company_name,
                "prob_up_raw": prob_up,
                "prob_down_raw": prob_down,
                "confidence": confidence,
                "kind": kind,
                "feat_dict": feat_dict,
            }
        )

        base = {
            "symbol": sym,
            "company_name": company_name,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "confidence": round(confidence, 2),
        }

        if kind in ("STRONG_BUY", "BUY"):
            short, risk = build_buy_explanations(prob_up, prob_down, kind, feat_dict)
            buys.append({**base, "signal": "BUY", "kind": kind, "explanation_short": short, "explanation_risk": risk})
        else:
            short, risk = build_bear_explanations(prob_up, prob_down, kind, feat_dict)
            bears.append({**base, "signal": "SELL" if kind in ("STRONG_SELL", "SELL") else "HOLD",
                          "kind": kind, "explanation_short": short, "explanation_risk": risk})

    # ------ HOTLIST ------
    buys.sort(key=lambda x: x["prob_up"], reverse=True)
    hotlist = buys[:5]

    # fallback
    if len(hotlist) < 5:
        existing = {h["symbol"] for h in hotlist}
        extras = [e for e in everything if e["symbol"] not in existing]
        extras.sort(key=lambda x: x["prob_up_raw"], reverse=True)

        for extra in extras[: 5 - len(hotlist)]:
            short, risk = build_buy_explanations(
                extra["prob_up_raw"], extra["prob_down_raw"], "WATCHLIST_BUY", extra["feat_dict"]
            )
            hotlist.append(
                {
                    "symbol": extra["symbol"],
                    "company_name": extra["company_name"],
                    "prob_up": round(extra["prob_up_raw"], 4),
                    "prob_down": round(extra["prob_down_raw"], 4),
                    "confidence": round(extra["confidence"], 2),
                    "signal": "BUY",
                    "kind": "WATCHLIST_BUY",
                    "explanation_short": short,
                    "explanation_risk": risk,
                }
            )

    hotlist = hotlist[:5]

    # ------ BEARWATCH ------
    bears.sort(key=lambda x: x["prob_down"], reverse=True)
    bearwatch = bears[:5]

    now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    return (
        {"count": len(hotlist), "hotlist": hotlist, "updated_at": now},
        {"count": len(bearwatch), "bearwatch": bearwatch, "updated_at": now},
    )


# ---------------------------------------------------------
# SAVE TO FIRESTORE
# ---------------------------------------------------------
def save_docs(hotlist_doc, bearwatch_doc):
    db = get_db()
    col = db.collection("bullsignals_ai")

    col.document("market_hotlist").set(hotlist_doc, merge=True)
    log("Saved market_hotlist")

    col.document("market_bearwatch").set(bearwatch_doc, merge=True)
    log("Saved market_bearwatch")


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
def main():
    start = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    log(f"Scan started at {start}")

    try:
        hot, bear = compute_hotlist_and_bearwatch()
        save_docs(hot, bear)
        end = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        log(f"Scan completed at {end}")

    except Exception as e:
        log(f"Fatal cron error: {e}")


if __name__ == "__main__":
    main()
