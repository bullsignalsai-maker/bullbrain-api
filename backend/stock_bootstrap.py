# backend/stock_bootstrap.py
# ---------------------------------------------------------
# On-Demand Stock Bootstrapper
# - Uses Firestore candle cache
# - Computes BullBrain signal
# - Saves to stocks/{SYMBOL}
# ---------------------------------------------------------

from typing import Dict, Any

import main as backend
from backend.candle_store import get_candles
from backend.stock_repo import (
    get_stock,
    save_stock,
    is_stock_fresh,
    utc_now_iso,
)

from symbols_clean import COMPANY_NAMES
from backend.bull_insights import generate_bull_insights
from backend.stock_display_intelligence import build_display_intelligence


# ---------------------------------------------------------
# Ensure model loaded (same pattern as market_cron)
# ---------------------------------------------------------
def ensure_bullbrain_loaded():
    if backend.bullbrain_model is not None:
        return
    backend.bullbrain_model = backend.load_bullbrain_model()


# ---------------------------------------------------------
# Signal classification (copied exactly — DO NOT CHANGE)
# ---------------------------------------------------------
def classify_signal(prob_up: float, prob_down: float) -> str:
    edge_up = prob_up - prob_down
    edge_down = prob_down - prob_up

    if prob_up >= 0.58 and edge_up >= 0.08:
        return "STRONG_BUY"
    if prob_up >= 0.52 and edge_up >= 0.02:
        return "BUY"
    if prob_down >= 0.58 and edge_down >= 0.08:
        return "STRONG_SELL"
    if prob_down >= 0.52 and edge_down >= 0.02:
        return "SELL"

    return "HOLD"



# ---------------------------------------------------------
# Public entry point
# ---------------------------------------------------------
def bootstrap_stock(symbol: str) -> Dict[str, Any]:
    """
    Returns cached stock intelligence if fresh.
    Otherwise computes, saves, and returns.
    """

    symbol = symbol.upper()

    # 1️⃣ Cache check
    cached = get_stock(symbol)
    if cached and is_stock_fresh(cached):
        return cached

    # 2️⃣ Ensure model
    ensure_bullbrain_loaded()

    # 3️⃣ Candles (Firestore-backed)
    candles = get_candles(symbol, min_points=120)
    if not candles:
        raise RuntimeError(f"No candles available for {symbol}")

    # 4️⃣ Features
    feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles)
    if feats_vec is None:
        raise RuntimeError("Feature generation failed")

    # 5️⃣ Inference
    infer = backend.bullbrain_infer(feats_vec)
    if infer is None:
        raise RuntimeError("Inference failed")

    prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
    prob_down = float(infer.get("probability_down") or (1.0 - prob_up))

    kind = classify_signal(prob_up, prob_down)
    signal = (
        "BUY" if kind in ("BUY", "STRONG_BUY")
        else "SELL" if kind in ("SELL", "STRONG_SELL")
        else "HOLD"
    )

    confidence = round(max(prob_up, prob_down) * 100.0, 2)

    # 6️⃣ Quote (safe)
    quote = backend.backend_fetch_quote(symbol) or {}

    # ---------------------------------------------------------
    # 6.5️⃣ Bull Insights (backend single source of truth)
    # ---------------------------------------------------------
    try:
        insights = generate_bull_insights(
            symbol=symbol,
            features=feat_dict,
            bullbrain={
                "signal": signal,
                "kind": kind,
                "confidence": confidence,
                "prob_up": prob_up,
                "prob_down": prob_down,
            },
            technical=None,  # stock_bootstrap does not compute /technical
            seed_key=f"{symbol}:{utc_now_iso()}",
        )
    except Exception as e:
        print(f"[bull_insights] failed for {symbol}: {e}")
        insights = None

    # ---------------------------------------------------------
    # 6.6️⃣ Display Intelligence (single source of truth for
    # user-facing signal labels — mirrors market_cron.py so the
    # on-demand path stays in sync with the batch path)
    # ---------------------------------------------------------
    try:
        display_intelligence = build_display_intelligence(
            symbol=symbol,
            stock={
                "quote": {
                    "price": quote.get("price") or quote.get("close"),
                    "changePct": quote.get("changePct"),
                },
                "technical": {},
                "bullbrain": {"signal": signal, "confidence": confidence},
                "decision": {},
                "marketAwareness": {},
            },
        )
    except Exception as e:
        print(f"[display_intelligence] failed for {symbol}: {e}")
        display_intelligence = None

    # 7️⃣ Build document
    doc = {
        "symbol": symbol,
        "company_name": COMPANY_NAMES.get(symbol, symbol),
        "insights": insights,

        "profile": {
            "sector": quote.get("sector"),
        },

        "quote": {
            "price": quote.get("price") or quote.get("close"),
            "changePct": quote.get("changePct"),
        },
        "quote_updated_at": utc_now_iso(),

        "bullbrain": {
            "signal": signal,
            "kind": kind,
            "confidence": confidence,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
        },
        "displayIntelligence": display_intelligence,

        "features_meta": {
            "feature_version": "bb48",
            "rsi14": feat_dict.get("rsi14"),
            "trend_strength_20": feat_dict.get("trend_strength_20"),
        },

        "candles": {
            "source": "firestore",
            "count": len(candles.get("close", [])),
            "last_ts": candles.get("timestamp", [])[-1],
        },

        "computed_at": utc_now_iso(),
        "compute_ttl_minutes": 60,
        "schema_version": "v1",
    }

    # 8️⃣ Persist
    save_stock(symbol, doc)

    return doc
