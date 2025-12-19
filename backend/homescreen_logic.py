# backend/homescreen_logic.py
# ============================================================
# BullSignalsAI — HomeScreen Logic (MAG7 snapshot builder)
#
# Architecture:
# - Runs INSIDE API process
# - BullBrain model is loaded ONCE at startup
# - This file NEVER downloads or reloads model
# - Safe for background workers
# ============================================================

from typing import Dict, Any, List
import datetime
import math

from symbols_clean import COMPANY_NAMES
from backend.market_data import fetch_daily_candles, fetch_quote
from backend.bullbrain import (
    ensure_bullbrain_loaded,
    compute_bullbrain_features,
    bullbrain_infer,
)

# ------------------------------------------------------------
# MAG7 Universe
# ------------------------------------------------------------
MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_float(v):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


# ------------------------------------------------------------
# Core Builder
# ------------------------------------------------------------
def build_mag7_snapshot() -> Dict[str, Any]:
    """
    Build HomeScreen MAG7 snapshot.

    Guarantees:
    - BullBrain model is loaded (idempotent)
    - No model downloads here
    - Fail-soft per symbol
    - Deterministic output order
    """

    # 🔑 Assert model availability (SAFE + IDEMPOTENT)
    ensure_bullbrain_loaded()

    items: List[Dict[str, Any]] = []

    for symbol in MAG7:
        try:
            # --------------------------------------------
            # Market data
            # --------------------------------------------
            candles = fetch_daily_candles(symbol)
            if not candles:
                continue

            quote = fetch_quote(symbol) or {}
            price = _safe_float(quote.get("price"))
            change_pct = _safe_float(quote.get("changePct"))

            price_timestamp = None
            ts = quote.get("timestamp")
            if ts:
                try:
                    price_timestamp = {
                        "epoch": int(ts),
                        "iso": datetime.datetime.fromtimestamp(
                            int(ts), tz=datetime.timezone.utc
                        )
                        .isoformat()
                        .replace("+00:00", "Z"),
                    }
                except Exception:
                    pass

            # --------------------------------------------
            # BullBrain features + inference
            # --------------------------------------------
            feats_vec, _, _ = compute_bullbrain_features(candles)
            if feats_vec is None:
                continue

            infer = bullbrain_infer(feats_vec)
            if not infer.get("ok", True):
                continue

            # --------------------------------------------
            # Build item
            # --------------------------------------------
            item = {
                "symbol": symbol,
                "company_name": COMPANY_NAMES.get(symbol, symbol),
                "price": price,
                "changePct": change_pct,
                "price_timestamp": price_timestamp,
                "bullbrain": {
                    "signal": infer.get("signal", "HOLD"),
                    "confidence": float(infer.get("confidence", 50.0)),
                    "prob_up": round(float(infer.get("probability_up", 0.5)), 4),
                    "prob_down": round(float(infer.get("probability_down", 0.5)), 4),
                    "version": infer.get("version"),
                },
                "updated_at": _utc_now_iso(),
            }

            items.append(item)

        except Exception:
            # Fail-soft per ticker — never kill batch
            continue

    # Deterministic ordering for Firestore & UI diffing
    items.sort(key=lambda x: x["symbol"])

    return {
        "count": len(items),
        "items": items,
        "updated_at": _utc_now_iso(),
    }


# ------------------------------------------------------------
# Public Wrapper
# ------------------------------------------------------------
def build_homescreen_mag7_block() -> Dict[str, Any]:
    """
    Public entrypoint used by:
    - Background worker
    - Internal API (if needed)
    """
    return build_mag7_snapshot()
