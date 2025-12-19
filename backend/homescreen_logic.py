# backend/homescreen_logic.py
# ============================================================
# BullSignalsAI — HomeScreen Logic (MAG7 snapshot builder)
# ============================================================

from typing import Dict, Any, List
import datetime
import math

from symbols_clean import COMPANY_NAMES

from backend.market_data import (
    fetch_daily_candles,
    fetch_quote,
)

from backend.technicals import compute_bullbrain_features
from backend.bullbrain import bullbrain_infer


# ------------------------------------------------------------
# Constants
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
# Build MAG7 BullBrain snapshot
# ------------------------------------------------------------
def build_mag7_snapshot() -> Dict[str, Any]:
    """
    Builds HomeScreen MAG7 snapshot using:
      - Latest daily candles
      - BullBrain v2 (48 features)
      - Live quote (price + change)
    Assumes BullBrain model is ALREADY loaded by cron.
    """

    items: List[Dict[str, Any]] = []

    for symbol in MAG7:
        try:
            # ----------------------------
            # 1) Market data
            # ----------------------------
            candles = fetch_daily_candles(symbol)
            if not candles:
                continue

            quote = fetch_quote(symbol) or {}

            price = _safe_float(quote.get("price"))
            change_pct = _safe_float(quote.get("changePct"))

            price_ts = quote.get("timestamp")
            price_timestamp = None
            if price_ts:
                try:
                    price_timestamp = {
                        "epoch": int(price_ts),
                        "iso": datetime.datetime.fromtimestamp(
                            int(price_ts), tz=datetime.timezone.utc
                        )
                        .isoformat()
                        .replace("+00:00", "Z"),
                    }
                except Exception:
                    price_timestamp = None

            # ----------------------------
            # 2) BullBrain features + inference
            # ----------------------------
            feats_vec, feat_dict, last_close = compute_bullbrain_features(candles)
            if feats_vec is None:
                continue

            infer = bullbrain_infer(feats_vec)

            prob_up = float(infer.get("probability_up", 0.5))
            prob_down = float(infer.get("probability_down", 0.5))
            signal = infer.get("signal", "HOLD")
            confidence = float(infer.get("confidence", 50.0))

            # ----------------------------
            # 3) Assemble item
            # ----------------------------
            item = {
                "symbol": symbol,
                "company_name": COMPANY_NAMES.get(symbol, symbol),
                "price": price,
                "changePct": change_pct,
                "price_timestamp": price_timestamp,
                "bullbrain": {
                    "signal": signal,
                    "confidence": confidence,
                    "prob_up": round(prob_up, 4),
                    "prob_down": round(prob_down, 4),
                },
                "updated_at": _utc_now_iso(),
            }

            items.append(item)

        except Exception as e:
            # Fail-soft per ticker
            continue

    return {
        "count": len(items),
        "items": items,
        "updated_at": _utc_now_iso(),
    }


# ------------------------------------------------------------
# Public builder (used by homescreen_cron)
# ------------------------------------------------------------
def build_homescreen_mag7_block() -> Dict[str, Any]:
    """
    Public wrapper to keep cron clean.
    """
    return build_mag7_snapshot()
