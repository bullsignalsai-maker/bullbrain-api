# backend/stockdetail_logic.py
# ------------------------------------------------------------
# Stock Detail Orchestration Logic
#
# Purpose:
# - Central brain for StockDetail computation
# - Used by:
#     • main.py (read-only API)
#     • stockdetail_cron.py (Firestore precompute)
# - NO routing
# - NO Firestore writes
# ------------------------------------------------------------

from typing import Dict, Any, Optional

# ------------------------------------------------------------
# Market data
# ------------------------------------------------------------
from backend.market_data import (
    fetch_daily_candles,
    fetch_quote,
)

# ------------------------------------------------------------
# BullBrain (ML)
# ------------------------------------------------------------
from backend.bullbrain import (
    ensure_bullbrain_loaded,
    bullbrain_infer,
    compute_bullbrain_features,
)

# ------------------------------------------------------------
# Technical indicators
# ------------------------------------------------------------
from backend.technicals import build_technical_snapshot

# ------------------------------------------------------------
# Smart patterns
# ------------------------------------------------------------
from backend.smart_patterns import (
    detect_smart_pattern,
    scan_smart_pattern_history,
)

# ------------------------------------------------------------
# Grok / XAI
# ------------------------------------------------------------
from backend.grok_ai import get_stockdetail_grok

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
import datetime

def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )



# ============================================================
# PUBLIC API
# ============================================================

def build_stockdetail_payload(
    symbol: str,
    *,
    force_grok: bool = False,
    include_pattern_history: bool = True,
) -> Dict[str, Any]:
    """
    Build the FULL StockDetail payload.

    Used by:
      - GET /stockdetail/{symbol}
      - stockdetail_cron.py
      - Firestore cache refresh

    This function:
      ✔ orchestrates everything
      ✔ performs NO persistence
      ✔ performs NO routing
    """

    symbol = symbol.upper()

    # --------------------------------------------------------
    # 0) Ensure BullBrain model is loaded (SAFE, idempotent)
    # --------------------------------------------------------
    ensure_bullbrain_loaded()

    # --------------------------------------------------------
    # 1) Fetch market data
    # --------------------------------------------------------
    candles = fetch_daily_candles(symbol)
    quote = fetch_quote(symbol)

    if not candles or not quote:
        return {
            "symbol": symbol,
            "error": "Market data unavailable",
            "generatedAt": _utc_now_iso(),
        }

    # --------------------------------------------------------
    # 2) BullBrain features + inference
    # --------------------------------------------------------
    bullbrain_block: Optional[Dict[str, Any]] = None

    features_vec, feature_dict, last_close = compute_bullbrain_features(candles)

    if features_vec is not None:
        out = bullbrain_infer(features_vec) or {}

        prob_up = float(
            out.get("probability_up")
            or out.get("raw_output")
            or 0.5
        )

        bullbrain_block = {
            "signal": out.get("signal", "NEUTRAL"),
            "probability_up": round(prob_up * 100.0, 1),  # percent
            "confidence": (
                "High" if prob_up >= 0.66
                else "Moderate" if prob_up >= 0.55
                else "Low"
            ),
            "version": out.get("version") or "v2",
        }

    # --------------------------------------------------------
    # 3) Technical snapshot
    # --------------------------------------------------------
    technical = build_technical_snapshot(
        symbol=symbol,
        features=feature_dict,
        last_close=last_close,
    )

    # --------------------------------------------------------
    # 4) Smart Pattern (single-day)
    # --------------------------------------------------------
    smart_pattern = detect_smart_pattern(
        features=feature_dict,
        quote=quote,
        technical=technical,
    )

    # --------------------------------------------------------
    # 5) Smart Pattern History (optional, heavy)
    # --------------------------------------------------------
    pattern_history = None
    if include_pattern_history:
        try:
            pattern_history = scan_smart_pattern_history(
                symbol=symbol,
                candles=candles,
            )
        except Exception as e:
            pattern_history = {
                "error": "Pattern history unavailable",
                "detail": str(e),
            }

    # --------------------------------------------------------
    # 6) Grok AI (natural language)
    # --------------------------------------------------------
    grok_block = get_stockdetail_grok(
        symbol=symbol,
        quote=quote,
        technical=technical,
        force=force_grok,
    )

    # --------------------------------------------------------
    # 7) Final payload
    # --------------------------------------------------------
    return {
        "symbol": symbol,
        "quote": quote,
        "bullbrain": bullbrain_block,
        "technical": technical,
        "smartPattern": smart_pattern,
        "patternHistory": pattern_history,
        "grok": grok_block,
        "generatedAt": _utc_now_iso(),
    }
