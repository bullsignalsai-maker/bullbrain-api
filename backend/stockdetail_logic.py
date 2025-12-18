# backend/stockdetail_logic.py

from typing import Dict, Any, Optional
import time

# ----------------------------
# Data Sources
# ----------------------------
from backend.market_data import (
    fetch_quote,
    fetch_daily_candles,
    fetch_news,
)

# ----------------------------
# Models & Indicators
# ----------------------------
from backend.bullbrain import run_bullbrain
from backend.technicals import build_technical_snapshot

# ----------------------------
# Intelligence Engines
# ----------------------------
from backend.smart_patterns import build_smart_pattern
from backend.grok_ai import get_grok_stock_insight


# --------------------------------------------------------
# CORE ORCHESTRATION
# --------------------------------------------------------

def build_stockdetail_core(
    symbol: str,
    limit_candles: int = 180,
    force_grok: bool = False,
) -> Dict[str, Any]:
    """
    Builds the full Stock Detail payload.
    NO Firestore.
    NO FastAPI.
    NO caching.
    """

    symbol = symbol.upper()

    # ----------------------------------------------------
    # 1️⃣ Market Data
    # ----------------------------------------------------
    quote = fetch_quote(symbol)
    candles = fetch_daily_candles(symbol, limit=limit_candles)

    last_price = quote.get("current") if quote else None

    # ----------------------------------------------------
    # 2️⃣ BullBrain v2 (48 features)
    # ----------------------------------------------------
    bullbrain_block = None
    bull_prob_up = None
    features = None

    if candles:
        bb = run_bullbrain(candles)
        bullbrain_block = bb.get("block")
        bull_prob_up = bb.get("prob_up")
        features = bb.get("features")

    # ----------------------------------------------------
    # 3️⃣ Technical Snapshot (Human-readable)
    # ----------------------------------------------------
    technical = None
    if features and last_price:
        technical = build_technical_snapshot(
            symbol=symbol,
            features=features,
            last_price=last_price,
        )

    # ----------------------------------------------------
    # 4️⃣ Grok AI Reasoning
    # ----------------------------------------------------
    grok = get_grok_stock_insight(
        symbol=symbol,
        quote=quote,
        technical=technical,
        force=force_grok,
    )

    grok_prob_up = grok.get("prob_up")

    # ----------------------------------------------------
    # 5️⃣ Hybrid Signal (BullBrain + Grok)
    # ----------------------------------------------------
    hybrid_prob_up = _hybrid_prob(bull_prob_up, grok_prob_up)
    hybrid_signal = _signal_from_prob(hybrid_prob_up)

    # ----------------------------------------------------
    # 6️⃣ Smart Pattern Detection
    # ----------------------------------------------------
    smart_pattern = build_smart_pattern(
        symbol=symbol,
        candles=candles,
    )

    # ----------------------------------------------------
    # 7️⃣ News
    # ----------------------------------------------------
    news = fetch_news(symbol, limit=5)

    # ----------------------------------------------------
    # 8️⃣ Final Payload
    # ----------------------------------------------------
    return {
        "symbol": symbol,
        "asOf": int(time.time()),

        "quote": quote,
        "price": last_price,

        "bullbrain": bullbrain_block,
        "technical": technical,

        "grok": grok,

        "hybridProbUp": hybrid_prob_up,
        "hybridSignal": hybrid_signal,

        "smartPattern": smart_pattern,
        "news": news,
    }


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def _hybrid_prob(
    bull: Optional[float],
    grok: Optional[float],
) -> float:
    """
    Conservative blend.
    """
    if bull is None and grok is None:
        return 0.5
    if bull is None:
        return grok
    if grok is None:
        return bull
    return round((bull * 0.6) + (grok * 0.4), 4)


def _signal_from_prob(p: float) -> str:
    if p >= 0.65:
        return "BUY"
    if p <= 0.35:
        return "SELL"
    return "HOLD"
