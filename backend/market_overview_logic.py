# backend/market_overview_logic.py
# ---------------------------------------------------------
# BullSignalsAI — Market Overview Logic
# ---------------------------------------------------------
# Pure backend logic:
# - No FastAPI
# - No Firestore
# - No cron
# - Safe for reuse in cron and main.py
# ---------------------------------------------------------

import math
from typing import Dict, Any

from backend.market_data import (
    fetch_sp500_change_pct,
    fetch_vix_value,
    fetch_fear_greed_index,
)


# ---------------------------------------------------------
# Risk level classifier (unchanged behavior)
# ---------------------------------------------------------
def _compute_risk_from_stats(
    sp500_change: float,
    vix: float,
    fear_greed_value: int,
) -> str:
    """
    Convert macro stats into a simple risk label.
    Behavior matches original logic.
    """

    if vix >= 25 or fear_greed_value <= 25:
        return "High Risk"

    if vix >= 18 or fear_greed_value <= 40:
        return "Moderate Risk"

    if sp500_change <= -1.2:
        return "Moderate Risk"

    return "Low Risk"


# ---------------------------------------------------------
# Market overview builder (core logic)
# ---------------------------------------------------------
def build_market_overview() -> Dict[str, Any]:
    """
    Lightweight market overview snapshot.

    Returns schema identical to what UI already consumes:
    {
      sp500_change,
      vix,
      fearGreed: { value, label },
      risk_level
    }
    """

    try:
        sp500_change = fetch_sp500_change_pct()
        vix = fetch_vix_value()
        fear_greed = fetch_fear_greed_index()

        fear_value = int(fear_greed.get("value", 50))
        fear_label = fear_greed.get("label", "Neutral")

        risk_level = _compute_risk_from_stats(
            sp500_change=sp500_change,
            vix=vix,
            fear_greed_value=fear_value,
        )

        return {
            "sp500_change": round(sp500_change, 2),
            "vix": round(vix, 2),
            "fearGreed": {
                "value": fear_value,
                "label": fear_label,
            },
            "risk_level": risk_level,
        }

    except Exception:
        # Fallback — identical spirit to original safety net
        return {
            "sp500_change": 0.0,
            "vix": 15.0,
            "fearGreed": {
                "value": 50,
                "label": "Neutral",
            },
            "risk_level": "Moderate Risk",
        }
