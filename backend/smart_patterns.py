# backend/smart_patterns.py

from typing import Dict, Any, Optional


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def detect_smart_patterns(
    symbol: str,
    candles: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect current smart pattern + historical performance.

    Returns a UI-safe structure ALWAYS.
    No exceptions bubble up.
    """

    try:
        raw = _scan_pattern_history(symbol, candles)
        return _build_safe_response(raw)
    except Exception:
        return _empty_pattern_response()


# ------------------------------------------------------------
# Internal Logic (ported from existing implementation)
# ------------------------------------------------------------

def _scan_pattern_history(symbol: str, candles: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Your existing smart pattern detection logic goes here.
    This function may return None or a raw pattern object.
    """

    if not candles or "close" not in candles:
        return None

    # ⚠️ PLACEHOLDER for your real logic
    # Keep exactly the same algorithm you already have
    # This stub preserves structure

    return {
        "currentPattern": {
            "pattern": "Bullish Engulfing",
            "headline": "Bullish reversal detected",
            "winRate": 0.63,
        },
        "historyForCurrent": {
            "occurrences": 42,
            "samples": [
                "2024-01-12",
                "2023-11-07",
                "2023-08-18",
            ],
            "forwardReturns": {
                "1d": 0.004,
                "5d": 0.021,
                "20d": 0.083,
            },
        },
    }


# ------------------------------------------------------------
# Safe Output Builder
# ------------------------------------------------------------

def _build_safe_response(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert raw pattern output into UI-safe payload.
    """

    if not raw:
        return _empty_pattern_response()

    cp = raw.get("currentPattern") or {}
    hist = raw.get("historyForCurrent") or {}

    if not cp.get("pattern"):
        return _empty_pattern_response()

    return {
        "smartPattern": {
            "pattern": cp.get("pattern"),
            "headline": cp.get("headline"),
            "winRate": cp.get("winRate"),
            "occurrences": hist.get("occurrences", 0),
            "samples": hist.get("samples", []),
            "forwardReturns": hist.get("forwardReturns", {}),
        },
        "patternDates": (hist.get("samples") or [])[:5],
        "patternStats": raw,  # full raw object for debug / advanced UI
    }


def _empty_pattern_response() -> Dict[str, Any]:
    """
    Guaranteed-safe empty response.
    Frontend will never crash on this.
    """
    return {
        "smartPattern": {
            "pattern": None,
            "headline": None,
            "winRate": None,
            "occurrences": 0,
            "samples": [],
            "forwardReturns": {},
        },
        "patternDates": [],
        "patternStats": None,
    }
