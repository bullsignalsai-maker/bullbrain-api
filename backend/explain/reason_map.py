# backend/explain/reason_map.py
from __future__ import annotations

"""
Maps your existing decision reason codes (and common variants) into clean,
non-vague, user-facing explanations and tighter institutional phrasing.
"""

REASON_MAP: dict[str, dict] = {
    # Liquidity
    "Liquidity=POOR": {
        "title": "Liquidity filter failed",
        "short": "Liquidity is poor, so signals are suppressed.",
        "institutional": "Liquidity conditions are below threshold; directional signals are withheld to reduce slippage and noise."
    },
    "Liquidity=THIN": {
        "title": "Liquidity filter failed",
        "short": "Liquidity is thin, so confidence is reduced.",
        "institutional": "Thin liquidity reduces signal quality; the model prefers confirmation before acting."
    },

    # Pattern quality
    "PatternQualityFailed": {
        "title": "Pattern quality failed",
        "short": "Pattern history is too weak to trust.",
        "institutional": "Pattern statistics do not meet minimum sample/win-rate thresholds; the pattern is treated as context only."
    },

    # Regime mismatch
    "PatternNotAllowedInTRENDING": {
        "title": "Regime mismatch",
        "short": "Pattern doesn’t fit current regime.",
        "institutional": "Pattern is not statistically reliable in the current market regime; signal is suppressed."
    },
    "PatternNotAllowedInRANGING": {
        "title": "Regime mismatch",
        "short": "Pattern doesn’t fit current regime.",
        "institutional": "Pattern/regime compatibility filter failed; signal is suppressed."
    },
    "PatternNotAllowedInHIGH_VOL": {
        "title": "Regime mismatch",
        "short": "Pattern doesn’t fit high-volatility regime.",
        "institutional": "High-volatility regime reduces pattern edge; signal is withheld until structure stabilizes."
    },

    # Alignment and gates
    "SignalPatternConflict": {
        "title": "Signal conflict",
        "short": "Model and pattern bias disagree.",
        "institutional": "Directional model output conflicts with pattern bias; conflict resolution forces HOLD."
    },
    "TimeframeMisalignment": {
        "title": "Timeframe misalignment",
        "short": "Returns don’t confirm the direction.",
        "institutional": "Multi-timeframe agreement gate failed; directional follow-through is not confirmed."
    },
    "VolumeGateFailed": {
        "title": "Volume confirmation failed",
        "short": "Volume doesn’t confirm the move.",
        "institutional": "Participation is insufficient versus baselines; directional signal is withheld."
    },
    "WeakFeatureConsensus": {
        "title": "Weak consensus",
        "short": "Indicators don’t agree enough.",
        "institutional": "Feature consensus score is below threshold; edge is insufficient for a directional call."
    },
    "NoUpsidePressure": {
        "title": "No upside pressure",
        "short": "Upward pressure isn’t present.",
        "institutional": "Directional pressure score does not support upside continuation; signal is suppressed."
    },
    "NoDownsidePressure": {
        "title": "No downside pressure",
        "short": "Downward pressure isn’t present.",
        "institutional": "Directional pressure score does not support downside continuation; signal is suppressed."
    },
    "SignalTooFragile": {
        "title": "Fragile setup",
        "short": "Setup is unstable; HOLD.",
        "institutional": "Fragility index exceeds threshold; expected whipsaw risk is elevated."
    },
    "MomentumExhausted": {
        "title": "Momentum exhaustion",
        "short": "Move looks exhausted; wait.",
        "institutional": "Momentum exhaustion filter triggered; continuation odds are reduced."
    },
    "NegativeEV": {
        "title": "Negative expected value",
        "short": "Expected value is not attractive.",
        "institutional": "EV score is non-positive after fragility penalties; signal is suppressed."
    },
    "ALL_GATES_PASSED": {
        "title": "All gates passed",
        "short": "Conditions aligned for this signal.",
        "institutional": "All quality gates passed; conditions are aligned across structure, momentum, and participation."
    },
}


def explain_reason(code: str, tone: str = "institutional") -> dict:
    """
    Returns {title, text}. If unknown, returns a safe fallback.
    """
    block = REASON_MAP.get(code)
    if not block:
        return {"title": "Decision rule", "text": code}
    text = block.get(tone) or block.get("short") or code
    return {"title": block.get("title", "Decision rule"), "text": text}


def explain_reasons(codes: list[str] | None, tone: str = "institutional") -> list[dict]:
    if not codes:
        return []
    return [explain_reason(c, tone=tone) for c in codes]
