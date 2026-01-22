# backend/decision_explainer.py

from typing import Dict, Any, List


# -------------------------------------------------
# Evidence per gate (UI-friendly, optional)
# -------------------------------------------------
def build_gate_evidence(stock: Dict[str, Any], gate: str) -> List[str]:
    technical = stock.get("technical") or {}
    pattern = stock.get("pattern") or {}
    bull = stock.get("bullbrain") or {}

    evidence: List[str] = []

    if gate == "Feature Consensus":
        rsi = (technical.get("rsi") or {}).get("label")
        macd = (technical.get("macd") or {}).get("label")
        trend = (technical.get("trend") or {}).get("label")

        if rsi:
            evidence.append(f"RSI momentum was {rsi.lower()}")
        if macd:
            evidence.append(f"MACD momentum was {macd.lower()}")
        if trend:
            evidence.append(f"Trend regime was {trend.lower()}")

    elif gate == "Pattern Quality":
        evidence.append("Pattern historical performance did not meet strength threshold")

    elif gate == "Pattern Alignment":
        bias = pattern.get("bias")
        if bias:
            evidence.append(f"Pattern bias was {bias.lower()}")

    elif gate == "Expected Value":
        evidence.append("Historical outcomes showed limited reward relative to risk")

    elif gate == "Exhaustion / Fragility":
        evidence.append("Recent price move appeared extended or unstable")

    elif gate == "Liquidity":
        evidence.append("Trading activity was below reliability threshold")

    elif gate == "Market Regime":
        regime = (stock.get("decision") or {}).get("quality", {}).get("regime")
        if regime:
            evidence.append(f"Market regime classified as {regime.lower()}")

    return evidence


def explain_decision_ladder(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision", {}) or {}
    bull = stock.get("bullbrain", {}) or {}
    tech = stock.get("technical", {}) or {}
    features = stock.get("features_meta", {}) or {}
    pattern = stock.get("pattern", {}) or {}
    history = stock.get("patternHistory", {}) or {}

    final_signal = decision.get("finalSignal", "HOLD")
    confidence = bull.get("confidence")

    reasons = decision.get("decisionReasons", []) or []

    def failed(code: str) -> bool:
        return code in reasons

    ladder = []

    # 1️⃣ Liquidity
    ladder.append({
        "gate": "Liquidity",
        "status": "failed" if failed("Liquidity=POOR") else "passed",
        "explanation": (
            "Trading volume is too thin to trust price moves."
            if failed("Liquidity=POOR")
            else "Trading volume is healthy."
        ),
        "evidenceSummary": build_gate_evidence(stock, "Liquidity"),
    })

    # 2️⃣ Market Regime
    regime = (decision.get("quality") or {}).get("regime")
    ladder.append({
        "gate": "Market Regime",
        "status": "passed",
        "explanation": f"Market regime is {regime or 'normal'}.",
        "evidenceSummary": build_gate_evidence(stock, "Market Regime"),
    })

    # 3️⃣ Feature Consensus
    ladder.append({
        "gate": "Feature Consensus",
        "status": "failed" if failed("WeakFeatureConsensus") else "passed",
        "explanation": (
            "Trend, momentum, and volume indicators do not align."
            if failed("WeakFeatureConsensus")
            else "Key indicators agree on direction."
        ),
        "evidenceSummary": build_gate_evidence(stock, "Feature Consensus"),
    })

    # 4️⃣ Pattern Quality
    ladder.append({
        "gate": "Pattern Quality",
        "status": "failed" if failed("PatternQualityFailed") else "passed",
        "explanation": (
            "Pattern has weak historical performance."
            if failed("PatternQualityFailed")
            else "Pattern shows acceptable historical edge."
        ),
        "evidenceSummary": build_gate_evidence(stock, "Pattern Quality"),
    })

    # 5️⃣ Pattern Alignment
    ladder.append({
        "gate": "Pattern Alignment",
        "status": "failed" if failed("SignalPatternConflict") else "passed",
        "explanation": (
            "Pattern bias conflicts with model direction."
            if failed("SignalPatternConflict")
            else "Pattern bias aligns with model signal."
        ),
        "evidenceSummary": build_gate_evidence(stock, "Pattern Alignment"),
    })

    # 6️⃣ Expected Value
    ladder.append({
        "gate": "Expected Value",
        "status": "blocked" if failed("NegativeEV") else "passed",
        "explanation": (
            "Historical reward does not justify the risk."
            if failed("NegativeEV")
            else "Risk-reward profile is acceptable."
        ),
        "evidenceSummary": build_gate_evidence(stock, "Expected Value"),
    })

    # 7️⃣ Exhaustion / Fragility
    ladder.append({
        "gate": "Exhaustion / Fragility",
        "status": "failed" if failed("MomentumExhausted") else "passed",
        "explanation": (
            "Move appears stretched or unstable."
            if failed("MomentumExhausted")
            else "Price action is stable."
        ),
        "evidenceSummary": build_gate_evidence(stock, "Exhaustion / Fragility"),
    })


    # What would change?
    improvements = []
    if failed("WeakFeatureConsensus"):
        improvements.append("Stronger alignment between trend, momentum, and volume")
    if failed("SignalPatternConflict"):
        improvements.append("Pattern bias aligning with model direction")
    if failed("NegativeEV"):
        improvements.append("Improved historical reward relative to risk")

    return {
        "symbol": stock.get("symbol"),
        "finalSignal": final_signal,
        "confidence": confidence,
        "confidenceLabel": (
            "High" if confidence and confidence >= 70
            else "Moderate" if confidence and confidence >= 55
            else "Low"
        ),
        "summary": {
            "headline": f"{final_signal} — mixed signals" if final_signal == "HOLD" else f"{final_signal} signal",
            "why": improvements[0] if improvements else "All quality checks passed."
        },
        "decisionLadder": ladder,
        "whatWouldChange": improvements
    }
