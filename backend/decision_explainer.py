# backend/decision_explainer.py

from typing import Dict, Any, List


def _num(v, digits=2):
    return round(v, digits) if isinstance(v, (int, float)) else v


def _metric(label: str, value: Any, unit: str | None = None) -> Dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "unit": unit,
    }


def build_gate_metrics(stock: Dict[str, Any], gate: str) -> List[Dict[str, Any]]:
    decision = stock.get("decision") or {}
    quality = decision.get("quality") or {}
    features = stock.get("features_meta") or {}
    indicators = stock.get("indicator_states") or {}
    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}
    bull = stock.get("bullbrain") or {}
    raw = bull.get("raw") or {}

    days5 = (history.get("forwardReturns") or {}).get("days5") or {}

    if gate == "Liquidity":
        return [
            _metric("Volume vs 20D Avg", _num(features.get("volume_vs_ma20_pct")), "%"),
            _metric("Volume Z-Score", _num(features.get("volume_zscore_20"))),
            _metric("Liquidity Quality", quality.get("liquidity") or indicators.get("liquidity_quality")),
        ]

    if gate == "Market Regime":
        return [
            _metric("Regime", quality.get("regime") or indicators.get("regime_state")),
            _metric("Volatility 20D", _num(features.get("volatility_20d")), "%"),
            _metric("ATR 14", _num(features.get("atr14")), "pts"),
        ]

    if gate == "Feature Consensus":
        return [
            _metric("RSI 14", _num(features.get("rsi14"))),
            _metric("MACD", _num(features.get("macd"))),
            _metric("MACD Signal", _num(features.get("macd_signal"))),
            _metric("Trend State", indicators.get("trend_composite") or indicators.get("trend")),
            _metric("Momentum State", indicators.get("momentum_composite") or indicators.get("momentum")),
        ]

    if gate == "Pattern Quality":
        return [
            _metric("Pattern", pattern.get("pattern") or pattern.get("patternLabel")),
            _metric("5D Win Rate", _num(days5.get("winRate") * 100, 1) if isinstance(days5.get("winRate"), (int, float)) else None, "%"),
            _metric("Sample Count", days5.get("count")),
            _metric("Pattern Quality", indicators.get("pattern_winrate_5d")),
        ]

    if gate == "Pattern Alignment":
        return [
            _metric("Pattern Bias", pattern.get("bias")),
            _metric("Model Signal", decision.get("final") or decision.get("finalSignal") or bull.get("signal")),
            _metric("Pattern Name", pattern.get("pattern") or pattern.get("patternLabel")),
        ]

    if gate == "Expected Value":
        return [
            _metric("5D Avg Return", _num(days5.get("avg")), "%"),
            _metric("5D Best Return", _num(days5.get("best")), "%"),
            _metric("5D Worst Return", _num(days5.get("worst")), "%"),
            _metric("5D Edge", indicators.get("pattern_edge_5d")),
        ]

    if gate == "Exhaustion / Fragility":
        return [
            _metric("RSI 14", _num(features.get("rsi14"))),
            _metric("Gap", _num(features.get("gap_pct")), "%"),
            _metric("Intraday Range", _num(features.get("intraday_range_pct")), "%"),
            _metric("Extension Risk", indicators.get("extension_risk")),
        ]

    return []


def build_gate_evidence(stock: Dict[str, Any], gate: str) -> List[str]:
    technical = stock.get("technical") or {}
    pattern = stock.get("pattern") or {}
    decision = stock.get("decision") or {}
    quality = decision.get("quality") or {}
    features = stock.get("features_meta") or {}
    history = stock.get("patternHistory") or {}

    evidence: List[str] = []

    if gate == "Feature Consensus":
        rsi = (technical.get("rsi") or {}).get("label")
        macd = (technical.get("macd") or {}).get("label")
        trend = (technical.get("trend") or {}).get("label")

        if rsi:
            evidence.append(f"RSI momentum was {rsi.lower()}.")
        if macd:
            evidence.append(f"MACD momentum was {macd.lower()}.")
        if trend:
            evidence.append(f"Trend regime was {trend.lower()}.")

    elif gate == "Pattern Quality":
        days5 = ((history.get("forwardReturns") or {}).get("days5") or {})
        wr = days5.get("winRate")
        count = days5.get("count")

        if isinstance(wr, (int, float)):
            evidence.append(f"Pattern 5-day win rate is {wr * 100:.1f}%.")
        if count is not None:
            evidence.append(f"Pattern sample count is {count}.")

    elif gate == "Pattern Alignment":
        bias = pattern.get("bias")
        if bias:
            evidence.append(f"Pattern bias was {bias.lower()}.")

    elif gate == "Expected Value":
        days5 = ((history.get("forwardReturns") or {}).get("days5") or {})
        avg = days5.get("avg")
        worst = days5.get("worst")

        if isinstance(avg, (int, float)):
            evidence.append(f"Average 5-day forward return is {avg:.2f}%.")
        if isinstance(worst, (int, float)):
            evidence.append(f"Worst 5-day forward return is {worst:.2f}%.")

    elif gate == "Exhaustion / Fragility":
        rsi = features.get("rsi14")
        gap = features.get("gap_pct")

        if isinstance(rsi, (int, float)):
            evidence.append(f"RSI is {rsi:.1f}.")
        if isinstance(gap, (int, float)):
            evidence.append(f"Opening gap is {gap:.2f}%.")

    elif gate == "Liquidity":
        v = features.get("volume_vs_ma20_pct")
        liquidity = quality.get("liquidity")

        if isinstance(v, (int, float)):
            evidence.append(f"Volume is {v:.1f}% versus the 20-day average.")
        if liquidity:
            evidence.append(f"Liquidity quality is {liquidity.lower()}.")

    elif gate == "Market Regime":
        regime = quality.get("regime")
        vol = features.get("volatility_20d")

        if regime:
            evidence.append(f"Market regime is classified as {regime.lower()}.")
        if isinstance(vol, (int, float)):
            evidence.append(f"20-day volatility is {vol:.2f}%.")

    return evidence


def explain_decision_ladder(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    bull = stock.get("bullbrain") or {}
    quality = decision.get("quality") or {}

    final_signal = (
        decision.get("final")
        or decision.get("finalSignal")
        or bull.get("signal")
        or "HOLD"
    )

    confidence = (
        decision.get("confidence")
        or bull.get("confidence")
    )

    reasons = decision.get("decisionReasons") or decision.get("reasons") or []

    def failed(code: str) -> bool:
        return code in reasons

    liquidity = quality.get("liquidity")
    regime = quality.get("regime")

    liquidity_failed = (
        failed("Liquidity=POOR")
        or liquidity in ("THIN", "POOR")
    )

    ladder = [
        {
            "gate": "Liquidity",
            "status": "failed" if liquidity_failed else "passed",
            "explanation": (
                "Trading volume is below the reliability threshold, so price moves should be trusted with caution."
                if liquidity_failed
                else "Trading volume is healthy enough to support the signal."
            ),
            "evidenceSummary": build_gate_evidence(stock, "Liquidity"),
            "metrics": build_gate_metrics(stock, "Liquidity"),
        },
        {
            "gate": "Market Regime",
            "status": "passed",
            "explanation": f"Market regime is {regime or 'NORMAL'}, which affects how aggressive the signal can be.",
            "evidenceSummary": build_gate_evidence(stock, "Market Regime"),
            "metrics": build_gate_metrics(stock, "Market Regime"),
        },
        {
            "gate": "Feature Consensus",
            "status": "failed" if failed("WeakFeatureConsensus") else "passed",
            "explanation": (
                "Trend, momentum, and volume indicators do not align strongly."
                if failed("WeakFeatureConsensus")
                else "Key indicators show enough agreement to support the model view."
            ),
            "evidenceSummary": build_gate_evidence(stock, "Feature Consensus"),
            "metrics": build_gate_metrics(stock, "Feature Consensus"),
        },
        {
            "gate": "Pattern Quality",
            "status": "failed" if failed("PatternQualityFailed") else "passed",
            "explanation": (
                "The current pattern has weak historical performance or insufficient quality."
                if failed("PatternQualityFailed")
                else "The current pattern has acceptable historical context."
            ),
            "evidenceSummary": build_gate_evidence(stock, "Pattern Quality"),
            "metrics": build_gate_metrics(stock, "Pattern Quality"),
        },
        {
            "gate": "Pattern Alignment",
            "status": "failed" if failed("SignalPatternConflict") else "passed",
            "explanation": (
                "Pattern bias conflicts with the model direction."
                if failed("SignalPatternConflict")
                else "Pattern bias does not strongly conflict with the model signal."
            ),
            "evidenceSummary": build_gate_evidence(stock, "Pattern Alignment"),
            "metrics": build_gate_metrics(stock, "Pattern Alignment"),
        },
        {
            "gate": "Expected Value",
            "status": "blocked" if failed("NegativeEV") else "passed",
            "explanation": (
                "Historical reward does not justify the current risk."
                if failed("NegativeEV")
                else "Historical reward-risk profile is acceptable."
            ),
            "evidenceSummary": build_gate_evidence(stock, "Expected Value"),
            "metrics": build_gate_metrics(stock, "Expected Value"),
        },
        {
            "gate": "Exhaustion / Fragility",
            "status": "failed" if failed("MomentumExhausted") else "passed",
            "explanation": (
                "The move appears stretched or unstable."
                if failed("MomentumExhausted")
                else "Price action is not showing a major exhaustion warning."
            ),
            "evidenceSummary": build_gate_evidence(stock, "Exhaustion / Fragility"),
            "metrics": build_gate_metrics(stock, "Exhaustion / Fragility"),
        },
    ]

    failed_gates = [
        g["gate"] for g in ladder
        if g["status"] in ("failed", "blocked")
    ]

    improvements = []

    if liquidity_failed:
        improvements.append("Stronger trading volume and better liquidity quality")
    if failed("WeakFeatureConsensus"):
        improvements.append("Stronger alignment between trend, momentum, and volume")
    if failed("SignalPatternConflict"):
        improvements.append("Pattern bias aligning more clearly with model direction")
    if failed("NegativeEV"):
        improvements.append("Improved historical reward relative to risk")
    if failed("MomentumExhausted"):
        improvements.append("Reduced extension risk or cleaner momentum reset")

    if failed_gates:
        why = f"{len(failed_gates)} quality check(s) need attention: {', '.join(failed_gates)}."
    else:
        why = "All quality checks passed."

    return {
        "symbol": stock.get("symbol"),
        "finalSignal": final_signal,
        "confidence": confidence,
        "confidenceLabel": (
            "High" if isinstance(confidence, (int, float)) and confidence >= 70
            else "Moderate" if isinstance(confidence, (int, float)) and confidence >= 55
            else "Low"
        ),
        "summary": {
            "headline": (
                f"{final_signal} — mixed signals"
                if final_signal == "HOLD"
                else f"{final_signal} signal"
            ),
            "why": why,
        },
        "decisionLadder": ladder,
        "whatWouldChange": improvements,
    }