from typing import Dict, Any, List


def _num(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _clean(v):
    if not isinstance(v, str):
        return ""
    return " ".join(v.replace(",", " • ").split()).strip()


def _score_label(score: int, change_pct: float, risk_level: str):
    risk = str(risk_level or "").lower()

    if change_pct >= 6 and risk == "high":
        return "HIGH_RISK_MOMENTUM", "Momentum Surge", "positive"
    if score >= 82:
        return "STRONG_BULLISH", "Strong Bullish", "positive"
    if score >= 68:
        return "BULLISH_WATCH", "Bullish Watch", "positive"
    if score >= 58:
        return "MOMENTUM_WATCH", "Momentum Building", "positive"
    if score <= 35:
        return "BEARISH_WATCH", "Bearish Watch", "negative"
    if score <= 45:
        return "CAUTION", "Caution", "warning"

    return "HOLD", "Neutral", "neutral"


def build_display_intelligence(
    symbol: str,
    stock: Dict[str, Any],
    spreadsheet_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    symbol = symbol.upper()
    spreadsheet_meta = spreadsheet_meta or {}

    quote = stock.get("quote") or {}
    bull = stock.get("bullbrain") or {}
    decision = stock.get("decision") or {}
    technical = stock.get("technical") or {}
    market_awareness = stock.get("marketAwareness") or {}

    trend = technical.get("trend") or {}
    momentum = technical.get("momentum") or {}
    volume = technical.get("volume") or {}

    base_signal = (
        decision.get("finalSignal")
        or decision.get("final")
        or bull.get("signal")
        or "HOLD"
    )

    base_conf = _num(
        decision.get("confidence")
        if decision.get("confidence") is not None
        else bull.get("confidence"),
        50,
    )

    change_pct = _num(quote.get("changePct"), 0)
    price = _num(quote.get("price"), None)

    reason = (
        spreadsheet_meta.get("reason")
        or market_awareness.get("displayLine")
        or market_awareness.get("oneLiner")
        or market_awareness.get("summary")
        or ""
    )

    catalysts = spreadsheet_meta.get("primaryCatalysts") or ""
    candidate_type = spreadsheet_meta.get("candidateType") or ""
    risk_level = spreadsheet_meta.get("riskLevel") or ""
    market_sentiment = spreadsheet_meta.get("marketSentiment") or ""
    dominant_theme = spreadsheet_meta.get("dominantTheme") or ""

    trend_label = trend.get("label") or ""
    rsi_label = momentum.get("rsiLabel") or ""
    volume_label = volume.get("label") or ""

    score = 50

    raw = str(base_signal).upper()
    if raw in ("BUY", "BULLISH"):
        score += 12
    elif raw in ("SELL", "BEARISH"):
        score -= 12

    score += max(-8, min(10, (base_conf - 50) / 5))

    if change_pct >= 6:
        score += 18
    elif change_pct >= 3:
        score += 13
    elif change_pct >= 1.5:
        score += 7
    elif change_pct <= -6:
        score -= 18
    elif change_pct <= -3:
        score -= 13
    elif change_pct <= -1.5:
        score -= 7

    if reason:
        score += 6
    if catalysts:
        score += 5

    ct = str(candidate_type).lower()
    if ct in ("institutional_setup", "earnings_reaction", "analyst_action", "unusual_attention"):
        score += 7
    elif ct == "speculative_momentum":
        score += 3

    tl = trend_label.lower()
    if "strong uptrend" in tl:
        score += 8
    elif "uptrend" in tl:
        score += 5
    elif "strong downtrend" in tl:
        score -= 8
    elif "downtrend" in tl:
        score -= 5

    if "high" in volume_label.lower():
        score += 4

    risk = str(risk_level).lower()
    if risk == "high":
        score -= 6
    elif risk == "low":
        score += 3

    score = int(max(0, min(100, round(score))))
    signal, label, tone = _score_label(score, change_pct, risk_level)

    why_now: List[str] = []

    if reason:
        why_now.append(_clean(reason))
    if catalysts:
        why_now.append(f"Catalyst: {_clean(catalysts)}")
    if dominant_theme:
        why_now.append(f"Market theme: {_clean(dominant_theme)}")
    if trend_label:
        why_now.append(f"Trend context: {trend_label}")
    if volume_label:
        why_now.append(f"Participation: {volume_label}")

    if not why_now:
        why_now.append("Current rating blends BullBrain, quote movement, trend, and market context.")

    risk_notes: List[str] = []

    if risk == "high":
        risk_notes.append("Risk is elevated, so price movement may stay volatile.")
    if change_pct >= 6:
        risk_notes.append("Move is extended; pullback risk is higher.")
    if "overbought" in rsi_label.lower():
        risk_notes.append("RSI context suggests momentum may be stretched.")

    if not risk_notes:
        risk_notes.append("Watch for continued confirmation from price, volume, and trend.")

    headline = why_now[0]

    return {
        "signal": signal,
        "displaySignal": signal,
        "label": label,
        "tone": tone,
        "score": score,
        "baseSignal": base_signal,
        "baseConfidence": base_conf,
        "headline": headline,
        "summary": f"{label}: base BullBrain is {base_signal}, while current market context shows {change_pct:.2f}% movement.",
        "whyNow": why_now[:4],
        "riskNotes": risk_notes[:3],
        "context": {
            "symbol": symbol,
            "price": price,
            "changePct": change_pct,
            "candidateType": candidate_type,
            "riskLevel": risk_level,
            "marketSentiment": market_sentiment,
            "dominantTheme": dominant_theme,
            "reason": reason,
            "primaryCatalysts": catalysts,
            "trend": trend_label,
            "volume": volume_label,
        },
        "sourceMix": {
            "bullbrain": True,
            "quoteMomentum": quote.get("changePct") is not None,
            "catalyst": bool(reason or catalysts),
            "technical": bool(technical),
            "marketTheme": bool(dominant_theme or market_sentiment),
        },
    }