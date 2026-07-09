import re
import datetime
import pytz

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


_ET = pytz.timezone("America/New_York")
_SPREADSHEET_REASON_MAX_AGE_HOURS = 4


def _spreadsheet_reason_is_fresh(spreadsheet_meta: Dict[str, Any]) -> bool:
    """
    True if spreadsheet_meta's Grok-generated reason was produced within the
    last _SPREADSHEET_REASON_MAX_AGE_HOURS. generatedAt is a naive
    'YYYY-MM-DD HH:MM:SS' string in ET (matches the sessionType labels
    PREMARKET/MIDDAY/END_OF_DAY) — not the ISO-8601/Z format used elsewhere
    in this codebase, so it needs its own parser.
    """
    generated_at = spreadsheet_meta.get("generatedAt")
    if not generated_at:
        return False

    try:
        naive = datetime.datetime.strptime(generated_at, "%Y-%m-%d %H:%M:%S")
        generated = _ET.localize(naive)
    except Exception:
        return False

    now = datetime.datetime.now(_ET)
    age_hours = (now - generated).total_seconds() / 3600.0

    if age_hours < 0:
        # Clock skew or bad data — don't trust a timestamp claiming to be
        # from the future.
        return False

    return age_hours <= _SPREADSHEET_REASON_MAX_AGE_HOURS


_UP_WORDS = [
    "rally", "rallying", "surge", "surging", "breakout", "upgrade", "upgraded",
    "outperform", "beat", "beats", "record high", "accelerating", "accelerates",
    "strength", "strong", "bullish", "rebound", "rebounding",
]

_DOWN_WORDS = [
    "sell-off", "selloff", "plunge", "plunging", "downgrade", "downgraded",
    "miss", "misses", "weakness", "bearish", "pullback", "pulling back",
    "collapse", "slump", "slumping", "under pressure",
]

_UP_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _UP_WORDS) + r")\b", re.IGNORECASE
)
_DOWN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _DOWN_WORDS) + r")\b", re.IGNORECASE
)

_REASON_DEAD_ZONE_PCT = 0.4


def _reason_direction_conflicts(reason: str, change_pct: float | None) -> bool:
    """
    True only when `reason` contains an unambiguous directional word AND that
    direction contradicts today's actual price move. Absence of a directional
    word is NOT a conflict — this only rejects on a detected contradiction,
    not on lack of confirmation, so terse Grok phrasing outside this word
    list still passes through untouched.
    """
    if not reason or change_pct is None:
        return False

    if abs(change_pct) < _REASON_DEAD_ZONE_PCT:
        return False  # today's move is noise-level; nothing to contradict

    has_up = bool(_UP_PATTERN.search(reason))
    has_down = bool(_DOWN_PATTERN.search(reason))

    if has_up and has_down:
        return False  # mixed/ambiguous language — don't reject on ambiguity
    if has_up and change_pct < 0:
        return True
    if has_down and change_pct > 0:
        return True

    return False


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

    spreadsheet_reason = spreadsheet_meta.get("reason")
    fallback_reason = (
        market_awareness.get("displayLine")
        or market_awareness.get("oneLiner")
        or market_awareness.get("summary")
        or ""
    )

    if (
        spreadsheet_reason
        and _spreadsheet_reason_is_fresh(spreadsheet_meta)
        and not _reason_direction_conflicts(spreadsheet_reason, change_pct)
    ):
        reason = spreadsheet_reason
    else:
        reason = fallback_reason

    catalysts = spreadsheet_meta.get("primaryCatalysts") or ""
    candidate_type = spreadsheet_meta.get("candidateType") or ""
    risk_level = spreadsheet_meta.get("riskLevel") or ""
    market_sentiment = spreadsheet_meta.get("marketSentiment") or ""
    dominant_theme = spreadsheet_meta.get("dominantTheme") or ""

    trend_label = trend.get("label") or ""
    rsi_label = momentum.get("rsiLabel") or ""
    volume_label = volume.get("label") or ""

    score = 50
    factors: Dict[str, float] = {}

    raw = str(base_signal).upper()
    if raw in ("BUY", "BULLISH"):
        factors["baseSignal"] = 12
    elif raw in ("SELL", "BEARISH"):
        factors["baseSignal"] = -12
    else:
        factors["baseSignal"] = 0
    score += factors["baseSignal"]

    factors["confidence"] = max(-8, min(10, (base_conf - 50) / 5))
    score += factors["confidence"]

    if change_pct >= 6:
        factors["priceMove"] = 18
    elif change_pct >= 3:
        factors["priceMove"] = 13
    elif change_pct >= 1.5:
        factors["priceMove"] = 7
    elif change_pct <= -6:
        factors["priceMove"] = -18
    elif change_pct <= -3:
        factors["priceMove"] = -13
    elif change_pct <= -1.5:
        factors["priceMove"] = -7
    else:
        factors["priceMove"] = 0
    score += factors["priceMove"]

    factors["catalyst"] = 6 if reason else 0
    score += factors["catalyst"]

    factors["catalystDetail"] = 5 if catalysts else 0
    score += factors["catalystDetail"]

    ct = str(candidate_type).lower()
    if ct in ("institutional_setup", "earnings_reaction", "analyst_action", "unusual_attention"):
        factors["candidateType"] = 7
    elif ct == "speculative_momentum":
        factors["candidateType"] = 3
    else:
        factors["candidateType"] = 0
    score += factors["candidateType"]

    tl = trend_label.lower()
    if "strong uptrend" in tl:
        factors["trend"] = 8
    elif "uptrend" in tl:
        factors["trend"] = 5
    elif "strong downtrend" in tl:
        factors["trend"] = -8
    elif "downtrend" in tl:
        factors["trend"] = -5
    else:
        factors["trend"] = 0
    score += factors["trend"]

    factors["volume"] = 4 if "high" in volume_label.lower() else 0
    score += factors["volume"]

    risk = str(risk_level).lower()
    if risk == "high":
        factors["risk"] = -6
    elif risk == "low":
        factors["risk"] = 3
    else:
        factors["risk"] = 0
    score += factors["risk"]

    raw_total = int(round(score))
    score = int(max(0, min(100, round(score))))
    clamped_by = score - raw_total

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
        "scoreBreakdown": {
            "base": 50,
            "factors": factors,
            "rawTotal": raw_total,
            "clampedBy": clamped_by,
        },
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