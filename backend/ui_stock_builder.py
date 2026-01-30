# =========================================================
# UI Stock Builder — Stock Detail Contract v1.0 (FINAL)
# Source of truth: Firestore stock document
# NO external calls | NO hallucinated data
# =========================================================

from typing import Dict, Any, List


# ---------------------------------------------------------
# 🔧 Helpers (NORMALIZATION — CRITICAL)
# ---------------------------------------------------------

def _sentences(lines: List[str], min_count: int = 2) -> str:
    lines = [l.strip() for l in lines if isinstance(l, str) and l.strip()]
    if not lines:
        return ""
    if len(lines) >= min_count:
        return " ".join(lines)
    return " ".join(lines * min_count)


def _get_probabilities(stock: Dict[str, Any]):
    probs = stock.get("probabilities")
    if isinstance(probs, dict):
        up, down = probs.get("up"), probs.get("down")
        if isinstance(up, (int, float)) and isinstance(down, (int, float)):
            return up, down

    raw = (stock.get("bullbrain") or {}).get("raw") or {}
    up, down = raw.get("prob_up"), raw.get("prob_down")
    if isinstance(up, (int, float)) and isinstance(down, (int, float)):
        return up, down

    return None, None


def _get_confidence(stock: Dict[str, Any]):
    return (
        (stock.get("decision") or {}).get("confidence")
        or (stock.get("bullbrain") or {}).get("confidence")
    )


# ---------------------------------------------------------
# 1️⃣ SIGNAL BLOCK (AUTHORITATIVE)
# ---------------------------------------------------------

def build_signal_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}

    signal = decision.get("final") or "HOLD"
    confidence = _get_confidence(stock)

    if isinstance(confidence, (int, float)):
        tier = "High" if confidence >= 75 else "Moderate" if confidence >= 60 else "Low"
    else:
        tier = "Low"

    up, down = _get_probabilities(stock)
    if isinstance(up, float) and isinstance(down, float):
        bias = "Bullish" if up > down else "Bearish" if down > up else "Neutral"
    else:
        bias = "Neutral"

    expl = []
    if narratives.get("summary"):
        expl.append(narratives["summary"])
    else:
        expl.append(
            "Signals are mixed, with no strong directional conviction at current levels."
        )

    expl.append(
        f"Signal confidence is {tier.lower()}, indicating increased uncertainty in near-term outcomes."
    )

    return {
        "value": signal,
        "confidence": confidence,
        "confidenceTier": tier,
        "bias": bias,
        "explanation": _sentences(expl, 2),
    }


# ---------------------------------------------------------
# 2️⃣ PROBABILITY BLOCK (FIXED)
# ---------------------------------------------------------

def build_probability_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    narratives = stock.get("narratives") or {}
    up, down = _get_probabilities(stock)

    if not isinstance(up, float) or not isinstance(down, float):
        return {
            "bias": "Neutral",
            "explanation": "Probability data is currently unavailable for this symbol."
        }

    diff = abs(up - down)
    bias = "Neutral" if diff < 0.05 else "Bullish" if up > down else "Bearish"

    expl = []
    if narratives.get("probability"):
        expl.append(narratives["probability"])

    expl.append(
        f"Upside probability is approximately {up*100:.0f}%, while downside probability is around {down*100:.0f}%, "
        f"indicating a {bias.lower()} bias."
    )

    return {
        "up": round(up, 4),
        "down": round(down, 4),
        "bias": bias,
        "strengthPct": round(diff * 100, 1),
        "explanation": _sentences(expl, 2),
    }


# ---------------------------------------------------------
# 3️⃣ PATTERN BLOCK (STANDALONE EXPLANATION)
# ---------------------------------------------------------

def build_pattern_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}
    narratives = stock.get("narratives") or {}

    name = pattern.get("pattern") or pattern.get("patternLabel")
    bias = pattern.get("bias")
    win_rate = pattern.get("winRate5d")

    days5 = (history.get("forwardReturns") or {}).get("days5") or {}
    best, worst = days5.get("best"), days5.get("worst")

    expl = []

    if name:
        expl.append(
            f"The {name.replace('_', ' ').title()} pattern has recently emerged, reflecting short-term price structure."
        )

    if isinstance(win_rate, (int, float)):
        expl.append(
            f"Historically, this pattern has been favorable roughly {win_rate*100:.0f}% of the time over five days."
        )

    if isinstance(best, (int, float)) and isinstance(worst, (int, float)):
        expl.append(
            f"Past outcomes show gains up to {best:.1f}% and drawdowns near {worst:.1f}%, highlighting variability."
        )

    if not expl:
        expl.append(
            "Recent price behavior reflects a short-term pattern, though reliability is mixed."
        )

    return {
        "name": name,
        "bias": bias,
        "winRate5d": win_rate,
        "explanation": _sentences(expl, 3),
    }


# ---------------------------------------------------------
# 4️⃣ TECHNICAL SNAPSHOT (SUMMARY, NOT DUMP)
# ---------------------------------------------------------

def build_technical_snapshot_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    tech = stock.get("technical") or {}
    expl = []

    trend = (tech.get("trend") or {}).get("label")
    rsi = (tech.get("momentum") or {}).get("rsi14")
    vol = (tech.get("volatility") or {}).get("regime")

    if trend:
        expl.append(f"Trend structure is currently classified as {trend.lower()}.")

    if isinstance(rsi, (int, float)):
        expl.append(
            "Momentum is neutral."
            if 40 <= rsi <= 60
            else "Momentum is stretched and may limit follow-through."
        )

    if vol:
        expl.append(f"Volatility regime is {vol.lower()}, influencing risk dynamics.")

    if not expl:
        expl.append(
            "Technical conditions are mixed, without a dominant directional driver."
        )

    return {
        "summary": _sentences(expl, 2)
    }


# ---------------------------------------------------------
# 5️⃣ FEATURE INSIGHT (REPLACES FEATURE DUMP)
# ---------------------------------------------------------

def build_feature_insight_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    f = stock.get("features_meta") or {}
    expl = []

    rsi = f.get("rsi14")
    ret10 = f.get("return_10d")
    price = f.get("adj_close")
    sma20 = f.get("sma20")

    if isinstance(rsi, (int, float)):
        expl.append(
            "Momentum remains subdued."
            if rsi < 50 else
            "Momentum is balanced without extreme pressure."
        )

    if isinstance(ret10, (int, float)) and ret10 < 0:
        expl.append("Recent returns reflect sustained selling pressure.")

    if isinstance(price, (int, float)) and isinstance(sma20, (int, float)):
        expl.append(
            "Price is trading below its short-term average."
            if price < sma20 else
            "Price remains above its short-term average."
        )

    if not expl:
        expl.append(
            "Underlying feature signals do not currently show a strong directional bias."
        )

    return {
        "summary": _sentences(expl, 2)
    }


# ---------------------------------------------------------
# 6️⃣ TRADE IDEA
# ---------------------------------------------------------

def build_trade_idea_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}
    stance = decision.get("final") or "HOLD"
    confidence = _get_confidence(stock)

    expl = []
    if narratives.get("tradeIdea"):
        expl.append(narratives["tradeIdea"])
    else:
        expl.append(
            "Current conditions do not present a compelling risk–reward setup."
        )

    return {
        "stance": stance,
        "confidence": confidence,
        "explanation": _sentences(expl, 2),
    }


# ---------------------------------------------------------
# 7️⃣ FINAL RECOMMENDATION
# ---------------------------------------------------------

def build_final_recommendation_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}
    signal = decision.get("final") or "HOLD"
    confidence = _get_confidence(stock)

    expl = []
    if narratives.get("summary"):
        expl.append(narratives["summary"])
    else:
        expl.append(
            f"The model maintains a {signal.lower()} stance based on current conditions."
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "text": _sentences(expl, 2),
    }


# ---------------------------------------------------------
# 8️⃣ NEWS
# ---------------------------------------------------------

def build_news_block(stock: Dict[str, Any]) -> List[Dict[str, Any]]:
    news = stock.get("news") or []
    return [
        {
            "headline": n.get("headline"),
            "summary": n.get("summary"),
            "url": n.get("url"),
            "source": n.get("source"),
            "datetime": n.get("datetime"),
            "image": n.get("image"),
        }
        for n in news
        if isinstance(n, dict)
    ]


# ---------------------------------------------------------
# 🧠 ORCHESTRATOR — STOCK DETAIL v1.0 (FINAL)
# ---------------------------------------------------------

def build_stockdetail_v1(stock: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "signal": build_signal_block(stock),
        "probability": build_probability_block(stock),
        "pattern": build_pattern_block(stock),
        "technicalSnapshot": build_technical_snapshot_block(stock),
        "featureInsight": build_feature_insight_block(stock),
        "tradeIdea": build_trade_idea_block(stock),
        "finalRecommendation": build_final_recommendation_block(stock),
        "news": build_news_block(stock),
        "computed_at": stock.get("computed_at"),
    }
