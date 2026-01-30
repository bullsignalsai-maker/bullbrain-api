# =========================================================
# UI Stock Builder — Stock Detail Contract v1.0
# Source of truth: Firestore stock document
# NO external calls | NO hallucinated data
# =========================================================

from typing import Dict, Any, List


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------

def _sentences(lines: List[str], min_count: int = 2) -> str:
    """
    Join narrative lines into a paragraph.
    Ensures minimum sentence count without inventing facts.
    """
    lines = [l.strip() for l in lines if isinstance(l, str) and l.strip()]
    if not lines:
        return ""
    if len(lines) >= min_count:
        return " ".join(lines)
    # repeat last sentence for density (not invention)
    return " ".join(lines + lines[-1:])


def _safe(val, fallback=None):
    return val if val is not None else fallback


# ---------------------------------------------------------
# 1️⃣ SIGNAL BLOCK
# ---------------------------------------------------------

def build_signal_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}

    return {
        "signal": decision.get("final", "HOLD"),
        "confidence": decision.get("confidence"),
        "primary": narratives.get("summary"),
        "secondary": narratives.get("tradeIdea"),
    }


# ---------------------------------------------------------
# 2️⃣ PROBABILITY BLOCK
# ---------------------------------------------------------

def build_probability_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    narratives = stock.get("narratives") or {}
    probs = stock.get("probabilities") or {}

    up = probs.get("up")
    down = probs.get("down")

    explanation = narratives.get("probability")
    if isinstance(explanation, list):
        explanation = _sentences(explanation)

    return {
        "up": up,
        "down": down,
        "explanation": explanation,
    }


# ---------------------------------------------------------
# 3️⃣ PATTERN BLOCK
# ---------------------------------------------------------

def build_pattern_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    pattern = stock.get("pattern") or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    explanation = sections.get("pattern") or []
    explanation = _sentences(explanation, min_count=3)

    return {
        "name": pattern.get("pattern") or pattern.get("patternLabel"),
        "bias": pattern.get("bias"),
        "winRate5d": pattern.get("winRate5d"),
        "confidence": pattern.get("confidence"),
        "explanation": explanation,
    }


# ---------------------------------------------------------
# 4️⃣ TECHNICAL SNAPSHOT
# ---------------------------------------------------------

def build_technical_snapshot_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    technical = stock.get("technical") or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    def build_part(key: str, values: Dict[str, Any]) -> Dict[str, Any]:
        text = _sentences(sections.get(key) or [])
        return {
            "values": values,
            "explanation": text,
        }

    return {
        "trend": build_part("trend", technical.get("trend") or {}),
        "momentum": build_part("momentum", technical.get("momentum") or {}),
        "volatility": build_part("volatility", technical.get("volatility") or {}),
        "volume": build_part("volume", technical.get("volume") or {}),
    }


# ---------------------------------------------------------
# 5️⃣ FEATURES / INDICATORS
# ---------------------------------------------------------

def build_features_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    indicators = stock.get("indicator_states") or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    features = {}
    for key, data in indicators.items():
        explanation = sections.get(key) or []
        features[key] = {
            "value": data.get("value"),
            "state": data.get("state"),
            "explanation": _sentences(explanation),
        }

    return features


# ---------------------------------------------------------
# 6️⃣ OUTLOOK BLOCK
# ---------------------------------------------------------

def build_outlook_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    insights = stock.get("insights") or {}
    narratives = stock.get("narratives") or {}

    return {
        "shortTerm": insights.get("trendSummary"),
        "mediumTerm": narratives.get("summary"),
        "longTerm": insights.get("combinedTechnicalSummary"),
    }


# ---------------------------------------------------------
# 7️⃣ TRADE IDEA
# ---------------------------------------------------------

def build_trade_idea_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    narratives = stock.get("narratives") or {}
    decision = stock.get("decision") or {}

    return {
        "stance": decision.get("final", "HOLD"),
        "summary": narratives.get("tradeIdea"),
        "rationale": narratives.get("summary"),
    }


# ---------------------------------------------------------
# 8️⃣ RISKS & OPPORTUNITIES (NO NULLS)
# ---------------------------------------------------------

def build_risks_opportunities_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    risks = sections.get("risks") or []
    opps = sections.get("opportunities") or []

    risks = [_sentences([r]) for r in risks if isinstance(r, str)]
    opps = [_sentences([o]) for o in opps if isinstance(o, str)]

    return {
        "risks": risks,
        "opportunities": opps,
    }


# ---------------------------------------------------------
# 9️⃣ FINAL RECOMMENDATION
# ---------------------------------------------------------

def build_final_recommendation_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}

    return {
        "signal": decision.get("final"),
        "confidence": decision.get("confidence"),
        "text": narratives.get("summary"),
    }


# ---------------------------------------------------------
# 🔟 NEWS
# ---------------------------------------------------------

def build_news_block(stock: Dict[str, Any]) -> List[Dict[str, Any]]:
    news = stock.get("news") or []
    out = []

    for n in news:
        out.append({
            "headline": n.get("headline"),
            "summary": n.get("summary"),
            "url": n.get("url"),
            "source": n.get("source"),
            "datetime": n.get("datetime"),
            "image": n.get("image"),
        })

    return out


# ---------------------------------------------------------
# 🧠 ORCHESTRATOR — STOCK DETAIL v1.0
# ---------------------------------------------------------

def build_stockdetail_v1(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical Stock Detail builder
    Every field required by UI is produced here
    """

    return {
        "signal": build_signal_block(stock),
        "probability": build_probability_block(stock),
        "pattern": build_pattern_block(stock),
        "technicalSnapshot": build_technical_snapshot_block(stock),
        "features": build_features_block(stock),
        "outlook": build_outlook_block(stock),
        "tradeIdea": build_trade_idea_block(stock),
        "risksOpportunities": build_risks_opportunities_block(stock),
        "finalRecommendation": build_final_recommendation_block(stock),
        "news": build_news_block(stock),
        "computed_at": stock.get("computed_at"),
    }
