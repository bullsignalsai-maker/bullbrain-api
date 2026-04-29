# backend/astra_intent_router.py

import re
from typing import Dict, Any, List


KNOWN_INTENTS = {
    "overview": "portfolio_overview",
    "risk_exposure": "portfolio_risk",
    "overweight": "portfolio_allocation",
    "worst": "portfolio_attention",
    "ai_suggestions": "portfolio_suggestions",
    "explain_single": "stock_explain",
    "explain_two": "compare_symbols",
    "compare_two": "compare_symbols",
    "top3_explain": "portfolio_top_holdings",
    "compare_top3": "compare_symbols",
    "leader": "portfolio_leader",
}


def extract_symbols(text: str, available_symbols: List[str]) -> List[str]:
    text = (text or "").upper()
    found = []

    for sym in available_symbols:
        if re.search(rf"\b{re.escape(sym.upper())}\b", text):
            found.append(sym.upper())

    return found


def detect_astra_intent(
    question: str | None,
    question_id: str | None,
    available_symbols: List[str],
) -> Dict[str, Any]:
    q = (question or "").strip()
    q_lower = q.lower()

    if question_id and question_id in KNOWN_INTENTS:
        intent = KNOWN_INTENTS[question_id]
    elif any(w in q_lower for w in ["compare", "vs", "versus"]):
        intent = "compare_symbols"
    elif any(w in q_lower for w in ["risk", "exposure", "danger"]):
        intent = "portfolio_risk"
    elif any(w in q_lower for w in ["worst", "attention", "problem", "drag"]):
        intent = "portfolio_attention"
    elif any(w in q_lower for w in ["rebalance", "suggest", "optimize"]):
        intent = "portfolio_suggestions"
    elif any(w in q_lower for w in ["why", "signal", "hold", "buy", "sell"]):
        intent = "decision_explain"
    elif any(w in q_lower for w in ["pattern", "chart pattern"]):
        intent = "pattern_explain"
    elif any(w in q_lower for w in ["technical", "rsi", "macd", "volume", "trend"]):
        intent = "technical_explain"
    elif extract_symbols(q, available_symbols):
        intent = "stock_explain"
    elif any(
        w in q_lower
        for w in [
            "market",
            "market pulse",
            "spy",
            "qqq",
            "nasdaq",
            "crypto",
            "bitcoin",
            "btc",
            "eth",
            "commodities",
            "gold",
            "oil",
            "silver",
            "fear",
            "greed",
            "risk",
            "news",
            "movers",
            "gainers",
            "losers",
        ]
    ):
        intent = "market_pulse"    
    else:
        intent = "portfolio_overview"

    symbols = extract_symbols(q, available_symbols)

    return {
        "intent": intent,
        "symbols": symbols,
        "question": q,
        "question_id": question_id,
    }