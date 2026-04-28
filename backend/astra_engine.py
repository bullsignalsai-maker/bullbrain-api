# backend/astra_engine.py

import json
from typing import Dict, Any

from backend.astra_intent_router import detect_astra_intent
from backend.astra_context_builder import build_astra_context


def build_fast_astra_answer(context: Dict[str, Any]) -> str:
    intent = context.get("intent", {}).get("intent")
    portfolio = context.get("portfolio") or {}
    symbols = context.get("symbols") or []

    if not symbols:
        return (
            "I could not find enough stock intelligence yet. Add holdings or refresh the portfolio, "
            "then I can analyze signals, risk, patterns, and technicals."
        )

    if intent == "portfolio_risk":
        top = portfolio.get("top_holding") or {}
        worst = portfolio.get("worst_position") or {}

        return (
            f"Your biggest concentration is {top.get('symbol', 'N/A')} at "
            f"{top.get('allocation_pct', 'N/A')}% of the portfolio. "
            f"The position needing the most attention is {worst.get('symbol', 'N/A')}, "
            f"with an unrealized gain/loss of ${worst.get('gain', 'N/A')}. "
            "Astra is using allocation, price action, AI signal confidence, pattern history, and technical indicators to judge risk. "
            "This is AI-driven insight for education, not personal financial advice."
        )

    if intent in ["stock_explain", "decision_explain", "technical_explain", "pattern_explain"]:
        s = symbols[0]
        sig = s.get("aiSignal") or {}
        pat = s.get("pattern") or {}
        tech = s.get("technical") or {}
        port = s.get("portfolio") or {}

        return (
            f"{s.get('symbol')} is currently rated {sig.get('signal')} with "
            f"{sig.get('confidence')}% confidence. "
            f"The model shows {sig.get('prob_up')} upside probability and {sig.get('prob_down')} downside probability. "
            f"The current pattern is {pat.get('name')} with a 5-day win rate of {pat.get('winRate5d')}. "
            f"From your portfolio view, this position is {port.get('allocation_pct')}% of holdings with "
            f"{port.get('gain_pct')}% gain/loss. "
            "This is AI-driven insight for education, not personal financial advice."
        )

    if intent == "compare_symbols":
        names = [s.get("symbol") for s in symbols]
        lines = []

        for s in symbols:
            sig = s.get("aiSignal") or {}
            port = s.get("portfolio") or {}
            pat = s.get("pattern") or {}

            lines.append(
                f"{s.get('symbol')}: {sig.get('signal')} at {sig.get('confidence')}% confidence, "
                f"{port.get('allocation_pct')}% allocation, {port.get('gain_pct')}% gain/loss, "
                f"pattern {pat.get('name')}."
            )

        return (
            f"Comparing {', '.join(names)}: "
            + " ".join(lines)
            + " Use this comparison to understand concentration, signal quality, and relative risk. "
            "This is AI-driven insight for education, not personal financial advice."
        )

    top = portfolio.get("top_holding") or {}
    best = portfolio.get("best_position") or {}
    worst = portfolio.get("worst_position") or {}

    return (
        f"Your portfolio value is about ${portfolio.get('total_value')} with total gain/loss of "
        f"${portfolio.get('total_gain')} and today's movement of ${portfolio.get('today_gain')}. "
        f"Largest holding is {top.get('symbol')} at {top.get('allocation_pct')}%. "
        f"Best contributor is {best.get('symbol')} and weakest contributor is {worst.get('symbol')}. "
        "Astra is using portfolio allocation, AI signals, probabilities, patterns, and technical indicators. "
        "This is AI-driven insight for education, not personal financial advice."
    )


def build_astra_prompt(context: Dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are Astra, the AI intelligence engine inside BullSignalsAI. "
        "Use only the provided JSON data. Do not invent numbers. "
        "Explain clearly in simple language. "
        "Do not give financial advice. Use educational wording. "
        "Be specific to the user's portfolio and symbols. "
        "Avoid generic templates. Keep answers concise."
    )

    user_prompt = (
        "Answer the user's question using this Astra context only.\n\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Rules: mention real tickers, values, signal, probability, pattern, allocation, and risk when available. "
        "If data is missing, say it is unavailable. "
        "End with a short educational disclaimer."
    )

    return system_prompt, user_prompt


def run_astra(req, astra_llm_answer_fn) -> Dict[str, Any]:
    available_symbols = [p.symbol.upper() for p in (req.positions or [])]

    intent_payload = detect_astra_intent(
        question=req.question,
        question_id=req.question_id,
        available_symbols=available_symbols,
    )

    context = build_astra_context(req, intent_payload)

    fallback_answer = build_fast_astra_answer(context)

    system_prompt, user_prompt = build_astra_prompt(context)

    llm_answer = astra_llm_answer_fn(system_prompt, user_prompt)

    return {
        "answer": llm_answer or fallback_answer,
        "used_llm": llm_answer is not None,
        "intent": intent_payload,
        "contextSummary": {
            "symbols_used": [s.get("symbol") for s in context.get("symbols", [])],
            "portfolio_value": context.get("portfolio", {}).get("total_value"),
            "position_count": context.get("portfolio", {}).get("position_count"),
        },
        "analysis": context,
    }