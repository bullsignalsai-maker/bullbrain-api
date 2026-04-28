# backend/astra_engine.py

import json
import re
from typing import Dict, Any

from backend.astra_intent_router import detect_astra_intent
from backend.astra_context_builder import build_astra_context

def build_astra_cards(context: Dict[str, Any]) -> list[Dict[str, Any]]:
    intent = (context.get("intent") or {}).get("intent")
    symbols = context.get("symbols") or []
    portfolio = context.get("portfolio") or {}

    cards = []

    if intent in ["stock_explain", "decision_explain", "technical_explain", "pattern_explain"]:
        if symbols:
            s = symbols[0]
            sig = s.get("aiSignal") or {}
            pat = s.get("pattern") or {}
            tech = s.get("technical") or {}

            cards.append({
                "type": "signal",
                "title": "AI Signal",
                "value": sig.get("signal") or "N/A",
                "subtitle": f"{sig.get('confidence')}% confidence" if sig.get("confidence") is not None else "Confidence unavailable",
            })

            cards.append({
                "type": "probability",
                "title": "Probability",
                "value": f"{round((sig.get('prob_up') or 0) * 100)}% Up",
                "subtitle": f"{round((sig.get('prob_down') or 0) * 100)}% downside probability",
            })

            cards.append({
                "type": "pattern",
                "title": "Pattern",
                "value": pat.get("name") or "N/A",
                "subtitle": f"5D win rate {round((pat.get('winRate5d') or 0) * 100)}%" if pat.get("winRate5d") is not None else "Win rate unavailable",
            })

    elif intent == "compare_symbols":
        for s in symbols[:3]:
            sig = s.get("aiSignal") or {}
            pat = s.get("pattern") or {}

            cards.append({
                "type": "compare",
                "title": s.get("symbol"),
                "value": sig.get("signal") or "N/A",
                "subtitle": f"{sig.get('confidence')}% conf • {pat.get('name') or 'No pattern'}",
            })

    else:
        top = portfolio.get("top_holding") or {}
        best = portfolio.get("best_position") or {}
        worst = portfolio.get("worst_position") or {}

        if top:
            cards.append({
                "type": "portfolio",
                "title": "Largest Holding",
                "value": top.get("symbol") or "N/A",
                "subtitle": f"{top.get('allocation_pct')}% allocation",
            })

        if best:
            cards.append({
                "type": "portfolio",
                "title": "Best Contributor",
                "value": best.get("symbol") or "N/A",
                "subtitle": f"${best.get('gain')} gain/loss",
            })

        if worst:
            cards.append({
                "type": "risk",
                "title": "Needs Attention",
                "value": worst.get("symbol") or "N/A",
                "subtitle": f"${worst.get('gain')} gain/loss",
            })

    return cards[:4]
def resolve_followup_symbols(req, available_symbols: list[str]) -> tuple[str, list[str]]:
    """
    Resolves follow-up language like:
    - Compare it with TSLA
    - What about NVDA?
    - Compare that with MSFT

    Returns:
      resolved_question, extra_symbols
    """
    question = (req.question or "").strip()
    if not question:
        return question, []

    q_upper = question.upper()
    chat_history = getattr(req, "chat_history", []) or []

    # 1) Symbols explicitly mentioned in current question
    current_symbols = []
    for sym in available_symbols:
        if re.search(rf"\b{re.escape(sym.upper())}\b", q_upper):
            current_symbols.append(sym.upper())

    # 2) Resolve last discussed symbol from stock_detail context first
    last_symbol = None
    if getattr(req, "contextType", None) == "stock_detail" and getattr(req, "symbol", None):
        last_symbol = req.symbol.upper()

    # 3) If not stock_detail, inspect recent chat history
    if not last_symbol:
        for msg in reversed(chat_history[-8:]):
            text = (msg.get("text") or "").upper()
            for sym in available_symbols:
                if re.search(rf"\b{re.escape(sym.upper())}\b", text):
                    last_symbol = sym.upper()
                    break
            if last_symbol:
                break

    # 4) If user uses pronouns and mentioned another symbol, include both
    pronoun_followup = any(
        phrase in q_upper
        for phrase in [
            " IT ",
            " THAT ",
            " THIS ",
            " THIS STOCK",
            "WITH IT",
            "COMPARE IT",
            "COMPARE THAT",
            "WHAT ABOUT",
        ]
    )

    extra_symbols = []

    if pronoun_followup and last_symbol:
        extra_symbols.append(last_symbol)

    for sym in current_symbols:
        if sym not in extra_symbols:
            extra_symbols.append(sym)

    # 5) Rewrite question for LLM clarity
    resolved_question = question
    if last_symbol:
        resolved_question = re.sub(
            r"\bit\b|\bthat\b|\bthis stock\b|\bthis\b",
            last_symbol,
            resolved_question,
            flags=re.IGNORECASE,
        )

    return resolved_question, extra_symbols
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
        
    )

def build_suggested_followups(context: Dict[str, Any]) -> list[str]:
    intent = (context.get("intent") or {}).get("intent")
    question = ((context.get("intent") or {}).get("question") or "").lower()
    symbols = context.get("symbols") or []
    portfolio = context.get("portfolio") or {}
    chat_history = context.get("chat_history") or []

    first_symbol = symbols[0].get("symbol") if symbols else None
    second_symbol = symbols[1].get("symbol") if len(symbols) > 1 else None
    top = (portfolio.get("top_holding") or {}).get("symbol")
    worst = (portfolio.get("worst_position") or {}).get("symbol")

    sym = first_symbol or top or "this stock"

    asked_text = " ".join(
        (m.get("text") or "").lower()
        for m in chat_history
        if m.get("role") == "user"
    )

    asked_text = f"{asked_text} {question}"

    def already_asked(candidate: str) -> bool:
        c = candidate.lower()

        checks = [
            ("rated", ["rated", "why"]),
            ("improve", ["improve", "better", "change"]),
            ("pattern", ["pattern"]),
            ("technical", ["technical", "rsi", "macd", "trend"]),
            ("risk", ["risk"]),
            ("compare", ["compare", " vs ", " versus "]),
            ("overweight", ["overweight", "underweight"]),
            ("attention", ["attention", "weakest", "worst"]),
            ("monitor", ["monitor", "watch"]),
        ]

        for key, words in checks:
            if key in c and any(w in asked_text for w in words):
                return True

        # exact-ish fallback
        return c in asked_text

    if intent == "compare_symbols":
        pool = [
            "Which one has stronger signal?",
            "Which one has higher risk?",
            "Which one has better pattern quality?",
            "Which one has stronger technicals?",
            "Which one looks more stable?",
        ]

    elif intent in ["stock_explain", "decision_explain"]:
        pool = [
            f"Why is {sym} rated this way?",
            f"What would improve {sym}?",
            f"What is the biggest risk for {sym}?",
            f"Explain {sym} pattern risk",
            f"Explain {sym} technicals",
            f"What should I monitor next for {sym}?",
        ]

    elif intent == "pattern_explain":
        pool = [
            f"How reliable is this pattern for {sym}?",
            f"What could invalidate this pattern?",
            f"Compare {sym} pattern with another stock",
            f"What is the downside risk for {sym}?",
        ]

    elif intent == "technical_explain":
        pool = [
            f"Is {sym} momentum strong or weak?",
            f"What does RSI say for {sym}?",
            f"What does volume confirm for {sym}?",
            f"What would improve {sym} technicals?",
        ]

    elif intent == "portfolio_risk":
        pool = [
            "Which holding is most risky?",
            "How can I reduce concentration?",
            "Which stock needs attention first?",
            "Which holding is overweight?",
        ]

    elif intent == "portfolio_suggestions":
        pool = [
            "What should I monitor next?",
            "Which holding is overweight?",
            "Which position looks weakest?",
            "Which holding has the best signal?",
        ]

    else:
        pool = [
            "What is my biggest risk?",
            f"Explain {top or sym}",
            f"Why is {worst or sym} underperforming?",
            "Which stock needs attention first?",
            "What should I monitor next?",
        ]

    fresh = [q for q in pool if not already_asked(q)]

    # Fallback if all were filtered
    if len(fresh) < 2:
        fallback = [
            f"Compare {sym} with TSLA" if sym != "TSLA" else "Compare TSLA with NVDA",
            f"What would change the signal for {sym}?",
            f"What is the next thing to watch for {sym}?",
            "Which stock looks stronger?",
        ]

        for q in fallback:
            if q not in fresh and not already_asked(q):
                fresh.append(q)

    return fresh[:3]
def build_astra_prompt(context: Dict[str, Any]) -> tuple[str, str]:
    intent = (context.get("intent") or {}).get("intent")
    question = (context.get("intent") or {}).get("question") or ""

    chat_history = context.get("chat_history") or []

    history_text = ""
    if chat_history:
        history_text = "Recent conversation:\n"
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if text:
                history_text += f"{role}: {text}\n"

    system_prompt = (
        "You are Astra, the AI intelligence engine inside BullSignalsAI. "
        "Answer ONLY the user's question. Do not provide a full portfolio report unless asked. "
        "Use only the provided JSON data. Do not invent numbers. "
        "If the context contains multiple symbols, compare them directly and do not say another symbol's data is unavailable."
        "Be concise, direct, and practical. "
        "Use simple language for retail investors. "
        "Do not use markdown headings, bullet lists, asterisks, or long explanations. "
        "Do not give financial advice. Use educational wording only. "
        "Maximum answer length: 4 short sentences. "
        "If the question asks for one thing, answer that one thing first."
    )

    user_prompt = (
        f"User question: {question}\n"
        f"Detected intent: {intent}\n\n"
        "Use this JSON context only:\n"
        f"{history_text}\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Answer rules:\n"
        "- Start with the direct answer.\n"
        "- Mention only the most relevant tickers and numbers.\n"
        "- Do not summarize every holding.\n"
        "- Do not repeat the same disclaimer every time; use one short closing sentence only.\n"
        "- Keep the answer within 2 to 4 short sentences.\n"
    )

    return system_prompt, user_prompt

def run_astra(req, astra_llm_answer_fn) -> Dict[str, Any]:
    if getattr(req, "contextType", None) == "stock_detail" and getattr(req, "symbol", None):
        available_symbols = [req.symbol.upper()]
    else:
        available_symbols = [p.symbol.upper() for p in (req.positions or [])]

    # ✅ Add any symbols mentioned in the question, even if not in portfolio
    question_upper = (req.question or "").upper()
    mentioned_symbols = re.findall(r"\b[A-Z]{1,5}\b", question_upper)

    for sym in mentioned_symbols:
        if sym not in available_symbols and sym not in ["WHY", "WHAT", "WITH", "THIS", "THAT", "HOLD", "BUY", "SELL"]:
            available_symbols.append(sym)

    # ✅ Resolve follow-up pronouns like "it", "that", "this stock"
    resolved_question, resolved_symbols = resolve_followup_symbols(req, available_symbols)

    for sym in resolved_symbols:
        if sym not in available_symbols:
            available_symbols.append(sym)

    intent_payload = detect_astra_intent(
        question=resolved_question,
        question_id=req.question_id,
        available_symbols=available_symbols,
    )

    if resolved_symbols:
        intent_payload["symbols"] = resolved_symbols

    intent_payload["question"] = resolved_question
    context = build_astra_context(req, intent_payload)

    # ✅ Add short session memory from frontend
    context["chat_history"] = getattr(req, "chat_history", []) or []

    fallback_answer = build_fast_astra_answer(context)
    

    system_prompt, user_prompt = build_astra_prompt(context)

    llm_answer = astra_llm_answer_fn(system_prompt, user_prompt)

    return {
        "answer": llm_answer or fallback_answer,
        "used_llm": llm_answer is not None,
        "intent": intent_payload,
        "cards": build_astra_cards(context),
        "contextSummary": {
            "symbols_used": [s.get("symbol") for s in context.get("symbols", [])],
            "portfolio_value": context.get("portfolio", {}).get("total_value"),
            "position_count": context.get("portfolio", {}).get("position_count"),
        },
        "suggestedFollowups": build_suggested_followups(context),
        "analysis": context,
    }