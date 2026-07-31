# backend/astra_engine.py

import json
import re
from typing import Dict, Any

from backend.astra_intent_router import detect_astra_intent
from backend.astra_context_builder import build_astra_context
from symbols_clean import REAL_TICKERS

# A candidate word from the user's question is only treated as a stock
# symbol if it's a genuine, known ticker -- not merely "looks like one"
# (1-5 uppercase letters). The previous approach was a denylist of common
# short words ("WHY", "IS", ...), which is unbounded: any short English
# word not yet added ("DID", "YOU", "PICK") slipped through and got
# treated as a real symbol, producing empty/garbage context blocks
# alongside the real one. REAL_TICKERS is the same source of truth
# alpha_watch_logic.py/market_cron.py already use.
_REAL_TICKER_SET = frozenset(REAL_TICKERS)

# Real tickers that are ALSO common short English words -- confirmed to
# cause real, live problems twice: "IT" hijacked "why did you pick PANW"-
# style questions, "NOW" hijacked a ranking question via "right now" (see
# bullbrain_clara_astra_investigation memory). For exactly these words,
# extraction below additionally requires the user to have actually typed
# them in all-caps -- everywhere else, REAL_TICKERS membership alone is
# still sufficient (typing "panw" lowercase must keep working).
_AMBIGUOUS_TICKERS = frozenset({"IT", "ALL", "LOW", "NOW", "ON", "ARE", "GO", "SO"})

# displayIntelligence (System B) buckets, collapsed into Clara's three
# conversational phrases. Raw BUY/SELL/HOLD are also recognized so the
# fallback path (a symbol without displayIntelligence yet) still translates.
_BULLISH_SIGNALS = {"STRONG_BULLISH", "BULLISH_WATCH", "MOMENTUM_WATCH", "BUY"}
_NEUTRAL_SIGNALS = {"HOLD", "CAUTION"}
_BEARISH_SIGNALS = {"BEARISH_WATCH", "HIGH_RISK_MOMENTUM", "SELL"}

def clara_signal_label(signal: str | None) -> str:
    signal = (signal or "").upper()

    if signal in _BULLISH_SIGNALS:
        return "Bullish Setup"
    if signal in _BEARISH_SIGNALS:
        return "Risk Alert"
    if signal in _NEUTRAL_SIGNALS:
        return "Neutral Setup"

    return "Market Setup"

def sanitize_clara_answer(text: str | None) -> str:
    if not text:
        return ""

    out = str(text)

    replacements = {
        r"\bBUY\b": "Bullish Setup",
        r"\bSELL\b": "Risk Alert",
        r"\bHOLD\b": "Neutral Setup",
        r"\bbuy\b": "bullish setup",
        r"\bsell\b": "risk alert",
        r"\bhold\b": "neutral setup",
        # Defense-in-depth: scrub raw displayIntelligence enum tokens in case
        # the LLM echoes one verbatim from the JSON context it was given.
        r"\bSTRONG_BULLISH\b": "Bullish Setup",
        r"\bBULLISH_WATCH\b": "Bullish Setup",
        r"\bMOMENTUM_WATCH\b": "Bullish Setup",
        r"\bCAUTION\b": "Neutral Setup",
        r"\bBEARISH_WATCH\b": "Risk Alert",
        r"\bHIGH_RISK_MOMENTUM\b": "Risk Alert",
    }

    for raw, clean in replacements.items():
        out = re.sub(raw, clean, out)

    out = out.replace("rated Neutral Setup", "showing a Neutral Setup")
    out = out.replace("rated Bullish Setup", "showing a Bullish Setup")
    out = out.replace("rated Risk Alert", "showing a Risk Alert")

    return out

def build_astra_cards(context: Dict[str, Any]) -> list[Dict[str, Any]]:
    intent = (context.get("intent") or {}).get("intent")
    symbols = context.get("symbols") or []
    portfolio = context.get("portfolio") or {}

    cards = []

    if context.get("contextType") == "momentum_movers":
        momentum = context.get("momentum") or {}
        selected = momentum.get("selectedMover") or {}
        pulse = momentum.get("pulse") or {}

        cards.append({
            "type": "mover",
            "title": "Selected Mover",
            "value": selected.get("symbol") or "N/A",
            "subtitle": selected.get("reason") or selected.get("momentumLabel") or "Momentum context",
        })

        cards.append({
            "type": "score",
            "title": "Momentum Score",
            "value": str(round(selected.get("momentumScore") or selected.get("avgAlphaScore") or 0)),
            "subtitle": f"{selected.get('appearances') or selected.get('dailyMoverAppearances') or 0} sessions observed",
        })

        cards.append({
            "type": "market",
            "title": "Market Bias",
            "value": pulse.get("marketBias") or "N/A",
            "subtitle": pulse.get("topTheme") or "Theme unavailable",
        })

        return cards[:4]
        
    if context.get("contextType") == "market" or intent == "market_pulse":
        market = context.get("market") or {}
        overview = market.get("marketOverview") or {}

        fg = overview.get("fearGreed") or {}
        us = market.get("usMarket") or []
        crypto = market.get("crypto") or []
        commodities = market.get("commodities") or []

        cards.append({
            "type": "market",
            "title": "Market Mood",
            "value": overview.get("marketMood") or "N/A",
            "subtitle": overview.get("marketStatus") or "Market status unavailable",
        })

        cards.append({
            "type": "risk",
            "title": "Fear & Greed",
            "value": str(fg.get("value", "N/A")),
            "subtitle": fg.get("label") or "Sentiment unavailable",
        })

        if us:
            cards.append({
                "type": "market",
                "title": "US Market",
                "value": "SPY / QQQ",
                "subtitle": "Based on broad-market ETF movement",
            })

        if crypto or commodities:
            cards.append({
                "type": "market",
                "title": "Cross-Asset Pulse",
                "value": "Crypto + ETFs",
                "subtitle": "Checks risk appetite across assets",
            })

        return cards[:4]        

    if intent in ["stock_explain", "decision_explain", "technical_explain", "pattern_explain"]:
        if symbols:
            s = symbols[0]
            sig = s.get("aiSignal") or {}
            pat = s.get("pattern") or {}
            tech = s.get("technical") or {}

            cards.append({
                "type": "signal",
                "title": "Market View",
                "value": clara_signal_label(sig.get("signal")),
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
                "value": clara_signal_label(sig.get("signal")),
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
    if context.get("contextType") == "accuracy_track_record":
        report = context.get("accuracyReport") or {}
        summary = report.get("summary")

        if not summary:
            return (
                "I don't have enough resolved picks yet to report real accuracy numbers. "
                "Check back once more tracked picks have reached their outcome window."
            )

        horizon_days = str(summary.get("horizon") or "").rstrip("d") or "an unknown number of"
        date_range = summary.get("pick_date_range") or {}

        return (
            f"Based on {summary.get('n')} tracked picks ({horizon_days}-day outcomes, "
            f"{date_range.get('min')} to {date_range.get('max')}), "
            f"{summary.get('pct_positive')}% were positive with an average return of "
            f"{summary.get('mean_return_pct')}%. This reflects real, checked outcomes, not a forecast."
        )

    if context.get("contextType") == "alphaclara_picks_overview":
        accuracy_report = context.get("accuracyReport") or {}
        accuracy_summary = accuracy_report.get("summary")
        picks_overview = context.get("picksOverview") or {}
        counts = picks_overview.get("counts") or {}
        rankings = picks_overview.get("rankings") or {}
        question = ((context.get("intent") or {}).get("question") or "").lower()

        # 1) Ranking-question phrasing, checked BEFORE trusting an extracted
        # symbol -- confirmed on real data that "whats our best performer
        # right now" extracts "NOW" (a real ticker, ServiceNow) purely from
        # the word "now", which would otherwise hijack this into a wrong,
        # single-symbol answer instead of the intended ranking answer. Same
        # residual ticker/word collision class already disclosed for the
        # symbol-extraction fix -- worked around here by priority ordering
        # rather than re-solved at the source.
        ranking_words = ["best", "worst", "top", "most", "biggest", "performer", "moved the most"]
        is_ranking_question = rankings and any(w in question for w in ranking_words)

        if is_ranking_question:
            best = rankings.get("bestPerformer")
            worst = rankings.get("worstPerformer")
            most_picked = rankings.get("mostPicked")

            parts = []
            if best:
                parts.append(f"the best performer right now is {best.get('symbol')} at {best.get('returnPct')}%")
            if worst:
                parts.append(f"the worst is {worst.get('symbol')} at {worst.get('returnPct')}%")
            if most_picked:
                parts.append(f"the most-picked symbol is {most_picked.get('symbol')} ({most_picked.get('pick_count')} times)")

            if parts:
                return "Across all tracked picks, " + "; ".join(parts) + "."

        # 2) A specific symbol was mentioned -- answer about THAT pick.
        if symbols:
            sym_ctx = symbols[0]
            symbol = sym_ctx.get("symbol")
            tracking_summary = sym_ctx.get("pickTrackingSummary")

            if not tracking_summary:
                return (
                    f"I don't have a tracked pick on record for {symbol}. "
                    "It may not have been an Alphaclara pick, or it's outside the tracked window."
                )

            status = tracking_summary.get("status")
            pick_count = tracking_summary.get("pick_count") or 0
            times = "time" if pick_count == 1 else "times"

            if status == "checked":
                return (
                    f"{symbol} was picked {pick_count} {times}. Its resolved return is "
                    f"{tracking_summary.get('checked_return_pct')}%, checked on "
                    f"{str(tracking_summary.get('checked_at') or '')[:10]}."
                )
            if status == "tracking":
                return (
                    f"{symbol} was picked {pick_count} {times} and is still being tracked. "
                    f"It's moved {tracking_summary.get('livePctSinceFirstPick')}% since first "
                    f"picked on {tracking_summary.get('first_picked_date')}."
                )
            return (
                f"{symbol} was picked {pick_count} {times}, but its outcome couldn't be "
                "determined (price data unavailable)."
            )

        # 3) Defensive: an accuracy-shaped question, in case one ever reaches
        # here (structurally it shouldn't -- accuracy_track_record's intent
        # check runs before any contextType branch and always wins first).
        if accuracy_summary and any(w in question for w in ["accura", "track record", "win rate", "success rate"]):
            horizon_days = str(accuracy_summary.get("horizon") or "").rstrip("d") or "an unknown number of"
            return (
                f"Based on {accuracy_summary.get('n')} tracked picks ({horizon_days}-day outcomes), "
                f"{accuracy_summary.get('pct_positive')}% were positive with an average return of "
                f"{accuracy_summary.get('mean_return_pct')}%."
            )

        # 4) Sensible default: an aggregate status overview -- always true and
        # useful regardless of which specific sub-question actually triggered
        # this path, without guessing at an intent we have no real signal for.
        summary_line = (
            f"Alphaclara is currently tracking {counts.get('total', 0)} picks — "
            f"{counts.get('tracking', 0)} still active, {counts.get('checked', 0)} resolved."
        )
        if accuracy_summary:
            summary_line += (
                f" Of the resolved picks, {accuracy_summary.get('pct_positive')}% were positive "
                f"with an average return of {accuracy_summary.get('mean_return_pct')}%."
            )
        return summary_line

    if context.get("contextType") == "momentum_movers":
        momentum = context.get("momentum") or {}
        selected = momentum.get("selectedMover") or {}
        pulse = momentum.get("pulse") or {}

        symbols = context.get("symbols") or []
        stock_ctx = symbols[0] if symbols else {}
        tech = stock_ctx.get("technical") or {}
        pattern = stock_ctx.get("pattern") or {}
        sig = stock_ctx.get("aiSignal") or {}

        volume = tech.get("volume") or {}
        trend = tech.get("trend") or {}
        volatility = tech.get("volatility") or {}

        sym = selected.get("symbol") or "This mover"
        reason = (
            selected.get("reason")
            or selected.get("momentumLabel")
            or "momentum signals are strengthening"
        ).rstrip(".")

        score = round(selected.get("momentumScore") or selected.get("avgAlphaScore") or 0, 1)
        appearances = selected.get("appearances") or selected.get("dailyMoverAppearances") or 0
        lookback = selected.get("lookbackSnapshots") or momentum.get("lookbackSnapshots") or 0
        move = selected.get("netMovePct") or selected.get("changePct")
        theme = pulse.get("topTheme") or selected.get("sector") or "mixed"

        move_text = f"{round(move, 2)}%" if isinstance(move, (int, float)) else "not available"

        signal = sig.get("signal")
        confidence = sig.get("confidence")
        pattern_name = pattern.get("name")
        volume_label = volume.get("label")
        volume_vs_ma20 = tech.get("volume_vs_ma20_pct")
        trend_label = trend.get("label")
        volatility_label = volatility.get("label")
        rsi = tech.get("rsi14")

        extra_parts = []

        if signal:
            extra_parts.append(
                f"market view is {clara_signal_label(signal)}"
                + (f" with {round(confidence, 1)}% confidence" if isinstance(confidence, (int, float)) else "")
            )

        if pattern_name:
            extra_parts.append(f"pattern is {pattern_name}")

        if trend_label:
            extra_parts.append(f"trend is {trend_label}")

        if isinstance(volume_vs_ma20, (int, float)):
            extra_parts.append(f"volume is {round(volume_vs_ma20)}% above its 20-day average")
        elif volume_label:
            extra_parts.append(f"volume is {volume_label.lower()}")

        risk_parts = []

        if isinstance(rsi, (int, float)) and rsi >= 70:
            risk_parts.append(f"RSI is elevated near {round(rsi, 1)}")

        if volatility_label:
            risk_parts.append(f"volatility is {volatility_label.lower()}")

        if selected.get("riskLevel"):
            risk_parts.append(f"risk level is {selected.get('riskLevel')}")

        intelligence = ". ".join(extra_parts)
        risk_text = ". ".join(risk_parts)

        answer = (
            f"{sym} is moving because {reason}. "
            f"The move is {move_text}, with a {score}/100 momentum score and {appearances} appearances across {lookback} recent snapshots. "
        )

        if intelligence:
            answer += f"Clara also sees {intelligence}. "

        if risk_text:
            answer += f"The main caution is that {risk_text}."

        return answer
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

        market_view = clara_signal_label(sig.get("signal"))

        return (
            f"{s.get('symbol')} currently shows a {market_view} with "
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
                f"{s.get('symbol')}: {clara_signal_label(sig.get('signal'))} at {sig.get('confidence')}% confidence, "
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
    context_type = context.get("contextType") or "portfolio"

    first_symbol = symbols[0].get("symbol") if symbols else None
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

        if "rated" in c and ("rated" in asked_text or "why" in asked_text):
            return True
        if "improve" in c and ("improve" in asked_text or "change" in asked_text):
            return True
        if "risk" in c and "risk" in asked_text:
            return True
        if "technical" in c and any(w in asked_text for w in ["technical", "rsi", "macd", "trend"]):
            return True
        if "pattern" in c and "pattern" in asked_text:
            return True
        if "monitor" in c and any(w in asked_text for w in ["monitor", "watch"]):
            return True
        if "compare" in c and "compare" in asked_text:
            return True
        if "overweight" in c and "overweight" in asked_text:
            return True
        if "attention" in c and any(w in asked_text for w in ["attention", "weakest", "worst"]):
            return True

        return c in asked_text

    # ✅ Momentum Movers mode: only momentum-specific follow-ups
    if context_type == "momentum_movers":
        momentum = context.get("momentum") or {}
        selected = momentum.get("selectedMover") or {}
        movers = momentum.get("movers") or []
        ai_setups = momentum.get("aiSetups") or []

        symbol = selected.get("symbol") or first_symbol or "this mover"

        compare_symbol = None
        for m in movers:
            s = m.get("symbol")
            if s and s != symbol:
                compare_symbol = s
                break

        momentum_pool = [
            f"Is {symbol}'s move real?",
            f"What are the risks with {symbol}?",
            f"Can {symbol}'s momentum continue?",
            f"Compare {symbol} with {compare_symbol}" if compare_symbol else "Compare top movers",
            "Which mover has the strongest momentum?",
            "Which AI setup looks strongest?",
            "What is the market momentum theme?",
        ]

        fresh = [q for q in momentum_pool if not already_asked(q)]
        return fresh[:5]
    
    # ✅ Market mode: only market-pulse follow-ups
    if context_type == "market":
        market_pool = [
            "Explain today’s market pulse",
            "What is market risk now?",
            "What are SPY and QQQ showing?",
            "Are crypto and commodities confirming risk?",
            "Summarize market news",
            "Explain top gainers and losers",
        ]

        fresh = [q for q in market_pool if not already_asked(q)]

        if len(fresh) < 2:
            fallback = [
                "Is this market risk-on or risk-off?",
                "What should I watch next in the market?",
                "What is driving the current market mood?",
            ]
            for q in fallback:
                if q not in fresh and not already_asked(q):
                    fresh.append(q)

        return fresh[:3]
    # ✅ Stock Detail mode: only stock-specific follow-ups
    if context_type == "stock_detail":
        stock_pool = [
            f"Why is {sym} rated this way?",
            f"What would improve {sym} signal?",
            f"What is the biggest risk for {sym}?",
            f"Explain {sym} technicals",
            f"Explain {sym} pattern risk",
            f"What should I monitor next for {sym}?",
            f"Compare {sym} with TSLA" if sym != "TSLA" else "Compare TSLA with NVDA",
        ]

        fresh = [q for q in stock_pool if not already_asked(q)]

        if len(fresh) < 2:
            fallback = [
                f"What could change the signal for {sym}?",
                f"What is the next confirmation for {sym}?",
                f"Is {sym} risk increasing?",
            ]
            for q in fallback:
                if q not in fresh and not already_asked(q):
                    fresh.append(q)

        return fresh[:3]

    # ✅ Portfolio mode: portfolio-style follow-ups
    if intent == "compare_symbols":
        pool = [
            "Which one has stronger signal?",
            "Which one has higher risk?",
            "Which one has better pattern quality?",
            "Which one has stronger technicals?",
            "Which one looks more stable?",
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

    if len(fresh) < 2:
        fallback = [
            "Which stock looks stronger?",
            "Which holding has the highest risk?",
            "What should I monitor next?",
        ]
        for q in fallback:
            if q not in fresh and not already_asked(q):
                fresh.append(q)

    return fresh[:5]

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
        "You are Clara, a calm premium AI market assistant inside Alphaclara. "
        "Answer ONLY the user's question. Do not provide a full portfolio report unless asked. "
        "Use only the provided JSON data. Do not invent numbers. "
        "If the context contains multiple symbols, compare them directly and do not say another symbol's data is unavailable."
        "Be concise, direct, and practical. "
        "If contextType is market, explain only the provided Market Pulse data: SPY, QQQ, crypto, commodities ETFs, Fear & Greed, internal movers, and market news. Do not claim this represents the entire official market. "
        "Use simple language for retail investors. "
        "Do not use markdown headings, bullet lists, asterisks, or long explanations. "
        "Do not give financial advice. Use educational wording only. "
        "Maximum answer length: 4 short sentences. "
        "If the question asks for one thing, answer that one thing first."
        "Never say raw internal signal codes (e.g. BUY, HOLD, SELL, BULLISH_WATCH, BEARISH_WATCH, "
        "STRONG_BULLISH, CAUTION, HIGH_RISK_MOMENTUM). Use plain language instead: Bullish Setup, "
        "Neutral Setup, or Risk Alert. "
        "Explain in simple non-technical language. "
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

    # ✅ Add any symbols mentioned in the question, even if not in portfolio.
    # Extracted from the ORIGINAL, non-uppercased question -- casing is a
    # real signal ("PANW" vs "it"/"now") that a pre-uppercase step would
    # destroy before ever getting a chance to use it. Only required for
    # the narrow _AMBIGUOUS_TICKERS set; every other real ticker still
    # matches regardless of how the user typed it.
    mentioned_raw = re.findall(r"\b[A-Za-z]{1,5}\b", req.question or "")

    for raw in mentioned_raw:
        sym = raw.upper()
        if sym in available_symbols or sym not in _REAL_TICKER_SET:
            continue
        if sym in _AMBIGUOUS_TICKERS and raw != sym:
            # Ambiguous ticker, not typed in all-caps -- treat as the
            # ordinary English word it almost certainly is.
            continue
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

    # Momentum Movers already has structured internal data.
    # Use deterministic answer first to avoid vague LLM responses.
    if getattr(req, "contextType", None) == "momentum_movers":
        system_prompt = (
            "You are Clara, a premium market intelligence assistant. "
            "Rewrite the provided market facts into a natural, concise answer. "
            "Do not invent facts, prices, catalysts, ratings, or predictions. "
            "Do not sound robotic or templated. "
            "No markdown. No bullets. Maximum 3 short sentences. "
            "Educational wording only, no financial advice."
        )

        user_prompt = (
            f"User question: {resolved_question}\n\n"
            f"Facts:\n{fallback_answer}\n\n"
            "Write a natural answer that sounds like a thoughtful market analyst."
        )

        llm_answer = astra_llm_answer_fn(system_prompt, user_prompt)

        return {
            "answer": sanitize_clara_answer(llm_answer or fallback_answer),
            "usedLLM": llm_answer is not None,
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
    system_prompt, user_prompt = build_astra_prompt(context)

    llm_answer = astra_llm_answer_fn(system_prompt, user_prompt)

    return {
        "answer": sanitize_clara_answer(llm_answer or fallback_answer),
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