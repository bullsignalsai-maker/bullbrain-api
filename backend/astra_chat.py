# backend/astra_chat.py
# ---------------------------------------------------------------
# Astra Chat Engine (NO FastAPI)
# - Full parity with your current main.py/test.py Astra logic
# - Exposes:
#     - AstraPosition, AstraChatRequest (Pydantic models)
#     - astra_chat_engine(req) -> dict
# ---------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
import re

# ---- Backend dependencies (already migrated) ----
from backend.market_data import fetch_daily_candles
from backend.bullbrain import compute_bullbrain_features, bullbrain_infer
from backend.grok_ai import astra_llm_answer


# ---------------------------------------------------------------
# Astra Chat Request Models
# ---------------------------------------------------------------
class AstraPosition(BaseModel):
    symbol: str
    shares: float
    avg_cost: float
    price: float
    gain: float
    gain_pct: float
    allocation_pct: float
    today: float


class AstraChatRequest(BaseModel):
    # Either a free-form question or a predefined question_id from the app
    question: Optional[str] = ""
    question_id: Optional[str] = None

    total_value: float = 0.0
    total_gain: float = 0.0
    today_gain: float = 0.0

    positions: List[AstraPosition] = []


# ---------------------------------------------------------------
# Helper: lightweight market sentiment for a symbol
# (Full parity with your code, but NO FastAPI)
# ---------------------------------------------------------------
def astra_symbol_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Lightweight, resilient sentiment block for a single symbol.
    Uses:
      - Daily candles for price/vol move
      - Last close vs prev close
    Keeps it simple, two lines max for Astra to use.
    """
    symbol = symbol.upper()
    sentiment: Dict[str, Any] = {
        "symbol": symbol,
        "price_comment": "",
        "volume_comment": "",
        "summary": "",
    }

    try:
        candles = fetch_daily_candles(symbol)
        if not candles or len(candles) < 2:
            sentiment["summary"] = (
                f"{symbol} market sentiment could not be derived from recent price data."
            )
            return sentiment

        # Your existing code assumed a list of dict OHLCV candles in some places.
        # fetch_daily_candles in your backend typically returns dict-of-arrays.
        # We'll support BOTH shapes safely.

        def _get_candle_close(c):
            return float(c.get("c", c.get("close", 0)))

        def _get_candle_vol(c):
            return float(c.get("v", c.get("volume", 0)))

        # --- If dict-of-arrays shape ---
        if isinstance(candles, dict) and "close" in candles and isinstance(candles["close"], list):
            closes = candles.get("close", [])
            vols = candles.get("volume", [])
            if len(closes) < 2:
                sentiment["summary"] = (
                    f"{symbol} market sentiment could not be derived from recent price data."
                )
                return sentiment

            last_close = float(closes[-1] or 0)
            prev_close = float(closes[-2] or 0)
            last_vol = float(vols[-1] or 0) if vols else 0.0

            recent_vols = vols[-10:] if vols and len(vols) >= 10 else (vols or [])
            avg_vol = (
                sum(float(v or 0) for v in recent_vols) / max(len(recent_vols), 1)
                if recent_vols
                else 0.0
            )

        # --- Else list-of-dict shape ---
        elif isinstance(candles, list):
            last = candles[-1]
            prev = candles[-2]
            last_close = _get_candle_close(last)
            prev_close = _get_candle_close(prev)
            last_vol = _get_candle_vol(last)

            recent = candles[-10:] if len(candles) >= 10 else candles
            avg_vol = (
                sum(_get_candle_vol(c) for c in recent) / max(len(recent), 1)
                if recent
                else 0.0
            )
        else:
            sentiment["summary"] = f"{symbol} sentiment is unclear based on current data."
            return sentiment

        price_change_pct = (
            ((last_close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        )

        if price_change_pct > 3:
            price_comment = f"Price is up about {price_change_pct:.1f}% today."
        elif price_change_pct < -3:
            price_comment = f"Price is down about {price_change_pct:.1f}% today."
        elif abs(price_change_pct) < 0.5:
            price_comment = "Price is almost flat today."
        else:
            direction = "up" if price_change_pct > 0 else "down"
            price_comment = f"Price is {direction} about {abs(price_change_pct):.1f}% today."

        vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else 1.0

        if vol_ratio > 1.3:
            volume_comment = "Volume is higher than its recent average, so interest is elevated."
        elif vol_ratio < 0.7:
            volume_comment = "Volume is lower than usual, so moves may not be strongly confirmed."
        else:
            volume_comment = "Volume is close to its recent average."

        summary = f"{price_comment} {volume_comment}"

        sentiment["price_comment"] = price_comment
        sentiment["volume_comment"] = volume_comment
        sentiment["summary"] = summary.strip()
        return sentiment

    except Exception as e:
        print(f"Astra sentiment error for {symbol}:", e)
        sentiment["summary"] = f"{symbol} sentiment is unclear based on current data."
        return sentiment


# ---------------------------------------------------------------
# ASTRA CHAT — Portfolio AI Analyst (BullBrain v2 + Grok + Sentiment)
# (Full parity with your pasted logic)
# ---------------------------------------------------------------
def astra_chat_engine(req: AstraChatRequest) -> Dict[str, Any]:
    """
    Astra – Artificial Stock Trading & Risk Analyst for BullSignalsAI.

    Uses:
    - Daily candles
    - BullBrain v2 (48 features) per symbol
    - Allocation %, gain %, position value
    - Grok/XAI for natural language answers
    - Lightweight price/volume-based sentiment

    Works for both:
    - Predefined questions (question_id)
    - Custom free-form questions (question)
    """

    # 1) Basic validation
    if not req.positions or req.total_value <= 0:
        return {
            "answer": (
                "I need at least one holding with a non-zero portfolio value "
                "to analyze. Please add positions to your portfolio and try again."
            ),
            "used_llm": False,
            "analysis": {},
        }

    # 2) Resolve question (chips or custom text)
    q_map = {
        "overview": "Give a concise overview of how my portfolio is doing.",
        "risk_exposure": "Explain my overall risk exposure and concentration.",
        "overweight": "Which stocks are overweight or underweight, and what should I consider doing?",
        "worst": "Which positions need my urgent attention and why?",
        "ai_suggestions": "Give me AI-driven suggestions on how to optimize this portfolio.",
    }

    base_question = (req.question or "").strip()
    if not base_question and req.question_id:
        base_question = q_map.get(
            req.question_id,
            "Give a clear, practical analysis of this portfolio.",
        )

    if not base_question:
        base_question = "Give a clear, practical analysis of this portfolio."

    question_id = req.question_id or ""

    # 3) Sort positions by allocation (focus on biggest ones first)
    positions_sorted = sorted(req.positions, key=lambda p: p.allocation_pct, reverse=True)

    # Limit how many symbols we send to BullBrain (performance)
    max_symbols_for_model = 10
    focus_positions = positions_sorted[:max_symbols_for_model]

    per_symbol_analysis: List[Dict[str, Any]] = []
    bullbrain_failures: List[str] = []

    # 4) Per-symbol BullBrain v2 analysis
    for pos in focus_positions:
        symbol = pos.symbol.upper()

        try:
            candles = fetch_daily_candles(symbol)
            if not candles:
                bullbrain_failures.append(symbol)
                continue

            features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
            if features_vec is None:
                bullbrain_failures.append(symbol)
                continue

            out = bullbrain_infer(features_vec)
            prob_up = float(out.get("probability_up") or out.get("raw_output") or 0.5)
            signal = out.get("signal") or "NEUTRAL"

            vol = float(feature_dict.get("volatility_5d", 0.02) or 0.02)
            # Approx expected move (5d-ish) on a -1..+1 scale
            expected_move = vol * (prob_up * 2 - 1)
            # Convert to %
            expected_move_pct = round(expected_move * 100, 2)

            # Confidence bucket
            if prob_up >= 0.66:
                confidence = "High"
            elif prob_up >= 0.55:
                confidence = "Moderate"
            else:
                confidence = "Low"

            # Risk bucket from volatility
            if vol < 0.015:
                risk = "Low"
            elif vol < 0.035:
                risk = "Medium"
            else:
                risk = "High"

            # Simple pattern from SMAs
            sma5 = float(feature_dict.get("sma5", 0) or 0)
            sma20 = float(feature_dict.get("sma20", 0) or 0)
            if sma5 > sma20:
                pattern = "Short-term momentum"
            elif sma5 < sma20:
                pattern = "Reversal risk"
            else:
                pattern = "Sideways consolidation"

            # Over/under-weight (vs equal-weight baseline)
            equal_weight = 100.0 / max(len(req.positions), 1)
            allocation = float(pos.allocation_pct or 0.0)
            if allocation > equal_weight * 1.8:
                weight_flag = "Strongly Overweight"
            elif allocation > equal_weight * 1.2:
                weight_flag = "Overweight"
            elif allocation < equal_weight * 0.5:
                weight_flag = "Underweight"
            else:
                weight_flag = "Balanced"

            per_symbol_analysis.append(
                {
                    "symbol": symbol,
                    "allocation_pct": round(float(pos.allocation_pct or 0.0), 2),
                    "gain_pct": round(float(pos.gain_pct or 0.0), 2),
                    "unrealized_gain": round(float(pos.gain or 0.0), 2),
                    "today_pl": round(float(pos.today or 0.0), 2),
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "price": pos.price,
                    "bullbrain": {
                        "signal": signal,
                        "prob_up": round(prob_up * 100, 1),  # %
                        "expected_move_pct": expected_move_pct,
                        "risk": risk,
                        "confidence": confidence,
                        "pattern": pattern,
                    },
                    "weight_flag": weight_flag,
                }
            )

        except Exception as e:
            print(f"Astra BullBrain error for {symbol}:", e)
            bullbrain_failures.append(symbol)
            continue

    if not per_symbol_analysis:
        # If BullBrain failed for everything, we at least return basic metrics.
        return {
            "answer": (
                "I could not compute AI signals for your holdings at this time. "
                "Please try again later. Your portfolio still has a positive value, "
                "but BullBrain analysis is temporarily unavailable."
            ),
            "used_llm": False,
            "analysis": {
                "total_value": req.total_value,
                "total_gain": req.total_gain,
                "today_gain": req.today_gain,
                "positions": [p.symbol for p in req.positions],
                "bullbrain_unavailable": bullbrain_failures,
            },
        }

    # 5) Portfolio-level aggregates

    # Overall return %
    overall_return_pct = (
        (req.total_gain / max(req.total_value - req.total_gain, 1)) * 100.0
        if req.total_value > 0
        else 0.0
    )

    # Weighted average bullish probability (by allocation)
    total_alloc_for_weight = sum(p["allocation_pct"] for p in per_symbol_analysis) or 1
    weighted_prob_up = (
        sum(p["allocation_pct"] * p["bullbrain"]["prob_up"] for p in per_symbol_analysis)
        / total_alloc_for_weight
    )

    # Top contributors / laggards
    sorted_by_gain = sorted(per_symbol_analysis, key=lambda p: p["unrealized_gain"], reverse=True)
    top_gainer = sorted_by_gain[0] if sorted_by_gain else None
    worst_loser = sorted_by_gain[-1] if sorted_by_gain else None

    # Risk exposure heuristic: max allocation and high-vol names
    max_alloc = max(p["allocation_pct"] for p in per_symbol_analysis)
    high_risk_names = [
        p for p in per_symbol_analysis
        if p["bullbrain"]["risk"] == "High" and p["allocation_pct"] >= 10
    ]

    if max_alloc > 40:
        portfolio_risk_label = "High (concentrated in a single position)"
    elif max_alloc > 25:
        portfolio_risk_label = "Moderate to High (few large positions)"
    else:
        portfolio_risk_label = "Balanced to Moderate"

    # 6) Hybrid sentiment selection (FULL parity)
    portfolio_symbols = [p["symbol"] for p in per_symbol_analysis]
    sentiment_targets: List[str] = []

    # a) For predefined questions, pick a small, meaningful set of symbols
    if question_id == "overview":
        # focus on largest holding + top gainer + worst loser
        if per_symbol_analysis:
            sentiment_targets.append(per_symbol_analysis[0]["symbol"])
        if top_gainer:
            sentiment_targets.append(top_gainer["symbol"])
        if worst_loser:
            sentiment_targets.append(worst_loser["symbol"])

    elif question_id == "risk_exposure":
        # high risk concentration names
        sentiment_targets.extend([p["symbol"] for p in high_risk_names])

    elif question_id == "overweight":
        overweight = [p for p in per_symbol_analysis if "Overweight" in p["weight_flag"]]
        sentiment_targets.extend([p["symbol"] for p in overweight[:4]])

    elif question_id == "worst":
        if worst_loser:
            sentiment_targets.append(worst_loser["symbol"])

    elif question_id == "ai_suggestions":
        overweight = [p for p in per_symbol_analysis if "Overweight" in p["weight_flag"]]
        if top_gainer:
            sentiment_targets.append(top_gainer["symbol"])
        if worst_loser:
            sentiment_targets.append(worst_loser["symbol"])
        sentiment_targets.extend([p["symbol"] for p in overweight[:3]])

    # b) For custom questions, detect mentioned tickers
    else:
        upper_q = base_question.upper()
        mentioned: List[str] = []
        for sym in portfolio_symbols:
            if re.search(rf"\b{re.escape(sym)}\b", upper_q):
                mentioned.append(sym)
        if mentioned:
            sentiment_targets.extend(mentioned[:5])

    # remove duplicates while preserving order
    seen = set()
    uniq_targets: List[str] = []
    for s in sentiment_targets:
        if s and s not in seen:
            seen.add(s)
            uniq_targets.append(s)

    sentiment_map: Dict[str, Any] = {}
    for sym in uniq_targets:
        sentiment_map[sym] = astra_symbol_sentiment(sym)

    # 7) Build analysis object
    analysis_obj: Dict[str, Any] = {
        "question": base_question,
        "question_id": question_id,
        "total_value": round(req.total_value, 2),
        "total_gain": round(req.total_gain, 2),
        "today_gain": round(req.today_gain, 2),
        "overall_return_pct": round(overall_return_pct, 2),
        "weighted_bull_prob_pct": round(weighted_prob_up, 1),
        "portfolio_risk_label": portfolio_risk_label,
        "top_gainer": top_gainer,
        "worst_loser": worst_loser,
        "high_risk_concentrations": high_risk_names,
        "per_symbol": per_symbol_analysis,
        "bullbrain_failed_symbols": bullbrain_failures,
        "sentiment": sentiment_map,
    }

    # 8) Build LLM prompt for Grok (Astra) — FULL parity
    system_prompt = (
        "You are Astra, an expert AI portfolio analyst inside a mobile app "
        "called BullSignalsAI. The user is a retail investor.\n"
        "You must be clear, calm, and practical.\n"
        "Use real numbers from the JSON (values, percentages, tickers, shares).\n"
        "Do not invent numbers. If a value is missing, simply say it is not available.\n"
        "Do NOT use markdown, bullet symbols, asterisks, or headings.\n"
        "Write in plain text with short paragraphs.\n"
        "Tailor your answer to the specific question. Do not repeat the same template every time.\n"
        "Avoid financial jargon. Do not say buy or sell; instead say things like "
        "consider reducing exposure or consider increasing exposure.\n"
    )

    user_prompt = (
        f"User question: {base_question}\n\n"
        "Here is their portfolio and AI analysis as JSON. "
        "This includes BullBrain v2 outputs and a small sentiment block for some symbols.\n\n"
        f"{json.dumps(analysis_obj, indent=2)}\n\n"
        "Now answer the user's question using this data only.\n"
        "Use the following style:\n"
        "- If the question is about the whole portfolio (overview, risk, suggestions), "
        "talk about total value, total gain or loss, overall return percent, "
        "and mention 2–3 key tickers with their allocation and gain or loss.\n"
        "- If the question is about a specific stock, focus on that ticker's allocation, "
        "gain or loss, BullBrain signal, probability_up and expected_move_pct, "
        "plus the sentiment summary for that ticker if available.\n"
        "- For overweight or underweight questions, mention which tickers are flagged as overweight "
        "or underweight and what the user could consider doing.\n"
        "- Keep the answer concise. Usually 1 to 3 short paragraphs are enough.\n"
        "- Always close with one short sentence reminding the user that this is AI-driven insight, "
        "not personal financial advice.\n"
    )

    llm_answer = astra_llm_answer(system_prompt, user_prompt)
    used_llm = llm_answer is not None

    # 9) Fallback answer if Grok fails — FULL parity
    if not llm_answer:
        lines: List[str] = []
        lines.append(
            f"Your portfolio is currently valued at about ${req.total_value:,.2f} "
            f"with an overall return of approximately {overall_return_pct:+.2f} percent."
        )
        lines.append(
            f"Across your largest positions, BullBrain sees an average bullish probability "
            f"of around {weighted_prob_up:.1f} percent."
        )
        lines.append(f"Risk level is classified as {portfolio_risk_label}.")

        if top_gainer:
            lines.append(
                f"Your main positive contributor right now is {top_gainer['symbol']}, "
                f"with an unrealized gain of about ${top_gainer['unrealized_gain']:,.2f} "
                f"and a change of roughly {top_gainer['gain_pct']:+.1f} percent."
            )
        if worst_loser:
            lines.append(
                f"The biggest drag is {worst_loser['symbol']}, "
                f"at about ${worst_loser['unrealized_gain']:,.2f} "
                f"and {worst_loser['gain_pct']:+.1f} percent."
            )

        if sentiment_map:
            # Use first sentiment as example
            s_sym, s_val = next(iter(sentiment_map.items()))
            if s_val.get("summary"):
                lines.append(f"For example, {s_sym}: {s_val['summary']}")

        lines.append(
            "Treat this as AI-supported analysis to guide your thinking, not as personal financial advice."
        )

        llm_answer = " ".join(lines)

    # 10) Return structured response — FULL parity
    return {
        "answer": llm_answer,
        "used_llm": used_llm,
        "analysis": analysis_obj,
    }
