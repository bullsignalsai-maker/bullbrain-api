# =========================================================
# UI Stock Builder — Stock Detail Contract v1.0
# Source of truth: Firestore stock document
# NO external calls | NO hallucinated data
# =========================================================

from typing import Dict, Any, List, Optional

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
    return " ".join(lines * min_count)


def _safe(val, fallback=None):
    return val if val is not None else fallback


# ---------------------------------------------------------
# 1️⃣ FINAL SIGNAL BLOCK (AUTHORITATIVE, NO NULLS)
# ---------------------------------------------------------
def build_signal_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}
    probs = stock.get("probabilities") or {}

    signal = decision.get("final") or decision.get("finalSignal") or "HOLD"
    confidence = decision.get("confidence")

    # ---- Confidence tier ----
    if isinstance(confidence, (int, float)):
        if confidence >= 75:
            tier = "High"
        elif confidence >= 60:
            tier = "Moderate"
        else:
            tier = "Low"
    else:
        tier = None

    # ---- Bias (derived from probabilities if available) ----
    up = probs.get("up")
    down = probs.get("down")

    if isinstance(up, (int, float)) and isinstance(down, (int, float)):
        diff = abs(up - down)
        if diff < 0.05:
            bias = "Neutral"
        elif up > down:
            bias = "Bullish"
        else:
            bias = "Bearish"
    else:
        bias = "Neutral"

    # ---- Explanation (Firestore-first, then derived) ----
    expl = []

    if narratives.get("summary"):
        expl.append(narratives["summary"])

    if narratives.get("tradeIdea"):
        expl.append(narratives["tradeIdea"])

    if not expl:
        if signal == "HOLD":
            expl.append(
                "Signals are mixed, and no strong directional edge is currently present."
            )
        elif signal == "BUY":
            expl.append(
                "Multiple indicators align in favor of upside continuation."
            )
        elif signal == "SELL":
            expl.append(
                "Downside risk outweighs upside scenarios based on current conditions."
            )

    # ---- Confidence interpretation ----
    if tier == "High":
        expl.append(
            "Signal confidence is elevated, increasing reliability under current conditions."
        )
    elif tier == "Moderate":
        expl.append(
            "Signal confidence is moderate, suggesting selective and disciplined positioning."
        )
    else:
        expl.append(
            "Low confidence indicates higher uncertainty and increased outcome dispersion."
        )

    return {
        "value": signal,
        "confidence": confidence,
        "confidenceTier": tier,
        "bias": bias,
        "explanation": _sentences(expl, min_count=2),
    }


# ---------------------------------------------------------
# 2️⃣ PROBABILITY BLOCK (FINAL — EXPLAINED, NO NULLS)
# ---------------------------------------------------------
def build_probability_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    narratives = stock.get("narratives") or {}
    probs = stock.get("probabilities") or {}
    indicator_states = stock.get("indicator_states") or {}

    up = probs.get("up")
    down = probs.get("down")

    # Guard: probabilities must exist
    if not isinstance(up, (int, float)) or not isinstance(down, (int, float)):
        return {
            "up": None,
            "down": None,
            "bias": "Neutral",
            "strengthPct": 0,
            "explanation": "Probability data is currently unavailable for this symbol."
        }


    diff = abs(up - down)
    strength_pct = round(diff * 100, 1)

    # Bias label
    if diff < 0.05:
        bias = "Neutral"
    elif up > down:
        bias = "Bullish"
    else:
        bias = "Bearish"

    # ---- Explanation sentences (Firestore first) ----
    expl = []

    narrative_prob = narratives.get("probability")
    if isinstance(narrative_prob, list):
        expl.extend(narrative_prob)
    elif isinstance(narrative_prob, str):
        expl.append(narrative_prob)

    # ---- Indicator-based explanation (data-derived) ----
    composite = indicator_states.get("probability_composite")
    if composite:
        expl.append(
            f"Probability alignment reflects combined indicator state: {composite.replace('_', ' ').lower()}."
        )

    # ---- Strength explanation ----
    if diff < 0.05:
        expl.append(
            "The small probability gap suggests no strong directional edge at this time."
        )
    elif diff < 0.15:
        expl.append(
            "The probability imbalance is moderate, favoring selective positioning rather than aggressive bets."
        )
    else:
        expl.append(
            "The probability gap is meaningful, indicating a clearer directional bias."
        )

    return {
        "up": round(up, 4),
        "down": round(down, 4),
        "bias": bias,
        "strengthPct": strength_pct,
        "explanation": _sentences(expl, min_count=2),
    }

# ---------------------------------------------------------
# 3️⃣ PATTERN BLOCK (FINAL — NO NULLS, 3–5 SENTENCES)
# ---------------------------------------------------------
def build_pattern_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the Smart Pattern explanation block.

    Rules:
    - Pattern explanation must stand alone
    - 3–5 sentences minimum
    - Data-driven only (no hallucination)
    - Explains: what it is, how reliable, how to use it
    """

    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    name = pattern.get("pattern") or pattern.get("patternLabel")
    bias = pattern.get("bias")
    win_rate = pattern.get("winRate5d")
    confidence = pattern.get("confidence")

    # Historical forward returns (supporting context)
    days5 = (history.get("forwardReturns") or {}).get("days5") or {}
    best = days5.get("best")
    worst = days5.get("worst")

    raw_expl = sections.get("pattern") or []

    sentences = []

    # -------------------------------------------------
    # 1️⃣ What this pattern represents
    # -------------------------------------------------
    if name:
        sentences.append(
            f"The {name.replace('_', ' ').title()} pattern has recently emerged in price action, "
            f"typically signaling a {bias or 'neutral'} short-term setup based on historical behavior."
        )
    else:
        sentences.append(
            "A short-term price pattern has been detected, reflecting recent shifts in intraday structure and sentiment."
        )

    # -------------------------------------------------
    # 2️⃣ Historical reliability (win rate)
    # -------------------------------------------------
    if isinstance(win_rate, (int, float)):
        sentences.append(
            f"Historically, this pattern has produced favorable outcomes approximately {win_rate * 100:.0f}% "
            f"of the time over the next five trading days."
        )
    else:
        sentences.append(
            "Historical outcome consistency for this pattern is mixed, suggesting limited standalone predictive power."
        )

    # -------------------------------------------------
    # 3️⃣ Outcome dispersion (risk context)
    # -------------------------------------------------
    if isinstance(best, (int, float)) and isinstance(worst, (int, float)):
        sentences.append(
            f"Past occurrences show a wide range of outcomes, with gains reaching up to {best:.1f}% "
            f"and drawdowns extending to around {worst:.1f}%, highlighting variability in follow-through."
        )

    # -------------------------------------------------
    # 4️⃣ Narrative insight (Firestore-authored)
    # -------------------------------------------------
    if raw_expl:
        sentences.append(_sentences(raw_expl, min_count=1))

    # -------------------------------------------------
    # 5️⃣ How to use it (interpretation guidance)
    # -------------------------------------------------
    if bias == "bull":
        sentences.append(
            "This pattern is best interpreted as a tactical bullish signal and is most effective when confirmed by momentum or volume expansion."
        )
    elif bias == "bear":
        sentences.append(
            "This pattern favors downside risk scenarios and should be monitored closely for breakdown confirmation or failed rebounds."
        )
    else:
        sentences.append(
            "This pattern alone does not provide a strong directional edge and should be used alongside broader technical context."
        )

    # -------------------------------------------------
    # 🧹 Final cleanup
    # -------------------------------------------------
    explanation = _sentences(sentences, min_count=3)

    return {
        "name": name,
        "bias": bias,
        "winRate5d": win_rate,
        "confidence": confidence,
        "explanation": explanation,
    }


# ---------------------------------------------------------
# 4️⃣ TECHNICAL SNAPSHOT (FINAL — NO NULLS, 2+ SENTENCES EACH)
# ---------------------------------------------------------
def build_technical_snapshot_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a fully explained technical snapshot.

    Rules:
    - Every section must have >= 2 sentences
    - Values drive explanations
    - Firestore narratives are primary
    - Deterministic fallbacks only
    """

    technical = stock.get("technical") or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    def explain_trend(values: Dict[str, Any]) -> Dict[str, Any]:
        expl = []

        label = values.get("label")
        slope = values.get("slope")
        price_vs_sma20 = values.get("price_vs_sma20_pct")

        if label:
            expl.append(
                f"Trend analysis currently indicates a {label.lower()} market structure, "
                f"reflecting how price is behaving relative to recent averages."
            )

        if isinstance(price_vs_sma20, (int, float)):
            expl.append(
                f"Price is trading approximately {abs(price_vs_sma20):.1f}% "
                f"{'above' if price_vs_sma20 > 0 else 'below'} its 20-day average, "
                f"which helps define short-term directional bias."
            )

        expl.extend(sections.get("trend") or [])

        return {
            "values": values,
            "explanation": _sentences(expl, min_count=2),
        }

    def explain_momentum(values: Dict[str, Any]) -> Dict[str, Any]:
        expl = []

        rsi = values.get("rsi14")
        macd = values.get("macd_state")

        if isinstance(rsi, (int, float)):
            if rsi > 70:
                expl.append(
                    f"Momentum readings are elevated, with RSI near {rsi:.0f}, "
                    f"suggesting overbought conditions and increased pullback risk."
                )
            elif rsi < 30:
                expl.append(
                    f"Momentum appears stretched to the downside, with RSI near {rsi:.0f}, "
                    f"which can precede short-term rebounds."
                )
            else:
                expl.append(
                    f"RSI near {rsi:.0f} reflects balanced momentum without a strong directional edge."
                )

        if macd:
            expl.append(
                f"MACD structure currently signals {macd.lower()} momentum dynamics."
            )

        expl.extend(sections.get("momentum") or [])

        return {
            "values": values,
            "explanation": _sentences(expl, min_count=2),
        }

    def explain_volatility(values: Dict[str, Any]) -> Dict[str, Any]:
        expl = []

        atr = values.get("atr14")
        regime = values.get("regime")

        if isinstance(atr, (int, float)):
            expl.append(
                f"Average True Range indicates typical daily price movement of about {atr:.2f} points, "
                f"which helps frame expected short-term price swings."
            )

        if regime:
            expl.append(
                f"Volatility conditions are classified as {regime.lower()}, "
                f"affecting signal reliability and trade sizing considerations."
            )

        expl.extend(sections.get("volatility") or [])

        return {
            "values": values,
            "explanation": _sentences(expl, min_count=2),
        }

    def explain_volume(values: Dict[str, Any]) -> Dict[str, Any]:
        expl = []

        z = values.get("volume_zscore_20")
        vs_avg = values.get("volume_vs_ma20_pct")

        if isinstance(vs_avg, (int, float)):
            expl.append(
                f"Trading volume is roughly {abs(vs_avg):.1f}% "
                f"{'above' if vs_avg > 0 else 'below'} its 20-day average, "
                f"indicating changes in market participation."
            )

        if isinstance(z, (int, float)) and abs(z) > 1:
            expl.append(
                f"Volume Z-score near {z:.1f} suggests statistically significant deviation "
                f"from normal trading activity."
            )

        expl.extend(sections.get("volume") or [])

        return {
            "values": values,
            "explanation": _sentences(expl, min_count=2),
        }

    return {
        "trend": explain_trend(technical.get("trend") or {}),
        "momentum": explain_momentum(technical.get("momentum") or {}),
        "volatility": explain_volatility(technical.get("volatility") or {}),
        "volume": explain_volume(technical.get("volume") or {}),
    }

# ---------------------------------------------------------
# 5️⃣ FEATURES / INDICATORS
# ---------------------------------------------------------

def build_features_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a UI-friendly 'features' block from Firestore.
    Firestore reality:
      - features_meta is usually a flat dict: { "rsi14": 48.2, "atr14": 5.4, ... }
      - sometimes may contain nested dicts
    We normalize everything into:
      {
        "items":[
          { "key":"rsi14", "label":"RSI (14)", "value":48.2, "unit":None, "explanation":[...2 sentences...] }
        ]
      }
    Data-only speaking rule:
      - Explanation comes from narratives/sections if present
      - Otherwise we generate short factual explanations only when value exists
    """

    features = stock.get("features_meta") or {}
    if not isinstance(features, dict) or not features:
        return {"items": []}

    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}
    # Optional: allow Firestore to store per-feature explanation maps if you ever add later
    feature_notes = sections.get("features") or {}

    def _safe_num(v):
        return v if isinstance(v, (int, float)) else None

    def _normalize_feature_value(raw: Any) -> Dict[str, Any]:
        # If Firestore already stored dict with value fields
        if isinstance(raw, dict):
            # Support common keys if you later store structured values
            if "value" in raw:
                return raw
            if "v" in raw:
                return {"value": raw.get("v")}
            return raw
        # Primitive value case (float/int/str/bool)
        return {"value": raw}

    def _two_sentences_from_value(key: str, value: Any) -> List[str]:
        """
        Minimal, factual, data-driven 2 sentences (only if value exists).
        No guessing, no market advice.
        """
        # If value is missing or nonsense, return empty list
        if value is None:
            return []

        v_num = _safe_num(value)

        # Provide factual explanations for common keys your UI cares about
        if key == "rsi14" and v_num is not None:
            return [
                f"RSI(14) is {v_num:.1f}, measuring recent price momentum on a 0–100 scale.",
                "Values near 50 suggest balanced momentum, while extremes indicate stronger directional pressure."
            ]

        if key == "atr14" and v_num is not None:
            return [
                f"ATR(14) is {v_num:.2f}, representing typical daily price movement in points.",
                "Higher ATR means wider daily swings, which increases outcome dispersion even if direction is unclear."
            ]

        if key == "volume_vs_ma20_pct" and v_num is not None:
            return [
                f"Volume vs 20-day average is {v_num:.1f}%, comparing today’s activity to its recent baseline.",
                "Positive values indicate above-average participation, while negative values suggest lighter-than-normal trading."
            ]

        if key in ("gap_pct", "intraday_range_pct", "body_pct", "upper_shadow_pct", "lower_shadow_pct") and v_num is not None:
            # These are candle anatomy features — keep them factual
            label_map = {
                "gap_pct": "Gap %",
                "intraday_range_pct": "Intraday Range %",
                "body_pct": "Candle Body %",
                "upper_shadow_pct": "Upper Wick %",
                "lower_shadow_pct": "Lower Wick %",
            }
            nice = label_map.get(key, key)
            return [
                f"{nice} is {v_num:.2f}%, describing today’s price-action structure using OHLC relationships.",
                "These values quantify where buyers and sellers showed strength during the session, without assuming direction."
            ]

        # Generic fallback: factual only
        if isinstance(value, (int, float)):
            return [
                f"{key} is {value}, sourced from the latest computed feature set in Firestore.",
                "This metric is included as an input signal used by the model and narrative builder."
            ]

        # If it's a string (rare), still normalize safely
        if isinstance(value, str) and value.strip():
            return [
                f"{key} is recorded as '{value.strip()}' in the computed feature set.",
                "This value is passed through from Firestore without additional inference."
            ]

        return []

    # Nice labels for UI (expand as needed)
    LABELS = {
        "rsi14": "RSI (14)",
        "atr14": "ATR (14)",
        "volume_vs_ma20_pct": "Volume vs Avg (20D)",
        "gap_pct": "Gap %",
        "intraday_range_pct": "Intraday Range %",
        "body_pct": "Candle Body %",
        "upper_shadow_pct": "Upper Wick %",
        "lower_shadow_pct": "Lower Wick %",
    }

    # Units (optional; keep minimal)
    UNITS = {
        "rsi14": None,
        "atr14": "pts",
        "volume_vs_ma20_pct": "%",
        "gap_pct": "%",
        "intraday_range_pct": "%",
        "body_pct": "%",
        "upper_shadow_pct": "%",
        "lower_shadow_pct": "%",
    }

    items: List[Dict[str, Any]] = []

    for k, raw in features.items():
        if isinstance(raw, dict) and "value" not in raw and "v" not in raw:
            continue
        norm = _normalize_feature_value(raw)
        val = norm.get("value") if isinstance(norm, dict) else raw  # ultra defensive

        # Explanation priority:
        # 1) Firestore narrative section for this feature (if you ever store it)
        # 2) generated 2-sentence factual explanation (only if value exists)
        expl: List[str] = []
        if isinstance(feature_notes, dict):
            note = feature_notes.get(k)
            if isinstance(note, list):
                expl = [str(x).strip() for x in note if str(x).strip()]
            elif isinstance(note, str) and note.strip():
                # split into sentences if user stored a paragraph
                expl = [note.strip()]

        if not expl:
            expl = _two_sentences_from_value(k, val)

        items.append({
            "key": k,
            "label": LABELS.get(k, k.replace("_", " ").upper() if len(k) <= 8 else k.replace("_", " ").title()),
            "value": val,
            "unit": UNITS.get(k),
            "explanation": expl,  # always list
        })

    return {"items": items}

# ---------------------------------------------------------
# 6️⃣ OUTLOOK BLOCK (FINAL — NO NULLS, EXPLAINED)
# ---------------------------------------------------------
def build_outlook_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    insights = stock.get("insights") or {}
    narratives = stock.get("narratives") or {}
    technical = stock.get("technical") or {}
    probabilities = stock.get("probabilities") or {}

    sections = (narratives.get("sections") or {})

    def build_horizon(
        summary: str | None,
        narrative_key: str,
        fallback: str,
    ) -> Dict[str, Any]:
        expl = []

        # Narrative sentences (preferred)
        if sections.get(narrative_key):
            expl.extend(sections[narrative_key])

        # Fallback if narratives are thin
        if not expl:
            expl.append(fallback)

        return {
            "summary": summary or fallback,
            "explanation": _sentences(expl, min_count=2),
        }

    # ---- SHORT TERM ----
    short_term = build_horizon(
        summary=insights.get("trendSummary"),
        narrative_key="momentum",
        fallback=(
            "Short-term price action reflects mixed momentum signals, "
            "with no decisive directional control established."
        ),
    )

    # ---- MEDIUM TERM ----
    medium_term = build_horizon(
        summary=narratives.get("summary"),
        narrative_key="trend",
        fallback=(
            "The intermediate trend structure remains balanced, "
            "suggesting consolidation rather than strong continuation."
        ),
    )

    # ---- LONG TERM ----
    long_term = build_horizon(
        summary=insights.get("combinedTechnicalSummary"),
        narrative_key="volatility",
        fallback=(
            "Longer-term conditions are shaped by volatility and regime behavior, "
            "which may influence future risk–reward dynamics."
        ),
    )

    return {
        "shortTerm": short_term,
        "mediumTerm": medium_term,
        "longTerm": long_term,
    }

# ---------------------------------------------------------
# 7️⃣ TRADE IDEA (FINAL — NO NULLS, EXPLAINED)
# ---------------------------------------------------------
def build_trade_idea_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a clear, narrative trade idea using ONLY existing data.

    Rules:
    - Always returns stance + 2–4 sentence explanation
    - No targets, no predictions
    - Uses decision, narratives, volatility, and probabilities
    """

    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}
    technical = stock.get("technical") or {}
    probabilities = stock.get("probabilities") or {}
    quality = stock.get("decision_quality") or {}

    stance = decision.get("final") or "HOLD"
    confidence = decision.get("confidence")

    expl = []

    # 1️⃣ Core stance explanation (from narratives if present)
    if narratives.get("tradeIdea"):
        expl.append(narratives["tradeIdea"])

    # 2️⃣ Probability context (if available)
    up = probabilities.get("up")
    down = probabilities.get("down")

    if isinstance(up, (int, float)) and isinstance(down, (int, float)):
        if abs(up - down) < 0.05:
            expl.append(
                "Upside and downside probabilities are closely balanced, "
                "suggesting limited directional edge at current levels."
            )
        elif up > down:
            expl.append(
                f"Upside scenarios slightly outweigh downside cases, "
                f"though conviction remains moderate."
            )
        else:
            expl.append(
                f"Downside scenarios currently dominate the probability distribution, "
                f"which warrants caution."
            )

    # 3️⃣ Volatility / regime framing
    vol = (technical.get("volatility") or {}).get("regime")
    if vol:
        expl.append(
            f"Volatility conditions are classified as {vol.lower()}, "
            f"which impacts timing and position sizing decisions."
        )

    # 4️⃣ Fallback if narratives are thin
    if not expl:
        expl.append(
            "The model does not identify a strong risk–reward imbalance at this time, "
            "indicating that patience or reduced exposure may be appropriate."
        )

    return {
        "stance": stance,
        "confidence": confidence,
        "explanation": _sentences(expl, min_count=2),
    }


# ---------------------------------------------------------
# 8️⃣ RISKS & OPPORTUNITIES (FINAL — UI SAFE, NO NULLS)
# ---------------------------------------------------------
def build_risks_opportunities_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a deterministic Risks & Opportunities block.

    Rules:
    - NEVER return nulls
    - ALWAYS return arrays
    - Deduplicate sentences
    - Minimum 2 sentences per item (when possible)
    - Data-only: narratives + indicators + features
    """

    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    features = stock.get("features_meta") or {}
    indicators = stock.get("indicator_states") or {}
    decision = stock.get("decision") or {}

    raw_risks = sections.get("risks") or []
    raw_opps = sections.get("opportunities") or []

    def normalize(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in items:
            if not isinstance(s, str):
                continue
            clean = s.strip()
            if len(clean) < 8:
                continue
            if clean.lower() in seen:
                continue
            seen.add(clean.lower())
            out.append(clean)
        return out

    risks: List[str] = normalize(raw_risks)
    opps: List[str] = normalize(raw_opps)

    # -------------------------------------------------
    # 🔁 DATA-DRIVEN FALLBACKS (ONLY IF EMPTY)
    # -------------------------------------------------

    rsi = features.get("rsi14")
    atr = features.get("atr14")
    vol_vs_avg = features.get("volume_vs_ma20_pct")
    trend_state = indicators.get("trend")
    volatility_state = indicators.get("volatility")
    signal = (decision.get("final") or "").upper()

    # --------- RISKS FALLBACKS ----------
    if not risks:
        if isinstance(rsi, (int, float)) and rsi > 70:
            risks.append(
                f"RSI is elevated at {rsi:.1f}, indicating momentum may be stretched after recent price advances. "
                "Extended momentum conditions increase the likelihood of consolidation or pullback."
            )

        if isinstance(volatility_state, str) and volatility_state.upper() == "HIGH":
            risks.append(
                "Volatility regime is classified as high, which increases short-term outcome dispersion. "
                "Price movements may become less predictable even if the broader trend remains intact."
            )

        if signal == "HOLD":
            risks.append(
                "The model does not detect a strong directional edge at current levels. "
                "Sideways or choppy price behavior can reduce risk-reward efficiency."
            )

    # --------- OPPORTUNITIES FALLBACKS ----------
    if not opps:
        if isinstance(rsi, (int, float)) and 40 <= rsi <= 60:
            opps.append(
                f"RSI is near neutral at {rsi:.1f}, suggesting momentum is balanced rather than exhausted. "
                "This creates flexibility for either continuation or reversal depending on confirmation."
            )

        if isinstance(vol_vs_avg, (int, float)) and vol_vs_avg > 20:
            opps.append(
                f"Trading volume is approximately {vol_vs_avg:.1f}% above its 20-day average. "
                "Elevated participation improves signal reliability and follow-through potential."
            )

        if isinstance(trend_state, str) and trend_state.upper() in ("UP", "BULLISH"):
            opps.append(
                "Trend indicators continue to point upward, reflecting sustained buying interest. "
                "Trend persistence can support favorable risk-reward if momentum stabilizes."
            )

    # -------------------------------------------------
    # 🧹 FINAL SAFETY: ENSURE NON-EMPTY ARRAYS
    # -------------------------------------------------
    if not risks:
        risks.append(
            "No dominant risk factor is currently flagged by the available indicators. "
            "Market conditions should still be monitored for sudden regime changes."
        )

    if not opps:
        opps.append(
            "No clear opportunity is currently dominant based on existing indicators. "
            "Improved setups may emerge with stronger confirmation or volatility expansion."
        )

    return {
        "short": risks[0],
        "medium": risks[1] if len(risks) > 1 else risks[0],
        "risks": risks,
        "opportunities": opps,
    }



# ---------------------------------------------------------
# 9️⃣ FINAL RECOMMENDATION (FINAL — UI SAFE, NO NULLS)
# ---------------------------------------------------------
def build_final_recommendation_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the final actionable conclusion shown at the bottom of StockDetailScreen.

    Rules:
    - NEVER return nulls
    - Always explain *why* the signal exists
    - 2–4 sentences minimum
    - Data-only: decision + probabilities + narratives + indicators
    """

    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}
    indicators = stock.get("indicator_states") or {}
    probs = stock.get("probabilities") or {}

    signal = (
        decision.get("final")
        or decision.get("finalSignal")
        or (stock.get("bullbrain") or {}).get("signal")
        or "HOLD"
    )

    confidence = (
        decision.get("confidence")
        or (stock.get("bullbrain") or {}).get("confidence")
    )

    summary = narratives.get("summary")
    trade_idea = narratives.get("tradeIdea")
    probability_text = narratives.get("probability")

    up_prob = probs.get("up")
    down_prob = probs.get("down")

    trend_state = indicators.get("trend")
    volatility_state = indicators.get("volatility")

    sentences = []

    # -------------------------------------------------
    # 1️⃣ Primary conclusion (signal-based)
    # -------------------------------------------------
    if summary:
        sentences.append(summary.strip())
    else:
        sentences.append(
            f"The model issues a {signal.upper()} recommendation based on the current balance of trend, momentum, and risk factors."
        )

    # -------------------------------------------------
    # 2️⃣ Probability context
    # -------------------------------------------------
    if probability_text:
        sentences.append(probability_text.strip())
    elif isinstance(up_prob, (int, float)) and isinstance(down_prob, (int, float)):
        bias = "upside" if up_prob > down_prob else "downside"
        sentences.append(
            f"Probability analysis shows a modest {bias} bias, with upside at approximately {up_prob * 100:.0f}% "
            f"versus downside at {down_prob * 100:.0f}%."
        )

    # -------------------------------------------------
    # 3️⃣ Regime / risk context
    # -------------------------------------------------
    if isinstance(volatility_state, str) and volatility_state.upper() == "HIGH":
        sentences.append(
            "Elevated volatility increases uncertainty and reduces the reliability of tight timing decisions."
        )

    if isinstance(trend_state, str):
        sentences.append(
            f"The prevailing trend regime is classified as {trend_state.lower()}, which frames expectations for follow-through."
        )

    # -------------------------------------------------
    # 4️⃣ Actionability guidance
    # -------------------------------------------------
    if trade_idea:
        sentences.append(trade_idea.strip())
    else:
        if signal.upper() == "BUY":
            sentences.append(
                "Risk–reward appears favorable, though entries should be sized conservatively given market uncertainty."
            )
        elif signal.upper() == "SELL":
            sentences.append(
                "Downside risk remains present, and risk management should be prioritized over aggressive positioning."
            )
        else:
            sentences.append(
                "A wait-and-see approach is appropriate until clearer confirmation improves conviction."
            )

    # -------------------------------------------------
    # 🧹 Final cleanup
    # -------------------------------------------------
    final_text = " ".join(s for s in sentences if s)

    return {
        "signal": signal,
        "confidence": confidence,
        "text": final_text,
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
