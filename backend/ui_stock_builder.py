# =========================================================
# UI Stock Builder — Stock Detail Contract v1.0 (FINAL)
# Source of truth: Firestore stock document
# NO external calls | NO hallucinated data
# =========================================================

from typing import Dict, Any, List


# ---------------------------------------------------------
# 🔧 Helpers (NORMALIZATION — CRITICAL)
# ---------------------------------------------------------

def _sentences(lines: List[str], min_count: int = 3) -> str:
    clean = []

    seen = set()
    for x in lines:
        if not isinstance(x, str):
            continue

        s = x.strip()
        if not s:
            continue

        key = s.lower()
        if key in seen:
            continue

        seen.add(key)
        clean.append(s)

    return " ".join(clean)

def _get_probabilities(stock: Dict[str, Any]):
    probs = stock.get("probabilities")

    if isinstance(probs, dict):
        up = probs.get("up")
        down = probs.get("down")
        if isinstance(up, (int, float)) and isinstance(down, (int, float)):
            return float(up), float(down)

    raw = (stock.get("bullbrain") or {}).get("raw") or {}

    up = raw.get("prob_up")
    down = raw.get("prob_down")

    if isinstance(up, (int, float)) and isinstance(down, (int, float)):
        return float(up), float(down)

    return None, None


def _get_confidence(stock: Dict[str, Any]):
    decision = stock.get("decision") or {}
    bullbrain = stock.get("bullbrain") or {}

    val = decision.get("confidence")
    if isinstance(val, (int, float)):
        return float(val)

    val = bullbrain.get("confidence")
    if isinstance(val, (int, float)):
        return float(val)

    return None


def _probability_bias(up, down) -> str:
    """
    Shared threshold for probability-derived bias, used by both the signal
    block and the probability block so they can't disagree with each other
    on the same stock. Requires a 5-point edge before calling a direction —
    a razor-thin split (e.g. 50.1%/49.9%) isn't a meaningful lean.
    """
    if not isinstance(up, float) or not isinstance(down, float):
        return "Neutral"

    if abs(up - down) < 0.05:
        return "Neutral"

    return "Bullish" if up > down else "Bearish"
def build_sparkline_from_prices(
    prices: List[float],
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prices = [p for p in prices if isinstance(p, (int, float))]

    if len(prices) < 2:
        return {}

    min_p = min(prices)
    max_p = max(prices)
    span = max_p - min_p or 1

    points = []

    for i, price in enumerate(prices):
        x = round(i * 100 / (len(prices) - 1), 1)
        y = round((max_p - price) * 30 / span, 1)
        points.append(f"{x},{y}")

    out = {
    "path": "M " + " L ".join(points),
    "min": round(min_p, 2),
    "max": round(max_p, 2),
    "direction": "up" if prices[-1] >= prices[0] else "down",
    }

    if isinstance(meta, dict):
        out["range"] = meta.get("range", "1Y")
        out["basis"] = meta.get("basis", "close")
        out["source"] = meta.get("source", "candle_store")
        out["rangeStats"] = {
            "closeLow": meta.get("closeLow"),
            "closeHigh": meta.get("closeHigh"),
            "intradayLow": meta.get("intradayLow"),
            "intradayHigh": meta.get("intradayHigh"),
            "firstClose": meta.get("firstClose"),
            "lastClose": meta.get("lastClose"),
            "returnPct": meta.get("returnPct"),
            "candleCount": meta.get("candleCount"),
        }

    return out


def build_sparkline(
    candles: List[Dict[str, Any]],
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not isinstance(candles, list):
        return {}

    closes = []

    for c in candles:
        if isinstance(c, dict) and isinstance(c.get("close"), (int, float)):
            closes.append(c["close"])

    return build_sparkline_from_prices(closes, meta=meta)

# ---------------------------------------------------------
# 1️⃣ SIGNAL BLOCK (AUTHORITATIVE)
# ---------------------------------------------------------

def build_signal_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}
    display_intelligence = stock.get("displayIntelligence") or {}

    # displayIntelligence (System B) is the single source of truth for the
    # user-facing signal everywhere else in the app (Watchlist, Home,
    # Momentum) — mirror that here instead of showing the raw model call.
    signal = display_intelligence.get("signal") or decision.get("final") or "HOLD"
    label = display_intelligence.get("label")
    confidence = _get_confidence(stock)

    if isinstance(confidence, (int, float)):
        tier = "High" if confidence >= 75 else "Moderate" if confidence >= 60 else "Low"
    else:
        tier = "Low"

    up, down = _get_probabilities(stock)
    probability_bias = _probability_bias(up, down)

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
        "label": label,
        "confidence": confidence,
        "confidenceTier": tier,
        "probabilityBias": probability_bias,
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
    bias = _probability_bias(up, down)

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
    indicators = stock.get("indicator_states") or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    name = (
        pattern.get("pattern")
        or pattern.get("patternLabel")
        or stock.get("patternLabel")
    )

    bias = (
        pattern.get("bias")
        or pattern.get("patternBias")
        or stock.get("patternBias")
    )

    days5 = ((history.get("forwardReturns") or {}).get("days5") or {})

    win_rate = pattern.get("winRate5d")
    if not isinstance(win_rate, (int, float)):
        win_rate = days5.get("winRate")

    best = days5.get("best")
    worst = days5.get("worst")
    avg = days5.get("avg")
    count = days5.get("count")

    pattern_state = indicators.get("pattern_winrate_5d")
    edge_state = indicators.get("pattern_edge_5d")
    sample_state = indicators.get("pattern_sample_count_5d")

    expl = []

    if name:
        expl.append(
            f"The {str(name).replace('_', ' ').title()} pattern has recently emerged, showing the latest short-term price-action structure."
        )
    else:
        expl.append(
            "A short-term price pattern is present, but the pattern name is not available in the current snapshot."
        )

    if bias:
        expl.append(
            f"The stored pattern bias is {str(bias).lower()}, so this setup should be interpreted with that directional context."
        )

    if isinstance(win_rate, (int, float)):
        expl.append(
            f"Historically, this pattern has been favorable about {win_rate * 100:.0f}% of the time over the next five trading days."
        )
    elif pattern_state:
        expl.append(
            f"The pattern win-rate state is {str(pattern_state).replace('_', ' ').lower()}, based on the stored indicator evaluation."
        )
    else:
        expl.append(
            "A reliable five-day win rate is not available in the current pattern history, so confidence should remain limited."
        )

    if isinstance(avg, (int, float)):
        expl.append(
            f"The average five-day forward return for this pattern is about {avg:.2f}%, which gives context for the typical historical follow-through."
        )

    if isinstance(best, (int, float)) and isinstance(worst, (int, float)):
        expl.append(
            f"Past outcomes ranged from approximately {worst:.1f}% on the downside to {best:.1f}% on the upside, showing that results can vary meaningfully."
        )

    if isinstance(count, (int, float)):
        expl.append(
            f"The five-day sample count is {int(count)}, so the pattern is being evaluated against a measurable historical sample."
        )
    elif sample_state:
        expl.append(
            f"The sample quality state is {str(sample_state).replace('_', ' ').lower()}, which helps judge how much weight to give this pattern."
        )

    if edge_state:
        expl.append(
            f"The stored pattern edge is {str(edge_state).replace('_', ' ').lower()}, which summarizes whether the historical edge is positive, negative, or mixed."
        )

    raw_pattern_notes = sections.get("pattern")
    if isinstance(raw_pattern_notes, list):
        expl.extend([x for x in raw_pattern_notes if isinstance(x, str) and x.strip()])
    elif isinstance(raw_pattern_notes, str) and raw_pattern_notes.strip():
        expl.append(raw_pattern_notes.strip())

    if not expl:
        expl.append(
            "Recent price behavior reflects a short-term setup, but the available pattern data is limited."
        )

    return {
        "name": name,
        "bias": bias,
        "winRate5d": win_rate,
        "patternState": pattern_state,
        "edgeState": edge_state,
        "sampleState": sample_state,
        "stats": {
            "avg5d": avg,
            "best5d": best,
            "worst5d": worst,
            "count5d": count,
        },
        "explanation": _sentences(expl, 3),
    }
# ---------------------------------------------------------
# 4️⃣ TECHNICAL SNAPSHOT (SUMMARY, NOT DUMP)
# ---------------------------------------------------------

def build_technical_snapshot_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    technical = stock.get("technical") or {}
    features = stock.get("features_meta") or {}
    indicators = stock.get("indicator_states") or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    trend = technical.get("trend") or {}
    rsi = technical.get("rsi") or {}
    macd = technical.get("macd") or {}
    volatility = technical.get("volatility") or {}
    volume = technical.get("volume") or {}
    price_position = technical.get("pricePosition") or {}

    # -----------------------------
    # Trend explanation
    # -----------------------------
    trend_lines = []

    trend_label = trend.get("label")
    trend_comment = trend.get("comment")
    trend_strength = features.get("trend_strength_20")
    price_vs_sma20 = features.get("price_vs_sma20_pct")
    price_pos_label = price_position.get("label")

    if trend_label:
        trend_lines.append(
            (
                f"Trend is labeled {trend_label}, meaning price is showing strong directional structure."
                if trend_label in ("Strong Uptrend", "Strong Downtrend")
                else f"Trend is labeled {trend_label}, meaning price is trending with a clear directional bias."
                if trend_label in ("Uptrend", "Downtrend")
                else f"Trend is labeled {trend_label}, meaning price is not showing a clean one-direction trend right now."
            )
        )

    if price_pos_label and isinstance(price_vs_sma20, (int, float)):
        trend_lines.append(
            f"Price position is {price_pos_label}, with price about {price_vs_sma20:.1f}% versus the 20-day average."
        )

    if isinstance(trend_strength, (int, float)):
        trend_lines.append(
            f"Trend strength is {trend_strength:.2f}, which helps measure whether the move has enough structure behind it."
        )

    if trend_comment:
        trend_lines.append(trend_comment)

    if isinstance(sections.get("trend"), list):
        trend_lines.extend(sections.get("trend"))

    # -----------------------------
    # Momentum explanation
    # -----------------------------
    momentum_lines = []

    rsi_label = rsi.get("label")
    rsi_value = rsi.get("value")
    rsi_comment = rsi.get("comment")

    macd_label = macd.get("label")
    macd_value = macd.get("value")
    macd_signal = macd.get("signal")
    macd_comment = macd.get("comment")

    if rsi_label and isinstance(rsi_value, (int, float)):
        momentum_lines.append(
            f"RSI is {rsi_label} at {rsi_value:.1f}, showing the current momentum pressure."
        )

    if macd_label and isinstance(macd_value, (int, float)):
        momentum_lines.append(
            f"MACD is {macd_label}, with MACD value {macd_value:.2f} versus signal {macd_signal:.2f}."
            if isinstance(macd_signal, (int, float))
            else f"MACD is {macd_label}, with value {macd_value:.2f}."
        )

    if rsi_comment:
        momentum_lines.append(rsi_comment)

    if macd_comment:
        momentum_lines.append(macd_comment)

    if isinstance(sections.get("momentum"), list):
        momentum_lines.extend(sections.get("momentum"))

    # -----------------------------
    # Volatility explanation
    # -----------------------------
    volatility_lines = []

    vol_label = volatility.get("label")
    vol_value = volatility.get("volatility_20d")
    vol_comment = volatility.get("comment")
    atr14 = features.get("atr14")
    regime = (stock.get("decision") or {}).get("quality", {}).get("regime")

    if vol_label and isinstance(vol_value, (int, float)):
        volatility_lines.append(
            f"Volatility is {vol_label}, with 20-day volatility around {vol_value:.2f}%."
        )

    if isinstance(atr14, (int, float)):
        volatility_lines.append(
            f"ATR(14) is {atr14:.2f}, showing the typical daily price movement in points."
        )

    if regime:
        volatility_lines.append(
            f"Decision regime is {regime}, which affects how reliable short-term signals may be."
        )

    if vol_comment:
        volatility_lines.append(vol_comment)

    if isinstance(sections.get("volatility"), list):
        volatility_lines.extend(sections.get("volatility"))

    # -----------------------------
    # Volume explanation
    # -----------------------------
    volume_lines = []

    volume_label = volume.get("label")
    volume_vs_ma20 = volume.get("volume_vs_ma20_pct")
    volume_comment = volume.get("comment")
    # volume_zscore_20 from features_meta is inflated ~6.66x (see
    # bullbrain_gate_ladder_audit memory). Prefer the corrected value.
    volume_z = stock.get("volume_zscore_20_corrected")
    if volume_z is None:
        volume_z = features.get("volume_zscore_20")

    if volume_label and isinstance(volume_vs_ma20, (int, float)):
        volume_lines.append(
            f"Volume is {volume_label}, trading about {volume_vs_ma20:.1f}% versus the 20-day average."
        )

    if isinstance(volume_z, (int, float)):
        volume_lines.append(
            f"Volume Z-score is {volume_z:.1f}, showing whether participation is unusually high or low compared with recent history."
        )

    if volume_comment:
        volume_lines.append(volume_comment)

    if isinstance(sections.get("volume"), list):
        volume_lines.extend(sections.get("volume"))

    # -----------------------------
    # Section summary
    # -----------------------------
    summary_lines = []

    if trend_label:
        summary_lines.append(f"Trend is {trend_label}, so directional follow-through is not fully confirmed.")

    if rsi_label and macd_label:
        summary_lines.append(f"Momentum is supported by {rsi_label} RSI and {macd_label} MACD readings.")

    if vol_label:
        summary_lines.append(f"Volatility is {vol_label}, which shapes expected price swings.")

    if volume_label:
        summary_lines.append(f"Volume is {volume_label}, showing participation is close to recent behavior.")

    return {
        "summary": _sentences(summary_lines, 3),
        "trend": {
            "label": trend_label,
            "value": trend_strength,
            "priceVsSma20Pct": price_vs_sma20,
            "explanation": _sentences(trend_lines, 3),
        },
        "momentum": {
            "rsiLabel": rsi_label,
            "rsi": rsi_value,
            "macdLabel": macd_label,
            "macd": macd_value,
            "macdSignal": macd_signal,
            "explanation": _sentences(momentum_lines, 3),
        },
        "volatility": {
            "label": vol_label,
            "volatility20d": vol_value,
            "atr14": atr14,
            "regime": regime,
            "explanation": _sentences(volatility_lines, 3),
        },
        "volume": {
            "label": volume_label,
            "volumeVsMa20Pct": volume_vs_ma20,
            "volumeZscore20": volume_z,
            "explanation": _sentences(volume_lines, 3),
        },
    }

# ---------------------------------------------------------
# 5️⃣ FEATURE INSIGHT (REPLACES FEATURE DUMP)
# ---------------------------------------------------------

def build_feature_insight_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    f = stock.get("features_meta") or {}
    indicators = stock.get("indicator_states") or {}

    rsi = f.get("rsi14")
    return_10d = f.get("return_10d")
    return_5d = f.get("return_5d")
    price_vs_sma20 = f.get("price_vs_sma20_pct")
    volume_vs_ma20 = f.get("volume_vs_ma20_pct")
    atr14 = f.get("atr14")

    lines = []

    if isinstance(rsi, (int, float)):
        lines.append(
            f"RSI is {rsi:.1f}, and the stored state is {indicators.get('rsi14', 'UNKNOWN')}."
        )

    if isinstance(price_vs_sma20, (int, float)):
        lines.append(
            f"Price is {price_vs_sma20:.1f}% versus the 20-day average, with state {indicators.get('price_vs_sma20_pct', 'UNKNOWN')}."
        )

    if isinstance(return_10d, (int, float)) and isinstance(return_5d, (int, float)):
        lines.append(
            f"Recent returns show {return_5d:.1f}% over 5 days and {return_10d:.1f}% over 10 days."
        )

    if isinstance(volume_vs_ma20, (int, float)):
        lines.append(
            f"Volume is {volume_vs_ma20:.1f}% versus the 20-day average, with state {indicators.get('volume_vs_ma20_pct', 'UNKNOWN')}."
        )

    if isinstance(atr14, (int, float)):
        lines.append(
            f"ATR(14) is {atr14:.2f}, giving context for normal daily movement."
        )

    if not lines:
        lines.append(
            "Feature signals are mixed, and no single indicator is dominating the setup."
        )

    return {
        "summary": _sentences(lines, 3),
        "highlights": {
            "rsi14": rsi,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "price_vs_sma20_pct": price_vs_sma20,
            "volume_vs_ma20_pct": volume_vs_ma20,
            "atr14": atr14,
        },
    }

# ---------------------------------------------------------
# 6️⃣ OUTLOOK
# ---------------------------------------------------------

def build_outlook_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}
    insights = stock.get("insights") or {}

    trend_lines = sections.get("trend") if isinstance(sections.get("trend"), list) else []
    momentum_lines = sections.get("momentum") if isinstance(sections.get("momentum"), list) else []
    volatility_lines = sections.get("volatility") if isinstance(sections.get("volatility"), list) else []

    short_lines = []
    medium_lines = []
    long_lines = []

    if momentum_lines:
        short_lines.extend(momentum_lines)
    else:
        short_lines.append("Short-term outlook is driven by current momentum and probability conditions.")

    if trend_lines:
        medium_lines.extend(trend_lines)
    else:
        medium_lines.append("Medium-term outlook depends on whether trend structure confirms or remains range-bound.")

    if volatility_lines:
        long_lines.extend(volatility_lines)
    else:
        long_lines.append("Longer-term outlook is shaped by volatility regime and the stability of current signals.")

    return {
        "shortTerm": {
            "summary": insights.get("trendSummary") or short_lines[0],
            "explanation": _sentences(short_lines, 3),
        },
        "mediumTerm": {
            "summary": narratives.get("summary") or medium_lines[0],
            "explanation": _sentences(medium_lines, 3),
        },
        "longTerm": {
            "summary": insights.get("combinedTechnicalSummary") or long_lines[0],
            "explanation": _sentences(long_lines, 3),
        },
    }


# ---------------------------------------------------------
# 7️⃣ RISKS & OPPORTUNITIES
# ---------------------------------------------------------

def build_risks_opportunities_block(stock: Dict[str, Any]) -> Dict[str, Any]:
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}
    features = stock.get("features_meta") or {}

    risks = []
    opportunities = []

    raw_risks = sections.get("risks") or sections.get("risk") or []
    raw_opps = sections.get("opportunities") or sections.get("opportunity") or []

    if isinstance(raw_risks, list):
        risks.extend([x for x in raw_risks if isinstance(x, str) and x.strip()])
    elif isinstance(raw_risks, str) and raw_risks.strip():
        risks.append(raw_risks.strip())

    if isinstance(raw_opps, list):
        opportunities.extend([x for x in raw_opps if isinstance(x, str) and x.strip()])
    elif isinstance(raw_opps, str) and raw_opps.strip():
        opportunities.append(raw_opps.strip())

    rsi = features.get("rsi14")
    volume_vs_ma20 = features.get("volume_vs_ma20_pct")

    up, down = _get_probabilities(stock)

    if not risks:
        if isinstance(up, float) and isinstance(down, float) and down > up:
            risks.append(
                f"Downside probability is higher than upside probability, with downside near {down * 100:.0f}% versus upside near {up * 100:.0f}%."
            )

        if isinstance(rsi, (int, float)) and rsi > 70:
            risks.append(
                f"RSI is elevated at {rsi:.1f}, which may increase pullback or consolidation risk."
            )

        if not risks:
            risks.append(
                "No dominant risk factor is explicitly flagged, but confirmation is still needed before assuming directional follow-through."
            )

    if not opportunities:
        if isinstance(up, float) and isinstance(down, float) and up > down:
            opportunities.append(
                f"Upside probability is higher than downside probability, with upside near {up * 100:.0f}% versus downside near {down * 100:.0f}%."
            )

        if isinstance(volume_vs_ma20, (int, float)) and volume_vs_ma20 > 20:
            opportunities.append(
                f"Volume is {volume_vs_ma20:.1f}% above the 20-day average, showing stronger-than-normal participation."
            )

        if not opportunities:
            opportunities.append(
                "No strong opportunity is dominant yet, but a clearer setup may emerge if momentum, trend, and volume begin to align."
            )

    return {
        "risks": [_sentences([r], 1) for r in risks],
        "opportunities": [_sentences([o], 1) for o in opportunities],
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
    display_intelligence = stock.get("displayIntelligence") or {}

    # raw_signal drives the narrative sentence below (describing the model's
    # own directional call); the returned "signal"/"label" fields mirror
    # displayIntelligence (System B), the app-wide source of truth for the
    # user-facing signal.
    raw_signal = decision.get("final") or "HOLD"
    signal = display_intelligence.get("signal") or raw_signal
    label = display_intelligence.get("label")
    confidence = _get_confidence(stock)

    expl = []
    if narratives.get("summary"):
        expl.append(narratives["summary"])
    else:
        expl.append(
            f"The model maintains a {raw_signal.lower()} stance based on current conditions."
        )

    return {
        "signal": signal,
        "label": label,
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
        "outlook": build_outlook_block(stock),
        "tradeIdea": build_trade_idea_block(stock),
        "risksOpportunities": build_risks_opportunities_block(stock),
        "finalRecommendation": build_final_recommendation_block(stock),
        "news": build_news_block(stock),
        "computed_at": stock.get("computed_at"),
    }


def build_stockdetail_ui_v1(stock: Dict[str, Any]) -> Dict[str, Any]:
    full = build_stockdetail_v1(stock)

    sparkline = full.get("sparkline") or {}
    range_stats = sparkline.get("rangeStats") or {}

    return {
        "signal": full.get("signal"),

        "probability": {
            "up": (full.get("probability") or {}).get("up"),
            "down": (full.get("probability") or {}).get("down"),
            "bias": (full.get("probability") or {}).get("bias"),
        },

        "pattern": {
            "name": (full.get("pattern") or {}).get("name"),
            "bias": (full.get("pattern") or {}).get("bias"),
            "winRate5d": (full.get("pattern") or {}).get("winRate5d"),
            "patternState": (full.get("pattern") or {}).get("patternState"),
            "edgeState": (full.get("pattern") or {}).get("edgeState"),
            "sampleState": (full.get("pattern") or {}).get("sampleState"),

            "stats": {
                "avg5d": (((full.get("pattern") or {}).get("stats") or {}).get("avg5d")),
                "best5d": (((full.get("pattern") or {}).get("stats") or {}).get("best5d")),
                "worst5d": (((full.get("pattern") or {}).get("stats") or {}).get("worst5d")),
                "count5d": (((full.get("pattern") or {}).get("stats") or {}).get("count5d")),
            },
        },

        "technicalSnapshot": full.get("technicalSnapshot"),

        "outlook": {
            "shortTerm": (full.get("outlook") or {}).get("shortTerm"),
            "mediumTerm": (full.get("outlook") or {}).get("mediumTerm"),
            "longTerm": (full.get("outlook") or {}).get("longTerm"),
        },

        "risksOpportunities": full.get("risksOpportunities"),

        "news": [
            {
                "headline": n.get("headline"),
                "summary": n.get("summary"),
                "source": n.get("source"),
                "datetime": n.get("datetime"),
                "url": n.get("url"),
            }
            for n in (full.get("news") or [])
            if isinstance(n, dict)
        ],

        "computed_at": full.get("computed_at"),

        "sparkline": {
            "path": sparkline.get("path"),
            "min": sparkline.get("min"),
            "max": sparkline.get("max"),
            "direction": sparkline.get("direction"),
            "range": sparkline.get("range"),
            "basis": sparkline.get("basis"),

            "rangeStats": {
                "closeLow": range_stats.get("closeLow"),
                "closeHigh": range_stats.get("closeHigh"),
                "returnPct": range_stats.get("returnPct"),
                "candleCount": range_stats.get("candleCount"),
            },
        } if sparkline.get("path") else None,
    }