from typing import Dict, Any


def _pct(v, digits=0):
    if isinstance(v, (int, float)):
        return f"{v * 100:.{digits}f}%"
    return None


def _num(v, digits=1):
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return None


def resolve_watchlist_summary(stock: Dict[str, Any]) -> str:
    quote = stock.get("quote") or {}
    bull = stock.get("bullbrain") or {}
    raw = bull.get("raw") or {}
    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}
    features = stock.get("features_meta") or {}
    technical = stock.get("technical") or {}
    insights = stock.get("insights") or {}

    sym = stock.get("symbol") or quote.get("symbol") or "This stock"

    change_pct = quote.get("changePct")
    signal = bull.get("signal") or "HOLD"
    confidence = bull.get("confidence")
    prob_up = raw.get("prob_up")
    prob_down = raw.get("prob_down")

    rsi = features.get("rsi14")
    volume_vs_ma20 = features.get("volume_vs_ma20_pct")
    price_vs_sma20 = features.get("price_vs_sma20_pct")
    return_5d = features.get("return_5d")

    pattern_name = pattern.get("pattern") or pattern.get("patternLabel")
    pattern_bias = pattern.get("bias") or pattern.get("patternBias")
    pattern_headline = pattern.get("headline")

    days5 = ((history.get("forwardReturns") or {}).get("days5") or {})
    win_rate = days5.get("winRate")
    sample_count = days5.get("count")

    parts = []

    # 1) Price action first — makes every ticker feel different
    if isinstance(change_pct, (int, float)):
        if change_pct >= 2:
            parts.append(f"{sym} is showing strong upside movement today, up {change_pct:.2f}%.")
        elif change_pct >= 0.5:
            parts.append(f"{sym} is trading higher today, up {change_pct:.2f}%.")
        elif change_pct <= -2:
            parts.append(f"{sym} is under selling pressure today, down {abs(change_pct):.2f}%.")
        elif change_pct <= -0.5:
            parts.append(f"{sym} is slightly weaker today, down {abs(change_pct):.2f}%.")
        else:
            parts.append(f"{sym} is mostly flat today, with no strong price direction yet.")

    # 2) Technical condition
    if isinstance(rsi, (int, float)):
        if rsi >= 70:
            parts.append(f"RSI near {rsi:.0f} shows overbought momentum, so pullback risk is higher.")
        elif rsi <= 30:
            parts.append(f"RSI near {rsi:.0f} shows oversold pressure, so a rebound setup may be forming.")
        elif rsi >= 55:
            parts.append(f"RSI near {rsi:.0f} shows buyers still have short-term momentum.")
        elif rsi <= 45:
            parts.append(f"RSI near {rsi:.0f} shows momentum remains soft.")
        else:
            parts.append(f"RSI near {rsi:.0f} shows balanced momentum.")

    if isinstance(price_vs_sma20, (int, float)):
        if price_vs_sma20 >= 3:
            parts.append(f"Price is {price_vs_sma20:.1f}% above the 20-day average, showing short-term strength.")
        elif price_vs_sma20 <= -3:
            parts.append(f"Price is {abs(price_vs_sma20):.1f}% below the 20-day average, keeping trend pressure negative.")

    if isinstance(volume_vs_ma20, (int, float)):
        if volume_vs_ma20 >= 20:
            parts.append(f"Volume is {volume_vs_ma20:.0f}% above average, showing stronger participation.")
        elif volume_vs_ma20 <= -20:
            parts.append(f"Volume is {abs(volume_vs_ma20):.0f}% below average, so conviction is weaker.")

    # 3) AI probability
    if isinstance(prob_up, (int, float)) and isinstance(prob_down, (int, float)):
        if abs(prob_up - prob_down) < 0.08:
            parts.append(f"AI probabilities are close: {prob_up*100:.0f}% upside vs {prob_down*100:.0f}% downside.")
        elif prob_down > prob_up:
            parts.append(f"AI leans cautious with {prob_down*100:.0f}% downside probability.")
        else:
            parts.append(f"AI leans constructive with {prob_up*100:.0f}% upside probability.")

    # 4) Pattern context
    if pattern_name:
        if isinstance(win_rate, (int, float)) and sample_count:
            parts.append(f"{pattern_name} has a {win_rate*100:.0f}% 5-day win rate across {sample_count} samples.")
        elif pattern_headline:
            parts.append(f"{pattern_name}: {pattern_headline}")

    # 5) Fallback from existing insights
    if insights.get("oneLiner"):
        parts.append(insights["oneLiner"])

    # Deduplicate
    clean = []
    seen = set()
    for p in parts:
        if not isinstance(p, str):
            continue
        s = p.strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            clean.append(s)

    return " ".join(clean[:3]) or "Watchlist view is based on price action, AI probability, technical indicators, and current pattern behavior."