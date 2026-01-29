def resolve_watchlist_summary(stock: dict) -> str:
    bull = stock.get("bullbrain") or {}
    raw = bull.get("raw") or {}
    states = stock.get("indicator_states") or {}

    signal = bull.get("signal", "HOLD")
    confidence = float(bull.get("confidence") or 0)
    badge = bull.get("confidenceBadge") or "MEDIUM"

    prob_up = raw.get("prob_up")
    prob_down = raw.get("prob_down")

    trend = states.get("trend_strength_20")
    momentum = states.get("momentum_composite")
    volatility = states.get("volatility_composite")
    volume = states.get("volume_vs_ma20_pct")
    liquidity = states.get("liquidity_quality")

    pattern = stock.get("pattern") or {}
    pname = pattern.get("pattern") or pattern.get("patternLabel")

    # ==================================================
    # BUY
    # ==================================================
    if signal == "BUY":
        conviction = (
            "Strong signal conviction"
            if badge == "HIGH"
            else "Constructive conditions"
        )

        trend_clause = (
            "trend structure is firmly supportive"
            if trend in ("STRONG_UPTREND", "UPTREND")
            else "trend structure is improving"
        )

        return (
            f"{conviction} supports upside participation as {trend_clause}. "
            "Probability and momentum alignment favor controlled long exposure."
        )

    # ==================================================
    # SELL
    # ==================================================
    if signal == "SELL":
        pressure = (
            "Downside pressure is firmly established"
            if badge == "HIGH"
            else "Risk conditions remain skewed to the downside"
        )

        trend_clause = (
            "trend deterioration is accelerating"
            if trend in ("STRONG_DOWNTREND", "DOWNTREND")
            else "trend support has weakened"
        )

        return (
            f"{pressure}, with downside scenarios dominating. "
            f"Current structure suggests {trend_clause} rather than stabilization."
        )

    # ==================================================
    # HOLD — layered explanation (primary + qualifier)
    # ==================================================

    primary = None
    secondary = None

    # 1️⃣ Primary HOLD reason (dominant)
    if prob_up is not None and prob_down is not None:
        delta = abs(prob_up - prob_down)
        if delta >= 0.12:
            primary = (
                "Upside probability currently dominates"
                if prob_up > prob_down
                else "Downside probability currently dominates"
            )

    if not primary and volatility == "VOLATILITY_EXPANDING":
        primary = "Expanding volatility increases outcome dispersion and execution risk"

    if not primary and liquidity in ("THIN", "POOR"):
        primary = "Thin liquidity reduces signal reliability"

    # 2️⃣ Secondary qualifier (symbol-specific color)
    if trend in ("STRONG_DOWNTREND", "DOWNTREND"):
        secondary = "trend structure remains unsupportive of sustained upside"

    elif momentum in ("MOMENTUM_BULL_STRETCHED", "MOMENTUM_BEAR_STRETCHED"):
        secondary = "momentum conditions appear stretched and unstable"

    elif volume in ("LOW", "VERY_LOW"):
        secondary = "participation remains too weak to confirm directional intent"

    elif pname:
        secondary = f"the current {pname.lower()} pattern offers context but limited edge"

    # 3️⃣ Final assembly (guaranteed 2 sentences)
    if primary:
        if secondary:
            return f"{primary}. {secondary.capitalize()}."
        return f"{primary}. Conviction remains insufficient for directional commitment."

    # --------------------------------------------------
    # Safe fallback (guaranteed 2 sentences)
    # --------------------------------------------------
    return (
        "Market inputs remain mixed with no dominant directional driver. "
        "Waiting for clearer alignment before committing capital remains prudent."
    )

