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
    # HOLD — probability imbalance (primary HOLD reason)
    # ==================================================
    if prob_up is not None and prob_down is not None:
        delta = abs(prob_up - prob_down)
        if delta >= 0.12:
            dominant = "upside" if prob_up > prob_down else "downside"
            return (
                f"{dominant.capitalize()} probability currently dominates, "
                "but conviction remains insufficient for directional commitment."
            )

    # ==================================================
    # HOLD — trend vs momentum conflict
    # ==================================================
    if trend in ("STRONG_DOWNTREND", "DOWNTREND") and momentum in (
        "MOMENTUM_BULLISH",
        "MOMENTUM_BULL_STRETCHED",
    ):
        return (
            "Momentum stabilization contrasts with a still-weak trend backdrop. "
            "This divergence reduces the reliability of early directional entries."
        )

    # ==================================================
    # HOLD — volatility regime
    # ==================================================
    if volatility == "VOLATILITY_EXPANDING":
        return (
            "Expanding volatility increases outcome dispersion and execution risk. "
            "Directional positioning is less reliable under current conditions."
        )

    # ==================================================
    # HOLD — participation / liquidity
    # ==================================================
    if liquidity in ("THIN", "POOR"):
        return (
            "Thin liquidity limits follow-through and signal reliability. "
            "Improved participation would be needed to support conviction."
        )

    if volume in ("LOW", "VERY_LOW"):
        return (
            "Subdued volume reduces confirmation behind recent price moves. "
            "Directional setups remain fragile without stronger participation."
        )

    # ==================================================
    # HOLD — pattern-only context
    # ==================================================
    if pname:
        return (
            f"The current pattern ({pname}) provides contextual insight but limited standalone edge. "
            "Broader trend and probability alignment remain inconclusive."
        )

    # ==================================================
    # Fallback (guaranteed, non-generic)
    # ==================================================
    return (
        "Market inputs remain mixed with no dominant directional driver. "
        "Waiting for clearer alignment before committing capital remains prudent."
    )
