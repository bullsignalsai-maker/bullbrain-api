def resolve_watchlist_summary(stock: dict) -> str:
    bull = stock.get("bullbrain") or {}
    raw = bull.get("raw") or {}
    indicators = stock.get("indicator_states") or {}

    signal = bull.get("signal", "HOLD")
    confidence = bull.get("confidence") or 0
    badge = bull.get("confidenceBadge") or "MEDIUM"

    prob_up = raw.get("prob_up")
    prob_down = raw.get("prob_down")

    volatility = indicators.get("volatility_20d")
    volume = indicators.get("volume_vs_ma20_pct")

    pattern = stock.get("pattern") or {}
    pname = pattern.get("pattern") or pattern.get("patternLabel")

    # --------------------------------------------------
    # BUY SIGNAL
    # --------------------------------------------------
    if signal == "BUY":
        tone = (
            "Strong conviction supports participation"
            if badge == "HIGH"
            else "Conditions are constructive but not aggressive"
        )

        return (
            f"{tone}, with upside probability exceeding downside risk. "
            "Trend and momentum alignment favor controlled long exposure."
        )

    # --------------------------------------------------
    # SELL SIGNAL
    # --------------------------------------------------
    if signal == "SELL":
        tone = (
            "Downside pressure is firmly established"
            if badge == "HIGH"
            else "Risk conditions favor defensive positioning"
        )

        return (
            f"{tone}, as downside probability outweighs upside scenarios. "
            "Trend structure and momentum argue against long exposure."
        )

    # --------------------------------------------------
    # HOLD — probability dominance
    # --------------------------------------------------
    if prob_up is not None and prob_down is not None:
        delta = abs(prob_up - prob_down)

        if delta >= 0.12:
            dominant = "upside" if prob_up > prob_down else "downside"
            return (
                f"{dominant.capitalize()} probability currently dominates, "
                "but conviction remains insufficient for directional commitment. "
                "Additional confirmation is required before acting on this imbalance."
            )

    # --------------------------------------------------
    # HOLD — confidence tone
    # --------------------------------------------------
    if badge == "HIGH":
        return (
            "Overall signal confidence is elevated, reflecting strong underlying inputs. "
            "However, conflicting conditions prevent a decisive directional trade."
        )

    if badge == "LOW":
        return (
            "Low confidence reflects unresolved conflict across key indicators. "
            "Market structure does not currently support directional exposure."
        )

    # --------------------------------------------------
    # HOLD — volatility / volume dominance
    # --------------------------------------------------
    if volatility in ("HIGH", "VERY_HIGH"):
        return (
            "Elevated volatility increases outcome dispersion and execution risk. "
            "Waiting for stabilization before directional positioning is advised."
        )

    if volume in ("LOW", "VERY_LOW"):
        return (
            "Subdued volume limits follow-through and conviction. "
            "Directional setups remain fragile under current participation levels."
        )

    # --------------------------------------------------
    # HOLD — pattern as supporting context only
    # --------------------------------------------------
    if pname:
        return (
            f"The current pattern ({pname}) provides contextual insight but limited standalone edge. "
            "Confirmation from trend and momentum indicators remains insufficient."
        )

    # --------------------------------------------------
    # Safe fallback (guaranteed 2 sentences)
    # --------------------------------------------------
    return (
        "Market signals remain mixed with no dominant directional driver. "
        "Waiting for clearer alignment before considering a trade."
    )

