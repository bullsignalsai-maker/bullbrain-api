# backend/explain/indicator_templates.py
# ============================================================
# BullSignalsAI — Indicator Narration Templates
#
# Tone: Institutional trading desk
# Style: Objective, professional, risk-aware
#
# Rules:
# - NO computation
# - NO randomness
# - NO Firestore access
# - ONLY deterministic text templates
#
# Mapping:
#   indicator -> state -> list[str]
#
# The narrative engine will select later.
# ============================================================


INDICATOR_TEMPLATES = {

    # ========================================================
    # MOMENTUM — RSI (rsi14)
    # ========================================================
    "rsi14": {
        "EXTREMELY_OVERSOLD": [
            "RSI is deeply oversold, indicating selling pressure has reached an extreme level. While reflexive bounces can occur, this alone does not confirm a durable reversal.",
            "Momentum is severely compressed to the downside. Historically, such conditions often attract short-covering, but trend confirmation remains critical."
        ],
        "OVERSOLD": [
            "RSI is in oversold territory, indicating downside momentum may be stretched and selling pressure is elevated.",
            "While oversold conditions can support short-term stabilization, they do not by themselves confirm a durable reversal."
        ],

        "BEARISH": [
            "RSI remains below neutral, reflecting bearish momentum conditions.",
            "Momentum continues to favor sellers, with limited evidence of accumulation."
        ],
        "NEUTRAL": [
            "RSI is in a neutral range, suggesting balanced momentum between buyers and sellers.",
            "Momentum indicators do not currently provide a directional edge."
        ],
        "BULLISH": [
            "RSI is above neutral, signaling improving upside momentum.",
            "Momentum conditions favor buyers, though confirmation from trend structure is still important."
        ],
        "OVERBOUGHT": [
            "RSI is overbought, indicating upside momentum is becoming stretched.",
            "Risk of consolidation or pullback typically increases in this zone."
        ],
        "EXTREMELY_OVERBOUGHT": [
            "RSI is extremely overbought, suggesting momentum is vulnerable to mean reversion.",
            "Upside continuation becomes statistically less efficient at these levels."
        ],
        "UNKNOWN": [
            "RSI data is unavailable or unreliable, limiting momentum assessment."
        ]
    },

    # ========================================================
    # MOMENTUM — MACD HISTOGRAM (macd_hist)
    # ========================================================
    "macd_hist": {
        "STRONG_BULLISH": [
            "MACD momentum is strongly positive, indicating accelerating upside pressure.",
            "Bullish momentum expansion often aligns with sustained trend continuation."
        ],
        "BULLISH": [
            "MACD histogram is positive, reflecting constructive upside momentum.",
            "Momentum favors higher prices, though acceleration is moderate."
        ],
        "MILD_BULLISH": [
            "MACD momentum is marginally positive, suggesting early or weakening upside pressure.",
            "Upside momentum exists but lacks strong conviction."
        ],
        "FLAT": [
            "MACD momentum is flat, indicating a lack of directional impulse.",
            "Such conditions often precede consolidation."
        ],
        "MILD_BEARISH": [
            "MACD momentum has turned slightly negative, signaling early downside pressure.",
            "Momentum deterioration is present but not yet decisive."
        ],
        "BEARISH": [
            "MACD histogram is negative, indicating bearish momentum conditions.",
            "Downside momentum currently favors sellers."
        ],
        "STRONG_BEARISH": [
            "MACD momentum is strongly negative, reflecting accelerating downside pressure.",
            "This regime is often associated with failed rallies and sustained declines."
        ],
        "UNKNOWN": [
            "MACD momentum data is unavailable."
        ]
    },

    # ========================================================
    # TREND — TREND STRENGTH (trend_strength_20)
    # ========================================================
    "trend_strength_20": {
        "STRONG_UPTREND": [
            "Trend strength signals a strong uptrend, with price advancing persistently.",
            "Market structure strongly favors trend-following strategies."
        ],
        "UPTREND": [
            "Trend conditions are positive, suggesting sustained upside bias.",
            "Pullbacks in this regime often find demand."
        ],
        "MILD_UP": [
            "Trend strength is mildly positive, indicating early or weakening upside structure.",
            "Directional edge exists but remains fragile."
        ],
        "SIDEWAYS": [
            "Trend strength is neutral, consistent with a range-bound environment.",
            "Directional trades tend to have reduced edge in sideways regimes."
        ],
        "MILD_DOWN": [
            "Trend strength is mildly negative, reflecting developing downside pressure.",
            "Bearish structure is emerging but not yet dominant."
        ],
        "DOWNTREND": [
            "Trend conditions favor the downside, with sellers controlling structure.",
            "Rallies are more likely to encounter resistance."
        ],
        "STRONG_DOWNTREND": [
            "Trend strength indicates a strong downtrend, with persistent selling pressure.",
            "Countertrend trades in this regime typically carry elevated risk."
        ],
        "UNKNOWN": [
            "Trend strength data is unavailable."
        ]
    },

    # ========================================================
    # PRICE POSITION — PRICE VS SMA20 (price_vs_sma20_pct)
    # ========================================================
    "price_vs_sma20_pct": {
        "FAR_ABOVE_TREND": [
            "Price is significantly extended above its short-term average, indicating stretched upside conditions.",
            "Such extensions often precede consolidation or mean reversion."
        ],
        "ABOVE_TREND": [
            "Price is trading above its short-term average, supporting bullish structure.",
            "Trend-following conditions remain constructive."
        ],
        "SLIGHTLY_ABOVE": [
            "Price is modestly above trend, reflecting mild upside bias.",
            "Structure remains supportive but not extended."
        ],
        "AT_TREND": [
            "Price is trading near its short-term average, reflecting equilibrium.",
            "This area often serves as a decision point for direction."
        ],
        "SLIGHTLY_BELOW": [
            "Price is slightly below trend, indicating mild weakness.",
            "Upside attempts may require stronger momentum to succeed."
        ],
        "BELOW_TREND": [
            "Price is below its short-term average, signaling bearish structure.",
            "Rallies may face resistance near trend levels."
        ],
        "FAR_BELOW_TREND": [
            "Price is significantly below trend, indicating pronounced weakness.",
            "Downside risk remains elevated in this configuration."
        ],
        "UNKNOWN": [
            "Trend positioning data is unavailable."
        ]
    },

    # ========================================================
    # VOLATILITY — VOLATILITY 20D (volatility_20d)
    # ========================================================
    "volatility_20d": {
        "LOW_VOL": [
            "Volatility is compressed, suggesting price is coiling rather than trending aggressively.",
            "Low volatility environments often precede directional expansion."
        ],
        "NORMAL_VOL": [
            "Volatility is within normal ranges, supporting more reliable signal interpretation.",
            "Price movement remains orderly."
        ],
        "ELEVATED_VOL": [
            "Volatility is elevated, increasing risk and reducing signal reliability.",
            "Larger price swings require disciplined risk management."
        ],
        "HIGH_VOL": [
            "Volatility is high, indicating unstable market conditions.",
            "Directional signals in this environment tend to degrade quickly."
        ],
        "UNKNOWN": [
            "Volatility data is unavailable."
        ]
    },

    # ========================================================
    # VOLUME — VOLUME VS MA20 (volume_vs_ma20_pct)
    # ========================================================
    "volume_vs_ma20_pct": {
        "FAR_ABOVE_AVG": [
            "Trading volume is significantly above average, suggesting strong institutional participation.",
            "High participation increases the credibility of recent price action."
        ],
        "ABOVE_AVG": [
            "Volume is above normal, supporting the validity of recent moves.",
            "Participation conditions are constructive."
        ],
        "AROUND_AVG": [
            "Volume is near average, indicating balanced participation.",
            "Price movement lacks a strong volume tailwind."
        ],
        "BELOW_AVG": [
            "Volume is below average, reducing confidence in directional moves.",
            "Thin participation increases the risk of false signals."
        ],
        "FAR_BELOW_AVG": [
            "Volume is significantly below normal, indicating weak market engagement.",
            "Signals generated in low-volume environments tend to be less reliable."
        ],
        "UNKNOWN": [
            "Volume comparison data is unavailable."
        ]
    },

    # ========================================================
    # PROBABILITY — HYBRID PROBABILITY UP (hybrid_prob_up)
    # ========================================================
    "hybrid_prob_up": {
        "VERY_HIGH": [
            "The model assigns a very high probability to upside outcomes, indicating strong alignment across inputs.",
            "Upside scenarios dominate the current probabilistic outlook."
        ],
        "HIGH": [
            "Upside probability is high, suggesting favorable conditions for bullish outcomes.",
            "Model confidence leans decisively toward upside scenarios."
        ],
        "LEAN_HIGH": [
            "Upside probability moderately exceeds downside risk.",
            "An edge exists, though it is not overwhelming."
        ],
        "BALANCED": [
            "Upside and downside probabilities are closely balanced.",
            "The model does not identify a strong directional advantage."
        ],
        "LEAN_LOW": [
            "Upside probability is below average, with downside scenarios carrying greater statistical weight.",
            "Risk-reward conditions are not favorable enough to justify aggressive bullish positioning."
        ],

        "LOW": [
            "Upside probability is low, with downside scenarios clearly dominating the probabilistic distribution.",
            "This probability skew argues for caution rather than directional exposure."
        ],

        "VERY_LOW": [
            "Upside probability is extremely low, suggesting strong downside bias.",
            "Bullish trades in this regime carry elevated risk."
        ],
        "UNKNOWN": [
            "Hybrid probability data is unavailable."
        ]
    },
    # ========================================================
    # ACTION BLOCKER — HOLD JUSTIFICATION (action_blocker)
    # ========================================================
    "action_blocker": {
        "DOWNSIDE_DOMINANT": [
            "Downside risk currently outweighs upside potential, limiting the attractiveness of new positions.",
            "Market conditions favor capital preservation over directional exposure."
        ],
        "LIQUIDITY_CONSTRAINED": [
            "Liquidity conditions are thin, reducing execution quality and increasing noise risk.",
            "Such environments typically warrant patience rather than active positioning."
        ],
        "NO_BLOCKER": [
            "No dominant structural blocker is currently present."
        ],
        "UNKNOWN": [
            "Structural constraints on action cannot be fully assessed."
        ]
    },


    # ========================================================
    # PATTERN — EDGE QUALITY (pattern_edge_5d)
    # ========================================================
    "pattern_edge_5d": {
        "POSITIVE_EDGE": [
            "Historical samples show a favorable win rate and positive average returns following this pattern.",
            "Pattern statistics suggest a measurable edge when aligned with broader trend conditions."
        ],
        "LEAN_POSITIVE": [
            "Pattern outcomes lean positive, though the edge is modest.",
            "Effectiveness improves when confirmed by trend and momentum alignment."
        ],
        "MIXED_EDGE": [
            "Pattern outcomes have been mixed, with no consistent directional advantage.",
            "Reliance on this pattern alone offers limited predictive value."
        ],
        "NEGATIVE_EDGE": [
            "Historical outcomes following this pattern skew negative.",
            "Pattern statistics caution against assuming upside follow-through."
        ],
        "INSUFFICIENT_SAMPLES": [
            "Historical sample size is limited, reducing statistical confidence.",
            "Pattern reliability cannot be firmly established."
        ],
        "UNKNOWN": [
            "Pattern edge statistics are unavailable."
        ]
    },

    # ========================================================
    # MOMENTUM — COMPOSITE (momentum_composite)
    # ========================================================
    "momentum_composite": {
        "MOMENTUM_BULLISH": [
            "Momentum conditions are aligned to the upside, with indicators reinforcing bullish continuation."
        ],
        "MOMENTUM_BEARISH": [
            "Momentum conditions are aligned to the downside, reflecting sustained selling pressure."
        ],
        "MOMENTUM_BULL_STRETCHED": [
            "Upside momentum remains positive but increasingly stretched, reducing follow-through efficiency."
        ],
        "MOMENTUM_BEAR_STRETCHED": [
            "Downside momentum is extended, increasing the risk of reflexive stabilization."
        ],
        "MOMENTUM_MIXED": [
            "Momentum signals are mixed, limiting directional conviction."
        ],
        "UNKNOWN": [
            "Momentum alignment cannot be fully assessed."
        ]
    },

    # ========================================================
    # PROBABILITY — COMPOSITE (probability_composite)
    # ========================================================
    "probability_composite": {
        "PROB_STRONGLY_UP": [
            "Upside scenarios dominate the probability distribution with strong conviction."
        ],
        "PROB_WEAKLY_UP": [
            "Upside probability holds a modest edge over downside risk."
        ],
        "PROB_BALANCED": [
            "Upside and downside probabilities remain closely balanced."
        ],
        "PROB_WEAKLY_DOWN": [
            "Downside probability modestly outweighs upside scenarios."
        ],
        "PROB_STRONGLY_DOWN": [
            "Downside scenarios dominate the probability distribution."
        ],
        "UNKNOWN": [
            "Probability alignment cannot be fully determined."
        ]
    },

    # ========================================================
    # VOLATILITY — COMPOSITE (volatility_composite)
    # ========================================================
    "volatility_composite": {
        "VOLATILITY_EXPANDING": [
            "Volatility is expanding, increasing outcome dispersion and reducing signal reliability."
        ],
        "VOLATILITY_CONTRACTING": [
            "Volatility is compressed, often preceding directional expansion."
        ],
        "VOLATILITY_NORMAL": [
            "Volatility conditions are stable, supporting clearer signal interpretation."
        ],
        "UNKNOWN": [
            "Volatility regime cannot be clearly classified."
        ]
    },


}
