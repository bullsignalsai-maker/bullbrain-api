# backend/explain/screen_specs.py
# ============================================================
# BullSignalsAI — Screen Indicator Specifications
#
# Purpose:
# - Define WHICH indicators appear on WHICH screen
# - Grouped by logical sections
#
# Rules:
# - NO computation
# - NO text
# - ONLY indicator keys that exist in indicator_states.py
# ============================================================


SCREEN_SPECS = {

    # ========================================================
    # HOMESCREEN (very compact)
    # ========================================================
    "HOMESCREEN": {
        # 2-line signal explanation should come from:
        # - summary (trend/momentum) + action_blocker (why HOLD)
        "signal": [
            "action_blocker",
        ],
        "momentum": [
            "rsi14",
            "macd_hist",
        ],
        "trend": [
            "price_vs_sma20_pct",
            "trend_strength_20",
        ],
        "volatility": [
            "volatility_20d",
        ],
        "probability": [
            "hybrid_prob_up",
        ],
        "pattern": [
            "pattern_edge_5d",
        ],
    },


    # ========================================================
    # WATCHLIST
    # ========================================================
    "WATCHLIST": {
        "signal": [
            "action_blocker",
        ],

        "momentum": [
            "return_1d",
            "rsi14",
            "macd_hist",
        ],
        "trend": [
            "price_vs_sma20_pct",
        ],
        "volume": [
            "volume_zscore_20",
        ],
        "probability": [
            "hybrid_prob_up",
        ],
    },

    # ========================================================
    # STOCK DETAIL (main screen)
    # ========================================================
    "STOCK_DETAIL": {
        "signal": [
            "action_blocker",
        ],
        "trend": [
            "trend_strength_20",
            "price_vs_sma20_pct",
            "sma5_sma20_pct",
            "sma20_sma50_pct",
        ],
        "momentum": [
            "rsi14",
            "macd_hist",
        ],
        "volume": [
            "volume_zscore_20",
            "volume_vs_ma20_pct",
        ],
        "volatility": [
            "volatility_20d",
            "atr14",
        ],
        "candle": [
            "gap_pct",
            "body_pct",
            "upper_shadow_pct",
            "lower_shadow_pct",
            "intraday_range_pct",
        ],
        "probability": [
            "hybrid_prob_up",
        ],
        "regime": [
            "liquidity_quality",
            "regime_state",
        ],
        "pattern": [
            "pattern_edge_5d",
            "pattern_winrate_5d",
            "pattern_avg_5d",
            "pattern_sample_count_5d",
        ],
    },

    # ========================================================
    # FULL SIGNAL DETAILS
    # ========================================================
    "FULL_SIGNAL_DETAILS": {
        "signal": [
            "action_blocker",
        ],
        "probability": [
            "hybrid_prob_up",
            "model_prob_up",
            "model_confidence",
        ],
        "regime": [
            "liquidity_quality",
            "regime_state",
        ],
    },


    # ========================================================
    # FULL PATTERN DETAILS
    # ========================================================
    "FULL_PATTERN_DETAILS": {
        "pattern": [
            "pattern_edge_5d",
            "pattern_winrate_5d",
            "pattern_avg_5d",
            "pattern_sample_count_5d",
            "pattern_occurrences",
        ],
        "regime": [
            "regime_state",
        ],
    },

    # ========================================================
    # FULL TECHNICAL DETAILS
    # ========================================================
    "FULL_TECHNICAL_DETAILS": {
        "trend": [
            "trend_strength_20",
            "price_vs_sma20_pct",
            "sma5_sma20_pct",
            "sma20_sma50_pct",
        ],
        "momentum": [
            "rsi14",
            "macd_hist",
            "williams_r_14",
            "stoch_k_14",
            "stoch_d_3",
        ],
        "volume": [
            "volume_zscore_20",
            "volume_vs_ma20_pct",
            "obv_slope_10",
        ],
        "volatility": [
            "volatility_20d",
            "atr14",
            "intraday_range_pct",
        ],
    },

    # ========================================================
    # FULL CANDLE DETAILS
    # ========================================================
    "FULL_CANDLE_DETAILS": {
        "candle": [
            "gap_pct",
            "body_pct",
            "upper_shadow_pct",
            "lower_shadow_pct",
            "true_range",
            "atr14",
            "intraday_range_pct",
        ],
    },

    # ========================================================
    # PORTFOLIO
    # ========================================================
    "PORTFOLIO": {
        "trend": [
            "trend_strength_20",
        ],
        "momentum": [
            "return_5d",
        ],
        "volatility": [
            "volatility_20d",
        ],
        "probability": [
            "hybrid_prob_up",
        ],
    },

    # ========================================================
    # MARKET MOVERS / GAINERS / LOSERS
    # ========================================================
    "MARKET_MOVERS": {
        "momentum": [
            "return_1d",
            "return_5d",
        ],
        "volume": [
            "volume_zscore_20",
        ],
        "volatility": [
            "volatility_20d",
        ],
    },
}
