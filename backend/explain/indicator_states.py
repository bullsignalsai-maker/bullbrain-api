# backend/explain/indicator_states.py
# ============================================================
# BullSignalsAI — Indicator States (DETERMINISTIC + PURE)
#
# ✅ Pure functions only (no Firestore, no network, no time I/O)
# ✅ Reusable across cron + API
# ✅ Outputs STRING states only (per your choice A)
#
# Goal:
#   Raw numeric inputs -> normalized semantic "states"
#   (Used later by templates/narrative engine to generate text)
#
# Coverage:
#   - 48 BullBrain engineered features (BULLBRAIN_FEATURES)
#   - +20 display-layer indicators used across screens (quote/decision/pattern)
#   = 68 total
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math


# ------------------------------------------------------------
# 68 INDICATORS (authoritative list)
# ------------------------------------------------------------
BULLBRAIN_FEATURES_48 = [
    "adj_close",
    "close",
    "high",
    "low",
    "open",
    "volume",
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "sma5",
    "sma10",
    "sma20",
    "sma50",
    "sma200",
    "sma5_sma20_pct",
    "sma20_sma50_pct",
    "price_vs_sma20_pct",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema12",
    "ema26",
    "ema_ratio",
    "williams_r_14",
    "stoch_k_14",
    "stoch_d_3",
    "volume_change_1d",
    "volume_ma5",
    "volume_ma20",
    "volume_vs_ma5_pct",
    "volume_vs_ma20_pct",
    "obv",
    "obv_slope_10",
    "intraday_range_pct",
    "true_range",
    "atr14",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "gap_pct",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "volume_zscore_20",
    "trend_strength_20",
]

DISPLAY_LAYER_20 = [
    # Quote-level
    "quote_change_pct",
    "quote_change_abs",
    "quote_gap_pct",          # derived from open vs prevClose if available
    "quote_range_pct",        # derived from high/low vs price if available
    "quote_volume_vs_ma20",   # derived using features if both exist

    # Model probabilities + confidence
    "model_prob_up",
    "model_prob_down",
    "model_confidence",

    # Hybrid / decision probability
    "hybrid_prob_up",
    "hybrid_prob_down",
    "decision_bias_strength",

    # Decision ladder / quality
    "liquidity_quality",      # state based on volume_zscore/vol_vs_ma20/intraday_range
    "regime_state",           # uses your regime thresholds

    # Pattern stats (when present)
    "pattern_winrate_5d",
    "pattern_avg_5d",
    "pattern_sample_count_5d",
    "pattern_occurrences",
    "pattern_edge_5d",        # winrate & avg combined

    # Freshness (when present)
    "freshness_minutes_ago",
    "freshness_state",
]

ALL_68_INDICATORS = BULLBRAIN_FEATURES_48 + DISPLAY_LAYER_20


# ------------------------------------------------------------
# Helpers (pure)
# ------------------------------------------------------------
def _is_num(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    return False


def _to_float(x: Any) -> Optional[float]:
    if not _is_num(x):
        return None
    return float(x)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get(obj: Any, path: str) -> Any:
    """
    Safe dot-path getter. Pure.
    Example:
      _get(payload, "quote.price")
      _get(payload, "patternHistory.forwardReturns.days5.winRate")
    """
    if obj is None:
        return None
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _pct(x: Optional[float]) -> Optional[float]:
    """Convert decimal to percent if user accidentally passes 0.12 meaning 12%.
    We DO NOT auto-convert generally; used only in a few display derivations."""
    if x is None:
        return None
    return x * 100.0


# ------------------------------------------------------------
# Core state bucketizers (pure)
# ------------------------------------------------------------
def _state_rsi(rsi: Optional[float]) -> str:
    if rsi is None:
        return "UNKNOWN"
    # Extra granularity for your “institutional” feel
    if rsi <= 10:
        return "EXTREMELY_OVERSOLD"
    if rsi < 30:
        return "OVERSOLD"
    if rsi < 45:
        return "BEARISH"
    if rsi <= 55:
        return "NEUTRAL"
    if rsi <= 70:
        return "BULLISH"
    if rsi <= 85:
        return "OVERBOUGHT"
    return "EXTREMELY_OVERBOUGHT"


def _state_macd_hist(h: Optional[float]) -> str:
    if h is None:
        return "UNKNOWN"
    # magnitude tiers (MACD histogram varies by price scale)
    if h >= 2.0:
        return "STRONG_BULLISH"
    if h >= 0.5:
        return "BULLISH"
    if h > 0.0:
        return "MILD_BULLISH"
    if h <= -2.0:
        return "STRONG_BEARISH"
    if h <= -0.5:
        return "BEARISH"
    if h < 0.0:
        return "MILD_BEARISH"
    return "FLAT"


def _state_trend_strength(t: Optional[float]) -> str:
    if t is None:
        return "UNKNOWN"
    # Align with your regime logic (abs(trend) > 0.4 -> TRENDING)
    if t >= 0.8:
        return "STRONG_UPTREND"
    if t >= 0.4:
        return "UPTREND"
    if t <= -0.8:
        return "STRONG_DOWNTREND"
    if t <= -0.4:
        return "DOWNTREND"
    # inside regime “ranging” band
    if t >= 0.15:
        return "MILD_UP"
    if t <= -0.15:
        return "MILD_DOWN"
    return "SIDEWAYS"


def _state_price_vs_sma20(p: Optional[float]) -> str:
    if p is None:
        return "UNKNOWN"
    if p >= 8.0:
        return "FAR_ABOVE_TREND"
    if p >= 3.0:
        return "ABOVE_TREND"
    if p > 1.0:
        return "SLIGHTLY_ABOVE"
    if p >= -1.0:
        return "AT_TREND"
    if p >= -3.0:
        return "SLIGHTLY_BELOW"
    if p >= -8.0:
        return "BELOW_TREND"
    return "FAR_BELOW_TREND"


def _state_ma_spread(p: Optional[float]) -> str:
    """Used for sma5_sma20_pct and sma20_sma50_pct."""
    if p is None:
        return "UNKNOWN"
    if p >= 3.0:
        return "BULLISH_STACK"
    if p >= 0.8:
        return "BULLISH_LEAN"
    if p >= -0.8:
        return "FLAT_STACK"
    if p >= -3.0:
        return "BEARISH_LEAN"
    return "BEARISH_STACK"


def _state_stoch(x: Optional[float]) -> str:
    if x is None:
        return "UNKNOWN"
    if x <= 5:
        return "EXTREMELY_OVERSOLD"
    if x < 20:
        return "OVERSOLD"
    if x <= 80:
        return "NEUTRAL"
    if x < 95:
        return "OVERBOUGHT"
    return "EXTREMELY_OVERBOUGHT"


def _state_williams_r(x: Optional[float]) -> str:
    # Williams %R is typically [-100, 0]
    if x is None:
        return "UNKNOWN"
    if x <= -95:
        return "EXTREMELY_OVERSOLD"
    if x < -80:
        return "OVERSOLD"
    if x <= -20:
        return "NEUTRAL"
    if x < -5:
        return "OVERBOUGHT"
    return "EXTREMELY_OVERBOUGHT"


def _state_volatility(vol20: Optional[float]) -> str:
    if vol20 is None:
        return "UNKNOWN"
    # Match your existing interpretation
    if vol20 < 1.0:
        return "LOW_VOL"
    if vol20 < 2.5:
        return "NORMAL_VOL"
    if vol20 < 4.0:
        return "ELEVATED_VOL"
    return "HIGH_VOL"


def _state_intraday_range(x: Optional[float]) -> str:
    if x is None:
        return "UNKNOWN"
    if x < 1.0:
        return "TIGHT"
    if x < 2.0:
        return "NORMAL"
    if x < 4.0:
        return "WIDE"
    if x < 6.0:
        return "VERY_WIDE"
    return "EXTREME"


def _state_gap(g: Optional[float]) -> str:
    if g is None:
        return "UNKNOWN"
    if g >= 2.0:
        return "BIG_GAP_UP"
    if g >= 0.7:
        return "GAP_UP"
    if g <= -2.0:
        return "BIG_GAP_DOWN"
    if g <= -0.7:
        return "GAP_DOWN"
    return "FLAT"


def _state_volume_z(z: Optional[float]) -> str:
    if z is None:
        return "UNKNOWN"
    if z >= 3.0:
        return "INSTITUTIONAL_SPIKE"
    if z >= 2.0:
        return "VOLUME_SPIKE"
    if z >= 1.0:
        return "ELEVATED"
    if z > -1.0:
        return "NORMAL"
    if z > -2.0:
        return "LOW"
    return "VERY_LOW"


def _state_volume_vs_ma(p: Optional[float]) -> str:
    if p is None:
        return "UNKNOWN"
    if p >= 30:
        return "FAR_ABOVE_AVG"
    if p >= 10:
        return "ABOVE_AVG"
    if p >= -10:
        return "AROUND_AVG"
    if p >= -30:
        return "BELOW_AVG"
    return "FAR_BELOW_AVG"


def _state_obv_slope(s: Optional[float]) -> str:
    if s is None:
        return "UNKNOWN"
    # OBV slope magnitude can be huge; focus on sign + relative tiers
    if s >= 5_000_000:
        return "STRONG_ACCUMULATION"
    if s > 0:
        return "ACCUMULATION"
    if s <= -5_000_000:
        return "STRONG_DISTRIBUTION"
    if s < 0:
        return "DISTRIBUTION"
    return "FLAT"


def _state_return(r: Optional[float]) -> str:
    if r is None:
        return "UNKNOWN"
    if r >= 5.0:
        return "BIG_UP"
    if r >= 1.5:
        return "UP"
    if r >= 0.3:
        return "MILD_UP"
    if r <= -5.0:
        return "BIG_DOWN"
    if r <= -1.5:
        return "DOWN"
    if r <= -0.3:
        return "MILD_DOWN"
    return "FLAT"


def _state_distance_from_extreme(d: Optional[float], *, kind: str) -> str:
    """
    distance_from_20d_high: typically negative when below the high.
    distance_from_20d_low : typically positive when above the low.
    """
    if d is None:
        return "UNKNOWN"
    if kind == "HIGH":
        # near high is closer to 0, far below is big negative
        if d >= -1.0:
            return "NEAR_HIGH"
        if d >= -5.0:
            return "BELOW_HIGH"
        if d >= -12.0:
            return "FAR_BELOW_HIGH"
        return "EXTREMELY_BELOW_HIGH"
    # LOW
    if d <= 1.0:
        return "NEAR_LOW"
    if d <= 5.0:
        return "ABOVE_LOW"
    if d <= 12.0:
        return "FAR_ABOVE_LOW"
    return "EXTREMELY_ABOVE_LOW"


def _state_candle_body(body_pct: Optional[float]) -> str:
    if body_pct is None:
        return "UNKNOWN"
    ab = abs(body_pct)
    if ab < 0.15:
        return "DOJI_SMALL_BODY"
    if body_pct > 0:
        if ab >= 2.0:
            return "STRONG_GREEN_BODY"
        if ab >= 0.7:
            return "GREEN_BODY"
        return "SMALL_GREEN_BODY"
    # body_pct < 0
    if ab >= 2.0:
        return "STRONG_RED_BODY"
    if ab >= 0.7:
        return "RED_BODY"
    return "SMALL_RED_BODY"


def _state_wick(x: Optional[float]) -> str:
    if x is None:
        return "UNKNOWN"
    if x >= 2.0:
        return "VERY_LONG"
    if x >= 0.9:
        return "LONG"
    if x >= 0.3:
        return "NORMAL"
    return "SHORT"


def _state_prob(p01: Optional[float]) -> str:
    """Probability in 0..1 -> tiers."""
    if p01 is None:
        return "UNKNOWN"
    p = _clamp(p01, 0.0, 1.0)
    if p >= 0.80:
        return "VERY_HIGH"
    if p >= 0.65:
        return "HIGH"
    if p >= 0.55:
        return "LEAN_HIGH"
    if p >= 0.45:
        return "BALANCED"
    if p >= 0.35:
        return "LEAN_LOW"
    if p >= 0.20:
        return "LOW"
    return "VERY_LOW"


def _state_confidence(c: Optional[float]) -> str:
    """Confidence 0..100 -> tiers."""
    if c is None:
        return "UNKNOWN"
    c = _clamp(float(c), 0.0, 100.0)
    if c >= 80:
        return "VERY_HIGH"
    if c >= 65:
        return "HIGH"
    if c >= 55:
        return "MODERATE"
    if c >= 45:
        return "LOW"
    return "VERY_LOW"


def _detect_regime_from_features(trend_strength_20: Optional[float], vol20: Optional[float], vol60: Optional[float], atr14: Optional[float]) -> str:
    """
    Mirrors your detect_market_regime() semantics, deterministically.
    """
    if trend_strength_20 is None or vol20 is None:
        return "UNKNOWN"
    # HIGH_VOL condition (same spirit as your code)
    base60 = vol60 if vol60 is not None else vol20
    if vol20 > 1.5 * base60:
        return "HIGH_VOL"
    if atr14 is not None and vol20 is not None and atr14 > 1.2 * vol20:
        return "HIGH_VOL"
    if abs(trend_strength_20) > 0.4:
        return "TRENDING"
    return "RANGING"


def _liquidity_quality_from_features(vol_z: Optional[float], vol_vs_ma20: Optional[float], intraday_range: Optional[float], vol20: Optional[float]) -> str:
    """
    Deterministic liquidity quality classification (aligns with your earlier logic).
    """
    if vol_z is None or vol_vs_ma20 is None:
        return "POOR"
    if vol_z < -1.0 or vol_vs_ma20 < -20:
        return "POOR"
    if vol_z < 0.3 or vol_vs_ma20 < 0:
        return "THIN"
    if intraday_range is not None and vol20 is not None:
        if intraday_range > 6.0 and vol20 > 4.0:
            return "THIN"
    return "GOOD"


def _freshness_state(minutes_ago: Optional[float]) -> str:
    if minutes_ago is None:
        return "UNKNOWN"
    m = max(0.0, float(minutes_ago))
    if m <= 20:
        return "FRESH"
    if m <= 90:
        return "RECENT"
    if m <= 240:
        return "AGING"
    if m <= 1440:
        return "STALE"
    return "VERY_STALE"


def _pattern_edge_state(winrate: Optional[float], avg_ret_pct: Optional[float], count: Optional[float]) -> str:
    """
    Institutional-ish “edge” state:
      winrate in 0..1, avg_ret_pct in percent units (e.g., +1.2), count in samples
    """
    if winrate is None or avg_ret_pct is None or count is None:
        return "UNKNOWN"
    n = int(count)
    if n < 8:
        return "INSUFFICIENT_SAMPLES"
    wr = float(winrate)
    ar = float(avg_ret_pct)
    # conservative tiers
    if wr >= 0.65 and ar >= 1.0:
        return "POSITIVE_EDGE"
    if wr >= 0.60 and ar > 0.0:
        return "LEAN_POSITIVE"
    if wr <= 0.45 and ar <= 0.0:
        return "NEGATIVE_EDGE"
    return "MIXED_EDGE"


# ------------------------------------------------------------
# Public API (pure)
# ------------------------------------------------------------
def compute_indicator_states(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
      payload can be your Firestore symbol doc OR your API assembled payload.
      (No I/O, so caller decides what to pass.)

    Output:
      {
        "states": { indicator_name: "STATE_STRING", ... }   # for 68 indicators
        "values": { indicator_name: float|None, ... }       # normalized numeric values
      }
    """
    # ---- value extraction (supports both Firestore doc and API response shapes) ----
    # Prefer Firestore: features_meta.*
    features = _get(payload, "features_meta") or _get(payload, "features") or {}

    # quote values might live under "quote" (Firestore) OR "header.quote" (API)
    quote = _get(payload, "quote") or _get(payload, "header.quote") or {}

    # bullbrain might live under "bullbrain.raw" (Firestore) OR "bullbrain.raw" (API)
    bullbrain = _get(payload, "bullbrain") or {}
    bull_raw = _get(bullbrain, "raw") or _get(bullbrain, "probabilities") or {}

    # decision data might live under payload.decision.* (Firestore) OR header/content.ui.decision (API)
    decision = _get(payload, "decision") or _get(payload, "content.ui.decision") or {}
    decision_prob = _get(decision, "probability") or {}
    decision_bias = _get(decision, "bias") or {}

    # pattern history (Firestore) OR content.patternHistory (API)
    patt_hist = _get(payload, "patternHistory") or _get(payload, "content.patternHistory") or {}
    days5 = _get(patt_hist, "forwardReturns.days5") or {}
    # days5 fields in your doc: avg/best/worst/count/winRate are in "percent units" already (avg is percent)
    # (your scanner multiplies *100, so avg is already %)

    # freshness can live under content.ui.freshness.minutesAgo (API) or not present in Firestore doc
    freshness_minutes = _to_float(_get(payload, "content.ui.freshness.minutesAgo"))

    # ---- Normalize the 48 feature values (float/None) ----
    values: Dict[str, Optional[float]] = {}
    for k in BULLBRAIN_FEATURES_48:
        values[k] = _to_float(features.get(k))

    # ---- Add 20 display-layer values (float/None) ----
    # quote_change_pct: try quote.changePct (could be decimal in API header.quote), or quote.changePct percent in Firestore quote
    q_change_pct = _to_float(quote.get("changePct"))
    # If API header.quote.changePct is decimal (-0.0668), convert to percent
    if q_change_pct is not None and abs(q_change_pct) <= 2.0:
        q_change_pct = q_change_pct * 100.0

    q_change_abs = _to_float(quote.get("change"))

    # quote_gap_pct: use open vs prevClose if present
    q_open = _to_float(quote.get("open"))
    q_prev = _to_float(quote.get("prevClose"))
    q_gap_pct = None
    if q_open is not None and q_prev is not None and q_prev != 0:
        q_gap_pct = (q_open / q_prev - 1.0) * 100.0

    # quote_range_pct: (high-low)/price * 100 if possible
    q_high = _to_float(quote.get("high"))
    q_low = _to_float(quote.get("low"))
    q_price = _to_float(quote.get("price")) or values.get("close")
    q_range_pct = None
    if q_high is not None and q_low is not None and q_price is not None and q_price != 0:
        q_range_pct = ((q_high - q_low) / q_price) * 100.0

    # quote_volume_vs_ma20: use feature volume_vs_ma20_pct if available
    q_vol_vs_ma20 = values.get("volume_vs_ma20_pct")

    # model probabilities + confidence (Firestore: bullbrain.raw.prob_up/down)
    model_prob_up = _to_float(_get(bullbrain, "raw.prob_up")) or _to_float(_get(bullbrain, "raw.probability_up"))
    model_prob_down = _to_float(_get(bullbrain, "raw.prob_down")) or _to_float(_get(bullbrain, "raw.probability_down"))
    model_conf = _to_float(_get(bullbrain, "confidence"))

    # hybrid probs:
    # Prefer API: content.ui.hybridProbUp or content.ui.decision.probability.up (already 0..1 in your sample)
    hybrid_prob_up = _to_float(_get(payload, "content.ui.hybridProbUp"))
    if hybrid_prob_up is None:
        hybrid_prob_up = _to_float(decision_prob.get("up"))
    hybrid_prob_down = _to_float(_get(payload, "content.ui.decision.probability.down"))
    if hybrid_prob_down is None:
        hybrid_prob_down = _to_float(decision_prob.get("down"))
    if hybrid_prob_up is not None and hybrid_prob_down is None:
        hybrid_prob_down = 1.0 - hybrid_prob_up
    if hybrid_prob_down is not None and hybrid_prob_up is None:
        hybrid_prob_up = 1.0 - hybrid_prob_down

    bias_strength = _to_float(decision_bias.get("strength"))

    # liquidity + regime derived (deterministic)
    vol_z = values.get("volume_zscore_20")
    vol_vs_ma20 = values.get("volume_vs_ma20_pct")
    intraday = values.get("intraday_range_pct")
    vol20 = values.get("volatility_20d")
    vol60 = values.get("volatility_60d")
    atr14 = values.get("atr14")
    liq_q = _liquidity_quality_from_features(vol_z, vol_vs_ma20, intraday, vol20)
    regime = _detect_regime_from_features(values.get("trend_strength_20"), vol20, vol60, atr14)

    # pattern stats (5d)
    patt_win = _to_float(days5.get("winRate"))
    patt_avg = _to_float(days5.get("avg"))          # already % in your storage
    patt_cnt = _to_float(days5.get("count"))
    patt_occ = _to_float(patt_hist.get("occurrences"))

    values.update({
        "quote_change_pct": q_change_pct,
        "quote_change_abs": q_change_abs,
        "quote_gap_pct": q_gap_pct,
        "quote_range_pct": q_range_pct,
        "quote_volume_vs_ma20": q_vol_vs_ma20,

        "model_prob_up": model_prob_up,
        "model_prob_down": model_prob_down,
        "model_confidence": model_conf,

        "hybrid_prob_up": hybrid_prob_up,
        "hybrid_prob_down": hybrid_prob_down,
        "decision_bias_strength": bias_strength,

        # for these, "values" is not numeric in a meaningful way; we keep None
        "liquidity_quality": None,
        "regime_state": None,

        "pattern_winrate_5d": patt_win,
        "pattern_avg_5d": patt_avg,
        "pattern_sample_count_5d": patt_cnt,
        "pattern_occurrences": patt_occ,
        "pattern_edge_5d": None,

        "freshness_minutes_ago": freshness_minutes,
        "freshness_state": None,
    })

    # ---- Compute states for all 68 ----
    states: Dict[str, str] = {}

    # 48 feature states
    # Raw price/MA levels we keep as "VALUE_ONLY" since they are not normalized signals by themselves.
    VALUE_ONLY = {
        "adj_close", "close", "high", "low", "open",
        "volume", "sma5", "sma10", "sma20", "sma50", "sma200",
        "ema12", "ema26", "macd", "macd_signal",
        "volume_ma5", "volume_ma20", "obv", "true_range", "atr14"
    }

    for k in BULLBRAIN_FEATURES_48:
        v = values.get(k)

        if k in VALUE_ONLY:
            states[k] = "VALUE_ONLY" if v is not None else "UNKNOWN"
            continue

        if k in ("return_1d", "return_5d", "return_10d"):
            states[k] = _state_return(v)
            continue

        if k in ("volatility_5d", "volatility_20d", "volatility_60d"):
            # Use vol20 thresholds for all; it's consistent for UI meaning
            states[k] = _state_volatility(v)
            continue

        if k in ("sma5_sma20_pct", "sma20_sma50_pct", "ema_ratio"):
            states[k] = _state_ma_spread(v)
            continue

        if k == "price_vs_sma20_pct":
            states[k] = _state_price_vs_sma20(v)
            continue

        if k == "rsi14":
            states[k] = _state_rsi(v)
            continue

        if k == "macd_hist":
            states[k] = _state_macd_hist(v)
            continue

        if k in ("stoch_k_14", "stoch_d_3"):
            states[k] = _state_stoch(v)
            continue

        if k == "williams_r_14":
            states[k] = _state_williams_r(v)
            continue

        if k in ("volume_change_1d",):
            states[k] = _state_return(v)  # same scale meaning (+/-)
            continue

        if k in ("volume_vs_ma5_pct", "volume_vs_ma20_pct"):
            states[k] = _state_volume_vs_ma(v)
            continue

        if k == "volume_zscore_20":
            states[k] = _state_volume_z(v)
            continue

        if k == "obv_slope_10":
            states[k] = _state_obv_slope(v)
            continue

        if k == "intraday_range_pct":
            states[k] = _state_intraday_range(v)
            continue

        if k in ("upper_shadow_pct", "lower_shadow_pct"):
            states[k] = _state_wick(v)
            continue

        if k == "body_pct":
            states[k] = _state_candle_body(v)
            continue

        if k == "gap_pct":
            states[k] = _state_gap(v)
            continue

        if k == "distance_from_20d_high":
            states[k] = _state_distance_from_extreme(v, kind="HIGH")
            continue

        if k == "distance_from_20d_low":
            states[k] = _state_distance_from_extreme(v, kind="LOW")
            continue

        if k == "trend_strength_20":
            states[k] = _state_trend_strength(v)
            continue

        # default
        states[k] = "UNKNOWN" if v is None else "VALUE_ONLY"

    # 20 display-layer states
    # Quote movement
    states["quote_change_pct"] = _state_return(values.get("quote_change_pct"))
    states["quote_change_abs"] = "VALUE_ONLY" if values.get("quote_change_abs") is not None else "UNKNOWN"
    states["quote_gap_pct"] = _state_gap(values.get("quote_gap_pct"))
    states["quote_range_pct"] = _state_intraday_range(values.get("quote_range_pct"))
    states["quote_volume_vs_ma20"] = _state_volume_vs_ma(values.get("quote_volume_vs_ma20"))

    # Model probs/conf
    states["model_prob_up"] = _state_prob(values.get("model_prob_up"))
    states["model_prob_down"] = _state_prob(values.get("model_prob_down"))
    states["model_confidence"] = _state_confidence(values.get("model_confidence"))

    # Hybrid probs/bias
    states["hybrid_prob_up"] = _state_prob(values.get("hybrid_prob_up"))
    states["hybrid_prob_down"] = _state_prob(values.get("hybrid_prob_down"))
    # bias strength in [0..100] typically
    bs = values.get("decision_bias_strength")
    if bs is None:
        states["decision_bias_strength"] = "UNKNOWN"
    else:
        if bs >= 70:
            states["decision_bias_strength"] = "STRONG"
        elif bs >= 45:
            states["decision_bias_strength"] = "MODERATE"
        elif bs >= 20:
            states["decision_bias_strength"] = "WEAK"
        else:
            states["decision_bias_strength"] = "VERY_WEAK"

    # Derived liquidity & regime states
    states["liquidity_quality"] = liq_q
    states["regime_state"] = regime

    # Pattern stats
    states["pattern_winrate_5d"] = _state_prob(values.get("pattern_winrate_5d"))
    # avg 5d is in percent units (e.g., +1.03)
    patt_avg_v = values.get("pattern_avg_5d")
    states["pattern_avg_5d"] = _state_return(patt_avg_v)  # same tiering works well
    # sample count
    cnt = values.get("pattern_sample_count_5d")
    if cnt is None:
        states["pattern_sample_count_5d"] = "UNKNOWN"
    else:
        n = int(cnt)
        if n >= 50:
            states["pattern_sample_count_5d"] = "ROBUST"
        elif n >= 20:
            states["pattern_sample_count_5d"] = "ADEQUATE"
        elif n >= 8:
            states["pattern_sample_count_5d"] = "LOW"
        else:
            states["pattern_sample_count_5d"] = "INSUFFICIENT"

    occ = values.get("pattern_occurrences")
    if occ is None:
        states["pattern_occurrences"] = "UNKNOWN"
    else:
        o = int(occ)
        if o >= 25:
            states["pattern_occurrences"] = "COMMON"
        elif o >= 10:
            states["pattern_occurrences"] = "OCCASIONAL"
        elif o >= 3:
            states["pattern_occurrences"] = "RARE"
        else:
            states["pattern_occurrences"] = "VERY_RARE"

    states["pattern_edge_5d"] = _pattern_edge_state(
        values.get("pattern_winrate_5d"),
        values.get("pattern_avg_5d"),
        values.get("pattern_sample_count_5d"),
    )

    # Freshness
    states["freshness_minutes_ago"] = "VALUE_ONLY" if values.get("freshness_minutes_ago") is not None else "UNKNOWN"
    states["freshness_state"] = _freshness_state(values.get("freshness_minutes_ago"))

    # Ensure all 68 exist
    for k in ALL_68_INDICATORS:
        if k not in states:
            states[k] = "UNKNOWN"

    return {
        "states": states,
        "values": values,
        "meta": {
            "count": len(ALL_68_INDICATORS),
            "version": "indicator_states_v1",
        },
    }


def list_all_indicators() -> list[str]:
    """Pure helper: returns the 68 indicator keys in order."""
    return list(ALL_68_INDICATORS)
