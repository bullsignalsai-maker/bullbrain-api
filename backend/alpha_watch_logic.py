# backend/alpha_watch_logic.py
# =========================================================
# Alphaclara — AI Opportunity Watch
# Cross-Sectional Multi-Factor Opportunity Ranking Engine
# =========================================================

import datetime
import math
from typing import Any, Dict, List, Optional, Tuple
from firebase_admin import firestore

try:
    from backend.firestore_utils import utc_now_iso
except Exception:
    def utc_now_iso() -> str:
        return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


COL_ROOT = "bullsignals_ai"
DOC_ALPHA_WATCH = "alpha_watch"
HISTORY_ROOT = "alpha_watch_history"

MIN_PRICE = 12.0
MIN_AVG_VOLUME_20D = 1_200_000
MAX_FRESHNESS_HOURS = 36
MIN_PROB_UP_SOFT = 0.42
MIN_PROB_UP_STRONG = 0.48
MIN_FINAL_SCORE = 62.0
NEGATIVE_MOVE_PENALTY_TRIGGER = -2.0
MAX_EARLY_EXPANSION_DEFAULT = 85.0
MAX_ITEMS = 8
MAX_PER_THEME = 2

BASE_WEIGHTS = {
    "momentum": 0.28,
    "trend": 0.22,
    "pattern": 0.20,
    "bullbrain": 0.15,
    "volume": 0.10,
    "early_expansion": 0.05,
}

THEME_BUCKETS = {
    "AI_Semis": {"NVDA", "AMD", "AVGO", "MU", "SMCI", "ARM", "ASML", "AMAT", "LRCX", "KLAC", "QCOM", "INTC", "DELL", "WDC", "STX"},
    "Cloud_Software": {"MSFT", "GOOGL", "META", "CRM", "NOW", "SNOW", "DDOG", "NET", "ZS", "MDB", "CRWD", "PANW", "PLTR", "ADBE", "ORCL"},
    "Consumer_Internet": {"AMZN", "NFLX", "UBER", "SHOP", "TSLA", "AAPL", "RBLX", "SPOT", "RDDT"},
    "Financial_Crypto": {"JPM", "BAC", "GS", "MS", "V", "MA", "COIN", "MSTR", "HOOD", "SOFI", "AFRM", "UPST"},
    "Energy_Industrials": {"XOM", "CVX", "GE", "CAT", "RTX", "LMT", "VRT", "FIX", "OKLO", "SMR"},
}


def _num(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _age_hours(iso: Optional[str]) -> Optional[float]:
    try:
        if not iso:
            return None
        ts = datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - ts).total_seconds() / 3600.0
    except Exception:
        return None


def _fresh_enough(stock: Dict[str, Any]) -> bool:
    ts = (
        stock.get("computed_at")
        or (stock.get("quote") or {}).get("updated_at")
        or stock.get("updated_at")
    )
    age = _age_hours(ts)
    return age is not None and age <= MAX_FRESHNESS_HOURS


def _theme_for_symbol(symbol: str) -> str:
    s = str(symbol or "").upper()
    for theme, symbols in THEME_BUCKETS.items():
        if s in symbols:
            return theme
    return "Other"


def _signal(stock: Dict[str, Any]) -> str:
    bull = stock.get("bullbrain") or {}
    decision = stock.get("decision") or {}
    return decision.get("finalSignal") or decision.get("final") or bull.get("signal") or "HOLD"


def _prob_up(stock: Dict[str, Any]) -> float:
    bull = stock.get("bullbrain") or {}
    raw = bull.get("raw") or {}
    return _num(raw.get("prob_up"), 0.5) or 0.5


def _confidence(stock: Dict[str, Any]) -> float:
    bull = stock.get("bullbrain") or {}
    return _num(bull.get("confidence"), 50.0) or 50.0


def passes_quality_filter(stock: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    symbol = str(stock.get("symbol") or "").upper().strip()
    quote = stock.get("quote") or {}
    features = stock.get("features_meta") or {}
    bull = stock.get("bullbrain") or {}
    decision = stock.get("decision") or {}
    pattern = stock.get("pattern")
    quality = decision.get("quality") or {}

    if not symbol:
        reasons.append("missing_symbol")

    price = _num(quote.get("price"))
    if price is None or price < MIN_PRICE:
        reasons.append("price_below_minimum")

    volume_ma20 = _num(features.get("volume_ma20"))
    if volume_ma20 is not None and volume_ma20 < MIN_AVG_VOLUME_20D:
        reasons.append("avg_volume_below_minimum")

    if not _fresh_enough(stock):
        reasons.append("stale_data")

    if not isinstance(features, dict) or not features:
        reasons.append("missing_features_meta")

    if not isinstance(bull, dict) or not bull:
        reasons.append("missing_bullbrain")

    if not isinstance(decision, dict) or not decision:
        reasons.append("missing_decision")

    if not pattern:
        reasons.append("missing_pattern")

    liquidity = quality.get("liquidity")
    if liquidity == "POOR":
        reasons.append("liquidity_POOR")

    fragility = _num(quality.get("fragility"), 0.0)
    if fragility is not None and fragility > 2:
        reasons.append("fragility_above_limit")

    if _signal(stock) == "SELL":
        reasons.append("sell_signal_excluded")

    return len(reasons) == 0, reasons


def detect_regime(stock: Dict[str, Any]) -> str:
    f = stock.get("features_meta") or {}

    trend = _num(f.get("trend_strength_20"), 0.0) or 0.0
    vol20 = _num(f.get("volatility_20d"), 0.0) or 0.0
    vol60 = _num(f.get("volatility_60d"), vol20) or vol20
    r5 = _num(f.get("return_5d"), 0.0) or 0.0
    r10 = _num(f.get("return_10d"), 0.0) or 0.0
    vol_vs = _num(f.get("volume_vs_ma20_pct"), 0.0) or 0.0

    if vol20 > max(4.5, vol60 * 1.6):
        return "HIGH_VOL"

    if trend > 0.25 and r5 > 0 and r10 > 0 and vol_vs >= 0:
        return "RISK_ON"

    if trend < -0.25 and r5 < 0 and r10 < 0:
        return "RISK_OFF"

    return "NEUTRAL"


def dynamic_weights(regime: str) -> Dict[str, float]:
    w = dict(BASE_WEIGHTS)

    if regime == "RISK_ON":
        w["momentum"] += 0.04
        w["trend"] += 0.02
        w["early_expansion"] += 0.03
        w["pattern"] -= 0.03
        w["bullbrain"] -= 0.03
        w["volume"] -= 0.03

    elif regime in {"RISK_OFF", "HIGH_VOL"}:
        w["trend"] += 0.04
        w["bullbrain"] += 0.03
        w["volume"] += 0.02
        w["momentum"] -= 0.04
        w["early_expansion"] -= 0.03
        w["pattern"] -= 0.02

    total = sum(w.values()) or 1.0
    return {k: round(v / total, 4) for k, v in w.items()}


def score_momentum(f: Dict[str, Any]) -> float:
    r1 = _num(f.get("return_1d"), 0.0) or 0.0
    r5 = _num(f.get("return_5d"), 0.0) or 0.0
    r10 = _num(f.get("return_10d"), 0.0) or 0.0
    rsi = _num(f.get("rsi14"), 50.0) or 50.0
    macd = _num(f.get("macd_hist"), 0.0) or 0.0

    score = 50.0
    score += _clamp(r5 * 3.6, -25, 30)
    score += _clamp(r10 * 2.1, -20, 25)
    score += _clamp(r1 * 1.4, -10, 12)

    if 45 <= rsi <= 68:
        score += 14
    elif 68 < rsi <= 75:
        score += 5
    elif rsi > 78:
        score -= 18
    elif rsi < 40:
        score -= 12

    if macd > 0:
        score += 10
    elif macd < 0:
        score -= 10

    return round(_clamp(score), 2)


def score_trend(f: Dict[str, Any]) -> float:
    trend = _num(f.get("trend_strength_20"), 0.0) or 0.0
    vs20 = _num(f.get("price_vs_sma20_pct"), 0.0) or 0.0
    vs50 = _num(f.get("sma20_sma50_pct"), 0.0) or 0.0
    sma5_20 = _num(f.get("sma5_sma20_pct"), 0.0) or 0.0

    score = 50.0
    score += _clamp(trend * 28, -30, 35)
    score += _clamp(vs20 * 2.2, -18, 24)
    score += _clamp(vs50 * 2.0, -15, 20)
    score += _clamp(sma5_20 * 1.5, -10, 12)

    return round(_clamp(score), 2)


def score_pattern(stock: Dict[str, Any]) -> float:
    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}
    name = pattern.get("pattern") or pattern.get("patternLabel") or ""
    bias = str(pattern.get("bias") or stock.get("patternBias") or "").lower()

    days5 = ((history.get("forwardReturns") or {}).get("days5") or {})
    win_rate = _num(days5.get("winRate"))
    avg_ret = _num(days5.get("avg"))
    count = _num(days5.get("count"), 0.0) or 0.0

    score = 45.0
    bullish = {
        "GAP UP & RUNNING",
        "VOLUME BREAKOUT",
        "TREND ACCELERATION",
        "BUY THE DIP (UPTREND)",
        "HAMMER REVERSAL",
        "OVERSOLD BOUNCE",
    }

    if name in bullish or bias == "bull":
        score += 22
    elif bias == "bear":
        score -= 25

    if win_rate is not None:
        score += (win_rate - 0.5) * 90

    if avg_ret is not None:
        score += _clamp(avg_ret * 4.8, -25, 25)

    reliability = min(count / 15.0, 1.0)
    score = 50 + (score - 50) * reliability

    return round(_clamp(score), 2)


def score_bullbrain(stock: Dict[str, Any]) -> float:
    conf = _confidence(stock)
    prob = _prob_up(stock)
    sig = _signal(stock)

    score = 45.0
    score += (conf - 50) * 0.8
    score += (prob - 0.5) * 80

    if sig == "BUY":
        score += 18
    elif sig == "SELL":
        score -= 30

    return round(_clamp(score), 2)


def score_volume(f: Dict[str, Any]) -> float:
    vs = _num(f.get("volume_vs_ma20_pct"), 0.0) or 0.0
    z = _num(f.get("volume_zscore_20"), 0.0) or 0.0
    obv = _num(f.get("obv_slope_10"), 0.0) or 0.0

    score = 50.0
    score += _clamp(vs * 0.52, -25, 28)
    score += _clamp(z * 11, -20, 23)

    if obv > 0:
        score += 10
    elif obv < 0:
        score -= 10

    return round(_clamp(score), 2)


def score_early_expansion(f: Dict[str, Any]) -> float:
    r1 = _num(f.get("return_1d"), 0.0) or 0.0
    r5 = _num(f.get("return_5d"), 0.0) or 0.0
    r10 = _num(f.get("return_10d"), 0.0) or 0.0
    vs20 = _num(f.get("price_vs_sma20_pct"), 0.0) or 0.0
    vol_vs = _num(f.get("volume_vs_ma20_pct"), 0.0) or 0.0
    rsi = _num(f.get("rsi14"), 50.0) or 50.0
    macd = _num(f.get("macd_hist"), 0.0) or 0.0
    vol20 = _num(f.get("volatility_20d"), 2.0) or 2.0
    dist_high = _num(f.get("distance_from_20d_high"), 0.0) or 0.0

    score = 42.0

    if r5 > 0 and r10 > 0:
        score += 12
    elif r5 > 0:
        score += 6

    if r1 > 0 and r5 > 0:
        score += 5

    if 0 < vs20 < 7:
        score += 12
    elif 7 <= vs20 <= 12:
        score += 5
    elif vs20 > 15:
        score -= 12

    if vol_vs > 25:
        score += 14
    elif vol_vs > 10:
        score += 8
    elif vol_vs < -10:
        score -= 12

    if macd > 0:
        score += 8
    else:
        score -= 6

    if 48 <= rsi <= 66:
        score += 10
    elif rsi > 74:
        score -= 14

    if vol20 < 3.5:
        score += 6

    if -5 <= dist_high <= 0:
        score += 5

    return round(_clamp(score), 2)

def stability_profile(stock: Dict[str, Any]) -> Dict[str, Any]:
    q = ((stock.get("decision") or {}).get("quality") or {})
    f = stock.get("features_meta") or {}

    mult = 1.0
    flags = []

    frag = _num(q.get("fragility"), 0.0) or 0.0
    liq = q.get("liquidity")
    rsi = _num(f.get("rsi14"), 50.0) or 50.0
    vol20 = _num(f.get("volatility_20d"), 0.0) or 0.0
    range_pct = _num(f.get("intraday_range_pct"), 0.0) or 0.0
    vol_vs = _num(f.get("volume_vs_ma20_pct"), 0.0) or 0.0

    if frag >= 3:
        mult *= 0.52
        flags.append("High fragility")
    elif frag >= 2:
        mult *= 0.84
        flags.append("Moderate fragility")

    if liq == "THIN":
        mult *= 0.88
        flags.append("Liquidity thin")

    elif liq == "POOR":
        mult *= 0.65
        flags.append("Liquidity poor")

    if rsi > 78:
        mult *= 0.70
        flags.append("Momentum extended")

    if vol20 > 5:
        mult *= 0.80
        flags.append("High volatility")

    if range_pct > 7:
        mult *= 0.78
        flags.append("Wide intraday range")

    if vol_vs < -20:
        mult *= 0.75
        flags.append("Weak participation")

    mult = max(0.35, min(1.0, mult))

    risk = "Controlled" if mult >= 0.90 else "Moderate" if mult >= 0.75 else "Elevated"

    return {
        "multiplier": round(mult, 3),
        "riskLevel": risk,
        "riskFlags": flags[:3],
    }


def derive_setup_label(stock: Dict[str, Any], scores: Dict[str, float]) -> str:
    pattern = stock.get("pattern") or {}
    pattern_name = pattern.get("pattern") or pattern.get("patternLabel")

    if pattern_name in {"TREND ACCELERATION", "GAP UP & RUNNING", "VOLUME BREAKOUT"}:
        return str(pattern_name).title()

    if scores.get("momentum", 0) >= 72 and scores.get("volume", 0) >= 65:
        return "Momentum Expansion"

    if scores.get("trend", 0) >= 72 and scores.get("volume", 0) >= 60:
        return "Trend Acceleration with Volume"

    if scores.get("early_expansion", 0) >= 70:
        return "Early Breakout Pressure"

    if scores.get("pattern", 0) >= 70:
        return "Pattern-Supported Setup"

    return "Improving Opportunity Setup"


def build_reason(stock: Dict[str, Any], scores: Dict[str, float]) -> str:
    awareness = stock.get("marketAwareness") or {}
    one_liner = awareness.get("oneLiner") or awareness.get("summary")

    if isinstance(one_liner, str) and one_liner.strip():
        return clean_reason_text(one_liner)

    strongest = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    names = [k.replace("_", " ").title() for k, _ in strongest]
    return f"{stock.get('symbol')} ranks highly due to {', '.join(names).lower()} alignment."


def build_why_now(scores: Dict[str, float]) -> List[str]:
    points = []

    if scores.get("momentum", 0) >= 65:
        points.append("Momentum persistence is improving")
    if scores.get("trend", 0) >= 65:
        points.append("Trend structure remains constructive")
    if scores.get("pattern", 0) >= 65:
        points.append("Pattern intelligence supports the setup")
    if scores.get("bullbrain", 0) >= 65:
        points.append("BullBrain conviction is above average")
    if scores.get("volume", 0) >= 65:
        points.append("Volume participation is confirming the move")
    if scores.get("early_expansion", 0) >= 65:
        points.append("Early expansion pressure is forming")

    return points[:3] or ["Multiple Alphaclara intelligence factors are aligning"]

def clean_reason_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    s = (
        text.replace("â", "'")
            .replace("â", '"')
            .replace("â", '"')
            .replace("â", "-")
            .replace("â", "-")
            .replace("  ", " ")
            .strip()
    )

    # Avoid ugly partial endings
    for sep in [". ", "; ", " - "]:
        if sep in s[:190]:
            parts = s[:190].split(sep)
            if len(parts[0]) > 35:
                return parts[0].strip() + "."

    return s[:170].rstrip(" ,;:-") + "."

def score_stock(stock: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ok, reject_reasons = passes_quality_filter(stock)
    if not ok:
        return None

    symbol = str(stock.get("symbol") or "").upper()
    quote = stock.get("quote") or {}
    features = stock.get("features_meta") or {}

    prob_up = _prob_up(stock)
    signal = _signal(stock)
    change_pct = _num(quote.get("changePct"), 0.0) or 0.0

    # Strong upside opportunity must have at least reasonable upside probability.
    # BUY can survive slightly lower probability because decision ladder may override based on pattern/momentum.

    probability_penalty = 1.0
    if prob_up < 0.40:
        probability_penalty = 0.78
    elif prob_up < 0.46:
        probability_penalty = 0.90

    regime = detect_regime(stock)
    weights = dynamic_weights(regime)

    factor_scores = {
        "momentum": score_momentum(features),
        "trend": score_trend(features),
        "pattern": score_pattern(stock),
        "bullbrain": score_bullbrain(stock),
        "volume": score_volume(features),
        "early_expansion": score_early_expansion(features),
    }
    pattern_name = (
        (stock.get("pattern") or {}).get("pattern")
        or (stock.get("pattern") or {}).get("patternLabel")
        or ""
    )

    bearish_patterns = {
        "GAP DOWN & PRESSURE",
        "BEARISH ENGULFING",
        "BREAKDOWN",
        "DISTRIBUTION DAY",
    }

    bearish_pattern_setup = pattern_name in bearish_patterns

    if bearish_pattern_setup:
        factor_scores["pattern"] = min(factor_scores["pattern"], 35)
    # Early expansion should not dominate unless model/probability confirms it.
    if not (
        prob_up >= 0.55
        and change_pct >= 0
        and factor_scores["volume"] >= 65
        and factor_scores["momentum"] >= 65
    ):
        factor_scores["early_expansion"] = min(
            factor_scores["early_expansion"],
            MAX_EARLY_EXPANSION_DEFAULT,
        )

    opportunity_score = sum(factor_scores[k] * weights[k] for k in weights)
    stability = stability_profile(stock)

    final_score = opportunity_score * stability["multiplier"] * probability_penalty

    # Negative daily move penalty.
    # It may still qualify, but only as a pullback setup, not acceleration.
    pullback_setup = False
    if change_pct <= NEGATIVE_MOVE_PENALTY_TRIGGER:
        final_score *= 0.82
        pullback_setup = True

    final_score = round(_clamp(final_score), 2)

    if final_score < MIN_FINAL_SCORE:
        return None

    setup_label = derive_setup_label(stock, factor_scores)

    if bearish_pattern_setup:
        setup_label = "Cautionary Momentum Watch"

    if prob_up < 0.40:
        setup_label = "Developing Momentum Watch"
    elif prob_up < 0.46 and signal != "BUY":
        setup_label = "Momentum Watch"

    if pullback_setup:
        setup_label = "Constructive Pullback Watch"

    return {
        "symbol": symbol,
        "companyName": stock.get("company_name") or symbol,
        "theme": _theme_for_symbol(symbol),
        "score": final_score,
        "opportunityScore": round(opportunity_score, 2),
        "stabilityMultiplier": stability["multiplier"],
        "confidence": round(_clamp(_confidence(stock)), 2),
        "probUp": round(prob_up, 4),
        "signal": signal,
        "price": quote.get("price"),
        "change": quote.get("change"),
        "changePct": quote.get("changePct"),
        "quote_updated_at": quote.get("updated_at"),
        "computed_at": stock.get("computed_at"),
        "marketRegime": regime,
        "setupLabel": setup_label,
        "reason": build_reason(stock, factor_scores),
        "whyNow": build_why_now(factor_scores),
        "riskLevel": stability["riskLevel"],
        "riskFlags": stability["riskFlags"],
        "factorScores": {k: round(v, 1) for k, v in factor_scores.items()},
        "weights": weights,
        "pattern": (stock.get("pattern") or {}).get("pattern")
            or (stock.get("pattern") or {}).get("patternLabel"),
        "schema_version": "alpha_watch_item_v4",
    }

def apply_diversity_filter(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    theme_count = {}

    for item in items:
        theme = item.get("theme", "Other")
        count = theme_count.get(theme, 0)

        if theme != "Other" and count >= MAX_PER_THEME:
            continue

        selected.append(item)
        theme_count[theme] = count + 1

        if len(selected) >= MAX_ITEMS:
            break

    return selected


def build_alpha_watch_from_docs(stock_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = []
    reject_counts: Dict[str, int] = {}

    for doc in stock_docs:
        try:
            ok, reasons = passes_quality_filter(doc)

            if not ok:
                for r in reasons:
                    reject_counts[r] = reject_counts.get(r, 0) + 1
                continue

            item = score_stock(doc)
            if item:
                scored.append(item)
            else:
                reject_counts["score_below_threshold_or_prob_gate"] = (
                    reject_counts.get("score_below_threshold_or_prob_gate", 0) + 1
                )

        except Exception:
            reject_counts["exception"] = reject_counts.get("exception", 0) + 1

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    selected = apply_diversity_filter(scored)

    regimes = {}
    for item in selected:
        r = item.get("marketRegime", "NEUTRAL")
        regimes[r] = regimes.get(r, 0) + 1

    market_regime = (
        sorted(regimes.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        if regimes else "NEUTRAL"
    )

    return {
        "title": "AI Opportunity Watch",
        "subtitle": "Highest-conviction setups ranked by momentum, trend quality, pattern edge, BullBrain conviction, and participation.",
        "updated_at": utc_now_iso(),
        "market_regime": market_regime,
        "count": len(selected),
        "items": selected,
        "methodology": {
            "type": "cross_sectional_multi_factor_opportunity_ranking",
            "minimum_price": MIN_PRICE,
            "minimum_avg_volume_20d": MIN_AVG_VOLUME_20D,
            "freshness_hours": MAX_FRESHNESS_HOURS,
            "minimum_final_score": MIN_FINAL_SCORE,
            "max_items": MAX_ITEMS,
            "max_per_theme": MAX_PER_THEME,
            "base_weights": BASE_WEIGHTS,
            "regime_adjusted": True,
            "stability_adjusted": True,
            "diversity_control": True,
        },
        "quality": {
            "input_docs": len(stock_docs),
            "qualified_before_diversity": len(scored),
            "selected": len(selected),
            "reject_counts": reject_counts,
        },
        "disclaimer": "AI Opportunity Watch is probabilistic and for informational and educational purposes only. Not financial advice.",
        "schema_version": "alpha_watch_v2_debug",
    }

def persist_alpha_watch(db: firestore.Client, stock_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = build_alpha_watch_from_docs(stock_docs)

    db.collection(COL_ROOT).document(DOC_ALPHA_WATCH).set(payload, merge=True)

    date_key = datetime.datetime.utcnow().date().isoformat()
    db.collection(COL_ROOT).document(HISTORY_ROOT).collection("days").document(date_key).set(
        {
            "date": date_key,
            "updated_at": payload.get("updated_at"),
            "market_regime": payload.get("market_regime"),
            "count": payload.get("count"),
            "items": payload.get("items", []),
            "schema_version": "alpha_watch_history_v1",
        },
        merge=True,
    )

    return payload