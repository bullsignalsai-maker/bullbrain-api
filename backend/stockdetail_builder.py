# backend/stockdetail_builder.py

import datetime
from typing import Any, Dict, Optional

from backend.schema_versions import (
    STOCKDETAIL_SCHEMA_VERSION,
    TTL_STOCKDETAIL_SECONDS,
)

# We import from main.py because those functions already exist there.
# This is OK because main.py already defines them earlier in the file.
from main import (
    backend_fetch_quote,
    fetch_daily_candles,
    compute_bullbrain_features,
    bullbrain_infer,
    build_technical_snapshot,
    _class_probs_from_prob_up,
    get_symbol_news,
    get_stockdetail_grok,
    _hybrid_from_probs,
    scan_smart_pattern_history,
    bullbrain_model,
    BULLBRAIN_VERSION,
)

DEFAULT_LIMIT_CANDLES = 180


def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


def iso(dt: datetime.datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"


def safe_float(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def build_candles_payload(symbol: str, candles: dict, limit_candles: int = DEFAULT_LIMIT_CANDLES) -> Optional[dict]:
    if not candles:
        return None

    closes = candles.get("close") or []
    highs = candles.get("high") or []
    lows = candles.get("low") or []
    opens = candles.get("open") or closes
    vols = candles.get("volume") or []
    ts_list = candles.get("timestamp") or []

    n = len(closes)
    if n == 0:
        return None

    use_n = min(limit_candles, n)
    start = n - use_n

    items = []
    for i in range(start, n):
        t_raw = ts_list[i] if i < len(ts_list) and ts_list[i] else None
        if t_raw:
            dt = datetime.datetime.utcfromtimestamp(t_raw / 1000.0).replace(microsecond=0)
        else:
            dt = utcnow() - datetime.timedelta(days=(n - 1 - i))

        items.append(
            {
                "t": iso(dt),
                "open": safe_float(opens[i]),
                "high": safe_float(highs[i]),
                "low": safe_float(lows[i]),
                "close": safe_float(closes[i]),
                "volume": safe_float(vols[i]),
            }
        )

    return {
        "symbol": symbol,
        "source": candles.get("source", "polygon"),
        "candles": items,
    }


def build_bullbrain_block(candles: dict):
    if not candles or bullbrain_model is None:
        return None, None, None, None

    features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
    inference = bullbrain_infer(features_vec)

    prob_up = float(inference.get("probability_up") or inference.get("raw_output") or 0.5)
    class_probs = _class_probs_from_prob_up(prob_up)

    bullbrain = {
        "version": BULLBRAIN_VERSION,
        "signal": inference.get("signal"),
        "confidence": inference.get("confidence"),
        "probabilities": class_probs,
        "raw": {"prob_up": prob_up, "prob_down": 1.0 - prob_up},
    }
    return bullbrain, feature_dict, float(last_close), prob_up


def build_smart_pattern_safe(symbol: str, candles: dict) -> Dict[str, Any]:
    raw = None
    try:
        raw = scan_smart_pattern_history(symbol, candles)
    except Exception:
        raw = None

    safe = {
        "pattern": None,
        "headline": None,
        "winRate": None,
        "occurrences": 0,
        "samples": [],
        "forwardReturns": {},
    }
    dates = []

    if raw:
        cp = raw.get("currentPattern")
        hist = raw.get("historyForCurrent")
        if cp and cp.get("pattern"):
            safe = {
                "pattern": cp.get("pattern"),
                "headline": cp.get("headline"),
                "winRate": cp.get("winRate"),
                "occurrences": hist.get("occurrences") if hist else 0,
                "samples": hist.get("samples") if hist else [],
                "forwardReturns": hist.get("forwardReturns") if hist else {},
            }
            if hist and hist.get("samples"):
                dates = hist["samples"][:5]

    return {"smartPattern": safe, "patternDates": dates, "patternStats": raw}


def build_stockdetail_payload(symbol: str, force_grok: bool = False, limit_candles: int = DEFAULT_LIMIT_CANDLES) -> Dict[str, Any]:
    symbol = symbol.upper()
    now = utcnow()
    expires_at_ts = int(now.timestamp()) + TTL_STOCKDETAIL_SECONDS

    quote = backend_fetch_quote(symbol)
    candles = fetch_daily_candles(symbol)

    bullbrain, feature_dict, last_close, bull_prob_up = build_bullbrain_block(candles)

    if last_close is None and quote:
        last_close = safe_float(quote.get("current")) or 0.0

    technical = None
    if feature_dict is not None and last_close is not None:
        technical = build_technical_snapshot(symbol, feature_dict, last_close)

    candles_payload = build_candles_payload(symbol, candles, limit_candles=limit_candles)

    news = get_symbol_news(symbol, limit=8)
    grok = get_stockdetail_grok(symbol, quote, technical, force=force_grok)
    grok_prob_up = grok.get("prob_up")

    hybrid_p, hybrid_signal, hybrid_conf = _hybrid_from_probs(bull_prob_up, grok_prob_up)

    sp = build_smart_pattern_safe(symbol, candles)

    return {
        "symbol": symbol,
        "schemaVersion": STOCKDETAIL_SCHEMA_VERSION,
        "asOf": iso(now),
        "computedAt": iso(now),
        "expiresAt": expires_at_ts,  # epoch seconds (fast TTL check)

        "quote": quote,
        "price": last_close,
        "bullbrain": bullbrain,
        "features": feature_dict,
        "technical": technical,
        "candles": candles_payload,
        "news": news,
        "grok": grok,

        "hybridProbUp": hybrid_p,
        "hybridSignal": hybrid_signal,
        "hybridScore": hybrid_conf,

        "smartPattern": sp["smartPattern"],
        "patternDates": sp["patternDates"],
        "patternStats": sp["patternStats"],
    }
