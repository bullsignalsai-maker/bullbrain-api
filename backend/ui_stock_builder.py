# backend/ui_stock_builder.py
# ---------------------------------------------------------
# UI Stock Builder (PLAN B — FINAL, SAFE)
# ---------------------------------------------------------
# Builds ONE canonical StockDetail UI document per symbol
# All computation happens HERE
# /stockdetail reads Firestore only
# ---------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, Optional, List
import datetime
import math

# Firestore (DO NOT change)
from backend.firestore_utils import get_db, iso_now

# Data sources
from backend.stock_repo import get_stock
from backend.candle_store import get_candles
from backend.technicals import build_technical_snapshot
from backend.smart_patterns import detect_smart_pattern, scan_smart_pattern_history
from backend.news_repo import fetch_symbol_news


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _iso_from_ms(ts_ms: Optional[int]) -> Optional[str]:
    try:
        if not ts_ms:
            return None
        dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000).replace(microsecond=0)
        return dt.isoformat() + "Z"
    except Exception:
        return None


# ---------------------------------------------------------
# Candles → UI payload
# ---------------------------------------------------------

def _build_candles_payload(
    symbol: str,
    candles: Dict[str, Any],
    limit: int = 180,
) -> Dict[str, Any]:

    closes = candles.get("close") or []
    if not closes:
        return {"symbol": symbol, "source": candles.get("source"), "candles": []}

    opens = candles.get("open") or []
    highs = candles.get("high") or []
    lows = candles.get("low") or []
    vols = candles.get("volume") or []
    ts = candles.get("timestamp") or []

    n = len(closes)
    start = max(0, n - limit)

    out: List[Dict[str, Any]] = []
    for i in range(start, n):
        t = _iso_from_ms(ts[i]) if i < len(ts) else None
        if not t:
            t = (
                datetime.datetime.utcnow()
                - datetime.timedelta(days=(n - i))
            ).replace(microsecond=0).isoformat() + "Z"

        out.append(
            {
                "t": t,
                "open": _safe_float(opens[i]),
                "high": _safe_float(highs[i]),
                "low": _safe_float(lows[i]),
                "close": _safe_float(closes[i]),
                "volume": _safe_float(vols[i]),
            }
        )

    return {
        "symbol": symbol,
        "source": candles.get("source", "firestore"),
        "candles": out,
    }


# ---------------------------------------------------------
# Sparkline
# ---------------------------------------------------------

def build_sparkline(
    candles: Dict[str, Any], max_points: int = 60
) -> Optional[Dict[str, Any]]:

    closes = candles.get("close") or []
    if len(closes) < 2:
        return None

    step = max(1, len(closes) // max_points)
    data = [float(v) for v in closes[::step] if v is not None]
    if len(data) < 2:
        return None

    lo, hi = min(data), max(data)
    rng = hi - lo or 1.0

    pts = []
    for i, v in enumerate(data):
        x = i * (100 / max(len(data) - 1, 1))
        y = 32 - ((v - lo) / rng) * 32
        pts.append(f"{x:.1f},{y:.1f}")

    return {
        "path": "M " + " L ".join(pts),
        "min": lo,
        "max": hi,
        "direction": "up" if data[-1] >= data[0] else "down",
        "updatedAt": iso_now(),
    }


# ---------------------------------------------------------
# Quote + OHLCV
# ---------------------------------------------------------

def _build_quote_block(
    symbol: str,
    quote: Dict[str, Any],
    candles: Dict[str, Any],
) -> Dict[str, Any]:

    closes = candles.get("close") or []
    if not closes:
        return {}

    last = len(closes) - 1
    prev = last - 1 if last > 0 else None

    price = _safe_float(quote.get("price")) or _safe_float(closes[last])
    prev_close = _safe_float(closes[prev]) if prev is not None else None

    change_pct = _safe_float(quote.get("changePct"))
    if change_pct is None and price and prev_close:
        change_pct = ((price / prev_close) - 1) * 100

    return {
        "symbol": symbol,
        "price": price,
        "changePct": change_pct,
        "updatedAt": quote.get("updated_at") or iso_now(),
        "ohlcv": {
            "t": _iso_from_ms((candles.get("timestamp") or [None])[last]),
            "open": _safe_float((candles.get("open") or [None])[last]),
            "high": _safe_float((candles.get("high") or [None])[last]),
            "low": _safe_float((candles.get("low") or [None])[last]),
            "close": _safe_float((candles.get("close") or [None])[last]),
            "prevClose": prev_close,
            "volume": _safe_float((candles.get("volume") or [None])[last]),
        },
    }


# ---------------------------------------------------------
# Main Builder (FINAL)
# ---------------------------------------------------------

def build_and_save_stock_ui_doc(
    symbol: str, *, candle_limit: int = 180, news_limit: int = 8
) -> Dict[str, Any]:

    symbol = (symbol or "").upper().strip()
    if not symbol.isalnum():
        raise RuntimeError("Invalid symbol")

    db = get_db()

    # --- BullBrain cache ---
    stock = get_stock(symbol) or {}
    company_name = stock.get("company_name") or stock.get("companyName") or symbol

    # --- Quote cache ---
    quote_ref = (
        db.collection("bullsignals_ai")
        .document("quotes")
        .collection("symbols")
        .document(symbol)
    )
    quote_doc = quote_ref.get()
    quote_cache = quote_doc.to_dict() if quote_doc.exists else {}

    # --- Candles ---
    candles = get_candles(symbol, min_points=candle_limit)
    if not candles or not candles.get("close"):
        raise RuntimeError("Candles unavailable")

    # --- Header ---
    quote_block = _build_quote_block(symbol, quote_cache, candles)

    # --- Technical ---
    technical = build_technical_snapshot(
        symbol,
        stock.get("features_meta") or {},
        quote_block.get("price"),
    )

    # --- Sparkline ---
    sparkline = build_sparkline(candles)

    # --- Candles payload ---
    candles_payload = _build_candles_payload(symbol, candles, candle_limit)

    # --- Smart Pattern (SAFE ADAPTER) ---
    smart_pattern = None
    pattern_dates = []
    pattern_stats = None

    try:
        quote_for_pattern = {
            "price": quote_block.get("price"),
            "changePct": quote_block.get("changePct"),
        }

        sp = detect_smart_pattern(
            stock.get("features_meta") or {},
            quote_for_pattern,
            technical,
        )

        if sp and sp.get("pattern") != "NO CLEAR PATTERN":
            hist = scan_smart_pattern_history(symbol, candles) or {}
            smart_pattern = sp
            pattern_stats = hist

            hf = hist.get("historyForCurrent") or {}
            pattern_dates = (hf.get("samples") or [])[:5]

    except Exception:
        pass

    # --- News ---
    news = fetch_symbol_news(
        symbol=symbol,
        company_name=stock.get("company_name"),
        limit=news_limit,
    )

    # --- Final UI doc ---
    ui_doc: Dict[str, Any] = {
        "schemaVersion": "stockdetail_ui_v1",
        "computedAt": iso_now(),
        "symbol": symbol,
        "companyName": company_name,
        "quote": quote_block,
        "sparkline": sparkline,
        "bullbrain": stock.get("bullbrain"),
        "insights": stock.get("insights"),
        "technical": technical,
        "candles": candles_payload,
        "smartPattern": smart_pattern,
        "patternDates": pattern_dates,
        "patternStats": pattern_stats,
        "news": news,
        "ttl": {
            "quoteSeconds": 30,
            "candlesMinutes": 30,
            "technicalMinutes": 60,
            "signalMinutes": 60,
            "newsMinutes": 30,
        },
    }

    # --- Persist ---
    (
        db.collection("bullsignals_ai")
        .document("stocks")
        .collection("symbols")
        .document(symbol)
        .set(ui_doc, merge=True)
    )

    return ui_doc
