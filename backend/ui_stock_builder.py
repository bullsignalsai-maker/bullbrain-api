# backend/ui_stock_builder.py
# ---------------------------------------------------------
# UI Stock Builder (Plan B)
# ---------------------------------------------------------
# Purpose:
# - Build ONE canonical Firestore UI document per symbol
# - Compute everything OUTSIDE /stockdetail
# - /stockdetail reads Firestore only (ultra fast)
# ---------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, Optional, List
import datetime
import math
import inspect

from backend.firestore_utils import get_db, utc_now_iso
from backend.stock_repo import get_stock
from backend.candle_store import get_candles
from backend.technicals import build_technical_snapshot
from backend.smart_patterns import detect_smart_pattern, scan_smart_pattern_history
from backend.news_repo import fetch_symbol_news


# -----------------------------
# Helpers
# -----------------------------

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
        dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000.0).replace(microsecond=0)
        return dt.isoformat() + "Z"
    except Exception:
        return None


def _build_candles_payload(
    symbol: str,
    candles: Dict[str, Any],
    limit: int = 180
) -> Dict[str, Any]:
    """
    Convert raw candle_store dict into UI-ready list for chart.
    """
    closes = candles.get("close") or []
    highs = candles.get("high") or []
    lows = candles.get("low") or []
    opens = candles.get("open") or []
    vols = candles.get("volume") or []
    ts_list = candles.get("timestamp") or []

    n = len(closes)
    if n <= 0:
        return {"symbol": symbol, "source": candles.get("source", "unknown"), "candles": []}

    use_n = min(limit, n)
    start = n - use_n

    out: List[Dict[str, Any]] = []
    for i in range(start, n):
        t_iso = None
        if i < len(ts_list) and ts_list[i]:
            t_iso = _iso_from_ms(int(ts_list[i]))
        if not t_iso:
            # fallback: approximate daily sequence
            base = datetime.datetime.utcnow().replace(microsecond=0)
            t_iso = (base - datetime.timedelta(days=(n - 1 - i))).isoformat() + "Z"

        out.append(
            {
                "t": t_iso,
                "open": _safe_float(opens[i]),
                "high": _safe_float(highs[i]),
                "low": _safe_float(lows[i]),
                "close": _safe_float(closes[i]),
                "volume": _safe_float(vols[i]),
            }
        )

    return {"symbol": symbol, "source": candles.get("source", "unknown"), "candles": out}


def build_sparkline(candles: Dict[str, Any], max_points: int = 60) -> Optional[Dict[str, Any]]:
    """
    Sparkline path for mini chart rendering.
    """
    closes = candles.get("close") or []
    if not closes:
        return None

    # Downsample
    step = max(1, len(closes) // max_points)
    data = [float(x) for x in closes[::step] if x is not None]
    if len(data) < 2:
        return None

    lo = min(data)
    hi = max(data)
    rng = (hi - lo) or 1.0

    points = []
    for i, v in enumerate(data):
        x = i * (100 / max(len(data) - 1, 1))
        y = 32 - ((v - lo) / rng) * 32
        points.append(f"{x:.1f},{y:.1f}")

    direction = "up" if data[-1] >= data[0] else "down"

    return {
        "path": "M " + " L ".join(points),
        "direction": direction,
        "min": lo,
        "max": hi,
        "updatedAt": utc_now_iso(),
    }


def _build_quote_ohlcv(
    symbol: str,
    quote_cache: Dict[str, Any],
    candles: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build header quote + OHLCV payload from Firestore quote cache + latest candles.
    """
    closes = candles.get("close") or []
    opens = candles.get("open") or []
    highs = candles.get("high") or []
    lows = candles.get("low") or []
    vols = candles.get("volume") or []
    ts_list = candles.get("timestamp") or []

    price = _safe_float(quote_cache.get("price"))
    change_pct = _safe_float(quote_cache.get("changePct"))

    last_idx = len(closes) - 1 if closes else -1
    prev_idx = last_idx - 1

    last_close = _safe_float(closes[last_idx]) if last_idx >= 0 else None
    prev_close = _safe_float(closes[prev_idx]) if prev_idx >= 0 else None

    # If quote cache missing price, use last close
    if price is None:
        price = last_close

    # If quote cache missing changePct, compute
    if change_pct is None and price is not None and prev_close:
        try:
            change_pct = ((price / prev_close) - 1.0) * 100.0
        except Exception:
            change_pct = None

    o = _safe_float(opens[last_idx]) if last_idx >= 0 else None
    h = _safe_float(highs[last_idx]) if last_idx >= 0 else None
    l = _safe_float(lows[last_idx]) if last_idx >= 0 else None
    c = _safe_float(closes[last_idx]) if last_idx >= 0 else None
    v = _safe_float(vols[last_idx]) if last_idx >= 0 else None

    ts = None
    if last_idx >= 0 and last_idx < len(ts_list) and ts_list[last_idx]:
        ts = _iso_from_ms(int(ts_list[last_idx]))

    return {
        "symbol": symbol,
        "price": price,
        "changePct": change_pct,
        "updatedAt": quote_cache.get("updated_at") or utc_now_iso(),
        "ohlcv": {
            "t": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "prevClose": prev_close,
        },
    }


def _detect_smart_pattern_safe(features: Dict[str, Any], quote_for_pattern: Dict[str, Any], technical: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Your detect_smart_pattern in smart_patterns.py expects:
      detect_smart_pattern(features, quote, technical)
    But we keep it signature-safe anyway.
    """
    try:
        n = len(inspect.signature(detect_smart_pattern).parameters)
    except Exception:
        n = 3

    try:
        if n == 3:
            return detect_smart_pattern(features, quote_for_pattern, technical)
        if n == 2:
            return detect_smart_pattern(features, quote_for_pattern)
        return detect_smart_pattern(features)
    except Exception:
        return None


# ---------------------------------------------------------
# Main Builder
# ---------------------------------------------------------

def build_and_save_stock_ui_doc(symbol: str, *, candle_limit: int = 180, news_limit: int = 8) -> Dict[str, Any]:
    """
    Builds and persists the canonical StockDetail UI document.

    Reads:
      - BullBrain cache: via get_stock(symbol)
      - Quote cache:     /bullsignals_ai/quotes/symbols/{SYMBOL}
      - Candles:         get_candles(symbol)

    Writes:
      - UI doc:          /bullsignals_ai/stocks/symbols/{SYMBOL}
    """
    symbol = (symbol or "").upper().strip()
    if not symbol or not symbol.isalnum():
        raise RuntimeError("Invalid symbol")

    db = get_db()

    # 1) BullBrain cache
    stock = get_stock(symbol) or {}
    company_name = stock.get("company_name") or stock.get("companyName") or symbol

    # 2) Quote cache (Firestore)
    quote_ref = (
        db.collection("bullsignals_ai")
          .document("quotes")
          .collection("symbols")
          .document(symbol)
    )
    quote_doc = quote_ref.get()
    quote_cache = quote_doc.to_dict() if quote_doc.exists else {}

    # 3) Candles (Firestore-backed via candle_store)
    candles = get_candles(symbol, min_points=max(180, candle_limit))
    if not candles or not (candles.get("close") or []):
        raise RuntimeError(f"Candles unavailable for {symbol}")

    # 4) Quote + OHLCV (for header)
    quote_block = _build_quote_ohlcv(symbol, quote_cache, candles)
    last_price = quote_block.get("price")

    # 5) Technical snapshot (from persisted features_meta)
    features_meta = stock.get("features_meta") or {}
    technical = build_technical_snapshot(symbol, features_meta, last_price)

    # 6) Sparkline
    sparkline = build_sparkline(candles)

    # 7) Candles payload for chart (UI-ready list)
    candles_payload = _build_candles_payload(symbol, candles, limit=candle_limit)

    # 8) Smart Pattern + History
    smart_pattern = None
    pattern_dates = []
    pattern_stats = None
    try:
        # Pattern detector uses (features, quote, technical)
        quote_for_pattern = {
            "changePct": quote_block.get("changePct"),
            "current": quote_block.get("price"),
            "price": quote_block.get("price"),
        }
        sp = _detect_smart_pattern_safe(features_meta, quote_for_pattern, technical)
        if sp and isinstance(sp, dict) and sp.get("pattern") and sp.get("pattern") != "NO CLEAR PATTERN":
            hist = scan_smart_pattern_history(symbol, candles) or {}
            pattern_stats = hist
            # hist may contain "historyForCurrent" samples
            hf = hist.get("historyForCurrent") if isinstance(hist, dict) else None
            samples = (hf.get("samples") if isinstance(hf, dict) else None) or []
            pattern_dates = samples[:5] if isinstance(samples, list) else []
            smart_pattern = {
                "pattern": sp.get("pattern"),
                "headline": sp.get("headline") or sp.get("explanation"),
                "winRate": sp.get("winRate"),
                "lastDetectedAt": utc_now_iso(),
                "history": hf or None,
            }
    except Exception:
        smart_pattern = None
        pattern_dates = []
        pattern_stats = None

    # 9) News (standalone, no circular imports)
    news = fetch_symbol_news(symbol, limit=news_limit)

    # 10) Assemble canonical UI doc
    ui_doc: Dict[str, Any] = {
        "schemaVersion": "stockdetail_ui_v1",
        "computedAt": utc_now_iso(),

        # Header basics
        "symbol": symbol,
        "companyName": company_name,

        # Header Quote + OHLCV
        "quote": quote_block,

        # Mini chart
        "sparkline": sparkline,

        # BullBrain + Insights (from cache)
        "bullbrain": stock.get("bullbrain"),
        "insights": stock.get("insights"),

        # Technical section
        "technical": technical,

        # Candles chart section (prebuilt list)
        "candles": candles_payload,

        # Smart pattern section
        "smartPattern": smart_pattern,
        "patternDates": pattern_dates,
        "patternStats": pattern_stats,

        # News section
        "news": news,

        # TTL hints (optional)
        "ttl": {
            "quoteSeconds": quote_cache.get("ttl_seconds", 30),
            "candlesMinutes": 30,
            "signalMinutes": stock.get("compute_ttl_minutes", 60),
            "technicalMinutes": 60,
            "newsMinutes": 30,
        },
    }

    # 11) Persist
    (
        db.collection("bullsignals_ai")
          .document("stocks")
          .collection("symbols")
          .document(symbol)
          .set(ui_doc, merge=True)
    )

    return ui_doc
