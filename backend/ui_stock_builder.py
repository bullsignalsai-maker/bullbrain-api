# ---------------------------------------------------------
# UI Stock Builder
# ---------------------------------------------------------
# Purpose:
# - Build ONE canonical Firestore UI document per symbol
# - Firestore-first (no frontend logic, no runtime compute leakage)
# - Safe to call from cron or lazy API
#
# Output:
#   /bullsignals_ai/stocks/{SYMBOL}
# ---------------------------------------------------------

from typing import Dict, Any, Optional, List

import firebase_admin
from firebase_admin import firestore  # type: ignore

from backend.stock_repo import get_stock
from backend.candle_store import get_candles
from backend.firestore_utils import utc_now_iso
from backend.technicals import build_technical_snapshot
from backend.smart_patterns import scan_smart_pattern_history
from backend.news import fetch_symbol_news


# ---------------------------------------------------------
# Firestore handle
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


# ---------------------------------------------------------
# Sparkline builder (UI-ready, zero frontend math)
# ---------------------------------------------------------
def build_sparkline(
    candles: Dict[str, List[float]],
    max_points: int = 60,
) -> Optional[Dict[str, Any]]:
    closes = candles.get("close", [])
    if not closes:
        return None

    step = max(1, len(closes) // max_points)
    data = closes[::step]

    lo = min(data)
    hi = max(data)
    rng = hi - lo or 1.0

    points = []
    for i, v in enumerate(data):
        x = i * (100 / max(len(data) - 1, 1))
        y = 32 - ((v - lo) / rng) * 32
        points.append(f"{x:.1f},{y:.1f}")

    return {
        "path": "M " + " L ".join(points),
        "direction": "up" if data[-1] >= data[0] else "down",
        "updatedAt": utc_now_iso(),
    }


# ---------------------------------------------------------
# Main Builder (CANONICAL)
# ---------------------------------------------------------
def build_and_save_stock_ui_doc(symbol: str) -> Dict[str, Any]:
    """
    Builds and persists the canonical Stock UI document.

    Reads:
    - /stocks/{SYMBOL}                 (BullBrain cache)
    - /bullsignals_ai/quotes/symbols   (quote cache)
    - /bullsignals_ai/candles          (raw candles)

    Writes:
    - /bullsignals_ai/stocks/{SYMBOL}
    """

    symbol = symbol.upper().strip()
    db = get_db()

    # -------------------------------------------------
    # 1️⃣ BullBrain model cache
    # -------------------------------------------------
    stock = get_stock(symbol)
    if not stock:
        raise RuntimeError(f"No BullBrain cache found for {symbol}")

    features = stock.get("features_meta")
    if not features:
        raise RuntimeError(f"No features available for {symbol}")

    # -------------------------------------------------
    # 2️⃣ Quote (Firestore-only)
    # -------------------------------------------------
    quote_ref = (
        db.collection("bullsignals_ai")
        .document("quotes")
        .collection("symbols")
        .document(symbol)
    )

    quote_doc = quote_ref.get()
    quote = quote_doc.to_dict() if quote_doc.exists else {}

    price = quote.get("price")
    change_pct = quote.get("changePct")

    # -------------------------------------------------
    # 3️⃣ Candles (Firestore-backed)
    # -------------------------------------------------
    candles = get_candles(symbol, min_points=180)
    if not candles or not candles.get("close"):
        raise RuntimeError(f"Candles unavailable for {symbol}")

    last_close = float(price) if price is not None else float(candles["close"][-1])

    # -------------------------------------------------
    # 4️⃣ Technical snapshot (✅ correct signature)
    # -------------------------------------------------
    technical = build_technical_snapshot(
        symbol=symbol,
        features=features,
        last_close=last_close,
    )

    # -------------------------------------------------
    # 5️⃣ Sparkline (mini chart)
    # -------------------------------------------------
    sparkline = build_sparkline(candles)

    # -------------------------------------------------
    # 6️⃣ Smart pattern (history-based, canonical)
    # -------------------------------------------------
    smart_pattern = None
    try:
        ph = scan_smart_pattern_history(symbol, candles)
        cp = ph.get("currentPattern") if isinstance(ph, dict) else None
        hist = ph.get("historyForCurrent") if isinstance(ph, dict) else None

        if cp and cp.get("pattern"):
            smart_pattern = {
                "pattern": cp.get("pattern"),
                "headline": cp.get("headline"),
                "winRate": cp.get("winRate"),
                "occurrences": hist.get("occurrences", 0) if hist else 0,
                "samples": hist.get("samples", []) if hist else [],
                "forwardReturns": hist.get("forwardReturns", {}) if hist else {},
                "lastDetectedAt": utc_now_iso(),
            }
    except Exception:
        smart_pattern = None

    # -------------------------------------------------
    # 7️⃣ News (thin wrapper)
    # -------------------------------------------------
    news = fetch_symbol_news(symbol, limit=6)

    # -------------------------------------------------
    # 8️⃣ Assemble UI document (STABLE SCHEMA)
    # -------------------------------------------------
    ui_doc = {
        "symbol": symbol,
        "companyName": stock.get("company_name", symbol),

        "quote": {
            "price": price,
            "changePct": change_pct,
            "updatedAt": quote.get("updated_at"),
        },

        "sparkline": sparkline,

        "bullbrain": stock.get("bullbrain"),
        "insights": stock.get("insights"),

        "technical": technical,
        "smartPattern": smart_pattern,
        "news": news,

        "meta": {
            "computedAt": utc_now_iso(),
            "schemaVersion": "ui_v1",
        },

        "ttl": {
            "quoteSeconds": quote.get("ttl_seconds", 30),
            "signalMinutes": stock.get("compute_ttl_minutes", 60),
            "technicalMinutes": 60,
            "newsMinutes": 30,
        },
    }

    # -------------------------------------------------
    # 9️⃣ Persist (MERGE-safe)
    # -------------------------------------------------
    (
        db.collection("bullsignals_ai")
        .document("stocks")
        .collection("symbols")
        .document(symbol)
        .set(ui_doc, merge=True)
    )

    return ui_doc
