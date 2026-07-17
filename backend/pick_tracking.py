# backend/pick_tracking.py
# =========================================================
# Pick outcome tracking — write side only.
#
# Records every appearance of a stock in a ranked "opportunity" list as its
# own row (no dedup — each cron cycle's appearance is a separate tracked
# pick), with a snapshot of price/reasoning/model view at that moment.
# `source` is generic so other lists (daily movers, etc.) can write here
# later without a schema change. Read side (checking 5d/20d outcomes) is
# not built yet — every horizon starts "pending" and stays that way until a
# future checker fills it in.
# =========================================================

import datetime
from typing import Any, Dict, List, Optional

COL_ROOT = "bullsignals_ai"
PICK_TRACKING_COLLECTION = "pick_tracking"

HORIZONS_TRADING_DAYS = [5, 20]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _build_pick_record(
    date_key: str,
    source: str,
    item: Dict[str, Any],
    stock: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    stock = stock or {}
    quote = stock.get("quote") or {}
    display_intelligence = stock.get("displayIntelligence") or {}
    pattern_history = stock.get("patternHistory") or {}
    days5 = (pattern_history.get("forwardReturns") or {}).get("days5") or {}

    return {
        "symbol": item.get("symbol"),
        "source": source,
        "pick_date": date_key,
        "recorded_at": _now_iso(),

        "pick_price": item.get("price"),
        # quote (not the alpha_watch item) carries the freshness flags —
        # get_canonical_quote() (market_cron.py) can fall back to a stale
        # cached value or a candle-close price when a live fetch fails,
        # always marked needs_refresh=True on that path. Recording it here
        # means a degraded-price pick can be identified/excluded later
        # instead of silently looking identical to a clean one.
        "pick_price_source": quote.get("source"),
        "pick_price_needs_refresh": quote.get("needs_refresh", False),
        "pick_change_pct": item.get("changePct"),

        "pick_score": item.get("score"),
        "pick_setup_label": item.get("setupLabel"),
        "pick_reason": item.get("reason"),
        "pick_why_now": item.get("whyNow"),
        "pick_market_regime": item.get("marketRegime"),
        "pick_factor_scores": item.get("factorScores"),

        "pick_model_view": display_intelligence.get("modelView"),
        "pick_pattern_stats": {
            "pattern": pattern_history.get("pattern"),
            "winRate": days5.get("winRate"),
            "avg": days5.get("avg"),
            "count": days5.get("count"),
        } if days5 else None,

        "horizons": {
            f"{h}d": {
                "trading_days": h,
                "status": "pending",
                "price": None,
                "return_pct": None,
                "checked_at": None,
            }
            for h in HORIZONS_TRADING_DAYS
        },

        "schema_version": "pick_tracking_v1",
    }


def record_picks_for_tracking(
    db,
    date_key: str,
    items: List[Dict[str, Any]],
    stock_docs: List[Dict[str, Any]],
    source: str = "alpha_watch",
) -> int:
    """
    Writes one new row per item, every call — no dedup by design: each
    cron cycle's appearance of a symbol is its own tracked pick event.
    """
    if not items:
        return 0

    stock_by_symbol = {
        (d.get("symbol") or "").upper(): d
        for d in stock_docs
        if isinstance(d, dict) and d.get("symbol")
    }

    collection = (
        db.collection(COL_ROOT)
          .document(PICK_TRACKING_COLLECTION)
          .collection("picks")
    )

    written = 0
    for item in items:
        symbol = (item.get("symbol") or "").upper()
        if not symbol:
            continue
        record = _build_pick_record(date_key, source, item, stock_by_symbol.get(symbol))
        collection.document().set(record)  # auto-generated ID — always a new row
        written += 1

    return written
