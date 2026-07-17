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

from backend.market_calendar import load_recent_trading_days, trading_days_elapsed
from backend.stock_repo import get_stock

COL_ROOT = "bullsignals_ai"
PICK_TRACKING_COLLECTION = "pick_tracking"

HORIZONS_TRADING_DAYS = [5, 20]

# Calendar-day width of the pick_date range the checker scans each run.
# Comfortably covers the 20-trading-day (~28 calendar day) max horizon
# with margin -- see homescreen_replacement_track_scoping memory for the
# volume math confirming this stays cheap even at 12 months of history,
# since cost is bounded by this window, not by total collection size.
CHECKER_WINDOW_DAYS = 35


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


# =========================================================
# Outcome checker — read side.
# =========================================================

def _today_str() -> str:
    # Matches how pick_date/date_key are written in record_picks_for_tracking
    # (UTC date) -- checked safe against the actual run windows
    # persist_alpha_watch() executes in (9:30 AM-4:45 PM ET is always the
    # same UTC calendar day), so this intentionally isn't switched to ET.
    return datetime.datetime.utcnow().date().isoformat()


def _get_checker_state(db) -> Dict[str, Any]:
    doc = db.collection(COL_ROOT).document(PICK_TRACKING_COLLECTION).get()
    return doc.to_dict() if doc.exists else {}


def _mark_checker_ran(db, date_key: str, stats: Dict[str, Any]) -> None:
    db.collection(COL_ROOT).document(PICK_TRACKING_COLLECTION).set(
        {
            "last_checker_run_date": date_key,
            "last_checker_run_at": _now_iso(),
            "last_checker_stats": stats,
        },
        merge=True,
    )


def _lookup_current_price(symbol: str) -> Optional[float]:
    stock = get_stock(symbol)
    if not stock:
        return None
    price = (stock.get("quote") or {}).get("price")
    return float(price) if isinstance(price, (int, float)) else None


def check_pending_picks(db) -> Dict[str, Any]:
    """
    Once-per-day outcome checker: finds picks whose 5d/20d horizon has
    elapsed and is still "pending", fills in the real return, or marks
    "unavailable" if current price data can't be found (delisted, dropped
    from the tracked universe, etc.) -- never leaves a due row silently
    unresolved.

    Safe to call more than once on the same UTC calendar day -- guarded
    internally via a state doc, so only the first call each day does work.
    Bounded to a fixed-size lookback window (CHECKER_WINDOW_DAYS), so cost
    stays flat regardless of how large the collection grows over time.
    """
    today = _today_str()
    state = _get_checker_state(db)
    if state.get("last_checker_run_date") == today:
        return {"skipped": True, "reason": "already_ran_today"}

    # Today is trusted as a trading day by construction: the caller only
    # invokes this when market_cron's own gating has already confirmed
    # today is a live trading day. Added explicitly rather than relying on
    # today's own candle being published yet, which it usually isn't at
    # checker run time.
    trading_day_set = load_recent_trading_days()
    trading_day_set.add(today)

    window_start = (
        datetime.date.fromisoformat(today) - datetime.timedelta(days=CHECKER_WINDOW_DAYS)
    ).isoformat()

    collection = (
        db.collection(COL_ROOT)
          .document(PICK_TRACKING_COLLECTION)
          .collection("picks")
    )
    query = collection.where("pick_date", ">=", window_start).where("pick_date", "<=", today)

    scanned = 0
    checked = 0
    unavailable = 0
    updated_docs = 0

    for doc in query.stream():
        scanned += 1
        data = doc.to_dict()
        symbol = data.get("symbol")
        pick_date = data.get("pick_date")
        pick_price = data.get("pick_price")
        horizons = data.get("horizons") or {}

        updates: Dict[str, Any] = {}
        current_price: Optional[float] = None
        price_lookup_attempted = False

        for horizon_key, horizon in horizons.items():
            if not isinstance(horizon, dict) or horizon.get("status") != "pending":
                continue

            trading_days_needed = horizon.get("trading_days")
            if not isinstance(trading_days_needed, int):
                continue

            elapsed = trading_days_elapsed(pick_date, today, trading_day_set)
            if elapsed < trading_days_needed:
                continue  # not due yet

            if not price_lookup_attempted:
                current_price = _lookup_current_price(symbol)
                price_lookup_attempted = True

            prefix = f"horizons.{horizon_key}"
            if current_price is None:
                updates[f"{prefix}.status"] = "unavailable"
                updates[f"{prefix}.checked_at"] = _now_iso()
                unavailable += 1
            else:
                return_pct = (
                    round((current_price / pick_price - 1) * 100, 2)
                    if isinstance(pick_price, (int, float)) and pick_price
                    else None
                )
                updates[f"{prefix}.status"] = "checked"
                updates[f"{prefix}.price"] = current_price
                updates[f"{prefix}.return_pct"] = return_pct
                updates[f"{prefix}.checked_at"] = _now_iso()
                checked += 1

        if updates:
            doc.reference.update(updates)
            updated_docs += 1

    stats = {
        "scanned": scanned,
        "updated_docs": updated_docs,
        "checked": checked,
        "unavailable": unavailable,
    }
    _mark_checker_ran(db, today, stats)
    return stats
