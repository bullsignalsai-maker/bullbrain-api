# backend/universe_outcome_tracking.py
# =========================================================
# Unbiased, broad-universe outcome tracking -- one row per REAL_TICKERS
# symbol per day, independent of whether Alpha Watch selected it.
#
# pick_tracking.py answers "how did our PICKS do" -- this answers "how
# does the model's raw view do across the WHOLE universe," which
# pick_tracking structurally cannot answer (it only ever sees the ~10-15
# symbols/day Alpha Watch selects, itself a biased, selection-filtered
# subset). See ml_training_data_readiness_audit memory for the full
# scoping discussion.
#
# Deliberately decoupled from Alpha Watch's opinionated score_stock()
# composite: the primary recorded signal is the neutral raw model view
# (BullBrain's own prob_up/prob_down, via the same _model_view() helper
# stock_display_intelligence.py uses) and detect_regime()'s per-symbol
# classification -- both derived from run_bullbrain_from_inputs(), the
# same canonical "single authority" pipeline every other path in this app
# uses (main.py's own on-demand route, market_cron.py's compute_symbol()).
# factor_scores is recorded too, but only as an optional secondary field,
# computed by calling the six score_* functions directly rather than
# going through score_stock()'s MIN_FINAL_SCORE/diversity-filter gate --
# Alpha Watch's composite scoring formula changed materially 7+ times in
# one session; a future model should learn its own weighting from the
# raw components, not have today's formula's opinions baked in as ground
# truth.
#
# No dedup logic needed here (unlike pick_tracking): this writes exactly
# once per (symbol, date) by construction -- one deliberate daily pass,
# not one row per cron cycle a symbol happens to appear in a ranked list.
#
# Deliberately does NOT call save_stock() -- stays fully isolated from
# the live, user-facing stock_repo collection. Refreshing ~300 rarely-
# viewed symbols' displayed data as a side effect of a training-data job
# is a bigger blast radius than this feature needs.
# =========================================================

import datetime
import os
from typing import Any, Dict, List, Optional

import main as backend
from backend.candle_store import get_candles
from backend.stock_bootstrap import ensure_bullbrain_loaded
from backend.stock_display_intelligence import _model_view
from backend.alpha_watch_logic import (
    detect_regime,
    score_momentum,
    score_trend,
    score_pattern,
    score_bullbrain,
    score_volume,
    score_early_expansion,
)
from backend.quote_repo import get_quote_safe
from backend.market_calendar import load_recent_trading_days, trading_days_elapsed
from symbols_clean import REAL_TICKERS

COL_ROOT = "bullsignals_ai"
UNIVERSE_TRACKING_COLLECTION = "universe_outcome_tracking"
HORIZONS_TRADING_DAYS = [5, 20]

_WRITE_BATCH_SIZE = 400

# Same rationale as pick_tracking.py's CHECKER_WINDOW_DAYS: comfortably
# covers the 20-trading-day (~28 calendar day) max horizon with margin.
CHECKER_WINDOW_DAYS = 35


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _today_str() -> str:
    return datetime.datetime.utcnow().date().isoformat()


def _doc_id(symbol: str, date_key: str) -> str:
    return f"{symbol}_{date_key}"


def _build_factor_scores(core: Dict[str, Any], feat_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Calls the six score_* functions directly -- NOT score_stock() -- so
    this never runs into score_stock()'s `if final_score < MIN_FINAL_
    SCORE: return None` early-return. That gate (plus diversity
    filtering, market-momentum bonuses, pullback penalties) is Alpha
    Watch product logic, not a neutral feature computation; every scanned
    symbol gets its six component scores here regardless of whether it
    would ever qualify as a pick.
    """
    stock_like = {
        "bullbrain": core.get("bullbrain"),
        "decision": core.get("decision"),
        "pattern": core.get("pattern"),
        "patternHistory": core.get("patternHistory"),
        "patternBias": core.get("patternBias"),
        "features_meta": feat_dict,
        "trend_pct_20d": core.get("trend_pct_20d"),
    }
    return {
        "momentum": score_momentum(feat_dict),
        "trend": score_trend(feat_dict, trend_pct_20d=core.get("trend_pct_20d")),
        "pattern": score_pattern(stock_like),
        "bullbrain": score_bullbrain(stock_like),
        "volume": score_volume(feat_dict, vol_z_corrected=core.get("vol_zscore_20_corrected")),
        "early_expansion": score_early_expansion(feat_dict),
    }


def build_snapshot(symbol: str, date_key: str) -> Optional[Dict[str, Any]]:
    """
    Computes one symbol's neutral daily snapshot. Returns None (not
    raises) on any missing input -- same degrade-quietly convention as
    every other fetcher in this app (safe_json, quote/candle providers).
    Self-sufficient (calls ensure_bullbrain_loaded() itself, same as
    bootstrap_stock()) -- safe to call standalone, not just from
    record_daily_universe_snapshot()'s batch loop.
    """
    symbol = symbol.upper()
    ensure_bullbrain_loaded()

    candles = get_candles(symbol)
    if not candles:
        return None

    try:
        feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles)
    except Exception:
        return None
    if feats_vec is None or not feat_dict:
        return None

    try:
        core = backend.run_bullbrain_from_inputs(
            symbol, candles_arrays=candles, feat_dict=feat_dict
        )
    except Exception:
        return None

    bull_raw = (core.get("bullbrain") or {}).get("raw") or {}
    up = bull_raw.get("prob_up")
    down = bull_raw.get("prob_down")

    quote = get_quote_safe(symbol) or {}
    snapshot_price = quote.get("price")

    return {
        "symbol": symbol,
        "date": date_key,
        "recorded_at": _now_iso(),
        "snapshot_price": float(snapshot_price) if isinstance(snapshot_price, (int, float)) else None,
        "model_view": _model_view(
            float(up) if isinstance(up, (int, float)) else None,
            float(down) if isinstance(down, (int, float)) else None,
        ),
        "market_regime": detect_regime({
            "trend_pct_20d": core.get("trend_pct_20d"),
            "features_meta": feat_dict,
        }),
        "gate_signal": (core.get("decision") or {}).get("finalSignal"),
        "factor_scores": _build_factor_scores(core, feat_dict),
        # Which deploy produced this snapshot -- same RENDER_GIT_COMMIT
        # tagging as pick_tracking.py's pick_code_version, applied here
        # from day one rather than bolted on later.
        "pick_code_version": os.getenv("RENDER_GIT_COMMIT"),
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
        "schema_version": "universe_outcome_tracking_v1",
    }


def _get_state(db) -> Dict[str, Any]:
    doc = db.collection(COL_ROOT).document(UNIVERSE_TRACKING_COLLECTION).get()
    return doc.to_dict() if doc.exists else {}


def record_daily_universe_snapshot(
    db,
    date_key: Optional[str] = None,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Once-per-day pass over the full REAL_TICKERS universe (or an
    explicit `symbols` override, for a partial/manual run). Guarded by
    its own state doc, independent of pick_tracking's own once/day
    guards -- protects against an overlapping cron invocation doing this
    (potentially multi-minute, heaviest on its first run) work twice in
    one day.
    """
    date_key = date_key or _today_str()
    state = _get_state(db)
    if state.get("last_run_date") == date_key:
        return {"skipped": True, "reason": "already_ran_today"}

    ensure_bullbrain_loaded()

    target_symbols = symbols or REAL_TICKERS
    collection = (
        db.collection(COL_ROOT)
          .document(UNIVERSE_TRACKING_COLLECTION)
          .collection("symbols")
    )

    written = 0
    skipped = 0
    batch = db.batch()
    batch_count = 0

    for symbol in target_symbols:
        snapshot = build_snapshot(symbol, date_key)
        if snapshot is None:
            skipped += 1
            continue

        batch.set(collection.document(_doc_id(symbol, date_key)), snapshot, merge=True)
        batch_count += 1
        written += 1

        if batch_count >= _WRITE_BATCH_SIZE:
            batch.commit()
            batch = db.batch()
            batch_count = 0

    if batch_count:
        batch.commit()

    stats = {
        "date": date_key,
        "scanned": len(target_symbols),
        "written": written,
        "skipped": skipped,
    }
    db.collection(COL_ROOT).document(UNIVERSE_TRACKING_COLLECTION).set(
        {"last_run_date": date_key, "last_run_at": _now_iso(), "last_run_stats": stats},
        merge=True,
    )
    return stats


# =========================================================
# Outcome checker — read side. Mirrors pick_tracking.check_pending_picks(),
# except _lookup_current_price() reads quote_repo's lighter quote cache
# instead of the full stock_repo doc.
# =========================================================

def _lookup_current_price(symbol: str) -> Optional[float]:
    quote = get_quote_safe(symbol)
    price = (quote or {}).get("price")
    return float(price) if isinstance(price, (int, float)) else None


def check_pending_universe_snapshots(db) -> Dict[str, Any]:
    """
    Once-per-day outcome checker: finds snapshots whose 5d/20d horizon
    has elapsed and is still "pending", fills in the real return, or
    marks "unavailable" if a current price can't be found. Safe to call
    more than once on the same UTC calendar day -- guarded via the same
    state doc record_daily_universe_snapshot() uses, under a different
    key so the two don't clobber each other's stats.
    """
    today = _today_str()
    state = _get_state(db)
    if state.get("last_checker_run_date") == today:
        return {"skipped": True, "reason": "already_ran_today"}

    trading_day_set = load_recent_trading_days(lookback_days=CHECKER_WINDOW_DAYS + 10)
    trading_day_set.add(today)

    window_start = (
        datetime.date.fromisoformat(today) - datetime.timedelta(days=CHECKER_WINDOW_DAYS)
    ).isoformat()

    collection = (
        db.collection(COL_ROOT)
          .document(UNIVERSE_TRACKING_COLLECTION)
          .collection("symbols")
    )
    query = collection.where("date", ">=", window_start).where("date", "<=", today)

    scanned = 0
    checked = 0
    unavailable = 0
    updated_docs = 0

    for doc in query.stream():
        scanned += 1
        data = doc.to_dict()
        symbol = data.get("symbol")
        date_key = data.get("date")
        snapshot_price = data.get("snapshot_price")
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

            elapsed = trading_days_elapsed(date_key, today, trading_day_set)
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
                    round((current_price / snapshot_price - 1) * 100, 2)
                    if isinstance(snapshot_price, (int, float)) and snapshot_price
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
    db.collection(COL_ROOT).document(UNIVERSE_TRACKING_COLLECTION).set(
        {
            "last_checker_run_date": today,
            "last_checker_run_at": _now_iso(),
            "last_checker_stats": stats,
        },
        merge=True,
    )
    return stats
