# backend/market_regime_history.py
# =========================================================
# Independent, market-wide regime signal -- daily snapshot.
#
# Every existing "regime" value in this app is derived, not independent:
# alpha_watch_logic.py's detect_regime() is computed PER-SYMBOL from that
# symbol's own trend/volatility features (there is no SPY/VIX-level
# classifier anywhere in this codebase), and the daily Alpha Watch
# payload's top-level market_regime is just the mode of those per-symbol
# values among that day's SELECTED picks -- a selection-biased proxy, not
# a signal that exists independent of which symbols happened to qualify.
# See bullbrain_calibration_check memory's infra audit (finding 5).
#
# This module is the first real fix: SPY close/change (broad-market
# proxy) plus VIX (options-market-implied volatility), logged once per
# trading day regardless of which symbols got picked or how the pick-
# selection logic changes over time.
#
# Deliberately NOT classified into a regime label (RISK_ON/RISK_OFF/etc)
# yet -- picking thresholds today would just be another unvalidated guess
# like several already flagged elsewhere in this codebase (e.g.
# _MARKET_CONTEXT_POSITIVE=58, documented as "provisional... until
# checked on more dates"). Log the raw numbers now; calibrate real
# thresholds once there's enough history to do that honestly.
# =========================================================

import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from backend.quote_repo import get_quote_safe
from backend.candle_store import get_candles

COL_ROOT = "bullsignals_ai"
REGIME_HISTORY_COLLECTION = "market_regime_history"

# Unauthenticated Yahoo Finance chart endpoint -- the SAME source already
# used live in this exact codebase for VIX by /stats/live and
# /market-mood (main.py). No API key, not a new dependency. Verified live
# 2026-09-22 (returned VIX=14.21, a real current value) before wiring
# this in.
_VIX_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
_VIX_FETCH_TIMEOUT_SECONDS = 8


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _fetch_vix_close() -> Optional[float]:
    """
    Degrades quietly, same convention as every other fetcher in this app
    (safe_json, quote/candle providers) -- returns None on any failure
    rather than raising, so a snapshot with vix_close=None is still
    written (with spy_realized_vol_20d_pct as a real, independent
    fallback signal) instead of losing the whole day.
    """
    try:
        resp = requests.get(
            _VIX_CHART_URL,
            timeout=_VIX_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = resp.json()
        meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        return float(price) if isinstance(price, (int, float)) else None
    except Exception:
        return None


_SPY_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"


def _realized_vol_from_closes(closes: List[float]) -> Optional[float]:
    """Same daily_ret.rolling(20).std()*100 convention as
    features_meta.volatility_20d elsewhere in this app (main.py) and
    scripts/backfill_market_regime_history.py's _realized_vol_20d()."""
    if len(closes) < 21:
        return None

    recent = closes[-21:]
    daily_returns = [
        (recent[i] / recent[i - 1] - 1.0) * 100.0
        for i in range(1, len(recent))
        if recent[i - 1]
    ]
    if len(daily_returns) < 2:
        return None

    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    return round(variance ** 0.5, 3)


def _spy_closes_from_candle_store(date_key: str) -> Optional[List[float]]:
    """
    Polygon-backed candle_store closes, accepted only when the last bar IS
    date_key's -- a stale cached series (served silently during a Polygon
    429 cooldown) would otherwise yield yesterday's vol stamped as today's.
    """
    candles = get_candles("SPY", min_points=21)
    if not candles:
        return None

    closes = candles.get("close") or []
    stamps = candles.get("timestamp") or []
    if len(closes) < 21 or not stamps:
        return None

    last_date = datetime.datetime.utcfromtimestamp(stamps[-1] / 1000).date().isoformat()
    if last_date != date_key:
        print(
            f"[regime] SPY candle_store last bar={last_date} != {date_key} → yahoo fallback",
            flush=True,
        )
        return None
    return closes


def _spy_closes_from_yahoo(date_key: str) -> Optional[List[float]]:
    """
    Same unauthenticated Yahoo chart source as _fetch_vix_close() and the
    9/21-9/22 backfill. Needs no Polygon quota -- the Polygon key's plan is
    capped at 5 requests/min (confirmed 2026-09-25: 6th request -> 429),
    which final_close_intelligence routinely exhausts before this runs.
    Uses meta.regularMarketPrice for the latest bar when Yahoo's close
    array has a null there (same observed quirk the backfill handles).
    """
    try:
        resp = requests.get(
            _SPY_CHART_URL,
            params={"range": "3mo", "interval": "1d"},
            timeout=_VIX_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        result = ((resp.json().get("chart") or {}).get("result") or [{}])[0]
    except Exception as e:
        print(f"[regime] SPY yahoo fetch failed | {e}", flush=True)
        return None

    timestamps = result.get("timestamp") or []
    raw_closes = (result.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
    meta = result.get("meta") or {}
    meta_price = meta.get("regularMarketPrice")
    meta_time = meta.get("regularMarketTime")
    meta_date = (
        datetime.datetime.utcfromtimestamp(meta_time).date().isoformat()
        if isinstance(meta_time, (int, float)) else None
    )

    by_date: Dict[str, float] = {}
    for ts, close in zip(timestamps, raw_closes):
        d = datetime.datetime.utcfromtimestamp(ts).date().isoformat()
        if close is None and d == meta_date and isinstance(meta_price, (int, float)):
            close = meta_price
        if close is not None and d <= date_key:
            by_date[d] = float(close)

    if date_key not in by_date:
        print(f"[regime] SPY yahoo series has no bar for {date_key}", flush=True)
        return None
    return [by_date[d] for d in sorted(by_date)]


def _spy_realized_vol_20d_pct(date_key: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Backward-looking companion to VIX's options-implied measure -- always
    computed, not just an emergency substitute when the VIX fetch fails,
    since it's a genuinely different signal (no options-market
    expectation baked in). Returns (value, source).

    Polygon candle_store first (so SPY is cached like every other
    symbol), Yahoo as fallback. Before this fallback existed the field
    was null on every live write: SPY had no cached candle doc, and the
    Polygon 429 cooldown tripped earlier in final_close_intelligence made
    candle_store skip the first-ever full fetch outright.
    """
    closes = _spy_closes_from_candle_store(date_key)
    if closes:
        return _realized_vol_from_closes(closes), "polygon_candle_store"

    closes = _spy_closes_from_yahoo(date_key)
    if closes:
        return _realized_vol_from_closes(closes), "yahoo_chart_api"

    return None, None


def build_market_regime_snapshot(date_key: str) -> Dict[str, Any]:
    spy_quote = get_quote_safe("SPY") or {}
    spy_close = spy_quote.get("price")
    spy_change_pct = spy_quote.get("changePct")

    vix_close = _fetch_vix_close()
    realized_vol, realized_vol_source = _spy_realized_vol_20d_pct(date_key)

    return {
        "date": date_key,
        "recorded_at": _now_iso(),
        "spy_close": float(spy_close) if isinstance(spy_close, (int, float)) else None,
        "spy_change_pct": float(spy_change_pct) if isinstance(spy_change_pct, (int, float)) else None,
        "vix_close": vix_close,
        "vix_source": "yahoo_chart_api" if vix_close is not None else None,
        "spy_realized_vol_20d_pct": realized_vol,
        "spy_realized_vol_source": realized_vol_source,
        "schema_version": "market_regime_history_v1",
    }


def record_daily_market_regime(db, date_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Once-per-day snapshot of independent, market-wide condition data --
    unlike detect_regime()'s per-symbol classification, this doesn't
    depend on which symbols got picked or scored well that day. Safe to
    call more than once on the same date: doc ID is the date itself, so a
    re-run just overwrites with that call's freshest numbers -- same
    idempotency pattern as persist_alpha_watch()'s alpha_watch_history
    write and save_accuracy_snapshot().
    """
    date_key = date_key or datetime.datetime.utcnow().date().isoformat()
    snapshot = build_market_regime_snapshot(date_key)

    (
        db.collection(COL_ROOT)
          .document(REGIME_HISTORY_COLLECTION)
          .collection("days")
          .document(date_key)
          .set(snapshot, merge=True)
    )

    return snapshot


def get_market_regime_day(db, date_key: str) -> Optional[Dict[str, Any]]:
    doc = (
        db.collection(COL_ROOT)
          .document(REGIME_HISTORY_COLLECTION)
          .collection("days")
          .document(date_key)
          .get()
    )
    return doc.to_dict() if doc.exists else None
