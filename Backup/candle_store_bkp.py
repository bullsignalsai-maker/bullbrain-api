# backend/candle_store.py
# ---------------------------------------------------------
# Candle Store — Firestore-backed candle cache with
# delta fetching, throttling, and rate-limit protection
#
# Source of truth for ALL candle usage across the app.
# ---------------------------------------------------------

import os
import time
import random
import datetime
from typing import Dict, Any, Optional

import firebase_admin
from firebase_admin import firestore  # type: ignore
import requests

# ---------------------------------------------------------
# ENV
# ---------------------------------------------------------
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
MAX_DAYS_BACK = 370          # fetch ~1 year initially
CANDLE_TTL_HOURS = 24        # candles considered fresh for 24h
MIN_POINTS_DEFAULT = 120     # minimum candles required
RATE_SLEEP_MIN = 0.15
RATE_SLEEP_MAX = 0.25

# ---------------------------------------------------------
# FIRESTORE INIT
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


# ---------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------
def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------
# RATE LIMIT GUARD
# ---------------------------------------------------------
def rate_limit_sleep():
    time.sleep(random.uniform(RATE_SLEEP_MIN, RATE_SLEEP_MAX))

def normalize_polygon_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")

# ---------------------------------------------------------
# POLYGON FETCHERS
# ---------------------------------------------------------
def _polygon_fetch(symbol: str, start_ts: int, end_ts: int) -> Optional[list]:
    if not POLYGON_KEY:
        return None

    poly_symbol = normalize_polygon_symbol(symbol)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{poly_symbol}/range/1/day/"
        f"{start_ts}/{end_ts}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_KEY}"
    )

    resp = requests.get(url, timeout=12)

    if resp.status_code == 429:
        raise RuntimeError("429 rate limit")

    if not resp.ok:
        return None

    data = resp.json()
    return data.get("results")


def fetch_full_history(symbol: str) -> Optional[list]:
    end = int(utc_now().timestamp() * 1000)
    start = int((utc_now() - datetime.timedelta(days=MAX_DAYS_BACK)).timestamp() * 1000)

    rate_limit_sleep()
    return _polygon_fetch(symbol, start, end)


def fetch_delta_history(symbol: str, last_ts_ms: int) -> Optional[list]:
    # Polygon timestamps are in milliseconds
    start = last_ts_ms + 86400000  # next trading day (1 day in ms)
    end = int(utc_now().timestamp() * 1000)

    rate_limit_sleep()
    return _polygon_fetch(symbol, start, end)


# ---------------------------------------------------------
# FIRESTORE READ / WRITE
# ---------------------------------------------------------
def _read_firestore_candles(symbol: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    doc = (
        db.collection("bullsignals_ai")
          .document("candles")
          .collection("symbols")
          .document(symbol)
          .get()
    )

    if not doc.exists:
        return None

    return doc.to_dict()


def _save_firestore_candles(symbol: str, payload: Dict[str, Any]) -> None:
    db = get_db()
    (
        db.collection("bullsignals_ai")
          .document("candles")
          .collection("symbols")
          .document(symbol)
          .set(payload, merge=True)
    )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------
def _candles_fresh(meta: dict) -> bool:
    try:
        last_fetch = datetime.datetime.fromisoformat(
            meta["last_fetch"].replace("Z", "")
        ).replace(tzinfo=datetime.timezone.utc)

        age = utc_now() - last_fetch
        return age.total_seconds() < CANDLE_TTL_HOURS * 3600
    except Exception:
        return False


def _normalize_polygon_results(results: list) -> Dict[str, list]:
    return {
        "open":   [r.get("o") for r in results],
        "high":   [r.get("h") for r in results],
        "low":    [r.get("l") for r in results],
        "close":  [r.get("c") for r in results],
        "volume": [r.get("v") for r in results],
        "ts":     [r.get("t") for r in results],  # ms
    }


# ---------------------------------------------------------
# Global Polygon rate-limit circuit breaker
# ---------------------------------------------------------
_POLYGON_COOLDOWN_UNTIL = 0  # unix ts
_POLYGON_COOLDOWN_SECS = 10 * 60  # 10 minutes


def _polygon_available() -> bool:
    return time.time() >= _POLYGON_COOLDOWN_UNTIL


def _trip_polygon_cooldown(reason: str):
    global _POLYGON_COOLDOWN_UNTIL
    _POLYGON_COOLDOWN_UNTIL = time.time() + _POLYGON_COOLDOWN_SECS
    print(
        f"[candles] ⚠️ POLYGON DISABLED for {_POLYGON_COOLDOWN_SECS}s | reason={reason}",
        flush=True,
    )

# ---------------------------------------------------------
# PUBLIC API — SINGLE ENTRY POINT
# ---------------------------------------------------------
def get_candles(
    symbol: str,
    min_points: int = MIN_POINTS_DEFAULT,
    **_ignored_kwargs,
) -> Optional[Dict[str, list]]:

    if "min_points" in _ignored_kwargs:
        try:
            min_points = int(_ignored_kwargs["min_points"])
        except Exception:
            pass

    symbol = symbol.upper()

    print(
        f"[candles] ▶ START {symbol} | min_points={min_points}",
        flush=True,
    )

    def normalize_candles(c: Dict[str, list]) -> Optional[Dict[str, list]]:
        try:
            usable = min(
                len(c.get("open", [])),
                len(c.get("high", [])),
                len(c.get("low", [])),
                len(c.get("close", [])),
                len(c.get("volume", [])),
                len(c.get("ts", [])),
            )
            if usable <= 0:
                return None

            return {
                "open": c["open"][-usable:],
                "high": c["high"][-usable:],
                "low": c["low"][-usable:],
                "close": c["close"][-usable:],
                "volume": c["volume"][-usable:],
                "timestamp": c["ts"][-usable:],
            }
        except Exception:
            return None

    # =====================================================
    # 1️⃣ Firestore cache
    # =====================================================
    cached = _read_firestore_candles(symbol)

    if cached:
        candles = cached.get("candles") or {}
        meta = cached.get("meta") or {}

        normalized = normalize_candles(candles)
        usable_len = len(normalized["close"]) if normalized else 0

        print(
            f"[candles] {symbol} | cache-found | usable={usable_len}",
            flush=True,
        )

        if normalized and usable_len >= min_points:

            if _candles_fresh(meta):
                print(
                    f"[candles] {symbol} | cache-fresh → serve",
                    flush=True,
                )
                return normalized

            print(
                f"[candles] {symbol} | cache-stale → delta-attempt",
                flush=True,
            )

            if not _polygon_available():
                print(
                    f"[candles] {symbol} | polygon-disabled → serve-stale",
                    flush=True,
                )
                return normalized

            try:
                last_ts = candles["ts"][-1]
                delta = fetch_delta_history(symbol, last_ts)

                if delta:
                    print(
                        f"[candles] {symbol} | delta-fetched | rows={len(delta)}",
                        flush=True,
                    )
                    norm_delta = _normalize_polygon_results(delta)
                    for k in candles:
                        candles[k].extend(norm_delta.get(k, []))
                else:
                    print(
                        f"[candles] {symbol} | delta-empty (market closed)",
                        flush=True,
                    )

                meta["last_fetch"] = utc_now_iso()
                meta["last_ts"] = candles["ts"][-1]
                meta["count"] = len(candles["close"])

                _save_firestore_candles(symbol, {
                    "candles": candles,
                    "meta": meta,
                })

                return normalize_candles(candles)

            except RuntimeError as e:
                if "429" in str(e):
                    _trip_polygon_cooldown("delta-429")
                    print(
                        f"[candles] {symbol} | 429 → serve-stale",
                        flush=True,
                    )
                    return normalized

            except Exception as e:
                print(
                    f"[candles] {symbol} | delta-error | {e}",
                    flush=True,
                )
                return normalized

        print(
            f"[candles] {symbol} | cache-insufficient → full-fetch",
            flush=True,
        )

    # =====================================================
    # 2️⃣ Full Polygon fetch
    # =====================================================
    if not _polygon_available():
        print(
            f"[candles] {symbol} | polygon-disabled → skip-full-fetch",
            flush=True,
        )
        return None

    try:
        print(
            f"[candles] {symbol} | FULL-FETCH start",
            flush=True,
        )

        full = fetch_full_history(symbol)

        if not full:
            print(
                f"[candles] {symbol} | FULL-FETCH empty",
                flush=True,
            )
            return None

        norm = _normalize_polygon_results(full)
        normalized = normalize_candles(norm)

        if not normalized:
            print(
                f"[candles] {symbol} | normalize-failed",
                flush=True,
            )
            return None

        usable_len = len(normalized["close"])
        print(
            f"[candles] {symbol} | FULL-FETCH done | usable={usable_len}",
            flush=True,
        )

        if usable_len < min_points:
            if usable_len < 60:
                print(
                    f"[candles] {symbol} | <60 candles → reject",
                    flush=True,
                )
                return None
            min_points = 60
            print(
                f"[candles] {symbol} | relaxed-min → 60",
                flush=True,
            )

        _save_firestore_candles(symbol, {
            "candles": norm,
            "meta": {
                "symbol": symbol,
                "source": "polygon",
                "first_ts": norm["ts"][0],
                "last_ts": norm["ts"][-1],
                "last_fetch": utc_now_iso(),
                "count": len(norm["close"]),
            },
        })

        print(
            f"[candles] {symbol} | saved-to-firestore",
            flush=True,
        )

        return normalized

    except RuntimeError as e:
        if "429" in str(e):
            _trip_polygon_cooldown("full-fetch-429")
            print(
                f"[candles] {symbol} | HARD 429 → abort",
                flush=True,
            )
            return None

        print(
            f"[candles] {symbol} | full-fetch-error | {e}",
            flush=True,
        )
        return None
