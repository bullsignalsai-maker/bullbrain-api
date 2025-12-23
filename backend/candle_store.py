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


# ---------------------------------------------------------
# POLYGON FETCHERS
# ---------------------------------------------------------
def _polygon_fetch(symbol: str, start_ts: int, end_ts: int) -> Optional[list]:
    if not POLYGON_KEY:
        return None

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
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
    end = int(utc_now().timestamp())
    start = int((utc_now() - datetime.timedelta(days=MAX_DAYS_BACK)).timestamp())

    rate_limit_sleep()
    return _polygon_fetch(symbol, start, end)


def fetch_delta_history(symbol: str, last_ts_ms: int) -> Optional[list]:
    # Polygon timestamps are in milliseconds
    start = int(last_ts_ms / 1000) + 1
    end = int(utc_now().timestamp())

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
# PUBLIC API — SINGLE ENTRY POINT
# ---------------------------------------------------------
def get_candles(
    symbol: str,
    min_points: int = MIN_POINTS_DEFAULT,
) -> Optional[Dict[str, list]]:
    """
    Returns normalized candle dict:
      { open, high, low, close, volume, timestamp }

    Firestore is source of truth.
    Polygon is used ONLY for delta or initial fill.
    """

    symbol = symbol.upper()

    # -----------------------------------------------------
    # 1) Read cache
    # -----------------------------------------------------
    cached = _read_firestore_candles(symbol)

    if cached:
        candles = cached.get("candles", {})
        meta = cached.get("meta", {})

        # Validate candle count
        if candles and len(candles.get("close", [])) >= min_points:
            # Fresh → return immediately
            if _candles_fresh(meta):
                return {
                    "open": candles["open"],
                    "high": candles["high"],
                    "low": candles["low"],
                    "close": candles["close"],
                    "volume": candles["volume"],
                    "timestamp": candles["ts"],
                }

            # Stale → fetch delta
            try:
                last_ts = candles["ts"][-1]
                delta = fetch_delta_history(symbol, last_ts)

                if delta:
                    norm = _normalize_polygon_results(delta)

                    for k in candles:
                        candles[k].extend(norm[k])

                    meta["last_fetch"] = utc_now_iso()

                    _save_firestore_candles(symbol, {
                        "candles": candles,
                        "meta": meta,
                    })

                return {
                    "open": candles["open"],
                    "high": candles["high"],
                    "low": candles["low"],
                    "close": candles["close"],
                    "volume": candles["volume"],
                    "timestamp": candles["ts"],
                }

            except RuntimeError as e:
                if "429" in str(e):
                    # rate-limited → serve stale but valid data
                    return {
                        "open": candles["open"],
                        "high": candles["high"],
                        "low": candles["low"],
                        "close": candles["close"],
                        "volume": candles["volume"],
                        "timestamp": candles["ts"],
                    }
                return None

    # -----------------------------------------------------
    # 2) No cache → full fetch
    # -----------------------------------------------------
    try:
        full = fetch_full_history(symbol)
        if not full or len(full) < min_points:
            return None

        norm = _normalize_polygon_results(full)

        payload = {
            "candles": norm,
            "meta": {
                "symbol": symbol,
                "source": "polygon",
                "first_ts": norm["ts"][0],
                "last_ts": norm["ts"][-1],
                "last_fetch": utc_now_iso(),
                "count": len(norm["close"]),
            },
        }

        _save_firestore_candles(symbol, payload)

        return {
            "open": norm["open"],
            "high": norm["high"],
            "low": norm["low"],
            "close": norm["close"],
            "volume": norm["volume"],
            "timestamp": norm["ts"],
        }

    except RuntimeError as e:
        if "429" in str(e):
            return None
        return None
