# =========================================================
# BullSignalsAI — Market Intelligence Cron
#
# Schedule:
#   */15 * * * 1-5
#
# Responsibilities:
#   1) Refresh global stock intelligence (/stocks/{SYMBOL})
#   2) Use user relevance (active_symbols)
#   3) Rotate SP500 discovery (discovery_cursor)
#   4) Always include MAG-7
#   5) Build Hotlist + BearWatch
#   6) Update Homescreen snapshot (MAG-7 only)
#
# DOES NOT:
#   - Fetch news
#   - Do sentiment
#   - Do UI grouping
# =========================================================

import datetime
import math
import random
import time
from typing import Dict, Any, List

import firebase_admin
from firebase_admin import firestore

import main as backend
from symbols_clean import REAL_TICKERS, COMPANY_NAMES

from backend.candle_store import get_candles
from backend.bull_insights import generate_bull_insights


# =========================================================
# CONSTANTS
# =========================================================

MAG7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

ACTIVE_SYMBOL_LIMIT = 60
DISCOVERY_LIMIT = 50
TOTAL_SCAN_LIMIT = 120

DISCOVERY_SHARDS = 8

COL_ROOT = "bullsignals_ai"
COL_STOCKS = "stocks"
DOC_ACTIVE = "active_symbols"
DOC_DISCOVERY = "discovery_cursor"
DOC_HOTLIST = "market_hotlist"
DOC_BEARWATCH = "market_bearwatch"
DOC_HOMESCREEN = "homescreen_snapshot"


# =========================================================
# FIRESTORE / TIME HELPERS
# =========================================================

def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# =========================================================
# MODEL LOADER
# =========================================================

def ensure_bullbrain_loaded():
    if backend.bullbrain_model is None:
        backend.bullbrain_model = backend.load_bullbrain_model()


# =========================================================
# ACTIVE SYMBOLS (READ-ONLY)
# =========================================================

def load_active_symbols() -> Dict[str, Dict[str, Any]]:
    db = get_db()
    snap = db.collection(COL_ROOT).document(DOC_ACTIVE).get()
    if not snap.exists:
        return {}
    return (snap.to_dict() or {}).get("symbols", {})


def rank_active_symbols(active: Dict[str, Dict[str, Any]]) -> List[str]:
    now = datetime.datetime.now(datetime.timezone.utc)

    def score(meta: Dict[str, Any]) -> float:
        base = float(meta.get("count", 0))
        last_seen = meta.get("last_seen")
        boost = 0.0
        try:
            if last_seen:
                ts = datetime.datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                mins = (now - ts).total_seconds() / 60.0
                if mins < 60:
                    boost = (60 - mins) / 60.0
        except Exception:
            pass
        return base + boost

    ranked = sorted(
        active.items(),
        key=lambda kv: score(kv[1]),
        reverse=True,
    )

    return [sym for sym, _ in ranked[:ACTIVE_SYMBOL_LIMIT]]


# =========================================================
# DISCOVERY CURSOR (CRON-OWNED)
# =========================================================

def get_discovery_symbols() -> List[str]:
    db = get_db()
    ref = db.collection(COL_ROOT).document(DOC_DISCOVERY)

    snap = ref.get()
    data = snap.to_dict() or {}
    shard_index = int(data.get("shard_index", 0))

    total = len(REAL_TICKERS)
    shard_size = math.ceil(total / DISCOVERY_SHARDS)

    start = shard_index * shard_size
    end = min(start + shard_size, total)

    shard = REAL_TICKERS[start:end][:DISCOVERY_LIMIT]

    next_index = (shard_index + 1) % DISCOVERY_SHARDS
    ref.set(
        {
            "shard_index": next_index,
            "total_shards": DISCOVERY_SHARDS,
            "updated_at": utc_now_iso(),
        },
        merge=True,
    )

    return shard


# =========================================================
# SCAN UNIVERSE
# =========================================================

def build_scan_universe() -> List[str]:
    active = rank_active_symbols(load_active_symbols())
    discovery = get_discovery_symbols()

    universe = list(dict.fromkeys(MAG7 + active + discovery))
    return universe[:TOTAL_SCAN_LIMIT]


# =========================================================
# PER-SYMBOL COMPUTE
# =========================================================

def compute_symbol(symbol: str) -> Dict[str, Any] | None:
    candles = get_candles(symbol, min_points=120)
    if not candles:
        return None

    feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles)
    if feats_vec is None:
        return None

    infer = backend.bullbrain_infer(feats_vec)
    if infer is None:
        return None

    prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
    prob_down = float(infer.get("probability_down") or (1.0 - prob_up))

    if prob_up >= 0.58:
        signal = "BUY"
    elif prob_down >= 0.58:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(max(prob_up, prob_down) * 100.0, 2)

    insights = generate_bull_insights(
        symbol=symbol,
        features=feat_dict,
        bullbrain={
            "signal": signal,
            "confidence": confidence,
            "prob_up": prob_up,
            "prob_down": prob_down,
        },
        technical=None,
        seed_key=f"{symbol}:{utc_now_iso()}",
    )

    doc = {
        "symbol": symbol,
        "company_name": COMPANY_NAMES.get(symbol, symbol),
        "bullbrain": {
            "signal": signal,
            "confidence": confidence,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
        },
        "features_meta": feat_dict,
        "insights": insights,
        "computed_at": utc_now_iso(),
        "schema_version": "v1",
    }

    db = get_db()
    db.collection(COL_ROOT).document(COL_STOCKS) \
        .collection("symbols").document(symbol) \
        .set(doc, merge=True)

    return doc


# =========================================================
# PHASE 2 — HOTLIST + BEARWATCH
# =========================================================

def build_hotlist_bearwatch(results: List[Dict[str, Any]]):
    buys = []
    sells = []

    for r in results:
        bb = r.get("bullbrain", {})
        if bb.get("signal") == "BUY":
            buys.append(r)
        elif bb.get("signal") == "SELL":
            sells.append(r)

    buys.sort(key=lambda x: x["bullbrain"]["confidence"], reverse=True)
    sells.sort(key=lambda x: x["bullbrain"]["confidence"], reverse=True)

    return buys[:5], sells[:5]


def save_market_lists(hotlist, bearwatch):
    db = get_db()

    db.collection(COL_ROOT).document(DOC_HOTLIST).set(
        {
            "count": len(hotlist),
            "hotlist": hotlist,
            "updated_at": utc_now_iso(),
        },
        merge=True,
    )

    db.collection(COL_ROOT).document(DOC_BEARWATCH).set(
        {
            "count": len(bearwatch),
            "bearwatch": bearwatch,
            "updated_at": utc_now_iso(),
        },
        merge=True,
    )


# =========================================================
# MAG-7 HOMESCREEN SNAPSHOT (UNCHANGED CONTRACT)
# =========================================================

def build_mag7_snapshot():
    db = get_db()
    items = []

    for sym in MAG7:
        snap = (
            db.collection(COL_ROOT)
              .document(COL_STOCKS)
              .collection("symbols")
              .document(sym)
              .get()
        )
        if snap.exists:
            items.append(snap.to_dict())

    return items


def save_homescreen_snapshot():
    db = get_db()
    db.collection(COL_ROOT).document(DOC_HOMESCREEN).set(
        {
            "mag7": build_mag7_snapshot(),
            "updated_at": utc_now_iso(),
            "version": "v1",
        },
        merge=True,
    )


# =========================================================
# ENTRYPOINT
# =========================================================

def main():
    ensure_bullbrain_loaded()

    scan_symbols = build_scan_universe()
    results = []

    for sym in scan_symbols:
        try:
            r = compute_symbol(sym)
            if r:
                results.append(r)
        except Exception:
            pass

        time.sleep(random.uniform(0.15, 0.25))

    hotlist, bearwatch = build_hotlist_bearwatch(results)
    save_market_lists(hotlist, bearwatch)

    save_homescreen_snapshot()


if __name__ == "__main__":
    main()
