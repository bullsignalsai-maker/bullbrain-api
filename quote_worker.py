# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher (30s loop)
#
# Runs as a LONG-RUNNING background worker (NOT cron)
# Refreshes quotes for:
#   - Home screen carousel (SPY/QQQ/GLD/SLV/USO etc. parsed from labels)
#   - MAG7
#   - Hotlist
#   - Bearwatch
#
# Frontend reads Firestore only.
# ---------------------------------------------------------

import os
import json
import time
import datetime
import requests
import random
from typing import Dict, Any, Set, Optional, List

import firebase_admin
from firebase_admin import credentials, firestore


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
def log(msg: str) -> None:
    print(f"[quote-worker] {msg}", flush=True)


# ---------------------------------------------------------
# Time helper
# ---------------------------------------------------------
def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------
# Firebase Admin init (worker-safe)
# ---------------------------------------------------------
def init_firebase_admin():
    """
    Initialize Firebase Admin exactly once using FIREBASE_ADMIN_JSON.
    Expected: FIREBASE_ADMIN_JSON contains a valid JSON service account.
    """
    if firebase_admin._apps:
        return

    firebase_json = os.getenv("FIREBASE_ADMIN_JSON")
    if not firebase_json:
        raise RuntimeError("FIREBASE_ADMIN_JSON missing")

    try:
        cred_dict = json.loads(firebase_json)  # must be strict JSON (double quotes)
    except Exception as e:
        raise RuntimeError(
            f"FIREBASE_ADMIN_JSON is not valid JSON. "
            f"Tip: wrap the entire JSON in one line in Render env. Error: {e}"
        )

    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    log("🔥 Firebase initialized in quote_worker")


init_firebase_admin()
db = firestore.client()

log("✅ quote_worker Firestore client ready")


# ---------------------------------------------------------
# Safe quote fetch (lazy import, NO THROW)
# ---------------------------------------------------------
def fetch_quote_safe(symbol: str) -> Dict[str, Any]:
    """
    Imports main lazily so we don't crash at import-time if main.py does
    heavy initialization. Returns {} on failure.
    """
    try:
        import main as backend  # lazy import (inside function)

        q = backend.backend_fetch_quote(symbol)
        if isinstance(q, dict):
            return q
    except Exception as e:
        log(f"Quote fetch failed for {symbol}: {e}")

    return {}


def normalize_change_pct(v: Any):
    """
    Normalize changePct to percentage units.
    Some quote providers return decimals (0.0086 => 0.86%).
    """
    try:
        v = float(v)
        if abs(v) <= 1.5:  # treat as decimal ratio
            return v * 100.0
        return v
    except Exception:
        return None


def parse_symbol_from_label(label: str) -> str:
    """
    Extract symbol inside parentheses: "S&P 500 (SPY)" -> "SPY"
    Returns "" if not parseable.
    """
    try:
        if not label:
            return ""
        if "(" in label and ")" in label:
            sym = label.split("(")[-1].split(")")[0].strip()
            # Keep it conservative. Tickers like BRK.B won't pass isalpha(),
            # so use a safer filter.
            if 1 <= len(sym) <= 10:
                return sym.upper()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------
# Collect ALL tickers that need quotes
# ---------------------------------------------------------
def collect_quote_tickers(db_client) -> Set[str]:
    tickers: Set[str] = set()

    # ---------------------------
    # 1) Home screen snapshot
    # ---------------------------
    snap = (
        db_client.collection("bullsignals_ai")
        .document("homescreen_snapshot")
        .get()
    )

    if snap.exists:
        data = snap.to_dict() or {}

        # MAG7 (stored as list of maps)
        for item in data.get("mag7", []) or []:
            if isinstance(item, dict):
                sym = (item.get("symbol") or "").strip().upper()
                if sym:
                    tickers.add(sym)

        # Carousel proxies (symbols parsed from label "Gold (GLD)", etc.)
        for card in data.get("carousel", []) or []:
            if not isinstance(card, dict):
                continue
            for it in card.get("items", []) or []:
                if not isinstance(it, dict):
                    continue
                label = it.get("label", "") or ""
                sym = parse_symbol_from_label(label)
                if sym:
                    tickers.add(sym)

    # ---------------------------
    # 2) Hotlist
    # ---------------------------
    hot = (
        db_client.collection("bullsignals_ai")
        .document("market_hotlist")
        .get()
    )

    if hot.exists:
        hot_data = hot.to_dict() or {}
        for h in hot_data.get("hotlist", []) or []:
            if isinstance(h, dict):
                sym = (h.get("symbol") or "").strip().upper()
                if sym:
                    tickers.add(sym)

    # ---------------------------
    # 3) Bearwatch
    # ---------------------------
    bear = (
        db_client.collection("bullsignals_ai")
        .document("market_bearwatch")
        .get()
    )

    if bear.exists:
        bear_data = bear.to_dict() or {}
        for b in bear_data.get("bearwatch", []) or []:
            if isinstance(b, dict):
                sym = (b.get("symbol") or "").strip().upper()
                if sym:
                    tickers.add(sym)

    return tickers


# ---------------------------------------------------------
# Update Firestore docs with quotes (MERGE SAFE)
# ---------------------------------------------------------
def update_quotes(db_client, quotes: Dict[str, Dict[str, Any]]) -> None:
    now = utc_now_iso()

    # ---------------------------
    # Home screen snapshot
    # ---------------------------
    hs_ref = db_client.collection("bullsignals_ai").document("homescreen_snapshot")
    hs_doc = hs_ref.get()

    if hs_doc.exists:
        data = hs_doc.to_dict() or {}

        # Update MAG7 list items
        mag7_list = data.get("mag7", []) or []
        for item in mag7_list:
            if not isinstance(item, dict):
                continue
            sym = (item.get("symbol") or "").strip().upper()
            if sym and sym in quotes:
                item["price"] = quotes[sym].get("price")
                item["changePct"] = quotes[sym].get("changePct")
                item["quote_updated_at"] = now

        # Update carousel item values based on parsed symbol
        carousel = data.get("carousel", []) or []
        for card in carousel:
            if not isinstance(card, dict):
                continue
            items = card.get("items", []) or []
            for it in items:
                if not isinstance(it, dict):
                    continue
                sym = parse_symbol_from_label(it.get("label", "") or "")
                if sym and sym in quotes:
                    chg = quotes[sym].get("changePct")
                    it["value"] = f"{chg:+.2f}%" if chg is not None else "--"

        # Persist updates
        hs_ref.set(
            {
                "mag7": mag7_list,
                "carousel": carousel,
                "quote_updated_at": now,
            },
            merge=True,
        )

    # ---------------------------
    # Hotlist
    # ---------------------------
    hot_ref = db_client.collection("bullsignals_ai").document("market_hotlist")
    hot_doc = hot_ref.get()

    if hot_doc.exists:
        hot_data = hot_doc.to_dict() or {}
        hotlist = hot_data.get("hotlist", []) or []

        for h in hotlist:
            if not isinstance(h, dict):
                continue
            sym = (h.get("symbol") or "").strip().upper()
            if sym and sym in quotes:
                h["price"] = quotes[sym].get("price")
                h["changePct"] = quotes[sym].get("changePct")
                h["quote_updated_at"] = now

        hot_ref.set({"hotlist": hotlist, "quote_updated_at": now}, merge=True)

    # ---------------------------
    # Bearwatch
    # ---------------------------
    bear_ref = db_client.collection("bullsignals_ai").document("market_bearwatch")
    bear_doc = bear_ref.get()

    if bear_doc.exists:
        bear_data = bear_doc.to_dict() or {}
        bearwatch = bear_data.get("bearwatch", []) or []

        for b in bearwatch:
            if not isinstance(b, dict):
                continue
            sym = (b.get("symbol") or "").strip().upper()
            if sym and sym in quotes:
                b["price"] = quotes[sym].get("price")
                b["changePct"] = quotes[sym].get("changePct")
                b["quote_updated_at"] = now

        bear_ref.set({"bearwatch": bearwatch, "quote_updated_at": now}, merge=True)


# ---------------------------------------------------------
# MAIN LOOP — every 30 seconds
# ---------------------------------------------------------
def main():
    log("🚀 Quote worker started (30s loop)")

    while True:
        cycle_started = utc_now_iso()
        try:
            tickers = collect_quote_tickers(db)
            log(f"Refreshing quotes for {len(tickers)} tickers | cycle={cycle_started}")

            quotes: Dict[str, Dict[str, Any]] = {}

            for sym in sorted(tickers):
                q = fetch_quote_safe(sym)
                price = q.get("price") or q.get("close")
                chg = normalize_change_pct(q.get("changePct"))

                quotes[sym] = {
                    "price": price,
                    "changePct": chg,
                }

                time.sleep(0.10)  # gentle throttling per symbol

            update_quotes(db, quotes)
            log(f"✅ Quote refresh cycle completed | tickers={len(tickers)}")

        except Exception as e:
            log(f"❌ Quote worker cycle error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
