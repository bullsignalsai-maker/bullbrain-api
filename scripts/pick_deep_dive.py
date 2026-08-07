# scripts/pick_deep_dive.py
# One-off analysis script (not a cron/route) for the 2026-08-06 winners-vs-
# losers deep dive. Read-only: fetches pick_tracking data and prints a
# report. Uses the same serviceAccountKey.json credential pattern as the
# other scripts/*.py files, not main.py's FIREBASE_ADMIN_JSON env var, so it
# can run locally without the full app's env config.

import datetime
import json
import sys

import firebase_admin
from firebase_admin import credentials, firestore

sys.path.insert(0, ".")

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

from backend.pick_tracking import get_checked_picks_for_report
from backend.pick_accuracy_report import (
    dedupe_checked_picks,
    build_accuracy_report,
    render_markdown_report,
)
from backend.market_calendar import load_recent_trading_days


def part_1_and_2_and_3():
    raw_docs = get_checked_picks_for_report(db, since=None)
    deduped = dedupe_checked_picks(raw_docs)
    report = build_accuracy_report(deduped)
    print("=" * 70)
    print("PART 1-3: Fresh accuracy report (factor scores, pick_source,")
    print("setup_label, market_regime, model_view.bias)")
    print("=" * 70)
    print(render_markdown_report(report))
    print()
    print(f"[raw doc count: {len(raw_docs)}, deduped distinct picks: {len(deduped)}]")
    return report


def part_4_winners_losers():
    print("=" * 70)
    print("PART 4: Big winners vs big losers (currently tracking, by")
    print("livePctSinceFirstPick)")
    print("=" * 70)

    WINDOW_DAYS = 30
    trading_days_sorted = sorted(load_recent_trading_days(lookback_days=60), reverse=True)
    recent_trading_days = trading_days_sorted[:WINDOW_DAYS]
    window_start_date = recent_trading_days[-1] if recent_trading_days else (
        datetime.date.today() - datetime.timedelta(days=WINDOW_DAYS)
    ).isoformat()

    picks_col = db.collection("bullsignals_ai").document("pick_tracking").collection("picks")
    raw_items = [d.to_dict() or {} for d in picks_col.where("pick_date", ">=", window_start_date).stream()]

    earliest_by_symbol = {}
    latest_by_symbol = {}
    for it in raw_items:
        symbol = str(it.get("symbol") or "").upper()
        if not symbol:
            continue
        if symbol not in earliest_by_symbol or (it.get("recorded_at") or "") < (earliest_by_symbol[symbol].get("recorded_at") or ""):
            earliest_by_symbol[symbol] = it
        if symbol not in latest_by_symbol or (it.get("recorded_at") or "") > (latest_by_symbol[symbol].get("recorded_at") or ""):
            latest_by_symbol[symbol] = it

    symbols_needed = list(latest_by_symbol.keys())
    stock_refs = [
        db.collection("bullsignals_ai").document("stocks").collection("symbols").document(sym)
        for sym in symbols_needed
    ]
    stock_by_symbol = {}
    if stock_refs:
        for snap in db.get_all(stock_refs):
            stock_by_symbol[snap.id] = snap.to_dict() if snap.exists else {}

    rows = []
    for symbol, latest in latest_by_symbol.items():
        horizons = latest.get("horizons") or {}
        h5 = horizons.get("5d") or {}
        h20 = horizons.get("20d") or {}
        # Only "still tracking" (no resolved horizon yet) for this part --
        # the checked-picks breakdown is already covered in part 1-3.
        if h5.get("status") in ("checked", "unavailable") or h20.get("status") in ("checked", "unavailable"):
            continue

        first = earliest_by_symbol.get(symbol) or {}
        first_price = first.get("pick_price")
        quote = (stock_by_symbol.get(symbol) or {}).get("quote") or {}
        current_price = quote.get("price")

        if not (isinstance(current_price, (int, float)) and isinstance(first_price, (int, float)) and first_price):
            continue

        live_pct = round((current_price / first_price - 1) * 100, 2)
        rows.append({
            "symbol": symbol,
            "livePctSinceFirstPick": live_pct,
            "first_picked_date": first.get("pick_date"),
            "first_picked_price": first_price,
            "current_price": current_price,
            "pick_source": latest.get("pick_source"),
            "pick_setup_label": latest.get("pick_setup_label"),
            "pick_market_regime": latest.get("pick_market_regime"),
            "pick_model_view": latest.get("pick_model_view"),
            "pick_factor_scores": latest.get("pick_factor_scores"),
            "pick_pattern_stats": latest.get("pick_pattern_stats"),
            "pick_score": latest.get("pick_score"),
            "pick_decision_reasons": latest.get("pick_decision_reasons"),
        })

    rows.sort(key=lambda r: r["livePctSinceFirstPick"], reverse=True)

    print(f"[{len(rows)} distinct symbols currently tracking with valid live pct]\n")

    print("--- TOP 5 WINNERS ---")
    for r in rows[:5]:
        print(json.dumps(r, indent=2, default=str))
    print()
    print("--- BOTTOM 5 LOSERS ---")
    for r in rows[-5:][::-1]:
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    part_1_and_2_and_3()
    print()
    part_4_winners_losers()
