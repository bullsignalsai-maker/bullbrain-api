# scripts/backfill_market_regime_history.py
# =========================================================
# One-off backfill: populates bullsignals_ai/market_regime_history/days/{date}
# for the gap before backend/market_regime_history.py went live
# (2026-09-22 session) -- from pick_tracking's real start date (2026-07-17)
# through the day before the first live capture.
#
# NOT part of the regular cron. Run manually, once. Uses the same
# serviceAccountKey.json credential pattern as the other scripts/*.py
# files (see scripts/pick_deep_dive.py), not main.py's FIREBASE_ADMIN_JSON
# env var, so it runs locally without the full app's env config.
#
# Source: the same unauthenticated Yahoo Finance chart endpoint already
# used LIVE in this exact codebase for VIX (main.py's /stats/live,
# /market-mood) -- not candle_store/Polygon, since SPY's candle cache
# hasn't been seeded yet and this needs a real historical range, not a
# point-in-time quote. Every backfilled row is tagged backfilled=True so
# it's always distinguishable from a real-time capture.
#
# Defaults to DRY RUN (prints the full preview table, writes nothing).
# Pass --confirm to actually write to Firestore.
# =========================================================

import sys
import datetime
import statistics
import argparse

import requests
import firebase_admin
from firebase_admin import credentials, firestore

sys.path.insert(0, ".")

BACKFILL_START = "2026-07-17"   # pick_tracking's real first pick_date
BACKFILL_END = "2026-09-22"     # last date before the first live capture

# Extra calendar-day padding before BACKFILL_START so the very first
# backfilled date still has a real previous-close for change_pct, and
# enough trailing closes for a real 20-trading-day realized-vol window
# (~28 calendar days needed; padded further for weekends/holidays).
_LOOKBACK_PADDING_DAYS = 45
_TRAILING_PADDING_DAYS = 5

_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_FETCH_TIMEOUT_SECONDS = 15

COL_ROOT = "bullsignals_ai"
REGIME_HISTORY_COLLECTION = "market_regime_history"

# The mislabeled doc from the 2026-09-22 session's ad-hoc verification
# call -- see this script's accompanying investigation. Its data (SPY
# 773.38, VIX 14.21) is really 2026-09-22's real close, written under the
# wrong date key because the manual call ran after midnight UTC while the
# quote cache still held the prior trading day's values. Deleted (not
# overwritten) before the backfill runs, since the backfill will write
# the correct data under the correct key (2026-09-22).
STALE_MISLABELED_DOC_DATE = "2026-09-23"


def _fetch_yahoo_series(ticker: str, period1: int, period2: int):
    """
    Returns {date_str: close} for one ticker over [period1, period2).
    Falls back to meta.regularMarketPrice for the most recent bar when
    Yahoo's own indicators.quote[].close array has a null there (a real,
    observed quirk on the last/most-recent bar -- confirmed 2026-09-22:
    the array had None for that date while meta carried the correct
    773.38, matching what the live path independently captured the same
    day).
    """
    url = _YAHOO_CHART_URL.format(ticker=ticker)
    resp = requests.get(
        url,
        params={"period1": period1, "period2": period2, "interval": "1d"},
        timeout=_FETCH_TIMEOUT_SECONDS,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    data = resp.json()
    result = (data.get("chart") or {}).get("result") or [{}]
    result = result[0]

    timestamps = result.get("timestamp") or []
    closes = (result.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
    meta = result.get("meta") or {}
    meta_price = meta.get("regularMarketPrice")
    meta_time = meta.get("regularMarketTime")
    meta_date = (
        datetime.datetime.utcfromtimestamp(meta_time).date().isoformat()
        if isinstance(meta_time, (int, float)) else None
    )

    series = {}
    for ts, close in zip(timestamps, closes):
        date_str = datetime.datetime.utcfromtimestamp(ts).date().isoformat()
        if close is None and date_str == meta_date and isinstance(meta_price, (int, float)):
            close = meta_price
        if close is not None:
            series[date_str] = float(close)

    return series


def _realized_vol_20d(spy_series_sorted_dates, spy_by_date, as_of_date):
    """Same daily_ret.rolling(20).std()*100 convention as the live
    backend/market_regime_history.py._spy_realized_vol_20d_pct() and
    features_meta.volatility_20d elsewhere in this app."""
    eligible = [d for d in spy_series_sorted_dates if d <= as_of_date]
    if len(eligible) < 21:
        return None
    window = eligible[-21:]
    closes = [spy_by_date[d] for d in window]
    daily_returns = [
        (closes[i] / closes[i - 1] - 1.0) * 100.0
        for i in range(1, len(closes))
        if closes[i - 1]
    ]
    if len(daily_returns) < 2:
        return None
    return round(statistics.stdev(daily_returns), 3)


def build_backfill_rows():
    from backend.market_calendar import load_recent_trading_days

    start = datetime.date.fromisoformat(BACKFILL_START)
    end = datetime.date.fromisoformat(BACKFILL_END)
    fetch_start = start - datetime.timedelta(days=_LOOKBACK_PADDING_DAYS)
    fetch_end = end + datetime.timedelta(days=_TRAILING_PADDING_DAYS)

    period1 = int(datetime.datetime.combine(fetch_start, datetime.time.min, tzinfo=datetime.timezone.utc).timestamp())
    period2 = int(datetime.datetime.combine(fetch_end, datetime.time.min, tzinfo=datetime.timezone.utc).timestamp())

    print(f"Fetching VIX history {fetch_start} -> {fetch_end} ...")
    vix_by_date = _fetch_yahoo_series("%5EVIX", period1, period2)
    print(f"  {len(vix_by_date)} VIX daily closes fetched")

    print(f"Fetching SPY history {fetch_start} -> {fetch_end} ...")
    spy_by_date = _fetch_yahoo_series("SPY", period1, period2)
    print(f"  {len(spy_by_date)} SPY daily closes fetched")

    spy_dates_sorted = sorted(spy_by_date.keys())

    # Cross-check only -- NOT the date source. backend.market_calendar
    # derives trading days from AAPL/MSFT's own cached candle history,
    # which can lag up to a day behind the real market close (their cache
    # hadn't picked up 2026-09-22 yet at the time this was run, even
    # though Yahoo's meta.regularMarketTime confirms a real 20:00 UTC
    # close that date, matching the same value the live path
    # independently captured). Yahoo's own returned dates are what
    # actually drive which rows get built, below -- this block only
    # flags disagreements for a human to look at.
    needed_lookback = (datetime.date.today() - start).days + 10
    trading_days = load_recent_trading_days(lookback_days=needed_lookback)
    target_trading_days = sorted(
        d for d in trading_days if BACKFILL_START <= d <= BACKFILL_END
    )

    yahoo_dates_in_range = sorted(
        d for d in spy_by_date if BACKFILL_START <= d <= BACKFILL_END
    )

    print(f"\nbackend.market_calendar trading days in range: {len(target_trading_days)}")
    print(f"Yahoo SPY close dates in range: {len(yahoo_dates_in_range)}  <- driving the backfill rows below")

    missing_from_yahoo = sorted(set(target_trading_days) - set(yahoo_dates_in_range))
    extra_in_yahoo = sorted(set(yahoo_dates_in_range) - set(target_trading_days))
    if missing_from_yahoo:
        print(f"  WARNING: market_calendar trading days with NO Yahoo SPY close -- these rows will be skipped: {missing_from_yahoo}")
    if extra_in_yahoo:
        print(f"  NOTE: Yahoo has SPY closes on dates market_calendar doesn't (yet) count as trading days -- included anyway, real market_calendar candle-cache lag, not a Yahoo data problem: {extra_in_yahoo}")
    if not missing_from_yahoo and not extra_in_yahoo:
        print("  MATCH: Yahoo's trading dates exactly match this app's own trading-day calendar for this range.")

    rows = []
    for date_str in yahoo_dates_in_range:
        spy_close = spy_by_date.get(date_str)
        vix_close = vix_by_date.get(date_str)

        prior_dates = [d for d in spy_dates_sorted if d < date_str]
        spy_change_pct = None
        if spy_close is not None and prior_dates:
            prev_close = spy_by_date[prior_dates[-1]]
            if prev_close:
                spy_change_pct = round((spy_close / prev_close - 1.0) * 100.0, 4)

        rows.append({
            "date": date_str,
            "recorded_at": None,  # filled in at write time
            "spy_close": spy_close,
            "spy_change_pct": spy_change_pct,
            "vix_close": vix_close,
            "vix_source": "yahoo_chart_api" if vix_close is not None else None,
            "spy_realized_vol_20d_pct": _realized_vol_20d(spy_dates_sorted, spy_by_date, date_str),
            "schema_version": "market_regime_history_v1",
            "backfilled": True,
            "backfill_source": "yahoo_chart_api_historical",
        })

    return rows


def print_preview(rows):
    print(f"\n{'date':<12} {'spy_close':>10} {'spy_chg%':>9} {'vix':>7} {'vol20d%':>8}")
    for r in rows:
        print(
            f"{r['date']:<12} "
            f"{r['spy_close'] if r['spy_close'] is not None else '—':>10} "
            f"{r['spy_change_pct'] if r['spy_change_pct'] is not None else '—':>9} "
            f"{r['vix_close'] if r['vix_close'] is not None else '—':>7} "
            f"{r['spy_realized_vol_20d_pct'] if r['spy_realized_vol_20d_pct'] is not None else '—':>8}"
        )

    incomplete = [r for r in rows if r["spy_close"] is None or r["vix_close"] is None]
    if incomplete:
        print(f"\nWARNING: {len(incomplete)} row(s) missing spy_close or vix_close: {[r['date'] for r in incomplete]}")
    no_vol = [r for r in rows if r["spy_realized_vol_20d_pct"] is None]
    if no_vol:
        print(f"NOTE: {len(no_vol)} row(s) with no spy_realized_vol_20d_pct (insufficient trailing history): {[r['date'] for r in no_vol]}")


def write_rows(db, rows):
    picks_written = 0
    skipped_existing_live = 0

    # One-time correction: delete the mislabeled doc from the 2026-09-22
    # session's ad-hoc verification call. Its data is a duplicate of what
    # this backfill writes under the correct key below.
    stale_ref = (
        db.collection(COL_ROOT).document(REGIME_HISTORY_COLLECTION)
          .collection("days").document(STALE_MISLABELED_DOC_DATE)
    )
    stale_doc = stale_ref.get()
    if stale_doc.exists and not (stale_doc.to_dict() or {}).get("backfilled"):
        print(f"Deleting mislabeled live doc at days/{STALE_MISLABELED_DOC_DATE} (see investigation notes)...")
        stale_ref.delete()

    for row in rows:
        ref = (
            db.collection(COL_ROOT).document(REGIME_HISTORY_COLLECTION)
              .collection("days").document(row["date"])
        )
        existing = ref.get()
        if existing.exists and not (existing.to_dict() or {}).get("backfilled"):
            # Never overwrite a genuine real-time capture with backfilled data.
            print(f"  SKIP {row['date']}: real live capture already exists, not overwriting")
            skipped_existing_live += 1
            continue

        row = dict(row)
        row["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        ref.set(row, merge=True)
        picks_written += 1

    return {"written": picks_written, "skipped_existing_live": skipped_existing_live}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="Actually write to Firestore. Omit for a dry run (preview only, no writes).")
    args = parser.parse_args()

    # Needed even in dry-run mode: load_recent_trading_days() does a
    # real (read-only) Firestore lookup of AAPL/MSFT's cached candles to
    # derive the authoritative trading-day calendar for the preview. No
    # writes happen unless --confirm is passed.
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    rows = build_backfill_rows()
    print_preview(rows)

    if not args.confirm:
        print(f"\nDRY RUN -- {len(rows)} rows computed above, nothing written. Re-run with --confirm to write.")
    else:
        stats = write_rows(db, rows)
        print(f"\nWROTE: {stats}")
