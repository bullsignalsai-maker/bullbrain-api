# backend/candle_refresh_grouped.py
# =========================================================
# Refresh candle docs from Polygon's grouped-daily endpoint (one call
# returns every US ticker's bar for a date) instead of one aggs call per
# symbol. Needed because the plan caps Polygon at 5 calls/min and the
# per-symbol delta path in candle_store.get_candles() burns that budget in
# the first seconds of each cron run, leaving most of the universe stale.
#
# Verified 2026-10-05 against the live plan: grouped-daily serves history
# back to ~2 years (2024-10-07 OK, 2024-10-04 403) but refuses the current
# day until after the close ("today's data before end of day").
#
# Strategy per symbol, from its last stored bar date L to `through`:
#   (unverified overlap -> splits-endpoint check -> append, or full refetch)
#   current      L >= through                       no call
#   grouped      gap <= max_grouped_days weekdays   shares one grouped call
#                                                   per date across symbols
#   per_symbol   deeper gap                         1 aggs call from L
#   full         no candle doc yet                  1 full-history call
#   not_listed   absent from grouped(through)       skipped (delisted/merged)
#
# Appending adjusted bars onto older stored bars is only safe if no split
# happened in between, so every symbol is overlap-checked: the bar at its
# own last stored date L (from grouped(L), or the first per-symbol bar) is
# compared with the stored close. A difference > 0.1% flags the symbol for
# a full refetch instead of an append.
#
# Matching is by DATE, never timestamp: grouped bars carry 20:00 UTC while
# stored bars use midnight ET (04:00/05:00 UTC). Appended bars are given
# the stored midnight-ET convention.
#
# Dry-run by default: makes NO Polygon calls and NO writes (Firestore
# reads only). Pacing stays <= 4.8 calls/min. Resumable: grouped responses
# are cached on disk, and appends skip dates a doc already has, so a
# re-run after an interruption redoes nothing.
#
# Usage:
#   python -m backend.candle_refresh_grouped --through 2026-10-02            # dry run
#   python -m backend.candle_refresh_grouped --through 2026-10-02 --apply
# =========================================================

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pytz
import requests

from backend import candle_store
from backend.candle_store import (
    _normalize_polygon_results,
    _polygon_fetch,
    _read_firestore_candles,
    _save_firestore_candles,
    fetch_full_history,
    normalize_polygon_symbol,
)
from backend.firestore_utils import get_db
from symbols_clean import REAL_TICKERS

_ET = pytz.timezone("America/New_York")

GROUPED_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
SPLITS_URL = "https://api.polygon.io/v3/reference/splits"
PACE_SECONDS = 12.5            # 4.8 calls/min, under the 5/min cap
RATE_LIMIT_BACKOFF_SECONDS = 65
OVERLAP_TOLERANCE = 0.001      # 0.1%
DEFAULT_MAX_GROUPED_DAYS = 20
DEFAULT_CACHE_DIR = ".candle_refresh_cache"

CANDLE_KEYS = ("open", "high", "low", "close", "volume", "ts")


class PolygonStop(Exception):
    """Polygon refused in a way retrying won't fix (entitlement, repeated 429)."""


def log(msg: str) -> None:
    print(f"[candle-refresh] {msg}", flush=True)


# ---------------------------------------------------------
# DATE HELPERS
# ---------------------------------------------------------
def _ms_to_date(ms: float) -> datetime.date:
    # Stored bars are midnight ET == 04:00/05:00 UTC, same UTC calendar date.
    return datetime.datetime.utcfromtimestamp(ms / 1000).date()


def _midnight_et_ms(d: datetime.date) -> int:
    local = _ET.localize(datetime.datetime(d.year, d.month, d.day))
    return int(local.timestamp() * 1000)


def _weekdays_after(start: datetime.date, end: datetime.date) -> List[datetime.date]:
    """Weekdays strictly after `start`, up to and including `end`."""
    out, d = [], start + datetime.timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


# ---------------------------------------------------------
# GROUPED-DAILY FETCH (paced, cached)
# ---------------------------------------------------------
class GroupedClient:
    def __init__(self, cache_dir: str, wanted: Set[str], max_calls: Optional[int]):
        self.cache_dir = cache_dir
        self.wanted = wanted
        self.max_calls = max_calls
        self.calls = 0
        self._last_call = 0.0
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, d: datetime.date) -> str:
        return os.path.join(self.cache_dir, f"grouped_{d.isoformat()}.json")

    def cached(self, d: datetime.date) -> Optional[Dict[str, Any]]:
        try:
            with open(self._path(d)) as f:
                blob = json.load(f)
        except Exception:
            return None
        # Reusable only if it was filtered to a superset of today's symbols.
        if not self.wanted.issubset(set(blob.get("covers", []))):
            return None
        return blob

    def pace(self) -> None:
        wait = PACE_SECONDS - (time.time() - self._last_call)
        if self._last_call and wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def budget_left(self) -> bool:
        return self.max_calls is None or self.calls < self.max_calls

    def get(self, d: datetime.date) -> Dict[str, Any]:
        """{"count": N, "bars": {ticker: [o,h,l,c,v]}} for universe tickers."""
        blob = self.cached(d)
        if blob is not None:
            return blob
        if not self.budget_left():
            raise PolygonStop("max_calls reached")
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            raise PolygonStop("POLYGON_API_KEY not set")

        for attempt in (1, 2):
            self.pace()
            self.calls += 1
            resp = requests.get(
                GROUPED_URL.format(date=d.isoformat()),
                params={"adjusted": "true", "apiKey": key},
                timeout=60,
            )
            if resp.status_code == 429 and attempt == 1:
                log(f"grouped {d} | HTTP 429 | backing off {RATE_LIMIT_BACKOFF_SECONDS}s")
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            break

        if resp.status_code != 200:
            try:
                reason = str(resp.json().get("message") or resp.json().get("error"))[:160]
            except Exception:
                reason = (resp.text or "")[:160]
            kind = "RATE-LIMITED" if resp.status_code == 429 else f"HTTP-{resp.status_code}"
            raise PolygonStop(f"grouped {d} | {kind} | {reason.replace(key, '<redacted>')}")

        results = resp.json().get("results") or []
        blob = {
            "date": d.isoformat(),
            "count": len(results),
            "covers": sorted(self.wanted),
            "bars": {
                r["T"]: [r.get("o"), r.get("h"), r.get("l"), r.get("c"), r.get("v")]
                for r in results
                if r.get("T") in self.wanted
            },
        }
        with open(self._path(d), "w") as f:
            json.dump(blob, f)
        log(f"grouped {d} | HTTP 200 | tickers={blob['count']} universe_bars={len(blob['bars'])}")
        return blob


# ---------------------------------------------------------
# PURE LOGIC (unit-testable)
# ---------------------------------------------------------
def classify(
    last_date: Optional[datetime.date],
    through: datetime.date,
    listed: Optional[bool],
    max_grouped_days: int,
) -> str:
    if last_date is None:
        return "not_listed" if listed is False else "full"
    if listed is False:
        return "not_listed"
    if last_date >= through:
        return "current"
    gap = len(_weekdays_after(last_date, through))
    return "grouped" if gap <= max_grouped_days else "per_symbol"


def overlap_ok(stored_close: Optional[float], fresh_close: Optional[float]) -> Optional[bool]:
    """True/False, or None when it can't be checked."""
    if not stored_close or not fresh_close:
        return None
    return abs(fresh_close - stored_close) / abs(stored_close) <= OVERLAP_TOLERANCE


def append_bars(
    candles: Dict[str, list],
    bars_by_date: Dict[datetime.date, List[float]],
) -> int:
    """
    Appends [o,h,l,c,v] bars in date order, skipping dates the doc already
    has (match on date, not ts). Returns bars added.
    """
    have = {_ms_to_date(t) for t in candles.get("ts", [])}
    added = 0
    for d in sorted(bars_by_date):
        if d in have:
            continue
        o, h, l, c, v = bars_by_date[d]
        candles["open"].append(o)
        candles["high"].append(h)
        candles["low"].append(l)
        candles["close"].append(c)
        candles["volume"].append(v)
        candles["ts"].append(_midnight_et_ms(d))
        added += 1
    return added


def _close_on(candles: Dict[str, list], d: datetime.date) -> Optional[float]:
    for t, c in zip(reversed(candles.get("ts", [])), reversed(candles.get("close", []))):
        if _ms_to_date(t) == d:
            return c
        if _ms_to_date(t) < d:
            break
    return None


def _finish_meta(meta: Dict[str, Any], candles: Dict[str, list], through: datetime.date) -> Dict[str, Any]:
    meta = dict(meta or {})
    meta["last_ts"] = candles["ts"][-1]
    meta["count"] = len(candles["close"])
    meta["first_ts"] = candles["ts"][0]
    meta["last_fetch"] = candle_store.utc_now_iso()
    meta["last_grouped_refresh_through"] = through.isoformat()
    meta.setdefault("source", "polygon")
    return meta


# ---------------------------------------------------------
# UNIVERSE + PLAN
# ---------------------------------------------------------
def load_symbols(explicit: Optional[List[str]], universe_only: bool = False) -> List[str]:
    if explicit:
        return sorted({s.upper() for s in explicit})
    if universe_only:
        return sorted(set(REAL_TICKERS))
    db = get_db()
    docs = {
        r.id
        for r in db.collection("bullsignals_ai").document("candles").collection("symbols").list_documents()
    }
    return sorted(set(REAL_TICKERS) | docs)


def load_last_dates(symbols: List[str]) -> Dict[str, Optional[datetime.date]]:
    db = get_db()
    col = db.collection("bullsignals_ai").document("candles").collection("symbols")
    out: Dict[str, Optional[datetime.date]] = {s: None for s in symbols}
    for i in range(0, len(symbols), 100):
        refs = [col.document(s) for s in symbols[i:i + 100]]
        for d in db.get_all(refs, field_paths=["meta"]):
            if d.exists:
                ts = ((d.to_dict() or {}).get("meta") or {}).get("last_ts")
                if isinstance(ts, (int, float)):
                    out[d.id] = _ms_to_date(ts)
    return out


def build_plan(
    symbols: List[str],
    last_dates: Dict[str, Optional[datetime.date]],
    through: datetime.date,
    listed: Optional[Set[str]],
    closed_days: Set[datetime.date],
    max_grouped_days: int,
) -> Dict[str, Any]:
    kinds: Dict[str, str] = {}
    bars_to_add: Dict[str, int] = {}
    grouped_dates: Set[datetime.date] = set()
    for s in symbols:
        last = last_dates.get(s)
        poly = normalize_polygon_symbol(s)
        k = classify(last, through, None if listed is None else (poly in listed), max_grouped_days)
        kinds[s] = k
        if k in ("grouped", "per_symbol"):
            missing = [d for d in _weekdays_after(last, through) if d not in closed_days]
            bars_to_add[s] = len(missing)
            if k == "grouped":
                grouped_dates.update(missing)
                grouped_dates.add(last)  # overlap check
    return {"kinds": kinds, "bars_to_add": bars_to_add, "grouped_dates": sorted(grouped_dates)}


def print_dry_run(plan: Dict[str, Any], through: datetime.date, have_listing: bool) -> None:
    kinds, bars = plan["kinds"], plan["bars_to_add"]
    by_kind: Dict[str, List[str]] = {}
    for s, k in kinds.items():
        by_kind.setdefault(k, []).append(s)
    n_grouped_calls = len(plan["grouped_dates"])
    n_per_symbol = len(by_kind.get("per_symbol", [])) + len(by_kind.get("full", []))
    total_calls = n_grouped_calls + n_per_symbol
    log(f"DRY RUN through={through} | no Polygon calls, no writes")
    if not have_listing:
        log("note: no cached grouped listing for `through`; delisted/merged symbols cannot be "
            "excluded, so per_symbol/full counts are upper bounds")
    for k in ("current", "grouped", "per_symbol", "full", "not_listed"):
        log(f"  {k:<11} {len(by_kind.get(k, [])):>4} symbols")
    log(f"  grouped calls      {n_grouped_calls} (one per distinct date, overlap dates included)")
    log(f"  per-symbol calls   {n_per_symbol} ({len(by_kind.get('per_symbol', []))} deep gap + "
        f"{len(by_kind.get('full', []))} no doc)")
    log(f"  TOTAL Polygon calls {total_calls} -> {total_calls / (60 / PACE_SECONDS):.0f} min at "
        f"{60 / PACE_SECONDS:.1f}/min")
    log(f"  bars to add (est.) {sum(bars.values())}; max per symbol {max(bars.values(), default=0)}")
    log("  per-symbol detail (symbol kind calls bars):")
    for s in sorted(kinds):
        k = kinds[s]
        if k in ("current", "not_listed"):
            continue
        calls = {"grouped": 0, "per_symbol": 1, "full": 1}[k]
        log(f"    {s:<8} {k:<10} per_symbol_calls={calls} bars={bars.get(s, '~370 (full history)')}")
    if by_kind.get("not_listed"):
        log(f"  skipped (not in grouped(through)): {', '.join(sorted(by_kind['not_listed']))}")


# ---------------------------------------------------------
# APPLY
# ---------------------------------------------------------
def _per_symbol_call(client: GroupedClient, fn, *args):
    if not client.budget_left():
        raise PolygonStop("max_calls reached")
    client.pace()
    client.calls += 1
    try:
        return fn(*args)
    except RuntimeError as e:
        if "429" in str(e):
            log(f"per-symbol | RATE-LIMITED | backing off {RATE_LIMIT_BACKOFF_SECONDS}s")
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
            client.pace()
            client.calls += 1
            try:
                return fn(*args)
            except RuntimeError as e2:
                raise PolygonStop(f"per-symbol RATE-LIMITED twice: {e2}")
        raise


def _splits_since(client: GroupedClient, poly_symbol: str, since: datetime.date) -> Optional[list]:
    """
    Splits for one symbol executing on/after `since`. [] = verified none;
    None = the check itself failed (caller must NOT treat that as "no split").
    Counts against the call budget and pacing like every other call.
    """
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        return None
    for attempt in (1, 2):
        if not client.budget_left():
            raise PolygonStop("max_calls reached")
        client.pace()
        client.calls += 1
        try:
            resp = requests.get(
                SPLITS_URL,
                params={"ticker": poly_symbol, "execution_date.gte": since.isoformat(),
                        "limit": 100, "apiKey": key},
                timeout=30,
            )
        except requests.RequestException as e:
            log(f"{poly_symbol} | splits check ERROR | {type(e).__name__}")
            return None
        if resp.status_code == 429 and attempt == 1:
            log(f"{poly_symbol} | splits check RATE-LIMITED | backing off {RATE_LIMIT_BACKOFF_SECONDS}s")
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
            continue
        break
    if resp.status_code == 429:
        raise PolygonStop("splits check RATE-LIMITED twice")
    if resp.status_code != 200:
        log(f"{poly_symbol} | splits check HTTP-{resp.status_code}")
        return None
    results = resp.json().get("results")
    return results if isinstance(results, list) else None


def _full_refetch(client: GroupedClient, symbol: str, through: datetime.date, reason: str) -> bool:
    results = _per_symbol_call(client, fetch_full_history, symbol)
    if not results:
        log(f"{symbol} | full refetch EMPTY ({reason}) | doc left unchanged")
        return False
    norm = _normalize_polygon_results(results)
    candles = {"open": norm["open"], "high": norm["high"], "low": norm["low"],
               "close": norm["close"], "volume": norm["volume"], "ts": norm["ts"]}
    meta = _finish_meta({"symbol": symbol}, candles, through)
    _save_firestore_candles(symbol, {"candles": candles, "meta": meta})
    log(f"{symbol} | full refetch done ({reason}) | bars={len(candles['close'])}")
    return True


def apply_plan(
    symbols: List[str],
    last_dates: Dict[str, Optional[datetime.date]],
    through: datetime.date,
    plan: Dict[str, Any],
    client: GroupedClient,
    closed_days: Set[datetime.date],
) -> Dict[str, Any]:
    stats = {"appended_symbols": 0, "bars_added": 0, "flagged": [], "full": 0, "skipped_no_data": [],
             "unverified_clean": 0, "unverified_refetched": 0, "unverified_failed": [],
             "stopped": None}
    kinds = plan["kinds"]
    try:
        grouped: Dict[datetime.date, Dict[str, list]] = {}
        for d in plan["grouped_dates"]:
            blob = client.get(d)
            grouped[d] = blob["bars"]
            if blob["count"] == 0:
                closed_days.add(d)

        for s in symbols:
            k = kinds[s]
            poly = normalize_polygon_symbol(s)
            if k == "full":
                if _full_refetch(client, s, through, "no candle doc"):
                    stats["full"] += 1
                continue
            if k not in ("grouped", "per_symbol"):
                continue

            doc = _read_firestore_candles(s)
            if not doc or not (doc.get("candles") or {}).get("ts"):
                continue
            candles = {key: list(doc["candles"].get(key, [])) for key in CANDLE_KEYS}
            last = last_dates[s]
            new_bars: Dict[datetime.date, List[float]] = {}
            overlap: Optional[bool] = None

            if k == "grouped":
                lb = grouped.get(last, {}).get(poly)
                overlap = overlap_ok(_close_on(candles, last), lb[3] if lb else None)
                for d, bars in grouped.items():
                    if d > last and poly in bars and d <= through:
                        new_bars[d] = bars[poly]
            else:
                start_ms = _midnight_et_ms(last)
                end_ms = int((datetime.datetime.combine(through, datetime.time()) +
                              datetime.timedelta(days=2)).timestamp() * 1000)
                results = _per_symbol_call(client, _polygon_fetch, poly, start_ms, end_ms)
                for r in results or []:
                    d = _ms_to_date(r["t"])
                    if d == last:
                        overlap = overlap_ok(_close_on(candles, last), r.get("c"))
                    elif last < d <= through:
                        new_bars[d] = [r.get("o"), r.get("h"), r.get("l"), r.get("c"), r.get("v")]

            if overlap is False:
                stats["flagged"].append(s)
                log(f"{s} | OVERLAP MISMATCH on {last} (>0.1%) -> full refetch instead of append")
                if _full_refetch(client, s, through, "overlap mismatch / possible split"):
                    stats["full"] += 1
                continue
            if not new_bars:
                stats["skipped_no_data"].append(s)
                continue

            unverified = overlap is None
            if unverified:
                # Overlap couldn't run (no bar at `last`). Never append blind:
                # confirm no split since the last stored bar, else refetch.
                splits = _splits_since(client, poly, last)
                if splits is None or splits:
                    why = "splits check failed" if splits is None else f"split since {last}"
                    log(f"{s} | overlap UNVERIFIED + {why} -> full refetch instead of append")
                    if _full_refetch(client, s, through, f"unverified overlap, {why}"):
                        stats["full"] += 1
                        stats["unverified_refetched"] += 1
                    else:
                        stats["unverified_failed"].append(s)
                    continue

            added = append_bars(candles, new_bars)
            if added:
                meta = _finish_meta(doc.get("meta"), candles, through)
                _save_firestore_candles(s, {"candles": candles, "meta": meta})
                stats["appended_symbols"] += 1
                stats["bars_added"] += added
                if unverified:
                    stats["unverified_clean"] += 1
                    log(f"{s} | appended {added} bar(s) | overlap unverified, splits check clean")
    except PolygonStop as e:
        stats["stopped"] = str(e)
        log(f"STOPPED: {e} | re-run to resume (cached grouped days and already-appended dates are skipped)")
    stats["polygon_calls"] = client.calls
    return stats


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def _default_through() -> datetime.date:
    d = datetime.datetime.now(_ET).date() - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--through", type=datetime.date.fromisoformat, default=None,
                    help="refresh up to this date (default: previous weekday, ET)")
    ap.add_argument("--apply", action="store_true", help="make Polygon calls and write (default: dry run)")
    ap.add_argument("--max-calls", type=int, default=None, help="cap Polygon calls this invocation")
    ap.add_argument("--max-grouped-days", type=int, default=DEFAULT_MAX_GROUPED_DAYS)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--universe-only", action="store_true",
                    help="REAL_TICKERS only (default also includes every existing candle doc, "
                         "e.g. ETFs and one-off scan symbols)")
    args = ap.parse_args(argv)

    through = args.through or _default_through()
    symbols = load_symbols(args.symbols, args.universe_only)
    wanted = {normalize_polygon_symbol(s) for s in symbols}
    last_dates = load_last_dates(symbols)
    client = GroupedClient(args.cache_dir, wanted, args.max_calls)

    closed_days: Set[datetime.date] = set()
    listing = client.cached(through)
    if listing is None and args.apply:
        try:
            listing = client.get(through)
        except PolygonStop as e:
            log(f"cannot fetch grouped({through}): {e}")
            return 2
    listed = set(listing["bars"]) if listing else None
    if listing is not None and listing["count"] == 0:
        log(f"grouped({through}) returned no tickers (holiday?) -- pick another --through")
        return 2
    # Closed days are known only for dates already in the cache.
    for p in os.listdir(args.cache_dir):
        if p.startswith("grouped_"):
            try:
                blob = json.load(open(os.path.join(args.cache_dir, p)))
                if blob.get("count") == 0:
                    closed_days.add(datetime.date.fromisoformat(blob["date"]))
            except Exception:
                pass

    plan = build_plan(symbols, last_dates, through, listed, closed_days, args.max_grouped_days)
    if not args.apply:
        print_dry_run(plan, through, listed is not None)
        return 0

    stats = apply_plan(symbols, last_dates, through, plan, client, closed_days)
    log(f"DONE {json.dumps(stats)}")
    return 1 if stats["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
