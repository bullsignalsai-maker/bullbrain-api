# Offline checks for backend/candle_refresh_grouped.py -- no network, no Firestore.
# Run: PYTHONPATH=. python3 scripts/test_candle_refresh_grouped.py
import datetime as dt
import json
import tempfile

import backend.candle_refresh_grouped as m

D = dt.date
THROUGH, LAST = D(2026, 10, 2), D(2026, 9, 30)
mid = m._midnight_et_ms


def doc(close, last=LAST):
    return {"candles": {"open": [1], "high": [1], "low": [1], "close": [close], "volume": [1],
                        "ts": [mid(last)]}, "meta": {"source": "polygon"}}


def setup(stored, grouped_bars, splits_fn):
    store = {k: json.loads(json.dumps(v)) for k, v in stored.items()}
    saved, refetched = {}, []
    m._read_firestore_candles = lambda s: json.loads(json.dumps(store[s])) if s in store else None
    m._save_firestore_candles = lambda s, p: saved.__setitem__(s, p)

    def full(client, symbol, through, reason):
        refetched.append((symbol, reason))
        saved[symbol] = {"candles": {"close": [999.0]}, "meta": {}}
        return True
    m._full_refetch = full
    m._splits_since = lambda client, poly, since: splits_fn(poly)
    m.PACE_SECONDS = 0
    d = tempfile.mkdtemp()
    syms = sorted(stored)
    for day, bars in grouped_bars.items():
        json.dump({"date": day, "count": 5000, "covers": syms, "bars": bars}, open(f"{d}/grouped_{day}.json", "w"))
    client = m.GroupedClient(d, set(syms), None)
    last_dates = {s: LAST for s in syms}
    plan = m.build_plan(syms, last_dates, THROUGH, set(syms), set(), 20)
    return syms, last_dates, plan, client, saved, refetched


def bars(sym_close_by_day):
    out = {}
    for day, per_sym in sym_close_by_day.items():
        out[day] = {s: [1, 2, 1, c, 5] for s, c in per_sym.items()}
    return out


# 1) verified overlap -> append; mismatch -> refetch (no splits call needed)
g = bars({"2026-09-30": {"OK": 100.05, "BAD": 50.0}, "2026-10-01": {"OK": 101, "BAD": 50.2},
          "2026-10-02": {"OK": 102, "BAD": 50.5}})
syms, ld, plan, cl, saved, refetched = setup({"OK": doc(100.0), "BAD": doc(100.0)}, g,
                                             lambda p: (_ for _ in ()).throw(AssertionError("splits not expected")))
st = m.apply_plan(syms, ld, THROUGH, plan, cl, set())
assert st["appended_symbols"] == 1 and st["flagged"] == ["BAD"] and st["unverified_clean"] == 0
assert saved["OK"]["candles"]["close"] == [100.0, 101, 102]

# 2) unverified (no grouped bar at last date) + no split -> append, unverified_clean
g = bars({"2026-09-30": {}, "2026-10-01": {"U": 101}, "2026-10-02": {"U": 102}})
syms, ld, plan, cl, saved, refetched = setup({"U": doc(100.0)}, g, lambda p: [])
st = m.apply_plan(syms, ld, THROUGH, plan, cl, set())
assert st["unverified_clean"] == 1 and st["unverified_refetched"] == 0 and not refetched, st
assert saved["U"]["candles"]["close"] == [100.0, 101, 102]

# 3) unverified + split since last bar -> refetch, nothing appended
syms, ld, plan, cl, saved, refetched = setup({"U": doc(100.0)}, g, lambda p: [{"ticker": p, "split_from": 1, "split_to": 2}])
st = m.apply_plan(syms, ld, THROUGH, plan, cl, set())
assert st["unverified_refetched"] == 1 and st["unverified_clean"] == 0 and st["appended_symbols"] == 0, st
assert refetched and saved["U"]["candles"]["close"] == [999.0]

# 4) unverified + splits check failed (None) -> refetch, never append blind
syms, ld, plan, cl, saved, refetched = setup({"U": doc(100.0)}, g, lambda p: None)
st = m.apply_plan(syms, ld, THROUGH, plan, cl, set())
assert st["unverified_refetched"] == 1 and st["appended_symbols"] == 0 and "failed" in refetched[0][1], st

# 5) idempotent: a doc already at `through` appends nothing
c = doc(100.0)["candles"]
assert m.append_bars(c, {D(2026, 10, 1): [1, 1, 1, 1, 1]}) == 1
assert m.append_bars(c, {D(2026, 10, 1): [1, 1, 1, 1, 1]}) == 0
print("all candle_refresh_grouped checks passed")
