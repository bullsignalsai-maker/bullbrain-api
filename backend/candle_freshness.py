# backend/candle_freshness.py
# =========================================================
# Per-row candle freshness, derived from candle data a caller has
# ALREADY loaded -- no Polygon/API calls.
#
# get_candles() silently serves stale cache once Polygon is in its 429
# cooldown, so a tracking row built from it is indistinguishable from one
# built from current bars. These three fields make that visible later:
#   candle_last_bar_date       newest bar's date (YYYY-MM-DD)
#   candle_age_trading_days    trading days between that bar and the
#                              previous close (0 = caught up)
#   candle_stale               True when the newest bar is at least one
#                              trading day behind the previous close
#
# "Previous close" = the latest trading day strictly before today (ET).
# Deliberately not "today's close": Polygon's plan only serves a day's
# bar after end of day, so even a perfectly fresh symbol lags the
# same-day bar at the 4:35 PM ET close run.
#
# Trading-day calendar: same candle-derived approach as market_calendar
# (static holiday lists miss ad-hoc closures). A weekday inside the span
# the reference symbols cover but absent from their bars is a closed day;
# weekdays past the span are assumed open. The reference read is 2
# Firestore doc reads, cached per process.
# =========================================================

import datetime
import time
from typing import Any, Dict, Optional, Set

import pytz

_ET = pytz.timezone("America/New_York")
_CALENDAR_TTL_SECONDS = 3600
_calendar_cache: Dict[str, Any] = {"loaded_at": 0.0, "days": None}

_NULL_FRESHNESS = {
    "candle_last_bar_date": None,
    "candle_age_trading_days": None,
    "candle_stale": None,
}


def _trading_day_set() -> Set[str]:
    now = time.time()
    if (
        _calendar_cache["days"] is None
        or now - _calendar_cache["loaded_at"] > _CALENDAR_TTL_SECONDS
    ):
        try:
            from backend.market_calendar import load_recent_trading_days
            _calendar_cache["days"] = load_recent_trading_days()
        except Exception:
            _calendar_cache["days"] = set()
        _calendar_cache["loaded_at"] = now
    return _calendar_cache["days"]


def _is_trading_day(d: datetime.date, known: Set[str]) -> bool:
    if d.weekday() >= 5:
        return False
    if not known:
        return True
    iso = d.isoformat()
    if iso in known:
        return True
    # Weekday with no reference bar: closed only if it sits inside the
    # span the reference bars cover; past that span we can't tell a
    # closure from a not-yet-refreshed reference, so assume open.
    return iso > max(known) or iso < min(known)


def _bar_date(ts: Any) -> Optional[datetime.date]:
    if not isinstance(ts, (int, float)):
        return None
    seconds = ts / 1000 if ts > 10_000_000_000 else ts
    return datetime.datetime.utcfromtimestamp(seconds).date()


def candle_freshness(
    candles: Optional[Dict[str, Any]],
    today: Optional[datetime.date] = None,
    trading_days: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    `candles` is whatever get_candles() returned (its timestamps live
    under "timestamp"; the raw cache layout uses "ts"). All-None on
    missing/unparseable input rather than raising -- same degrade-quietly
    convention as the rest of the app.
    """
    try:
        ts_list = (candles or {}).get("timestamp") or (candles or {}).get("ts")
        last_bar = _bar_date(ts_list[-1]) if ts_list else None
        if last_bar is None:
            return dict(_NULL_FRESHNESS)

        today = today or datetime.datetime.now(_ET).date()
        known = trading_days if trading_days is not None else _trading_day_set()

        prev_close = today - datetime.timedelta(days=1)
        while not _is_trading_day(prev_close, known):
            prev_close -= datetime.timedelta(days=1)

        age = 0
        d = last_bar + datetime.timedelta(days=1)
        while d <= prev_close:
            if _is_trading_day(d, known):
                age += 1
            d += datetime.timedelta(days=1)

        return {
            "candle_last_bar_date": last_bar.isoformat(),
            "candle_age_trading_days": age,
            "candle_stale": age >= 1,
        }
    except Exception:
        return dict(_NULL_FRESHNESS)
