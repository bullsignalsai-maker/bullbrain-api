# backend/market_calendar.py
# =========================================================
# Trading-day calendar derived from real candle data, not a hardcoded
# holiday list.
#
# A static weekday+holiday-list calendar was checked against AAPL's real
# 393-day cached candle history before choosing this approach: it missed
# 2025-01-09, a real NYSE/NASDAQ closure (National Day of Mourning) that
# isn't in any recurring annual holiday list. Cross-checked AAPL against
# MSFT over the same range -- identical trading-date sets (393/393) --
# confirming 2 reliable reference symbols are a trustworthy proxy for
# "was the market open," without needing a maintained holiday list at all.
#
# Reads already-cached candle data only (_read_firestore_candles is a pure
# Firestore read) -- no new fetch, no API cost. Both reference symbols are
# in CORE_UNIVERSE and already refreshed every market_cron cycle.
# =========================================================

import datetime
from typing import Set

from backend.candle_store import _read_firestore_candles

REFERENCE_SYMBOLS = ["AAPL", "MSFT"]
DEFAULT_LOOKBACK_DAYS = 40


def _candle_dates(symbol: str) -> Set[str]:
    cached = _read_firestore_candles(symbol)
    if not cached:
        return set()

    ts_list = (cached.get("candles") or {}).get("ts") or []
    dates: Set[str] = set()
    for t in ts_list:
        if not isinstance(t, (int, float)):
            continue
        seconds = t / 1000 if t > 10_000_000_000 else t
        dates.add(datetime.datetime.utcfromtimestamp(seconds).date().isoformat())
    return dates


def load_recent_trading_days(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Set[str]:
    """
    Union of real recorded trading dates from the reference symbols,
    bounded to the last `lookback_days` calendar days -- comfortably
    covers the 20-trading-day (~28 calendar day) checker window with
    margin. Union, not intersection: one symbol's own data gap on a given
    day shouldn't be mistaken for the whole market being closed.
    """
    cutoff = (
        datetime.datetime.utcnow().date() - datetime.timedelta(days=lookback_days)
    ).isoformat()

    all_dates: Set[str] = set()
    for sym in REFERENCE_SYMBOLS:
        all_dates |= _candle_dates(sym)

    return {d for d in all_dates if d >= cutoff}


def trading_days_elapsed(
    pick_date_str: str,
    as_of_date_str: str,
    trading_day_set: Set[str],
) -> int:
    """
    Number of trading days strictly after pick_date_str, up to and
    including as_of_date_str, per the real trading_day_set. pick_date
    itself is day 0, not counted.

    trading_day_set should include "today" explicitly if the caller has
    already confirmed today is a trading day by other means (e.g. the
    checker only ever runs when market_cron's own gating has already
    established this) -- today's own candle may not be published yet at
    checker run time, so this function deliberately does not special-case
    "today"; that trust is the caller's responsibility to add to the set.
    """
    pick_date = datetime.date.fromisoformat(pick_date_str)
    as_of = datetime.date.fromisoformat(as_of_date_str)

    count = 0
    d = pick_date + datetime.timedelta(days=1)
    while d <= as_of:
        if d.isoformat() in trading_day_set:
            count += 1
        d += datetime.timedelta(days=1)

    return count
