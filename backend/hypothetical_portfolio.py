# backend/hypothetical_portfolio.py
# =========================================================
# Hypothetical $1,000-across-every-pick illustration -- pure logic.
#
# No Firestore, no FastAPI -- build_hypothetical_portfolio() takes
# already-fetched deduped picks and a {symbol: current_price} dict, so
# it's directly unit-testable. The route (main.py) does the I/O: raw
# pick_tracking docs via backend.pick_tracking.get_checked_picks_for_report,
# current prices via backend.stock_repo.get_stocks().
#
# Deliberately NOT built on top of pick_accuracy_report.py's
# dedupe_checked_picks()/build_accuracy_report() -- those answer "how did
# resolved picks do against their own 5d/20d checkpoint", filtered to
# horizons.status == "checked". This answers a different question ("what
# would every pick, resolved or still open, be worth AT CURRENT PRICE
# today"), so it never looks at horizons/status at all -- only pick_price
# vs. a fresh current-price lookup, for every distinct pick.
#
# Regulatory-adjacent: this is investment-performance-adjacent marketing
# copy. The disclaimer strings below are a careful draft, not a certified
# compliance artifact -- see the feature's plan doc for that caveat.
# =========================================================

from typing import Any, Dict, List, Optional

STARTING_AMOUNT = 1000.0

METHODOLOGY_VERSION = "hypothetical_portfolio_v1"

SHORT_DISCLAIMER = (
    "Hypothetical illustration only — not a real account or investment "
    "recommendation. Assumes an equal $1,000 ÷ N allocation across "
    "every tracked pick, bought at its recorded price and held with no "
    "rebalancing, fees, taxes, or slippage. Past performance does not "
    "predict future results."
)

SURVIVORSHIP_BIAS_SENTENCE_TEMPLATE = (
    "This calculation excludes {n_excluded} pick(s) where a current price "
    "is unavailable (for example, the stock was delisted). Excluding "
    "these can make the result look better than reality, since stocks "
    "removed from tracking may have performed worse than the ones still "
    "priced."
)

LONG_DISCLAIMER_TEMPLATE = (
    "This is a hypothetical illustration, not a real investment or "
    "account. It shows what a $1,000 balance, split equally across every "
    "distinct Alphaclara Pick, would be worth today if you had bought "
    "each pick at its recorded price on its pick date and simply held it "
    "— no rebalancing, no reinvesting, no adding new picks after the "
    "fact.\n\n"
    "It ignores trading fees, bid/ask spreads, slippage, taxes, and "
    "dividends, and assumes you could have traded at the exact recorded "
    "price at the exact recorded time, which is rarely possible in real "
    "markets.{survivorship_clause}\n\n"
    "Past performance — real or hypothetical — never guarantees future "
    "results. This is for informational and educational purposes only "
    "and is not financial advice."
)


def dedupe_picks_for_valuation(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapses pick_tracking's one-row-per-cron-cycle write pattern down to
    one row per real (symbol, pick_date) pick -- same dedup key as
    dedupe_checked_picks() in pick_accuracy_report.py, but WITHOUT its
    horizons.status == "checked" filter, so still-open ("pending") picks
    are kept too. Duplicates share the same real-world pick_price by
    construction (same pick event), so which duplicate survives doesn't
    matter.
    """
    seen: Dict[tuple, Dict[str, Any]] = {}

    for doc in raw_docs:
        symbol = doc.get("symbol")
        pick_date = doc.get("pick_date")
        pick_price = doc.get("pick_price")

        if not symbol or not pick_date:
            continue

        key = (symbol, pick_date)
        if key in seen:
            continue

        seen[key] = {
            "symbol": symbol,
            "pick_date": pick_date,
            "pick_price": pick_price,
        }

    return list(seen.values())


def build_hypothetical_portfolio(
    deduped_picks: List[Dict[str, Any]],
    current_prices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Equal-weight $1,000 across every INCLUDED pick (see module docstring
    for exclusion rules). allocation_per_pick is computed over n_included,
    not the total universe -- the classic definition of survivorship
    bias, which SURVIVORSHIP_BIAS_SENTENCE_TEMPLATE names explicitly
    whenever n_excluded > 0, rather than silently absorbing it.
    """
    included: List[Dict[str, Any]] = []
    excluded_price_unavailable = 0
    excluded_invalid_pick_price = 0

    for pick in deduped_picks:
        pick_price = pick.get("pick_price")
        if not isinstance(pick_price, (int, float)) or pick_price <= 0:
            excluded_invalid_pick_price += 1
            continue

        current_price = current_prices.get(pick["symbol"])
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            excluded_price_unavailable += 1
            continue

        included.append({
            "symbol": pick["symbol"],
            "pick_date": pick["pick_date"],
            "pick_price": pick_price,
            "current_price": current_price,
            "return_pct": round((current_price / pick_price - 1) * 100, 2),
        })

    n_included = len(included)
    n_excluded = excluded_price_unavailable + excluded_invalid_pick_price

    if n_included == 0:
        return {
            "schema_version": METHODOLOGY_VERSION,
            "starting_amount": STARTING_AMOUNT,
            "n_included": 0,
            "n_excluded": n_excluded,
            "n_excluded_price_unavailable": excluded_price_unavailable,
            "n_excluded_invalid_pick_price": excluded_invalid_pick_price,
            "hypothetical_current_value": None,
            "hypothetical_total_return_pct": None,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "pick_date_range": {"min": None, "max": None},
            "disclaimer": SHORT_DISCLAIMER,
            "long_disclaimer": _build_long_disclaimer(n_excluded),
        }

    allocation_per_pick = STARTING_AMOUNT / n_included

    for p in included:
        p["allocated_amount"] = round(allocation_per_pick, 2)
        p["current_value"] = round(allocation_per_pick * (p["current_price"] / p["pick_price"]), 2)

    hypothetical_current_value = round(sum(p["current_value"] for p in included), 2)
    hypothetical_total_return_pct = round((hypothetical_current_value / STARTING_AMOUNT - 1) * 100, 2)

    n_positive = sum(1 for p in included if p["return_pct"] > 0)
    n_negative = sum(1 for p in included if p["return_pct"] < 0)
    n_zero = sum(1 for p in included if p["return_pct"] == 0)

    pick_dates = [p["pick_date"] for p in included]

    return {
        "schema_version": METHODOLOGY_VERSION,
        "starting_amount": STARTING_AMOUNT,
        "n_included": n_included,
        "n_excluded": n_excluded,
        "n_excluded_price_unavailable": excluded_price_unavailable,
        "n_excluded_invalid_pick_price": excluded_invalid_pick_price,
        "hypothetical_current_value": hypothetical_current_value,
        "hypothetical_total_return_pct": hypothetical_total_return_pct,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "pick_date_range": {"min": min(pick_dates), "max": max(pick_dates)},
        "disclaimer": SHORT_DISCLAIMER,
        "long_disclaimer": _build_long_disclaimer(n_excluded),
    }


def _build_long_disclaimer(n_excluded: int) -> str:
    survivorship_clause = (
        "\n\n" + SURVIVORSHIP_BIAS_SENTENCE_TEMPLATE.format(n_excluded=n_excluded)
        if n_excluded > 0
        else ""
    )
    return LONG_DISCLAIMER_TEMPLATE.format(survivorship_clause=survivorship_clause)
