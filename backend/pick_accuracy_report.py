# backend/pick_accuracy_report.py
# =========================================================
# Alphaclara pick accuracy report — pure logic.
#
# No Firestore, no FastAPI, no cron — safe to reuse from a route, a script,
# or a future cron job. Firestore reads live in backend/pick_tracking.py
# (get_checked_picks_for_report()); this module only transforms data
# already in hand.
#
# Built from a real post-mortem (2026-07-31) that found record_picks_for_
# tracking() writes one row per cron cycle a symbol appears in a ranked
# list, by design, with no dedup -- a naive analysis over raw rows
# overcounts by however many cron cycles a symbol happened to sit in the
# list (one real example: the same pick appeared 96 times with an
# identical checked return every time). dedupe_checked_picks() is the
# fix. That same post-mortem found a "pattern" (Neutral model view ->
# 9.1% win rate) that dissolved on inspection into one repeatedly-re-
# picked losing symbol counted 4 times among only 6 distinct symbols --
# the confounding guard below (distinct_symbols/dominant_symbol_share)
# is built specifically to catch that case automatically.
# =========================================================

import statistics
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict

FACTOR_SCORE_KEYS = [
    "momentum", "trend", "pattern", "bullbrain", "volume", "early_expansion",
]

# Same thresholds used to catch the fake PANW/Neutral signal in the
# 2026-07-31 post-mortem: that subgroup had 6 distinct symbols (< 10) and
# one symbol (PANW) at 36% share (> 0.30).
MIN_DISTINCT_SYMBOLS = 10
MAX_DOMINANT_SYMBOL_SHARE = 0.30


def dedupe_checked_picks(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapses pick_tracking's one-row-per-cron-cycle write pattern down to
    one row per real distinct (symbol, pick_date, horizon) outcome. Keeps
    the first-seen row for each combo -- duplicates share the same
    checked_return_pct by construction (same real-world pick, same
    resolution), so which duplicate survives doesn't matter.
    """
    seen: Dict[tuple, Dict[str, Any]] = {}

    for doc in raw_docs:
        symbol = doc.get("symbol")
        pick_date = doc.get("pick_date")
        horizons = doc.get("horizons") or {}

        if not symbol or not pick_date or not isinstance(horizons, dict):
            continue

        for horizon_key, horizon in horizons.items():
            if not isinstance(horizon, dict) or horizon.get("status") != "checked":
                continue

            key = (symbol, pick_date, horizon_key)
            if key in seen:
                continue

            seen[key] = {
                "symbol": symbol,
                "pick_date": pick_date,
                "horizon": horizon_key,
                "checked_return_pct": horizon.get("return_pct"),
                "checked_at": horizon.get("checked_at"),
                "pick_source": doc.get("pick_source"),
                "pick_decision_reasons": doc.get("pick_decision_reasons"),
                "pick_pattern_stats": doc.get("pick_pattern_stats"),
                "pick_model_view": doc.get("pick_model_view"),
                "pick_market_regime": doc.get("pick_market_regime"),
                "pick_factor_scores": doc.get("pick_factor_scores"),
                "pick_setup_label": doc.get("pick_setup_label"),
                "pick_score": doc.get("pick_score"),
            }

    return list(seen.values())


def _return_stats(picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    returns = [
        p["checked_return_pct"] for p in picks
        if isinstance(p.get("checked_return_pct"), (int, float))
    ]

    if not returns:
        return {
            "n": 0, "positive": 0, "negative": 0, "zero": 0,
            "pct_positive": None, "mean_return_pct": None,
            "median_return_pct": None, "stdev_return_pct": None,
        }

    positive = sum(1 for r in returns if r > 0)
    negative = sum(1 for r in returns if r < 0)
    zero = sum(1 for r in returns if r == 0)

    return {
        "n": len(returns),
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "pct_positive": round(100 * positive / len(returns), 1),
        "mean_return_pct": round(statistics.mean(returns), 2),
        "median_return_pct": round(statistics.median(returns), 2),
        "stdev_return_pct": round(statistics.stdev(returns), 2) if len(returns) > 1 else None,
    }


def _confounding_guard(picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The check that would have caught the fake PANW/Neutral signal: any
    subgroup this small or this dominated by one repeated symbol gets
    flagged, not silently trusted.
    """
    symbol_counts = Counter(p.get("symbol") for p in picks if p.get("symbol"))
    distinct_symbols = len(symbol_counts)
    distinct_pick_dates = len(set(p.get("pick_date") for p in picks if p.get("pick_date")))

    dominant_symbol_share = 0.0
    dominant_symbol = None
    if symbol_counts and picks:
        dominant_symbol, dominant_count = symbol_counts.most_common(1)[0]
        dominant_symbol_share = round(dominant_count / len(picks), 3)

    low_confidence = (
        distinct_symbols < MIN_DISTINCT_SYMBOLS
        or dominant_symbol_share > MAX_DOMINANT_SYMBOL_SHARE
    )

    return {
        "distinct_symbols": distinct_symbols,
        "distinct_pick_dates": distinct_pick_dates,
        "dominant_symbol": dominant_symbol if dominant_symbol_share > MAX_DOMINANT_SYMBOL_SHARE else None,
        "dominant_symbol_share": dominant_symbol_share,
        "low_confidence": low_confidence,
    }


def _subgroup_breakdown(picks: List[Dict[str, Any]], key_fn) -> Dict[str, Any]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    missing = 0

    for p in picks:
        key = key_fn(p)
        if key is None:
            missing += 1
            continue
        groups[key].append(p)

    total = len(picks)
    if total and missing == total:
        return {
            "insufficient_data": True,
            "reason": "field is missing/null on every checked pick in this window",
            "groups": {},
        }

    out = {}
    for key, group_picks in groups.items():
        out[str(key)] = {
            **_return_stats(group_picks),
            **_confounding_guard(group_picks),
        }

    return {"insufficient_data": False, "groups": out, "missing_field_count": missing}


def _factor_score_comparison(winners: List[Dict[str, Any]], losers: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for factor in FACTOR_SCORE_KEYS:
        winner_vals = [
            (p.get("pick_factor_scores") or {}).get(factor) for p in winners
            if isinstance((p.get("pick_factor_scores") or {}).get(factor), (int, float))
        ]
        loser_vals = [
            (p.get("pick_factor_scores") or {}).get(factor) for p in losers
            if isinstance((p.get("pick_factor_scores") or {}).get(factor), (int, float))
        ]

        out[factor] = {
            "winners_mean": round(statistics.mean(winner_vals), 1) if winner_vals else None,
            "winners_n": len(winner_vals),
            "losers_mean": round(statistics.mean(loser_vals), 1) if loser_vals else None,
            "losers_n": len(loser_vals),
            "delta": (
                round(statistics.mean(winner_vals) - statistics.mean(loser_vals), 1)
                if winner_vals and loser_vals else None
            ),
        }

    return out


def _report_for_horizon(picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    winners = [p for p in picks if isinstance(p.get("checked_return_pct"), (int, float)) and p["checked_return_pct"] > 0]
    losers = [p for p in picks if isinstance(p.get("checked_return_pct"), (int, float)) and p["checked_return_pct"] < 0]

    return {
        "overall": _return_stats(picks),
        "by_pick_source": _subgroup_breakdown(picks, lambda p: p.get("pick_source")),
        "by_market_regime": _subgroup_breakdown(picks, lambda p: p.get("pick_market_regime")),
        "by_model_view_bias": _subgroup_breakdown(
            picks, lambda p: (p.get("pick_model_view") or {}).get("bias")
        ),
        "factor_scores_winners_vs_losers": _factor_score_comparison(winners, losers),
        "pick_date_range": {
            "min": min((p["pick_date"] for p in picks if p.get("pick_date")), default=None),
            "max": max((p["pick_date"] for p in picks if p.get("pick_date")), default=None),
        },
    }


def _horizon_sort_key(horizon: str) -> int:
    # "5d"/"20d" sort lexicographically as ["20d", "5d"] (ASCII '2' < '5'),
    # not shortest-first -- extract the numeric value so "shortest horizon"
    # below is actually shortest, not alphabetical.
    try:
        return int("".join(ch for ch in horizon if ch.isdigit()))
    except Exception:
        return 0


def _build_summary(horizons_report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    A small, flat, stable summary for lightweight consumers (e.g. a UI
    disclosure line) that shouldn't need to know about the horizon-keyed
    breakdown structure. Picks the shortest horizon with any resolved
    picks -- resolves "which horizon does this describe" once, here,
    instead of leaving every caller to guess.
    """
    for horizon in sorted(horizons_report.keys(), key=_horizon_sort_key):
        overall = horizons_report[horizon]["overall"]
        if overall.get("n"):
            return {
                "horizon": horizon,
                "n": overall["n"],
                "pct_positive": overall["pct_positive"],
                "mean_return_pct": overall["mean_return_pct"],
                "pick_date_range": horizons_report[horizon]["pick_date_range"],
            }

    return None


def build_accuracy_report(deduped_picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Takes already-deduped picks (see dedupe_checked_picks()) and returns
    the full report, segmented by horizon -- a 5d return and a 20d return
    are different bets and are never pooled together.
    """
    by_horizon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in deduped_picks:
        horizon = p.get("horizon")
        if horizon:
            by_horizon[horizon].append(p)

    horizons_report = {
        horizon: _report_for_horizon(picks)
        for horizon, picks in sorted(by_horizon.items(), key=lambda kv: _horizon_sort_key(kv[0]))
    }

    return {
        "schema_version": "pick_accuracy_report_v1",
        "total_distinct_picks": len(deduped_picks),
        "summary": _build_summary(horizons_report),
        "horizons": horizons_report,
        "confounding_guard": {
            "min_distinct_symbols": MIN_DISTINCT_SYMBOLS,
            "max_dominant_symbol_share": MAX_DOMINANT_SYMBOL_SHARE,
        },
    }


def _render_subgroup_breakdown_md(title: str, breakdown: Dict[str, Any]) -> List[str]:
    lines = [f"### {title}"]

    if breakdown.get("insufficient_data"):
        lines.append(f"_Insufficient data: {breakdown.get('reason')}_")
        return lines

    groups = breakdown.get("groups") or {}
    if not groups:
        lines.append("_No groups._")
        return lines

    for key, g in sorted(groups.items(), key=lambda kv: kv[1].get("n", 0), reverse=True):
        flag = " ⚠️ LOW CONFIDENCE" if g.get("low_confidence") else ""
        lines.append(
            f"- **{key}**: n={g['n']}, {g['pct_positive']}% positive, "
            f"mean={g['mean_return_pct']}%, median={g['median_return_pct']}%"
            f" — {g['distinct_symbols']} distinct symbols, {g['distinct_pick_dates']} distinct days"
            f"{flag}"
        )
        if g.get("dominant_symbol"):
            lines.append(
                f"  - dominant symbol: {g['dominant_symbol']} "
                f"({round(g['dominant_symbol_share']*100)}% of this subgroup)"
            )

    return lines


def render_markdown_report(report: Dict[str, Any]) -> str:
    """
    Human-readable rendering of build_accuracy_report()'s output, for quick
    manual review (?format=markdown) instead of reading raw JSON.
    """
    lines = [
        "# Alphaclara Pick Accuracy Report",
        "",
        f"Total distinct picks analyzed: **{report.get('total_distinct_picks', 0)}** "
        "(deduped from raw pick_tracking rows — see dedupe_checked_picks()).",
        "",
    ]

    for horizon, h in sorted((report.get("horizons") or {}).items()):
        o = h["overall"]
        date_range = h.get("pick_date_range") or {}
        lines.append(f"## {horizon} horizon")
        lines.append(
            f"pick_dates {date_range.get('min')} → {date_range.get('max')} | "
            f"n={o['n']}, {o['pct_positive']}% positive "
            f"({o['positive']} pos / {o['negative']} neg / {o['zero']} zero)"
        )
        lines.append(
            f"mean={o['mean_return_pct']}%, median={o['median_return_pct']}%, "
            f"stdev={o['stdev_return_pct']}%"
        )
        lines.append("")

        lines += _render_subgroup_breakdown_md("By pick_source", h["by_pick_source"])
        lines.append("")
        lines += _render_subgroup_breakdown_md("By market_regime", h["by_market_regime"])
        lines.append("")
        lines += _render_subgroup_breakdown_md("By model_view.bias", h["by_model_view_bias"])
        lines.append("")

        lines.append("### Factor scores — winners vs. losers (mean)")
        for factor, fs in h["factor_scores_winners_vs_losers"].items():
            if fs["delta"] is None:
                continue
            lines.append(
                f"- **{factor}**: winners={fs['winners_mean']} (n={fs['winners_n']}), "
                f"losers={fs['losers_mean']} (n={fs['losers_n']}), delta={fs['delta']:+}"
            )
        lines.append("")

    guard = report.get("confounding_guard") or {}
    lines.append(
        f"_Confounding guard: subgroups flagged LOW CONFIDENCE when distinct_symbols < "
        f"{guard.get('min_distinct_symbols')} or one symbol is > "
        f"{round((guard.get('max_dominant_symbol_share') or 0)*100)}% of the subgroup's rows._"
    )

    return "\n".join(lines)
