from typing import Any, Dict, List
from backend.firestore_utils import utc_now_iso


def _num(v: Any) -> float | None:
    try:
        if isinstance(v, (int, float)):
            return float(v)
        return None
    except Exception:
        return None


def _fmt_pct(v: Any) -> str:
    n = _num(v)
    if n is None:
        return "--"
    return f"{abs(n):.2f}%"


def build_market_awareness(
    symbol: str,
    company_name: str | None = None,
    quote: Dict[str, Any] | None = None,
    technical: Dict[str, Any] | None = None,
    pattern: Dict[str, Any] | None = None,
    bullbrain: Dict[str, Any] | None = None,
    decision: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    symbol = (symbol or "").upper().strip()
    company = company_name or symbol

    quote = quote or {}
    technical = technical or {}
    pattern = pattern or {}
    bullbrain = bullbrain or {}
    decision = decision or {}

    change_pct = _num(quote.get("changePct"))
    change = _num(quote.get("change"))
    price = _num(quote.get("price"))

    direction = "mixed"
    if change_pct is not None:
        if change_pct >= 1:
            direction = "rising"
        elif change_pct <= -1:
            direction = "pulling back"
        elif change_pct > 0:
            direction = "slightly higher"
        elif change_pct < 0:
            direction = "slightly lower"
        else:
            direction = "flat"

    pattern_name = (
        pattern.get("pattern")
        or pattern.get("patternLabel")
        or pattern.get("name")
    )

    pattern_bias = (
        pattern.get("bias")
        or pattern.get("patternBias")
    )

    signal = (
        decision.get("finalSignal")
        or bullbrain.get("signal")
        or "HOLD"
    )

    confidence = bullbrain.get("confidence")

    drivers: List[str] = []

    if change_pct is not None:
        move_word = "gained" if change_pct >= 0 else "declined"
        dollar_part = ""
        if change is not None:
            dollar_part = f" (${abs(change):.2f})"
        drivers.append(
            f"Price action: {symbol} {move_word} {_fmt_pct(change_pct)}{dollar_part} today."
        )

    if pattern_name:
        if pattern_bias:
            drivers.append(
                f"Pattern context: {pattern_name} is showing a {pattern_bias} setup."
            )
        else:
            drivers.append(f"Pattern context: {pattern_name} is active.")

    if signal:
        conf_part = ""
        if isinstance(confidence, (int, float)):
            conf_part = f" with {round(float(confidence))}% confidence"
        drivers.append(
            f"AI context: BullBrain rates the setup as {signal}{conf_part}."
        )

    # Lightweight technical context from common nested fields if available
    trend_label = None
    try:
        trend_label = (technical.get("trend") or {}).get("label")
    except Exception:
        trend_label = None

    if trend_label:
        drivers.append(f"Trend context: the current technical trend is {trend_label}.")

    if not drivers:
        drivers.append(
            "No single catalyst is visible from the available quote and technical data."
        )

    if change_pct is None:
        one_liner = f"{symbol} has limited fresh movement data available right now."
    elif change_pct >= 1:
        one_liner = f"{symbol} is rising today, up {_fmt_pct(change_pct)}, with price action showing stronger buyer interest."
    elif change_pct <= -1:
        one_liner = f"{symbol} is pulling back today, down {_fmt_pct(change_pct)}, with sellers controlling the latest move."
    elif change_pct > 0:
        one_liner = f"{symbol} is slightly higher today, up {_fmt_pct(change_pct)}, but the move remains modest."
    elif change_pct < 0:
        one_liner = f"{symbol} is slightly lower today, down {_fmt_pct(change_pct)}, with limited downside pressure so far."
    else:
        one_liner = f"{symbol} is mostly flat today, with no major price move yet."

    # Summary for Home/Watchlist
    if pattern_name:
        summary = (
            f"{company} is {direction} today. "
            f"The move is supported by current price action and the active {pattern_name} pattern. "
            f"AI signal remains {signal}, so users should view this as market context, not a trade recommendation."
        )
    else:
        summary = (
            f"{company} is {direction} today. "
            f"The move is mainly reflected in current price action and technical conditions. "
            f"AI signal remains {signal}, so users should view this as market context, not a trade recommendation."
        )

    return {
        "title": f"Why {symbol} is moving today",
        "oneLiner": one_liner,
        "summary": summary,
        "drivers": drivers[:4],
        "confidence": "medium" if len(drivers) >= 2 else "low",
        "source": "quote+technicals+pattern+bullbrain",
        "price": price,
        "change": change,
        "changePct": change_pct,
        "updated_at": utc_now_iso(),
        "schema_version": "v1",
    }