from typing import Any, Dict, List, Optional
from backend.firestore_utils import utc_now_iso


CATALYST_KEYWORDS = {
    "price_target": [
        "price target", "raises target", "raised target", "target to",
        "street high", "analyst"
    ],
    "ai": [
        "ai", "artificial intelligence", "chip", "chips", "gpu",
        "data center", "infrastructure", "grok"
    ],
    "earnings": [
        "earnings", "revenue", "profit", "eps", "beats", "misses",
        "guidance", "outlook", "quarter"
    ],
    "deal": [
        "partnership", "deal", "investment", "contract", "agreement",
        "acquisition", "merger"
    ],
    "delivery": [
        "deliveries", "delivery", "shipments", "production", "sales"
    ],
    "regulatory": [
        "approval", "regulatory", "sec", "doj", "ftc", "export",
        "tariff", "probe", "investigation"
    ],
    "product": [
        "launch", "update", "software", "model", "iphone", "wwdc",
        "product", "release"
    ],
}


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


def _clean_text(v: Any, limit: int = 220) -> str:
    s = str(v or "").replace("\n", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s[:limit]


def _detect_catalyst_type(text: str) -> Optional[str]:
    t = text.lower()
    for ctype, words in CATALYST_KEYWORDS.items():
        if any(w in t for w in words):
            return ctype
    return None


def _pick_best_catalyst(news_items: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not isinstance(news_items, list):
        return None

    best = None
    best_score = 0

    for item in news_items:
        headline = _clean_text(
            item.get("headline") or item.get("title") or "",
            240,
        )
        summary = _clean_text(item.get("summary") or "", 260)
        source = item.get("source") or "News"
        url = item.get("url") or item.get("link")

        if not headline:
            continue

        combined = f"{headline} {summary}"
        ctype = _detect_catalyst_type(combined)

        if not ctype:
            continue

        score = 1

        # prefer stronger catalyst types
        if ctype in {"price_target", "earnings", "deal", "ai"}:
            score += 3
        elif ctype in {"delivery", "product", "regulatory"}:
            score += 2

        # prefer headline match over summary-only match
        if _detect_catalyst_type(headline):
            score += 2

        # prefer known financial sources
        if source in {"CNBC", "Reuters", "Bloomberg", "MarketWatch", "Yahoo", "Benzinga", "SeekingAlpha", "Zacks"}:
            score += 1

        if score > best_score:
            best_score = score
            best = {
                "type": ctype,
                "headline": headline,
                "summary": summary,
                "source": source,
                "url": url,
                "confidence": "high" if score >= 6 else "medium",
            }

    return best


def _human_catalyst_phrase(catalyst: Dict[str, Any]) -> str:
    headline = catalyst.get("headline") or ""
    ctype = catalyst.get("type")

    if ctype == "price_target":
        return f"after analyst news: {headline}"
    if ctype == "ai":
        return f"as AI-related headlines supported sentiment: {headline}"
    if ctype == "earnings":
        return f"after earnings-related news: {headline}"
    if ctype == "deal":
        return f"after deal or partnership news: {headline}"
    if ctype == "delivery":
        return f"after delivery, shipment, or sales-related news: {headline}"
    if ctype == "regulatory":
        return f"as regulatory or policy news influenced sentiment: {headline}"
    if ctype == "product":
        return f"after product or software-related news: {headline}"

    return f"after company-specific news: {headline}"


def _today_tone(change_pct: float | None) -> str:
    if change_pct is None:
        return "Mixed"
    if change_pct >= 1:
        return "Bullish"
    if change_pct <= -1:
        return "Bearish"
    if change_pct > 0:
        return "Slightly Bullish"
    if change_pct < 0:
        return "Slightly Bearish"
    return "Mixed"


def build_market_awareness(
    symbol: str,
    company_name: str | None = None,
    quote: Dict[str, Any] | None = None,
    technical: Dict[str, Any] | None = None,
    pattern: Dict[str, Any] | None = None,
    bullbrain: Dict[str, Any] | None = None,
    decision: Dict[str, Any] | None = None,
    news_items: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    symbol = (symbol or "").upper().strip()
    company = company_name or symbol

    quote = quote or {}
    technical = technical or {}
    pattern = pattern or {}
    bullbrain = bullbrain or {}
    decision = decision or {}
    news_items = news_items or []

    change_pct = _num(quote.get("changePct"))
    change = _num(quote.get("change"))
    price = _num(quote.get("price"))

    tone = _today_tone(change_pct)

    pattern_name = (
        pattern.get("pattern")
        or pattern.get("patternLabel")
        or pattern.get("name")
    )

    signal = (
        decision.get("finalSignal")
        or bullbrain.get("signal")
        or "HOLD"
    )

    confidence = bullbrain.get("confidence")

    catalyst = _pick_best_catalyst(news_items)

    drivers: List[str] = []

    if catalyst:
        drivers.append(
            f"Catalyst: {catalyst.get('headline')} ({catalyst.get('source')})."
        )

    if change_pct is not None:
        move_word = "gained" if change_pct >= 0 else "declined"
        dollar_part = f" (${abs(change):.2f})" if change is not None else ""
        drivers.append(
            f"Price action: {symbol} {move_word} {_fmt_pct(change_pct)}{dollar_part} today."
        )

    if pattern_name:
        drivers.append(f"Pattern context: {pattern_name} is active.")

    if signal:
        conf_part = ""
        if isinstance(confidence, (int, float)):
            conf_part = f" with {round(float(confidence))}% confidence"
        drivers.append(
            f"AI context: BullBrain remains {signal}{conf_part} on forward edge."
        )

    # ---------------------------------------------------------
    # Catalyst-aware one-liner + summary
    # ---------------------------------------------------------
    if catalyst and change_pct is not None:
        catalyst_phrase = _human_catalyst_phrase(catalyst)

        if change_pct >= 0:
            one_liner = (
                f"{symbol} is rising today, up {_fmt_pct(change_pct)}, "
                f"{catalyst_phrase}."
            )
        else:
            one_liner = (
                f"{symbol} is pulling back today, down {_fmt_pct(change_pct)}, "
                f"{catalyst_phrase}."
            )

        summary = (
            f"{company} is showing a {tone.lower()} tone today. "
            f"The move appears linked to {catalyst_phrase}. "
            f"BullBrain remains {signal}, so this separates today's market reaction "
            f"from the model's forward-looking rating."
        )

        awareness_source = "quote+company_news+technicals+pattern+bullbrain"
        awareness_confidence = catalyst.get("confidence") or "medium"

    else:
        # Safe fallback when no clear news catalyst exists
        if change_pct is None:
            one_liner = f"{symbol} has limited fresh movement data available right now."
        elif change_pct >= 1:
            one_liner = (
                f"{symbol} is rising today, up {_fmt_pct(change_pct)}, "
                f"with price action showing stronger buyer interest."
            )
        elif change_pct <= -1:
            one_liner = (
                f"{symbol} is pulling back today, down {_fmt_pct(change_pct)}, "
                f"with sellers controlling the latest move."
            )
        elif change_pct > 0:
            one_liner = (
                f"{symbol} is slightly higher today, up {_fmt_pct(change_pct)}, "
                f"but no clear company-specific catalyst was detected."
            )
        elif change_pct < 0:
            one_liner = (
                f"{symbol} is slightly lower today, down {_fmt_pct(change_pct)}, "
                f"but no clear company-specific catalyst was detected."
            )
        else:
            one_liner = f"{symbol} is mostly flat today, with no major price move yet."

        if pattern_name:
            summary = (
                f"{company} is showing a {tone.lower()} tone today. "
                f"No clear company-specific news catalyst was detected from available sources, "
                f"so the move is currently explained by price action and the active {pattern_name} pattern. "
                f"BullBrain remains {signal} on forward edge."
            )
        else:
            summary = (
                f"{company} is showing a {tone.lower()} tone today. "
                f"No clear company-specific news catalyst was detected from available sources, "
                f"so the move is currently explained by price action and technical conditions. "
                f"BullBrain remains {signal} on forward edge."
            )

        awareness_source = "quote+technicals+pattern+bullbrain"
        awareness_confidence = "medium" if len(drivers) >= 2 else "low"

    return {
        "title": f"Why {symbol} is moving today",
        "todayTone": tone,
        "oneLiner": one_liner,
        "summary": summary,
        "drivers": drivers[:4],
        "catalyst": catalyst,
        "confidence": awareness_confidence,
        "source": awareness_source,
        "price": price,
        "change": change,
        "changePct": change_pct,
        "updated_at": utc_now_iso(),
        "schema_version": "v2",
    }