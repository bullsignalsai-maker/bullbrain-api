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


def _company_tokens(company_name: str | None) -> List[str]:
    raw = (company_name or "").lower()
    for ch in [",", ".", "(", ")", "-", "&"]:
        raw = raw.replace(ch, " ")

    ignore = {
        "inc", "corp", "corporation", "company", "class",
        "holdings", "holding", "limited", "ltd", "plc",
        "common", "stock", "group"
    }

    return [
        t.strip()
        for t in raw.split()
        if len(t.strip()) > 3 and t.strip() not in ignore
    ]


def _headline_is_primary_for_symbol(
    headline: str,
    symbol: str,
    company_name: str | None,
) -> bool:
    h = f" {headline.lower()} "
    sym = (symbol or "").lower().strip()

    if sym and f" {sym} " in h:
        return True

    tokens = _company_tokens(company_name)
    return any(f" {tok} " in h for tok in tokens) 


def _pick_best_catalyst(
    news_items: List[Dict[str, Any]],
    symbol: str,
    company_name: str | None = None,
) -> Dict[str, Any] | None:
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

        bad_headline_phrases = (
            "?", "which ", "what ", "why ", "how ", "should ", "could ", "would ",
            "better after", "looks better", "what it means",
        )

        headline_l = headline.lower().strip()
        if any(p in headline_l for p in bad_headline_phrases):
            continue    

        comparison_words = (" vs. ", " versus ", " compared with ", " compared to ")
        if any(w in headline_l for w in comparison_words):
            continue

        if not _headline_is_primary_for_symbol(headline, symbol, company_name):
            continue

        combined = f"{headline} {summary}"
        ctype = _detect_catalyst_type(combined)

        if not ctype:
            continue

        score = 1
        if ctype in {"price_target", "earnings", "deal", "ai"}:
            score += 3
        elif ctype in {"delivery", "product", "regulatory"}:
            score += 2
        if _detect_catalyst_type(headline):
            score += 2

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


# ====================== NEW HELPERS ======================

def _shorten(s: str, limit: int = 148) -> str:
    s = _clean_text(s)
    if len(s) <= limit:
        return s
    return s[:limit - 1].rstrip(" ,.;:-") + "…"


def _confidence_band(confidence: Any) -> str:
    n = _num(confidence)
    if n is None:
        return "medium"
    if n >= 78:
        return "very_high"
    if n >= 65:
        return "high"
    if n >= 52:
        return "medium"
    return "low"


def _pattern_phrase(pattern_name: str | None) -> str:
    if not pattern_name:
        return ""
    p = str(pattern_name).replace("_", " ").title()
    if any(x in p for x in ["Trend", "Acceleration", "Breakout"]):
        return f"with {p.lower()} still supporting the setup"
    if any(x in p for x in ["Hammer", "Reversal"]):
        return f"as a {p.lower()} points to possible stabilization"
    if "Compression" in p or "Inside" in p:
        return f"as compression keeps the next move in focus"
    return f"with the {p} pattern active"


def _catalyst_short(catalyst: Dict[str, Any] | None) -> str:
    if not catalyst:
        return ""
    headline = _clean_text(catalyst.get("headline") or "", 88)
    ctype = catalyst.get("type")

    phrases = {
        "price_target": f"analyst conviction improved after {headline}",
        "ai": f"AI-related momentum stayed in focus after {headline}",
        "earnings": f"earnings expectations shifted after {headline}",
        "deal": f"deal momentum improved after {headline}",
        "delivery": f"delivery and demand signals moved sentiment after {headline}",
        "product": f"product momentum drew attention after {headline}",
    }
    return phrases.get(ctype, f"company-specific news shaped sentiment after {headline}")


def _build_smart_one_liner(
    symbol: str,
    change_pct: float | None,
    signal: str,
    confidence: Any,
    catalyst: Dict[str, Any] | None,
    pattern_name: str | None,
) -> str:
    catalyst_text = _catalyst_short(catalyst)
    pattern_text = _pattern_phrase(pattern_name)

    if catalyst_text:
        if change_pct is None or change_pct >= 0:
            return _shorten(f"{symbol} is gaining traction as {catalyst_text}.")
        else:
            return _shorten(f"{symbol} is pulling back despite {catalyst_text}.")

    if change_pct is None:
        return _shorten(f"{symbol} trading with {signal.lower()} model bias.")
    elif change_pct >= 2.0 and _confidence_band(confidence) in ("very_high", "high"):
        return _shorten(f"{symbol} showing strong momentum, up {_fmt_pct(change_pct)}.")
    elif change_pct >= 0.8:
        return _shorten(f"{symbol} is higher with improving price action.")
    elif change_pct <= -2.0:
        return _shorten(f"{symbol} is under pressure, down {_fmt_pct(change_pct)}.")
    elif pattern_text:
        return _shorten(f"{symbol} consolidating {pattern_text}.")
    else:
        return _shorten(f"{symbol} showing a {signal.lower()} setup in the current session.")


def _build_smart_summary(
    company: str,
    tone: str,
    signal: str,
    confidence: Any,
    catalyst: Dict[str, Any] | None,
    pattern_name: str | None,
) -> str:
    conf_text = f" with {round(float(confidence))}% confidence" if isinstance(confidence, (int, float)) else ""

    if catalyst:
        cat_text = _catalyst_short(catalyst)
        return f"{company} is showing a {tone.lower()} tone today as {cat_text}. BullBrain remains {signal}{conf_text}."

    if pattern_name:
        return f"{company} is showing a {tone.lower()} tone with active {pattern_name.lower()} pattern. BullBrain remains {signal}{conf_text}."

    return f"{company} is showing a {tone.lower()} tone today. BullBrain remains {signal}{conf_text} based on price action and technical conditions."


# ====================== MAIN FUNCTION ======================

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

    catalyst = _pick_best_catalyst(news_items, symbol, company)

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

    # ====================== NEW LOGIC ======================
    one_liner = _build_smart_one_liner(
        symbol=symbol,
        change_pct=change_pct,
        signal=signal,
        confidence=confidence,
        catalyst=catalyst,
        pattern_name=pattern_name,
    )

    summary = _build_smart_summary(
        company=company,
        tone=tone,
        signal=signal,
        confidence=confidence,
        catalyst=catalyst,
        pattern_name=pattern_name,
    )

    awareness_source = (
        "quote+company_news+technicals+pattern+bullbrain"
        if catalyst
        else "quote+technicals+pattern+bullbrain"
    )

    awareness_confidence = (
        catalyst.get("confidence")
        if catalyst
        else ("medium" if len(drivers) >= 2 else "low")
    )

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
        "schema_version": "v3",
    }