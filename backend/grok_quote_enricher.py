# backend/grok_quote_enricher.py

from typing import Dict, List, Any

from backend.grok_candidate_builder import build_grok_candidates
from quote_provider import fetch_equity_quote


MIN_PRICE = 3.0


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _get_cached_quote(symbol: str, quote_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    symbol = str(symbol or "").upper().strip()

    if not symbol:
        return {}

    if symbol in quote_cache:
        return quote_cache[symbol]

    try:
        quote = fetch_equity_quote(symbol) or {}
    except Exception as e:
        print(f"[grok_quote_enricher] quote fetch failed for {symbol}: {e}", flush=True)
        quote = {}

    quote_cache[symbol] = quote
    return quote


def enrich_symbol(candidate: Dict[str, Any], quote_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    symbol = str(candidate.get("symbol") or "").upper().strip()

    if not symbol:
        return {}

    quote = _get_cached_quote(symbol, quote_cache)

    if not quote:
        print(f"[grok_quote_enricher] no quote returned for {symbol}; keeping unverified", flush=True)
        return {
            **candidate,
            "quote_verified": False,
            "quote": {},
        }
    raw_price = quote.get("price")
    price = _safe_float(raw_price, default=None)
    change = _safe_float(quote.get("change"))
    change_pct = _safe_float(quote.get("changePct"))

    if price is not None and price < MIN_PRICE:
        print(f"[grok_quote_enricher] skipped {symbol}: price {price} < {MIN_PRICE}", flush=True)
        return {}

    return {
        **candidate,
        "quote_verified": True,
        "quote": {
            "price": round(price, 2) if price is not None else None,
            "change": round(change, 2),
            "changePct": round(change_pct, 2),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "prevClose": quote.get("prevClose"),
            "source": quote.get("source"),
        },
    }


def enrich_candidates() -> Dict[str, List[Dict[str, Any]]]:
    candidates = build_grok_candidates()
    quote_cache: Dict[str, Dict[str, Any]] = {}

    result = {
        "premarket_gainers": [],
        "premarket_losers": [],
        "alpha_opportunities": [],
    }

    # IMPORTANT: alpha first, so best opportunities do not suffer from rate limits
    section_order = [
        "alpha_opportunities",
        "premarket_gainers",
        "premarket_losers",
    ]

    for section in section_order:
        for candidate in candidates.get(section, []):
            enriched = enrich_symbol(candidate, quote_cache)
            if enriched:
                result[section].append(enriched)

    return result


if __name__ == "__main__":
    data = enrich_candidates()

    print("Enriched gainers:", len(data["premarket_gainers"]))
    print("Enriched losers:", len(data["premarket_losers"]))
    print("Enriched opportunities:", len(data["alpha_opportunities"]))

    if data["alpha_opportunities"]:
        print("\nSample enriched opportunity:")
        print(data["alpha_opportunities"][0])