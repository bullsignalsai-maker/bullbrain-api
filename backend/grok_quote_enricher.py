# backend/grok_quote_enricher.py

from typing import Dict, List, Any

from backend.grok_candidate_builder import build_grok_candidates

# quote_provider.py is in project root, not backend/
from quote_provider import fetch_equity_quote


MIN_PRICE = 3.0


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def enrich_symbol(candidate: Dict[str, Any]) -> Dict[str, Any]:
    symbol = candidate.get("symbol")

    if not symbol:
        return {}

    try:
        quote = fetch_equity_quote(symbol)
    except Exception as e:
        print(f"[grok_quote_enricher] quote fetch failed for {symbol}: {e}")
        return {}

    if not quote:
        return {}

    price = _safe_float(quote.get("price"))
    change = _safe_float(quote.get("change"))
    change_pct = _safe_float(quote.get("changePct"))

    # Filter out penny / invalid stocks for app-facing quality
    if price < MIN_PRICE:
        return {}

    return {
        **candidate,
        "quote_verified": True,
        "quote": {
            "price": round(price, 2),
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

    result = {
        "premarket_gainers": [],
        "premarket_losers": [],
        "alpha_opportunities": [],
    }

    for section in result.keys():
        for candidate in candidates.get(section, []):
            enriched = enrich_symbol(candidate)
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