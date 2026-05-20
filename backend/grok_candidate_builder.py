# backend/grok_candidate_builder.py

from typing import Dict, List, Any

from backend.market_memory_sheet import get_all_market_memory_candidates


def _clean_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()

    # Fix accidental repeated symbols like AMDAMD, LLYLLY, JPMJPM
    if len(s) % 2 == 0:
        half = len(s) // 2
        if s[:half] == s[half:]:
            s = s[:half]

    return s

def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_candidate(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    symbol = _clean_symbol(row.get("symbol"))

    return {
        "symbol": symbol,
        "source": source,  # premarket_gainer / premarket_loser / alpha_opportunity
        "sector": str(row.get("sector", "")).strip(),
        "mover_quality": str(row.get("mover_quality", "")).strip(),
        "primary_catalysts": str(row.get("primary_catalysts", "")).strip(),
        "reason": str(row.get("reason", "")).strip(),
        "risk_level": str(row.get("risk_level", "")).strip(),
        "grok_alpha_priority_score": _to_int(row.get("alpha_priority_score")),
        "generated_at": str(row.get("generated_at", "")).strip(),
    }


def build_grok_candidates() -> Dict[str, List[Dict[str, Any]]]:
    data = get_all_market_memory_candidates()

    gainers = [
        _normalize_candidate(row, "premarket_gainer")
        for row in data.get("premarket_gainers", [])
        if _clean_symbol(row.get("symbol"))
    ]

    losers = [
        _normalize_candidate(row, "premarket_loser")
        for row in data.get("premarket_losers", [])
        if _clean_symbol(row.get("symbol"))
    ]

    opportunities = [
        _normalize_candidate(row, "alpha_opportunity")
        for row in data.get("alpha_opportunities", [])
        if _clean_symbol(row.get("symbol"))
    ]

    return {
        "premarket_gainers": gainers,
        "premarket_losers": losers,
        "alpha_opportunities": opportunities,
    }


def build_unique_symbol_list() -> List[str]:
    candidates = build_grok_candidates()

    symbols = []
    seen = set()

    for group in [
        candidates["alpha_opportunities"],
        candidates["premarket_gainers"],
        candidates["premarket_losers"],
    ]:
        for item in group:
            symbol = item["symbol"]
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)

    return symbols


if __name__ == "__main__":
    candidates = build_grok_candidates()
    symbols = build_unique_symbol_list()

    print("Premarket gainers:", len(candidates["premarket_gainers"]))
    print("Premarket losers:", len(candidates["premarket_losers"]))
    print("Alpha opportunities:", len(candidates["alpha_opportunities"]))
    print("Unique symbols:", len(symbols))
    print(symbols)