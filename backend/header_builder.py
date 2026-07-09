# backend/header_builder.py

from typing import Dict, Any


def build_stock_header(stock: Dict[str, Any]) -> Dict[str, Any]:
    quote = stock.get("quote") or {}
    bull = stock.get("bullbrain") or {}
    decision = stock.get("decision") or {}
    technical = stock.get("technical") or {}
    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}
    profile = stock.get("profile") or {}
    logo_url = profile.get("logoUrl") or stock.get("logoUrl")
    days5 = (history.get("forwardReturns") or {}).get("days5") or {}
    display_intelligence = stock.get("displayIntelligence") or {}

    # displayIntelligence (System B) is the single source of truth for the
    # user-facing signal everywhere else in the app; mirror that here instead
    # of the raw model call, so the header can't disagree with the rest of
    # the Stock Detail payload for the same stock at the same moment.
    raw_signal = decision.get("finalSignal") or bull.get("signal")
    final_signal = display_intelligence.get("signal") or raw_signal
    final_label = display_intelligence.get("label")

    return {
        # -------------------------------------------------
        # Identity
        # -------------------------------------------------
        "symbol": stock.get("symbol"),
        "companyName": stock.get("company_name"),
        "logoUrl": logo_url,
        # -------------------------------------------------
        # Quote (authoritative – Firestore only)
        # -------------------------------------------------
        "quote": {
            "price": quote.get("price"),
            "change": quote.get("change"),
            "changePct": quote.get("changePct"),

            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "prevClose": quote.get("prevClose"),

            "updated_at": quote.get("updated_at"),
            "source": quote.get("source"),
        },

        # -------------------------------------------------
        # Signal
        # -------------------------------------------------
        "signal": {
            "final": final_signal,
            "label": final_label,
            "confidence": bull.get("confidence"),
        },

        # -------------------------------------------------
        # Badges (small, composable) — human-readable text, so use the
        # display label here rather than the raw enum/signal value.
        # -------------------------------------------------
        "badges": [
            final_label or final_signal,
            (technical.get("trend") or {}).get("label"),
            pattern.get("pattern") or pattern.get("patternLabel"),
        ],

        # -------------------------------------------------
        # Pattern Chip (for header row)
        # -------------------------------------------------
        "pattern": {
            "name": pattern.get("pattern") or pattern.get("patternLabel"),
            "bias": pattern.get("bias") or pattern.get("patternBias"),
            "winRate5d": days5.get("winRate"),
        },
    }
