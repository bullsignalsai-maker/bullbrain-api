# backend/header_builder.py

from typing import Dict, Any


def build_stock_header(stock: Dict[str, Any]) -> Dict[str, Any]:
    quote = stock.get("quote") or {}
    bull = stock.get("bullbrain") or {}
    decision = stock.get("decision") or {}
    technical = stock.get("technical") or {}
    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}

    days5 = (history.get("forwardReturns") or {}).get("days5") or {}

    final_signal = decision.get("finalSignal") or bull.get("signal")

    return {
        # -------------------------------------------------
        # Identity
        # -------------------------------------------------
        "symbol": stock.get("symbol"),
        "companyName": stock.get("company_name"),

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
            "confidence": bull.get("confidence"),
        },

        # -------------------------------------------------
        # Badges (small, composable)
        # -------------------------------------------------
        "badges": [
            final_signal,
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
