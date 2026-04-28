# backend/astra_context_builder.py

from typing import Dict, Any, List
from backend.stock_repo import get_stock


def _safe_round(v, digits=2):
    return round(v, digits) if isinstance(v, (int, float)) else v


def build_symbol_context(symbol: str, portfolio_position: Dict[str, Any] | None = None) -> Dict[str, Any]:
    stock = get_stock(symbol) or {}

    quote = stock.get("quote") or {}
    bull = stock.get("bullbrain") or {}
    raw = bull.get("raw") or {}
    pattern = stock.get("pattern") or {}
    history = stock.get("patternHistory") or {}
    technical = stock.get("technical") or {}
    features = stock.get("features_meta") or {}
    indicators = stock.get("indicator_states") or {}
    decision = stock.get("decision") or {}
    narratives = stock.get("narratives") or {}

    days5 = ((history.get("forwardReturns") or {}).get("days5") or {})

    return {
        "symbol": symbol,
        "companyName": stock.get("company_name") or symbol,

        "portfolio": portfolio_position or {},

        "quote": {
            "price": quote.get("price"),
            "change": quote.get("change"),
            "changePct": quote.get("changePct"),
            "updated_at": quote.get("updated_at"),
        },

        "aiSignal": {
            "signal": decision.get("final") or decision.get("finalSignal") or bull.get("signal") or "HOLD",
            "confidence": decision.get("confidence") or bull.get("confidence"),
            "prob_up": raw.get("prob_up"),
            "prob_down": raw.get("prob_down"),
        },

        "pattern": {
            "name": pattern.get("pattern") or pattern.get("patternLabel"),
            "bias": pattern.get("bias") or pattern.get("patternBias"),
            "headline": pattern.get("headline"),
            "winRate5d": days5.get("winRate"),
            "avgReturn5d": days5.get("avg"),
            "sampleCount": days5.get("count"),
            "best5d": days5.get("best"),
            "worst5d": days5.get("worst"),
        },

        "technical": {
            "trend": technical.get("trend") or {},
            "momentum": technical.get("momentum") or {},
            "volume": technical.get("volume") or {},
            "volatility": technical.get("volatility") or {},
            "rsi14": features.get("rsi14"),
            "macd": features.get("macd"),
            "macd_signal": features.get("macd_signal"),
            "atr14": features.get("atr14"),
            "volume_vs_ma20_pct": features.get("volume_vs_ma20_pct"),
            "price_vs_sma20_pct": features.get("price_vs_sma20_pct"),
        },

        "indicatorStates": indicators,

        "narratives": {
            "summary": narratives.get("summary"),
            "tradeIdea": narratives.get("tradeIdea"),
            "probability": narratives.get("probability"),
            "sections": narratives.get("sections") or {},
        },

        "computed_at": stock.get("computed_at"),
    }


def build_astra_context(req, intent_payload: Dict[str, Any]) -> Dict[str, Any]:
    positions = req.positions or []
        # Stock Detail mode: no portfolio required
    # Stock Detail mode: no portfolio required
    if getattr(req, "contextType", None) == "stock_detail":
        primary_sym = (getattr(req, "symbol", "") or "").upper()

        requested_symbols = intent_payload.get("symbols") or []

        # Always include current Stock Detail symbol first
        symbols_to_load = []
        if primary_sym:
            symbols_to_load.append(primary_sym)

        # Add resolved / mentioned symbols like TSLA, NVDA, MSFT
        for s in requested_symbols:
            s = (s or "").upper()
            if s and s not in symbols_to_load:
                symbols_to_load.append(s)

        # Safety fallback
        if not symbols_to_load and primary_sym:
            symbols_to_load = [primary_sym]

        return {
            "intent": {
                **intent_payload,
                "intent": intent_payload.get("intent") or "stock_explain",
                "symbols": symbols_to_load,
            },
            "contextType": "stock_detail",
            "portfolio": {
                "total_value": None,
                "total_gain": None,
                "today_gain": None,
                "position_count": 0,
                "top_holding": None,
                "best_position": None,
                "worst_position": None,
            },
            "symbols": [
                build_symbol_context(sym, None)
                for sym in symbols_to_load
            ],
        }
    position_map = {
        p.symbol.upper(): {
            "shares": p.shares,
            "avg_cost": p.avg_cost,
            "price": p.price,
            "gain": _safe_round(p.gain),
            "gain_pct": _safe_round(p.gain_pct),
            "allocation_pct": _safe_round(p.allocation_pct),
            "today": _safe_round(p.today),
        }
        for p in positions
    }

    available_symbols = list(position_map.keys())

    target_symbols = intent_payload.get("symbols") or []

    if not target_symbols:
        if intent_payload.get("intent") in ["portfolio_top_holdings", "compare_symbols"]:
            target_symbols = sorted(
                available_symbols,
                key=lambda s: position_map[s].get("allocation_pct") or 0,
                reverse=True,
            )[:3]
        else:
            target_symbols = available_symbols[:10]

    symbol_contexts = [
        build_symbol_context(sym, position_map.get(sym))
        for sym in target_symbols
    ]

    total_value = req.total_value or 0
    total_gain = req.total_gain or 0
    today_gain = req.today_gain or 0

    top_holding = None
    worst_position = None
    best_position = None

    if position_map:
        values = list(position_map.items())

        top_holding = max(values, key=lambda x: x[1].get("allocation_pct") or 0)
        worst_position = min(values, key=lambda x: x[1].get("gain") or 0)
        best_position = max(values, key=lambda x: x[1].get("gain") or 0)

    return {
        "intent": intent_payload,
        "portfolio": {
            "total_value": _safe_round(total_value),
            "total_gain": _safe_round(total_gain),
            "today_gain": _safe_round(today_gain),
            "position_count": len(positions),
            "top_holding": {
                "symbol": top_holding[0],
                **top_holding[1],
            } if top_holding else None,
            "best_position": {
                "symbol": best_position[0],
                **best_position[1],
            } if best_position else None,
            "worst_position": {
                "symbol": worst_position[0],
                **worst_position[1],
            } if worst_position else None,
        },
        "symbols": symbol_contexts,
    }