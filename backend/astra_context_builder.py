# backend/astra_context_builder.py

from typing import Dict, Any, List
from backend.stock_repo import get_stock
from backend.news.market_news_repo import get_market_news
from backend.news.market_highlights import build_market_highlights
from backend.pick_tracking import get_checked_picks_for_report
from backend.pick_accuracy_report import dedupe_checked_picks, build_accuracy_report

from firebase_admin import firestore
db = firestore.client()
def _safe_round(v, digits=2):
    return round(v, digits) if isinstance(v, (int, float)) else v

def build_market_context_from_firestore() -> Dict[str, Any]:
    """
    Market Pulse context for Astra.
    Uses same backend sources as MarketScreen:
    - homescreen_snapshot / homescreen-context
    - market movers source
    - market news source
    """

    # 1) Homescreen snapshot: market overview + carousel
    snap_doc = (
        db.collection("bullsignals_ai")
          .document("homescreen_snapshot")
          .get()
    )

    snap = snap_doc.to_dict() if snap_doc.exists else {}
    market = snap.get("market_overview") or {}
    carousel = snap.get("carousel") or []

    def find_card(card_id: str) -> Dict[str, Any]:
        for c in carousel:
            if c.get("id") == card_id:
                return c or {}
        return {}

    us_market = find_card("us_market")
    crypto = find_card("crypto")
    commodities = find_card("commodities")
    sentiment = find_card("sentiment")
    sectors = find_card("sectors")

    # 2) Market movers
    movers = {"gainers": [], "losers": []}
    try:
        movers_doc = (
            db.collection("bullsignals_ai")
              .document("market_movers")
              .get()
        )
        movers_data = movers_doc.to_dict() if movers_doc.exists else {}

        raw_movers = movers_data.get("movers") or []
        gainers = []
        losers = []

        for m in raw_movers:
            item = {
                "symbol": m.get("symbol"),
                "company": m.get("company") or m.get("symbol"),
                "direction": m.get("direction"),
                "quote": m.get("quote") or {},
                "trend": m.get("trend") or {},
                "pattern": m.get("pattern") or {},
                "oneLiner": m.get("oneLiner"),
            }

            if m.get("direction") == "up":
                gainers.append(item)
            elif m.get("direction") == "down":
                losers.append(item)

        movers = {
            "gainers": gainers[:10],
            "losers": losers[:10],
            "updated_at": movers_data.get("updated_at"),
            "note": "Market movers are based on AlphaWise internal tracked universe, not official full-market movers.",
        }
    except Exception as e:
        movers = {
            "gainers": [],
            "losers": [],
            "error": str(e),
            "note": "Market movers were unavailable.",
        }

    # 3) Market news — use same source as /market-news endpoint
    news = []
    news_summary = {"headline": None, "highlights": []}

    try:
        news_data = get_market_news() or {}
        news = news_data.get("items", []) or []
        news_summary = build_market_highlights(news)
    except Exception as e:
        news = []
        news_summary = {
            "headline": None,
            "highlights": [],
            "error": str(e),
        }

    return {
        "scope": "Market Pulse uses SPY, QQQ, crypto, commodities ETFs, Fear & Greed, internal movers, and market news.",
        "marketOverview": {
            "marketStatus": market.get("marketStatus"),
            "marketMood": market.get("marketMood"),
            "risk_level": market.get("risk_level"),
            "fearGreed": market.get("fearGreed"),
            "updated_at": market.get("updated_at") or snap.get("updated_at"),
        },
        "usMarket": us_market.get("items") or [],
        "crypto": crypto.get("items") or [],
        "commodities": commodities.get("items") or [],
        "sentiment": sentiment.get("items") or [],
        "sectors": sectors.get("items") or [],
        "internalMovers": movers,
        "marketNews": news[:12],
        "marketNewsSummary": news_summary,
        "updated_at": snap.get("updated_at"),
    }

def _find_current_alpha_watch_item(symbol: str) -> Dict[str, Any] | None:
    """
    Alpha Watch's OWN selection reasoning for a symbol -- genuinely
    different from BullBrain's raw signal/pattern below (see
    bullbrain_area_d_momentum_override_audit memory: the two systems are
    substantially decoupled). Only checks the live /bullsignals_ai/
    alpha_watch doc (today's current picks) -- a symbol that qualified
    previously but has since rotated out won't be found here. The
    historical case ("was a pick") would need a pick_tracking query by
    symbol, deliberately out of scope here (needs its own care around
    Firestore indexing given how many rows a single symbol can have).
    """
    doc = db.collection("bullsignals_ai").document("alpha_watch").get()
    if not doc.exists:
        return None

    items = (doc.to_dict() or {}).get("items") or []
    symbol = symbol.upper()

    for item in items:
        if str(item.get("symbol") or "").upper() == symbol:
            return item

    return None


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
    display_intelligence = stock.get("displayIntelligence") or {}

    days5 = ((history.get("forwardReturns") or {}).get("days5") or {})

    alpha_watch_item = _find_current_alpha_watch_item(symbol)
    alpha_watch_reasoning = (
        {
            "isCurrentPick": True,
            "setupLabel": alpha_watch_item.get("setupLabel"),
            "reason": alpha_watch_item.get("reason"),
            "whyNow": alpha_watch_item.get("whyNow"),
            "factorScores": alpha_watch_item.get("factorScores"),
            "score": alpha_watch_item.get("score"),
            "marketRegime": alpha_watch_item.get("marketRegime"),
            "riskLevel": alpha_watch_item.get("riskLevel"),
            "riskFlags": alpha_watch_item.get("riskFlags"),
        }
        if alpha_watch_item else None
    )

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
            # displayIntelligence (System B) is the app-wide source of truth
            # for the user-facing signal; fall back to the raw model call
            # only when it hasn't been computed for this symbol yet.
            "signal": (
                display_intelligence.get("signal")
                or decision.get("final")
                or decision.get("finalSignal")
                or bull.get("signal")
                or "HOLD"
            ),
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

        # Genuinely separate from aiSignal/pattern above: that's BullBrain's
        # raw model view; this is Alpha Watch's OWN selection reasoning
        # (score_stock() in alpha_watch_logic.py), decoupled from the raw
        # signal per the Area D audit. Honestly None when the symbol isn't
        # a current Alpha Watch pick -- never fabricated.
        "alphaWatchReasoning": alpha_watch_reasoning,

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


def _pick_performance_metric(item: Dict[str, Any]) -> float | None:
    # Two differently-named fields answer the same real question ("how
    # has this performed since picked"): a resolved pick's return is
    # frozen in checked_return_pct, a still-open pick's is live in
    # livePctSinceFirstPick. Unified here so ranking doesn't depend on
    # correctly reconciling both field names per item.
    if item.get("status") == "checked":
        return item.get("checked_return_pct")
    if item.get("status") == "tracking":
        return item.get("livePctSinceFirstPick")
    return None


def _compute_picks_rankings(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pre-computed server-side, not left for the LLM to scan-and-rank a
    (potentially 100+-item) JSON array itself -- deterministic, always
    correct, same discipline as every other cross-item computation in
    this project.
    """
    def _summary(item: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not item:
            return None
        return {
            "symbol": item.get("symbol"),
            "status": item.get("status"),
            "tier": item.get("tier"),
            "returnPct": _pick_performance_metric(item),
            "pick_count": item.get("pick_count"),
        }

    ranked = [
        (item, m) for item in items
        if isinstance((m := _pick_performance_metric(item)), (int, float))
    ]

    best = max(ranked, key=lambda pair: pair[1])[0] if ranked else None
    worst = min(ranked, key=lambda pair: pair[1])[0] if ranked else None
    most_picked = max(items, key=lambda i: i.get("pick_count") or 0) if items else None

    return {
        "bestPerformer": _summary(best),
        "worstPerformer": _summary(worst),
        "mostPicked": _summary(most_picked),
    }


def build_astra_context(req, intent_payload: Dict[str, Any]) -> Dict[str, Any]:
    positions = req.positions or []

    # Accuracy/track-record questions are answered the same way regardless
    # of which screen the user is on -- checked first, before any
    # contextType branch, so "how accurate are you" asked from a stock
    # detail screen still gets the real track record, not that screen's
    # unrelated symbol context.
    if intent_payload.get("intent") == "accuracy_track_record":
        raw_docs = get_checked_picks_for_report(db)
        deduped = dedupe_checked_picks(raw_docs)
        report = build_accuracy_report(deduped)

        return {
            "intent": intent_payload,
            "contextType": "accuracy_track_record",
            "portfolio": {
                "total_value": None,
                "total_gain": None,
                "today_gain": None,
                "position_count": 0,
                "top_holding": None,
                "best_position": None,
                "worst_position": None,
            },
            "symbols": [],
            "accuracyReport": report,
        }

    # Alphaclara Picks LIST screen: bundles BOTH the accuracy report and
    # the real tiered picks list, so either an aggregate question ("how
    # accurate", "best performer") or a specific-pick question ("why is
    # HIMS still tracking", "PANW's return since picked") can be answered
    # from the same context. Reuses /alphaclara-tracking's existing dedup/
    # tiering logic directly rather than rebuilding it -- imported lazily
    # (not at module top) so that scripts which only need
    # astra_context_builder.py for unrelated reasons aren't forced through
    # main.py's full startup (BullBrain model load, etc.). Safe here
    # because this branch only ever runs during a live /astra-chat
    # request, by which point main.py is already fully loaded.
    if getattr(req, "contextType", None) == "alphaclara_picks_overview":
        from main import get_alphaclara_tracking

        raw_docs = get_checked_picks_for_report(db)
        deduped = dedupe_checked_picks(raw_docs)
        accuracy_report = build_accuracy_report(deduped)

        # Fetched once, full and uncapped within the 30-day window (not
        # limit=30) -- confirmed on real data that a specifically-
        # mentioned symbol (PANW, HIMS) can sit well outside even a
        # 30-item recency-sorted slice despite being a real, resolvable
        # pick, so a mentioned-symbol lookup must never depend on the
        # truncated aggregate list below.
        tracking = get_alphaclara_tracking(limit=None, window_days=30)
        all_items = tracking.get("items") or []
        tracking_by_symbol = {
            str(item.get("symbol") or "").upper(): item
            for item in all_items
        }

        # Same pattern as stock_detail: any symbol mentioned in the
        # question still gets its own real per-symbol lookup, enriched
        # with its pick-tracking record (return/pick_count) from the full,
        # untruncated set -- distinct from alphaWatchReasoning (why it was
        # EVER picked); this is how it has actually PERFORMED since picked.
        requested_symbols = intent_payload.get("symbols") or []
        symbol_contexts = []
        for sym in requested_symbols:
            ctx = build_symbol_context(sym, None)
            ctx["pickTrackingSummary"] = tracking_by_symbol.get(sym.upper())
            symbol_contexts.append(ctx)

        return {
            "intent": intent_payload,
            "contextType": "alphaclara_picks_overview",
            "portfolio": {
                "total_value": None,
                "total_gain": None,
                "today_gain": None,
                "position_count": 0,
                "top_holding": None,
                "best_position": None,
                "worst_position": None,
            },
            "symbols": symbol_contexts,
            "accuracyReport": accuracy_report,
            "picksOverview": {
                # Truncated for prompt size -- rankings below are computed
                # over the full all_items set, not this slice, so ranking
                # accuracy never depends on this truncation.
                "items": all_items[:30],
                "counts": tracking.get("counts") or {},
                "window_days": tracking.get("window_days"),
                "updated_at": tracking.get("updated_at"),
                "rankings": _compute_picks_rankings(all_items),
            },
        }

    # Momentum Movers mode: no portfolio required
    if getattr(req, "contextType", None) == "momentum_movers":
        selected = getattr(req, "selectedMover", None) or {}
        movers = getattr(req, "movers", None) or []
        ai_setups = getattr(req, "aiSetups", None) or []
        pullbacks = getattr(req, "pullbacks", None) or []
        pulse = getattr(req, "pulse", None) or {}

        selected_symbol = str(selected.get("symbol") or "").upper() if isinstance(selected, dict) else ""

        return {
            "intent": {
                **intent_payload,
                "intent": intent_payload.get("intent") or "mover_why",
            },
            "contextType": "momentum_movers",
            "portfolio": {
                "total_value": None,
                "total_gain": None,
                "today_gain": None,
                "position_count": 0,
                "top_holding": None,
                "best_position": None,
                "worst_position": None,
            },
            "momentum": {
                "selectedMover": selected,
                "movers": movers[:12] if isinstance(movers, list) else [],
                "aiSetups": ai_setups[:8] if isinstance(ai_setups, list) else [],
                "pullbacks": pullbacks[:8] if isinstance(pullbacks, list) else [],
                "pulse": pulse,
                "updatedAt": getattr(req, "updatedAt", None),
                "lookbackSnapshots": getattr(req, "lookbackSnapshots", None),
            },
            "symbols": [
                build_symbol_context(selected_symbol, None)
            ] if selected_symbol else [],
        }
        # Market Pulse mode: no portfolio required
    if getattr(req, "contextType", None) == "market":
        return {
            "intent": {
                **intent_payload,
                "intent": intent_payload.get("intent") or "market_pulse",
            },
            "contextType": "market",
            "portfolio": {
                "total_value": None,
                "total_gain": None,
                "today_gain": None,
                "position_count": 0,
                "top_holding": None,
                "best_position": None,
                "worst_position": None,
            },
            "symbols": [],
            "market": build_market_context_from_firestore(),
        }    
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