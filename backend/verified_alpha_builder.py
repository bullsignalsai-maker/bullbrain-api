# backend/verified_alpha_builder.py

from typing import Dict, List, Any
import datetime

from backend.quote_repo import _db
from backend.grok_quote_enricher import enrich_candidates


COL_ROOT = "bullsignals_ai"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _score_item(item: Dict[str, Any]) -> float:
    grok_score = float(item.get("grok_alpha_priority_score") or 0)
    quote = item.get("quote") or {}
    change_pct = abs(float(quote.get("changePct") or 0))

    # V1 hybrid score: Grok reasoning + real market movement
    return round((grok_score * 0.75) + (min(change_pct, 10) * 2.5), 2)

def _logo_url_for_symbol(db, symbol: str) -> str | None:
    try:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return None

        snap = (
            db.collection(COL_ROOT)
            .document("stocks")
            .collection("symbols")
            .document(sym)
            .get()
        )

        if snap.exists:
            stock = snap.to_dict() or {}
            profile = stock.get("profile") or {}
            return profile.get("logoUrl")
    except Exception:
        pass

    return None

def _normalize_for_app(db, item: Dict[str, Any]) -> Dict[str, Any]:
    quote = item.get("quote") or {}
    final_score = _score_item(item)
    symbol = item.get("symbol")
    logo_url = _logo_url_for_symbol(db, symbol)

    return {
        "symbol": item.get("symbol"),
        "logoUrl": logo_url,
        "sector": item.get("sector"),
        "source": item.get("source"),
        "reason": item.get("reason"),
        "primaryCatalysts": item.get("primary_catalysts"),
        "moverQuality": item.get("mover_quality"),
        "riskLevel": item.get("risk_level"),
        "grokAlphaPriorityScore": item.get("grok_alpha_priority_score"),
        "finalAlphaScore": final_score,
        "quote": quote,
        "price": quote.get("price"),
        "change": quote.get("change"),
        "changePct": quote.get("changePct"),
        "quoteVerified": item.get("quote_verified") is True,
        "generatedAt": item.get("generated_at"),
    }

def _empty_payload() -> Dict[str, Any]:
    return {
        "source": "grok_google_sheet_verified",
        "updated_at": _now_iso(),
        "fallback_required": True,
        "counts": {
            "premarket_gainers": 0,
            "premarket_losers": 0,
            "alpha_opportunities": 0,
        },
        "premarket_gainers": [],
        "premarket_losers": [],
        "alpha_opportunities": [],
        "schema_version": "verified_alpha_v1",
    }


def build_verified_alpha_payload() -> Dict[str, Any]:
    db = _db()

    snap = db.collection(COL_ROOT).document("grok_market_memory").get()

    if not snap.exists:
        return _empty_payload()

    memory = snap.to_dict() or {}
    enriched = memory.get("premarket_latest") or {}

    gainers_raw = enriched.get("premarket_gainers", []) or []
    losers_raw = enriched.get("premarket_losers", []) or []
    alpha_raw = enriched.get("alpha_opportunities", []) or []

    gainers = [_normalize_for_app(db, x) for x in gainers_raw]
    losers = [_normalize_for_app(db, x) for x in losers_raw]
    opportunities = [_normalize_for_app(db, x) for x in alpha_raw]

    gainers = [x for x in gainers if x.get("symbol")]
    losers = [x for x in losers if x.get("symbol")]
    opportunities = [x for x in opportunities if x.get("symbol")]

    gainers.sort(key=lambda x: float(x.get("changePct") or 0), reverse=True)
    losers.sort(key=lambda x: float(x.get("changePct") or 0))
    opportunities.sort(
        key=lambda x: float(x.get("finalAlphaScore") or 0),
        reverse=True,
    )

    fallback_required = (
        len(gainers) == 0
        and len(losers) == 0
        and len(opportunities) == 0
    )

    return {
        "source": "grok_google_sheet_verified",
        "updated_at": _now_iso(),
        "session_type": enriched.get("session_type", "PREMARKET"),
        "market_summary": enriched.get("market_summary", {}),
        "fallback_required": fallback_required,
        "counts": {
            "premarket_gainers": len(gainers),
            "premarket_losers": len(losers),
            "alpha_opportunities": len(opportunities),
        },
        "premarket_gainers": gainers[:10],
        "premarket_losers": losers[:10],
        "alpha_opportunities": opportunities[:10],
        "schema_version": "verified_alpha_v1",
    }

def save_verified_alpha_payload() -> Dict[str, Any]:
    payload = build_verified_alpha_payload()

    if payload.get("fallback_required"):
        return {
            "ok": False,
            "saved": 0,
            "fallback_required": True,
            "updated_at": payload["updated_at"],
        }

    db = _db()

    db.collection(COL_ROOT).document("verified_alpha_opportunities").set(
        payload,
        merge=True,
    )

    print(
        "[verified-alpha] saved | "
        f"gainers={payload['counts']['premarket_gainers']} "
        f"losers={payload['counts']['premarket_losers']} "
        f"alpha={payload['counts']['alpha_opportunities']}",
        flush=True,
    )

    return {
        "ok": True,
        "saved": sum(payload["counts"].values()),
        "fallback_required": False,
        "updated_at": payload["updated_at"],
    }


if __name__ == "__main__":
    result = save_verified_alpha_payload()
    print(result)