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


def _normalize_for_app(item: Dict[str, Any]) -> Dict[str, Any]:
    quote = item.get("quote") or {}
    final_score = _score_item(item)

    return {
        "symbol": item.get("symbol"),
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


def build_verified_alpha_payload() -> Dict[str, Any]:
    """
    Builds app-facing verified alpha + movers payload.

    Safe rule:
    - If Grok/Sheets fail or produce no verified data, caller can fall back to internal docs.
    """

    db = _db()
    snap = db.collection(COL_ROOT).document("grok_market_memory").get()

    if not snap.exists:
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

    memory = snap.to_dict() or {}
    enriched = memory.get("premarket_latest") or {}

    gainers = [_normalize_for_app(x) for x in enriched.get("premarket_gainers", [])]
    losers = [_normalize_for_app(x) for x in enriched.get("premarket_losers", [])]
    opportunities = [_normalize_for_app(x) for x in enriched.get("alpha_opportunities", [])]

    gainers.sort(key=lambda x: float(x.get("changePct") or 0), reverse=True)
    losers.sort(key=lambda x: float(x.get("changePct") or 0))
    opportunities.sort(key=lambda x: float(x.get("finalAlphaScore") or 0), reverse=True)

    return {
        "source": "grok_google_sheet_verified",
        "updated_at": _now_iso(),
        "fallback_required": len(opportunities) == 0 and len(gainers) == 0 and len(losers) == 0,
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