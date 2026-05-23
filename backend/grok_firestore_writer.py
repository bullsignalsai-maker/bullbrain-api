# backend/grok_firestore_writer.py

from typing import Dict, List, Any
import datetime

from backend.quote_repo import _db
from backend.grok_quote_enricher import enrich_candidates
from backend.daily_alpha_intelligence import save_daily_alpha_session

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _flatten(items: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for section, rows in items.items():
        for row in rows:
            out.append({
                **row,
                "section": section,
            })
    return out

def _detect_alpha_session(alpha_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect market_day + session_type from alpha_opportunities.
    Uses the first valid row because enrich_candidates() already returns latest_rows().
    """

    for item in alpha_items:
        market_day = str(item.get("market_day") or "").strip()
        session_type = str(item.get("session_type") or "").strip().upper()

        if market_day and session_type:
            return {
                "market_day": market_day,
                "session_type": session_type,
            }

    return {
        "market_day": "",
        "session_type": "",
    }

def save_grok_market_memory() -> Dict[str, Any]:
    """
    Enhancement layer only:
    - Tries Grok/Google Sheets candidates
    - Enriches with verified quote data
    - Saves to Firestore if available
    - Does NOT break app if empty/fails
    """

    try:
        enriched = enrich_candidates()
    except Exception as e:
        print(f"[grok-writer] Grok sheet/enrichment failed: {e}", flush=True)
        enriched = {
            "premarket_gainers": [],
            "premarket_losers": [],
            "alpha_opportunities": [],
        }

    flat = _flatten(enriched)
    now = _now_iso()

    if not flat:
        print("[grok-writer] No Grok candidates available. Internal fallback should handle app output.", flush=True)
        return {
            "ok": False,
            "source": "grok_google_sheet",
            "saved": 0,
            "fallback_required": True,
            "updated_at": now,
        }

    payload = {
        "source": "grok_google_sheet",
        "updated_at": now,
        "fallback_required": False,
        "counts": {
            "premarket_gainers": len(enriched["premarket_gainers"]),
            "premarket_losers": len(enriched["premarket_losers"]),
            "alpha_opportunities": len(enriched["alpha_opportunities"]),
            "total": len(flat),
        },
        "premarket_gainers": enriched["premarket_gainers"],
        "premarket_losers": enriched["premarket_losers"],
        "alpha_opportunities": enriched["alpha_opportunities"],
    }

    db = _db()

    db.collection("bullsignals_ai").document("grok_market_memory").set(
        {
            "premarket_latest": payload,
            "updated_at": now,
        },
        merge=True,
    )

    db.collection("bullsignals_ai").document("alpha_opportunities").set(
        {
            "source": "grok_google_sheet_verified",
            "updated_at": now,
            "items": enriched["alpha_opportunities"],
        },
        merge=True,
    )

    # ---------------------------------------------------------
    # Historical Alpha Intelligence Memory
    # Save latest AlphaOpportunities session into:
    # /bullsignals_ai/market_memory/daily_alpha_intelligence/{YYYY-MM-DD}
    # ---------------------------------------------------------
    alpha_history_result = {
        "ok": False,
        "reason": "not_attempted",
        "saved": 0,
    }

    try:
        alpha_items = enriched.get("alpha_opportunities", []) or []
        session_meta = _detect_alpha_session(alpha_items)

        if alpha_items and session_meta.get("market_day") and session_meta.get("session_type"):
            alpha_history_result = save_daily_alpha_session(
                market_day=session_meta["market_day"],
                session_type=session_meta["session_type"],
                items=alpha_items,
                source="grok_google_sheet_verified_auto",
            )

            print(
                "[grok-writer] daily alpha intelligence saved | "
                f"day={session_meta['market_day']} "
                f"session={session_meta['session_type']} "
                f"saved={alpha_history_result.get('saved')}",
                flush=True,
            )
        else:
            alpha_history_result = {
                "ok": False,
                "reason": "missing_alpha_session_meta",
                "saved": 0,
            }

    except Exception as e:
        print(f"[grok-writer] daily alpha intelligence save failed: {e}", flush=True)
        alpha_history_result = {
            "ok": False,
            "reason": str(e),
            "saved": 0,
        }


    print(f"[grok-writer] Saved {len(flat)} verified Grok candidates.", flush=True)

    return {
        "ok": True,
        "source": "grok_google_sheet",
        "saved": len(flat),
        "fallback_required": False,
        "alpha_history": alpha_history_result,
        "updated_at": now,
    }


if __name__ == "__main__":
    result = save_grok_market_memory()
    print(result)