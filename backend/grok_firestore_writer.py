# backend/grok_firestore_writer.py

from typing import Dict, List, Any
import datetime

from backend.quote_repo import _db
from backend.grok_candidate_builder import build_grok_candidates


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _light_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": item.get("symbol"),
        "source": item.get("source"),
        "market_day": item.get("market_day"),
        "session_type": item.get("session_type"),
        "sector": item.get("sector"),
        "primary_catalysts": item.get("primary_catalysts"),
        "reason": item.get("reason"),
        "mover_quality": item.get("mover_quality"),
        "risk_level": item.get("risk_level"),
        "generated_at": item.get("generated_at"),
    }


def save_grok_market_memory() -> Dict[str, Any]:
    """
    Lightweight spreadsheet memory only.
    No quote enrichment.
    No verified-alpha write.
    No UI pipeline.
    Cron uses this only as discovery seed metadata.
    """
    now = _now_iso()

    try:
        candidates = build_grok_candidates()
    except Exception as e:
        print(f"[grok-writer] spreadsheet candidate load failed: {e}", flush=True)
        candidates = {
            "premarket_gainers": [],
            "premarket_losers": [],
            "alpha_opportunities": [],
        }

    payload = {
        "source": "google_sheet_lightweight_symbols",
        "updated_at": now,
        "premarket_gainers": [_light_item(x) for x in candidates.get("premarket_gainers", [])],
        "premarket_losers": [_light_item(x) for x in candidates.get("premarket_losers", [])],
        "alpha_opportunities": [_light_item(x) for x in candidates.get("alpha_opportunities", [])],
        "schema_version": "grok_lightweight_v1",
    }

    total = (
        len(payload["premarket_gainers"])
        + len(payload["premarket_losers"])
        + len(payload["alpha_opportunities"])
    )

    db = _db()
    db.collection("bullsignals_ai").document("grok_market_memory").set(
        {
            "premarket_latest": payload,
            "updated_at": now,
        },
        merge=True,
    )

    print(f"[grok-writer] lightweight spreadsheet symbols saved | total={total}", flush=True)

    return {
        "ok": True,
        "saved": total,
        "source": "google_sheet_lightweight_symbols",
        "updated_at": now,
    }


if __name__ == "__main__":
    print(save_grok_market_memory())