# backend/daily_alpha_intelligence.py

import datetime
from typing import Dict, List, Any
from collections import Counter

from backend.quote_repo import _db


COL_ROOT = "bullsignals_ai"
MARKET_MEMORY_DOC = "market_memory"
DAILY_ALPHA_COLLECTION = "daily_alpha_intelligence"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_symbol(symbol: Any) -> str:
    s = str(symbol or "").strip().upper()

    if len(s) % 2 == 0:
        half = len(s) // 2
        if s[:half] == s[half:]:
            s = s[:half]

    return s


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_session(value: Any) -> str:
    s = str(value or "").strip().upper()

    if s in {"PRE", "PRE_MARKET", "PREMARKET"}:
        return "PREMARKET"

    if s in {"MID", "MID_DAY", "MIDDAY"}:
        return "MIDDAY"

    if s in {"EOD", "ENDDAY", "END_OF_DAY", "END OF DAY"}:
        return "END_OF_DAY"

    return s or "UNKNOWN"


def normalize_alpha_row(row: Dict[str, Any]) -> Dict[str, Any]:
    symbol = _clean_symbol(row.get("symbol"))

    return {
        "symbol": symbol,
        "session_type": _normalize_session(row.get("session_type")),
        "market_day": str(row.get("market_day") or "").strip(),
        "generated_at": str(row.get("generated_at") or "").strip(),
        "sector": str(row.get("sector") or "").strip(),
        "moverQuality": str(row.get("mover_quality") or row.get("moverQuality") or "").strip(),
        "primaryCatalysts": str(row.get("primary_catalysts") or row.get("primaryCatalysts") or "").strip(),
        "reason": str(row.get("reason") or "").strip(),
        "riskLevel": str(row.get("risk_level") or row.get("riskLevel") or "").strip(),
        "grokAlphaPriorityScore": _to_int(
            row.get("alpha_priority_score")
            or row.get("grok_alpha_priority_score")
            or row.get("grokAlphaPriorityScore")
        ),
        "source": "alpha_opportunity",
        "schema_version": "daily_alpha_intelligence_item_v1",
    }


def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}

    for item in items:
        symbol = _clean_symbol(item.get("symbol"))
        if not symbol:
            continue

        score = _to_int(item.get("grokAlphaPriorityScore"))
        current = best.get(symbol)

        if current is None or score >= _to_int(current.get("grokAlphaPriorityScore")):
            best[symbol] = {
                **item,
                "symbol": symbol,
            }

    out = list(best.values())
    out.sort(key=lambda x: _to_int(x.get("grokAlphaPriorityScore")), reverse=True)
    return out


def save_daily_alpha_session(
    *,
    market_day: str,
    session_type: str,
    items: List[Dict[str, Any]],
    source: str = "grok_google_sheet_alpha_opportunities",
) -> Dict[str, Any]:
    """
    Saves AlphaOpportunities historical memory by date + session.

    Firestore:
    /bullsignals_ai/market_memory/daily_alpha_intelligence/{YYYY-MM-DD}
    """

    market_day = str(market_day or "").strip()
    session_type = _normalize_session(session_type)

    if not market_day:
        return {
            "ok": False,
            "reason": "missing_market_day",
            "saved": 0,
        }

    clean_items = []

    for row in items:
        normalized = normalize_alpha_row(row)

        if not normalized.get("symbol"):
            continue

        normalized["market_day"] = market_day
        normalized["session_type"] = session_type
        clean_items.append(normalized)

    clean_items = _dedupe_items(clean_items)

    if not clean_items:
        return {
            "ok": False,
            "reason": "no_valid_items",
            "market_day": market_day,
            "session_type": session_type,
            "saved": 0,
        }

    now = _now_iso()
    db = _db()

    ref = (
        db.collection(COL_ROOT)
        .document(MARKET_MEMORY_DOC)
        .collection(DAILY_ALPHA_COLLECTION)
        .document(market_day)
    )

    snap = ref.get()
    existing = snap.to_dict() if snap.exists else {}
    sessions = existing.get("sessions") or {}

    sessions[session_type] = {
        "session_type": session_type,
        "saved_at": now,
        "source": source,
        "row_count": len(clean_items),
        "unique_symbols": [x["symbol"] for x in clean_items],
        "items": clean_items,
    }

    all_symbols = []
    total_rows = 0
    sector_counter = Counter()

    for session_payload in sessions.values():
        rows = session_payload.get("items") or []
        total_rows += len(rows)

        for item in rows:
            sym = _clean_symbol(item.get("symbol"))
            if sym and sym not in all_symbols:
                all_symbols.append(sym)

            sector = item.get("sector")
            if sector:
                sector_counter[sector] += 1

    payload = {
        "date": market_day,
        "updated_at": now,
        "source": source,
        "sessions": sessions,
        "unique_symbols": all_symbols,
        "counts": {
            "sessions": len(sessions),
            "total_rows": total_rows,
            "unique_symbols": len(all_symbols),
        },
        "sector_counts": dict(sector_counter),
        "schema_version": "daily_alpha_intelligence_v1",
    }

    ref.set(payload, merge=True)

    return {
        "ok": True,
        "market_day": market_day,
        "session_type": session_type,
        "saved": len(clean_items),
        "unique_symbols": len(all_symbols),
        "updated_at": now,
    }