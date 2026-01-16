# backend/news/market_news_repo.py
from typing import Dict, Any
import datetime

from backend.firestore_utils import get_db, utc_now_iso
from backend.news.market_news_sources import fetch_market_news_raw
from backend.news.market_news_cleaner import clean_market_news

DOC_ID = "market_news"
TTL_MINUTES = 30


def _is_expired(updated_at: str | None) -> bool:
    if not updated_at:
        return True
    try:
        ts = datetime.datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds()
        return age > TTL_MINUTES * 60
    except Exception:
        return True


def get_market_news() -> Dict[str, Any]:
    db = get_db()
    ref = db.collection("bullsignals_ai").document(DOC_ID)
    snap = ref.get()

    if snap.exists:
        data = snap.to_dict() or {}
        if not _is_expired(data.get("updated_at")):
            return {
                "source": "cache",
                "updated_at": data.get("updated_at"),
                "count": len(data.get("items", [])),
                "items": data.get("items", []),
            }

    raw = fetch_market_news_raw()
    cleaned = clean_market_news(raw)

    payload = {
        "items": cleaned,
        "updated_at": utc_now_iso(),
        "ttl_minutes": TTL_MINUTES,
    }

    ref.set(payload, merge=True)

    return {
        "source": "fresh",
        "updated_at": payload["updated_at"],
        "count": len(cleaned),
        "items": cleaned,
    }
