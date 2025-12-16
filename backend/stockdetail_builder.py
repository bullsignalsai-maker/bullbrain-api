# backend/stockdetail_builder.py
"""
StockDetail Builder

Responsibilities:
- Call core stockdetail logic
- Attach schemaVersion, computedAt, expiresAt
- Used by BOTH API and cron
"""

import os
import datetime
from typing import Dict, Any

from backend.schema_versions import STOCKDETAIL_SCHEMA_VERSION
from backend.stockdetail_logic import build_stockdetail_core

# TTL in minutes (default 15)
DEFAULT_TTL_MINUTES = int(os.getenv("STOCKDETAIL_TTL_MINUTES", "15"))


def utc_iso() -> str:
    return (
        datetime.datetime.utcnow()
        .replace(microsecond=0)
        .isoformat()
        + "Z"
    )


def build_stockdetail_payload(
    symbol: str,
    force_grok: bool = False,
) -> Dict[str, Any]:
    """
    FINAL stockdetail payload builder.
    Safe to use from:
    - stockdetail_cron.py
    - FastAPI endpoint
    """

    now = datetime.datetime.utcnow()

    core = build_stockdetail_core(
        symbol=symbol,
        force_grok=force_grok,
    )

    payload = {
        **core,
        "schemaVersion": STOCKDETAIL_SCHEMA_VERSION,
        "computedAt": utc_iso(),
        "expiresAt": (
            now + datetime.timedelta(minutes=DEFAULT_TTL_MINUTES)
        )
        .replace(microsecond=0)
        .isoformat()
        + "Z",
    }

    return payload
