# backend/firestore_utils.py

import datetime
from typing import Optional

from google.cloud import firestore

# -------------------------------------------------------------------
# Firestore Client (Singleton-style)
# -------------------------------------------------------------------

_firestore_client: Optional[firestore.Client] = None


def get_db() -> firestore.Client:
    """
    Returns a singleton Firestore client.
    Safe to call multiple times.
    """
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.Client()
    return _firestore_client


# -------------------------------------------------------------------
# Time Helpers (UTC only — always)
# -------------------------------------------------------------------

def utcnow() -> datetime.datetime:
    """
    Current UTC datetime with timezone.
    """
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)


def iso_now() -> str:
    """
    Current UTC time as ISO-8601 string with Z suffix.
    Example: 2025-02-10T18:42:11Z
    """
    return (
        utcnow()
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def compute_expires_at(minutes: int) -> str:
    """
    Compute ISO timestamp for TTL expiry.
    """
    exp = utcnow() + datetime.timedelta(minutes=minutes)
    return (
        exp.replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_expired(expires_at: Optional[str]) -> bool:
    """
    Check whether an ISO timestamp is expired.
    Returns True if missing or invalid.
    """
    if not expires_at:
        return True
    try:
        exp = datetime.datetime.fromisoformat(
            expires_at.replace("Z", "+00:00")
        )
        return exp <= utcnow()
    except Exception:
        return True
