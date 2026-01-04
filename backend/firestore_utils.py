# backend/firestore_utils.py
import datetime
import json
import os
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore  # type: ignore

# -------------------------------------------------------------------
# Firestore Client (Singleton-style, Firebase Admin)
# -------------------------------------------------------------------

_firestore_client: Optional["firestore.Client"] = None


def _init_firebase_admin_if_needed() -> None:
    """
    Initialize Firebase Admin SDK once, using FIREBASE_ADMIN_JSON
    if provided (Render env var), otherwise default initialization.
    """
    if firebase_admin._apps:
        return

    raw = os.getenv("FIREBASE_ADMIN_JSON")

    # If Render env var is set to the JSON string, use it.
    if raw:
        try:
            # Some platforms store JSON with escaped newlines in private_key
            data = json.loads(raw)

            pk = data.get("private_key")
            if isinstance(pk, str):
                data["private_key"] = pk.replace("\\n", "\n")

            cred = credentials.Certificate(data)
            firebase_admin.initialize_app(cred)
            return
        except Exception as e:
            # If parsing fails, fall back to default init so we don't hard-crash
            print("FIREBASE_ADMIN_JSON parse/init failed, falling back:", str(e))

    # Fallback: default credentials (works on GCP environments, or if ADC is set)
    firebase_admin.initialize_app()


def get_db() -> "firestore.Client":
    """
    Returns a singleton Firestore client from Firebase Admin SDK.
    Safe to call multiple times.
    """
    global _firestore_client
    if _firestore_client is None:
        _init_firebase_admin_if_needed()
        _firestore_client = firestore.client()
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
    Example: 2026-01-02T21:01:09Z
    """
    return (
        utcnow()
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def utc_now_iso() -> str:
    """
    Backward-compatible alias used across your codebase.
    """
    return iso_now()


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
        exp = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        return exp <= utcnow()
    except Exception:
        return True
