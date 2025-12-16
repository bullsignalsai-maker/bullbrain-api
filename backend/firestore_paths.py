# backend/firestore_paths.py

import json
import os
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

from backend.schema_versions import (
    STOCKDETAIL_SCHEMA_VERSION,
    GROK_SCHEMA_VERSION,
)

# ==========================
# Firebase init (safe to call multiple times)
# ==========================
def init_firebase_admin_once():
    if firebase_admin._apps:
        return firebase_admin._apps[0]

    firebase_json = os.getenv("FIREBASE_ADMIN_JSON")
    if not firebase_json:
        raise RuntimeError("FIREBASE_ADMIN_JSON is missing in environment variables")

    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    return firebase_admin.initialize_app(cred)


def get_db():
    """
    Returns Firestore client using firebase-admin.
    Safe for API + cron.
    """
    if not firebase_admin._apps:
        init_firebase_admin_once()
    return firestore.client()


# ==========================
# COLLECTION NAMES
# ==========================
def stockdetail_collection() -> str:
    return f"stockdetail_cache_{STOCKDETAIL_SCHEMA_VERSION}"


def grok_collection() -> str:
    return f"grok_cache_{GROK_SCHEMA_VERSION}"


def stockdetail_meta_collection() -> str:
    return f"stockdetail_meta_{STOCKDETAIL_SCHEMA_VERSION}"


# ==========================
# DOCUMENT HELPERS
# ==========================
def stockdetail_doc_id(symbol: str) -> str:
    return symbol.upper()


def grok_doc_id(symbol: str) -> str:
    return symbol.upper()


# ==========================
# DOCUMENT REFERENCES
# ==========================
def stockdetail_doc_ref(symbol: str, db=None):
    """
    Usage:
      ref = stockdetail_doc_ref("AAPL")          # uses default db()
      ref = stockdetail_doc_ref("AAPL", db=db)   # uses passed db
    """
    db = db or get_db()
    return db.collection(stockdetail_collection()).document(stockdetail_doc_id(symbol))
