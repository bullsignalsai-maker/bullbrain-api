# backend/firestore_paths.py

from schema_versions import (
    STOCKDETAIL_SCHEMA_VERSION,
    GROK_SCHEMA_VERSION,
)

# ==========================
# COLLECTION NAMES
# ==========================

def stockdetail_collection():
    return f"stockdetail_cache_{STOCKDETAIL_SCHEMA_VERSION}"


def grok_collection():
    return f"grok_cache_{GROK_SCHEMA_VERSION}"


def stockdetail_meta_collection():
    return f"stockdetail_meta_{STOCKDETAIL_SCHEMA_VERSION}"


# ==========================
# DOCUMENT HELPERS
# ==========================

def stockdetail_doc(symbol: str) -> str:
    return symbol.upper()


def grok_doc(symbol: str) -> str:
    return symbol.upper()
