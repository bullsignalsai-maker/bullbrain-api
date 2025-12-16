# backend/schema_versions.py

"""
Central place for Firestore schema versions.
Bumping a version = new collection (safe migrations).
"""

# ---------- Stock Detail ----------
STOCKDETAIL_SCHEMA_VERSION = "v1"
STOCKDETAIL_BUILD_VERSION = "2025-01-01"

# ---------- Grok ----------
GROK_SCHEMA_VERSION = "v1"

# ---------- TTLs (seconds) ----------
# Used by cron & API validation logic
TTL_STOCKDETAIL_SECONDS = 15 * 60        # 15 minutes
TTL_QUOTE_SECONDS = 60                   # 1 minute
TTL_GROK_SECONDS = 6 * 60 * 60            # 6 hours
TTL_NEWS_SECONDS = 30 * 60                # 30 minutes
