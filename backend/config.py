# backend/config.py
import os

# ----------------------------------------------------------
# Environment keys (Render dashboard)
# ----------------------------------------------------------
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ----------------------------------------------------------
# LLM / Caching defaults
# ----------------------------------------------------------
MODEL = os.getenv("XAI_MODEL", "grok-4-fast-reasoning")

GROK_STOCK_CACHE_HOURS = int(os.getenv("GROK_STOCK_CACHE_HOURS", "6"))
WATCH_GROK_CACHE_HOURS = int(os.getenv("WATCH_GROK_CACHE_HOURS", "6"))

# ----------------------------------------------------------
# Versions
# ----------------------------------------------------------
BULLBRAIN_VERSION = os.getenv("BULLBRAIN_VERSION", "v2")
SCHEMA_VERSION = os.getenv("SCHEMA_VERSION", "v1")
