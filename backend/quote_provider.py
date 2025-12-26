# backend/quote_provider.py
import os
import requests

# --------------------------------------------------------------------
# ENV + CONSTANTS
# --------------------------------------------------------------------
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

def fetch_quote(symbol: str) -> dict:
    """
    Pure quote fetcher.
    - No Firestore
    - No candles
    - No FastAPI
    - Safe for workers
    """
    try:
        if not FINNHUB_KEY:
            return {}

        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": symbol, "token": FINNHUB_KEY}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()

        # Finnhub returns { c, d, dp, h, l, o, pc }
        if not isinstance(data, dict):
            return {}

        return {
            "price": data.get("c"),
            "changePct": data.get("dp"),
        }

    except Exception:
        return {}
