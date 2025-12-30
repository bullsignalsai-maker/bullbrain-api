# quote_provider.py
# ---------------------------------------------------------
# Central Quote Provider
# - Stocks / ETFs via Finnhub
# - Crypto via CoinGecko (coins/markets endpoint - like your HomeScreen.js)
# - Sector snapshot via ETF proxies
# ---------------------------------------------------------

import os
import requests
from typing import Dict, Any, Optional

FINNHUB_KEY = os.getenv("FINNHUB_KEY")

# Use a session so headers are ALWAYS present (CoinGecko is picky)
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "BullSignalsAI/1.0 (homescreen quote worker)",
        "Accept": "application/json",
    }
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_pct(v: Optional[float]) -> Optional[float]:
    try:
        if v is None:
            return None
        v = float(v)
        # If it's a decimal (e.g., 0.008 = 0.8%), convert
        if abs(v) <= 1.5:
            return v * 100.0
        return v
    except Exception:
        return None


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------
# EQUITIES / ETFs (Finnhub)
# ---------------------------------------------------------
def fetch_equity_quote(symbol: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "price": float | None,
        "changePct": float | None
      }
    """
    if not FINNHUB_KEY:
        return {}

    try:
        resp = _SESSION.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": FINNHUB_KEY},
            timeout=10,
        )
        data = _safe_json(resp)
        if not isinstance(data, dict):
            return {}

        # Finnhub: c=current, pc=previous close
        price = data.get("c")
        prev = data.get("pc")

        change_pct = None
        try:
            if isinstance(price, (int, float)) and isinstance(prev, (int, float)) and prev != 0:
                change_pct = ((float(price) - float(prev)) / float(prev)) * 100.0
        except Exception:
            change_pct = None

        return {
            "price": price if isinstance(price, (int, float)) else None,
            "changePct": normalize_pct(change_pct),
        }

    except Exception:
        return {}


# ---------------------------------------------------------
# CRYPTO SNAPSHOT (CoinGecko – free)
# Mirrors your old HomeScreen.js coins/markets logic
# ---------------------------------------------------------
def fetch_crypto_snapshot() -> Dict[str, Optional[float]]:
    """
    Returns top symbols (market-cap top10 scan) mapped to 24h % change.
    Explicitly returns BTC/ETH/SOL/XRP/DOGE keys for your carousel.
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 10,
            "page": 1,
        }

        # IMPORTANT: UA header must be present (CoinGecko blocks without it)
        resp = _SESSION.get(url, params=params, timeout=12)
        data = _safe_json(resp)

        out: Dict[str, float] = {}

        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                sym = (row.get("symbol") or "").upper().strip()
                chg = row.get("price_change_percentage_24h")

                if sym and isinstance(chg, (int, float)):
                    out[sym] = float(chg)

        # Return only what your carousel expects (stable schema)
        return {
            "BTC": out.get("BTC"),
            "ETH": out.get("ETH"),
            "SOL": out.get("SOL"),
            "XRP": out.get("XRP"),
            "DOGE": out.get("DOGE"),
        }

    except Exception as e:
        print(f"[quote-provider] CoinGecko fetch failed: {e}", flush=True)
        return {
            "BTC": None,
            "ETH": None,
            "SOL": None,
            "XRP": None,
            "DOGE": None,
        }


# ---------------------------------------------------------
# SECTOR SNAPSHOT (ETF proxies)
# ---------------------------------------------------------
def fetch_sector_snapshot() -> Dict[str, Optional[float]]:
    """
    Returns:
      {
        "Technology": +x.xx,
        "Financials": +x.xx,
        ...
      }
    """
    sectors = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer": "XLY",
    }

    out: Dict[str, Optional[float]] = {}
    for name, etf in sectors.items():
        q = fetch_equity_quote(etf)
        chg = q.get("changePct")
        out[name] = chg if isinstance(chg, (int, float)) else None

    return out
