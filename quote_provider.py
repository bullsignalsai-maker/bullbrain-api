# quote_provider.py
# ---------------------------------------------------------
# Central Quote Provider (Production)
# - Stocks / ETFs via Finnhub
# - Crypto via CoinGecko (coins/markets endpoint)  ✅ (your working logic style)
# - Sector snapshot via ETF proxies
# ---------------------------------------------------------

import os
import requests
from typing import Dict, Any, Optional

FINNHUB_KEY = os.getenv("FINNHUB_KEY")

# Reuse a session for performance
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "BullSignalsAI/1.0 (+https://bullsignals.ai)",
        "Accept": "application/json",
    }
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_pct(v: Optional[float]) -> Optional[float]:
    """
    Normalizes change % into a consistent percent number.
    (Some sources return 0.008 => 0.8)
    """
    try:
        v = float(v)
        if abs(v) <= 1.5:
            return v * 100.0
        return v
    except Exception:
        return None


# ---------------------------------------------------------
# EQUITIES / ETFs (Finnhub quote)
# ---------------------------------------------------------
def fetch_equity_quote(symbol: str) -> Dict[str, Any]:
    """
    Returns:
      {"price": float|None, "changePct": float|None}
    """
    if not FINNHUB_KEY:
        return {}

    try:
        resp = _SESSION.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": FINNHUB_KEY},
            timeout=10,
        )
        data = resp.json() if resp.ok else {}

        price = data.get("c")
        prev = data.get("pc")

        change_pct = None
        if isinstance(price, (int, float)) and isinstance(prev, (int, float)) and prev:
            change_pct = ((price - prev) / prev) * 100.0

        return {"price": price, "changePct": normalize_pct(change_pct)}
    except Exception:
        return {}


# ---------------------------------------------------------
# CRYPTO SNAPSHOT (CoinGecko – coins/markets) ✅ working-style
# ---------------------------------------------------------
def fetch_crypto_snapshot(
    symbols: Optional[list[str]] = None,
    per_page: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Backend-safe CoinGecko fetch.
    Mirrors your HomeScreen.js logic using:
      /coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1

    Returns mapping: { "BTC": float|None, "ETH": ..., ... }
    """
    wanted = symbols or ["BTC", "ETH", "SOL", "XRP", "DOGE"]

    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": 1,
        }

        resp = _SESSION.get(url, params=params, timeout=12)
        data = resp.json() if resp.ok else None

        # Build symbol -> 24h pct change map
        out: Dict[str, Optional[float]] = {s: None for s in wanted}

        if isinstance(data, list):
            for row in data:
                sym = (row.get("symbol") or "").upper()
                chg = row.get("price_change_percentage_24h")
                if sym in out and isinstance(chg, (int, float)):
                    out[sym] = float(chg)

        return out

    except Exception:
        # Do NOT throw; worker will keep running
        return {s: None for s in wanted}


# ---------------------------------------------------------
# SECTOR SNAPSHOT (ETF proxies)
# ---------------------------------------------------------
def fetch_sector_snapshot() -> Dict[str, Optional[float]]:
    """
    Returns:
      { "Technology": +1.2, "Energy": -0.7, ... }  (values are percent)
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
        out[name] = q.get("changePct") if isinstance(q, dict) else None

    return out
