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

def fetch_equity_quote(symbol: str) -> Dict[str, Any]:
    if not FINNHUB_KEY:
        print("[quote-provider] FINNHUB_KEY missing", flush=True)
        return {}

    try:
        resp = _SESSION.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": FINNHUB_KEY},
            timeout=10,
        )

        data = _safe_json(resp)

        if resp.status_code != 200:
            print(
                f"[quote-provider] Finnhub HTTP {resp.status_code} for {symbol}: {data}",
                flush=True,
            )
            return {}

        if isinstance(data, dict) and data.get("error"):
            print(
                f"[quote-provider] Finnhub error for {symbol}: {data.get('error')}",
                flush=True,
            )
            return {}

        if not isinstance(data, dict):
            return {}

        def num(v):
            return float(v) if isinstance(v, (int, float)) else None

        return {
            "price": num(data.get("c")),
            "changePct": num(data.get("dp")),
            "change": num(data.get("d")),
            "open": num(data.get("o")),
            "high": num(data.get("h")),
            "low": num(data.get("l")),
            "prevClose": num(data.get("pc")),
            "timestamp": int(data.get("t")) if isinstance(data.get("t"), int) else None,
            "source": "finnhub",
        }

    except Exception as e:
        print(f"[quote-provider] Finnhub failed for {symbol}: {e}", flush=True)
        return {}
# ---------------------------------------------------------
# CRYPTO SNAPSHOT — CoinGecko SIMPLE PRICE (LOW RATE)
# ---------------------------------------------------------
def fetch_crypto_simple_snapshot() -> Dict[str, Dict[str, Optional[float]]]:
    """
    Uses CoinGecko simple/price endpoint.
    Returns price + 24h % change.
    Rate-limit safe.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum,solana,ripple,dogecoin",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        }

        resp = _SESSION.get(url, params=params, timeout=10)
        data = _safe_json(resp)

        if not isinstance(data, dict):
            return {}

        def num(v):
            return float(v) if isinstance(v, (int, float)) else None

        return {
            "BTC": {
                "price": num(data.get("bitcoin", {}).get("usd")),
                "changePct": num(data.get("bitcoin", {}).get("usd_24h_change")),
            },
            "ETH": {
                "price": num(data.get("ethereum", {}).get("usd")),
                "changePct": num(data.get("ethereum", {}).get("usd_24h_change")),
            },
            "SOL": {
                "price": num(data.get("solana", {}).get("usd")),
                "changePct": num(data.get("solana", {}).get("usd_24h_change")),
            },
            "XRP": {
                "price": num(data.get("ripple", {}).get("usd")),
                "changePct": num(data.get("ripple", {}).get("usd_24h_change")),
            },
            "DOGE": {
                "price": num(data.get("dogecoin", {}).get("usd")),
                "changePct": num(data.get("dogecoin", {}).get("usd_24h_change")),
            },
        }

    except Exception as e:
        print(f"[quote-provider] Crypto price fetch failed: {e}", flush=True)
        return {}


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
