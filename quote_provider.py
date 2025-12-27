# quote_provider.py
# ---------------------------------------------------------
# Central Quote Provider
# - Stocks / ETFs via Finnhub
# - Crypto via CoinGecko
# - Indices & sector proxies
# ---------------------------------------------------------

import os
import requests
from typing import Dict, Any, Optional

FINNHUB_KEY = os.getenv("FINNHUB_KEY")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_pct(v: Optional[float]) -> Optional[float]:
    try:
        v = float(v)
        # Finnhub sometimes returns decimals like 0.008 → 0.8%
        if abs(v) <= 1.5:
            return v * 100.0
        return v
    except Exception:
        return None


# ---------------------------------------------------------
# EQUITIES / ETFs
# ---------------------------------------------------------
def fetch_equity_quote(symbol: str) -> Dict[str, Any]:
    if not FINNHUB_KEY:
        return {}

    try:
        r = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": FINNHUB_KEY},
            timeout=10,
        )
        d = r.json()

        price = d.get("c")
        prev = d.get("pc")

        change_pct = None
        if price is not None and prev:
            change_pct = ((price - prev) / prev) * 100.0

        return {
            "price": price,
            "changePct": normalize_pct(change_pct),
        }
    except Exception:
        return {}


# ---------------------------------------------------------
# INDEX SNAPSHOT (proxies)
# ---------------------------------------------------------
def fetch_index_snapshot() -> Dict[str, Any]:
    return {
        "SPY": fetch_equity_quote("SPY"),
        "QQQ": fetch_equity_quote("QQQ"),
        "VIX": fetch_equity_quote("VIX"),
    }


# ---------------------------------------------------------
# CRYPTO SNAPSHOT (CoinGecko – free)
# ---------------------------------------------------------
def fetch_crypto_snapshot() -> Dict[str, Optional[float]]:
    """
    CoinGecko requires a User-Agent from cloud environments.
    Without this, it returns empty / throttled responses.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum,solana,ripple,dogecoin",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        }

        headers = {
            "User-Agent": "BullSignalsAI/1.0 (contact: support@bullsignals.ai)"
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()

        def chg(key):
            try:
                return float(data[key]["usd_24h_change"])
            except Exception:
                return None

        return {
            "BTC": chg("bitcoin"),
            "ETH": chg("ethereum"),
            "SOL": chg("solana"),
            "XRP": chg("ripple"),
            "DOGE": chg("dogecoin"),
        }

    except Exception as e:
        print(f"[crypto] CoinGecko fetch failed: {e}")
        return {}


# ---------------------------------------------------------
# SECTOR SNAPSHOT (ETF proxies)
# ---------------------------------------------------------
def fetch_sector_snapshot() -> Dict[str, Optional[float]]:
    sectors = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer": "XLY",
    }

    out = {}
    for name, sym in sectors.items():
        out[name] = fetch_equity_quote(sym).get("changePct")

    return out
