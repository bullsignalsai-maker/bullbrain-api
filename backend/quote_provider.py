# quote_provider.py
# ---------------------------------------------------------
# Central Quote Provider (NO Firestore, NO loops)
# ---------------------------------------------------------

import os
import requests
from typing import Dict, Any, Optional

FINNHUB_KEY = os.getenv("FINNHUB_KEY")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _normalize_pct(v: Optional[float]) -> Optional[float]:
    try:
        v = float(v)
        # Finnhub sometimes returns 0.008 → 0.8%
        if abs(v) <= 1.5:
            return v * 100.0
        return v
    except Exception:
        return None


# ---------------------------------------------------------
# Equity / ETF / Index Quote (Finnhub)
# ---------------------------------------------------------
def fetch_equity_quote(symbol: str) -> Dict[str, Any]:
    """
    Returns:
      {
        price: float | None,
        changePct: float | None
      }
    """
    if not FINNHUB_KEY:
        return {}

    try:
        url = "https://finnhub.io/api/v1/quote"
        resp = requests.get(
            url,
            params={"symbol": symbol, "token": FINNHUB_KEY},
            timeout=10,
        )
        data = resp.json()

        price = data.get("c")
        prev_close = data.get("pc")

        change_pct = None
        if price is not None and prev_close:
            change_pct = ((price - prev_close) / prev_close) * 100.0

        return {
            "price": price,
            "changePct": _normalize_pct(change_pct),
        }

    except Exception:
        return {}


# ---------------------------------------------------------
# Market Index Snapshot
# ---------------------------------------------------------
def fetch_index_snapshot() -> Dict[str, Any]:
    """
    SPY → S&P 500
    QQQ → Nasdaq
    VIX → Volatility index
    """
    spy = fetch_equity_quote("SPY")
    qqq = fetch_equity_quote("QQQ")
    vix = fetch_equity_quote("VIX")

    return {
        "sp500_change": spy.get("changePct"),
        "nasdaq_change": qqq.get("changePct"),
        "vix": vix.get("price"),
    }


# ---------------------------------------------------------
# Crypto Snapshot (CoinGecko – free)
# ---------------------------------------------------------
def fetch_crypto_snapshot() -> Dict[str, Any]:
    """
    Top crypto movers (24h %)
    """
    try:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,solana,ripple,dogecoin"
            "&vs_currencies=usd"
            "&include_24hr_change=true"
        )
        data = requests.get(url, timeout=10).json()

        def pct(k):
            try:
                return float(data[k]["usd_24h_change"])
            except Exception:
                return None

        return {
            "BTC": pct("bitcoin"),
            "ETH": pct("ethereum"),
            "SOL": pct("solana"),
            "XRP": pct("ripple"),
            "DOGE": pct("dogecoin"),
        }

    except Exception:
        return {}


# ---------------------------------------------------------
# Sector Snapshot (ETF Proxies)
# ---------------------------------------------------------
def fetch_sector_snapshot() -> Dict[str, Any]:
    """
    Uses free ETF proxies (industry standard)
    """
    sectors = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer": "XLY",
    }

    out = {}
    for name, sym in sectors.items():
        q = fetch_equity_quote(sym)
        out[name] = q.get("changePct")

    return out
