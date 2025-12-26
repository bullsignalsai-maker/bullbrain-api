# quote_provider.py
# ---------------------------------------------------------
# Central Quote Provider (NO Firestore, NO loops)
# ---------------------------------------------------------

import os
import requests
from typing import Dict, Any, Optional

FINNHUB_KEY = os.getenv("FINNHUB_KEY")


def _normalize_pct(v: Optional[float]) -> Optional[float]:
    try:
        v = float(v)
        if abs(v) <= 1.5:
            return v * 100.0
        return v
    except Exception:
        return None


def fetch_equity_quote(symbol: str) -> Dict[str, Any]:
    """
    Uses Finnhub quote endpoint.
    Returns:
      { "price": float|None, "changePct": float|None }
    """
    if not FINNHUB_KEY:
        return {}

    try:
        url = "https://finnhub.io/api/v1/quote"
        resp = requests.get(url, params={"symbol": symbol, "token": FINNHUB_KEY}, timeout=10)
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


def fetch_index_snapshot() -> Dict[str, Any]:
    """
    SPY -> S&P proxy
    QQQ -> Nasdaq proxy
    VIX -> volatility
    """
    spy = fetch_equity_quote("SPY")
    qqq = fetch_equity_quote("QQQ")
    vix = fetch_equity_quote("VIX")

    return {
        "sp500_change": spy.get("changePct"),
        "nasdaq_change": qqq.get("changePct"),
        "vix": vix.get("price"),
    }


def fetch_crypto_snapshot() -> Dict[str, Any]:
    """
    Top crypto 24h changes (CoinGecko free)
    """
    try:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,solana,ripple,dogecoin"
            "&vs_currencies=usd"
            "&include_24hr_change=true"
        )
        data = requests.get(url, timeout=10).json()

        def pct(k: str) -> Optional[float]:
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


def fetch_sector_snapshot() -> Dict[str, Any]:
    """
    ETF proxies for sector performance (simple + reliable)
    """
    sectors = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer": "XLY",
    }

    out: Dict[str, Any] = {}
    for name, sym in sectors.items():
        q = fetch_equity_quote(sym)
        out[name] = q.get("changePct")

    return out
