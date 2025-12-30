# backend/grok_ai.py
"""
Grok / XAI AI layer for BullSignalsAI.

Design rules (strict):
- NO FastAPI endpoints here
- NO Firestore here
- NO candle fetching here
- NO BullBrain feature computation here

This module is ONLY:
- Grok API calling
- Prompt building
- Safe parsing
- Lightweight in-memory caching (per Grok call type)
- Reusable helpers that other backend modules can import

Other modules (astra_chat.py, stockdetail_logic.py, etc.) should import and call:
- astra_llm_answer(...)
- grok_prob_up(...)
- get_stockdetail_grok(...)
- grok_watchlist_sentiment(...)
- compute_hybrid_signal(...)

Env vars expected:
- XAI_API_KEY
- GROK_MODEL (optional)
- GROK_STOCK_CACHE_HOURS (optional)
- WATCH_GROK_CACHE_HOURS (optional)
"""

from __future__ import annotations

import os
import time
import json
import re
import requests
from typing import Optional, Dict, Any, Tuple

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
MODEL = os.getenv("GROK_MODEL", "grok-4-fast-reasoning")

# Cache TTLs
def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "", "None") else default
    except Exception:
        return default


GROK_STOCK_CACHE_HOURS = _env_float("GROK_STOCK_CACHE_HOURS", 6.0)
WATCH_GROK_CACHE_HOURS = _env_float("WATCH_GROK_CACHE_HOURS", 6.0)

# Grok endpoint
XAI_CHAT_URL = os.getenv("XAI_CHAT_URL", "https://api.x.ai/v1/chat/completions")

# -------------------------------------------------------------------
# Internal cache (Grok-only)
# key = ("prob_up", "AAPL") or ("stockdetail", "AAPL", fingerprint) etc.
# -------------------------------------------------------------------
_GROK_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _cache_get(key: Tuple[Any, ...], ttl_seconds: float) -> Optional[Any]:
    item = _GROK_CACHE.get(key)
    if not item:
        return None
    if _now() - float(item.get("ts", 0)) > ttl_seconds:
        return None
    return item.get("data")


def _cache_set(key: Tuple[Any, ...], data: Any) -> None:
    _GROK_CACHE[key] = {"ts": _now(), "data": data}


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _safe_round(v: Any, ndigits: int = 4) -> Optional[float]:
    try:
        if v is None:
            return None
        fv = float(v)
        if fv != fv:  # NaN
            return None
        return round(fv, ndigits)
    except Exception:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _fingerprint_for_stockdetail(symbol: str, quote: Optional[dict], technical: Optional[dict]) -> str:
    """
    Build a stable-ish fingerprint so we don't cache stale responses forever
    across changing market data.
    """
    q = quote or {}
    t = technical or {}
    parts = [
        symbol.upper(),
        str(_safe_round(q.get("price"), 4)),
        str(_safe_round(q.get("changePct"), 4)),
        str(_safe_round(q.get("prevClose"), 4)),
        str(_safe_round(t.get("rsi14") or t.get("rsi"), 2)),
        str(_safe_round(t.get("sma20"), 4)),
        str(_safe_round(t.get("sma50"), 4)),
        str(_safe_round(t.get("macd_hist") or t.get("macdHist"), 4)),
    ]
    return "|".join(parts)


def extract_json_block(text: str) -> Optional[dict]:
    """
    Attempts to extract a JSON object from an LLM response.

    Handles:
    - raw JSON
    - JSON inside ```json ... ```
    - JSON embedded in text

    Returns dict or None.
    """
    if not text or not isinstance(text, str):
        return None

    s = text.strip()

    # 1) Code-fence JSON
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2) Any code-fence
    m = re.search(r"```\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) First full object from first "{" to last "}"
    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            blob = s[start : end + 1]
            try:
                return json.loads(blob)
            except Exception:
                pass

    return None


# -------------------------------------------------------------------
# Low-level Grok call
# -------------------------------------------------------------------
def _call_grok(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.35,
    max_tokens: int = 700,
    timeout_sec: int = 18,
) -> Optional[str]:
    if not XAI_API_KEY:
        return None

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(XAI_CHAT_URL, headers=headers, json=payload, timeout=timeout_sec)
        if resp.status_code >= 400:
            # Don’t throw; fail gracefully
            return None

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None

        msg = choices[0].get("message") or {}
        content = msg.get("content")
        return content if isinstance(content, str) else None

    except Exception:
        return None


def astra_llm_answer(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Public: main helper used across modules (Astra Chat, stockdetail, etc.)
    """
    return _call_grok(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.4,
        max_tokens=750,
        timeout_sec=20,
    )


# -------------------------------------------------------------------
# Prompt builders (centralized)
# -------------------------------------------------------------------
def build_prob_up_prompt(symbol: str, quote: Optional[dict], technical: Optional[dict]) -> Tuple[str, str]:
    sym = (symbol or "").upper().strip()
    q = quote or {}
    t = technical or {}

    system = (
        "You are an expert market analyst. "
        "Return ONLY valid JSON. No markdown. No extra text."
    )

    # Keep the prompt compact and numeric.
    user = {
        "task": "Estimate probability that the stock closes higher over the next 5 trading days (0..1).",
        "symbol": sym,
        "inputs": {
            "quote": {
                "price": q.get("price"),
                "changePct": q.get("changePct"),
                "high": q.get("high"),
                "low": q.get("low"),
                "open": q.get("open"),
                "prevClose": q.get("prevClose"),
            },
            "technical": {
                "rsi14": t.get("rsi14") or t.get("rsi"),
                "sma20": t.get("sma20"),
                "sma50": t.get("sma50"),
                "sma200": t.get("sma200"),
                "macd_hist": t.get("macd_hist") or t.get("macdHist"),
                "atr14": t.get("atr14"),
                "volatility_20d": t.get("volatility_20d"),
            },
        },
        "output_schema": {
            "probability_up": "number in [0,1]",
            "confidence": "Low|Moderate|High",
            "rationale": "one short sentence, no financial advice",
        },
    }

    return system, json.dumps(user)


def build_stockdetail_grok_prompt(symbol: str, quote: Optional[dict], technical: Optional[dict]) -> Tuple[str, str]:
    sym = (symbol or "").upper().strip()
    q = quote or {}
    t = technical or {}

    system = (
        "You are a calm, practical stock analyst inside a mobile app. "
        "Return ONLY valid JSON. No markdown. No bullets. No headings."
    )

    user = {
        "task": "Generate a concise stock insight block for the app UI (no advice).",
        "symbol": sym,
        "inputs": {
            "quote": {
                "price": q.get("price"),
                "changePct": q.get("changePct"),
                "high": q.get("high"),
                "low": q.get("low"),
                "open": q.get("open"),
                "prevClose": q.get("prevClose"),
            },
            "technical": {
                "rsi14": t.get("rsi14") or t.get("rsi"),
                "sma20": t.get("sma20"),
                "sma50": t.get("sma50"),
                "sma200": t.get("sma200"),
                "macd": t.get("macd"),
                "macd_signal": t.get("macd_signal") or t.get("macdSignal"),
                "macd_hist": t.get("macd_hist") or t.get("macdHist"),
                "atr14": t.get("atr14"),
                "volatility_5d": t.get("volatility_5d"),
                "volatility_20d": t.get("volatility_20d"),
            },
        },
        "output_schema": {
            "summary": "1-2 short sentences",
            "outlook": "Bullish|Neutral|Bearish",
            "risk": "Low|Medium|High",
            "confidence": "Low|Moderate|High",
            "key_drivers": ["array of 2-3 short strings"],
            "disclaimer": "single short sentence: not financial advice",
        },
    }

    return system, json.dumps(user)


def build_watchlist_sentiment_prompt(symbol: str, change_pct: Optional[float]) -> Tuple[str, str]:
    sym = (symbol or "").upper().strip()

    system = (
        "You are an assistant inside a stock watchlist screen. "
        "Write ONE short sentence only. No markdown. No advice. No hype."
    )

    user = {
        "task": "Write a one-sentence watchlist sentiment comment.",
        "symbol": sym,
        "changePct": change_pct,
        "rules": [
            "Must be one sentence.",
            "No 'buy'/'sell'.",
            "No price targets.",
            "No emojis.",
        ],
    }

    return system, json.dumps(user)


# -------------------------------------------------------------------
# Public: Grok probability_up
# -------------------------------------------------------------------
def grok_prob_up(
    symbol: str,
    quote: Optional[dict] = None,
    technical: Optional[dict] = None,
    *,
    force: bool = False,
) -> Optional[float]:
    """
    Returns probability_up as float in [0,1], or None.
    Uses in-memory cache keyed by symbol + light fingerprint.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return None

    ttl = int(GROK_STOCK_CACHE_HOURS * 3600)
    fp = _fingerprint_for_stockdetail(sym, quote, technical)
    key = ("prob_up", sym, fp)

    if not force:
        cached = _cache_get(key, ttl)
        if cached is not None:
            try:
                return float(cached)
            except Exception:
                pass

    system, user = build_prob_up_prompt(sym, quote, technical)
    raw = _call_grok(system, user, temperature=0.25, max_tokens=220, timeout_sec=16)
    if not raw:
        return None

    obj = extract_json_block(raw)
    if not obj:
        return None

    p = obj.get("probability_up")
    try:
        p = float(p)
        if p != p:
            return None
        p = _clamp(p, 0.0, 1.0)
        _cache_set(key, p)
        return p
    except Exception:
        return None


# -------------------------------------------------------------------
# Public: Hybrid signal computation
# -------------------------------------------------------------------
def compute_hybrid_signal(
    bull_conf: float,
    grok_prob: float,
    *,
    bull_weight: float = 0.65,
    grok_weight: float = 0.35,
) -> dict:
    """
    Combine BullBrain confidence (0..1) with Grok probability_up (0..1).
    Returns a stable structure your UI can rely on.

    Note:
    - bull_conf: model probability_up (0..1)
    - grok_prob: grok probability_up (0..1)
    """
    try:
        b = _clamp(float(bull_conf), 0.0, 1.0)
    except Exception:
        b = 0.5

    try:
        g = _clamp(float(grok_prob), 0.0, 1.0)
    except Exception:
        g = 0.5

    bw = float(bull_weight)
    gw = float(grok_weight)
    if bw < 0 or gw < 0 or (bw + gw) <= 0:
        bw, gw = 0.65, 0.35

    hybrid = (b * bw) + (g * gw)
    hybrid = _clamp(hybrid, 0.0, 1.0)

    def _signal(p: float) -> str:
        # keep consistent with your app’s expectations
        if p >= 0.60:
            return "BUY"
        if p <= 0.40:
            return "SELL"
        return "NEUTRAL"

    bull_signal = _signal(b)
    grok_signal = _signal(g)
    hybrid_signal = _signal(hybrid)

    agreement = (
        "agree" if bull_signal == grok_signal else
        "partial" if hybrid_signal in (bull_signal, grok_signal) else
        "conflict"
    )

    return {
        "bullProbUp": round(b, 4),
        "grokProbUp": round(g, 4),
        "hybridProbUp": round(hybrid, 4),
        "bullSignal": bull_signal,
        "grokSignal": grok_signal,
        "hybridSignal": hybrid_signal,
        "agreement": agreement,
        "weights": {"bull": round(bw, 3), "grok": round(gw, 3)},
    }


# -------------------------------------------------------------------
# Public: StockDetail Grok block (structured)
# -------------------------------------------------------------------
def get_stockdetail_grok(
    symbol: str,
    quote: Optional[dict],
    technical: Optional[dict],
    *,
    force: bool = False,
) -> Optional[dict]:
    """
    Returns a dict like:
    {
      "summary": "...",
      "outlook": "Bullish|Neutral|Bearish",
      "risk": "Low|Medium|High",
      "confidence": "Low|Moderate|High",
      "key_drivers": [...],
      "disclaimer": "..."
    }
    or None if unavailable.

    Cached by symbol + fingerprint.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return None

    ttl = int(GROK_STOCK_CACHE_HOURS * 3600)
    fp = _fingerprint_for_stockdetail(sym, quote, technical)
    key = ("stockdetail", sym, fp)

    if not force:
        cached = _cache_get(key, ttl)
        if isinstance(cached, dict):
            return cached

    system, user = build_stockdetail_grok_prompt(sym, quote, technical)
    raw = _call_grok(system, user, temperature=0.35, max_tokens=420, timeout_sec=18)
    if not raw:
        return None

    obj = extract_json_block(raw)
    if not isinstance(obj, dict):
        return None

    # Validate / normalize minimal shape
    summary = obj.get("summary")
    outlook = obj.get("outlook")
    risk = obj.get("risk")
    confidence = obj.get("confidence")
    key_drivers = obj.get("key_drivers")
    disclaimer = obj.get("disclaimer")

    if not isinstance(summary, str) or not summary.strip():
        return None

    out = {
        "summary": summary.strip()[:380],
        "outlook": (outlook or "Neutral"),
        "risk": (risk or "Medium"),
        "confidence": (confidence or "Moderate"),
        "key_drivers": key_drivers if isinstance(key_drivers, list) else [],
        "disclaimer": (disclaimer or "AI-generated insight, not financial advice."),
        "model": MODEL,
    }

    _cache_set(key, out)
    return out


# -------------------------------------------------------------------
# Public: Watchlist / Market quick sentiment
# -------------------------------------------------------------------
def grok_watchlist_sentiment(
    symbol: str,
    change_pct: Optional[float],
    *,
    force: bool = False,
) -> Optional[str]:
    """
    Returns ONE short sentence (string) or None.
    Cached by symbol + change_pct (rounded).
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return None

    ttl = int(WATCH_GROK_CACHE_HOURS * 3600)
    cp = _safe_round(change_pct, 2)
    key = ("watch_sent", sym, cp)

    if not force:
        cached = _cache_get(key, ttl)
        if isinstance(cached, str) and cached.strip():
            return cached

    system, user = build_watchlist_sentiment_prompt(sym, cp)
    raw = _call_grok(system, user, temperature=0.5, max_tokens=60, timeout_sec=14)
    if not raw:
        return None

    sentence = raw.strip().replace("\n", " ").strip()
    # Ensure "one sentence" style: trim extra whitespace
    sentence = re.sub(r"\s+", " ", sentence).strip()

    # Hard guard: keep it short
    if len(sentence) > 180:
        sentence = sentence[:180].rsplit(" ", 1)[0].strip() + "."

    if sentence:
        _cache_set(key, sentence)
        return sentence

    return None


# -------------------------------------------------------------------
# Optional helper: Clear cache (useful for debugging)
# -------------------------------------------------------------------
def grok_cache_clear(prefix: Optional[str] = None) -> int:
    """
    Clears in-memory Grok cache.
    If prefix provided (e.g., 'stockdetail'), removes only those keys.
    Returns number of removed entries.
    """
    if not prefix:
        n = len(_GROK_CACHE)
        _GROK_CACHE.clear()
        return n

    to_del = [k for k in _GROK_CACHE.keys() if k and str(k[0]) == str(prefix)]
    for k in to_del:
        _GROK_CACHE.pop(k, None)
    return len(to_del)
