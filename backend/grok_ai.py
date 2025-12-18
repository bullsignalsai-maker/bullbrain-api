# backend/grok_ai.py

import os
import json
import time
import hashlib
import requests
from typing import Dict, Any, Optional

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-fast-reasoning")
GROK_TIMEOUT = int(os.getenv("GROK_TIMEOUT", "20"))


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_grok_stock_insight(
    symbol: str,
    quote: Dict[str, Any],
    technical: Optional[Dict[str, Any]],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Returns Grok reasoning + probability in a UI-safe structure.
    NEVER raises.
    """

    if not XAI_API_KEY:
        return _empty_grok("Missing XAI_API_KEY")

    try:
        prompt = _build_prompt(symbol, quote, technical)
        raw = _call_grok(prompt)
        return _parse_grok_response(raw)
    except Exception as e:
        return _empty_grok(str(e))


# ------------------------------------------------------------
# Prompt Engineering
# ------------------------------------------------------------

def _build_prompt(
    symbol: str,
    quote: Dict[str, Any],
    technical: Optional[Dict[str, Any]],
) -> str:
    """
    Controlled, explainable prompt.
    Grok should explain like a market analyst, not a quant paper.
    """

    price = quote.get("current")
    change = quote.get("changePct")

    tech_summary = json.dumps(technical, indent=2) if technical else "N/A"

    return f"""
You are a senior market analyst.

Analyze the stock {symbol} using:
- Current price: {price}
- Daily change (%): {change}
- Technical indicators: {tech_summary}

Your task:
1. Explain the technical outlook in **plain English**
2. Mention key risks and opportunities
3. Provide a probability (0–1) that the stock moves UP in the short term
4. Do NOT use jargon without explanation
5. Keep the explanation under 120 words

Return STRICT JSON in this format:

{{
  "summary": "...",
  "technicalInsight": "...",
  "risks": ["...", "..."],
  "opportunities": ["...", "..."],
  "prob_up": 0.00
}}
""".strip()


# ------------------------------------------------------------
# Grok Call
# ------------------------------------------------------------

def _call_grok(prompt: str) -> str:
    """
    Raw call to xAI Grok.
    """
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 500,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=GROK_TIMEOUT)
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]


# ------------------------------------------------------------
# Parsing & Safety
# ------------------------------------------------------------

def _parse_grok_response(raw: str) -> Dict[str, Any]:
    """
    Enforces schema + safety.
    """

    try:
        data = json.loads(raw)
    except Exception:
        return _empty_grok("Invalid JSON from Grok")

    return {
        "summary": data.get("summary"),
        "technicalInsight": data.get("technicalInsight"),
        "risks": _safe_list(data.get("risks")),
        "opportunities": _safe_list(data.get("opportunities")),
        "prob_up": _safe_prob(data.get("prob_up")),
        "source": "grok",
    }


def _safe_list(x) -> list:
    return x if isinstance(x, list) else []


def _safe_prob(x) -> float:
    try:
        p = float(x)
        return max(0.0, min(1.0, p))
    except Exception:
        return 0.5


def _empty_grok(reason: str) -> Dict[str, Any]:
    """
    Guaranteed fallback.
    """
    return {
        "summary": None,
        "technicalInsight": None,
        "risks": [],
        "opportunities": [],
        "prob_up": 0.5,
        "source": "grok",
        "error": reason,
    }
