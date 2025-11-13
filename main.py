# main.py — Clean Backend (No FMP, No Yahoo v7)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KEYS
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

@app.get("/")
def root():
    return {"status": "BullSignalsAI Backend Running"}

# ----------------------------------------------------------
# Finnhub Single Quote
# ----------------------------------------------------------
@app.get("/quote/{symbol}")
def quote(symbol: str):
    try:
        # Finnhub primary
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
        data = requests.get(url, timeout=4).json()

        # Fallback to Yahoo Chart API
        if not data or not data.get("c"):
            yurl = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d"
            ydata = requests.get(yurl, timeout=4).json()

            meta = (
                ydata.get("chart", {})
                .get("result", [{}])[0]
                .get("meta", {})
            )
            close = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose")

            return {
                "price": close,
                "change": (close - prev) if prev else None,
                "changePct": ((close - prev) / prev * 100) if prev else None,
            }

        return {
            "price": data["c"],
            "change": data["d"],
            "changePct": data["dp"],
            "high": data["h"],
            "low": data["l"],
            "open": data["o"],
            "prevClose": data["pc"],
            "timestamp": data["t"],
        }

    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# Analyst Recommendations
# ----------------------------------------------------------
@app.get("/recommendations/{symbol}")
def recommendations(symbol: str):
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_KEY}"
        return requests.get(url, timeout=4).json()
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# Grok Summary
# ----------------------------------------------------------
@app.post("/grok-summary")
def grok_summary(payload: dict):
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        return requests.post(url, json=payload, headers=headers).json()
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# Full Ticker Data
# ----------------------------------------------------------
@app.get("/ticker-full/{symbol}")
def ticker_full(symbol: str):
    try:
        quote = requests.get(f"https://bullbrain-api.onrender.com/quote/{symbol}").json()
        rec = requests.get(f"https://bullbrain-api.onrender.com/recommendations/{symbol}").json()

        return {
            "symbol": symbol,
            "quote": quote,
            "recommendations": rec,
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# Multi-symbol quotes
# ----------------------------------------------------------
@app.get("/quotes")
def quotes(symbols: str):
    try:
        out = {}
        for s in symbols.split(","):
            q = requests.get(f"https://bullbrain-api.onrender.com/quote/{s}").json()
            out[s] = q
        return out
    except Exception as e:
        return {"error": str(e)}
