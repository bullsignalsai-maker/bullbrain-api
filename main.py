# main.py — Clean Full Backend for BullSignalsAI

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# Allow Expo app to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load API Keys from Render Environment
# ---------------------------
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
def root():
    return {"status": "BullSignalsAI Backend Running"}

# ----------------------------------------------------------
# 1. Ticker Primary (FinancialModelingPrep)
# ----------------------------------------------------------
@app.get("/ticker/primary")
def ticker_primary():
    try:
        FMP_KEY = os.getenv("FMP_API_KEY")
        url = (
            "https://financialmodelingprep.com/api/v3/quote/"
            "%5EGSPC,%5EDJI,%5EIXIC,AAPL,MSFT,NVDA,TSLA,AMZN,META"
            f"?apikey={FMP_KEY}"
        )
        r = requests.get(url, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------
# 2. Yahoo Fallback
# ----------------------------------------------------------
@app.get("/ticker/yahoo")
def ticker_yahoo():
    try:
        symbols = "%5EGSPC,%5EDJI,%5EIXIC,AAPL,MSFT,NVDA,TSLA,AMZN,META"
        url = f"https://query2.finance.yahoo.com/v7/finance/quote?symbols={symbols}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://finance.yahoo.com",
        }

        resp = requests.get(url, headers=headers, timeout=5)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------
# 3. Single Quote
# ----------------------------------------------------------
@app.get("/quote/{symbol}")
def quote(symbol: str):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
        r = requests.get(url).json()

        if not r or r.get("c") in [None, 0]:
            yahoo = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d"
            y = requests.get(yahoo).json()
            meta = y.get("chart", {}).get("result", [{}])[0].get("meta", {})
            close = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose")
            return {
                "price": close,
                "change": close - prev if prev else None,
                "changePct": ((close - prev) / prev * 100) if prev else None,
            }

        return {
            "price": r["c"],
            "change": r["d"],
            "changePct": r["dp"],
            "high": r["h"],
            "low": r["l"],
            "open": r["o"],
            "prevClose": r["pc"],
            "timestamp": r["t"],
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 4. Recommendations
# ----------------------------------------------------------
@app.get("/recommendations/{symbol}")
def recommendations(symbol: str):
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_KEY}"
        r = requests.get(url).json()
        return {"data": r}
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 5. Grok Summary
# ----------------------------------------------------------
@app.post("/grok-summary")
def grok_summary(payload: dict):
    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        r = requests.post(
            "https://api.x.ai/v1/chat/completions", 
            json=payload,
            headers=headers
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 6. Full Ticker Data (Safe - No self-calls)
# ----------------------------------------------------------
@app.get("/ticker-full/{symbol}")
def ticker_full(symbol: str):
    try:
        quote_data = quote(symbol)
        rec_data = recommendations(symbol)

        return {
            "symbol": symbol,
            "quote": quote_data,
            "recommendations": rec_data,
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------------------------
# 7. Multiple Quotes
# ----------------------------------------------------------
@app.get("/quotes")
def quotes(symbols: str):
    try:
        sym_list = symbols.split(",")
        out = {}

        for s in sym_list:
            out[s] = quote(s)

        return out
    except Exception as e:
        return {"error": str(e)}
