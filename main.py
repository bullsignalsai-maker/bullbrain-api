# main.py
# ============================================================
# BullSignalsAI — API Gateway (READ-ONLY)
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime

# ------------------------------------------------------------
# Internal imports
# ------------------------------------------------------------
from backend.firestore_paths import get_db

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(title="BullSignalsAI API", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Logging helper
# ------------------------------------------------------------
def log(msg: str):
    print(f"[api] {msg}")


# ============================================================
# HEALTH
# ============================================================
@app.get("/")
def root():
    return {
        "status": "BullSignalsAI backend running",
        "time": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ============================================================
# HOME SCREEN (READ FROM FIRESTORE)
# ============================================================
@app.get("/homescreen")
def get_homescreen():
    db = get_db()

    doc = (
        db.collection("bullsignals_ai")
        .document("homescreen_snapshot")
        .get()
    )

    if not doc.exists:
        return {
            "status": "unavailable",
            "message": "HomeScreen snapshot not ready yet",
        }

    return doc.to_dict()


# ============================================================
# MARKET TAB (READ-ONLY)
# ============================================================

@app.get("/market/overview")
def get_market_overview():
    db = get_db()
    doc = db.collection("bullsignals_ai").document("market_overview_live").get()
    return doc.to_dict() if doc.exists else {}


@app.get("/market/hotlist")
def get_market_hotlist():
    db = get_db()
    doc = db.collection("bullsignals_ai").document("market_hotlist").get()
    return doc.to_dict() if doc.exists else {"count": 0, "hotlist": []}


@app.get("/market/bearwatch")
def get_market_bearwatch():
    db = get_db()
    doc = db.collection("bullsignals_ai").document("market_bearwatch").get()
    return doc.to_dict() if doc.exists else {"count": 0, "bearwatch": []}


@app.get("/market/pulse")
def get_market_pulse():
    db = get_db()
    doc = db.collection("bullsignals_ai").document("market_pulse").get()
    return doc.to_dict() if doc.exists else {}


# ============================================================
# STOCK DETAIL (READ-ONLY)
# ============================================================

@app.get("/stock/detail/{symbol}")
def get_stock_detail(symbol: str):
    db = get_db()
    symbol = symbol.upper()

    doc = (
        db.collection("bullsignals_ai")
        .document("stock_details")
        .collection("items")
        .document(symbol)
        .get()
    )

    if not doc.exists:
        return {"symbol": symbol, "error": "No data available"}

    return doc.to_dict()


# ============================================================
# SMART PATTERN (READ-ONLY)
# ============================================================

@app.get("/stock/smart-pattern/{symbol}")
def get_smart_pattern(symbol: str):
    db = get_db()
    symbol = symbol.upper()

    doc = (
        db.collection("bullsignals_ai")
        .document("smart_patterns")
        .collection("items")
        .document(symbol)
        .get()
    )

    return doc.to_dict() if doc.exists else {
        "symbol": symbol,
        "pattern": None,
    }


# ============================================================
# ASTRA CHAT (READ-ONLY)
# ============================================================

@app.get("/astra/context/{symbol}")
def get_astra_context(symbol: str):
    db = get_db()
    symbol = symbol.upper()

    doc = (
        db.collection("bullsignals_ai")
        .document("astra_context")
        .collection("items")
        .document(symbol)
        .get()
    )

    return doc.to_dict() if doc.exists else {
        "symbol": symbol,
        "context": None,
    }


# ============================================================
# MARKET NEWS (READ-ONLY)
# ============================================================

@app.get("/market/news")
def get_market_news():
    db = get_db()
    doc = db.collection("bullsignals_ai").document("market_news").get()
    return doc.to_dict() if doc.exists else {"data": []}


# ============================================================
# WATCHLIST / PORTFOLIO (READ-ONLY)
# ============================================================

@app.get("/portfolio/{user_id}")
def get_portfolio(user_id: str):
    db = get_db()
    doc = db.collection("portfolios").document(user_id).get()
    return doc.to_dict() if doc.exists else {"holdings": []}


@app.get("/watchlist/{user_id}")
def get_watchlist(user_id: str):
    db = get_db()
    doc = db.collection("watchlists").document(user_id).get()
    return doc.to_dict() if doc.exists else {"symbols": []}
