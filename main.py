# main.py
# ============================================================
# BullSignalsAI — API Gateway (READ-ONLY + Background Worker)
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
import asyncio

import firebase_admin
from firebase_admin import credentials, firestore  # type: ignore

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
# Globals
# ------------------------------------------------------------
db = None


# ------------------------------------------------------------
# Firebase Init
# ------------------------------------------------------------
def init_firebase_admin():
    global db

    if firebase_admin._apps:
        if db is None:
            db = firestore.client()
        return

    cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not cred_json:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT env var missing")

    cred = credentials.Certificate(eval(cred_json))
    firebase_admin.initialize_app(cred)
    db = firestore.client()


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
# MARKET TAB (READ-ONLY)
# ============================================================

@app.get("/market/overview")
def get_market_overview():
    init_firebase_admin()
    doc = db.collection("bullsignals_ai").document("market_overview_live").get()
    return doc.to_dict() if doc.exists else {}


@app.get("/market/hotlist")
def get_market_hotlist():
    init_firebase_admin()
    doc = db.collection("bullsignals_ai").document("market_hotlist").get()
    return doc.to_dict() if doc.exists else {"count": 0, "hotlist": []}


@app.get("/market/bearwatch")
def get_market_bearwatch():
    init_firebase_admin()
    doc = db.collection("bullsignals_ai").document("market_bearwatch").get()
    return doc.to_dict() if doc.exists else {"count": 0, "bearwatch": []}


@app.get("/market/pulse")
def get_market_pulse():
    init_firebase_admin()
    doc = db.collection("bullsignals_ai").document("market_pulse").get()
    return doc.to_dict() if doc.exists else {}


# ============================================================
# STOCK DETAIL (READ-ONLY)
# ============================================================

@app.get("/stock/detail/{symbol}")
def get_stock_detail(symbol: str):
    init_firebase_admin()
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
    init_firebase_admin()
    symbol = symbol.upper()

    doc = (
        db.collection("bullsignals_ai")
        .document("smart_patterns")
        .collection("items")
        .document(symbol)
        .get()
    )

    return doc.to_dict() if doc.exists else {"symbol": symbol, "pattern": None}


# ============================================================
# ASTRA CHAT (READ-ONLY)
# ============================================================

@app.get("/astra/context/{symbol}")
def get_astra_context(symbol: str):
    init_firebase_admin()
    symbol = symbol.upper()

    doc = (
        db.collection("bullsignals_ai")
        .document("astra_context")
        .collection("items")
        .document(symbol)
        .get()
    )

    return doc.to_dict() if doc.exists else {"symbol": symbol, "context": None}


# ============================================================
# MARKET NEWS (READ-ONLY)
# ============================================================

@app.get("/market/news")
def get_market_news():
    init_firebase_admin()
    doc = db.collection("bullsignals_ai").document("market_news").get()
    return doc.to_dict() if doc.exists else {"data": []}


# ============================================================
# WATCHLIST / PORTFOLIO (READ-ONLY)
# ============================================================

@app.get("/portfolio/{user_id}")
def get_portfolio(user_id: str):
    init_firebase_admin()
    doc = db.collection("portfolios").document(user_id).get()
    return doc.to_dict() if doc.exists else {"holdings": []}


@app.get("/watchlist/{user_id}")
def get_watchlist(user_id: str):
    init_firebase_admin()
    doc = db.collection("watchlists").document(user_id).get()
    return doc.to_dict() if doc.exists else {"symbols": []}


# ============================================================
# HOME SCREEN (READ-ONLY)
# ============================================================

@app.get("/homescreen")
def get_homescreen():
    init_firebase_admin()

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
# BACKGROUND — HOME SCREEN AUTO REFRESH (NO CRON)
# ============================================================

@app.on_event("startup")
async def homescreen_background_worker():
    """
    Runs INSIDE the API process.
    - One container
    - One BullBrain model
    - No Render cron
    - No OOM
    """

    async def loop():
        # Allow API to fully start
        await asyncio.sleep(10)

        # Load BullBrain ONCE
        from backend.bullbrain import ensure_bullbrain_loaded
        ensure_bullbrain_loaded()

        while True:
            try:
                log("HomeScreen background job started")

                init_firebase_admin()

                from backend.homescreen_logic import build_homescreen_mag7_block
                from backend.homescreen_macro_logic import build_homescreen_macro_snapshot
                from backend.firestore_utils import utc_now_iso

                mag7 = build_homescreen_mag7_block()
                macro = build_homescreen_macro_snapshot()

                payload = {
                    "schema_version": "homescreen_v1",
                    "updated_at": utc_now_iso(),
                    "market": macro.get("market"),
                    "macro": {
                        "carousel": macro.get("carousel", [])
                    },
                    "mag7": mag7,
                    "meta": {
                        "computed_by": "api_background_worker",
                        "refresh_minutes": 15,
                    },
                }

                db.collection("bullsignals_ai").document(
                    "homescreen_snapshot"
                ).set(payload, merge=True)

                log("HomeScreen snapshot updated successfully")

            except Exception as e:
                log(f"❌ HomeScreen background job failed: {e}")

            await asyncio.sleep(15 * 60)  # 15 minutes

    asyncio.create_task(loop())
