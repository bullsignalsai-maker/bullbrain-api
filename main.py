from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import datetime
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import base64

app = FastAPI()

# Allow Expo mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later to your app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# Load keys from environment (Render dashboard)
# ----------------------------------------------------------
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

MODEL = "grok-4-fast-reasoning"
GROK_STOCK_CACHE_HOURS = 6
WATCH_GROK_CACHE_HOURS = 12

# New BullBrain v2 model path (downloaded from Google Drive at startup)
FULLMODEL_LOCAL_PATH = "models/bullbrain_v2_40f.json"

# ----------------------------------------------------------
# Helpers: HTTP + simple cache
# ----------------------------------------------------------
def safe_json(url, headers=None, timeout=10):
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        print(f"safe_json non-200: {r.status_code} {url}")
    except Exception as e:
        print(f"safe_json error for {url}: {e}")
    return None

def now_utc():
    return datetime.datetime.utcnow()

def is_cache_fresh(cached_at: datetime.datetime, max_age_hours: float):
    return (now_utc() - cached_at).total_seconds() < max_age_hours * 3600

# In-memory caches (simple)
stock_grok_cache = {}
watch_grok_cache = {}

# ----------------------------------------------------------
# Grok (xAI) helper
# ----------------------------------------------------------
def call_grok(system_prompt: str, user_prompt: str):
    """
    Call xAI Grok model using the XAI_API_KEY.
    """
    if not XAI_API_KEY:
        print("⚠️ XAI_API_KEY missing")
        return None

    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "temperature": 0.4,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"call_grok error: {resp.status_code} {resp.text}")
            return None

        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"call_grok exception: {e}")
        return None

# ----------------------------------------------------------
# Global: BullBrain model
# ----------------------------------------------------------
bullbrain_model = None

# v2 feature names (47 total)
BULLBRAIN_FEATURES = ['close','open','high','low','volume',
 'sma5','sma10','sma20','sma50',
 'ema5','ema12','ema26','ema50',
 'vwap',
 'rsi14','rsi7',
 'macd','macd_signal','macd_hist',
 'stoch_k','stoch_d',
 'boll_mid','boll_upper','boll_lower',
 'kelt_upper','kelt_lower',
 'atr14','true_range',
 'momentum10','roc10',
 'pct_change','pct_change5','pct_change10',
 'vol_change','vol_change5','vol_change10',
 'highlowrange_pct',
 'range5','range10',
 'obv','mfi14',
 'day_of_week',
 'slope_close','slope_volume',
 'zscore_close','zscore_volume'
]


def load_bullbrain_model():
    global bullbrain_model

    # 1) Ensure local folder exists
    os.makedirs(os.path.dirname(FULLMODEL_LOCAL_PATH), exist_ok=True)

    # 2) Download from Google Drive (same file ID you used in Colab)
    file_id = "1ATNs7bsuXpVQWXMog-b7ILkKK14rOo30"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

    try:
        print("🚀 Backend starting… loading BullBrain v2 model")
        print("🔥 Downloading BullBrain model from Google Drive...")

        with requests.get(gdrive_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(FULLMODEL_LOCAL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("🔥 Model downloaded successfully.")
    except Exception as e:
        print(f"⚠️ Could not download model from Google Drive: {e}")

    # 3) Load XGBoost model
    try:
        bullbrain_model = xgb.Booster()
        bullbrain_model.load_model(FULLMODEL_LOCAL_PATH)

        # Optional: verify feature count
        model_config = json.loads(bullbrain_model.save_config())
        n_feat = int(model_config["learner"]["learner_model_param"]["num_feature"])
        print(f"🔥 REAL MODEL FEATURE COUNT: {n_feat}")
    except Exception as e:
        print(f"❌ Failed to load BullBrain model from {FULLMODEL_LOCAL_PATH}: {e}")
        bullbrain_model = None

@app.on_event("startup")
def startup_event():
    load_bullbrain_model()

# ----------------------------------------------------------
# Finnhub helpers (quote + profile)
# ----------------------------------------------------------
def backend_fetch_quote(symbol: str):
    symbol = symbol.upper()
    try:
        quote = None
        profile = {}

        if FINNHUB_KEY:
            # Finnhub quote
            q_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
            quote = requests.get(q_url, timeout=8).json()

            # Finnhub profile for name (non-critical)
            try:
                p_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_KEY}"
                profile = requests.get(p_url, timeout=8).json()
            except Exception as e:
                print(f"Finnhub profile error: {e}")

        return quote, profile
    except Exception as e:
        print(f"backend_fetch_quote error: {e}")
        return None, {}

# ----------------------------------------------------------
# Polygon candles (daily)
# ----------------------------------------------------------
def fetch_daily_candles(symbol: str, min_points: int = 60):
    symbol = symbol.upper()
    try:
        now = datetime.datetime.utcnow()
        end = int(now.timestamp())
        start = int((now - datetime.timedelta(days=400)).timestamp())

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_KEY}"
        )

        j = safe_json(url)

        # ❌ INVALID or FAILED RESPONSE
        if not j or "results" not in j:
            print("⚠️ Polygon returned no results for:", symbol)
            return None

        results = j["results"]
        if len(results) < min_points:
            print("⚠️ Polygon data too short:", symbol)
            return None

        closes = [r.get("c") for r in results]
        highs  = [r.get("h") for r in results]
        lows   = [r.get("l") for r in results]
        vols   = [r.get("v") for r in results]

        # Validate numeric
        if any(x is None for x in closes):
            print("⚠️ Missing close values for:", symbol)
            return None

        return {
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": vols,
            "source": "polygon"
        }

    except Exception as e:
        print("Polygon error:", e)
        return None

# ----------------------------------------------------------
# BullBrain v2 Feature Engineering from candles
# ----------------------------------------------------------
def compute_bullbrain_features(candles: dict):
    closes = candles["close"]
    highs  = candles["high"]
    lows   = candles["low"]
    vols   = candles["volume"]

    # Build DF
    df = pd.DataFrame({
        "close": closes,
        "high": highs,
        "low": lows,
        "volume": vols,
    })
    df["open"] = df["close"].shift(1)

    # ========== BASIC MOVING AVERAGES ==========
    df["sma5"]  = df["close"].rolling(5).mean()
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    df["ema5"]  = df["close"].ewm(span=5).mean()
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    # ========== VWAP ==========
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()

    # ========== RSI ==========
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df["rsi14"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
    df["rsi7"]  = 100 - (100 / (1 + gain.rolling(7).mean()  / loss.rolling(7).mean()))

    # ========== MACD ==========
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ========== STOCH ==========
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = (df["close"] - low14) / (high14 - low14) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ========== BOLLINGER ==========
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["boll_mid"]   = mid
    df["boll_upper"] = mid + 2 * std
    df["boll_lower"] = mid - 2 * std

    # ========== KELTNER ==========
    typical_range = (df["high"] - df["low"]).rolling(20).mean()
    df["kelt_upper"] = mid + 1.5 * typical_range
    df["kelt_lower"] = mid - 1.5 * typical_range

    # ========== ATR ==========
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["true_range"] = tr
    df["atr14"] = tr.rolling(14).mean()

    # ========== MOMENTUM ==========
    df["momentum10"] = df["close"] - df["close"].shift(10)
    df["roc10"] = df["close"].pct_change(10) * 100

    # ========== pct changes ==========
    df["pct_change"]  = df["close"].pct_change() * 100
    df["pct_change5"] = df["close"].pct_change(5) * 100
    df["pct_change10"] = df["close"].pct_change(10) * 100

    df["vol_change"]  = df["volume"].pct_change() * 100
    df["vol_change5"] = df["volume"].pct_change(5) * 100
    df["vol_change10"] = df["volume"].pct_change(10) * 100

    # ========== ranges ==========
    df["highlowrange_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    df["range5"]  = df["close"].rolling(5).apply(lambda x: x.max() - x.min())
    df["range10"] = df["close"].rolling(10).apply(lambda x: x.max() - x.min())

    # ========== OBV ==========
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # ========== MFI ==========
    money_flow = typical * df["volume"]
    pos = money_flow.where(typical.diff() > 0, 0).rolling(14).sum()
    neg = money_flow.where(typical.diff() < 0, 0).rolling(14).sum()
    df["mfi14"] = 100 - (100 / (1 + pos / neg))

    # ========== slopes ==========
    df["slope_close"]  = df["close"].rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0])
    df["slope_volume"] = df["volume"].rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0])

    # ========== z-scores ==========
    df["zscore_close"]  = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    df["zscore_volume"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

    df["day_of_week"] = datetime.datetime.utcnow().weekday()

    # ========= FINAL ROW ===========
    df_feat = df.dropna().iloc[-1]
    feature_dict = {f: float(df_feat[f]) for f in BULLBRAIN_FEATURES}

    features_vector = np.array([feature_dict[f] for f in BULLBRAIN_FEATURES]).reshape(1, -1)
    return features_vector, feature_dict, float(df_feat["close"])


# ----------------------------------------------------------
# 🔮 BullBrain v2 Inference
# ----------------------------------------------------------
def bullbrain_infer(features_vector: np.ndarray):
    """
    Run XGBoost booster on a 1x47 feature vector.
    Assumes model outputs probability of price going up (0-1).
    """
    global bullbrain_model

    if bullbrain_model is None:
        return None

    try:
        arr = np.array(features_vector, dtype=float).reshape(1, -1)

        # XGBoost DMatrix with explicit feature names
        dmat = xgb.DMatrix(arr, feature_names=BULLBRAIN_FEATURES)
        prob_up = float(bullbrain_model.predict(dmat)[0])

        # Basic ternary classification
        if prob_up >= 0.53:
            signal = "BUY"
        elif prob_up <= 0.47:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = round(abs(prob_up - 0.5) * 200, 2)  # 0–100 scale

        return {
            "probability_up": round(prob_up, 4),
            "probability_down": round(1 - prob_up, 4),
            "signal": signal,
            "confidence": confidence,
            "raw_output": prob_up,
        }

    except Exception as e:
        print(f"🔥 bullbrain_infer ERROR: {e}")
        return None

# ----------------------------------------------------------
# 🔮 API: BullBrain prediction
# ----------------------------------------------------------
@app.get("/predict/{symbol}")
def predict_symbol(symbol: str):
    """
    Full BullBrain v2 prediction for a given symbol using Polygon candles.
    """
    symbol = symbol.upper()

    # 1) Fetch candles
    candles = fetch_daily_candles(symbol)
    if not candles:
        return {"error": f"Could not fetch candles for {symbol}", "symbol": symbol}

    # 2) Build features
    feat_vec, feat_dict, last_close = compute_bullbrain_features(candles)
    if feat_vec is None:
        return {"error": "Could not compute features", "symbol": symbol}

    # 3) Run model
    model_out = bullbrain_infer(feat_vec)
    if not model_out:
        return {"error": "Model not available", "symbol": symbol}

    return {
        "symbol": symbol,
        "source": "polygon",
        "price": round(last_close, 4),
        "features": feat_dict,
        "model": model_out,
    }

# ----------------------------------------------------------
# Existing endpoints (health, quote, watchlist-item, etc.)
# ----------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "BullSignalsAI Backend Running",
        "bullbrain_loaded": bullbrain_model is not None,
        "features": BULLBRAIN_FEATURES,
    }

@app.get("/quote/{symbol}")
def get_quote(symbol: str):
    """
    Simple Finnhub quote (price etc.)
    """
    symbol = symbol.upper()
    quote, profile = backend_fetch_quote(symbol)
    if not quote:
        return {"error": f"Could not fetch quote for {symbol}"}
    return {
        "symbol": symbol,
        "price": quote.get("c"),
        "change": quote.get("d"),
        "changePct": quote.get("dp"),
        "high": quote.get("h"),
        "low": quote.get("l"),
        "open": quote.get("o"),
        "prevClose": quote.get("pc"),
        "profile": profile,
        "timestamp": quote.get("t"),
    }

@app.get("/watchlist-item/{symbol}")
def watchlist_item(symbol: str):
    """
    Simple combined endpoint the app already uses:
    - Finnhub quote
    - Minimal extra fields
    """
    symbol = symbol.upper()
    quote, profile = backend_fetch_quote(symbol)
    if not quote:
        return {"error": f"Could not fetch quote for {symbol}"}

    price = quote.get("c") or 0
    prev = quote.get("pc") or 0
    change = price - prev if prev else 0
    change_pct = (change / prev * 100) if prev else 0

    return {
        "symbol": symbol,
        "name": profile.get("name") or symbol,
        "price": round(price, 2),
        "prevClose": prev,
        "change": round(change, 2),
        "changePct": round(change_pct, 2),
        "currency": profile.get("currency", "USD"),
        "exchange": profile.get("exchange"),
        "logo": profile.get("logo"),
        "ticker": profile.get("ticker", symbol),
    }
