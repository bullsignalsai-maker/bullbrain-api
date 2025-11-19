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
BULLBRAIN_FEATURES = [
    "close", "open", "high", "low", "volume",
    "sma5", "sma10", "sma20", "sma50",
    "ema5", "ema12", "ema26", "ema50",
    "vwap",
    "rsi14", "rsi7",
    "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d",
    "boll_mid", "boll_upper", "boll_lower", "kelt_upper", "kelt_lower",
    "atr14", "true_range",
    "momentum10", "roc10",
    "pct_change", "pct_change5", "pct_change10",
    "vol_change", "vol_change5", "vol_change10",
    "highlowrange_pct", "range5", "range10",
    "obv", "mfi14", "day_of_week",
    "slope_close", "slope_volume", "zscore_close", "zscore_volume", "trend_strength",
]

def load_bullbrain_model():
    global bullbrain_model

    # 1) Ensure local folder exists
    os.makedirs(os.path.dirname(FULLMODEL_LOCAL_PATH), exist_ok=True)

    # 2) Download from Google Drive (same file ID you used in Colab)
    file_id = "1qDZ0NvErxV6AWft4fkt3EVZW4S9fEAo2"
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
        start = int((now - datetime.timedelta(days=365)).timestamp())

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_KEY}"
        )
        j = safe_json(url)
        if j and "results" in j:
            closes = [r.get("c") for r in j["results"]]
            highs = [r.get("h") for r in j["results"]]
            lows = [r.get("l") for r in j["results"]]
            opens = [r.get("o", r.get("c")) for r in j["results"]]
            vols = [r.get("v") for r in j["results"]]

            if len(closes) >= min_points:
                return {
                    "c": closes,
                    "h": highs,
                    "l": lows,
                    "o": opens,
                    "v": vols,
                }
    except Exception as e:
        print(f"fetch_daily_candles error for {symbol}: {e}")
    return None

# ----------------------------------------------------------
# BullBrain v2 Feature Engineering from candles
# ----------------------------------------------------------
def compute_bullbrain_features(candles: dict):
    """
    BullBrain v2 (40+ features) inference feature builder.

    Input:  candles dict with at least:
            - "c": closes
            - "h": highs
            - "l": lows
            - "v": volumes
            - optionally "o": opens (if missing, we approximate with close)

    Output:
        features_vector: list[float] in the exact order of BULLBRAIN_FEATURES (len == 47)
        feature_dict:    dict[str, float] same keys as BULLBRAIN_FEATURES
        last_close:      float, latest close used for price
    """
    try:
        closes = candles.get("c") or []
        highs = candles.get("h") or []
        lows = candles.get("l") or []
        vols = candles.get("v") or []
        opens = candles.get("o") or closes

        if not (closes and highs and lows and vols):
            raise ValueError("compute_bullbrain_features: missing candle arrays")

        # Use last N points for stability
        N = min(len(closes), 120)
        c = np.array(closes[-N:], dtype=float)
        h = np.array(highs[-N:], dtype=float)
        l = np.array(lows[-N:], dtype=float)
        v = np.array(vols[-N:], dtype=float)
        if opens:
            o = np.array(opens[-N:], dtype=float)
        else:
            o = c.copy()

        # Build DataFrame for convenience
        df = pd.DataFrame(
            {
                "close": c,
                "open": o,
                "high": h,
                "low": l,
                "volume": v,
            }
        )

        # ---------- Helpers ----------
        def rolling_slope(series, window=10):
            if len(series) < window:
                return np.nan
            y = series[-window:]
            x = np.arange(window, dtype=float)
            A = np.vstack([x, np.ones_like(x)]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return float(m)

        # ---------- Moving averages ----------
        df["sma5"] = df["close"].rolling(5).mean()
        df["sma10"] = df["close"].rolling(10).mean()
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()

        df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        # VWAP
        cum_vol = df["volume"].cumsum()
        cum_pv = (df["close"] * df["volume"]).cumsum()
        df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)

        # ---------- RSI ----------
        def compute_rsi(close_series, window=14):
            delta = close_series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
            avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df["rsi14"] = compute_rsi(df["close"], 14)
        df["rsi7"] = compute_rsi(df["close"], 7)

        # ---------- MACD ----------
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = macd - signal

        # ---------- Stochastic ----------
        low_min = df["low"].rolling(14).min()
        high_max = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min).replace(
            0, np.nan
        )
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ---------- Bollinger & Keltner ----------
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["boll_mid"] = mid
        df["boll_upper"] = mid + 2 * std
        df["boll_lower"] = mid - 2 * std

        # ATR for Keltner
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["true_range"] = true_range
        df["atr14"] = true_range.rolling(14).mean()

        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        df["kelt_upper"] = ema20 + 2 * df["atr14"]
        df["kelt_lower"] = ema20 - 2 * df["atr14"]

        # ---------- Momentum ----------
        df["momentum10"] = df["close"] - df["close"].shift(10)
        df["roc10"] = (df["close"] / df["close"].shift(10) - 1) * 100

        df["pct_change"] = df["close"].pct_change() * 100
        df["pct_change5"] = df["close"].pct_change(5) * 100
        df["pct_change10"] = df["close"].pct_change(10) * 100

        df["vol_change"] = df["volume"].pct_change() * 100
        df["vol_change5"] = df["volume"].pct_change(5) * 100
        df["vol_change10"] = df["volume"].pct_change(10) * 100

        df["highlowrange_pct"] = (
            (df["high"] - df["low"]) / df["close"].replace(0, np.nan) * 100
        )
        df["range5"] = (
            df["close"].rolling(5).max() - df["close"].rolling(5).min()
        ) / df["close"].rolling(5).mean().replace(0, np.nan) * 100
        df["range10"] = (
            df["close"].rolling(10).max() - df["close"].rolling(10).min()
        ) / df["close"].rolling(10).mean().replace(0, np.nan) * 100

        # ---------- OBV ----------
        direction = np.sign(df["close"].diff().fillna(0))
        df["obv"] = (direction * df["volume"]).cumsum()

        # ---------- MFI ----------
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]
        pos_mf = raw_money_flow.where(typical_price > typical_price.shift(), 0.0)
        neg_mf = raw_money_flow.where(typical_price < typical_price.shift(), 0.0)
        pos_mf_sum = pos_mf.rolling(14).sum()
        neg_mf_sum = neg_mf.rolling(14).sum()
        mfr = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
        df["mfi14"] = 100 - (100 / (1 + mfr))

        # ---------- Slopes & z-scores ----------
        df["slope_close"] = df["close"].rolling(10).apply(rolling_slope, raw=False)
        df["slope_volume"] = df["volume"].rolling(10).apply(rolling_slope, raw=False)

        df["zscore_close"] = df["close"].rolling(20).apply(
            lambda s: (s.iloc[-1] - s.mean()) / (s.std() if s.std() != 0 else 1),
            raw=False,
        )
        df["zscore_volume"] = df["volume"].rolling(20).apply(
            lambda s: (s.iloc[-1] - s.mean()) / (s.std() if s.std() != 0 else 1),
            raw=False,
        )

        # Day-of-week (approx; last bar only, 0–4) – here set 0 as generic weekday
        df["day_of_week"] = 0

        # Trend strength (synthetic index)
        df["trend_strength"] = (
            (df["slope_close"].fillna(0)
             / df["highlowrange_pct"].replace(0, np.nan).fillna(1))
            * 10
        )

        # Take last row
        last = df.iloc[-1].copy()

        # Fallback: replace NaNs / inf
        last = last.replace([np.inf, -np.inf], np.nan)
        last = last.fillna(0.0)

        feature_dict = {}
        for name in BULLBRAIN_FEATURES:
            feature_dict[name] = float(last.get(name, 0.0))

        features_vector = [feature_dict[name] for name in BULLBRAIN_FEATURES]
        last_close = float(last["close"])

        return features_vector, feature_dict, last_close

    except Exception as e:
        print(f"🔥 compute_bullbrain_features ERROR: {e}")
        return None, None, None

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
