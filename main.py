# main.py — BullSignalsAI Backend (Production, with BullBrain v1 Full Model)


from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import datetime
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import gdown  # <-- ADDED

app = FastAPI()

# Allow Expo mobile app access
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # you can restrict later
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
POLYGON_KEY = os.getenv("POLYGON_API_KEY")  # <— NEW

MODEL = "grok-4-fast-reasoning"
GROK_STOCK_CACHE_HOURS = 1
WATCH_GROK_CACHE_HOURS = 1
BULLBRAIN_VERSION = "v2-48f"


# ----------------------------------------------------------
# 🔥 BullBrain v1 Full Model (XGBoost JSON)
# ----------------------------------------------------------
# Google Drive model (REPLACE FILE_ID BELOW)
# Google Drive model (REPLACE FILE_ID BELOW)
# 🔥 BullBrain v2 model (XGBoost JSON)
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1TeutMa8jQ5l4Lw-ZaN1gP1iGfDp5spAJ"
FULLMODEL_LOCAL_PATH = "models/bullbrain_v2_48f.json"

BULLBRAIN_FEATURES = [
    "adj_close",
    "close",
    "high",
    "low",
    "open",
    "volume",
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "sma5",
    "sma10",
    "sma20",
    "sma50",
    "sma200",
    "sma5_sma20_pct",
    "sma20_sma50_pct",
    "price_vs_sma20_pct",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema12",
    "ema26",
    "ema_ratio",
    "williams_r_14",
    "stoch_k_14",
    "stoch_d_3",
    "volume_change_1d",
    "volume_ma5",
    "volume_ma20",
    "volume_vs_ma5_pct",
    "volume_vs_ma20_pct",
    "obv",
    "obv_slope_10",
    "intraday_range_pct",
    "true_range",
    "atr14",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "gap_pct",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "volume_zscore_20",
    "trend_strength_20",
]

bullbrain_model = None  # will hold xgb.Booster instance

# Simple in-memory cache for Grok and watchlist summaries
cache = {}

# ----------------------------------------------------------
# 🌐 Utility: Safe JSON fetch
# ----------------------------------------------------------
def safe_json(url, timeout=10):
   try:
       r = requests.get(url, timeout=timeout)
       if r.status_code != 200:
           return None
       return r.json()
   except Exception:
       return None

# ----------------------------------------------------------
# 📦 NEW: Load BullBrain model from Google Drive
# ----------------------------------------------------------

# Google Drive model link (your file)
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1TeutMa8jQ5l4Lw-ZaN1gP1iGfDp5spAJ"
FULLMODEL_LOCAL_PATH = "models/bullbrain_v2_48f.json"

def load_bullbrain_model():
   """Download latest BullBrain model from Google Drive and load it."""
   print("🔥 BullBrain: Preparing model directory")
   os.makedirs("models", exist_ok=True)


   try:
       print("🔥 Downloading latest model from Google Drive...")
       gdown.download(
           MODEL_DRIVE_URL,
           FULLMODEL_LOCAL_PATH,
           quiet=False,
           fuzzy=True
       )
       print("🔥 Model download completed.")
   except Exception as e:
       print("⚠️ Download failed — attempting to load existing local model.")
       print("Drive error:", e)


   if not os.path.exists(FULLMODEL_LOCAL_PATH):
       raise FileNotFoundError(
           f"❌ No model found locally or on Drive: {FULLMODEL_LOCAL_PATH}"
       )


   booster = xgb.Booster()
   booster.load_model(FULLMODEL_LOCAL_PATH)


   print("🔥 BullBrain model LOADED from:", FULLMODEL_LOCAL_PATH)
   print("🔥 Booster num_features:", booster.num_features())
   print("🔥 Booster feature_names:", booster.feature_names)
   return booster

# ----------------------------------------------------------
# 🚀 Startup hook — load BullBrain model
# ----------------------------------------------------------
@app.on_event("startup")
def on_startup():
   global bullbrain_model
   print("🚀 Backend starting… loading BullBrain v1 model")
   try:
       bullbrain_model = load_bullbrain_model()
   except Exception as e:
       print("⚠️ Failed to load BullBrain model on startup:", e)
# ----------------------------------------------------------
@app.get("/")
def root():
   return {
       "status": "BullSignalsAI Backend Running",
       "bullbrain_loaded": bullbrain_model is not None,
       "features": BULLBRAIN_FEATURES,
   }

def _class_probs_from_prob_up(prob_up: float) -> dict:
    """
    Convert a single 'probability_up' from the model into a
    3-way distribution over SELL / HOLD / BUY.

    NOTE: Model is binary under the hood; this mapping is a
    heuristic for UI / visualization, not a separate training.
    """
    p = float(prob_up)
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0

    # Three bands:
    # - Strong bearish (p <= 0.4)
    # - Neutral (0.4 < p < 0.6)
    # - Strong bullish (p >= 0.6)

    if p >= 0.6:
        # Mostly BUY, small share to HOLD
        buy = p
        hold = 1.0 - p
        sell = 0.0
    elif p <= 0.4:
        # Mostly SELL, small share to HOLD
        sell = 1.0 - p
        hold = p
        buy = 0.0
    else:
        # Neutral band – mix BUY/SELL around HOLD
        # Map p in [0.4, 0.6] → balanced distribution
        center_offset = p - 0.5  # -0.1 to +0.1
        hold = 0.6  # dominant in neutral band
        buy = max(0.0, 0.2 + center_offset * 2.0)   # 0.0..0.4
        sell = max(0.0, 0.2 - center_offset * 2.0)  # 0.0..0.4

    total = buy + hold + sell
    if total <= 0:
        return {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}

    return {
        "SELL": sell / total,
        "HOLD": hold / total,
        "BUY": buy / total,
    }
# ----------------------------------------------------------
# Helper: backend quote fetch (Finnhub + Yahoo fallback)
# ----------------------------------------------------------
def backend_fetch_quote(symbol: str):
   symbol = symbol.upper()

   try:
       quote = None
       profile = {}
       # 1️⃣ FINNHUB — Primary
       if FINNHUB_KEY:
           q_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
           quote = safe_json(q_url, timeout=8)
           # Profile (non-critical)
           p_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_KEY}"
           profile = safe_json(p_url, timeout=8) or {}

       # 2️⃣ FALLBACK — If Finnhub bad, try Yahoo
       if not quote or "c" not in quote or quote["c"] in [None, 0]:
           y_url = (
               f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{symbol}?range=1d&interval=1d"
           )
           y = safe_json(y_url, timeout=8)
           if not y:
               return {
                   "symbol": symbol,
                   "name": symbol,
                   "current": None,
                   "change": 0,
                   "changePct": 0,
                   "high": 0,
                   "low": 0,
                   "open": 0,
                   "prevClose": 0,
                   "timestamp": int(datetime.datetime.utcnow().timestamp()),
               }


           try:
               meta = (
                   y.get("chart", {})
                   .get("result", [{}])[0]
                   .get("meta", {})
               )
           except Exception:
               meta = {}


           close = meta.get("regularMarketPrice")
           prev = meta.get("previousClose") or meta.get("chartPreviousClose")


           if close is None:
               return {
                   "symbol": symbol,
                   "name": symbol,
                   "current": None,
                   "change": 0,
                   "changePct": 0,
                   "high": 0,
                   "low": 0,
                   "open": 0,
                   "prevClose": 0,
                   "timestamp": int(datetime.datetime.utcnow().timestamp()),
               }


           change = (close - prev) if prev else 0.0
           change_pct = ((close - prev) / prev * 100) if prev else 0.0


           return {
               "symbol": symbol,
               "name": profile.get("name") or symbol,
               "current": float(close),
               "change": float(change),
               "changePct": float(change_pct),
               "high": float(close),
               "low": float(close),
               "open": float(prev) if prev else float(close),
               "prevClose": float(prev) if prev else float(close),
               "timestamp": int(datetime.datetime.utcnow().timestamp()),
           }


       # 3️⃣ NORMAL FINNHUB PATH
       price = float(quote["c"])
       prev = float(quote.get("pc") or price)


       change = float(quote.get("d") or (price - prev))
       change_pct = float(
           quote.get("dp") or ((price - prev) / prev * 100 if prev else 0)
       )


       return {
           "symbol": symbol,
           "name": profile.get("name") or symbol,
           "current": price,
           "change": change,
           "changePct": change_pct,
           "high": float(quote.get("h") or price),
           "low": float(quote.get("l") or price),
           "open": float(quote.get("o") or prev),
           "prevClose": float(prev),
           "timestamp": int(
               quote.get("t") or datetime.datetime.utcnow().timestamp()
           ),
       }


   except Exception as e:
       print("backend_fetch_quote fatal error:", e)
       return {
           "symbol": symbol,
           "name": symbol,
           "current": None,
           "change": 0,
           "changePct": 0,
           "high": 0,
           "low": 0,
           "open": 0,
           "prevClose": 0,
           "timestamp": int(datetime.datetime.utcnow().timestamp()),
       }

# ----------------------------------------------------------
# 📈 Helper: Fetch OHLCV candles (Polygon → Yahoo fallback)
# ----------------------------------------------------------
# 📈 Helper: Fetch OHLCV candles (Polygon → Yahoo fallback)
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
            closes = [r["c"] for r in j["results"]]
            highs = [r["h"] for r in j["results"]]
            lows = [r["l"] for r in j["results"]]
            vols = [r["v"] for r in j["results"]]
            opens = [r.get("o", r["c"]) for r in j["results"]]  # fallback to close if missing

            if len(closes) >= min_points:
                return {
                    "source": "polygon",
                    "close": closes,
                    "high": highs,
                    "low": lows,
                    "open": opens,
                    "volume": vols,
                }

    except Exception as e:
        print("Polygon error:", e)

    return None


# ----------------------------------------------------------
# 🧮 Helper: Compute 10-engineered features for BullBrain
# ----------------------------------------------------------
# ----------------------------------------------------------
# 🧮 Helper: Compute 48 engineered features for BullBrain v2
# ----------------------------------------------------------
def _rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_rolling = pd.Series(gain, index=series.index).rolling(period).mean()
    loss_rolling = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = gain_rolling / (loss_rolling + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _stoch_kd(df, period=14, smooth_k=3, smooth_d=3):
    lowest_low = df["low"].rolling(period).min()
    highest_high = df["high"].rolling(period).max()
    k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-9)
    k_smooth = k.rolling(smooth_k).mean()
    d = k_smooth.rolling(smooth_d).mean()
    return k_smooth, d


def _atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def compute_bullbrain_features(candles: dict):
    """
    Reproduces the Colab engineer_features() logic for a single symbol.

    Input candles dict must have:
      - close, high, low, open, volume  (lists of floats)

    Returns:
      features_vector: np.array shape (1, 48) in correct order
      feature_dict: {feature_name: value or None}
      last_close: float
    """
    closes = candles["close"]
    highs = candles["high"]
    lows = candles["low"]
    vols = candles["volume"]
    opens = candles.get("open") or closes

    df = pd.DataFrame(
        {
            "close": closes,
            "high": highs,
            "low": lows,
            "open": opens,
            "volume": vols,
        }
    ).reset_index(drop=True)

    # adj_close ≈ close at inference time
    df["adj_close"] = df["close"]

    # ---------------------------
    # 1. Returns
    # ---------------------------
    df["return_1d"] = df["close"].pct_change() * 100.0
    df["return_5d"] = df["close"].pct_change(5) * 100.0
    df["return_10d"] = df["close"].pct_change(10) * 100.0

    # ---------------------------
    # 2. Volatility (rolling std of returns)
    # ---------------------------
    daily_ret = df["close"].pct_change()
    df["volatility_5d"] = daily_ret.rolling(5).std() * 100.0
    df["volatility_20d"] = daily_ret.rolling(20).std() * 100.0
    df["volatility_60d"] = daily_ret.rolling(60).std() * 100.0

    # ---------------------------
    # 3. Moving Averages
    # ---------------------------
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    # Relative positions
    df["sma5_sma20_pct"] = (df["sma5"] / df["sma20"] - 1.0) * 100.0
    df["sma20_sma50_pct"] = (df["sma20"] / df["sma50"] - 1.0) * 100.0
    df["price_vs_sma20_pct"] = (df["close"] / df["sma20"] - 1.0) * 100.0

    # ---------------------------
    # 4. RSI 14
    # ---------------------------
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # ---------------------------
    # 5. MACD + EMA12/26 + ratio
    # ---------------------------
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["ema12"] = ema12
    df["ema26"] = ema26
    df["ema_ratio"] = ema12 / ema26

    # ---------------------------
    # 6. Williams %R and Stoch K/D
    # ---------------------------
    hh14 = df["high"].rolling(14).max()
    ll14 = df["low"].rolling(14).min()
    df["williams_r_14"] = (df["close"] - hh14) / (hh14 - ll14) * 100.0

    df["stoch_k_14"] = (df["close"] - ll14) / (hh14 - ll14) * 100.0
    df["stoch_d_3"] = df["stoch_k_14"].rolling(3).mean()

    # ---------------------------
    # 7. Volume features
    # ---------------------------
    df["volume_change_1d"] = df["volume"].pct_change() * 100.0
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_vs_ma5_pct"] = (df["volume"] / df["volume_ma5"] - 1.0) * 100.0
    df["volume_vs_ma20_pct"] = (df["volume"] / df["volume_ma20"] - 1.0) * 100.0

    # ---------------------------
    # 8. OBV + OBV slope (10)
    # ---------------------------
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    def _slope_10(x):
        # x is a length-10 window
        return np.polyfit(range(10), x, 1)[0]

    df["obv_slope_10"] = df["obv"].rolling(10).apply(_slope_10, raw=False)

    # ---------------------------
    # 9. Price range features
    # ---------------------------
    df["intraday_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100.0

    # True range & ATR14 (same as Colab)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = tr
    df["atr14"] = tr.rolling(14).mean()

    # ---------------------------
    # 10. Candle anatomy
    # ---------------------------
    df["upper_shadow_pct"] = (df["high"] - df["close"]) / df["close"] * 100.0
    df["lower_shadow_pct"] = (df["close"] - df["low"]) / df["close"] * 100.0
    df["body_pct"] = (df["close"] - df["open"]) / df["open"] * 100.0

    # Gap vs previous close
    df["gap_pct"] = (df["open"] - df["close"].shift()) / df["close"].shift() * 100.0

    # ---------------------------
    # 11. Distance from 20-day extremes
    # ---------------------------
    rolling_high_20 = df["high"].rolling(20).max()
    rolling_low_20 = df["low"].rolling(20).min()
    df["distance_from_20d_high"] = (df["close"] / rolling_high_20 - 1.0) * 100.0
    df["distance_from_20d_low"] = (df["close"] / rolling_low_20 - 1.0) * 100.0

    # ---------------------------
    # 12. Volume Z-score (20)
    # ---------------------------
    vol_ma20 = df["volume_ma20"]
    vol_std20 = vol_ma20.rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - vol_ma20) / vol_std20

    # ---------------------------
    # 13. Trend strength (20) via slope
    # ---------------------------
    def _slope_20(x):
        return np.polyfit(range(20), x, 1)[0]

    df["trend_strength_20"] = df["close"].rolling(20).apply(_slope_20, raw=False)

    # ---------------------------
    # Pick the latest row for inference
    # ---------------------------
    row = df.iloc[-1]
    last_close = float(row["close"])

    # Build feature vector in exact training order
    values_for_model = []
    feature_dict = {}

    for name in BULLBRAIN_FEATURES:
        raw = row.get(name, np.nan)
        # For model we keep NaN as NaN (XGBoost can handle)
        values_for_model.append(float(raw) if pd.notna(raw) else np.nan)
        # For JSON, convert NaN to None
        feature_dict[name] = None if pd.isna(raw) else float(raw)

    features_vector = np.array([values_for_model], dtype=float)

    return features_vector, feature_dict, last_close



def _run_bullbrain_for_symbol(symbol: str):
    """
    Core BullBrain pipeline:
    - Fetch candles
    - Compute 48 features
    - Run model
    - Build structured result used by multiple endpoints
    """
    symbol = symbol.upper()

    if bullbrain_model is None:
        return None, {"error": "BullBrain model not loaded yet."}

    candles = fetch_daily_candles(symbol)
    if not candles:
        return None, {"error": f"Could not fetch candles for {symbol}"}

    # Features + inference
    features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
    inference = bullbrain_infer(features_vec)

    # Extract raw probability_up
    prob_up = inference.get("probability_up")
    if prob_up is None:
        prob_up = float(inference.get("raw_output", 0.5))
    prob_down = 1.0 - float(prob_up)

    class_probs = _class_probs_from_prob_up(prob_up)

    as_of = datetime.datetime.utcnow().isoformat()

    core = {
        "symbol": symbol,
        "asOf": as_of,
        "source": candles.get("source", "polygon"),
        "price": last_close,
        "features": feature_dict,
        "bullbrain": {
            "version": BULLBRAIN_VERSION,
            "signal": inference.get("signal"),
            "confidence": inference.get("confidence"),
            "probabilities": class_probs,
            "raw": {
                "prob_up": float(prob_up),
                "prob_down": float(prob_down),
            },
        },
        # Legacy model field for backward compatibility with existing app code
        "model": inference,
    }

    return core, None

# ----------------------------------------------------------
# 🔮 BullBrain v1 Inference
# ----------------------------------------------------------
def bullbrain_infer(features_vector: np.ndarray):
   """
   Run XGBoost booster on a 1x10 feature vector.
   Assumes model outputs probability of price going up (0-1).
   """
   global bullbrain_model
   if bullbrain_model is None:
       raise RuntimeError("BullBrain model not loaded")


   dmat = xgb.DMatrix(features_vector, feature_names=BULLBRAIN_FEATURES)
   preds = bullbrain_model.predict(dmat)


   # Handle shapes like (1,) or (1,1) safely
   arr = np.array(preds).ravel()
   if arr.size == 0:
       raise RuntimeError("Model returned no prediction")


   prob_up = float(arr[0])


   if prob_up >= 0.55:
       signal = "BUY"
   elif prob_up <= 0.45:
       signal = "SELL"
   else:
       signal = "HOLD"


   confidence = round(max(prob_up, 1 - prob_up) * 100.0, 2)


   return {
       "signal": signal,
       "confidence": confidence,
       "probability_up": round(prob_up, 4),
       "probability_down": round(1 - prob_up, 4),
       "raw_output": prob_up,
   }
# ----------------------------------------------------------
# 🔮 1. Main BullBrain Prediction (v2, structured)
# ----------------------------------------------------------
@app.get("/predict/{symbol}")
def predict_symbol(symbol: str):
    """
    Full BullBrain v2 signal for a ticker:
    - Fetch daily candles
    - Compute 48 engineered features
    - Run XGBoost model
    - Return structured result (B-style) + legacy keys
    """
    core, err = _run_bullbrain_for_symbol(symbol)
    if err is not None:
        return {"symbol": symbol.upper(), **err}

    # core already contains:
    # symbol, asOf, source, price, features, bullbrain, model(legacy)
    return core

# ----------------------------------------------------------
# 1. Quote (Primary Finnhub, fallback Yahoo)
# ----------------------------------------------------------
@app.get("/quote/{symbol}")
def quote(symbol: str):
   try:
       q = backend_fetch_quote(symbol)
       if not q:
           return {"error": "Quote unavailable"}


       return {
           "price": q["current"],
           "change": q["change"],
           "changePct": q["changePct"],
           "high": q["high"],
           "low": q["low"],
           "open": q["open"],
           "prevClose": q["prevClose"],
           "timestamp": q["timestamp"],
       }


   except Exception as e:
       return {"error": str(e)}

# ----------------------------------------------------------
# 🔮 2. Class probabilities endpoint
# ----------------------------------------------------------
@app.get("/predict-prob/{symbol}")
def predict_prob(symbol: str):
    """
    Returns only the class probability distribution for a symbol:
    {
      "symbol": "NVDA",
      "asOf": "...",
      "probabilities": {
        "SELL": 0.23,
        "HOLD": 0.17,
        "BUY": 0.60
      },
      "raw": {
        "prob_up": ...,
        "prob_down": ...
      }
    }
    """
    core, err = _run_bullbrain_for_symbol(symbol)
    if err is not None:
        return {"symbol": symbol.upper(), **err}

    bb = core["bullbrain"]
    return {
        "symbol": core["symbol"],
        "asOf": core["asOf"],
        "probabilities": bb["probabilities"],
        "raw": bb["raw"],
        "version": bb.get("version", BULLBRAIN_VERSION),
    }

# ----------------------------------------------------------
# 🔮 3. Batch prediction for multiple tickers
# ----------------------------------------------------------
@app.get("/predict-multi")
def predict_multi(tickers: str = Query(..., description="Comma-separated tickers")):
    """
    Batch BullBrain predictions for multiple symbols in one request.

    Example:
      /predict-multi?tickers=AAPL,TSLA,NVDA

    Returns:
    {
      "data": [
        { ... /predict response for AAPL ... },
        { ... /predict response for TSLA ... },
        ...
      ],
      "errors": [
        { "symbol": "XYZ", "error": "Could not fetch candles" }
      ]
    }
    """
    if not tickers:
        return {"data": [], "errors": []}

    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    results = []
    errors = []

    for sym in symbols:
        core, err = _run_bullbrain_for_symbol(sym)
        if err is not None:
            errors.append({"symbol": sym, "error": err.get("error", "Unknown error")})
        else:
            results.append(core)

    return {
        "data": results,
        "errors": errors,
    }

# ----------------------------------------------------------
# 2. Analyst recommendations
# ----------------------------------------------------------
@app.get("/recommendations/{symbol}")
def recommendations(symbol: str):
   try:
       if not FINNHUB_KEY:
           return {"data": []}
       url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_KEY}"
       data = requests.get(url, timeout=8).json()
       return {"data": data}
   except Exception as e:
       return {"error": str(e)}


# ----------------------------------------------------------
# 3. Generic Grok/XAI summary (kept if used elsewhere)
# ----------------------------------------------------------
@app.post("/grok-summary")
def grok_summary(payload: dict):
   try:
       headers = {
           "Authorization": f"Bearer {XAI_API_KEY}",
           "Content-Type": "application/json",
       }
       url = "https://api.x.ai/v1/chat/completions"
       resp = requests.post(url, json=payload, headers=headers, timeout=20)
       return resp.json()
   except Exception as e:
       return {"error": str(e)}

# ----------------------------------------------------------
# 4. Full ticker data (combines /quote + /recommendations)
# ----------------------------------------------------------
@app.get("/ticker-full/{symbol}")
def ticker_full(symbol: str):
   try:
       q = backend_fetch_quote(symbol)
       rec_data = recommendations(symbol)
       return {
           "symbol": symbol.upper(),
           "quote": q,
           "recommendations": rec_data,
       }
   except Exception as e:
       return {"error": str(e)}

# ----------------------------------------------------------
# 5. Multiple quotes
# ----------------------------------------------------------
@app.get("/quotes")
def quotes(symbols: str):
   try:
       out = {}
       for s in symbols.split(","):
           s = s.strip().upper()
           if not s:
               continue
           q = backend_fetch_quote(s)
           out[s] = q
       return out
   except Exception as e:
       return {"error": str(e)}

# ----------------------------------------------------------
# 6. Macro Watch (FMP)
# ----------------------------------------------------------
@app.get("/macro-watch")
def macro_watch():
   try:
       today = datetime.date.today()
       to_date = today + datetime.timedelta(days=10)


       url = (
           "https://financialmodelingprep.com/api/v3/economic_calendar"
           f"?from={today}&to={to_date}&apikey={FMP_API_KEY}"
       )
       data = requests.get(url, timeout=10).json()
       return {"data": data[:20] if isinstance(data, list) else []}
   except Exception as e:
       return {"data": [], "error": str(e)}

# ----------------------------------------------------------
# 7. Earnings
# ----------------------------------------------------------
@app.get("/earnings")
def earnings():
   try:
       today = datetime.date.today()
       next_week = today + datetime.timedelta(days=7)


       url = (
           "https://financialmodelingprep.com/api/v3/earning_calendar"
           f"?from={today}&to={next_week}&apikey={FMP_API_KEY}"
       )
       data = requests.get(url, timeout=10).json()
       return {"data": data[:20] if isinstance(data, list) else []}
   except Exception as e:
       return {"data": [], "error": str(e)}

# ----------------------------------------------------------
# 8. Live stats (fear & greed + VIX + S&P)
# ----------------------------------------------------------
@app.get("/stats/live")
def live_stats():
   try:
       # Fear & Greed Index (placeholder – can wire RapidAPI later)
       fearGreed = {"value": 50, "label": "Neutral"}


       # VIX
       vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
       vix_data = requests.get(vix_url, timeout=10).json()
       vix = (
           vix_data.get("chart", {})
           .get("result", [{}])[0]
           .get("meta", {})
           .get("regularMarketPrice", 15)
       )


       # S&P Change %
       sp_url = "https://query1.finance.yahoo.com/v8/finance/chart/^GSPC"
       sp_data = requests.get(sp_url, timeout=10).json()
       sp_meta = (
           sp_data.get("chart", {})
           .get("result", [{}])[0]
           .get("meta", {})
       )
       prev = sp_meta.get("previousClose")
       sp_change = (
           (sp_meta.get("regularMarketPrice") - prev) / prev * 100
           if prev
           else 0
       )


       return {
           "fearGreed": fearGreed,
           "vix": round(vix, 2),
           "sp500_change": round(sp_change, 2),
       }
   except Exception as e:
       return {
           "fearGreed": {"value": 50, "label": "Neutral"},
           "vix": 14.5,
           "sp500_change": 0.2,
           "error": str(e),
       }

# ----------------------------------------------------------
# 9. Market Mood (Fear & Greed + VIX) — for MoodService
# ----------------------------------------------------------
@app.get("/market-mood")
def market_mood():
   try:
       # Fear & Greed Index
       fng = requests.get(
           "https://api.alternative.me/fng/?limit=1&format=json",
           timeout=5,
       ).json()


       fear_value = int(fng.get("data", [{}])[0].get("value", 50))
       fear_label = fng.get("data", [{}])[0].get("value_classification", "Neutral")


       # VIX Index
       vix_json = requests.get(
           "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
           timeout=5,
       ).json()


       vix_price = (
           vix_json.get("chart", {})
           .get("result", [{}])[0]
           .get("meta", {})
           .get("regularMarketPrice", 15.0)
       )


       return {
           "data": {
               "fearGreed": {"value": fear_value, "label": fear_label},
               "vix": round(float(vix_price), 2),
           }
       }


   except Exception as e:
       return {
           "data": {
               "fearGreed": {"value": 50, "label": "Neutral"},
               "vix": 15.0,
           },
           "error": str(e),
       }

# ----------------------------------------------------------
# 10. Grok Stock Analysis (for StockDetailScreen)
# ----------------------------------------------------------
@app.get("/grok-stock/{symbol}")
def grok_stock(symbol: str, force: bool = False):
   """Full structured Grok analysis for StockDetailScreen."""
   now = datetime.datetime.utcnow()
   key = f"grok_stock_{symbol.upper()}"


   # Cache
   if not force:
       item = cache.get(key)
       if item:
           age_hours = (now - item["time"]).total_seconds() / 3600
           if age_hours < GROK_STOCK_CACHE_HOURS:
               return {"text": item["text"], "updatedAt": item["time"].isoformat()}


   quote = backend_fetch_quote(symbol)


   price_context = (
       f"Current Price: {quote['current']}\n"
       f"Change: {quote['change']} ({quote['changePct']:.2f}%)\n"
       f"Day Range: {quote['low']} – {quote['high']}\n"
       f"Open: {quote['open']}\n"
       f"Prev Close: {quote['prevClose']}\n"
       f"Company: {quote['name']}\n"
       if quote
       else f"Symbol: {symbol.upper()}"
   )


   prompt = f"""
Analyze {symbol.upper()} using this structure:


AI Signal
Predictions
Executive Summary
Key Statistics
Technical Outlook
News & Market Sentiment
Risks & Opportunities
Trade Idea
Recommendation


Market Context:
{price_context}


Keep each section concise but meaningful. Include NFA disclaimer at end.
"""


   try:
       if not XAI_API_KEY:
           raise RuntimeError("Missing XAI_API_KEY")


       res = requests.post(
           "https://api.x.ai/v1/chat/completions",
           headers={"Authorization": f"Bearer {XAI_API_KEY}"},
           json={
               "model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.45,
               "max_tokens": 1500,
           },
           timeout=20,
       )
       j = res.json()
       text = (
           j.get("choices", [{}])[0]
           .get("message", {})
           .get("content", "")
           .strip()
       )


       if not text:
           text = "⚠️ AI analysis unavailable."


       cache[key] = {"text": text, "time": now}
       return {"text": text, "updatedAt": now.isoformat()}


   except Exception as e:
       print("GROK STOCK ERROR:", e)
       return {"text": "⚠️ AI analysis unavailable.", "updatedAt": None}




# ----------------------------------------------------------
# 11. Market News (for AIPulse / NewsFeedService)
# ----------------------------------------------------------
@app.get("/market-news")
def market_news():
   import feedparser


   FEEDS = [
       "https://www.benzinga.com/rss/stock-news.xml",
       "https://seekingalpha.com/api/sa/combined/global_news.rss",
       "https://feeds.marketwatch.com/marketwatch/topstories/",
       "https://www.investing.com/rss/news.rss",
       "https://www.zacks.com/rss/news.xml",
       "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,TSLA,MSFT,NVDA,META,AMZN,GOOGL,AMD,INTC,JPM,BAC,GS&region=US&lang=en-US",
   ]


   KEYWORDS = [
       "dow", "nasdaq", "s&p", "fed", "inflation", "cpi", "ppi",
       "earnings", "guidance", "profit", "loss", "upgrade", "downgrade",
       "ipo", "merger", "acquisition", "forecast", "ai", "chip",
       "market", "stock", "recession", "treasury", "jobs", "rate", "futures",
   ]


   news = []


   for url in FEEDS:
       try:
           feed = feedparser.parse(url)
           for e in feed.entries[:20]:
               title = getattr(e, "title", "")
               summary = getattr(e, "summary", "")
               link = getattr(e, "link", "")
               pub_date = getattr(e, "published", datetime.datetime.utcnow().isoformat())


               text = (title + summary).lower()
               if any(k in text for k in KEYWORDS):
                   news.append({
                       "title": title,
                       "summary": (summary or "")[:220] + "...",
                       "link": link,
                       "pubDate": pub_date,
                       "source": getattr(e, "source", {}).get("title", "News"),
                   })
       except Exception as ex:
           print("RSS error:", ex)


   # Deduplicate by first 40 chars
   seen = set()
   uniq = []
   for n in news:
       key = (n["title"] or "")[:40].lower()
       if key not in seen:
           seen.add(key)
           uniq.append(n)


   # Sort by pubDate where possible
   try:
       uniq.sort(
           key=lambda x: x["pubDate"],
           reverse=True,
       )
   except Exception:
       pass


   return {"data": uniq[:50]}


# ----------------------------------------------------------
# 12. SEARCH + WATCHLIST endpoints (for WatchlistScreen)
# ----------------------------------------------------------
def compute_signal_and_conf(change_pct: float):
   try:
       cp = float(change_pct or 0.0)
   except Exception:
       cp = 0.0


   if cp > 0.8:
       signal = "BUY"
   elif cp < -0.8:
       signal = "SELL"
   else:
       signal = "HOLD"


   confidence = min(95, max(70, abs(cp) * 10 + 70))
   return signal, int(round(confidence))


def build_watchlist_item(symbol: str):
   symbol = symbol.upper()
   q = backend_fetch_quote(symbol)
   if not q:
       return {
           "symbol": symbol,
           "price": 0.0,
           "changePct": 0.0,
           "signal": "HOLD",
           "confidence": 75,
           "sentimentSummary": "Live data temporarily unavailable; showing neutral placeholder.",
       }


   price = q.get("current") or q.get("price") or 0.0
   change_pct = q.get("changePct") or 0.0
   signal, confidence = compute_signal_and_conf(change_pct)


   # Grok-style short sentiment line with backend cache
   now = datetime.datetime.utcnow()
   cache_key = f"watch_grok_{symbol}"
   summary = None


   # Try cache
   item = cache.get(cache_key)
   if item:
       age_hours = (now - item["time"]).total_seconds() / 3600
       if age_hours < WATCH_GROK_CACHE_HOURS:
           summary = item["text"]


   # If no cached summary and Grok key is present, call Grok
   if not summary and XAI_API_KEY:
       try:
           prompt = (
               f"In one concise line (max 15 words), describe {symbol}'s "
               f"market trend given a daily move of {change_pct:.2f}%."
           )
           res = requests.post(
               "https://api.x.ai/v1/chat/completions",
               headers={"Authorization": f"Bearer {XAI_API_KEY}"},
               json={
                   "model": "grok-beta",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 40,
                   "temperature": 0.6,
               },
               timeout=12,
           )
           j = res.json()
           text = (
               j.get("choices", [{}])[0]
               .get("message", {})
               .get("content", "")
               .strip()
           )
           if text:
               summary = text
               cache[cache_key] = {"text": summary, "time": now}
       except Exception as e:
           print("Watchlist Grok error:", e)


   # Fallback summaries if Grok not available / empty
   if not summary:
       if signal == "BUY":
           summary = "Strong bullish activity detected."
       elif signal == "SELL":
           summary = "Bearish pressure observed."
       else:
           summary = "Market appears neutral."


   try:
       price_val = float(price)
   except Exception:
       price_val = 0.0


   try:
       cp_val = float(change_pct)
   except Exception:
       cp_val = 0.0


   return {
       "symbol": symbol,
       "price": round(price_val, 2),
       "changePct": round(cp_val, 2),
       "signal": signal,
       "confidence": confidence,
       "sentimentSummary": summary,
   }




@app.get("/search")
def search(q: str, limit: int = 5):
   """Autocomplete for tickers (used by WatchlistScreen)."""
   try:
       if not FINNHUB_KEY:
           return {"data": []}


       url = f"https://finnhub.io/api/v1/search?q={q}&token={FINNHUB_KEY}"
       data = requests.get(url, timeout=8).json()
       out = []
       for item in data.get("result", [])[:limit]:
           sym = item.get("symbol")
           desc = item.get("description")
           if sym and desc:
               out.append({"symbol": sym, "description": desc})
       return {"data": out}
   except Exception as e:
       print("SEARCH error:", e)
       return {"data": []}




@app.get("/watchlist-item/{symbol}")
def watchlist_item(symbol: str):
   """Single watchlist item data for a ticker."""
   try:
       return build_watchlist_item(symbol)
   except Exception as e:
       return {"error": str(e)}




@app.get("/watchlist-batch")
def watchlist_batch(symbols: str = Query(..., description="Comma-separated tickers")):
   """Optional batch endpoint if needed later."""
   try:
       sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
       data = [build_watchlist_item(s) for s in sym_list]
       return {"data": data}
   except Exception as e:
       return {"error": str(e)}



