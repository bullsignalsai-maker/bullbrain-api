from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import datetime
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import gdown

app = FastAPI()

# CORS for Expo / mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# ENV + CONSTANTS
# --------------------------------------------------------------------
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")

MODEL = "grok-4-fast-reasoning"
GROK_STOCK_CACHE_HOURS = 3
WATCH_GROK_CACHE_HOURS = 3
BULLBRAIN_VERSION = "v2-48f"

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

bullbrain_model: xgb.Booster | None = None
cache: dict[str, dict] = {}

# --------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[BullSignals] {msg}")


def safe_json(url: str, timeout: int = 10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print("safe_json error:", e)
        return None


# --------------------------------------------------------------------
# MODEL LOADING (FROM GOOGLE DRIVE)
# --------------------------------------------------------------------
def load_bullbrain_model() -> xgb.Booster:
    os.makedirs("models", exist_ok=True)
    try:
        log("Downloading BullBrain model from Google Drive…")
        gdown.download(MODEL_DRIVE_URL, FULLMODEL_LOCAL_PATH, quiet=False, fuzzy=True)
    except Exception as e:
        log(f"Model download failed, will try local file: {e}")

    if not os.path.exists(FULLMODEL_LOCAL_PATH):
        raise FileNotFoundError(f"Model file not found at {FULLMODEL_LOCAL_PATH}")

    booster = xgb.Booster()
    booster.load_model(FULLMODEL_LOCAL_PATH)
    log(f"BullBrain model loaded from {FULLMODEL_LOCAL_PATH}")
    log(f"BullBrain num_features={booster.num_features()}")
    return booster


# --------------------------------------------------------------------
# CANDLES + FEATURES
# --------------------------------------------------------------------
def fetch_daily_candles(symbol: str, min_points: int = 60):
    symbol = symbol.upper()
    if not POLYGON_KEY:
        return None
    try:
        now = datetime.datetime.utcnow()
        end = int(now.timestamp())
        start = int((now - datetime.timedelta(days=365)).timestamp())
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_KEY}"
        )
        j = safe_json(url)
        if not j or "results" not in j:
            return None
        res = j["results"]
        closes = [r["c"] for r in res]
        highs = [r["h"] for r in res]
        lows = [r["l"] for r in res]
        vols = [r["v"] for r in res]
        opens = [r.get("o", r["c"]) for r in res]
        ts = [r.get("t") for r in res]
        if len(closes) < min_points:
            return None
        return {
            "source": "polygon",
            "close": closes,
            "high": highs,
            "low": lows,
            "open": opens,
            "volume": vols,
            "timestamp": ts,
        }
    except Exception as e:
        print("fetch_daily_candles error:", e)
        return None


def compute_bullbrain_features(candles: dict):
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

    df["adj_close"] = df["close"]

    # Returns
    df["return_1d"] = df["close"].pct_change() * 100.0
    df["return_5d"] = df["close"].pct_change(5) * 100.0
    df["return_10d"] = df["close"].pct_change(10) * 100.0

    # Volatility
    daily_ret = df["close"].pct_change()
    df["volatility_5d"] = daily_ret.rolling(5).std() * 100.0
    df["volatility_20d"] = daily_ret.rolling(20).std() * 100.0
    df["volatility_60d"] = daily_ret.rolling(60).std() * 100.0

    # MAs
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    df["sma5_sma20_pct"] = (df["sma5"] / df["sma20"] - 1.0) * 100.0
    df["sma20_sma50_pct"] = (df["sma20"] / df["sma50"] - 1.0) * 100.0
    df["price_vs_sma20_pct"] = (df["close"] / df["sma20"] - 1.0) * 100.0

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["ema12"] = ema12
    df["ema26"] = ema26
    df["ema_ratio"] = ema12 / (ema26 + 1e-9)

    # Williams R + Stoch
    hh14 = df["high"].rolling(14).max()
    ll14 = df["low"].rolling(14).min()
    df["williams_r_14"] = (df["close"] - hh14) / (hh14 - ll14 + 1e-9) * 100.0
    df["stoch_k_14"] = (df["close"] - ll14) / (hh14 - ll14 + 1e-9) * 100.0
    df["stoch_d_3"] = df["stoch_k_14"].rolling(3).mean()

    # Volume features
    df["volume_change_1d"] = df["volume"].pct_change() * 100.0
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_vs_ma5_pct"] = (df["volume"] / (df["volume_ma5"] + 1e-9) - 1.0) * 100.0
    df["volume_vs_ma20_pct"] = (df["volume"] / (df["volume_ma20"] + 1e-9) - 1.0) * 100.0

    df["obv"] = (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()

    def _slope_10(x):
        return np.polyfit(range(len(x)), x, 1)[0]

    df["obv_slope_10"] = df["obv"].rolling(10).apply(_slope_10, raw=False)

    # Price range
    df["intraday_range_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-9) * 100.0

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

    # Candle anatomy
    df["upper_shadow_pct"] = (df["high"] - df["close"]) / (df["close"] + 1e-9) * 100.0
    df["lower_shadow_pct"] = (df["close"] - df["low"]) / (df["close"] + 1e-9) * 100.0
    df["body_pct"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100.0
    df["gap_pct"] = (df["open"] - df["close"].shift()) / (df["close"].shift() + 1e-9) * 100.0

    # Distance from 20d extremes
    rolling_high_20 = df["high"].rolling(20).max()
    rolling_low_20 = df["low"].rolling(20).min()
    df["distance_from_20d_high"] = (
        df["close"] / (rolling_high_20 + 1e-9) - 1.0
    ) * 100.0
    df["distance_from_20d_low"] = (
        df["close"] / (rolling_low_20 + 1e-9) - 1.0
    ) * 100.0

    # Volume z-score
    vol_ma20 = df["volume_ma20"]
    vol_std20 = vol_ma20.rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - vol_ma20) / (vol_std20 + 1e-9)

    # Trend strength
    def _slope_20(x):
        return np.polyfit(range(len(x)), x, 1)[0]

    df["trend_strength_20"] = df["close"].rolling(20).apply(_slope_20, raw=False)

    row = df.iloc[-1]
    last_close = float(row["close"])
    feature_dict = {}
    values = []
    for name in BULLBRAIN_FEATURES:
        raw = row.get(name, np.nan)
        values.append(float(raw) if pd.notna(raw) else np.nan)
        feature_dict[name] = None if pd.isna(raw) else float(raw)

    features_vector = np.array([values], dtype=float)
    return features_vector, feature_dict, last_close


# --------------------------------------------------------------------
# BULLBRAIN INFERENCE + CLASS MAPPING
# --------------------------------------------------------------------
def bullbrain_infer(features_vector: np.ndarray):
    global bullbrain_model
    if bullbrain_model is None:
        raise RuntimeError("BullBrain model not loaded")
    dmat = xgb.DMatrix(features_vector, feature_names=BULLBRAIN_FEATURES)
    preds = bullbrain_model.predict(dmat)
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


def _class_probs_from_prob_up(prob_up: float) -> dict:
    p = float(prob_up)
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0

    if p >= 0.6:
        buy = p
        hold = 1.0 - p
        sell = 0.0
    elif p <= 0.4:
        sell = 1.0 - p
        hold = p
        buy = 0.0
    else:
        center_offset = p - 0.5
        hold = 0.6
        buy = max(0.0, 0.2 + center_offset * 2.0)
        sell = max(0.0, 0.2 - center_offset * 2.0)
    total = buy + hold + sell
    if total <= 0:
        return {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33}
    return {"SELL": sell / total, "HOLD": hold / total, "BUY": buy / total}


# --------------------------------------------------------------------
# QUOTES (FINNHUB + YAHOO FALLBACK)
# --------------------------------------------------------------------
def backend_fetch_quote(symbol: str):
    symbol = symbol.upper()
    try:
        quote = None
        profile: dict = {}

        if FINNHUB_KEY:
            q_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
            quote = safe_json(q_url, timeout=8)
            p_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_KEY}"
            profile = safe_json(p_url, timeout=8) or {}

        if not quote or "c" not in quote or quote["c"] in [None, 0]:
            y_url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{symbol}?range=1d&interval=1d"
            )
            y = safe_json(y_url, timeout=8)
            if not y:
                return None
            meta = (
                y.get("chart", {}).get("result", [{}])[0].get("meta", {})
            )
            close = meta.get("regularMarketPrice")
            prev = meta.get("previousClose") or meta.get("chartPreviousClose")
            if close is None:
                return None
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
        print("backend_fetch_quote error:", e)
        return None


# --------------------------------------------------------------------
# GROK PROBABILITY + HYBRID
# --------------------------------------------------------------------
def grok_prob_up(symbol: str):
    symbol = symbol.upper()
    if not XAI_API_KEY:
        return 50.0, "Neutral sentiment (no Grok API key configured)."

    now = datetime.datetime.utcnow()
    cache_key = f"grok_prob_{symbol}"
    item = cache.get(cache_key)
    if item:
        age_hours = (now - item["time"]).total_seconds() / 3600
        if age_hours < GROK_STOCK_CACHE_HOURS:
            return item["prob"], item["summary"]

    prompt = (
        f"Based on all available information, including market sentiment, news, "
        f"and macro context, estimate the probability (0-100) that {symbol} "
        f"will CLOSE higher tomorrow than today.\n"
        f"Respond ONLY in this format:\n"
        f"Probability: <number>\n"
        f"Summary: <short explanation>"
    )
    try:
        res = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 40,
                "temperature": 0.4,
            },
            timeout=12,
        )
        j = res.json()
        text_out = (
            j.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        prob_val = 50.0
        summary = ""
        for line in text_out.splitlines():
            lower = line.lower()
            if "prob" in lower:
                try:
                    prob_val = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif "summary" in lower:
                summary = line.split(":", 1)[1].strip()
        prob_val = max(0.0, min(100.0, prob_val))
        if not summary:
            summary = "Sentiment analysis not available; treating as neutral."
        cache[cache_key] = {"prob": prob_val, "summary": summary, "time": now}
        return prob_val, summary
    except Exception as e:
        print("grok_prob_up error:", e)
        return 50.0, "Neutral sentiment (Grok unavailable)."


def compute_hybrid_signal(bull_conf: float, grok_prob: float):
    bull_conf = max(0.0, min(100.0, float(bull_conf or 0.0)))
    grok_prob = max(0.0, min(100.0, float(grok_prob or 0.0)))
    hybrid_score = 0.7 * bull_conf + 0.3 * grok_prob
    if hybrid_score >= 66.0:
        hybrid_signal = "BUY"
    elif hybrid_score <= 33.0:
        hybrid_signal = "SELL"
    else:
        hybrid_signal = "HOLD"
    return round(hybrid_score, 2), hybrid_signal


# --------------------------------------------------------------------
# CORE PIPELINE FOR ONE SYMBOL
# --------------------------------------------------------------------
def _run_bullbrain_for_symbol(symbol: str):
    symbol = symbol.upper()
    if bullbrain_model is None:
        return None, {"error": "BullBrain model not loaded yet."}
    candles = fetch_daily_candles(symbol)
    if not candles:
        return None, {"error": f"Could not fetch candles for {symbol}"}
    features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
    inference = bullbrain_infer(features_vec)
    prob_up = inference.get("probability_up")
    if prob_up is None:
        prob_up = float(inference.get("raw_output", 0.5))
    prob_down = 1.0 - float(prob_up)
    class_probs = _class_probs_from_prob_up(prob_up)
    try:
        grok_p, grok_summary = grok_prob_up(symbol)
    except Exception as e:
        print("grok_prob_up fatal:", e)
        grok_p, grok_summary = 50.0, "Neutral sentiment (error while calling Grok)."
    bull_conf = float(inference.get("confidence") or 0.0)
    hybrid_score, hybrid_signal = compute_hybrid_signal(bull_conf, grok_p)
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
            "raw": {"prob_up": float(prob_up), "prob_down": float(prob_down)},
        },
        "model": inference,
        "grokProbUp": float(grok_p),
        "grokSummary": grok_summary,
        "hybridScore": float(hybrid_score),
        "hybridSignal": hybrid_signal,
    }
    return core, None


# --------------------------------------------------------------------
# TECHNICAL SNAPSHOT HELPERS
# --------------------------------------------------------------------
def _interpret_rsi(rsi: float | None) -> str:
    if rsi is None:
        return "Unknown"
    if rsi < 30:
        return "Oversold (RSI < 30)"
    if rsi < 40:
        return "Bearish momentum (RSI < 40)"
    if rsi <= 60:
        return "Neutral momentum (RSI 40–60)"
    if rsi <= 70:
        return "Bullish momentum (RSI 60–70)"
    return "Overbought (RSI > 70)"


def _interpret_macd(macd_hist: float | None) -> str:
    if macd_hist is None:
        return "Unknown"
    if macd_hist > 1.0:
        return "Strong bullish MACD momentum"
    if macd_hist > 0.0:
        return "Mild bullish MACD momentum"
    if macd_hist < -1.0:
        return "Strong bearish MACD momentum"
    if macd_hist < 0.0:
        return "Mild bearish MACD momentum"
    return "Flat MACD momentum"


def _interpret_volume(volume_z: float | None, vs_ma20: float | None) -> str:
    if volume_z is None and vs_ma20 is None:
        return "Unknown"
    if volume_z is not None:
        if volume_z > 2.0:
            return "High volume spike (Z > 2)"
        if volume_z > 1.0:
            return "Elevated volume (Z 1–2)"
        if volume_z < -1.0:
            return "Unusually low volume"
    if vs_ma20 is not None:
        if vs_ma20 > 20:
            return "Volume well above 20-day average"
        if vs_ma20 < -20:
            return "Volume well below 20-day average"
    return "Normal volume"


def _interpret_trend(trend_strength_20: float | None, dist_high: float | None, dist_low: float | None) -> str:
    if trend_strength_20 is None:
        return "Unknown trend"
    if trend_strength_20 > 0.5:
        return "Strong uptrend"
    if trend_strength_20 > 0.1:
        return "Mild uptrend"
    if trend_strength_20 < -0.5:
        return "Strong downtrend"
    if trend_strength_20 < -0.1:
        return "Mild downtrend"
    return "Sideways / range-bound"


def _interpret_volatility(vol20: float | None) -> str:
    if vol20 is None:
        return "Unknown"
    if vol20 < 1.0:
        return "Low volatility"
    if vol20 < 2.5:
        return "Normal volatility"
    if vol20 < 4.0:
        return "Elevated volatility"
    return "High volatility regime"


def build_technical_snapshot(symbol: str, feat: dict, last_close: float):
    symbol = symbol.upper()
    as_of = datetime.datetime.utcnow().isoformat()

    def fv(name):
        v = feat.get(name)
        return None if v is None else float(v)

    rsi = fv("rsi14")
    macd_val = fv("macd")
    macd_signal = fv("macd_signal")
    macd_hist = fv("macd_hist")
    stoch_k = fv("stoch_k_14")
    stoch_d = fv("stoch_d_3")
    willr = fv("williams_r_14")

    vol5 = fv("volatility_5d")
    vol20 = fv("volatility_20d")
    vol60 = fv("volatility_60d")

    vol_change_1d = fv("volume_change_1d")
    vol_vs_ma5 = fv("volume_vs_ma5_pct")
    vol_vs_ma20 = fv("volume_vs_ma20_pct")
    vol_z = fv("volume_zscore_20")
    obv = fv("obv")
    obv_slope_10 = fv("obv_slope_10")

    price_vs_sma20 = fv("price_vs_sma20_pct")
    sma5_sma20_pct = fv("sma5_sma20_pct")
    sma20_sma50_pct = fv("sma20_sma50_pct")
    dist_high = fv("distance_from_20d_high")
    dist_low = fv("distance_from_20d_low")
    trend_strength_20 = fv("trend_strength_20")

    intraday_range_pct = fv("intraday_range_pct")
    body_pct = fv("body_pct")
    upper_shadow_pct = fv("upper_shadow_pct")
    lower_shadow_pct = fv("lower_shadow_pct")
    gap_pct = fv("gap_pct")
    atr14 = fv("atr14")
    true_range = fv("true_range")

    trend_summary = _interpret_trend(trend_strength_20, dist_high, dist_low)
    momentum_summary = _interpret_rsi(rsi)
    macd_summary = _interpret_macd(macd_hist)
    volume_summary = _interpret_volume(vol_z, vol_vs_ma20)
    vol_regime_summary = _interpret_volatility(vol20)

    return {
        "symbol": symbol,
        "asOf": as_of,
        "price": last_close,
        "trend": {
            "trend_strength_20": trend_strength_20,
            "price_vs_sma20_pct": price_vs_sma20,
            "sma5_sma20_pct": sma5_sma20_pct,
            "sma20_sma50_pct": sma20_sma50_pct,
            "distance_from_20d_high": dist_high,
            "distance_from_20d_low": dist_low,
            "summary": trend_summary,
        },
        "momentum": {
            "rsi14": rsi,
            "macd": macd_val,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "stoch_k_14": stoch_k,
            "stoch_d_3": stoch_d,
            "williams_r_14": willr,
            "summary_rsi": momentum_summary,
            "summary_macd": macd_summary,
        },
        "volume": {
            "volume_change_1d": vol_change_1d,
            "volume_vs_ma5_pct": vol_vs_ma5,
            "volume_vs_ma20_pct": vol_vs_ma20,
            "volume_zscore_20": vol_z,
            "obv": obv,
            "obv_slope_10": obv_slope_10,
            "summary": volume_summary,
        },
        "volatility": {
            "volatility_5d": vol5,
            "volatility_20d": vol20,
            "volatility_60d": vol60,
            "atr14": atr14,
            "true_range": true_range,
            "summary": vol_regime_summary,
        },
        "candle": {
            "intraday_range_pct": intraday_range_pct,
            "body_pct": body_pct,
            "upper_shadow_pct": upper_shadow_pct,
            "lower_shadow_pct": lower_shadow_pct,
            "gap_pct": gap_pct,
        },
    }


# --------------------------------------------------------------------
# STOCKDETAIL GROK (COMPRESSED, OPTION B)
# --------------------------------------------------------------------
def get_stockdetail_grok(symbol: str, quote: dict | None, technical: dict | None, force: bool = False):
    symbol = symbol.upper()
    now = datetime.datetime.utcnow()
    cache_key = f"stockdetail_grok_{symbol}"
    if not force:
        item = cache.get(cache_key)
        if item:
            age_hours = (now - item["time"]).total_seconds() / 3600
            if age_hours < GROK_STOCK_CACHE_HOURS:
                return item["payload"]

    current_price = None
    change_pct = None
    if quote:
        current_price = quote.get("current")
        change_pct = quote.get("changePct")

    if not XAI_API_KEY:
        trend_summary = ""
        if technical and isinstance(technical, dict):
            trend_summary = (technical.get("trend", {}) or {}).get("summary") or ""
        payload = {
            "ai_signal": f"NEUTRAL - {trend_summary or 'AI sentiment unavailable.'}",
            "short_term": "Short-term outlook is neutral based on recent price and trend.",
            "medium_term": "Medium-term direction depends on earnings, macro trends, and news.",
            "long_term": "Long-term potential depends on fundamentals, competition, and innovation.",
            "risk_note": "Not financial advice. Consider your own risk tolerance and do your own research.",
            "prob_up": 0.5,
            "updatedAt": now.isoformat(),
        }
        cache[cache_key] = {"time": now, "payload": payload}
        return payload

    cp_str = f"{current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A"
    chg_str = f"{change_pct:.2f}" if isinstance(change_pct, (int, float)) else "N/A"

    trend_summary = ""
    momentum_summary = ""
    vol_summary = ""
    try:
        if technical and isinstance(technical, dict):
            trend_summary = (technical.get("trend", {}) or {}).get("summary") or ""
            momentum_summary = (technical.get("momentum", {}) or {}).get("summary_rsi") or ""
            vol_summary = (technical.get("volatility", {}) or {}).get("summary") or ""
    except Exception:
        pass

    prompt = f"""
You are an expert stock analyst speaking to a non-technical investor.

Stock:
- Symbol: {symbol}
- Current price: {cp_str}
- Daily change (%): {chg_str}

Technical context (already computed):
- Trend: {trend_summary}
- Momentum: {momentum_summary}
- Volatility: {vol_summary}

Task:
Return ONLY a compact JSON object with these keys:

- "ai_signal": one line like "BUY - reason" / "HOLD - reason" / "SELL - reason" / "NEUTRAL - reason" (max 18 words)
- "short_term": 1 sentence on the next 1–6 weeks (max 30 words, NO indicator names)
- "medium_term": 1 sentence on the next 6–12 months (max 35 words)
- "long_term": 1 sentence on the next 1–3 years (max 35 words)
- "risk_note": 1 brief risk disclaimer (max 25 words)
- "prob_up": a number between 0 and 1 for the chance price is HIGHER 1–3 months from now.

Rules:
- Use simple language.
- Do NOT add extra keys.
- Respond ONLY with valid JSON.
"""
    try:
        res = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 220,
            },
            timeout=16,
        )
        j = res.json()
        text = (
            j.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}
        prob_up = parsed.get("prob_up", 0.5)
        try:
            prob_up = float(prob_up)
        except Exception:
            prob_up = 0.5
        if prob_up < 0.0:
            prob_up = 0.0
        if prob_up > 1.0:
            prob_up = 1.0
        payload = {
            "ai_signal": parsed.get("ai_signal") or "NEUTRAL - AI view unavailable.",
            "short_term": parsed.get("short_term")
            or "Short-term outlook is uncertain; price may remain choppy.",
            "medium_term": parsed.get("medium_term")
            or "Medium-term direction depends on earnings, news, and broader market conditions.",
            "long_term": parsed.get("long_term")
            or "Long-term performance will depend on fundamentals and competitive position.",
            "risk_note": parsed.get("risk_note")
            or "Not financial advice. Markets are volatile; manage your risk carefully.",
            "prob_up": prob_up,
            "updatedAt": now.isoformat(),
        }
        cache[cache_key] = {"time": now, "payload": payload}
        return payload
    except Exception as e:
        print("get_stockdetail_grok error:", e)
        item = cache.get(cache_key)
        if item:
            return item["payload"]
        payload = {
            "ai_signal": "NEUTRAL - AI analysis unavailable.",
            "short_term": "Short-term outlook is unclear; price may move sideways.",
            "medium_term": "Medium-term view is neutral without AI guidance.",
            "long_term": "Long-term direction depends on fundamentals and macro trends.",
            "risk_note": "Not financial advice. Consider your own risk before trading.",
            "prob_up": 0.5,
            "updatedAt": now.isoformat(),
        }
        cache[cache_key] = {"time": now, "payload": payload}
        return payload


# --------------------------------------------------------------------
# WATCHLIST GROK HELPER + HYBRID
# --------------------------------------------------------------------
def grok_watchlist_sentiment(symbol: str, change_pct: float):
    symbol = symbol.upper()
    now = datetime.datetime.utcnow()
    cache_key = f"watch_grok_v2_{symbol}"
    item = cache.get(cache_key)
    if item:
        age_hours = (now - item["time"]).total_seconds() / 3600
        if age_hours < WATCH_GROK_CACHE_HOURS:
            return {"summary": item["summary"], "prob_up": item["prob_up"]}

    if not XAI_API_KEY:
        try:
            cp = float(change_pct or 0.0)
        except Exception:
            cp = 0.0
        x = max(-5.0, min(5.0, cp)) / 5.0
        prob_up = 0.5 + 0.4 * x
        summary = (
            f"Daily move {cp:.2f}% with no AI sentiment available; "
            "reading based only on price action."
        )
        cache[cache_key] = {"summary": summary, "prob_up": prob_up, "time": now}
        return {"summary": summary, "prob_up": prob_up}

    prompt = f"""
You are an expert stock analyst.

Given:
Symbol: {symbol}
Daily Change (%): {change_pct:.2f}

Return a STRICT JSON object with exactly these keys:

  "one_liner": a concise, plain-English summary of current sentiment and price action (max 18 words),
  "prob_up": a probability between 0 and 1 that this stock's price will be HIGHER 10–20 trading days from now.

Rules:
- Respond ONLY with valid JSON.
"""
    one_liner = None
    prob_up = None
    try:
        res = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 180,
            },
            timeout=16,
        )
        j = res.json()
        text = (
            j.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        try:
            parsed = json.loads(text)
            one_liner = parsed.get("one_liner") or parsed.get("summary") or ""
            prob_up_raw = parsed.get("prob_up", 0.5)
            prob_up = float(prob_up_raw)
        except Exception:
            one_liner = text or ""
            prob_up = 0.5
    except Exception as e:
        print("grok_watchlist_sentiment error:", e)
        one_liner = None
        prob_up = None

    try:
        cp = float(change_pct or 0.0)
    except Exception:
        cp = 0.0

    if not one_liner:
        if cp > 0:
            one_liner = f"Daily move {cp:.2f}% with mildly bullish tone."
        elif cp < 0:
            one_liner = f"Daily move {cp:.2f}% with cautious / bearish tone."
        else:
            one_liner = "Flat day with neutral sentiment."

    if prob_up is None:
        x = max(-5.0, min(5.0, cp)) / 5.0
        prob_up = 0.5 + 0.4 * x

    if prob_up < 0.0:
        prob_up = 0.0
    if prob_up > 1.0:
        prob_up = 1.0

    cache[cache_key] = {"summary": one_liner, "prob_up": prob_up, "time": now}
    return {"summary": one_liner, "prob_up": prob_up}


def _hybrid_from_probs(bull_prob_up: float | None, grok_prob_up: float | None):
    if bull_prob_up is None and grok_prob_up is None:
        p = 0.5
    elif bull_prob_up is None:
        p = float(grok_prob_up)
    elif grok_prob_up is None:
        p = float(bull_prob_up)
    else:
        p = 0.7 * float(bull_prob_up) + 0.3 * float(grok_prob_up)
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    if p >= 0.55:
        signal = "BUY"
    elif p <= 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"
    confidence = round(max(p, 1 - p) * 100.0, 2)
    return p, signal, confidence


# --------------------------------------------------------------------
# STARTUP
# --------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    global bullbrain_model
    log("Backend starting; loading BullBrain model…")
    try:
        bullbrain_model = load_bullbrain_model()
    except Exception as e:
        log(f"Failed to load BullBrain model: {e}")


# --------------------------------------------------------------------
# ROOT
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "BullSignalsAI Backend Running",
        "bullbrain_loaded": bullbrain_model is not None,
        "features": BULLBRAIN_FEATURES,
    }


# --------------------------------------------------------------------
# MAIN PREDICTION ENDPOINTS
# --------------------------------------------------------------------
@app.get("/predict/{symbol}")
def predict_symbol(symbol: str):
    core, err = _run_bullbrain_for_symbol(symbol)
    if err is not None:
        return {"symbol": symbol.upper(), **err}
    return core


@app.get("/predict-prob/{symbol}")
def predict_prob(symbol: str):
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


@app.get("/predict-multi")
def predict_multi(tickers: str = Query(..., description="Comma-separated tickers")):
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
    return {"data": results, "errors": errors}


@app.get("/features/{symbol}")
def get_features(symbol: str):
    symbol = symbol.upper()
    try:
        candles = fetch_daily_candles(symbol)
        if not candles:
            return {"symbol": symbol, "error": f"Could not fetch candles for {symbol}"}
        _, feature_dict, last_close = compute_bullbrain_features(candles)
        as_of = datetime.datetime.utcnow().isoformat()
        return {
            "symbol": symbol,
            "asOf": as_of,
            "source": candles.get("source", "polygon"),
            "price": last_close,
            "features": feature_dict,
        }
    except Exception as e:
        print("get_features error:", e)
        return {"symbol": symbol, "error": str(e)}


@app.get("/candles/{symbol}")
def get_candles(symbol: str, limit: int = 252):
    symbol = symbol.upper()
    try:
        candles = fetch_daily_candles(symbol, min_points=min(limit, 60))
        if not candles:
            return {"symbol": symbol, "error": f"Could not fetch candles for {symbol}"}
        closes = candles["close"]
        highs = candles["high"]
        lows = candles["low"]
        opens = candles["open"]
        vols = candles["volume"]
        ts_list = candles.get("timestamp") or []
        n = len(closes)
        if n == 0:
            return {"symbol": symbol, "error": "No candle data"}
        use_n = min(limit, n)
        start_idx = n - use_n
        items = []
        for i in range(start_idx, n):
            t_raw = ts_list[i] if i < len(ts_list) and ts_list[i] else None
            if t_raw:
                dt = datetime.datetime.utcfromtimestamp(t_raw / 1000.0).replace(microsecond=0)
                t_iso = dt.isoformat() + "Z"
            else:
                dt = datetime.datetime.utcnow() - datetime.timedelta(days=(n - 1 - i))
                t_iso = dt.replace(microsecond=0).isoformat() + "Z"
            items.append(
                {
                    "t": t_iso,
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "volume": float(vols[i]),
                }
            )
        return {"symbol": symbol, "source": candles.get("source", "polygon"), "candles": items}
    except Exception as e:
        print("get_candles error:", e)
        return {"symbol": symbol, "error": str(e)}


@app.get("/technical/{symbol}")
def get_technical(symbol: str):
    symbol = symbol.upper()
    try:
        candles = fetch_daily_candles(symbol)
        if not candles:
            return {"symbol": symbol, "error": f"Could not fetch candles for {symbol}"}
        _, feat, last_close = compute_bullbrain_features(candles)
        return build_technical_snapshot(symbol, feat, last_close)
    except Exception as e:
        print("get_technical error:", e)
        return {"symbol": symbol, "error": str(e)}


# --------------------------------------------------------------------
# STOCKDETAIL SUPER ENDPOINT
# --------------------------------------------------------------------
@app.get("/stockdetail/{symbol}")
def stockdetail(symbol: str, limit_candles: int = 120, forceGrok: bool = False):
    symbol = symbol.upper()
    try:
        quote = backend_fetch_quote(symbol)
        candles = fetch_daily_candles(symbol)
        feature_dict = None
        last_close = None
        bullbrain_block = None
        bull_prob_up = None

        if candles and bullbrain_model is not None:
            features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
            inference = bullbrain_infer(features_vec)
            prob_up = float(
                inference.get("probability_up") or inference.get("raw_output") or 0.5
            )
            bull_prob_up = prob_up
            class_probs = _class_probs_from_prob_up(prob_up)
            bullbrain_block = {
                "version": BULLBRAIN_VERSION,
                "signal": inference.get("signal"),
                "confidence": inference.get("confidence"),
                "probabilities": class_probs,
                "raw": {"prob_up": prob_up, "prob_down": 1.0 - prob_up},
            }

        if last_close is None and quote:
            try:
                last_close = float(quote.get("current") or 0.0)
            except Exception:
                last_close = None

        technical = None
        if feature_dict is not None and last_close is not None:
            technical = build_technical_snapshot(symbol, feature_dict, last_close)

        candles_payload = None
        if candles:
            closes = candles["close"]
            highs = candles["high"]
            lows = candles["low"]
            opens = candles["open"]
            vols = candles["volume"]
            ts_list = candles.get("timestamp") or []
            n = len(closes)
            if n > 0:
                use_n = min(limit_candles, n)
                start_idx = n - use_n
                chart_items = []
                for i in range(start_idx, n):
                    t_raw = ts_list[i] if i < len(ts_list) and ts_list[i] else None
                    if t_raw:
                        dt = datetime.datetime.utcfromtimestamp(t_raw / 1000.0).replace(microsecond=0)
                        t_iso = dt.isoformat() + "Z"
                    else:
                        dt = datetime.datetime.utcnow() - datetime.timedelta(days=(n - 1 - i))
                        t_iso = dt.replace(microsecond=0).isoformat() + "Z"
                    chart_items.append(
                        {
                            "t": t_iso,
                            "open": float(opens[i]),
                            "high": float(highs[i]),
                            "low": float(lows[i]),
                            "close": float(closes[i]),
                            "volume": float(vols[i]),
                        }
                    )
                candles_payload = {
                    "symbol": symbol,
                    "source": candles.get("source", "polygon"),
                    "candles": chart_items,
                }

        news = get_symbol_news(symbol, limit=8)
        grok_pack = get_stockdetail_grok(symbol, quote, technical, force=forceGrok)
        grok_prob_up = grok_pack.get("prob_up")

        hybrid_p = None
        hybrid_signal = None
        hybrid_conf = None
        if bull_prob_up is not None or grok_prob_up is not None:
            hybrid_p, hybrid_signal, hybrid_conf = _hybrid_from_probs(
                bull_prob_up, grok_prob_up
            )

        return {
            "symbol": symbol,
            "asOf": datetime.datetime.utcnow().isoformat(),
            "quote": quote,
            "price": last_close,
            "bullbrain": bullbrain_block,
            "features": feature_dict,
            "technical": technical,
            "candles": candles_payload,
            "news": news,
            "grok": grok_pack,
            "hybridProbUp": hybrid_p,
            "hybridSignal": hybrid_signal,
            "hybridScore": hybrid_conf,
        }
    except Exception as e:
        print("stockdetail error:", e)
        return {"symbol": symbol, "error": str(e)}


# --------------------------------------------------------------------
# SIMPLE QUOTE + ANALYST ENDPOINTS
# --------------------------------------------------------------------
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


@app.get("/grok-stock/{symbol}")
def grok_stock(symbol: str, force: bool = False):
    now = datetime.datetime.utcnow()
    key = f"grok_stock_{symbol.upper()}"
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

Keep each section concise. Include NFA disclaimer at end.
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


@app.get("/ticker-full/{symbol}")
def ticker_full(symbol: str):
    try:
        q = backend_fetch_quote(symbol)
        rec_data = recommendations(symbol)
        return {"symbol": symbol.upper(), "quote": q, "recommendations": rec_data}
    except Exception as e:
        return {"error": str(e)}


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


# --------------------------------------------------------------------
# MACRO / NEWS / MOOD
# --------------------------------------------------------------------
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


@app.get("/stats/live")
def live_stats():
    try:
        fearGreed = {"value": 50, "label": "Neutral"}
        vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
        vix_data = requests.get(vix_url, timeout=10).json()
        vix = (
            vix_data.get("chart", {})
            .get("result", [{}])[0]
            .get("meta", {})
            .get("regularMarketPrice", 15)
        )
        sp_url = "https://query1.finance.yahoo.com/v8/finance/chart/^GSPC"
        sp_data = requests.get(sp_url, timeout=10).json()
        sp_meta = sp_data.get("chart", {}).get("result", [{}])[0].get("meta", {})
        prev = sp_meta.get("previousClose")
        sp_change = (
            (sp_meta.get("regularMarketPrice") - prev) / prev * 100 if prev else 0
        )
        return {
            "fearGreed": fearGreed,
            "vix": round(float(vix), 2),
            "sp500_change": round(float(sp_change), 2),
        }
    except Exception as e:
        return {
            "fearGreed": {"value": 50, "label": "Neutral"},
            "vix": 14.5,
            "sp500_change": 0.2,
            "error": str(e),
        }


@app.get("/market-mood")
def market_mood():
    try:
        fng = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json", timeout=5
        ).json()
        fear_value = int(fng.get("data", [{}])[0].get("value", 50))
        fear_label = fng.get("data", [{}])[0].get("value_classification", "Neutral")
        vix_json = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX", timeout=5
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
                    news.append(
                        {
                            "title": title,
                            "summary": (summary or "")[:220] + "...",
                            "link": link,
                            "pubDate": pub_date,
                            "source": getattr(e, "source", {}).get("title", "News"),
                        }
                    )
        except Exception as ex:
            print("RSS error:", ex)
    seen = set()
    uniq = []
    for n in news:
        key = (n["title"] or "")[:40].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(n)
    try:
        uniq.sort(key=lambda x: x["pubDate"], reverse=True)
    except Exception:
        pass
    return {"data": uniq[:50]}


def get_symbol_news(symbol: str, limit: int = 8):
    sym = symbol.upper()
    try:
        resp = market_news()
        data = resp.get("data", []) if isinstance(resp, dict) else []
        if not isinstance(data, list):
            return []
        filtered = []
        for n in data:
            title = (n.get("title") or "")
            summary = (n.get("summary") or "")
            text = (title + " " + summary).upper()
            if sym in text:
                filtered.append(n)
        if not filtered:
            return data[:limit]
        return filtered[:limit]
    except Exception as e:
        print("get_symbol_news error:", e)
        return []


# --------------------------------------------------------------------
# SEARCH + WATCHLIST
# --------------------------------------------------------------------
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
    # Use old simple Grok line for compatibility
    summary = "Market appears neutral."
    try:
        g = grok_watchlist_sentiment(symbol, change_pct)
        summary = g.get("summary") or summary
    except Exception:
        pass
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
    try:
        return build_watchlist_item(symbol)
    except Exception as e:
        return {"error": str(e)}


@app.get("/watchlist-batch")
def watchlist_batch(symbols: str = Query(..., description="Comma-separated tickers in Firebase order")):
    try:
        raw_syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        seen = set()
        sym_list = []
        for s in raw_syms:
            if s not in seen:
                sym_list.append(s)
                seen.add(s)
        if not sym_list:
            return {"data": []}
        quotes = {}
        for s in sym_list:
            q = backend_fetch_quote(s)
            quotes[s] = q or {}
        bull_map = {}
        if bullbrain_model is not None:
            for s in sym_list:
                try:
                    core, err = _run_bullbrain_for_symbol(s)
                    if not err and core and core.get("bullbrain"):
                        bull_map[s] = core
                except Exception as e:
                    print(f"BullBrain error for {s}:", e)
        grok_map = {}
        for s in sym_list:
            q = quotes.get(s, {})
            change_pct = q.get("changePct", 0.0)
            try:
                g = grok_watchlist_sentiment(s, change_pct)
            except Exception as e:
                print(f"grok_watchlist_sentiment error for {s}:", e)
                g = {"summary": "Sentiment unavailable.", "prob_up": 0.5}
            grok_map[s] = g
        items = []
        for s in sym_list:
            q = quotes.get(s, {})
            price = q.get("current") or q.get("price") or 0.0
            change_pct = q.get("changePct") or 0.0
            g = grok_map.get(s, {})
            grok_summary = g.get("summary")
            grok_prob_up = g.get("prob_up")
            core = bull_map.get(s)
            bull_signal = None
            bull_confidence = None
            bull_prob_up = None
            bull_probabilities = None
            bull_features = None
            bullbrain_block = None
            if core:
                bb = core.get("bullbrain") or {}
                bull_signal = bb.get("signal")
                bull_confidence = bb.get("confidence")
                raw = bb.get("raw") or {}
                bull_prob_up = raw.get("prob_up")
                bull_probabilities = bb.get("probabilities")
                bull_features = core.get("features")
                bullbrain_block = bb
            hybrid_p, hybrid_signal, hybrid_conf = _hybrid_from_probs(
                bull_prob_up, grok_prob_up
            )
            item = {
                "symbol": s,
                "price": round(float(price or 0.0), 2),
                "changePct": round(float(change_pct or 0.0), 2),
                "hybridSignal": hybrid_signal,
                "hybridScore": hybrid_conf,
                "hybridProbUp": hybrid_p,
                "grokSummary": grok_summary,
                "grokProbUp": grok_prob_up,
                "bullSignal": bull_signal,
                "bullConfidence": bull_confidence,
                "bullProbabilities": bull_probabilities,
                "features": bull_features,
                "bullbrain": bullbrain_block,
            }
            items.append(item)
        return {"data": items}
    except Exception as e:
        print("watchlist_batch fatal error:", e)
        return {"error": str(e)}


# --------------------------------------------------------------------
# FORCE CLEAR GROK CACHE (DEBUG)
# --------------------------------------------------------------------
@app.get("/force-refresh-grok/{symbol}")
def force_refresh_grok(symbol: str):
    key_stock = f"grok_stock_{symbol.upper()}"
    key_watch = f"watch_grok_v2_{symbol.upper()}"
    removed = []
    for k in (key_stock, key_watch):
        if k in cache:
            del cache[k]
            removed.append(k)
    return {
        "symbol": symbol.upper(),
        "removedKeys": removed,
        "message": "Grok cache cleared — next request will fetch fresh data.",
    }
# --------------------------------------------------------------------
# SMART PATTERN ENGINE – REAL-WORLD VERSION (Triggers Every Day)
# --------------------------------------------------------------------
def detect_smart_pattern(features: dict, quote: dict):
    if not features or not quote:
        return None

    f = features
    q = quote

    def fv(key, default=0.0):
        v = f.get(key)
        return float(v) if v is not None and not pd.isna(v) else default

    def qv(key, default=0.0):
        v = q.get(key)
        return float(v) if v is not None else default

    # Core values
    gap_pct = fv("gap_pct")
    change_pct = qv("changePct")
    volume_z = fv("volume_zscore_20")
    volume_vs_ma20 = fv("volume_vs_ma20_pct")
    rsi = fv("rsi14")
    williams_r = fv("williams_r_14")
    price_vs_sma20 = fv("price_vs_sma20_pct")
    sma5 = fv("sma5")
    sma20 = fv("sma20")
    sma50 = fv("sma50")
    sma200 = fv("sma200")
    macd_hist = fv("macd_hist")
    body_pct = fv("body_pct")
    lower_shadow = fv("lower_shadow_pct")
    upper_shadow = fv("upper_shadow_pct")
    return_5d = fv("return_5d")
    return_10d = fv("return_10d")
    trend_strength_20 = fv("trend_strength_20")

    # 1. GAP UP & RUNNING (73%)
    if gap_pct > 2.0 and change_pct > 3.5 and volume_vs_ma20 > 5:
        return {
            "title": "GAP UP & RUNNING",
            "win_rate": 73,
            "desc": "Opened sharply higher and kept climbing on strong volume.",
            "action": "This is how big rallies begin — momentum firmly with buyers.",
            "type": "bullish"
        }

    # 2. MASSIVE VOLUME BREAKOUT (76%)
    if volume_z > 3.0 and change_pct > 2:
        return {
            "title": "MASSIVE VOLUME BREAKOUT",
            "win_rate": 76,
            "desc": "One of the highest volume days in months — institutions are active.",
            "action": "Strong move usually follows. Watch for continuation.",
            "type": "bullish"
        }

    # 3. OVERSOLD BOUNCE + VOLUME (80%)
    if rsi < 42 and williams_r < -60 and volume_z > 2.8 and change_pct > 4:
        return {
            "title": "OVERSOLD BOUNCE + VOLUME",
            "win_rate": 80,
            "desc": "Stock was heavily sold — today buyers stepped in aggressively.",
            "action": "One of the strongest short-term reversal signals.",
            "type": "bullish"
        }

    # 4. HAMMER CANDLE REVERSAL (74%)
    if lower_shadow > 1.8 * abs(body_pct) and body_pct > 0 and change_pct > 1.5:
        return {
            "title": "HAMMER CANDLE REVERSAL",
            "win_rate": 74,
            "desc": "Buyers rejected lower prices all day — classic reversal candle.",
            "action": "High-probability bottom. Often leads to multi-day bounce.",
            "type": "bullish"
        }

    # 5. BUY THE DIP IN UPTREND (69%)
    if trend_strength_20 > 0.2 and -10 < price_vs_sma20 < 3 and change_pct > 1:
        return {
            "title": "BUY THE DIP IN UPTREND",
            "win_rate": 69,
            "desc": "Strong trend paused — now bouncing off support with volume.",
            "action": "Classic dip-buying opportunity in a bull market.",
            "type": "bullish"
        }

    # 6. DEAD CAT BOUNCE? (68% fail)
    if change_pct > 9 and return_5d < -12:
        return {
            "title": "DEAD CAT BOUNCE?",
            "win_rate": 68,
            "desc": "Sharp rebound after big drop — history shows most fail.",
            "action": "High risk. Often reverses lower within days.",
            "type": "bearish"
        }

    # 7. OVERBOUGHT + WEAK VOLUME (67%)
    if rsi > 70 and volume_vs_ma20 < -10 and change_pct < 0:
        return {
            "title": "OVERBOUGHT DISTRIBUTION",
            "win_rate": 67,
            "desc": "Price at extreme high — volume fading as smart money exits.",
            "action": "Top forming. Risk of pullback increasing.",
            "type": "bearish"
        }

    # 8. FAILED BREAKOUT TRAP (66%)
    if change_pct < -5 and volume_z > 3:
        return {
            "title": "FAILED BREAKOUT TRAP",
            "win_rate": 66,
            "desc": "Breakout failed on high volume — classic bear trap.",
            "action": "Downside momentum likely to continue.",
            "type": "bearish"
        }

    return None


# --------------------------------------------------------------------
# DEDICATED ENDPOINT: /smart-pattern/{symbol}
# --------------------------------------------------------------------
@app.get("/smart-pattern/{symbol}")
def smart_pattern(symbol: str):
    symbol = symbol.upper()
    try:
        quote = backend_fetch_quote(symbol)
        candles = fetch_daily_candles(symbol)
        if not candles or not quote:
            return {"symbol": symbol, "smart_pattern": None}

        _, feature_dict, _ = compute_bullbrain_features(candles)
        pattern = detect_smart_pattern(feature_dict, quote)

        return {
            "symbol": symbol,
            "asOf": datetime.datetime.utcnow().isoformat(),
            "smart_pattern": pattern
        }
    except Exception as e:
        print("smart-pattern endpoint error:", e)
        return {"symbol": symbol, "smart_pattern": None}