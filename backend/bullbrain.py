# backend/bullbrain.py
# ============================================================
# BullSignalsAI — BullBrain (Model Loading + Feature + Inference)
# ============================================================

from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import xgboost as xgb


# ------------------------------------------------------------
# Versioning
# ------------------------------------------------------------
BULLBRAIN_VERSION = os.getenv("BULLBRAIN_VERSION", "v2")


# ------------------------------------------------------------
# Model location (Drive -> local path)
# ------------------------------------------------------------
BULLBRAIN_MODEL_PATH = os.getenv("BULLBRAIN_MODEL_PATH", "/tmp/bullbrain_model.json")
BULLBRAIN_MODEL_DRIVE_URL = os.getenv("BULLBRAIN_MODEL_DRIVE_URL", "").strip()
BULLBRAIN_MODEL_GDRIVE_ID = os.getenv("BULLBRAIN_MODEL_GDRIVE_ID", "").strip()
BULLBRAIN_SKIP_DOWNLOAD = os.getenv("BULLBRAIN_SKIP_DOWNLOAD", "false").lower() == "true"


# ------------------------------------------------------------
# Feature order (MUST match training order)
# ------------------------------------------------------------
BULLBRAIN_FEATURES: List[str] = [
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


# ------------------------------------------------------------
# Global model handle
# ------------------------------------------------------------
bullbrain_model: Optional[xgb.Booster] = None


# ------------------------------------------------------------
# Download helper (Google Drive via gdown)
# ------------------------------------------------------------
def _ensure_model_on_disk() -> str:
    path = BULLBRAIN_MODEL_PATH

    if os.path.exists(path) and BULLBRAIN_SKIP_DOWNLOAD:
        return path

    drive_url = BULLBRAIN_MODEL_DRIVE_URL
    if not drive_url and BULLBRAIN_MODEL_GDRIVE_ID:
        drive_url = f"https://drive.google.com/uc?id={BULLBRAIN_MODEL_GDRIVE_ID}"

    if not drive_url:
        return path

    try:
        import gdown  # type: ignore
    except Exception as e:
        print(f"[bullbrain] gdown not available, cannot download model: {e}")
        return path

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

    try:
        out = gdown.download(drive_url, path, quiet=True)
        if out:
            print(f"[bullbrain] ✅ Model downloaded to {path}")
        else:
            print(f"[bullbrain] ⚠️ Model download returned None, path={path}")
    except Exception as e:
        print(f"[bullbrain] ❌ Model download failed: {e}")

    return path


# ------------------------------------------------------------
# Model loader
# ------------------------------------------------------------
def load_bullbrain_model() -> Optional[xgb.Booster]:
    global bullbrain_model

    path = _ensure_model_on_disk()

    if not os.path.exists(path):
        print(f"[bullbrain] ❌ Model file not found: {path}")
        bullbrain_model = None
        return None

    try:
        booster = xgb.Booster()
        booster.load_model(path)
        bullbrain_model = booster
        print(f"[bullbrain] ✅ Model loaded: {path} | version={BULLBRAIN_VERSION}")
        return booster
    except Exception as e:
        print(f"[bullbrain] ❌ Failed to load model: {e}")
        bullbrain_model = None
        return None


def ensure_bullbrain_loaded() -> None:
    """
    Safe, idempotent: loads model once per process.
    """
    global bullbrain_model
    if bullbrain_model is not None:
        return
    load_bullbrain_model()


# ------------------------------------------------------------
# Numeric helpers
# ------------------------------------------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _pct(a: float, b: float) -> float:
    # percent difference (a vs b)
    if b == 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def _sma(arr: np.ndarray, n: int) -> float:
    if arr.size < n:
        return float(arr.mean()) if arr.size else 0.0
    return float(np.mean(arr[-n:]))


def _ema(arr: np.ndarray, n: int) -> float:
    if arr.size == 0:
        return 0.0
    alpha = 2.0 / (n + 1.0)
    ema = float(arr[0])
    for x in arr[1:]:
        ema = alpha * float(x) + (1.0 - alpha) * ema
    return float(ema)


def _rsi(arr: np.ndarray, n: int = 14) -> float:
    if arr.size < n + 1:
        return 50.0
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[-n:])
    avg_loss = np.mean(losses[-n:])

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _linear_slope(arr: np.ndarray) -> float:
    """
    Simple slope of last N points using least squares.
    Returns slope per step (not percent).
    """
    n = arr.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = arr.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(slope)


# ------------------------------------------------------------
# Feature computation (48 features)
# ------------------------------------------------------------
def compute_bullbrain_features(
    candles: List[Dict[str, Any]]
) -> Tuple[Optional[np.ndarray], Dict[str, float], Optional[float]]:
    """
    Input candles: list of dicts with keys like open/high/low/close/volume (or o/h/l/c/v)
    Output:
      - features_vec (np.ndarray shape (48,))
      - feature_dict (name->value)
      - last_close
    """
    if not candles or len(candles) < 25:
        return None, {}, None

    def get_key(d: Dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if k in d:
                return d.get(k)
        return None

    closes = np.array([_safe_float(get_key(c, "close", "c")) for c in candles], dtype=float)
    highs = np.array([_safe_float(get_key(c, "high", "h")) for c in candles], dtype=float)
    lows = np.array([_safe_float(get_key(c, "low", "l")) for c in candles], dtype=float)
    opens = np.array([_safe_float(get_key(c, "open", "o")) for c in candles], dtype=float)
    vols = np.array([_safe_float(get_key(c, "volume", "v")) for c in candles], dtype=float)

    if closes.size < 2:
        return None, {}, None

    last_close = float(closes[-1])

    # Returns (percent)
    ret1 = _pct(closes[-1], closes[-2]) if closes.size >= 2 else 0.0
    ret5 = _pct(closes[-1], closes[-6]) if closes.size >= 6 else 0.0
    ret10 = _pct(closes[-1], closes[-11]) if closes.size >= 11 else 0.0

    # Volatility (std dev of daily returns, percent)
    def vol_n(n: int) -> float:
        if closes.size < n + 1:
            return 0.0
        r = np.diff(closes[-(n + 1):]) / np.maximum(closes[-(n + 1):-1], 1e-9)
        return float(np.std(r) * 100.0)

    vol5 = vol_n(5)
    vol20 = vol_n(20)
    vol60 = vol_n(60)

    sma5 = _sma(closes, 5)
    sma10 = _sma(closes, 10)
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)

    sma5_sma20 = _pct(sma5, sma20) if sma20 != 0 else 0.0
    sma20_sma50 = _pct(sma20, sma50) if sma50 != 0 else 0.0
    price_vs_sma20 = _pct(closes[-1], sma20) if sma20 != 0 else 0.0

    rsi14 = _rsi(closes, 14)

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    ema_ratio = (ema12 / ema26) if ema26 != 0 else 1.0

    macd = ema12 - ema26
    # macd signal = EMA of MACD series
    # build MACD series fast:
    # (approx) compute ema arrays iteratively
    macd_series = []
    ema12_tmp = float(closes[0])
    ema26_tmp = float(closes[0])
    a12 = 2.0 / (12 + 1.0)
    a26 = 2.0 / (26 + 1.0)
    for x in closes:
        ema12_tmp = a12 * float(x) + (1 - a12) * ema12_tmp
        ema26_tmp = a26 * float(x) + (1 - a26) * ema26_tmp
        macd_series.append(ema12_tmp - ema26_tmp)
    macd_series = np.array(macd_series, dtype=float)
    macd_signal = _ema(macd_series, 9)
    macd_hist = macd - macd_signal

    # Williams %R (14)
    look = 14
    if highs.size >= look and lows.size >= look:
        hh = float(np.max(highs[-look:]))
        ll = float(np.min(lows[-look:]))
        denom = (hh - ll) if (hh - ll) != 0 else 1e-9
        will_r = -100.0 * (hh - float(closes[-1])) / denom
    else:
        will_r = -50.0

    # Stochastic %K (14) and %D (3 SMA of %K)
    if highs.size >= look and lows.size >= look:
        hh = float(np.max(highs[-look:]))
        ll = float(np.min(lows[-look:]))
        denom = (hh - ll) if (hh - ll) != 0 else 1e-9
        stoch_k = 100.0 * (float(closes[-1]) - ll) / denom
        # build last 3 K values (best-effort)
        k_vals = []
        for i in range(3):
            if highs.size >= look + i and lows.size >= look + i and closes.size >= look + i:
                hh_i = float(np.max(highs[-(look + i): -i if i else None]))
                ll_i = float(np.min(lows[-(look + i): -i if i else None]))
                denom_i = (hh_i - ll_i) if (hh_i - ll_i) != 0 else 1e-9
                c_i = float(closes[-(1 + i)])
                k_vals.append(100.0 * (c_i - ll_i) / denom_i)
        stoch_d = float(np.mean(k_vals)) if k_vals else stoch_k
    else:
        stoch_k = 50.0
        stoch_d = 50.0

    # Volume changes
    vol_chg_1d = _pct(vols[-1], vols[-2]) if vols.size >= 2 and vols[-2] != 0 else 0.0
    vol_ma5 = _sma(vols, 5)
    vol_ma20 = _sma(vols, 20)
    vol_vs_ma5 = _pct(vols[-1], vol_ma5) if vol_ma5 != 0 else 0.0
    vol_vs_ma20 = _pct(vols[-1], vol_ma20) if vol_ma20 != 0 else 0.0

    # OBV + slope 10
    obv = 0.0
    obv_series = [0.0]
    for i in range(1, closes.size):
        if closes[i] > closes[i - 1]:
            obv += vols[i]
        elif closes[i] < closes[i - 1]:
            obv -= vols[i]
        obv_series.append(obv)
    obv_series = np.array(obv_series, dtype=float)
    obv_slope_10 = _linear_slope(obv_series[-10:]) if obv_series.size >= 10 else _linear_slope(obv_series)

    # Intraday ranges / true range / ATR14
    intraday_range_pct = _pct(highs[-1], lows[-1]) if lows[-1] != 0 else 0.0

    prev_close = closes[-2] if closes.size >= 2 else closes[-1]
    tr = max(
        float(highs[-1] - lows[-1]),
        float(abs(highs[-1] - prev_close)),
        float(abs(lows[-1] - prev_close)),
    )
    true_range = tr

    # ATR14: SMA of true ranges over last 14 periods
    if closes.size >= 15:
        trs = []
        for i in range(closes.size - 14, closes.size):
            pc = closes[i - 1]
            trs.append(
                max(
                    float(highs[i] - lows[i]),
                    float(abs(highs[i] - pc)),
                    float(abs(lows[i] - pc)),
                )
            )
        atr14 = float(np.mean(trs)) if trs else float(true_range)
    else:
        atr14 = float(true_range)

    # Candle anatomy (percent of range)
    rng = float(highs[-1] - lows[-1])
    rng = rng if rng != 0 else 1e-9
    upper_shadow = float(highs[-1] - max(opens[-1], closes[-1]))
    lower_shadow = float(min(opens[-1], closes[-1]) - lows[-1])
    body = float(abs(closes[-1] - opens[-1]))

    upper_shadow_pct = (upper_shadow / rng) * 100.0
    lower_shadow_pct = (lower_shadow / rng) * 100.0
    body_pct = (body / rng) * 100.0

    # Gap vs previous close (percent)
    gap_pct = _pct(opens[-1], prev_close) if prev_close != 0 else 0.0

    # Distance from 20d high/low (percent)
    if closes.size >= 20:
        high20 = float(np.max(highs[-20:]))
        low20 = float(np.min(lows[-20:]))
    else:
        high20 = float(np.max(highs))
        low20 = float(np.min(lows))
    distance_from_20d_high = _pct(closes[-1], high20) if high20 != 0 else 0.0
    distance_from_20d_low = _pct(closes[-1], low20) if low20 != 0 else 0.0

    # Volume z-score 20
    if vols.size >= 20:
        v20 = vols[-20:]
        mu = float(np.mean(v20))
        sd = float(np.std(v20)) if float(np.std(v20)) != 0 else 1e-9
        volume_zscore_20 = float((vols[-1] - mu) / sd)
    else:
        volume_zscore_20 = 0.0

    # Trend strength 20: slope of close over last 20, scaled as % of mean price
    if closes.size >= 20:
        c20 = closes[-20:]
    else:
        c20 = closes
    slope = _linear_slope(c20)
    denom = float(np.mean(c20)) if float(np.mean(c20)) != 0 else 1e-9
    trend_strength_20 = float((slope / denom) * 100.0)

    feature_dict: Dict[str, float] = {
        "adj_close": float(closes[-1]),  # if no adjusted close, use close
        "close": float(closes[-1]),
        "high": float(highs[-1]),
        "low": float(lows[-1]),
        "open": float(opens[-1]),
        "volume": float(vols[-1]),
        "return_1d": float(ret1),
        "return_5d": float(ret5),
        "return_10d": float(ret10),
        "volatility_5d": float(vol5),
        "volatility_20d": float(vol20),
        "volatility_60d": float(vol60),
        "sma5": float(sma5),
        "sma10": float(sma10),
        "sma20": float(sma20),
        "sma50": float(sma50),
        "sma200": float(sma200),
        "sma5_sma20_pct": float(sma5_sma20),
        "sma20_sma50_pct": float(sma20_sma50),
        "price_vs_sma20_pct": float(price_vs_sma20),
        "rsi14": float(rsi14),
        "macd": float(macd),
        "macd_signal": float(macd_signal),
        "macd_hist": float(macd_hist),
        "ema12": float(ema12),
        "ema26": float(ema26),
        "ema_ratio": float(ema_ratio),
        "williams_r_14": float(will_r),
        "stoch_k_14": float(stoch_k),
        "stoch_d_3": float(stoch_d),
        "volume_change_1d": float(vol_chg_1d),
        "volume_ma5": float(vol_ma5),
        "volume_ma20": float(vol_ma20),
        "volume_vs_ma5_pct": float(vol_vs_ma5),
        "volume_vs_ma20_pct": float(vol_vs_ma20),
        "obv": float(obv_series[-1]),
        "obv_slope_10": float(obv_slope_10),
        "intraday_range_pct": float(intraday_range_pct),
        "true_range": float(true_range),
        "atr14": float(atr14),
        "upper_shadow_pct": float(upper_shadow_pct),
        "lower_shadow_pct": float(lower_shadow_pct),
        "body_pct": float(body_pct),
        "gap_pct": float(gap_pct),
        "distance_from_20d_high": float(distance_from_20d_high),
        "distance_from_20d_low": float(distance_from_20d_low),
        "volume_zscore_20": float(volume_zscore_20),
        "trend_strength_20": float(trend_strength_20),
    }

    vec = np.array([_safe_float(feature_dict.get(k), 0.0) for k in BULLBRAIN_FEATURES], dtype=float)
    return vec, feature_dict, last_close


# ------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def _signal_from_prob(prob_up: float) -> str:
    p = float(prob_up)
    if p >= 0.58:
        return "BUY"
    if p <= 0.42:
        return "SELL"
    return "HOLD"


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
def bullbrain_infer(features_vec: np.ndarray) -> Dict[str, Any]:
    if bullbrain_model is None:
        return {
            "ok": False,
            "error": "bullbrain_model_not_loaded",
            "probability_up": 0.5,
            "probability_down": 0.5,
            "signal": "HOLD",
            "confidence": 50.0,
            "version": BULLBRAIN_VERSION,
        }

    try:
        x = np.array(features_vec, dtype=float).reshape(1, -1)
        dmat = xgb.DMatrix(x)

        raw = bullbrain_model.predict(dmat)[0]

        if 0.0 <= float(raw) <= 1.0:
            prob_up = float(raw)
        else:
            prob_up = float(_sigmoid(float(raw)))

        prob_down = 1.0 - prob_up
        signal = _signal_from_prob(prob_up)
        confidence = max(prob_up, prob_down) * 100.0

        return {
            "ok": True,
            "raw_output": float(raw),
            "probability_up": float(prob_up),
            "probability_down": float(prob_down),
            "signal": signal,
            "confidence": float(round(confidence, 2)),
            "version": BULLBRAIN_VERSION,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "probability_up": 0.5,
            "probability_down": 0.5,
            "signal": "HOLD",
            "confidence": 50.0,
            "version": BULLBRAIN_VERSION,
        }
