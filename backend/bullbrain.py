from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import xgboost as xgb


# ------------------------------------------------------------
# Process Role (SAFE)
# ------------------------------------------------------------
IS_API_PROCESS = os.getenv("IS_API_PROCESS", "false").lower() == "true"


# ------------------------------------------------------------
# Versioning
# ------------------------------------------------------------
BULLBRAIN_VERSION = os.getenv("BULLBRAIN_VERSION", "v2")


# ------------------------------------------------------------
# Model location (Drive → local path)
# ------------------------------------------------------------
BULLBRAIN_MODEL_PATH = os.getenv(
    "BULLBRAIN_MODEL_PATH",
    "models/bullbrain_v2_48f.json"
)

BULLBRAIN_MODEL_DRIVE_URL = os.getenv("BULLBRAIN_MODEL_DRIVE_URL", "").strip()
BULLBRAIN_MODEL_GDRIVE_ID = os.getenv("BULLBRAIN_MODEL_GDRIVE_ID", "").strip()


# ------------------------------------------------------------
# Global model handle (singleton)
# ------------------------------------------------------------
bullbrain_model: Optional[xgb.Booster] = None



# ------------------------------------------------------------
# Feature order (MUST match training order)
# ------------------------------------------------------------
BULLBRAIN_FEATURES: List[str] = [
    "adj_close","close","high","low","open","volume",
    "return_1d","return_5d","return_10d",
    "volatility_5d","volatility_20d","volatility_60d",
    "sma5","sma10","sma20","sma50","sma200",
    "sma5_sma20_pct","sma20_sma50_pct","price_vs_sma20_pct",
    "rsi14","macd","macd_signal","macd_hist",
    "ema12","ema26","ema_ratio",
    "williams_r_14","stoch_k_14","stoch_d_3",
    "volume_change_1d","volume_ma5","volume_ma20",
    "volume_vs_ma5_pct","volume_vs_ma20_pct",
    "obv","obv_slope_10",
    "intraday_range_pct","true_range","atr14",
    "upper_shadow_pct","lower_shadow_pct","body_pct","gap_pct",
    "distance_from_20d_high","distance_from_20d_low",
    "volume_zscore_20","trend_strength_20",
]


# ------------------------------------------------------------
# Global model handle (SINGLETON)
# ------------------------------------------------------------
bullbrain_model: Optional[xgb.Booster] = None


# ------------------------------------------------------------
# Download helper (API PROCESS ONLY)
# ------------------------------------------------------------
def _ensure_model_on_disk() -> str:
    path = BULLBRAIN_MODEL_PATH

    # Model already exists
    if os.path.exists(path):
        return path

    # Build drive URL
    drive_url = BULLBRAIN_MODEL_DRIVE_URL
    if not drive_url and BULLBRAIN_MODEL_GDRIVE_ID:
        drive_url = f"https://drive.google.com/uc?id={BULLBRAIN_MODEL_GDRIVE_ID}"

    if not drive_url:
        print("[bullbrain] ❌ No Drive URL configured")
        return path

    try:
        import gdown  # type: ignore
    except Exception as e:
        print(f"[bullbrain] ❌ gdown unavailable: {e}")
        return path

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = gdown.download(drive_url, path, quiet=True)
        if out:
            print(f"[bullbrain] ✅ Model downloaded to {path}")
        else:
            print("[bullbrain] ❌ Download returned None")
    except Exception as e:
        print(f"[bullbrain] ❌ Download failed: {e}")

    return path


# ------------------------------------------------------------
# Model loader (FAIL-FAST)
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
        print(f"[bullbrain] ✅ Model loaded ({BULLBRAIN_VERSION})")
        return booster
    except Exception as e:
        print(f"[bullbrain] ❌ Failed to load model: {e}")
        bullbrain_model = None
        return None


def ensure_bullbrain_loaded() -> None:
    """
    Idempotent + FAIL FAST
    """
    global bullbrain_model
    if bullbrain_model is not None:
        return

    model = load_bullbrain_model()
    if model is None:
        raise RuntimeError("BullBrain model failed to load")


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
    n = arr.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = arr.astype(float)
    denom = np.sum((x - x.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


# ------------------------------------------------------------
# Feature computation (UNCHANGED, SAFE)
# ------------------------------------------------------------
def compute_bullbrain_features(
    candles: List[Dict[str, Any]]
) -> Tuple[Optional[np.ndarray], Dict[str, float], Optional[float]]:
    # 🔹 YOUR EXISTING IMPLEMENTATION
    # 🔹 LEFT 100% UNCHANGED
    # 🔹 (You already pasted it — keep as-is)
    ...
    # ⚠️ DO NOT MODIFY FEATURE LOGIC
    ...


# ------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def _signal_from_prob(prob_up: float) -> str:
    if prob_up >= 0.58:
        return "BUY"
    if prob_up <= 0.42:
        return "SELL"
    return "HOLD"


# ------------------------------------------------------------
# Inference (SAFE)
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

        prob_up = float(raw) if 0.0 <= float(raw) <= 1.0 else _sigmoid(float(raw))
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
