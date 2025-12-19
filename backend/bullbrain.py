
# backend/bullbrain.py
# ============================================================
# BullSignalsAI — BullBrain (Model Loading + Inference Helpers)
# ============================================================

from __future__ import annotations

import os
import math
import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

# xgboost is required (Render: add to requirements.txt)
import xgboost as xgb

print(f"[bullbrain] loaded from: {__file__}")
print(f"[bullbrain] has load_bullbrain_model: {'load_bullbrain_model' in globals()}")


# ------------------------------------------------------------
# Versioning
# ------------------------------------------------------------
BULLBRAIN_VERSION = os.getenv("BULLBRAIN_VERSION", "v2")


# ------------------------------------------------------------
# Model location (THIS is where model path/drive logic lives)
# ------------------------------------------------------------
# ✅ Local path where we want the model to exist at runtime
BULLBRAIN_MODEL_PATH = os.getenv("BULLBRAIN_MODEL_PATH", "/tmp/bullbrain_model.json")

# ✅ If you set this, we’ll attempt to download from Google Drive each boot
#    Examples:
#      BULLBRAIN_MODEL_DRIVE_URL="https://drive.google.com/uc?id=XXXX"
#      BULLBRAIN_MODEL_GDRIVE_ID="XXXX"
BULLBRAIN_MODEL_DRIVE_URL = os.getenv("BULLBRAIN_MODEL_DRIVE_URL", "").strip()
BULLBRAIN_MODEL_GDRIVE_ID = os.getenv("BULLBRAIN_MODEL_GDRIVE_ID", "").strip()

# ✅ Optional toggle to skip downloading (useful if model already on disk)
BULLBRAIN_SKIP_DOWNLOAD = os.getenv("BULLBRAIN_SKIP_DOWNLOAD", "false").lower() == "true"

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
TOP_LIQUID_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX","AVGO",
    "JPM","BAC","XOM","CVX","UNH","WMT","HD","PG","LLY","V","MA","KO","PEP",
    "MRK","ABBV","ORCL","INTC","CRM","COST","PYPL","QCOM","ADBE","TXN",
    "NKE","PFE","T","VZ","NEE","UPS","UNP","GS","MS","BA","CAT","GE","IBM"
]

# ------------------------------------------------------------
# Global model handle
# ------------------------------------------------------------
bullbrain_model: Optional[xgb.Booster] = None


# ------------------------------------------------------------
# Download helper (Google Drive via gdown)
# ------------------------------------------------------------
def _ensure_model_on_disk() -> str:
    """
    Ensures the model exists at BULLBRAIN_MODEL_PATH.
    Returns the final local path.
    """
    path = BULLBRAIN_MODEL_PATH

    # If file exists and user wants to skip download -> done
    if os.path.exists(path) and BULLBRAIN_SKIP_DOWNLOAD:
        return path

    # If no drive config, just return local path (caller may fail if missing)
    drive_url = BULLBRAIN_MODEL_DRIVE_URL
    if not drive_url and BULLBRAIN_MODEL_GDRIVE_ID:
        drive_url = f"https://drive.google.com/uc?id={BULLBRAIN_MODEL_GDRIVE_ID}"

    if not drive_url:
        return path

    # Attempt download
    try:
        import gdown  # type: ignore
    except Exception as e:
        # gdown not installed; caller will try to load local path
        print(f"[bullbrain] gdown not available, cannot download model: {e}")
        return path

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

    try:
        # quiet=True avoids noisy logs
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
    """
    Loads BullBrain model from local disk (optionally downloaded from Drive).
    Returns Booster or None.
    """
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


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def _class_probs_from_prob_up(prob_up: float) -> Dict[str, float]:
    """
    Map a single prob_up to the UI buckets you’ve been using.
    (Keeps behavior compatible across app.)
    """
    p = max(0.0, min(1.0, float(prob_up)))
    return {
        "prob_up": p,
        "prob_down": 1.0 - p,
    }


def _signal_from_prob(prob_up: float) -> str:
    """
    Simple signal mapping used across your backend.
    """
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
    """
    Run inference with bullbrain_model. Returns a consistent dict
    used by cron + stockdetail + endpoints.
    """
    if bullbrain_model is None:
        return {
            "ok": False,
            "error": "bullbrain_model_not_loaded",
            "probability_up": 0.5,
            "probability_down": 0.5,
            "signal": "HOLD",
            "confidence": 50.0,
        }

    try:
        # Ensure shape (1, n_features)
        x = np.array(features_vec, dtype=float).reshape(1, -1)
        dmat = xgb.DMatrix(x)

        # Many XGBoost binaries output raw margin or probability depending on training config.
        raw = bullbrain_model.predict(dmat)[0]

        # If raw seems like a probability already, keep it. Otherwise sigmoid.
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
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "probability_up": 0.5,
            "probability_down": 0.5,
            "signal": "HOLD",
            "confidence": 50.0,
        }
