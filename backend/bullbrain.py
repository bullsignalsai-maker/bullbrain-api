# backend/bullbrain.py

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import gdown
from typing import Dict, Tuple, Optional

# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------

MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1TeutMa8jQ5l4Lw-ZaN1gP1iGfDp5spAJ"
LOCAL_MODEL_PATH = "models/bullbrain_v2_48f.json"

BULLBRAIN_VERSION = "v2-48f"

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

# ------------------------------------------------------------------
# Model singleton
# ------------------------------------------------------------------

_model: Optional[xgb.Booster] = None


def load_model() -> xgb.Booster:
    global _model

    if _model is not None:
        return _model

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(LOCAL_MODEL_PATH):
        gdown.download(
            MODEL_DRIVE_URL,
            LOCAL_MODEL_PATH,
            quiet=False,
            fuzzy=True,
        )

    booster = xgb.Booster()
    booster.load_model(LOCAL_MODEL_PATH)

    if booster.num_features() != len(BULLBRAIN_FEATURES):
        raise RuntimeError(
            f"Feature mismatch: model={booster.num_features()} "
            f"expected={len(BULLBRAIN_FEATURES)}"
        )

    _model = booster
    return booster


# ------------------------------------------------------------------
# Feature computation (48 features)
# ------------------------------------------------------------------

def compute_features(
    candles: Dict
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Returns:
      - feature vector (1 x 48)
      - feature dictionary
      - last close price
    """

    df = pd.DataFrame({
        "close": candles["close"],
        "high": candles["high"],
        "low": candles["low"],
        "open": candles.get("open") or candles["close"],
        "volume": candles["volume"],
    })

    df["adj_close"] = df["close"]

    # Returns
    df["return_1d"] = df["close"].pct_change() * 100
    df["return_5d"] = df["close"].pct_change(5) * 100
    df["return_10d"] = df["close"].pct_change(10) * 100

    # Volatility
    daily_ret = df["close"].pct_change()
    df["volatility_5d"] = daily_ret.rolling(5).std() * 100
    df["volatility_20d"] = daily_ret.rolling(20).std() * 100
    df["volatility_60d"] = daily_ret.rolling(60).std() * 100

    # SMAs
    for n in [5, 10, 20, 50, 200]:
        df[f"sma{n}"] = df["close"].rolling(n).mean()

    df["sma5_sma20_pct"] = (df["sma5"] / df["sma20"] - 1) * 100
    df["sma20_sma50_pct"] = (df["sma20"] / df["sma50"] - 1) * 100
    df["price_vs_sma20_pct"] = (df["close"] / df["sma20"] - 1) * 100

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["ema12"] = ema12
    df["ema26"] = ema26
    df["ema_ratio"] = ema12 / (ema26 + 1e-9)

    # Williams / Stochastic
    hh = df["high"].rolling(14).max()
    ll = df["low"].rolling(14).min()
    df["williams_r_14"] = (df["close"] - hh) / (hh - ll + 1e-9) * 100
    df["stoch_k_14"] = (df["close"] - ll) / (hh - ll + 1e-9) * 100
    df["stoch_d_3"] = df["stoch_k_14"].rolling(3).mean()

    # Volume
    df["volume_change_1d"] = df["volume"].pct_change() * 100
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_vs_ma5_pct"] = (df["volume"] / (df["volume_ma5"] + 1e-9) - 1) * 100
    df["volume_vs_ma20_pct"] = (df["volume"] / (df["volume_ma20"] + 1e-9) - 1) * 100

    df["obv"] = (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()
    df["obv_slope_10"] = df["obv"].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0],
        raw=False,
    )

    # Range / ATR
    df["intraday_range_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-9) * 100
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
    df["upper_shadow_pct"] = (df["high"] - df["close"]) / (df["close"] + 1e-9) * 100
    df["lower_shadow_pct"] = (df["close"] - df["low"]) / (df["close"] + 1e-9) * 100
    df["body_pct"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
    df["gap_pct"] = (df["open"] - df["close"].shift()) / (df["close"].shift() + 1e-9) * 100

    # Extremes
    hi20 = df["high"].rolling(20).max()
    lo20 = df["low"].rolling(20).min()
    df["distance_from_20d_high"] = (df["close"] / (hi20 + 1e-9) - 1) * 100
    df["distance_from_20d_low"] = (df["close"] / (lo20 + 1e-9) - 1) * 100

    # Z-score + trend
    std20 = df["volume_ma20"].rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - df["volume_ma20"]) / (std20 + 1e-9)
    df["trend_strength_20"] = df["close"].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0],
        raw=False,
    )

    row = df.iloc[-1]

    feature_dict = {}
    values = []

    for name in BULLBRAIN_FEATURES:
        val = row.get(name)
        feature_dict[name] = None if pd.isna(val) else float(val)
        values.append(float(val) if pd.notna(val) else np.nan)

    return np.array([values], dtype=float), feature_dict, float(row["close"])


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def infer(features: np.ndarray) -> Dict:
    model = load_model()

    dmat = xgb.DMatrix(features, feature_names=BULLBRAIN_FEATURES)
    prob_up = float(model.predict(dmat)[0])

    if prob_up >= 0.55:
        signal = "BUY"
    elif prob_up <= 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(max(prob_up, 1 - prob_up) * 100, 2)

    return {
        "signal": signal,
        "confidence": confidence,
        "probability_up": round(prob_up, 4),
        "probability_down": round(1 - prob_up, 4),
        "raw_output": prob_up,
        "version": BULLBRAIN_VERSION,
    }
