# main.py

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from urllib.parse import urlparse
import os
import requests
import datetime
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import gdown
import re
import math
from symbols_clean import REAL_TICKERS
import firebase_admin
from firebase_admin import credentials, firestore
import time
from backend.candle_store import get_candles
from backend.candle_store import get_candles as get_cached_candles
from backend.stock_bootstrap import bootstrap_stock
from backend.quote_demand import ensure_quote
from backend.active_symbols import touch_active_symbol
from backend.firestore_utils import get_db
from backend.watchlist_snapshot import (
    build_watchlist_snapshot,
    get_watchlist_snapshot,
    is_snapshot_fresh,
)

from fastapi import APIRouter

from backend.news.market_news_repo import get_market_news


router = APIRouter()
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
TOP_LIQUID_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX","AVGO",
    "JPM","BAC","XOM","CVX","UNH","WMT","HD","PG","LLY","V","MA","KO","PEP",
    "MRK","ABBV","ORCL","INTC","CRM","COST","PYPL","QCOM","ADBE","TXN",
    "NKE","PFE","T","VZ","NEE","UPS","UNP","GS","MS","BA","CAT","GE","IBM"
]

bullbrain_model: xgb.Booster | None = None
cache: dict[str, dict] = {}

print("✅ get_candles loaded from:", get_candles.__module__)

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

# =========================================================
# Incremental Polygon Candle Fetch
# =========================================================
def fetch_polygon_candles_incremental(symbol: str, since_ms: int | None):
    if not POLYGON_KEY:
        return None

    now = datetime.datetime.utcnow()
    end = int(now.timestamp() * 1000)

    # If no cache → fetch 1 year
    if not since_ms:
        start = int((now - datetime.timedelta(days=365)).timestamp() * 1000)
    else:
        start = since_ms + 1  # avoid duplicate candle

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_KEY}"
    )

    data = safe_json(url)
    if not data or "results" not in data:
        return None

    res = data["results"]
    if not res:
        return None

    return {
        "source": "polygon",
        "close": [r["c"] for r in res],
        "high": [r["h"] for r in res],
        "low": [r["l"] for r in res],
        "open": [r.get("o", r["c"]) for r in res],
        "volume": [r["v"] for r in res],
        "timestamp": [r["t"] for r in res],
    }


def fetch_daily_candles(symbol: str, min_points: int = 60):
    """
    DEPRECATED: Redirects to Firestore-backed candle store.
    Kept for backward compatibility.
    """
    return get_candles(symbol, min_points=min_points)


def get_candles_cached(symbol: str):
    cached = load_cached_candles(symbol)

    since_ms = cached.get("last_t_ms") if cached else None
    new_data = fetch_polygon_candles_incremental(symbol, since_ms)

    if not cached and not new_data:
        return None

    if cached and new_data:
        c = cached["candles"]
        for k in ["close", "high", "low", "open", "volume", "timestamp"]:
            c[k].extend(new_data[k])
        save_cached_candles(symbol, c)
        return c

    if new_data:
        save_cached_candles(symbol, new_data)
        return new_data

    return cached["candles"]


# ============================================================
# SMART PATTERN CORE + HISTORY SCANNER
# ============================================================

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic RSI calculation on a pandas Series of closes."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Williams %R over a lookback window."""
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr

# -----------------------------------------------------------
# STEP 17: Elite Pattern Vocabulary (UI + Quality Control)
# -----------------------------------------------------------

ELITE_PATTERNS = {
    "GAP UP & RUNNING",
    "VOLUME BREAKOUT",
    "OVERSOLD BOUNCE",
    "HAMMER REVERSAL",
    "BUY THE DIP (UPTREND)",
    "TREND ACCELERATION",
    "FAILED BREAKOUT TRAP",
    "DEAD CAT BOUNCE",
}

ELITE_PATTERN_LABELS = {
    "BULLISH_ENGULFING_VOLUME": "Bullish Engulfing (Volume Confirmed)",
    "BEARISH_ENGULFING": "Bearish Engulfing",
    "POCKET_PIVOT": "Pocket Pivot",
    "CLIMAX_BAR": "Climax Bar",
    "EXHAUSTION_GAP": "Exhaustion Gap",
}
DEPRECATED_PATTERNS = {
    "HEAD_AND_SHOULDERS",
    "INVERSE_HEAD_AND_SHOULDERS",
    "SYMMETRICAL_TRIANGLE",
    "ASCENDING_TRIANGLE",
    "DESCENDING_TRIANGLE",
    "BULL_PENNANT",
    "BEAR_PENNANT",
}

def _evaluate_smart_pattern_row(
    *,
    gap: float | None,
    change: float | None,
    vol_z: float | None,
    vol_vs_ma: float | None,
    rsi: float | None,
    will_r: float | None,
    lower_shadow: float | None,
    upper_shadow: float | None,
    body_pct: float | None,
    price_vs_sma20: float | None,
    trend: float | None,
    ret3: float | None,
    ret5: float | None,
):
    """
    Core smart-pattern classifier.
    Takes pre-computed daily metrics and returns a single "best" pattern dict or None.

    We keep the UI simple (only the best pattern per day), but internally this engine
    can support many patterns without changing the API.
    """

    def ok(x):
        return x is not None and not np.isnan(x)

    # (score, pattern_dict)
    patterns: list[tuple[float, dict]] = []

    # 1) GAP UP & RUNNING – strong upside ignition
    if ok(gap) and ok(change) and ok(vol_vs_ma):
        if gap > 1.0 and change > 2.0 and vol_vs_ma > 20.0:
            patterns.append(
                (
                    0.9,
                    {
                        "pattern": "GAP UP & RUNNING",
                        "bias": "bull",
                        "headline": "Stock exploded higher at the open and buyers kept control all day.",
                        "explanation": (
                            "The stock opened noticeably above yesterday’s close and then continued "
                            "to push higher on well-above-average volume. This kind of gap-and-go move "
                            "often marks the start of short-term momentum runs."
                        ),
                    },
                )
            )

    # 2) MASSIVE VOLUME BREAKOUT – abnormal participation
    if ok(vol_z) and vol_z > 3.0:
        patterns.append(
            (
                0.85,
                {
                    "pattern": "VOLUME BREAKOUT",
                   
                    "bias": "bull",
                    "headline": "Unusually heavy trading volume – the big players are active.",
                    "explanation": (
                        "Today’s volume is far above the typical 20-day range, which usually only "
                        "happens when institutions or large funds are buying or selling aggressively. "
                        "Such volume shocks often precede strong follow-through moves."
                    ),
                },
            )
        )

    # 3) OVERSOLD BOUNCE – washout then reversal attempt
    if ok(rsi) and ok(will_r) and ok(vol_z):
        if rsi < 30 and will_r < -80 and vol_z > 2.0:
            patterns.append(
                (
                    0.9,
                    {
                        "pattern": "OVERSOLD BOUNCE",
                        
                        "bias": "bull",
                        "headline": "After heavy selling, dip-buyers finally stepped in with size.",
                        "explanation": (
                            "The stock had been deeply oversold and now shows a strong bounce on elevated "
                            "volume. Historically this kind of capitulation followed by high-conviction "
                            "buying often leads to sharp relief rallies."
                        ),
                    },
                )
            )

    # 4) HAMMER REVERSAL – intraday flush, close near highs
    if ok(lower_shadow) and ok(body_pct) and ok(change):
        # much longer lower wick, small body, green day
        if lower_shadow > 40.0 and abs(body_pct) < 40.0 and change > 0:
            patterns.append(
                (
                    0.8,
                    {
                        "pattern": "HAMMER REVERSAL",
                        
                        "bias": "bull",
                        "headline": "Bears pushed price down, but bulls slammed it back up by the close.",
                        "explanation": (
                            "Intraday the stock traded significantly lower, but buyers aggressively bought "
                            "the dip and forced price back toward the top of the day’s range. This hammer-style "
                            "candle often appears near local bottoms where selling pressure is finally exhausted."
                        ),
                    },
                )
            )

    # 5) BUY THE DIP (UPTREND) – pullback within strong trend
    if ok(trend) and ok(price_vs_sma20) and ok(change):
        if trend > 10.0 and price_vs_sma20 < -3.0 and change > 0:
            patterns.append(
                (
                    0.78,
                    {
                        "pattern": "BUY THE DIP (UPTREND)",
                       
                        "bias": "bull",
                        "headline": "Strong trend, normal pullback, and buyers stepping back in.",
                        "explanation": (
                            "The stock remains in a clear uptrend but had pulled back below its 20-day "
                            "trend line and is now bouncing. This is the classic 'buy the dip' profile "
                            "that many trend-followers use to add to winning positions."
                        ),
                    },
                )
            )

    # 6) DEAD CAT BOUNCE – weak rebound after big fall
    if ok(ret5) and ok(change) and ok(vol_z):
        if ret5 < -8.0 and change > 0 and vol_z < 1.0:
            patterns.append(
                (
                    0.75,
                    {
                        "pattern": "DEAD CAT BOUNCE",
                      
                        "bias": "bear",
                        "headline": "After a big drop, price is bouncing – but on weak conviction.",
                        "explanation": (
                            "The stock has sold off hard over the past few sessions and is now showing a small "
                            "bounce, but without a meaningful volume surge. Many such weak rebounds fail and "
                            "roll over again as sellers re-enter at slightly better prices."
                        ),
                    },
                )
            )

    # 7) OVERBOUGHT DISTRIBUTION – hot chart, cooling demand
    if ok(rsi) and ok(vol_vs_ma) and ok(change):
        if rsi > 70 and vol_vs_ma < 0:
            patterns.append(
                (
                    0.72,
                    {
                        "pattern": "OVERBOUGHT DISTRIBUTION",
                       
                        "bias": "bear",
                        "headline": "Sentiment is hot, but real demand is fading under the surface.",
                        "explanation": (
                            "Momentum has been strong and the chart looks extended, but today’s volume is no "
                            "longer beating its recent average. This can indicate that smart money is quietly "
                            "selling into late-stage enthusiasm near short-term peaks."
                        ),
                    },
                )
            )

    # 8) FAILED BREAKOUT TRAP – breakout hunters punished
    if ok(change) and ok(vol_z):
        if change < -2.0 and vol_z > 2.0:
            patterns.append(
                (
                    0.7,
                    {
                        "pattern": "FAILED BREAKOUT TRAP",
                    
                        "bias": "bear",
                        "headline": "Price broke higher, then reversed hard on heavy volume – classic bull trap.",
                        "explanation": (
                            "After recently attempting to move higher, the stock is now reversing sharply down "
                            "on strong volume. This pattern often marks failed breakouts where traders who "
                            "chased the move higher are now being forced to exit at a loss."
                        ),
                    },
                )
            )

    # 9) INSIDE RANGE COMPRESSION – energy coiling
    if ok(change) and ok(ret3) and ok(vol_vs_ma):
        if abs(change) < 0.8 and abs(ret3 or 0) < 2.0 and vol_vs_ma < 0:
            patterns.append(
                (
                    0.6,
                    {
                        "pattern": "INSIDE RANGE COMPRESSION",
                   
                        "bias": "neutral",
                        "headline": "Price is consolidating in a tight range after recent moves.",
                        "explanation": (
                            "The last few days show relatively small net movement and below-average volume. "
                            "This kind of quiet consolidation can precede a larger directional move once a new "
                            "trend leader emerges."
                        ),
                    },
                )
            )

    # 10) HIGH-WAVE INDECISION – long wicks both sides
    if ok(upper_shadow) and ok(lower_shadow) and ok(body_pct):
        if upper_shadow > 30.0 and lower_shadow > 30.0 and abs(body_pct) < 20.0:
            patterns.append(
                (
                    0.58,
                    {
                        "pattern": "HIGH-WAVE INDECISION",
                  
                        "bias": "neutral",
                        "headline": "Buyers and sellers both swung hard, but neither side won clearly.",
                        "explanation": (
                            "Today’s candle shows long upper and lower wicks with a small real body, "
                            "signaling strong intraday tug-of-war without a decisive close. Markets often "
                            "pause or pivot after such high-uncertainty sessions."
                        ),
                    },
                )
            )

    # 11) TREND ACCELERATION – trend with fresh follow-through
    if ok(trend) and ok(change) and ok(vol_vs_ma):
        if trend > 15.0 and change > 1.5 and vol_vs_ma > 5.0:
            patterns.append(
                (
                    0.7,
                    {
                        "pattern": "TREND ACCELERATION",
                      
                        "bias": "bull",
                        "headline": "Existing uptrend just got a fresh burst of momentum.",
                        "explanation": (
                            "The stock had already been trending higher and now shows another solid up day on "
                            "above-average volume. This kind of continuation behavior is typical of sustained "
                            "institutional accumulation phases."
                        ),
                    },
                )
            )

    # 12) GAP DOWN & PRESSURE – controlled selloff
    if ok(gap) and ok(change):
        if gap < -1.0 and change < -2.0:
            patterns.append(
                (
                    0.68,
                    {
                        "pattern": "GAP DOWN & PRESSURE",
                     
                        "bias": "bear",
                        "headline": "Stock opened sharply lower and sellers kept control.",
                        "explanation": (
                            "The session started with a clear downside gap versus yesterday and continued to "
                            "fade through the day. This can reflect negative news or widespread risk-off behavior "
                            "where buyers step aside rather than defend prior levels."
                        ),
                    },
                )
            )

    # -------------------------------------------------
    # STEP 17: Elite Pattern Enforcement
    # -------------------------------------------------
    patterns = [
        (score, p) for score, p in patterns
        if p["pattern"] not in DEPRECATED_PATTERNS
    ]
    if not patterns:
        return None

    # Pick the pattern with the highest internal score
    patterns.sort(key=lambda x: x[0], reverse=True)
    return patterns[0][1]


def scan_smart_pattern_history(
    symbol: str,
    candles: dict,
    lookahead_5: int = 5,
    lookahead_10: int = 10,
):
    """Scan ~1 year of daily candles and compute smart-pattern stats.

    Returns a dict with:
      - currentPattern: pattern dict for the most recent day (or None)
      - historyForCurrent: aggregated stats where the same pattern appeared in the past
      - allPatterns: basic counts for all detected patterns
    """
    closes = np.array(candles["close"], dtype=float)
    highs = np.array(candles["high"], dtype=float)
    lows = np.array(candles["low"], dtype=float)
    opens = np.array(candles["open"], dtype=float)
    vols = np.array(candles["volume"], dtype=float)
    ts_list = candles.get("timestamp") or []

    n = len(closes)
    if n < 40:
        return {
            "currentPattern": None,
            "historyForCurrent": None,
            "allPatterns": [],
            "note": "Not enough history to compute pattern stats.",
        }

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )

    # Timestamps → ISO
    if ts_list:
        df["ts"] = [
            datetime.datetime.utcfromtimestamp(t / 1000.0)
            .replace(microsecond=0)
            .isoformat()
            + "Z"
            if t
            else None
            for t in ts_list
        ]
    else:
        base = datetime.datetime.utcnow().replace(microsecond=0)
        df["ts"] = [
            (base - datetime.timedelta(days=(n - 1 - i))).isoformat() + "Z"
            for i in range(n)
        ]

    # Daily change & gap%
    df["changePct"] = df["close"].pct_change() * 100.0
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100.0

    # Volume stats vs 20d mean
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["vol_std20"] = df["volume"].rolling(20, min_periods=20).std()
    df["volume_vs_ma20_pct"] = (df["volume"] / df["vol_ma20"] - 1.0) * 100.0
    df["volume_zscore_20"] = (df["volume"] - df["vol_ma20"]) / df["vol_std20"]

    # RSI & Williams %R
    df["rsi14"] = _compute_rsi(df["close"], period=14)
    df["williams_r_14"] = _compute_williams_r(
        df["high"], df["low"], df["close"], period=14
    )

    # Candle anatomy (upper/lower wicks, body)
    full_range = df["high"] - df["low"]
    body = df["close"] - df["open"]
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    df["body_pct"] = np.where(full_range > 0, body / full_range * 100.0, 0.0)
    df["lower_shadow_pct"] = np.where(full_range > 0, lower / full_range * 100.0, 0.0)
    df["upper_shadow_pct"] = np.where(full_range > 0, upper / full_range * 100.0, 0.0)

    # Trend / distance from 20d trend
    df["sma20"] = df["close"].rolling(20, min_periods=20).mean()
    df["price_vs_sma20_pct"] = (df["close"] / df["sma20"] - 1.0) * 100.0
    df["trend_strength_20"] = (
        df["close"] / df["close"].shift(20) - 1.0
    ) * 100.0

    # 3-day and 5-day trailing returns
    df["ret3"] = df["close"].pct_change(3) * 100.0
    df["return_5d"] = df["close"].pct_change(5) * 100.0

    # Forward returns AFTER pattern
    df["fwd_5d"] = df["close"].shift(-lookahead_5) / df["close"] - 1.0
    df["fwd_10d"] = df["close"].shift(-lookahead_10) / df["close"] - 1.0

    pattern_rows = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        patt = _evaluate_smart_pattern_row(
            gap=row.get("gap_pct"),
            change=row.get("changePct"),
            vol_z=row.get("volume_zscore_20"),
            vol_vs_ma=row.get("volume_vs_ma20_pct"),
            rsi=row.get("rsi14"),
            will_r=row.get("williams_r_14"),
            lower_shadow=row.get("lower_shadow_pct"),
            upper_shadow=row.get("upper_shadow_pct"),
            body_pct=row.get("body_pct"),
            price_vs_sma20=row.get("price_vs_sma20_pct"),
            trend=row.get("trend_strength_20"),
            ret3=row.get("ret3"),
            ret5=row.get("return_5d"),
        )
        if not patt:
            continue

        pattern_rows.append(
            {
                "date": row["ts"],
                "pattern": patt["pattern"],
                "patternLabel": ELITE_PATTERN_LABELS.get(patt["pattern"], patt["pattern"]),
                "headline": patt["headline"],
               
                "bias": patt.get("bias"),
                "fwd_5d": float(row["fwd_5d"]) if pd.notna(row["fwd_5d"]) else None,
                "fwd_10d": float(row["fwd_10d"]) if pd.notna(row["fwd_10d"]) else None,
                "changePct": float(row["changePct"])
                if pd.notna(row["changePct"])
                else None,
            }
        )

    if not pattern_rows:
        return {
            "currentPattern": None,
            "historyForCurrent": None,
            "allPatterns": [],
            "note": "No recognizable smart patterns in the available history.",
        }

    # Current pattern = last valid pattern in history (ideally last trading day)
    current = pattern_rows[-1]
    current["patternLabel"] = ELITE_PATTERN_LABELS.get(
        current["pattern"], current["pattern"]
    )
    current_name = current["pattern"]

    from collections import defaultdict

    counts = defaultdict(int)
    for r in pattern_rows:
        counts[r["pattern"]] += 1

    all_patterns = [
        {"pattern": name, "occurrences": cnt} for name, cnt in counts.items()
    ]
    all_patterns.sort(key=lambda x: x["occurrences"], reverse=True)

    # Filter rows matching current pattern (excluding today for forward stats)
    history_matches = [r for r in pattern_rows[:-1] if r["pattern"] == current_name]
    # -------------------------------------------------
    # STEP 3A: Win-rate helper (positive forward return)
    # -------------------------------------------------
    def _win_rate(values):
        if not values:
            return None
        wins = [v for v in values if v > 0]
        return round(len(wins) / len(values), 4)


    def _agg(field: str):
        vals = [r[field] * 100.0 for r in history_matches if r[field] is not None]
        if not vals:
            return None

        return {
            "avg": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "best": float(np.max(vals)),
            "worst": float(np.min(vals)),
            "count": len(vals),
            "winRate": _win_rate(vals),  # ✅ STEP-3: dynamic win rate
        }

    stats_5d = _agg("fwd_5d")
    stats_10d = _agg("fwd_10d")

    # Last few occurrences (excluding today)
    sample_events = history_matches[-5:] if history_matches else []

    history_block = {
        "pattern": current_name,
        "occurrences": counts[current_name],
        "samples": sample_events,
        "forwardReturns": {
            "days5": stats_5d,
            "days10": stats_10d,
        },
    }

    return {
        "currentPattern": current,
        "historyForCurrent": history_block,
        "allPatterns": all_patterns,
        "note": None,
    }


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

def run_bullbrain_from_inputs(
    symbol: str,
    *,
    candles_arrays: dict,
    feat_dict: dict
) -> dict:
    """
    PURE BullBrain execution.
    No I/O. No fetching. No recomputation.

    Inputs:
      - candles_arrays: dict-of-arrays (open/high/low/close/volume/timestamp)
      - feat_dict: computed feature dictionary

    Returns:
      core decision payload used by cron + Firestore
    """

    # ---------------------------------------------------------
    # 1) Pattern scan (single authoritative engine)
    # ---------------------------------------------------------
    try:
        pattern_result = scan_smart_pattern_history(symbol, candles_arrays)
    except Exception:
        pattern_result = None

    # ---------------------------------------------------------
    # 2) ML prediction (SINGLE REAL MODEL PATH)
    # ---------------------------------------------------------
    # Rebuild feature vector in correct order
    values = [feat_dict.get(name) for name in BULLBRAIN_FEATURES]
    features_vector = np.array([values], dtype=float)

    inference = bullbrain_infer(features_vector)

    prob_up = float(inference.get("probability_up"))
    prob_down = float(inference.get("probability_down"))
    confidence = float(inference.get("confidence"))
    model_signal = inference.get("signal")

    # ---------------------------------------------------------
    # 3) STEP-16 Decision Ladder (single authority)
    # ---------------------------------------------------------
    decision = final_decision(
        model_signal=model_signal,
        features=feat_dict,
        pattern_name=(
            pattern_result.get("currentPattern", {}).get("pattern")
            if pattern_result else None
        ),
        pattern_history=(
            pattern_result.get("historyForCurrent")
            if pattern_result else None
        ),
        total_days=len(candles_arrays.get("close", [])),
    )

    final_signal = decision["finalSignal"]

    # ---------------------------------------------------------
    # 4) Normalized output (CONTRACT)
    # ---------------------------------------------------------
    return {
        "bullbrain": {
            "signal": final_signal,
            "confidence": confidence,
            "raw": {
                "prob_up": prob_up,
                "prob_down": prob_down,
            },
        },
        "decision": decision,
        "pattern": pattern_result.get("currentPattern") if pattern_result else None,
        "patternBias": (
            pattern_bias(
                pattern_result.get("currentPattern", {}).get("pattern")
            ) if pattern_result else None
        ),
        "patternHistory": (
            pattern_result.get("historyForCurrent")
            if pattern_result else None
        ),
    }

# --------------------------------------------------------------------
# CORE PIPELINE FOR ONE SYMBOL (FINAL – STEP-16 AUTHORITY)
# --------------------------------------------------------------------
def _run_bullbrain_for_symbol(symbol: str):
    symbol = symbol.upper()

    if bullbrain_model is None:
        return None, {"error": "BullBrain model not loaded yet."}

    candles = get_candles(symbol, min_points=120)
    if not candles:
        return None, {"error": f"Could not fetch candles for {symbol}"}

    # -------------------------------------------------
    # 1️⃣ Feature computation + model inference
    # -------------------------------------------------
    features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
    inference = bullbrain_infer(features_vec)

    model_signal = inference.get("signal")
    bull_conf_raw = float(inference.get("confidence") or 0.0)

    prob_up = inference.get("probability_up")
    if prob_up is None:
        prob_up = float(inference.get("raw_output", 0.5))
    prob_down = 1.0 - float(prob_up)
    class_probs = _class_probs_from_prob_up(prob_up)

    # -------------------------------------------------
    # 2️⃣ Pattern scan (single authoritative source)
    # -------------------------------------------------
    current_pattern = None
    pattern_history = None
    patt_name = None
    patt_bias = "neutral"

    try:
        pattern_scan = scan_smart_pattern_history(symbol, candles)
        current_pattern = pattern_scan.get("currentPattern")
        pattern_history = pattern_scan.get("historyForCurrent")

        if isinstance(current_pattern, dict):
            patt_name = current_pattern.get("pattern")

        patt_bias = pattern_bias(patt_name)

    except Exception as e:
        print("pattern scan error:", e)

    # -------------------------------------------------
    # 3️⃣ STEP-16 FINAL DECISION (SINGLE AUTHORITY)
    # -------------------------------------------------
    decision = final_decision(
        model_signal=model_signal,
        features=feature_dict,
        pattern_name=patt_name,
        pattern_history=pattern_history,
        total_days=len(candles.get("close", [])),
    )

    final_signal = decision["finalSignal"]
    decision_reasons = decision["decisionReasons"]
    quality = decision["quality"]

    # -------------------------------------------------
    # 4️⃣ Confidence (DESCRIPTIVE ONLY – no gating)
    # -------------------------------------------------
    regime_ok = quality.get("regime") is not None

    bull_conf = recalibrate_confidence(
        bull_conf_raw,
        pattern_history,
        regime_ok,
    )

    # Freshness + decay (post-decision hygiene)
    as_of = datetime.datetime.utcnow().isoformat()
    SIGNAL_MAX_AGE_HOURS = 24

    is_fresh = signal_is_fresh(as_of, SIGNAL_MAX_AGE_HOURS)
    if not is_fresh:
        final_signal = "HOLD"

    bull_conf = apply_confidence_decay(
        bull_conf,
        as_of,
        SIGNAL_MAX_AGE_HOURS,
    )

    signal_strength = derive_signal_strength(final_signal, bull_conf)

    # -------------------------------------------------
    # 5️⃣ Grok hybrid (informational, not authoritative)
    # -------------------------------------------------
    try:
        grok_p, grok_summary = grok_prob_up(symbol)
    except Exception as e:
        print("grok_prob_up fatal:", e)
        grok_p, grok_summary = 50.0, "Neutral sentiment (error while calling Grok)."

    hybrid_score, hybrid_signal = compute_hybrid_signal(bull_conf, grok_p)

    # -------------------------------------------------
    # 6️⃣ Final payload
    # -------------------------------------------------
    core = {
        "symbol": symbol,
        "asOf": as_of,
        "source": candles.get("source", "polygon"),
        "price": last_close,

        "features": feature_dict,

        "bullbrain": {
            "version": BULLBRAIN_VERSION,
            "signal": final_signal,
            "strength": signal_strength,
            "confidence": bull_conf,
            "probabilities": class_probs,
            "raw": {
                "prob_up": float(prob_up),
                "prob_down": float(prob_down),
            },
        },

        "decision": {
            "finalSignal": final_signal,
            "reasons": decision_reasons,
            "quality": quality,
        },

        "pattern": current_pattern,
        "patternBias": patt_bias,
        "patternHistory": pattern_history,

        "signalFresh": is_fresh,
        "signalExpiryHours": SIGNAL_MAX_AGE_HOURS,

        "model": inference,

        "grokProbUp": float(grok_p),
        "grokSummary": grok_summary,
        "hybridScore": float(hybrid_score),
        "hybridSignal": hybrid_signal,
    }

    return core, None

# -----------------------------------------------------------
# STEP 1: Pattern Bias Normalization
# -----------------------------------------------------------
def pattern_bias(pattern_name: str | None) -> str:
    if not pattern_name:
        return "neutral"

    name = pattern_name.upper()

    BULLISH = {
        "GAP UP & RUNNING",
        "VOLUME BREAKOUT",
        "MASSIVE VOLUME BREAKOUT",
        "OVERSOLD BOUNCE",
        "HAMMER REVERSAL",
        "BUY THE DIP (UPTREND)",
        "TREND ACCELERATION",
    }

    BEARISH = {
        "DEAD CAT BOUNCE",
        "OVERBOUGHT DISTRIBUTION",
        "FAILED BREAKOUT TRAP",
        "GAP DOWN & PRESSURE",
        "BEAR FLAG BREAKDOWN",
    }

    if name in BULLISH:
        return "bull"
    if name in BEARISH:
        return "bear"
    return "neutral"


# -----------------------------------------------------------
# STEP 2: Signal–Pattern Alignment Filter
# -----------------------------------------------------------
def alignment_filter(model_signal: str | None, patt_bias: str) -> bool:
    if not model_signal:
        return False

    signal = model_signal.upper()
    bias = (patt_bias or "neutral").lower()

    if bias == "neutral":
        return True
    if bias == "bull" and signal == "SELL":
        return False
    if bias == "bear" and signal == "BUY":
        return False
    return True

# -----------------------------------------------------------
# STEP 4: Pattern Forward-Return Quality Gate
# -----------------------------------------------------------

def pattern_quality_gate(history_block: dict | None) -> bool:
    """
    Determine whether a detected pattern is statistically valid
    based on historical forward returns.

    Returns True if pattern quality is acceptable.
    """

    if not history_block:
        return False

    fwd = history_block.get("forwardReturns", {})
    days5 = fwd.get("days5")

    if not days5:
        return False

    win_rate = days5.get("winRate")
    avg_return = days5.get("avg")
    count = days5.get("count", 0)

    # --- Hard minimum requirements ---
    MIN_SAMPLES = 20
    MIN_WINRATE = 0.65
    MIN_AVG_RETURN = 0.0

    if win_rate is None:
        return False

    if count < MIN_SAMPLES:
        return False

    if win_rate < MIN_WINRATE:
        return False

    if avg_return is None or avg_return <= MIN_AVG_RETURN:
        return False

    return True

# -----------------------------------------------------------
# STEP 5: Market Regime Detection
# -----------------------------------------------------------

def detect_market_regime(features: dict) -> str:
    """
    Detect market regime using existing features.
    Returns: 'TRENDING', 'RANGING', 'HIGH_VOL'
    """

    trend = features.get("trend_strength_20")
    vol20 = features.get("volatility_20d")
    vol60 = features.get("volatility_60d")
    atr = features.get("atr14")

    # Defensive
    if trend is None or vol20 is None:
        return "UNKNOWN"

    # High volatility regime
    if vol20 > 1.5 * (vol60 or vol20) or (atr and atr > 1.2 * vol20):
        return "HIGH_VOL"

    # Strong directional trend
    if abs(trend) > 0.4:
        return "TRENDING"

    # Otherwise range-bound
    return "RANGING"

# -----------------------------------------------------------
# STEP 7: Multi-Timeframe Agreement Gate
# -----------------------------------------------------------

def timeframe_alignment(features: dict, direction: str) -> bool:
    """
    Require agreement across 1D, 5D, and 10D returns.

    BUY  -> all returns >= 0
    SELL -> all returns <= 0
    """

    try:
        r1 = float(features.get("return_1d"))
        r5 = float(features.get("return_5d"))
        r10 = float(features.get("return_10d"))
    except (TypeError, ValueError):
        return False

    if direction == "BUY":
        return r1 >= 0 and r5 >= 0 and r10 >= 0

    if direction == "SELL":
        return r1 <= 0 and r5 <= 0 and r10 <= 0

    return False

# -----------------------------------------------------------
# STEP 8: Mandatory Volume Rule
# -----------------------------------------------------------

def volume_gate(features: dict) -> bool:
    """
    Enforce volume confirmation.

    BUY / SELL allowed only if:
    - volume_zscore_20 >= 0.5
    - volume_vs_ma20_pct >= 0
    """

    try:
        vol_z = float(features.get("volume_zscore_20"))
        vol_vs_ma = float(features.get("volume_vs_ma20_pct"))
    except (TypeError, ValueError):
        return False

    if vol_z < 0.5:
        return False

    if vol_vs_ma < 0:
        return False

    return True

# -----------------------------------------------------------
# STEP 9: Feature Consensus Score
# -----------------------------------------------------------

def feature_consensus_score(features: dict) -> int:
    """
    Compute directional consensus across feature groups:
    - Trend
    - Momentum
    - Volume

    Returns an integer score in range [-3, +3].
    """

    score = 0

    # ---- Trend vote ----
    trend = features.get("trend_strength_20")
    if trend is not None:
        if trend > 0:
            score += 1
        elif trend < 0:
            score -= 1

    # ---- Momentum vote ----
    rsi = features.get("rsi14")
    macd_hist = features.get("macd_hist")

    if rsi is not None and macd_hist is not None:
        if rsi > 50 and macd_hist > 0:
            score += 1
        elif rsi < 50 and macd_hist < 0:
            score -= 1

    # ---- Volume vote ----
    vol_z = features.get("volume_zscore_20")
    vol_vs_ma = features.get("volume_vs_ma20_pct")

    if vol_z is not None and vol_vs_ma is not None:
        if vol_z > 0 and vol_vs_ma > 0:
            score += 1
        elif vol_z < 0 and vol_vs_ma < 0:
            score -= 1

    return score

# -----------------------------------------------------------
# STEP 10: Directional Pressure Score
# -----------------------------------------------------------

def directional_pressure(features: dict) -> int:
    """
    Compute net directional pressure using:
    - Short / medium-term returns
    - MACD histogram
    - OBV slope

    Returns an integer score in range [-3, +3].
    """

    score = 0

    # ---- Returns pressure ----
    r1 = features.get("return_1d")
    r5 = features.get("return_5d")

    if r1 is not None and r5 is not None:
        if r1 > 0 and r5 > 0:
            score += 1
        elif r1 < 0 and r5 < 0:
            score -= 1

    # ---- Momentum pressure (MACD) ----
    macd_hist = features.get("macd_hist")
    if macd_hist is not None:
        if macd_hist > 0:
            score += 1
        elif macd_hist < 0:
            score -= 1

    # ---- Volume pressure (OBV slope) ----
    obv_slope = features.get("obv_slope_10")
    if obv_slope is not None:
        if obv_slope > 0:
            score += 1
        elif obv_slope < 0:
            score -= 1

    return score

# -----------------------------------------------------------
# STEP 11: Signal Fragility Index
# -----------------------------------------------------------

def signal_fragility(features: dict) -> int:
    """
    Detect fragile / unstable setups.

    Returns an integer fragility score:
    0 = very stable
    1–2 = moderate risk
    >=3 = fragile → should be HOLD
    """

    fragility = 0

    intraday_range = features.get("intraday_range_pct")
    body_pct = features.get("body_pct")
    vol_z = features.get("volume_zscore_20")
    vol20 = features.get("volatility_20d")

    # 1️⃣ Wide intraday swings → instability
    if intraday_range is not None and intraday_range > 5.0:
        fragility += 1

    # 2️⃣ Small candle body with big range → indecision
    if body_pct is not None and abs(body_pct) < 20.0:
        fragility += 1

    # 3️⃣ Abnormal volatility regime
    if vol20 is not None and vol20 > 4.0:
        fragility += 1

    # 4️⃣ Thin or suspicious volume
    if vol_z is not None and vol_z < -0.5:
        fragility += 1

    return fragility

# -----------------------------------------------------------
# STEP 12: Liquidity Quality
# -----------------------------------------------------------

def liquidity_quality(features: dict) -> str:
    """
    Classify liquidity quality using volume and volatility behavior.

    Returns:
    - 'GOOD'
    - 'THIN'
    - 'POOR'
    """

    vol_z = features.get("volume_zscore_20")
    vol_vs_ma20 = features.get("volume_vs_ma20_pct")
    intraday_range = features.get("intraday_range_pct")
    vol20 = features.get("volatility_20d")

    # Defensive
    if vol_z is None or vol_vs_ma20 is None:
        return "POOR"

    # --- POOR liquidity ---
    if vol_z < -1.0 or vol_vs_ma20 < -20:
        return "POOR"

    # --- THIN liquidity ---
    if vol_z < 0.3 or vol_vs_ma20 < 0:
        return "THIN"

    # --- Volatility-driven illiquidity ---
    if intraday_range is not None and vol20 is not None:
        if intraday_range > 6.0 and vol20 > 4.0:
            return "THIN"

    return "GOOD"

# -----------------------------------------------------------
# STEP 13: Momentum Exhaustion Detector
# -----------------------------------------------------------

def momentum_exhaustion(features: dict, direction: str) -> bool:
    """
    Detect momentum exhaustion for BUY or SELL direction.

    Returns True if exhaustion is detected (signal should be HOLD).
    """

    rsi = features.get("rsi14")
    willr = features.get("williams_r_14")
    dist_high = features.get("distance_from_20d_high")
    dist_low = features.get("distance_from_20d_low")
    vol_vs_ma20 = features.get("volume_vs_ma20_pct")
    macd_hist = features.get("macd_hist")

    direction = direction.upper() if direction else ""

    # ---------------- BUY exhaustion ----------------
    if direction == "BUY":
        if (
            rsi is not None and rsi > 72 and
            willr is not None and willr > -10 and
            dist_high is not None and dist_high > -1.0 and
            vol_vs_ma20 is not None and vol_vs_ma20 < 0 and
            macd_hist is not None and macd_hist < 0
        ):
            return True

    # ---------------- SELL exhaustion ----------------
    if direction == "SELL":
        if (
            rsi is not None and rsi < 28 and
            willr is not None and willr < -90 and
            dist_low is not None and dist_low > -1.0 and
            vol_vs_ma20 is not None and vol_vs_ma20 < 0 and
            macd_hist is not None and macd_hist > 0
        ):
            return True

    return False

# -----------------------------------------------------------
# STEP 14: Expected Value (EV) Score
# -----------------------------------------------------------

def expected_value_score(
    pattern_history: dict | None,
    fragility: int,
) -> float:
    """
    Compute Expected Value (EV) of the setup.

    EV combines:
    - Historical win rate
    - Average forward return
    - Fragility penalty

    Returns a float EV score.
    EV <= 0 → bad trade
    """

    if not pattern_history:
        return -1.0

    fwd = pattern_history.get("forwardReturns", {})
    days5 = fwd.get("days5")

    if not days5:
        return -1.0

    win_rate = days5.get("winRate")
    avg_ret = days5.get("avg")

    if win_rate is None or avg_ret is None:
        return -1.0

    # --- Core EV ---
    # Expected gain = win_rate * avg_return
    ev = win_rate * avg_ret

    # --- Fragility penalty ---
    # Each fragility point reduces EV
    ev -= fragility * 0.5

    return round(ev, 3)

# -----------------------------------------------------------
# STEP 15: Signal Rarity Index
# -----------------------------------------------------------

def signal_rarity(
    pattern_history: dict | None,
    total_days: int,
) -> float:
    """
    Compute rarity of a signal.

    Rarity = occurrences / total scanned days

    Returns a float between 0 and 1:
    - < 0.05 → very rare (high quality)
    - 0.05–0.15 → selective
    - > 0.20 → common (lower edge)
    """

    if not pattern_history or not total_days or total_days <= 0:
        return 1.0  # treat unknown as very common

    occurrences = pattern_history.get("occurrences")
    if not occurrences:
        return 1.0

    rarity = occurrences / total_days
    return round(min(max(rarity, 0.0), 1.0), 4)

def momentum_override_signal(
    *,
    model_signal: str,
    features: dict,
    pattern_name: str | None,
) -> str | None:
    """
    Allows strong real-time momentum setups to surface as BUY/SELL
    without weakening the whole decision ladder.
    Conservative App Store-safe override.
    """

    try:
        r1 = float(features.get("return_1d") or 0)
        r5 = float(features.get("return_5d") or 0)
        vol_vs_ma20 = float(features.get("volume_vs_ma20_pct") or 0)
        rsi = float(features.get("rsi14") or 50)
        macd_hist = float(features.get("macd_hist") or 0)
        trend = float(features.get("trend_strength_20") or 0)
    except Exception:
        return None

    patt_bias = pattern_bias(pattern_name)

    bullish_patterns = {
        "GAP UP & RUNNING",
        "VOLUME BREAKOUT",
        "HAMMER REVERSAL",
        "TREND ACCELERATION",
        "BUY THE DIP (UPTREND)",
        "OVERSOLD BOUNCE",
    }

    bearish_patterns = {
        "FAILED BREAKOUT TRAP",
        "GAP DOWN & PRESSURE",
        "OVERBOUGHT DISTRIBUTION",
        "DEAD CAT BOUNCE",
    }

    # Strong upside mover
    if (
        r1 >= 2.0
        and r5 >= 0
        and vol_vs_ma20 >= 10
        and macd_hist >= 0
        and rsi < 78
        and (
            patt_bias == "bull"
            or pattern_name in bullish_patterns
            or model_signal == "BUY"
        )
    ):
        return "BUY"

    # Strong downside mover
    if (
        r1 <= -2.0
        and r5 <= 0
        and vol_vs_ma20 >= 10
        and macd_hist <= 0
        and rsi > 22
        and (
            patt_bias == "bear"
            or pattern_name in bearish_patterns
            or model_signal == "SELL"
        )
    ):
        return "SELL"

    return None
# -----------------------------------------------------------
# STEP 16: Final Decision Ladder (Single Authority)
# -----------------------------------------------------------

def final_decision(
    *,
    model_signal: str,
    features: dict,
    pattern_name: str | None,
    pattern_history: dict | None,
    total_days: int,
) -> dict:
    """
    Enforces the full decision ladder.
    If ANY gate fails → HOLD.

    Returns:
    {
        "finalSignal": "BUY|SELL|HOLD",
        "decisionReasons": [...],
        "quality": {...}
    }
    """

    reasons = []

    # ---------------- 1️⃣ Liquidity ----------------
    liq = liquidity_quality(features)
    if liq != "GOOD":
        reasons.append(f"Liquidity={liq}")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"liquidity": liq}}

    # ---------------- 1.5️⃣ Momentum Override ----------------
    override = momentum_override_signal(
        model_signal=model_signal,
        features=features,
        pattern_name=pattern_name,
    )

    if override:
        return {
            "finalSignal": override,
            "decisionReasons": ["MOMENTUM_OVERRIDE"],
            "quality": {
                "liquidity": liq,
                "override": True,
                "overrideType": "strong_price_volume_momentum",
                "originalModelSignal": model_signal,
                "pattern": pattern_name,
            },
        }
    
    # ---------------- 2️⃣ Market Regime ----------------
    regime = detect_market_regime(features)

    # ---------------- 3️⃣ Pattern Quality ----------------
    if not pattern_quality_gate(pattern_history):
        reasons.append("PatternQualityFailed")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"regime": regime}}

    # ---------------- 4️⃣ Regime Compatibility ----------------
    if pattern_name:
        allowed = PATTERN_REGIME_COMPATIBILITY.get(pattern_name)
        if allowed and regime not in allowed:
            reasons.append(f"PatternNotAllowedIn{regime}")
            return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"regime": regime}}

    # ---------------- 5️⃣ Pattern–Model Alignment ----------------
    patt_bias = pattern_bias(pattern_name)
    if not alignment_filter(model_signal, patt_bias):
        reasons.append("SignalPatternConflict")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {}}

    # ---------------- 6️⃣ Multi-Timeframe Agreement ----------------
    if not timeframe_alignment(features, model_signal):
        reasons.append("TimeframeMisalignment")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {}}

    # ---------------- 7️⃣ Volume Confirmation ----------------
    if not volume_gate(features):
        reasons.append("VolumeGateFailed")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {}}

    # ---------------- 8️⃣ Feature Consensus ----------------
    consensus = feature_consensus_score(features)
    if abs(consensus) < 2:
        reasons.append("WeakFeatureConsensus")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"consensus": consensus}}

    # ---------------- 9️⃣ Directional Pressure ----------------
    pressure = directional_pressure(features)
    if model_signal == "BUY" and pressure <= 0:
        reasons.append("NoUpsidePressure")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"pressure": pressure}}
    if model_signal == "SELL" and pressure >= 0:
        reasons.append("NoDownsidePressure")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"pressure": pressure}}

    # ---------------- 🔟 Fragility ----------------
    frag = signal_fragility(features)
    if frag >= 3:
        reasons.append("SignalTooFragile")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"fragility": frag}}

    # ---------------- 1️⃣1️⃣ Momentum Exhaustion ----------------
    if momentum_exhaustion(features, model_signal):
        reasons.append("MomentumExhausted")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {}}

    # ---------------- 1️⃣2️⃣ Expected Value ----------------
    ev = expected_value_score(pattern_history, frag)
    if ev <= 0:
        reasons.append("NegativeEV")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {"EV": ev}}

    # ---------------- 1️⃣3️⃣ Rarity (context only) ----------------
    rarity = signal_rarity(pattern_history, total_days)

    # ---------------- ✅ PASSED ALL GATES ----------------
    return {
        "finalSignal": model_signal,
        "decisionReasons": ["ALL_GATES_PASSED"],
        "quality": {
            "liquidity": liq,
            "regime": regime,
            "consensus": consensus,
            "pressure": pressure,
            "fragility": frag,
            "EV": ev,
            "rarity": rarity,
        },
    }

# -----------------------------------------------------------
# STEP 5: Pattern ↔ Regime Compatibility
# -----------------------------------------------------------

PATTERN_REGIME_COMPATIBILITY = {
    # Trend continuation
    "TREND ACCELERATION": {"TRENDING"},
    "BUY THE DIP (UPTREND)": {"TRENDING"},
    "BULL FLAG": {"TRENDING"},
    "BEAR FLAG BREAKDOWN": {"TRENDING"},

    # Breakouts
    "GAP UP & RUNNING": {"TRENDING", "HIGH_VOL"},
    "VOLUME BREAKOUT": {"TRENDING", "HIGH_VOL"},
    "FAILED BREAKOUT TRAP": {"HIGH_VOL"},

    # Mean reversion
    "OVERSOLD BOUNCE": {"RANGING", "HIGH_VOL"},
    "HAMMER REVERSAL": {"RANGING"},
    "DEAD CAT BOUNCE": {"HIGH_VOL"},

    # Neutral / compression
    "INSIDE RANGE COMPRESSION": {"RANGING"},
    "HIGH-WAVE INDECISION": {"RANGING"},
}

# -----------------------------------------------------------
# STEP 6: Confidence Recalibration
# -----------------------------------------------------------

def recalibrate_confidence(
    model_conf: float,
    pattern_history: dict | None,
    regime_ok: bool,
) -> float:
    """
    Adjust model confidence using:
    - Pattern historical win rate
    - Forward returns strength
    - Market regime compatibility
    """

    conf = float(model_conf or 0.0)

    if not pattern_history:
        return round(conf, 2)

    days5 = pattern_history.get("forwardReturns", {}).get("days5")
    if not days5:
        return round(conf, 2)

    win_rate = days5.get("winRate")
    avg_ret = days5.get("avg")

    # --- Pattern strength adjustments ---
    if win_rate is not None:
        if win_rate >= 0.75:
            conf += 8
        elif win_rate >= 0.70:
            conf += 5
        elif win_rate < 0.60:
            conf -= 10

    if avg_ret is not None:
        if avg_ret >= 2.0:
            conf += 5
        elif avg_ret <= 0:
            conf -= 8

    # --- Regime penalty ---
    if not regime_ok:
        conf -= 15

    # Clamp
    conf = max(0.0, min(100.0, conf))
    return round(conf, 2)

# -----------------------------------------------------------
# STEP 7: Signal Strength Tiering
# -----------------------------------------------------------

def derive_signal_strength(signal: str, confidence: float) -> str:
    """
    Convert signal + confidence into strength tier.
    """

    if signal == "HOLD":
        return "HOLD"

    if signal == "BUY":
        if confidence >= 80:
            return "STRONG_BUY"
        if confidence >= 65:
            return "BUY"
        return "WEAK_BUY"

    if signal == "SELL":
        if confidence >= 80:
            return "STRONG_SELL"
        if confidence >= 65:
            return "SELL"
        return "WEAK_SELL"

    return "HOLD"

# -----------------------------------------------------------
# STEP 8: Signal Expiry & Freshness Control
# -----------------------------------------------------------

def signal_is_fresh(as_of_iso: str, max_age_hours: int = 24) -> bool:
    """
    Check whether a signal is still fresh based on its timestamp.
    """
    try:
        ts = datetime.datetime.fromisoformat(as_of_iso)
        age = datetime.datetime.utcnow() - ts
        return age.total_seconds() <= max_age_hours * 3600
    except Exception:
        return False
# -----------------------------------------------------------
# STEP 9: Confidence Time Decay
# -----------------------------------------------------------

def apply_confidence_decay(
    confidence: float,
    as_of_iso: str,
    max_age_hours: int,
    min_conf_floor: float = 40.0,
) -> float:
    """
    Linearly decay confidence over time until expiry.
    """

    try:
        ts = datetime.datetime.fromisoformat(as_of_iso)
        age_hours = (datetime.datetime.utcnow() - ts).total_seconds() / 3600.0
    except Exception:
        return round(confidence, 2)

    if age_hours <= 0:
        return round(confidence, 2)

    if age_hours >= max_age_hours:
        return round(min_conf_floor, 2)

    decay_ratio = age_hours / max_age_hours
    decayed = confidence * (1 - decay_ratio)

    return round(max(decayed, min_conf_floor), 2)

# -----------------------------------------------------------
# STEP 10: Probability Conflict Resolver
# -----------------------------------------------------------

def resolve_model_pattern_conflict(
    signal: str,
    model_prob_up: float,
    pattern_history: dict | None,
    min_agreement: float = 0.10,
) -> str:
    """
    Ensure model and pattern probabilities do not contradict.
    If disagreement exceeds threshold → HOLD.
    """

    if signal == "HOLD" or not pattern_history:
        return signal

    days5 = pattern_history.get("forwardReturns", {}).get("days5")
    if not days5:
        return signal

    patt_win = days5.get("winRate")
    if patt_win is None:
        return signal

    patt_prob_up = patt_win
    patt_prob_down = 1.0 - patt_win

    if signal == "BUY":
        if patt_prob_up + min_agreement < model_prob_up:
            return "HOLD"

    if signal == "SELL":
        if patt_prob_down + min_agreement < (1.0 - model_prob_up):
            return "HOLD"

    return signal

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

# ---------------------------------------------------------------
# Astra LLM Helper (Grok via XAI)
# ---------------------------------------------------------------
def astra_llm_answer(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Calls Grok (XAI) to generate a natural language answer.
    Returns None on failure so we can gracefully fall back.
    """
    try:
        if not XAI_API_KEY:
            print("Astra LLM: XAI_API_KEY missing, skipping Grok call")
            return None

        url = "https://api.x.ai/v1/chat/completions"

        payload = {
            "model": MODEL,  # e.g. "grok-4-fast-reasoning"
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.25,
            "max_tokens": 220,
        }

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            print("Astra LLM error:", resp.status_code, resp.text[:300])
            return None

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None

        content = choices[0]["message"]["content"]
        return content.strip()
    except Exception as e:
        print("Astra LLM exception:", e)
        return None


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



@app.get("/candles/{symbol}")
def candles_endpoint(symbol: str, limit: int = 252):
    """
    Returns OHLCV candles for charting.
    Data is sourced from Firestore-backed candle cache.
    """

    symbol = symbol.upper()

    try:
        # Fetch from Firestore-backed candle store
        candles = get_cached_candles(
            symbol,
            min_points=min(limit, 120),
        )

        if not candles:
            return {
                "symbol": symbol,
                "error": f"No candle data available for {symbol}",
            }

        closes = candles["close"]
        highs = candles["high"]
        lows = candles["low"]
        opens = candles["open"]
        volumes = candles["volume"]
        timestamps = candles.get("timestamp") or []

        n = len(closes)
        if n == 0:
            return {
                "symbol": symbol,
                "error": "Empty candle set",
            }

        # Trim to requested window
        use_n = min(limit, n)
        start_idx = n - use_n

        items = []
        for i in range(start_idx, n):
            ts = timestamps[i] if i < len(timestamps) else None

            if ts:
                dt = datetime.datetime.utcfromtimestamp(ts / 1000.0)
            else:
                # Fallback: synthetic spacing (should be rare)
                dt = datetime.datetime.utcnow() - datetime.timedelta(days=(n - 1 - i))

            items.append(
                {
                    "t": dt.replace(microsecond=0).isoformat() + "Z",
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "volume": float(volumes[i]),
                }
            )

        return {
            "symbol": symbol,
            "source": "firestore",
            "count": len(items),
            "candles": items,
        }

    except Exception as e:
        print("[candles_endpoint] error:", e)
        return {
            "symbol": symbol,
            "error": str(e),
        }


@app.get("/technical/{symbol}")
def get_technical(symbol: str):
    symbol = symbol.upper()
    try:
        candles = get_candles(symbol, min_points=120)
        if not candles:
            return {"symbol": symbol, "error": f"Could not fetch candles for {symbol}"}
        _, feat, last_close = compute_bullbrain_features(candles)
        return build_technical_snapshot(symbol, feat, last_close)
    except Exception as e:
        print("get_technical error:", e)
        return {"symbol": symbol, "error": str(e)}

# --------------------------------------------------------------------
# SMART PATTERN HISTORY ENDPOINT
# --------------------------------------------------------------------
@app.get("/patternhistory/{symbol}")
def pattern_history(symbol: str, lookahead_5: int = 5, lookahead_10: int = 10):
    symbol = symbol.upper()
    try:
        candles = get_candles(symbol, min_points=180)
        if not candles:
            return {
                "symbol": symbol,
                "error": "No candle data available for this symbol.",
            }

        summary = scan_smart_pattern_history(
            symbol,
            candles,
            lookahead_5=lookahead_5,
            lookahead_10=lookahead_10,
        )
        summary["symbol"] = symbol
        return summary
    except Exception as e:
        print("pattern_history error:", e)
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

@app.get("/prices")
def get_prices(symbols: str):
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    result = {}

    for sym in symbols_list:
        price = None
        prevClose = None

        # ---- Finnhub first attempt ----
        try:
            if FINNHUB_KEY:
                q_url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={FINNHUB_KEY}"
                q = requests.get(q_url, timeout=5).json()
                price = q.get("c")
                prevClose = q.get("pc")
        except:
            pass

        # ---- FMP fallback (RELIABLE) ----
        if price is None:
            try:
                if FMP_API_KEY:
                    fmp_url = f"https://financialmodelingprep.com/api/v3/quote/{sym}?apikey={FMP_API_KEY}"
                    fmp = requests.get(fmp_url, timeout=5).json()
                    if isinstance(fmp, list) and len(fmp) > 0:
                        price = fmp[0].get("price") or price
                        prevClose = fmp[0].get("previousClose") or prevClose
            except:
                pass

        result[sym] = {
            "price": price,
            "prevClose": prevClose,
        }

    return result


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


# ---------------------------------------------------------------
# AI INSIGHT (DYNAMIC) — BullBrain v2 + Rebalancing + 5-Day Trend
# ---------------------------------------------------------------

from functools import lru_cache
import time

# 15-min cache (900 sec)
AI_CACHE = {}  # key = (symbol, allocation, gainPct, posValue, totalValue)


def set_cache(key, data):
    AI_CACHE[key] = {
        "data": data,
        "ts": time.time()
    }


def get_cache(key):
    item = AI_CACHE.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > 900:  # 15 min expiry
        return None
    return item["data"]


@app.get("/portfolio-ai-insight/{symbol}")
def portfolio_ai_insight(
    symbol: str,
    allocation_pct: float = 0.0,
    gain_pct: float = 0.0,
    position_value: float = 0.0,
    portfolio_total_value: float = 0.0
):
    """
    Dynamic BullBrain v2 insight + 5-day trend probability + rebalancing suggestions.
    Lightweight, cached, and fast.
    """

    symbol = symbol.upper()

    # ------- CACHE CHECK -------
    cache_key = (symbol, round(allocation_pct, 2), round(gain_pct, 2),
                 round(position_value, 2), round(portfolio_total_value, 2))
    cached = get_cache(cache_key)
    if cached:
        return cached

    try:
        # 1) Fetch candles
        candles = get_candles(symbol, min_points=120)
        if not candles:
            return {"error": "Insufficient candle data"}

        # 2) Compute 48 features
        features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
        if features_vec is None:
            return {"error": "Feature computation failed"}

        # 3) Model inference
        out = bullbrain_infer(features_vec)
        prob_up = float(out.get("probability_up") or 0.5)
        signal = out.get("signal") or "NEUTRAL"

        # -------------------------------
        # TREND
        # -------------------------------
        if signal == "BUY":
            trend = "Bullish"
        elif signal == "SELL":
            trend = "Bearish"
        else:
            trend = "Neutral"

        # ------------------------------------
        # EXPECTED MOVE (VOL * probability)
        # ------------------------------------
        vol = feature_dict.get("volatility_5d", 0.02)
        expected_move = round(vol * (prob_up * 2 - 1), 4)
        expected_move_pct = f"{expected_move * 100:+.2f}%"

        # CONFIDENCE
        confidence_pct = f"{prob_up * 100:.0f}%"

        # RISK
        if vol < 0.015:
            risk = "Low"
        elif vol < 0.035:
            risk = "Medium"
        else:
            risk = "High"

        # PATTERN
        sma5 = feature_dict.get("sma5", 0)
        sma20 = feature_dict.get("sma20", 0)
        if sma5 > sma20:
            pattern = "Short-term Momentum"
        elif sma5 < sma20:
            pattern = "Reversal Risk"
        else:
            pattern = "Sideways Consolidation"

        # ------------------------------------
        # NEW: 5-DAY TREND PROBABILITY
        # ------------------------------------
        five_day_prob = f"{int(prob_up * 100)}% Bullish"

        # ------------------------------------
        # NEW: REBALANCING SUGGESTION
        # ------------------------------------
        suggestion = "No rebalancing needed."

        if portfolio_total_value > 0 and last_close > 0:
            ideal_pct = prob_up  # If model is 78% bullish, ideal weighting ~78%/100

            diff = (allocation_pct / 100) - prob_up
            diff_pct = round(abs(diff) * 100, 2)

            # Dollar difference
            dollar_diff = abs(diff) * portfolio_total_value

            # Shares difference
            shares_diff = round(dollar_diff / last_close)

            if diff > 0.02:  # overweight
                suggestion = (
                    f"Trim ~{shares_diff} shares (≈{diff_pct}% ≈ ${dollar_diff:,.0f}). "
                    f"This reduces {symbol} to an optimal allocation."
                )
            elif diff < -0.02:  # underweight
                suggestion = (
                    f"Add ~{shares_diff} shares (≈{diff_pct}% ≈ ${dollar_diff:,.0f}). "
                    f"{symbol} shows improving momentum — consider increasing exposure."
                )

        # ------------------------------------
        # Construct Human Message
        # ------------------------------------
        message = (
            f"AI View Today:\n"
            f"{symbol} trend: {trend}\n"
            f"Expected move: {expected_move_pct}\n"
            f"Risk: {risk}\n"
            f"Confidence: {confidence_pct}\n"
            f"Pattern: {pattern}\n"
            f"5-Day Bullish Probability: {five_day_prob}\n"
            f"(BullBrain v2)"
        )

        result = {
            "symbol": symbol,
            "trend": trend,
            "expected_move": expected_move_pct,
            "risk": risk,
            "confidence": confidence_pct,
            "pattern": pattern,
            "five_day_prob": five_day_prob,
            "rebalancing": suggestion,
            "last_price": last_close,
            "message": message,
        }

        # SAVE TO CACHE
        set_cache(cache_key, result)

        return result

    except Exception as e:
        print("AI insight error:", e)
        return {"error": "AI insight unavailable"}



import re  # make sure this is at top of main.py

class PushRegisterRequest(BaseModel):
    user_id: str
    token: str
    platform: str = "unknown"
# ---------------------------------------------------------------
# Astra Chat Request Models
# ---------------------------------------------------------------
class AstraPosition(BaseModel):
    symbol: str
    shares: float
    avg_cost: float
    price: float
    gain: float
    gain_pct: float
    allocation_pct: float
    today: float


class AstraChatRequest(BaseModel):
    # Either a free-form question or a predefined question_id from the app
    question: Optional[str] = ""
    question_id: Optional[str] = None

    contextType: Optional[str] = "portfolio"
    symbol: Optional[str] = None
    companyName: Optional[str] = None

    total_value: float = 0.0
    total_gain: float = 0.0
    today_gain: float = 0.0

    positions: List[AstraPosition] = []
    # ✅ NEW — short session memory from frontend
    chat_history: List[Dict[str, str]] = []


# ---------------------------------------------------------
# Push Notification Registration
# ---------------------------------------------------------

@app.post("/push/register")
def register_push_token(req: PushRegisterRequest):
    try:
        db = firestore.client()

        doc_ref = (
            db.collection("users")
            .document(req.user_id)
        )

        doc_ref.set({
            "expoPushToken": req.token,
            "pushPlatform": req.platform,
            "pushUpdatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        }, merge=True)

        return {
            "success": True,
            "message": "Push token registered successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ---------------------------------------------------------
# Push Notification Test
# ---------------------------------------------------------

@app.get("/push/test")
def send_test_push(user_id: str):
    try:
        db = firestore.client()

        doc = (
            db.collection("users")
            .document(user_id)
            .get()
        )

        if not doc.exists:
            return {
                "success": False,
                "message": "No user document found"
            }

        data = doc.to_dict() or {}

        token = data.get("expoPushToken") or data.get("expo_push_token")

        if not token:
            return {
                "success": False,
                "message": "Push token missing"
            }

        payload = {
            "to": token,
            "sound": "default",
            "title": "AlphaWise Alert",
            "body": "Welcome to AlphaWise push notifications 🚀",
            "data": {
                "type": "test_notification"
            }
        }

        r = requests.post(
            "https://exp.host/--/api/v2/push/send",
            json=payload,
            headers={
                "Content-Type": "application/json"
            },
            timeout=15
        )

        return {
            "success": True,
            "expo_response": r.json()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ---------------------------------------------------------------
# Helper: lightweight market sentiment for a symbol
# ---------------------------------------------------------------
def astra_symbol_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Lightweight, resilient sentiment block for a single symbol.
    Uses:
      - Daily candles (Polygon) for price/vol move
      - Last close vs prev close
    Keeps it simple, two lines max for Astra to use.
    """
    symbol = symbol.upper()
    sentiment = {
        "symbol": symbol,
        "price_comment": "",
        "volume_comment": "",
        "summary": "",
    }

    try:
        candles = get_candles(symbol, min_points=60)
        if not candles or len(candles) < 2:
            sentiment["summary"] = f"{symbol} market sentiment could not be derived from recent price data."
            return sentiment

        # candles is list of OHLCV dicts sorted by date (you already use this)
        last = candles[-1]
        prev = candles[-2]

        last_close = float(last.get("c", last.get("close", 0)))
        prev_close = float(prev.get("c", prev.get("close", 0)))
        last_vol = float(last.get("v", last.get("volume", 0)))

        # 10-day average volume
        recent = candles[-10:] if len(candles) >= 10 else candles
        avg_vol = sum(float(c.get("v", c.get("volume", 0))) for c in recent) / max(
            len(recent), 1
        )

        price_change_pct = (
            ((last_close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        )

        if price_change_pct > 3:
            price_comment = f"Price is up about {price_change_pct:.1f}% today."
        elif price_change_pct < -3:
            price_comment = f"Price is down about {price_change_pct:.1f}% today."
        elif abs(price_change_pct) < 0.5:
            price_comment = "Price is almost flat today."
        else:
            direction = "up" if price_change_pct > 0 else "down"
            price_comment = f"Price is {direction} about {abs(price_change_pct):.1f}% today."

        if avg_vol > 0:
            vol_ratio = last_vol / avg_vol
        else:
            vol_ratio = 1.0

        if vol_ratio > 1.3:
            volume_comment = "Volume is higher than its recent average, so interest is elevated."
        elif vol_ratio < 0.7:
            volume_comment = "Volume is lower than usual, so moves may not be strongly confirmed."
        else:
            volume_comment = "Volume is close to its recent average."

        summary = f"{price_comment} {volume_comment}"

        sentiment["price_comment"] = price_comment
        sentiment["volume_comment"] = volume_comment
        sentiment["summary"] = summary.strip()
        return sentiment

    except Exception as e:
        print(f"Astra sentiment error for {symbol}:", e)
        sentiment["summary"] = f"{symbol} sentiment is unclear based on current data."
        return sentiment



# ---------------------------------------------------------------
# ASTRA CHAT — App-wide Intelligence Engine
# ---------------------------------------------------------------
@app.post("/astra-chat")
def astra_chat(req: AstraChatRequest):
    if req.contextType not in ("stock_detail", "market"):
        if not req.positions or req.total_value <= 0:
            return {
                "answer": (
                    "I need at least one holding with a non-zero portfolio value "
                    "to analyze. Please add positions to your portfolio and try again."
                ),
                "used_llm": False,
                "analysis": {},
            }

    if req.contextType == "stock_detail" and not req.symbol:
        return {
            "answer": "I need a stock symbol to explain this stock.",
            "used_llm": False,
            "analysis": {},
        }
    try:
        from backend.astra_engine import run_astra

        return run_astra(
            req=req,
            astra_llm_answer_fn=astra_llm_answer,
        )

    except Exception as e:
        print("Astra engine error:", e)
        return {
            "answer": (
                "Astra could not complete the full intelligence analysis right now. "
                "Please try again shortly."
            ),
            "used_llm": False,
            "error": str(e),
            "analysis": {},
        }

@app.get("/market-pulse")
def market_pulse():
    """
    Firestore read-only endpoint.
    Cron job is the single writer.
    """
    try:
        db = firestore.client()
        doc = db.collection("bullsignals_ai").document("market_pulse").get()

        if not doc.exists:
            return {
                "highlights_grouped": {
                    "bullish": [],
                    "neutral": [],
                    "bearish": [],
                },
                "news_grouped": {
                    "today": [],
                    "yesterday": [],
                    "week": [],
                    "older": [],
                },
                "updated_at": None,
            }

        return doc.to_dict()

    except Exception as e:
        backend.log(f"[market-pulse] Firestore read error: {e}")
        return {
            "highlights_grouped": {
                "bullish": [],
                "neutral": [],
                "bearish": [],
            },
            "news_grouped": {
                "today": [],
                "yesterday": [],
                "week": [],
                "older": [],
            },
            "updated_at": None,
        }


@app.get("/market-overview")
def market_overview():
    try:
        db = firestore.client()
        doc = db.collection("bullsignals_ai").document("market_overview_live").get()

        if not doc.exists:
            return {}

        return doc.to_dict()

    except Exception as e:
        backend.log(f"[market-overview] Firestore read error: {e}")
        return {}


@app.get("/debug-bullbrain/{symbol}")
def debug_bullbrain(symbol: str):
    try:
        sym = symbol.upper()
        candles = get_candles(sym, min_points=120)

        features_vec, feat_dict, last_close = compute_bullbrain_features(candles)
        infer = bullbrain_infer(features_vec)

        return {
            "symbol": sym,
            "features_shape": len(features_vec),
            "infer": infer,
        }

    except Exception as e:
        return {"error": str(e)}



# ---------------------------------------------------------
# Firebase Admin init (shared by API + Cron)
# ---------------------------------------------------------
def init_firebase_admin():
    """
    Initialize Firebase Admin exactly once using FIREBASE_ADMIN_JSON.
    This is safe to call from both main API and cron scripts.
    """
    if firebase_admin._apps:
        # Already initialized
        return firebase_admin._apps[0]

    firebase_json = os.getenv("FIREBASE_ADMIN_JSON")
    if not firebase_json:
        print("❌ FIREBASE_ADMIN_JSON is missing!")
        return None

    try:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        app = firebase_admin.initialize_app(cred)
        print("🔥 Firebase Admin initialized")
        return app
    except Exception as e:
        print("❌ Firebase Admin init failed:", e)
        return None


# Initialize immediately for API process
init_firebase_admin()
db = firestore.client()

# =========================================================
# Firestore Candle Cache (shared by cron + APIs)
# =========================================================
def get_candle_doc(db, symbol: str):
    return (
        db.collection("bullsignals_ai_candles")
          .document(symbol.upper())
    )


def load_cached_candles(symbol: str):
    try:
        from firebase_admin import firestore
        db = firestore.client()
        doc = get_candle_doc(db, symbol).get()
        if not doc.exists:
            return None
        return doc.to_dict()
    except Exception:
        return None


def save_cached_candles(symbol: str, candles: dict):
    from firebase_admin import firestore
    db = firestore.client()

    payload = {
        "symbol": symbol,
        "source": candles.get("source", "polygon"),
        "candles": candles,
        "last_t_ms": max(candles.get("timestamp", []), default=0),
        "cached_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    get_candle_doc(db, symbol).set(payload, merge=True)


# ---------------------------------------------------------
# Generic helpers: save/read market AI cache (Firestore)
# ---------------------------------------------------------
def save_to_firestore_market_cache(doc_id: str, data: dict):
    """
    Save a document into bullsignals_ai/<doc_id>.
    Used by cron script OR any backend batch job.
    """
    try:
        if not firebase_admin._apps:
            init_firebase_admin()

        doc_ref = db.collection("bullsignals_ai").document(doc_id)
        doc_ref.set(data, merge=True)

        print(f"🔥 Saved AI Market Cache: {doc_id}")
    except Exception as e:
        print("save_to_firestore_market_cache error:", e)


def read_market_cache(doc_id: str):
    """
    Read a document from bullsignals_ai/<doc_id>.
    API endpoints use this to return cached Hotlist/BearWatch.
    No recompute, no TTL logic here — cron keeps it fresh.
    """
    try:
        if not firebase_admin._apps:
            init_firebase_admin()

        doc_ref = db.collection("bullsignals_ai").document(doc_id)
        snap = doc_ref.get()

        if not snap.exists:
            print(f"⚠️ No Firestore cache for {doc_id}")
            return None

        data = snap.to_dict()
        return data
    except Exception as e:
        print("read_market_cache error:", e)
        return None




# ---------------------------------------------------------
# /homescreen-context — UI-only data (NO intelligence NEW)
# ---------------------------------------------------------
@app.get("/homescreen-context")
def homescreen_context():
    doc = (
        db.collection("bullsignals_ai")
          .document("homescreen_snapshot")
          .get()
    )

    if not doc.exists:
        return {
            "market": None,
            "carousel": [],
            "updated_at": None,
            "version": "v2",
        }

    data = doc.to_dict() or {}

    return {
        "market": data.get("market_overview"),
        "carousel": data.get("carousel", []),
        "updated_at": data.get("updated_at"),
        "version": data.get("version", "v2"),
    }

@app.get("/homescreen-data")
def homescreen_data():
    cache = read_market_cache("homescreen_snapshot")

    if not cache:
        return {
            "status": "empty",
            "market_overview": {},
            "core_universe": [],
            "core_signals": [],
            "core_universe_count": 0,
            "core_signals_updated_at": None,
            "updated_at": None,
            "version": None,
            "schema_version": None,
        }

    return {
        "status": "ok",
        "market_overview": cache.get("market_overview", {}),
        "core_universe": cache.get("core_universe", []),
        "core_signals": cache.get("core_signals", []),
        "core_universe_count": cache.get("core_universe_count", 0),
        "core_signals_updated_at": cache.get("core_signals_updated_at"),
        "updated_at": cache.get("updated_at"),
        "version": cache.get("version"),
        "schema_version": cache.get("schema_version"),
    }
# ---------------------------------------------------------
# Stock Detail API — Canonical v1.0
# Narrative + Explanation driven
# ---------------------------------------------------------
@app.get("/stockdetail/{symbol}")
def stock_detail(symbol: str, source: str | None = None):
    sym = symbol.upper()
    print("STOCKDETAIL HIT:", sym, "source =", source)

    # ---------------------------------------------------------
    # 0️⃣ Track active symbol (UI only, never block)
    # ---------------------------------------------------------
    if source == "ui":
        try:
            from backend.active_symbols import touch_active_symbol
            touch_active_symbol(sym)
        except Exception:
            pass

    # ---------------------------------------------------------
    # 1️⃣ Firestore read (SINGLE source of truth)
    # ---------------------------------------------------------
    doc = (
        db.collection("bullsignals_ai")
          .document("stocks")
          .collection("symbols")
          .document(sym)
          .get()
    )

    if not doc.exists:
        return {
            "status": "not_ready",
            "symbol": sym,
        }

    stock = doc.to_dict() or {}
    stock["symbol"] = sym  # defensive

    # ---------------------------------------------------------
    # 2️⃣ Header (shared across screens)
    # ---------------------------------------------------------
    from backend.header_builder import build_stock_header
    header = build_stock_header(stock)

    # ---------------------------------------------------------
    # 3️⃣ Sparkline (optional, UI friendly)
    # Firestore has stock["sparkline"] as price array
    # ---------------------------------------------------------
    sparkline = None

    try:
        from backend.ui_stock_builder import (
            build_sparkline,
            build_sparkline_from_prices,
        )

        existing_prices = stock.get("sparkline")
        chart_meta = stock.get("chart") or {}

        # Priority 1: Firestore sparkline array
        if isinstance(existing_prices, list) and len(existing_prices) >= 2:
            sparkline = build_sparkline_from_prices(existing_prices, meta=chart_meta)

        # Priority 2: candles if available later
        if not sparkline:
            candles_block = stock.get("candles")

            if isinstance(candles_block, dict):
                candles = candles_block.get("candles", [])
            elif isinstance(candles_block, list):
                candles = candles_block
            else:
                candles = []

            if isinstance(candles, list) and len(candles) >= 2:
                sparkline = build_sparkline(candles, meta=chart_meta)

    except Exception as e:
        print("⚠️ stockdetail sparkline build failed:", str(e))
        sparkline = None
    # ---------------------------------------------------------
    # 4️⃣ Company News (external I/O — endpoint responsibility)
    # ---------------------------------------------------------
    try:
        from backend.news_repo import fetch_symbol_news
        stock["news"] = fetch_symbol_news(
            symbol=sym,
            company_name=stock.get("company_name"),
            limit=6,
        ) or []
    except Exception:
        stock["news"] = []

    # ---------------------------------------------------------
    # 5️⃣ FULL Stock Detail Report (v1.0 — deterministic)
    # ---------------------------------------------------------
    from backend.ui_stock_builder import build_stockdetail_v1
    content = build_stockdetail_v1(stock) or {}

    # Ensure sparkline is available at top-level for UI
    if sparkline:
        content["sparkline"] = sparkline

    # ---------------------------------------------------------
    # 6️⃣ Final Response (Clean & Stable)
    # ---------------------------------------------------------
    return {
        "header": header,
        "content": content,
    }

# ---------------------------------------------------------
# Stock Pattern Detail API — FULL PATTERN VIEW
# ---------------------------------------------------------
@app.get("/stockdetail/{symbol}/pattern")
def stock_pattern_detail(symbol: str):
    sym = symbol.upper()

    doc = (
        db.collection("bullsignals_ai")
          .document("stocks")
          .collection("symbols")
          .document(sym)
          .get()
    )

    if not doc.exists:
        return {"status": "not_ready", "symbol": sym}

    stock = doc.to_dict() or {}
    stock["symbol"] = sym

    # 1️⃣ Shared header
    from backend.header_builder import build_stock_header
    header = build_stock_header(stock)

    # 2️⃣ Pattern content (MAX DETAIL)
    content = {
        "pattern": stock.get("pattern"),
        "history": stock.get("patternHistory"),
        "forwardReturns": (stock.get("patternHistory") or {}).get("forwardReturns"),
        "occurrences": (stock.get("patternHistory") or {}).get("occurrences"),
        "samples": (stock.get("patternHistory") or {}).get("samples"),
        "explanation": {
            "whatItMeans": (stock.get("pattern") or {}).get("headline"),
            "bias": (stock.get("pattern") or {}).get("bias"),
            "note": "Pattern statistics are based on historical occurrences, not predictions.",
        },
    }

    return {
        "header": header,
        "content": content,
    }

# ---------------------------------------------------------
# Stock Decision Detail API — FULL MODEL EXPLANATION
# ---------------------------------------------------------
@app.get("/stockdetail/{symbol}/decision")
def stock_decision_detail(symbol: str):
    sym = symbol.upper()

    # ---------------------------------------------------------
    # 1️⃣ Firestore read — single source of truth
    # ---------------------------------------------------------
    doc = (
        db.collection("bullsignals_ai")
          .document("stocks")
          .collection("symbols")
          .document(sym)
          .get()
    )

    if not doc.exists:
        return {
            "status": "not_ready",
            "symbol": sym,
        }

    stock = doc.to_dict() or {}
    stock["symbol"] = sym

    # ---------------------------------------------------------
    # 2️⃣ Shared header
    # ---------------------------------------------------------
    from backend.header_builder import build_stock_header
    header = build_stock_header(stock)

    # ---------------------------------------------------------
    # 3️⃣ Decision ladder + gate metrics
    # ---------------------------------------------------------
    from backend.decision_explainer import explain_decision_ladder
    decision_payload = explain_decision_ladder(stock) or {}

    # ---------------------------------------------------------
    # 4️⃣ Final response
    # ---------------------------------------------------------
    return {
        "header": header,

        # ✅ Main UI payload
        "modelDecision": decision_payload,

        # ✅ Optional supporting context for frontend if needed later
        "context": {
            "decision": stock.get("decision") or {},
            "probabilities": stock.get("probabilities") or {},
            "indicatorStates": stock.get("indicator_states") or {},
            "pattern": stock.get("pattern") or {},
            "featuresMeta": stock.get("features_meta") or {},
        },

        "meta": {
            "computed_at": stock.get("computed_at"),
            "schema": "decision_v3",
        },
    }
# ---------------------------------------------------------
# Stock Technical Detail API — FULL TECH + FEATURES
# ---------------------------------------------------------
@app.get("/stockdetail/{symbol}/technical")
def stock_technical_detail(symbol: str):
    sym = symbol.upper()

    # 1️⃣ Firestore read
    doc = (
        db.collection("bullsignals_ai")
          .document("stocks")
          .collection("symbols")
          .document(sym)
          .get()
    )

    if not doc.exists:
        return {"status": "not_ready", "symbol": sym}

    stock = doc.to_dict() or {}
    stock["symbol"] = sym

    # 2️⃣ Shared header
    from backend.header_builder import build_stock_header
    header = build_stock_header(stock)

    # 3️⃣ Technical explainer
    from backend.technical_explainer import explain_technical
    technical_payload = explain_technical(stock)

    # 4️⃣ Final response
    return {
        "header": header,
        **technical_payload,

        # ✅ Add full feature metadata for deep UI sections
        "featuresMeta": stock.get("features_meta") or {},

        # ✅ Optional useful context
        "indicatorStates": stock.get("indicator_states") or {},
        "narratives": stock.get("narratives") or {},

        "meta": {
            "computed_at": stock.get("computed_at"),
            "schema": "technical_v1",
        },
    }


# -----------------------------
# Firestore helpers
# -----------------------------
def _db():
    return firestore.client()

def _norm_symbol(symbol: str) -> str:
    return (symbol or "").upper().strip().replace(".", "-")

def _watchlist_col(user_id: str):
    return (
        _db()
        .collection("users")
        .document(user_id)
        .collection("watchlist")
        
    )

# -----------------------------
# 1) READ watchlist (TTL SNAPSHOT)
# -----------------------------
@app.get("/watchlist/{user_id}")
def get_watchlist(user_id: str):
    snapshot = get_watchlist_snapshot(user_id)

    # ✅ REQUIRED: invalidate old logic snapshots
    SNAPSHOT_VERSION = "v6"

    if (
        snapshot
        and is_snapshot_fresh(snapshot)
        and snapshot.get("version") == SNAPSHOT_VERSION
    ):
        items = snapshot.get("items", [])
        return {
            "status": "ok",
            "count": len(items),
            "watchlist": items,
            "source": "snapshot",
        }

    # 🔥 Rebuild if stale OR version mismatch
    snapshot = build_watchlist_snapshot(user_id)
    items = snapshot.get("items", [])

    return {
        "status": "ok",
        "count": len(items),
        "watchlist": items,
        "source": "rebuilt",
    }

# -----------------------------
# 2) ADD symbol to watchlist
# -----------------------------
from backend.active_symbols import touch_active_symbol

# -----------------------------
# 2) ADD symbol to watchlist
# -----------------------------
from backend.active_symbols import touch_active_symbol
from backend.watchlist_snapshot import build_watchlist_snapshot

@app.post("/watchlist/{user_id}/add/{symbol}")
def add_watchlist_symbol(user_id: str, symbol: str):
    sym = _norm_symbol(symbol)
    if not sym.isalnum():
        return {"status": "error", "error": "Invalid symbol"}

    # 1️⃣ Save user intent
    _watchlist_col(user_id).document(sym).set(
        {"symbol": sym, "added_at": firestore.SERVER_TIMESTAMP},
        merge=True
    )

    # 2️⃣ Mark symbol as active (global relevance)
    try:
        touch_active_symbol(sym)
    except Exception:
        pass

    # 3️⃣ Warm caches (best effort, non-blocking)
    try:
        ensure_quote(sym)
        bootstrap_stock(sym)
    except Exception:
        pass

    # 4️⃣ 🔥 FORCE snapshot rebuild (ignore TTL)
    try:
        build_watchlist_snapshot(user_id)
    except Exception as e:
        print(f"[watchlist] snapshot rebuild failed: {e}")

    return {
        "status": "ok",
        "user_id": user_id,
        "symbol": sym,
    }


# -----------------------------
# 3) REMOVE symbol from watchlist
# -----------------------------
from backend.watchlist_snapshot import build_watchlist_snapshot

@app.delete("/watchlist/{user_id}/remove/{symbol}")
def remove_watchlist_symbol(user_id: str, symbol: str):
    sym = _norm_symbol(symbol)

    # 1️⃣ Remove from watchlist
    _watchlist_col(user_id).document(sym).delete()

    # 2️⃣ 🔥 FORCE snapshot rebuild
    try:
        build_watchlist_snapshot(user_id)
    except Exception as e:
        print(f"[watchlist] snapshot rebuild failed: {e}")

    return {
        "status": "ok",
        "user_id": user_id,
        "symbol": sym,
    }

# -----------------------------
# 4) Market Movers
# -----------------------------

@app.get("/market-movers")
def get_market_movers():
    movers_doc = (
        db.collection("bullsignals_ai")
          .document("market_movers")
          .get()
    )

    if not movers_doc.exists:
        return {
            "count": 0,
            "movers": [],
            "updated_at": None,
        }

    meta = movers_doc.to_dict() or {}
    movers = meta.get("movers", [])
    out = []

    for m in movers:
        sym = m.get("symbol")
        if not sym:
            continue

        stock_doc = (
            db.collection("bullsignals_ai")
              .document("stocks")
              .collection("symbols")
              .document(sym)
              .get()
        )

        if not stock_doc.exists:
            continue

        s = stock_doc.to_dict() or {}

        # ✅ live quote repo first
        quote_doc = (
            db.collection("bullsignals_ai")
              .document("quotes")
              .collection("symbols")
              .document(sym)
              .get()
        )

        live_q = quote_doc.to_dict() if quote_doc.exists else {}
        stock_q = s.get("quote", {}) or {}

        q = live_q if live_q else stock_q

        change_pct = q.get("changePct")
        direction = None

        if isinstance(change_pct, (int, float)):
            direction = "up" if change_pct >= 0 else "down"
        else:
            direction = m.get("direction")

        tech = s.get("technical", {}) or {}
        trend = tech.get("trend", {}) or {}
        pattern = s.get("pattern", {}) or {}
        insights = s.get("insights", {}) or {}
        market_awareness = s.get("marketAwareness", {}) or {}

        out.append({
            "symbol": sym,
            "company": s.get("company_name") or sym,

            "quote": {
                "price": q.get("price"),
                "change": q.get("change"),
                "changePct": q.get("changePct"),
                "updated_at": q.get("updated_at"),
                "source": q.get("source"),
                "needs_refresh": q.get("needs_refresh", False),
            },

            # ✅ now live direction based on latest quote
            "direction": direction,

            "trend": {
                "label": trend.get("label") or trend.get("summary"),
            },

            "pattern": {
                "name": pattern.get("pattern") or pattern.get("patternLabel"),
                "bias": pattern.get("bias") or pattern.get("patternBias"),
            },

            "oneLiner": (
                insights.get("oneLiner")
                or market_awareness.get("summary")
                or market_awareness.get("oneLiner")
            ),
        })

    # ✅ Re-sort using latest quote values
    gainers = [
        x for x in out
        if isinstance((x.get("quote") or {}).get("changePct"), (int, float))
        and (x.get("quote") or {}).get("changePct") >= 0
    ]

    losers = [
        x for x in out
        if isinstance((x.get("quote") or {}).get("changePct"), (int, float))
        and (x.get("quote") or {}).get("changePct") < 0
    ]

    gainers.sort(
        key=lambda x: (x.get("quote") or {}).get("changePct", 0),
        reverse=True,
    )

    losers.sort(
        key=lambda x: (x.get("quote") or {}).get("changePct", 0),
    )

    final_out = gainers + losers

    return {
        "count": len(final_out),
        "movers": final_out,
        "as_of": meta.get("as_of"),
        "updated_at": meta.get("updated_at"),
        "quote_refreshed": True,
        "version": "v3-live-quotes",
    }
# -----------------------------
# 5) Market News
# -----------------------------

@app.get("/market-news")
def market_news_legacy():
    """
    Market tab news.
    Cached, fast, App-Store safe.
    Strictly stock-market related.
    """
    data = get_market_news()
    items = data.get("items", [])

    return {
        "source": data.get("source"),
        "updated_at": data.get("updated_at"),
        "count": len(items),
        "news": items,
    }

# ---------------------------------------------------------
# Quotes Bulk API — Symbol-Driven (Reusable)
# ---------------------------------------------------------
@app.get("/quotes-bulk")
def quotes_bulk(
    symbols: str | None = None,
    scope: str | None = None,
):
    print("QUOTES-BULK HIT | scope =", scope, "| symbols =", symbols)

    if not symbols:
        return {
            "scope": scope,
            "count": 0,
            "quotes": {},
            "error": "symbols parameter required",
        }

    # ---------------------------------------------------------
    # 1️⃣ Parse symbols
    # ---------------------------------------------------------
    symbol_list = [
        s.strip().upper()
        for s in symbols.split(",")
        if s.strip()
    ][:100]

    quotes = {}

    # ---------------------------------------------------------
    # 2️⃣ Firestore reads (quotes ONLY)
    # ---------------------------------------------------------
    for sym in symbol_list:
        doc = (
            db.collection("bullsignals_ai")
              .document("quotes")
              .collection("symbols")
              .document(sym)
              .get()
        )

        if not doc.exists:
            continue

        data = doc.to_dict() or {}

        quotes[sym] = {
            "symbol": data.get("symbol", sym),
            "price": data.get("price"),
            "change": data.get("change"),
            "changePct": data.get("changePct"),
            "open": data.get("open"),
            "high": data.get("high"),
            "low": data.get("low"),
            "prevClose": data.get("prevClose"),
            "timestamp": data.get("timestamp"),
            "updated_at": data.get("updated_at"),
            "needs_refresh": data.get("needs_refresh", False),
            "ttl_seconds": data.get("ttl_seconds", 30),
            "source": data.get("source"),
        }

    return {
        "scope": scope,
        "count": len(quotes),
        "quotes": quotes,
    }


