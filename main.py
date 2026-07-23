# main.py

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from urllib.parse import urlparse
import os
import requests
import datetime
from zoneinfo import ZoneInfo
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
from backend.market_momentum import get_market_momentum_screen, save_market_momentum_screen

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
                "sector": profile.get("finnhubIndustry") or None,
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
            "sector": profile.get("finnhubIndustry") or None,
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
    trend_pct_20d = _trend_pct_20d(candles_arrays.get("close") or [])
    vol_zscore_20_corrected = _volume_zscore_20(candles_arrays.get("volume") or [])
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
        trend_pct_20d=trend_pct_20d,
        vol_zscore_20_corrected=vol_zscore_20_corrected,
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
        "trend_pct_20d": trend_pct_20d,
        "vol_zscore_20_corrected": vol_zscore_20_corrected,
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
    MIN_SAMPLES = 12
    MIN_WINRATE = 0.58
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

def _trend_pct_20d(closes: list) -> float | None:
    """True 20-day % return, price-comparable — see bullbrain_gate_ladder_audit
    memory, finding #2. compute_bullbrain_features()'s trend_strength_20 is a
    raw $/day regression slope, confounded by share price; this is the
    corrected value, used for gate logic only (the model's own DMatrix
    feature is deliberately left untouched)."""
    if not closes or len(closes) < 21:
        return None
    try:
        c_now = float(closes[-1])
        c_then = float(closes[-21])
        if c_then == 0:
            return None
        return (c_now / c_then - 1.0) * 100.0
    except (TypeError, ValueError, IndexError):
        return None


def _volume_zscore_20(volumes: list) -> float | None:
    """Correctly-scaled 20-day volume z-score — std of RAW volume, not std of
    volume's own 20-day moving average (the bug in compute_bullbrain_features(),
    inflates |z| ~6.66x — see bullbrain_gate_ladder_audit memory). Matches the
    formula already used correctly in scan_smart_pattern_history() and
    backend/smart_patterns.py. Used for gate/narrative logic only — the model's
    own DMatrix feature is deliberately left untouched."""
    if not volumes or len(volumes) < 20:
        return None
    try:
        window = np.array(volumes[-20:], dtype=float)
        mean20 = window.mean()
        std20 = window.std(ddof=1)
        if std20 == 0:
            return None
        return (float(volumes[-1]) - mean20) / std20
    except (TypeError, ValueError, IndexError):
        return None


def detect_market_regime(features: dict, trend_pct_20d: float | None = None) -> str:
    """
    Detect market regime using existing features.
    Returns: 'TRENDING', 'RANGING', 'HIGH_VOL'
    """

    vol20 = features.get("volatility_20d")
    vol60 = features.get("volatility_60d")
    atr = features.get("atr14")
    close = features.get("close")

    # trend_pct_20d (true 20-day % return) is preferred when the caller
    # provides it; falls back to the raw-slope model feature + its old
    # threshold only for callers that don't pass the corrected value.
    trend = trend_pct_20d if trend_pct_20d is not None else features.get("trend_strength_20")

    # Defensive
    if trend is None or vol20 is None:
        return "UNKNOWN"

    # High volatility regime
    # atr14 is a raw dollar range; normalize to % of price before comparing
    # against volatility_20d (already a percentage), otherwise this collapses
    # into a pure price-level test instead of a volatility test.
    atr_pct = (atr / close * 100.0) if (atr and close) else None
    if vol20 > 1.5 * (vol60 or vol20) or (atr_pct and atr_pct > 1.8 * vol20):
        return "HIGH_VOL"

    # Strong directional trend
    if trend_pct_20d is not None:
        if abs(trend) > 10.0:
            return "TRENDING"
    elif abs(trend) > 0.4:
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

def volume_gate(features: dict, vol_z_corrected: float | None = None) -> bool:
    """
    Enforce volume confirmation.

    BUY / SELL allowed only if:
    - volume_zscore_20 >= 0.5
    - volume_vs_ma20_pct >= 0
    """

    try:
        # volume_zscore_20 from compute_bullbrain_features() is inflated ~6.66x
        # (std of the volume MA, not raw volume — see bullbrain_gate_ladder_audit
        # memory). Prefer the corrected value; fall back to the buggy one for
        # callers that don't pass it, same as body_pct/trend_pct_20d.
        vol_z = float(vol_z_corrected) if vol_z_corrected is not None else float(features.get("volume_zscore_20"))
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

def signal_fragility(features: dict, vol_z_corrected: float | None = None) -> int:
    """
    Detect fragile / unstable setups.

    Returns an integer fragility score:
    0 = very stable
    1–2 = moderate risk
    >=3 = fragile → should be HOLD
    """

    fragility = 0

    intraday_range = features.get("intraday_range_pct")
    # body_pct from compute_bullbrain_features() is close/open-relative (a known
    # scale-mismatch bug — see bullbrain_gate_ladder_audit memory, finding #1).
    # Recompute the full-range-normalized version locally for this gate only;
    # the model's own DMatrix feature is deliberately left untouched.
    high = features.get("high")
    low = features.get("low")
    open_ = features.get("open")
    close = features.get("close")
    body_pct = None
    if None not in (high, low, open_, close):
        full_range = high - low
        body_pct = ((close - open_) / full_range * 100.0) if full_range > 0 else 0.0
    # volume_zscore_20 from compute_bullbrain_features() is inflated ~6.66x
    # (see bullbrain_gate_ladder_audit memory, finding #6). Prefer the
    # corrected value; fall back to the buggy one for callers that don't pass it.
    vol_z = vol_z_corrected if vol_z_corrected is not None else features.get("volume_zscore_20")
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

def liquidity_quality(features: dict, vol_z_corrected: float | None = None) -> str:
    """
    Classify liquidity quality using volume and volatility behavior.

    Returns:
    - 'GOOD'
    - 'THIN'
    - 'POOR'
    """

    # volume_zscore_20 from compute_bullbrain_features() is inflated ~6.66x
    # (std of the volume MA, not raw volume — see bullbrain_gate_ladder_audit
    # memory). Prefer the corrected value; fall back to the buggy one for
    # callers that don't pass it, same as body_pct/trend_pct_20d.
    vol_z = vol_z_corrected if vol_z_corrected is not None else features.get("volume_zscore_20")
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
    # Each fragility point reduces EV. 0.25 is data-grounded, not arbitrary:
    # it's ~p10 of the real win_rate*avg_ret distribution among symbols that
    # legitimately pass Pattern Quality (see bullbrain_gate_ladder_audit
    # memory, finding #5) — one fragility point should meaningfully threaten
    # only the weakest ~10% of validated edges, not a typical/good one. The
    # old 0.5 constant, combined with finding #1's guaranteed fragility floor,
    # was killing ~42% of legitimately-good setups regardless of edge quality.
    ev -= fragility * 0.25

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
        r1 >= 3.5
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
        r1 <= -3.5
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
    trend_pct_20d: float | None = None,
    vol_zscore_20_corrected: float | None = None,
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

    # quality dict shape: every branch below returns the SAME 11 keys, using
    # None for whatever this branch's execution path didn't reach/compute.
    # This is intentional — save_stock()'s Firestore write uses merge=True,
    # which lets nested map fields survive from a DIFFERENT historical run
    # if the current run's quality dict omits them (see
    # bullbrain_gate_ladder_audit memory, finding A). Always including every
    # key — even as None — makes every write fully overwrite the previous
    # one, so a stale regime/fragility/etc. from an earlier run can never
    # silently survive into this run's persisted decision.

    # ---------------- 1️⃣ Liquidity ----------------
    liq = liquidity_quality(features, vol_z_corrected=vol_zscore_20_corrected)
    if liq != "GOOD":
        reasons.append(f"Liquidity={liq}")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": None, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": None, "consensus": None, "pressure": None,
            "fragility": None, "EV": None, "rarity": None,
        }}

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
                "liquidity": liq, "override": True, "overrideType": "strong_price_volume_momentum",
                "originalModelSignal": model_signal, "pattern": pattern_name,
                "regime": None, "consensus": None, "pressure": None, "fragility": None, "EV": None, "rarity": None,
            },
        }

    # ---------------- 2️⃣ Market Regime ----------------
    regime = detect_market_regime(features, trend_pct_20d=trend_pct_20d)

    # ---------------- 3️⃣ Pattern Quality ----------------
    if not pattern_quality_gate(pattern_history):
        reasons.append("PatternQualityFailed")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": None, "pressure": None,
            "fragility": None, "EV": None, "rarity": None,
        }}

    # ---------------- 4️⃣ Regime Compatibility ----------------
    if pattern_name:
        allowed = PATTERN_REGIME_COMPATIBILITY.get(pattern_name)
        if allowed and regime not in allowed:
            reasons.append(f"PatternNotAllowedIn{regime}")
            return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
                "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
                "pattern": pattern_name, "regime": regime, "consensus": None, "pressure": None,
                "fragility": None, "EV": None, "rarity": None,
            }}

    # ---------------- 5️⃣ Pattern–Model Alignment ----------------
    patt_bias = pattern_bias(pattern_name)
    if not alignment_filter(model_signal, patt_bias):
        reasons.append("SignalPatternConflict")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": None, "pressure": None,
            "fragility": None, "EV": None, "rarity": None,
        }}

    # ---------------- 6️⃣ Multi-Timeframe Agreement ----------------
    if not timeframe_alignment(features, model_signal):
        reasons.append("TimeframeMisalignment")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": None, "pressure": None,
            "fragility": None, "EV": None, "rarity": None,
        }}

    # ---------------- 7️⃣ Volume Confirmation ----------------
    if not volume_gate(features, vol_z_corrected=vol_zscore_20_corrected):
        reasons.append("VolumeGateFailed")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": None, "pressure": None,
            "fragility": None, "EV": None, "rarity": None,
        }}

    # ---------------- 8️⃣ Feature Consensus ----------------
    consensus = feature_consensus_score(features)
    if abs(consensus) < 1:
        reasons.append("WeakFeatureConsensus")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": consensus, "pressure": None,
            "fragility": None, "EV": None, "rarity": None,
        }}

    # ---------------- 9️⃣ Directional Pressure ----------------
    pressure = directional_pressure(features)
    if model_signal == "BUY" and pressure <= 0:
        reasons.append("NoUpsidePressure")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": consensus, "pressure": pressure,
            "fragility": None, "EV": None, "rarity": None,
        }}
    if model_signal == "SELL" and pressure >= 0:
        reasons.append("NoDownsidePressure")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": consensus, "pressure": pressure,
            "fragility": None, "EV": None, "rarity": None,
        }}

    # ---------------- 🔟 Fragility ----------------
    frag = signal_fragility(features, vol_z_corrected=vol_zscore_20_corrected)
    if frag >= 3:
        reasons.append("SignalTooFragile")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": consensus, "pressure": pressure,
            "fragility": frag, "EV": None, "rarity": None,
        }}

    # ---------------- 1️⃣1️⃣ Momentum Exhaustion ----------------
    if momentum_exhaustion(features, model_signal):
        reasons.append("MomentumExhausted")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": consensus, "pressure": pressure,
            "fragility": frag, "EV": None, "rarity": None,
        }}

    # ---------------- 1️⃣2️⃣ Expected Value ----------------
    ev = expected_value_score(pattern_history, frag)
    if ev <= 0:
        reasons.append("NegativeEV")
        return {"finalSignal": "HOLD", "decisionReasons": reasons, "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": consensus, "pressure": pressure,
            "fragility": frag, "EV": ev, "rarity": None,
        }}

    # ---------------- 1️⃣3️⃣ Rarity (context only) ----------------
    rarity = signal_rarity(pattern_history, total_days)

    # ---------------- ✅ PASSED ALL GATES ----------------
    return {
        "finalSignal": model_signal,
        "decisionReasons": ["ALL_GATES_PASSED"],
        "quality": {
            "liquidity": liq, "override": False, "overrideType": None, "originalModelSignal": None,
            "pattern": pattern_name, "regime": regime, "consensus": consensus, "pressure": pressure,
            "fragility": frag, "EV": ev, "rarity": rarity,
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
    "GAP DOWN & PRESSURE": {"TRENDING", "HIGH_VOL"},
    "VOLUME BREAKOUT": {"TRENDING", "HIGH_VOL"},
    "FAILED BREAKOUT TRAP": {"HIGH_VOL"},

    # Mean reversion
    "OVERSOLD BOUNCE": {"RANGING", "HIGH_VOL"},
    "OVERBOUGHT DISTRIBUTION": {"RANGING", "HIGH_VOL"},
    "HAMMER REVERSAL": {"RANGING"},
    "DEAD CAT BOUNCE": {"HIGH_VOL"},

    # Neutral / compression
    "INSIDE RANGE COMPRESSION": {"RANGING"},
    "HIGH-WAVE INDECISION": {"RANGING"},
}

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

            if not sym or not desc:
                continue

            logo_url = None

            try:
                stock_doc = (
                    db.collection("bullsignals_ai")
                    .document("stocks")
                    .collection("symbols")
                    .document(sym.upper())
                    .get()
                )

                if stock_doc.exists:
                    stock = stock_doc.to_dict() or {}
                    profile = stock.get("profile") or {}
                    logo_url = profile.get("logoUrl")
            except Exception:
                pass

            out.append({
                "symbol": sym,
                "description": desc,
                "logoUrl": logo_url,
            })

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

        # 3) Model inference + full decision ladder (same gates as the
        #    canonical market_cron/stock_repo path — was previously a
        #    raw bullbrain_infer() call with no gates at all)
        core = run_bullbrain_from_inputs(
            symbol,
            candles_arrays=candles,
            feat_dict=feature_dict,
        )
        bull = core["bullbrain"]
        decision = core["decision"]

        signal = bull.get("signal") or "HOLD"  # gated final signal
        prob_up = float(bull.get("raw", {}).get("prob_up") or 0.5)

        # HOLD caused by a failed gate (not the model's own genuine
        # neutral read) — suppress the fields below that would
        # otherwise still reflect the raw, ungated probability.
        gate_forced_hold = (
            signal == "HOLD"
            and decision.get("decisionReasons") != ["ALL_GATES_PASSED"]
        )

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
        expected_move = 0.0 if gate_forced_hold else round(vol * (prob_up * 2 - 1), 4)
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
        five_day_prob = (
            "Neutral (no statistically valid setup)"
            if gate_forced_hold
            else f"{int(prob_up * 100)}% Bullish"
        )

        # ------------------------------------
        # NEW: REBALANCING SUGGESTION
        # ------------------------------------
        suggestion = "No rebalancing needed."

        if not gate_forced_hold and portfolio_total_value > 0 and last_close > 0:
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
    selectedMover: Optional[Dict[str, Any]] = None
    movers: List[Dict[str, Any]] = []
    aiSetups: List[Dict[str, Any]] = []
    pullbacks: List[Dict[str, Any]] = []
    pulse: Dict[str, Any] = {}
    updatedAt: Optional[str] = None
    lookbackSnapshots: Optional[int] = None
    chat_history: List[Dict[str, Any]] = []
    total_value: float = 0.0
    total_gain: float = 0.0
    today_gain: float = 0.0

    positions: List[AstraPosition] = []
    
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
    if req.contextType not in ("stock_detail", "market", "momentum_movers"):
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
        return firebase_admin.get_app()

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

def get_crypto_logo_url(symbol: str):
    try:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return None

        snap = (
            db.collection("bullsignals_ai")
              .document("logos")
              .collection("crypto")
              .document(sym)
              .get()
        )

        if snap.exists:
            data = snap.to_dict() or {}
            return data.get("logoUrl")
    except Exception:
        pass

    return None


def extract_context_symbol(label: str):
    label = str(label or "").strip()

    # Handles "Bitcoin (BTC)" or "S&P 500 (SPY)"
    if "(" in label and ")" in label:
        try:
            return label.split("(")[-1].split(")")[0].strip().upper()
        except Exception:
            pass

    return label.upper()


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

    carousel = data.get("carousel", []) or []

    enriched_carousel = []

    for card in carousel:
        new_card = dict(card)
        items = card.get("items", []) or []
        new_items = []

        for item in items:
            new_item = dict(item)
            symbol = (
                new_item.get("symbol")
                or extract_context_symbol(new_item.get("label"))
            )

            symbol = str(symbol or "").upper().strip()

            if symbol in {"BTC", "ETH", "SOL", "XRP", "DOGE"}:
                new_item["symbol"] = symbol
                new_item["logoUrl"] = get_crypto_logo_url(symbol)

            new_items.append(new_item)

        new_card["items"] = new_items
        enriched_carousel.append(new_card)

    return {
        "market": data.get("market_overview"),
        "carousel": enriched_carousel,
        "updated_at": data.get("updated_at"),
        "version": data.get("version", "v2"),
    }


@app.get("/homescreen-data")
def homescreen_data():
    cache = read_market_cache("homescreen_snapshot")
    alpha_watch = read_market_cache("alpha_watch") or {}

    safe_alpha_watch = {
        "title": alpha_watch.get("title", "AI Opportunity Watch"),
        "subtitle": alpha_watch.get(
            "subtitle",
            "AI-ranked setups showing momentum, trend quality, pattern edge, and participation.",
        ),
        "count": alpha_watch.get("count", 0),
        "items": alpha_watch.get("items", []),
        "market_regime": alpha_watch.get("market_regime"),
        "updated_at": alpha_watch.get("updated_at"),
        "disclaimer": alpha_watch.get("disclaimer"),
        "schema_version": alpha_watch.get("schema_version"),
    }

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
            "alpha_watch": safe_alpha_watch,
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
        "alpha_watch": safe_alpha_watch,
    }

def slim_sparkline_payload(sparkline):
    if not isinstance(sparkline, dict) or not sparkline.get("path"):
        return None

    range_stats = sparkline.get("rangeStats") or {}

    return {
        "path": sparkline.get("path"),
        "min": sparkline.get("min"),
        "max": sparkline.get("max"),
        "direction": sparkline.get("direction"),
        "range": sparkline.get("range"),
        "basis": sparkline.get("basis"),
        "rangeStats": {
            "closeLow": range_stats.get("closeLow"),
            "closeHigh": range_stats.get("closeHigh"),
            "returnPct": range_stats.get("returnPct"),
            "candleCount": range_stats.get("candleCount"),
        },
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
    stock["symbol"] = sym

    # ---------------------------------------------------------
    # 2️⃣ Header
    # ---------------------------------------------------------
    from backend.header_builder import build_stock_header
    header = build_stock_header(stock)

    # ---------------------------------------------------------
    # 3️⃣ Sparkline
    # ---------------------------------------------------------
    sparkline = None

    try:
        from backend.ui_stock_builder import (
            build_sparkline,
            build_sparkline_from_prices,
        )

        existing_prices = stock.get("sparkline")
        chart_meta = stock.get("chart") or {}

        if isinstance(existing_prices, list) and len(existing_prices) >= 2:
            sparkline = build_sparkline_from_prices(existing_prices, meta=chart_meta)

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
    # 4️⃣ Company News
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
    # 5️⃣ Stock Detail Content
    # ---------------------------------------------------------
    from backend.ui_stock_builder import (
        build_stockdetail_v1,
        build_stockdetail_ui_v1,
    )

    if source == "ui":
        content = build_stockdetail_ui_v1(stock) or {}
    else:
        content = build_stockdetail_v1(stock) or {}
    # ---------------------------------------------------------
    # 5.1️⃣ Canonical AI Market Rating
    # ---------------------------------------------------------
    display_intelligence = stock.get("displayIntelligence")

    if isinstance(display_intelligence, dict):
        content["displayIntelligence"] = display_intelligence
    # Keep full sparkline for non-UI, slim sparkline for UI
    if sparkline:
        content["sparkline"] = (
            slim_sparkline_payload(sparkline)
            if source == "ui"
            else sparkline
        )

    # ---------------------------------------------------------
    # 6️⃣ Final Response
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
    return (symbol or "").upper().strip()

MARKET_WATCHLIST_SYMBOLS = {"SPY", "QQQ", "GLD", "USO", "SLV"}
VALID_WATCHLIST_SYMBOLS = set(REAL_TICKERS) | MARKET_WATCHLIST_SYMBOLS

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
    SNAPSHOT_VERSION = "v8"

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
from backend.watchlist_symbols import increment_watchlist_symbol, decrement_watchlist_symbol

@app.post("/watchlist/{user_id}/add/{symbol}")
def add_watchlist_symbol(user_id: str, symbol: str):
    sym = _norm_symbol(symbol)
    if not sym.replace(".", "").isalnum():
        return {"status": "error", "error": "Invalid symbol"}

    if sym not in VALID_WATCHLIST_SYMBOLS:
        return {
            "status": "error",
            "error": f"'{sym}' is not a recognized ticker symbol and can't be added to your watchlist",
        }

    # 0️⃣ Was this symbol already on the user's watchlist?
    watchlist_doc_ref = _watchlist_col(user_id).document(sym)
    already_watched = watchlist_doc_ref.get().exists

    # 1️⃣ Save user intent
    watchlist_doc_ref.set(
        {"symbol": sym, "added_at": firestore.SERVER_TIMESTAMP},
        merge=True
    )

    # 1️⃣.5 Update global watchlist aggregate (skip if already watched)
    if not already_watched:
        try:
            increment_watchlist_symbol(sym)
        except Exception as e:
            print(f"[watchlist] aggregate increment failed for {sym}: {e}")

    # 2️⃣ Mark symbol as active (global relevance)
    try:
        touch_active_symbol(sym)
    except Exception:
        pass

    # 3️⃣ Warm quote immediately first
    quote_status = "pending"

    try:
        from backend.quote_provider import fetch_equity_quote
        from backend.quote_repo import save_quote

        live_quote = fetch_equity_quote(sym)

        if live_quote and live_quote.get("price") is not None:
            live_quote["needs_refresh"] = False
            save_quote(sym, live_quote)
            quote_status = "ready"
        else:
            ensure_quote(sym)
            quote_status = "pending"

    except Exception as e:
        print(f"[watchlist] quote warm failed for {sym}: {e}")
        try:
            ensure_quote(sym)
        except Exception:
            pass
        quote_status = "pending"


    # 4️⃣ Warm intelligence best-effort
    intelligence_status = "pending"

    try:
        bootstrap_stock(sym)
        intelligence_status = "ready"
    except Exception as e:
        print(f"[watchlist] bootstrap pending for {sym}: {e}")
        intelligence_status = "pending"

    # 4️⃣ 🔥 FORCE snapshot rebuild (ignore TTL)
    try:
        build_watchlist_snapshot(user_id)
    except Exception as e:
        print(f"[watchlist] snapshot rebuild failed: {e}")

    return {
        "status": "ok",
        "user_id": user_id,
        "symbol": sym,
        "quote_status": quote_status,
        "intelligence_status": intelligence_status,
    }


# -----------------------------
# 3) REMOVE symbol from watchlist
# -----------------------------
from backend.watchlist_snapshot import build_watchlist_snapshot

@app.delete("/watchlist/{user_id}/remove/{symbol}")
def remove_watchlist_symbol(user_id: str, symbol: str):
    sym = _norm_symbol(symbol)

    # 0️⃣ Was this symbol actually on the user's watchlist?
    watchlist_doc_ref = _watchlist_col(user_id).document(sym)
    was_watched = watchlist_doc_ref.get().exists

    # 1️⃣ Remove from watchlist
    watchlist_doc_ref.delete()

    # 1️⃣.5 Update global watchlist aggregate (skip if it wasn't actually watched)
    if was_watched:
        try:
            decrement_watchlist_symbol(sym)
        except Exception as e:
            print(f"[watchlist] aggregate decrement failed for {sym}: {e}")

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
def get_market_movers(mode: str = "preview"):
    try:
        snap = (
            db.collection("bullsignals_ai")
              .document("market_movers")
              .get()
        )

        if not snap.exists:
            return {
                "count": 0,
                "movers": [],
                "gainers": [],
                "losers": [],
                "as_of": None,
                "updated_at": None,
                "version": "empty",
            }

        data = snap.to_dict() or {}

        home_rising = data.get("home_rising", []) or []
        preview_rising = data.get("preview_rising", []) or []
        preview_falling = data.get("preview_falling", []) or []
        all_movers = data.get("movers", []) or []

        if mode == "home":
            movers = home_rising
        elif mode == "all":
            movers = all_movers
        else:
            movers = preview_rising + preview_falling

        gainers = [m for m in movers if m.get("direction") == "up"]
        losers = [m for m in movers if m.get("direction") == "down"]

        return {
            "count": len(movers),
            "movers": movers,
            "gainers": gainers,
            "losers": losers,
            "as_of": data.get("as_of"),
            "updated_at": data.get("updated_at"),
            "source": data.get("source"),
            "version": data.get("schema_version", "market_movers_v4"),
        }

    except Exception as e:
        print("[market-movers] error:", e)
        return {
            "count": 0,
            "movers": [],
            "gainers": [],
            "losers": [],
            "as_of": None,
            "updated_at": None,
            "error": str(e),
        }
    

@app.get("/verified-alpha")
def get_verified_alpha():
    """
    AI-curated opportunity endpoint, two honestly-labeled tiers:
    - tier="validated": alpha_watch's own quality+conviction screen
      (passes_quality_filter() + score>=55) -- shown first.
    - tier="momentum": MOMENTUM_OVERRIDE candidates (cached by
      market_cron.py's persist_momentum_override_candidates(), not
      computed here) backfilling remaining slots up to MAX_ITEMS, any
      symbol already used in tier "validated" excluded.

    riskLevel intentionally omitted, not fabricated -- confirmed dead/
    unrendered this week's Momentum risk-label fix.

    No Grok pipeline. No verified_alpha_opportunities dependency.
    No duplicated market movers.
    """
    MAX_ITEMS = 10

    try:
        alpha_snap = (
            db.collection("bullsignals_ai")
            .document("alpha_watch")
            .get()
        )

        alpha_data = alpha_snap.to_dict() if alpha_snap.exists else {}
        ai_items = alpha_data.get("items", []) or []

        alpha_opportunities = []
        used_symbols = set()

        for item in ai_items:
            if len(alpha_opportunities) >= MAX_ITEMS:
                break

            symbol = item.get("symbol")
            if not symbol:
                continue

            used_symbols.add(str(symbol).upper())
            alpha_opportunities.append({
                "symbol": symbol,
                "companyName": item.get("companyName"),
                "logoUrl": item.get("logoUrl"),
                "price": item.get("price"),
                "change": item.get("change"),
                "changePct": item.get("changePct"),
                "signal": item.get("signal"),
                "confidence": item.get("confidence"),
                "probUp": item.get("probUp"),
                "score": item.get("score"),
                "opportunityScore": item.get("opportunityScore"),
                "marketMomentumBonus": item.get("marketMomentumBonus"),
                "setupLabel": item.get("setupLabel"),
                "pattern": item.get("pattern"),
                "reason": item.get("reason"),
                "whyNow": item.get("whyNow") or [],
                "riskFlags": item.get("riskFlags") or [],
                "theme": item.get("theme"),
                "marketRegime": item.get("marketRegime"),
                "factorScores": item.get("factorScores") or {},
                "quote_updated_at": item.get("quote_updated_at"),
                "computed_at": item.get("computed_at"),
                "source": "alpha_watch",
                "tier": "validated",
            })

        momentum_items = []
        remaining = MAX_ITEMS - len(alpha_opportunities)

        if remaining > 0:
            mo_snap = (
                db.collection("bullsignals_ai")
                .document("momentum_override_candidates")
                .get()
            )
            mo_data = mo_snap.to_dict() if mo_snap.exists else {}

            for item in (mo_data.get("items") or []):
                if len(momentum_items) >= remaining:
                    break

                symbol = item.get("symbol")
                if not symbol or str(symbol).upper() in used_symbols:
                    continue

                momentum_items.append({
                    "symbol": symbol,
                    "companyName": item.get("companyName"),
                    "logoUrl": item.get("logoUrl"),
                    "price": item.get("price"),
                    "change": item.get("change"),
                    "changePct": item.get("changePct"),
                    "signal": item.get("signal"),
                    "confidence": item.get("confidence"),
                    "probUp": item.get("probUp"),
                    "score": None,
                    "opportunityScore": None,
                    "marketMomentumBonus": None,
                    "setupLabel": item.get("overrideType"),
                    "pattern": item.get("pattern"),
                    "reason": (
                        f"Momentum override: strong price/volume signal triggered a "
                        f"{item.get('originalModelSignal') or 'directional'} call."
                    ),
                    "whyNow": [],
                    "riskFlags": [],
                    "theme": None,
                    "marketRegime": None,
                    "factorScores": {},
                    "quote_updated_at": item.get("quote_updated_at"),
                    "computed_at": item.get("computed_at"),
                    "source": "momentum_override",
                    "tier": "momentum",
                })

        all_opportunities = alpha_opportunities + momentum_items

        return {
            "status": "ok",
            "source": "internal_alpha_watch",
            "updated_at": alpha_data.get("updated_at"),
            "market_regime": alpha_data.get("market_regime"),
            "title": "AI Opportunity Watch",
            "counts": {
                "alpha_opportunities": len(all_opportunities),
                "validated": len(alpha_opportunities),
                "momentum": len(momentum_items),
            },
            "alpha_opportunities": all_opportunities,
            "schema_version": "verified_alpha_internal_v4",
            "fallback_used": False,
        }

    except Exception as e:
        print("[verified-alpha] error:", e)

        return {
            "status": "error",
            "source": "internal_alpha_watch",
            "error": str(e),
            "counts": {
                "alpha_opportunities": 0,
                "validated": 0,
                "momentum": 0,
            },
            "alpha_opportunities": [],
            "schema_version": "verified_alpha_internal_v4",
        }


@app.get("/alphaclara-tracking")
def get_alphaclara_tracking(
    limit: Optional[int] = None,
    window_days: int = 3,
):
    """
    "Alphaclara is Tracking" -- honest, real-time accountability for past
    Alpha Watch picks, sourced from bullsignals_ai/pick_tracking/picks.

    Design, confirmed before implementation:
    - Deduped by symbol -- the cron records a fresh pick_tracking row every
      ~15min for each symbol still in alpha_watch (by design, see
      backend/pick_tracking.py), so the raw window is dominated by repeat
      rows of the same handful of symbols rather than distinct picks. This
      endpoint shows only the most recent row per symbol within the window;
      the full row-by-row history stays intact in pick_tracking, untouched.
    - Two independent queries, same WINDOW_DAYS constant reused for both,
      unioned: (1) picks whose pick_date falls in the window (still
      tracking, live price shown), (2) picks whose 5d horizon was checked
      within the same window (the "loop closes" case -- shows the frozen
      checked result, not a live price). No separate "extra day" constant,
      no separate "graduated" section -- everything sorts together by
      recency, newest first.
    - Gains and losses both shown identically, no filtering.
    - `limit` and `window_days` are both optional and additive: omitted,
      behavior is identical to the unparameterized endpoint (unlimited
      items, 3-day window) for the compact Home preview. A "View All"
      screen passes both explicitly, e.g. limit=None, window_days=30 for
      full history. window_days is capped at 30 -- raw (pre-dedupe) doc
      volume scales directly with it (3 days is already ~2,486 raw reads
      before dedup), so an uncapped value could blow up Firestore reads
      per request.
    """
    WINDOW_DAYS = min(window_days, 30)

    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        today = now.date()
        window_start_date = (today - datetime.timedelta(days=WINDOW_DAYS - 1)).isoformat()
        window_start_dt = (
            datetime.datetime.combine(
                today - datetime.timedelta(days=WINDOW_DAYS - 1),
                datetime.time.min,
                tzinfo=datetime.timezone.utc,
            ).isoformat().replace("+00:00", "Z")
        )

        picks_col = (
            db.collection("bullsignals_ai")
            .document("pick_tracking")
            .collection("picks")
        )

        # Firestore's raw dotted-string field-path parser chokes on a
        # segment starting with a digit ("5d") -- needs the escaped API
        # representation (backtick-quoted), not a plain "horizons.5d.
        # checked_at" string. Confirmed by reproducing the parser error
        # directly before fixing.
        from google.cloud.firestore_v1.field_path import FieldPath
        checked_at_path = FieldPath("horizons", "5d", "checked_at").to_api_repr()

        recent_docs = list(picks_col.where("pick_date", ">=", window_start_date).stream())
        graduated_docs = list(
            picks_col.where(checked_at_path, ">=", window_start_dt).stream()
        )

        seen_ids = set()
        raw_items = []
        for doc in recent_docs + graduated_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            raw_items.append(doc.to_dict() or {})

        # Dedupe by symbol -- keep only the most recently recorded pick per
        # symbol within the window. The cron re-records every symbol still
        # in alpha_watch each cycle, so without this a single symbol can
        # show dozens of near-identical cards. Full history is untouched
        # in pick_tracking; this only trims what this endpoint displays.
        #
        # The kept record's own pick_date/pick_price are always recent
        # (it's the latest recorded row), which silently hides both a real
        # multi-day streak and its true starting price -- a symbol picked
        # continuously since day 1 of the window looks exactly like one
        # picked for the first time today, AND its "since picked" price
        # keeps sliding forward every cron cycle instead of anchoring to
        # when it actually first qualified. earliest_by_symbol fixes both:
        # the full earliest-recorded row per symbol (by recorded_at, not by
        # pick_date string -- multiple rows can share the same pick_date,
        # and Firestore's stream() order isn't guaranteed, so recorded_at
        # is the only reliable tiebreak) across all raw (pre-dedupe) rows
        # in this same window, tracked in the same pass before the rest of
        # the rows are discarded -- no second lookup. Bounded by
        # window_days like everything else here -- a streak longer than
        # the requested window will show the window's own start, not the
        # true all-time first pick (that would need an unbounded scan).
        latest_by_symbol: Dict[str, Dict[str, Any]] = {}
        earliest_by_symbol: Dict[str, Dict[str, Any]] = {}
        for it in raw_items:
            symbol = str(it.get("symbol") or "").upper()
            if not symbol:
                continue
            existing_latest = latest_by_symbol.get(symbol)
            if existing_latest is None or (it.get("recorded_at") or "") > (existing_latest.get("recorded_at") or ""):
                latest_by_symbol[symbol] = it

            existing_earliest = earliest_by_symbol.get(symbol)
            if existing_earliest is None or (it.get("recorded_at") or "") < (existing_earliest.get("recorded_at") or ""):
                earliest_by_symbol[symbol] = it
        raw_items = list(latest_by_symbol.values())

        symbols_needed = {
            str(it.get("symbol") or "").upper()
            for it in raw_items
            if it.get("symbol")
        }
        stock_by_symbol = {}
        for sym in symbols_needed:
            snap = (
                db.collection("bullsignals_ai")
                .document("stocks")
                .collection("symbols")
                .document(sym)
                .get()
            )
            stock_by_symbol[sym] = snap.to_dict() if snap.exists else {}

        # Tier freshness uses the US/Eastern trading day, not naive UTC --
        # UTC midnight falls mid-evening ET (7-8pm depending on DST), hours
        # before a US user's own day is over, so a pick made mid-afternoon
        # ET could already read as "yesterday" once UTC has rolled over,
        # even though it's still today for anyone watching the market.
        # Mirrors quote_worker.py's is_market_open()/is_weekend() -- same
        # ZoneInfo("America/New_York") conversion, same reasoning: the
        # trading calendar is the one objective "day" here, not wherever
        # the requesting user happens to be. Scoped to this comparison only
        # -- today/window_start_date/window_start_dt above stay UTC, since
        # those bound the Firestore query against UTC-stamped pick_date/
        # checked_at values and must match that storage convention.
        today_str = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()

        items = []
        for it in raw_items:
            symbol = str(it.get("symbol") or "").upper()
            if not symbol:
                continue

            stock = stock_by_symbol.get(symbol) or {}
            profile = stock.get("profile") or {}
            quote = stock.get("quote") or {}
            pick_price = it.get("pick_price")
            first_pick = earliest_by_symbol.get(symbol) or {}
            first_picked_price = first_pick.get("pick_price")
            h5 = (it.get("horizons") or {}).get("5d") or {}
            h5_status = h5.get("status")

            entry = {
                "symbol": symbol,
                "companyName": stock.get("company_name"),
                "logoUrl": profile.get("logoUrl"),
                "pick_date": it.get("pick_date"),
                "first_picked_date": first_pick.get("pick_date"),
                "recorded_at": it.get("recorded_at"),
                # pick_price/livePct(SinceLastUpdate) reflect the most
                # recently re-recorded snapshot -- meaningful for a symbol
                # just picked today, but for a multi-day streak this keeps
                # sliding forward every cron cycle instead of anchoring to
                # when the symbol actually first qualified. first_picked_price/
                # livePctSinceFirstPick is the honest "since picked" number
                # for that case. Both exposed, unrenamed/untouched existing
                # fields kept as-is, so the frontend picks the one that
                # matches what it's labeling.
                "pick_price": pick_price,
                "first_picked_price": first_picked_price,
                "pick_reason": it.get("pick_reason"),
                "pick_setup_label": it.get("pick_setup_label"),
                "pick_model_view": it.get("pick_model_view"),
            }

            if h5_status == "checked":
                entry["status"] = "checked"
                entry["checked_price"] = h5.get("price")
                entry["checked_return_pct"] = h5.get("return_pct")
                entry["checked_at"] = h5.get("checked_at")
                entry["horizon"] = "5d"
            elif h5_status == "unavailable":
                entry["status"] = "unavailable"
                entry["unavailable_reason"] = h5.get("unavailable_reason")
                entry["checked_at"] = h5.get("checked_at")
            else:
                entry["status"] = "tracking"
                current_price = quote.get("price")
                entry["current_price"] = current_price
                entry["current_price_updated_at"] = quote.get("updated_at")
                if (
                    isinstance(current_price, (int, float))
                    and isinstance(pick_price, (int, float))
                    and pick_price
                ):
                    entry["livePct"] = round((current_price / pick_price - 1) * 100, 2)
                else:
                    entry["livePct"] = None
                entry["livePctSinceLastUpdate"] = entry["livePct"]

                if (
                    isinstance(current_price, (int, float))
                    and isinstance(first_picked_price, (int, float))
                    and first_picked_price
                ):
                    entry["livePctSinceFirstPick"] = round(
                        (current_price / first_picked_price - 1) * 100, 2
                    )
                else:
                    entry["livePctSinceFirstPick"] = None

            # Tier for the Pick Detail screen's three-way grouping. Checked
            # takes priority over freshness -- a completed horizon is never
            # "fresh" even if it happens to be first_picked_date == today.
            # "fresh" means genuinely new to the list (first_picked_date is
            # today, no earlier record in the window), not just re-recorded
            # -- distinct from "tracking," which covers every other still-
            # open pick regardless of how long it's been tracked.
            if entry["status"] in ("checked", "unavailable"):
                entry["tier"] = "checked"
            elif entry.get("first_picked_date") == today_str:
                entry["tier"] = "fresh"
            else:
                entry["tier"] = "tracking"

            items.append(entry)

        items.sort(key=lambda e: e.get("checked_at") or e.get("recorded_at") or "", reverse=True)

        if limit is not None:
            items = items[:limit]

        return {
            "status": "ok",
            "title": "Alphaclara is Tracking",
            "subtitle": "Real setups Alphaclara flagged, tracked honestly — wins and losses both.",
            "updated_at": now.isoformat().replace("+00:00", "Z"),
            "window_days": WINDOW_DAYS,
            "items": items,
            "counts": {
                "total": len(items),
                "tracking": sum(1 for i in items if i["status"] == "tracking"),
                "checked": sum(1 for i in items if i["status"] == "checked"),
                "unavailable": sum(1 for i in items if i["status"] == "unavailable"),
            },
        }

    except Exception as e:
        print("[alphaclara-tracking] error:", e)
        return {
            "status": "error",
            "error": str(e),
            "title": "Alphaclara is Tracking",
            "items": [],
            "counts": {"total": 0, "tracking": 0, "checked": 0, "unavailable": 0},
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
    # 2️⃣ Ensure quote docs exist / mark stale for refresh
    #    - No external API call here
    #    - Worker refreshes in background
    # ---------------------------------------------------------
    for sym in symbol_list:
        try:
            data = ensure_quote(sym) or {}

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

        except Exception as e:
            print(f"[quotes-bulk] failed for {sym}: {e}")

            quotes[sym] = {
                "symbol": sym,
                "price": None,
                "change": None,
                "changePct": None,
                "updated_at": None,
                "needs_refresh": True,
                "ttl_seconds": 30,
                "source": "error",
            }

    return {
        "scope": scope,
        "count": len(quotes),
        "quotes": quotes,
    }


@app.get("/market-momentum")
def market_momentum():
    return get_market_momentum_screen()


@app.post("/market-momentum/refresh")
def refresh_market_momentum():
    return save_market_momentum_screen()