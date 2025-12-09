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
                        "winRate": 0.73,
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
                    "winRate": 0.76,
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
                        "winRate": 0.80,
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
                        "winRate": 0.74,
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
                        "winRate": 0.69,
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
                        "winRate": 0.68,
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
                        "winRate": 0.67,
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
                        "winRate": 0.66,
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
                        "winRate": 0.62,
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
                        "winRate": 0.60,
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
                        "winRate": 0.70,
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
                        "winRate": 0.65,
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
                "headline": patt["headline"],
                "winRate": patt["winRate"],
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

# -----------------------------------------------------------
# SMART PATTERN DETECTOR (Hedge-Fund Level Pattern Engine)
# -----------------------------------------------------------
def detect_smart_pattern(features: dict, quote: dict, technical: dict):
    """
    Detect institutional-grade smart patterns using your 48-feature set,
    polygon daily candles, and the technical snapshot. Returns the strongest
    detected pattern with a human-friendly explanation and historical win rate.
    """

    if not features:
        return None

    # --- Extract key feature values (safe) ---
    gap = features.get("gap_pct")
    change = quote.get("changePct") if quote else None
    vol_z = features.get("volume_zscore_20")
    vol_ma20 = features.get("volume_vs_ma20_pct")
    rsi = features.get("rsi14")
    willr = features.get("williams_r_14")
    lower_shadow = features.get("lower_shadow_pct")
    body_pct = features.get("body_pct")
    price_vs_sma20 = features.get("price_vs_sma20_pct")
    trend = features.get("trend_strength_20")
    ret5 = features.get("return_5d")
    atr = features.get("atr14")
    range_pct = features.get("intraday_range_pct")
    stoch_k = features.get("stoch_k_14")
    stoch_d = features.get("stoch_d_3")
    sma5 = features.get("sma5")
    sma10 = features.get("sma10")
    sma20 = features.get("sma20")

    patterns = []

    # ------------------------------------------------------------
    # 1) GAP UP & RUNNING
    # ------------------------------------------------------------
    if gap and gap > 1 and change and change > 2 and vol_ma20 and vol_ma20 > 20:
        patterns.append({
            "pattern": "GAP UP & RUNNING",
            "winRate": 0.73,
            "explanation": (
                "The stock opened sharply higher than yesterday and kept climbing on strong volume. "
                "This is a classic sign of momentum ignition — big buyers stepped in early."
            )
        })

    # ------------------------------------------------------------
    # 2) MASSIVE VOLUME BREAKOUT
    # ------------------------------------------------------------
    if vol_z and vol_z > 3:
        patterns.append({
            "pattern": "MASSIVE VOLUME BREAKOUT",
            "winRate": 0.76,
            "explanation": (
                "Trading volume today is extremely high — the kind usually driven by large "
                "institutional activity. Such surges often precede major price moves."
            )
        })

    # ------------------------------------------------------------
    # 3) OVERSOLD BOUNCE
    # ------------------------------------------------------------
    if rsi and rsi < 30 and willr and willr < -80 and vol_z and vol_z > 2:
        patterns.append({
            "pattern": "OVERSOLD BOUNCE",
            "winRate": 0.80,
            "explanation": (
                "The stock reached an extreme oversold level, causing panic selling. "
                "But large buyers stepped in with strong volume, often leading to a sharp rebound."
            )
        })

    # ------------------------------------------------------------
    # 4) HAMMER REVERSAL
    # ------------------------------------------------------------
    if lower_shadow and lower_shadow > 2.5 and body_pct > -1 and change and change > 0:
        patterns.append({
            "pattern": "HAMMER REVERSAL",
            "winRate": 0.74,
            "explanation": (
                "Sellers pushed the stock down aggressively, but buyers reversed it and closed near the highs. "
                "This candle shape is a classic sign of a potential bottom forming."
            )
        })

    # ------------------------------------------------------------
    # 5) BUY THE DIP (UPTREND)
    # ------------------------------------------------------------
    if trend and trend > 1 and price_vs_sma20 and price_vs_sma20 < -3 and change > 0:
        patterns.append({
            "pattern": "BUY THE DIP (UPTREND)",
            "winRate": 0.69,
            "explanation": (
                "The stock is in a strong uptrend and recently pulled back to a normal level. "
                "Today’s bounce suggests buyers are stepping back in — a healthy continuation signal."
            )
        })

    # ------------------------------------------------------------
    # 6) DEAD CAT BOUNCE
    # ------------------------------------------------------------
    if ret5 and ret5 < -8 and change and change > 0 and (vol_z is not None and vol_z < 1):
        patterns.append({
            "pattern": "DEAD CAT BOUNCE",
            "winRate": 0.68,
            "explanation": (
                "After a major crash, the stock had a weak rebound with low volume — typically a fake recovery. "
                "These setups often fail and lead to another leg lower."
            )
        })

    # ------------------------------------------------------------
    # 7) OVERBOUGHT DISTRIBUTION
    # ------------------------------------------------------------
    if rsi and rsi > 70 and vol_ma20 and vol_ma20 < 0:
        patterns.append({
            "pattern": "OVERBOUGHT DISTRIBUTION",
            "winRate": 0.67,
            "explanation": (
                "The stock has risen too quickly into overbought territory. "
                "Volume is drying up, suggesting large investors may be quietly taking profits."
            )
        })

    # ------------------------------------------------------------
    # 8) FAILED BREAKOUT TRAP
    # ------------------------------------------------------------
    if change and change < -2 and vol_z and vol_z > 2:
        patterns.append({
            "pattern": "FAILED BREAKOUT TRAP",
            "winRate": 0.66,
            "explanation": (
                "The stock attempted a breakout but immediately failed on high volume — a classic bull trap. "
                "This often leads to accelerated downside pressure."
            )
        })

    # ------------------------------------------------------------
    # 9) BULL FLAG
    # ------------------------------------------------------------
    if trend and trend > 2 and price_vs_sma20 and -5 < price_vs_sma20 < 1:
        patterns.append({
            "pattern": "BULL FLAG",
            "winRate": 0.72,
            "explanation": (
                "After a strong rally, the stock is moving sideways on light volume. "
                "This calm pullback often leads to the next upward move."
            )
        })

    # ------------------------------------------------------------
    # 10) BEAR FLAG BREAKDOWN
    # ------------------------------------------------------------
    if trend and trend < -2 and ret5 and ret5 < -4 and change and change < 0:
        patterns.append({
            "pattern": "BEAR FLAG BREAKDOWN",
            "winRate": 0.71,
            "explanation": (
                "The stock fell sharply, attempted a weak recovery, and is now resuming its move down. "
                "This is a classic continuation pattern in downtrends."
            )
        })

    # ------------------------------------------------------------
    # 11) SHORT SQUEEZE SETUP
    # ------------------------------------------------------------
    if rsi and rsi < 35 and change and change > 3 and vol_z and vol_z > 2:
        patterns.append({
            "pattern": "SHORT SQUEEZE SETUP",
            "winRate": 0.78,
            "explanation": (
                "After a period of heavy shorting, a big green candle with strong volume suggests "
                "short sellers may be getting squeezed — often leading to rapid upside moves."
            )
        })

    # ------------------------------------------------------------
    # 12) LONG LIQUIDATION FLUSH
    # ------------------------------------------------------------
    if change and change < -3 and vol_z and vol_z > 2 and range_pct and range_pct > 5:
        patterns.append({
            "pattern": "LONG LIQUIDATION FLUSH",
            "winRate": 0.72,
            "explanation": (
                "A large red candle with high volume indicates forced selling by long holders. "
                "These panic flushes often mark short-term bottoms."
            )
        })

    # ------------------------------------------------------------
    # 13) VOLATILITY EXPANSION
    # ------------------------------------------------------------
    if atr and atr > 20 and range_pct and range_pct > 5:
        patterns.append({
            "pattern": "VOLATILITY EXPANSION",
            "winRate": 0.70,
            "explanation": (
                "Daily price swings are increasing sharply. The stock is entering a high-volatility phase — "
                "expect bigger moves in both directions."
            )
        })

    # ------------------------------------------------------------
    # 14) VOLATILITY COMPRESSION
    # ------------------------------------------------------------
    if atr and atr < 10 and vol_ma20 and vol_ma20 < 0 and range_pct and range_pct < 2:
        patterns.append({
            "pattern": "VOLATILITY COMPRESSION",
            "winRate": 0.64,
            "explanation": (
                "Price movement is tightening and volatility is shrinking. "
                "This calm period often precedes a strong breakout move."
            )
        })

    # ------------------------------------------------------------
    # 15) MOMENTUM REVERSAL WARNING
    # ------------------------------------------------------------
    if rsi and rsi < 60 and rsi > 40 and change and change < 0 and sma5 and sma10 and sma5 < sma10:
        patterns.append({
            "pattern": "MOMENTUM REVERSAL WARNING",
            "winRate": 0.68,
            "explanation": (
                "Short-term momentum is weakening and buyers are losing control. "
                "The stock may be preparing for a trend reversal."
            )
        })

    # ------------------------------------------------------------
    # 16) TREND ACCELERATION
    # ------------------------------------------------------------
    if sma5 and sma10 and sma20 and (sma5 > sma10 > sma20) and change and change > 1:
        patterns.append({
            "pattern": "TREND ACCELERATION",
            "winRate": 0.74,
            "explanation": (
                "Short, medium, and long-term trends are aligned. "
                "The stock is accelerating in the direction of the trend — a strong continuation signal."
            )
        })

    # ------------------------------------------------------------
    # Return strongest pattern (highest win rate)
    # ------------------------------------------------------------
    if patterns:
        return sorted(patterns, key=lambda x: x["winRate"], reverse=True)[0]

    return {
        "pattern": "NO CLEAR PATTERN",
        "winRate": None,
        "explanation": "Today's price action does not match any strong institutional pattern."
    }


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
            "temperature": 0.4,
            "max_tokens": 600,
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

        # BULLBRAIN
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
            last_close = float(quote.get("current") or 0.0)

        # TECHNICAL SNAPSHOT
        technical = None
        if feature_dict is not None and last_close is not None:
            technical = build_technical_snapshot(symbol, feature_dict, last_close)

        # CANDLES PAYLOAD
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
                    t_raw = ts_list[i] if i < len(ts_list) else None
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

        # NEWS + GROK
        news = get_symbol_news(symbol, limit=8)
        grok_pack = get_stockdetail_grok(symbol, quote, technical, force=forceGrok)
        grok_prob_up = grok_pack.get("prob_up")

        # HYBRID
        hybrid_p, hybrid_signal, hybrid_conf = _hybrid_from_probs(
            bull_prob_up, grok_prob_up
        )

        # -----------------------------------------------------------
        # SMART PATTERN + HISTORY (SAFE WRAPPER)
        # -----------------------------------------------------------
        raw_ph = scan_smart_pattern_history(symbol, candles)

        safe_smart_pattern = None
        safe_pattern_dates = []
        safe_pattern_stats = raw_ph  # return full stats for debugging/UI

        if raw_ph:
            cp = raw_ph.get("currentPattern")
            hist = raw_ph.get("historyForCurrent")

            # If we have a valid pattern for today
            if cp and cp.get("pattern"):
                safe_smart_pattern = {
                    "pattern": cp.get("pattern"),
                    "headline": cp.get("headline"),
                    "winRate": cp.get("winRate"),
                    "occurrences": hist.get("occurrences") if hist else 0,
                    "samples": hist.get("samples") if hist else [],
                    "forwardReturns": hist.get("forwardReturns") if hist else {},
                }

                if hist and hist.get("samples"):
                    safe_pattern_dates = hist["samples"][:5]

            else:
                # No pattern today — return clean structure (prevents frontend crash)
                safe_smart_pattern = {
                    "pattern": None,
                    "headline": None,
                    "winRate": None,
                    "occurrences": 0,
                    "samples": [],
                    "forwardReturns": {},
                }
                safe_pattern_dates = []

        else:
            # No history at all
            safe_smart_pattern = {
                "pattern": None,
                "headline": None,
                "winRate": None,
                "occurrences": 0,
                "samples": [],
                "forwardReturns": {},
            }
            safe_pattern_dates = []

        # FINAL RESPONSE
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

            # NEW — Smart pattern data for UI
            "smartPattern": safe_smart_pattern,
            "patternDates": safe_pattern_dates,
            "patternStats": safe_pattern_stats,
        }

    except Exception as e:
        print("stockdetail error:", e)
        return {"symbol": symbol, "error": str(e)}

# --------------------------------------------------------------------
# SMART PATTERN HISTORY ENDPOINT
# --------------------------------------------------------------------
@app.get("/patternhistory/{symbol}")
def pattern_history(symbol: str, lookahead_5: int = 5, lookahead_10: int = 10):
    symbol = symbol.upper()
    try:
        candles = fetch_daily_candles(symbol)
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

# -----------------------------------------
# News cleanup helpers for /market-news
# -----------------------------------------

SOURCE_MAP = {
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "finance.yahoo.com": "Yahoo Finance",
    "investing.com": "Investing.com",
    "investors.com": "Investor's Business Daily",
    "barrons.com": "Barron's",
}

# Tickers we KNOW are garbage from headlines (English words, etc.)
NOISY_TICKERS = {
    "A", "I", "U", "T", "ON", "UP", "DAY", "IT", "ARE", "HAS",
    "FAST", "COST", "TECH"
}


def clean_summary(summary: str | None, title: str) -> str:
    """
    - Replace '...' or empty with title.
    - Keep summary if it is non-trivial.
    """
    if not summary:
        return title

    s = summary.strip()
    if not s or s == "..." or len(s) < 10:
        return title

    return s


def normalize_source(source: str | None, link: str | None) -> str:
    """
    - Prefer explicit source if present.
    - Else derive from URL domain and map to pretty name.
    """
    if source:
        s = source.strip()
        if s:
            return s

    if not link:
        return "Unknown"

    try:
        domain = urlparse(link).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return SOURCE_MAP.get(domain, domain.split(":")[0].title())
    except Exception:
        return source or "Unknown"


def is_valid_ticker(t: str | None) -> bool:
    """
    Basic sanity:
    - 2–5 uppercase letters
    - not in noisy list
    """
    if not t:
        return False
    t = t.strip().upper()
    if not (2 <= len(t) <= 5):
        return False
    if not t.isalpha():
        return False
    if t in NOISY_TICKERS:
        return False
    return True


def extract_ticker_from_title(title: str) -> str | None:
    """
    Patterns like: 'Micron Technology (MU) Falls Hard ...'
    """
    if not title:
        return None
    m = re.search(r"\(([A-Z]{1,5})\)", title)
    if m:
        return m.group(1)
    return None


def extract_ticker_from_url(link: str) -> str | None:
    """
    Yahoo / Zacks / others often have '-TICKER-' inside the slug:
      .../lyondellbasell-lyb-loses-6-dividend-...
      .../ulta-ulta-earnings-q3-2025.html
    """
    if not link:
        return None
    try:
        path = urlparse(link).path
        m = re.search(r"-([A-Z]{1,5})-", path)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


def clean_ticker(raw_ticker: str | None, title: str, link: str) -> str | None:
    """
    - Try existing ticker (if it passes validity)
    - Else try from title (...) 
    - Else try from URL slug (-TICKER-)
    - Else None
    """
    if raw_ticker:
        t = raw_ticker.strip().upper()
        if is_valid_ticker(t):
            return t

    t = extract_ticker_from_title(title)
    if is_valid_ticker(t):
        return t

    t = extract_ticker_from_url(link)
    if is_valid_ticker(t):
        return t

    return None


def normalize_category(raw_category: str | None, title: str) -> str:
    """
    Light normalization – keep your existing label if it's already good,
    else infer a rough bucket.
    """
    allowed = {"Earnings", "Fed / Macro", "Tech / AI", "M&A", "Crypto", "General"}

    if raw_category in allowed:
        return raw_category

    c = (raw_category or "").lower()
    title_lower = (title or "").lower()
    txt = f"{c} {title_lower}"

    if any(k in txt for k in ["earnings", "q1", "q2", "q3", "q4", "results", "profit", "loss"]):
        return "Earnings"
    if any(k in txt for k in ["fed", "pce", "inflation", "rates", "treasury", "yields", "gdp", "jobs"]):
        return "Fed / Macro"
    if any(k in txt for k in ["ai", "semiconductor", "chip", "nvidia", "robotaxi", "cloud", "data center"]):
        return "Tech / AI"
    if any(k in txt for k in ["merger", "acquire", "acquisition", "takeover", "ipo", "spac"]):
        return "M&A"
    if any(k in txt for k in ["bitcoin", "crypto", "cryptocurrency", "ethereum", "ether", "token"]):
        return "Crypto"

    return "General"


def clean_news_items(items: list[dict]) -> list[dict]:
    """
    Final sanitizer for /market-news:
    - dedupe by (title, link)
    - clean summary
    - normalize source
    - fix ticker
    - normalize category
    - sort newest -> oldest
    """
    seen = set()
    cleaned: list[dict] = []

    for item in items:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        if not title or not link:
            continue

        key = (title, link)
        if key in seen:
            continue
        seen.add(key)

        summary = clean_summary(item.get("summary"), title)
        source = normalize_source(item.get("source"), link)
        ticker = clean_ticker(item.get("ticker"), title, link)
        category = normalize_category(item.get("category"), title)
        pub_date = item.get("pubDate")

        cleaned.append(
            {
                "title": title,
                "summary": summary,
                "link": link,
                "pubDate": pub_date,
                "source": source,
                "ticker": ticker,
                "category": category,
            }
        )

    # pubDate is ISO string, so string sort works fine
    cleaned.sort(key=lambda x: x.get("pubDate") or "", reverse=True)
    return cleaned

@app.get("/market-news")
def market_news():
    import feedparser
    import re
    import datetime
    from urllib.parse import urlparse
    from sp500_list_optimized import extract_ticker, detect_category

    FEEDS = [
        "https://seekingalpha.com/api/sa/combined/global_news.rss",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://www.investing.com/rss/news.rss",
        "https://www.zacks.com/rss/news.xml",
        "https://finance.yahoo.com/rss/topstories",
        "https://finance.yahoo.com/topic/earnings/rss",
        "https://finance.yahoo.com/rss/tech",
        "https://finance.yahoo.com/rss/pharma",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    ]

    BLOCK_KEYWORDS = [
        "why ", "how ", "what ", "should ", "could ",
        "wife", "husband", "family", "children",
        "tv", "celebrity", "gossip",
        "crime", "murder", "scam",
        "recipe", "diet", "health",
        "war", "ukraine", "russia",
    ]

    HARD_KEYWORDS = [
        "earnings", "revenue", "profit", "loss", "guidance", "forecast",
        "ipo", "merger", "acquisition", "m&a",
        "stocks", "market", "dow", "nasdaq", "s&p", "fed",
    ]

    # -----------------------------------------
    # Local helpers (summary / source / ticker)
    # -----------------------------------------

    NOISY_TICKERS = {
        "A", "I", "U", "T", "ON", "UP", "DAY", "IT", "ARE", "HAS",
        "FAST", "COST", "TECH"
    }

    def clean_summary(summary: str | None, title: str) -> str:
        """
        - If summary is missing, '...', or too tiny -> use title.
        - Otherwise, trim to ~240 chars and add ellipsis if long.
        """
        if not summary:
            return title

        s = summary.strip()
        if not s or s == "..." or len(s) < 10:
            return title

        if len(s) > 240:
            return s[:240].rstrip() + "..."
        return s

    def get_source_from_url(url: str):
        try:
            hostname = urlparse(url).hostname or ""
            hostname = hostname.lower()
            if "cnbc" in hostname:
                return "CNBC"
            if "yahoo" in hostname:
                return "Yahoo Finance"
            if "marketwatch" in hostname:
                return "MarketWatch"
            if "zacks" in hostname:
                return "Zacks"
            if "seekingalpha" in hostname:
                return "Seeking Alpha"
            if "investing.com" in hostname:
                return "Investing.com"
            if "investors.com" in hostname:
                return "Investor's Business Daily"
            return hostname.replace("www.", "")
        except:
            return "News"

    def parse_pubdate(entry):
        pd = getattr(entry, "published", None)
        if not pd:
            return datetime.datetime.utcnow()

        try:
            return datetime.datetime(*entry.published_parsed[:6])
        except:
            try:
                return datetime.datetime.fromisoformat(pd.replace("Z", ""))
            except:
                return datetime.datetime.utcnow()

    def is_valid_ticker(t: str | None) -> bool:
        """
        Basic sanity for tickers:
        - 2–5 uppercase letters
        - alphabetic
        - not in noisy list of English words
        """
        if not t:
            return False
        t = t.strip().upper()
        if not (2 <= len(t) <= 5):
            return False
        if not t.isalpha():
            return False
        if t in NOISY_TICKERS:
            return False
        return True

    def extract_ticker_from_title_local(title: str) -> str | None:
        """
        Pattern like: 'Micron Technology (MU) Falls Hard ...'
        """
        if not title:
            return None
        m = re.search(r"\(([A-Z]{1,5})\)", title)
        if m:
            return m.group(1)
        return None

    def extract_ticker_from_url_local(link: str) -> str | None:
        """
        Patterns like: .../lyondellbasell-lyb-loses-6-dividend-...
        """
        if not link:
            return None
        try:
            path = urlparse(link).path or ""
            m = re.search(r"-([A-Z]{1,5})-", path)
            if m:
                return m.group(1)
        except Exception:
            return None
        return None

    def clean_ticker(raw_ticker: str | None, title: str, link: str) -> str | None:
        """
        - Use your existing extract_ticker output if valid.
        - Else try from title '(XXXX)'.
        - Else try from URL '-XXXX-'.
        - Else None.
        """
        if is_valid_ticker(raw_ticker):
            return raw_ticker.strip().upper()

        t = extract_ticker_from_title_local(title.upper())
        if is_valid_ticker(t):
            return t

        t = extract_ticker_from_url_local(link.upper())
        if is_valid_ticker(t):
            return t

        return None

    # -----------------------------------------
    # Main RSS aggregation
    # -----------------------------------------

    all_news = []

    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:25]:
                title = getattr(e, "title", "") or ""
                summary_raw = getattr(e, "summary", "") or ""

                if any(b in title.lower() for b in BLOCK_KEYWORDS):
                    continue

                combined = (title + " " + summary_raw).lower()

                allowed = (
                    any(k in combined for k in HARD_KEYWORDS)
                    or extract_ticker(combined.upper())
                )
                if not allowed:
                    continue

                pub_date = parse_pubdate(e)
                link_val = getattr(e, "link", "") or ""

                raw_ticker = extract_ticker((title + " " + summary_raw).upper())
                ticker = clean_ticker(raw_ticker, title, link_val)

                category = detect_category((title + summary_raw).upper())
                source = get_source_from_url(link_val)
                summary = clean_summary(summary_raw, title)

                all_news.append({
                    "title": title.strip(),
                    "summary": summary,
                    "link": link_val,
                    "pubDate": pub_date.isoformat(),
                    "source": source,
                    "ticker": ticker,
                    "category": category,
                })

        except Exception as ex:
            print("RSS error:", ex)

    # NORMAL DEDUPE (not aggressive) – keep your logic
    seen = set()
    result = []
    for n in all_news:
        key = n["title"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(n)

    # SORT NEWEST FIRST
    result.sort(key=lambda x: x["pubDate"], reverse=True)

    return {"data": result[:80]}


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
        candles = fetch_daily_candles(symbol)
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

    total_value: float = 0.0
    total_gain: float = 0.0
    today_gain: float = 0.0

    positions: List[AstraPosition] = []


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
        candles = fetch_daily_candles(symbol)
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
# ASTRA CHAT — Portfolio AI Analyst (BullBrain v2 + Grok + Sentiment)
# ---------------------------------------------------------------
@app.post("/astra-chat")
def astra_chat(req: AstraChatRequest):
    """
    Astra – Artificial Stock Trading & Risk Analyst for BullSignalsAI.

    Uses:
    - Live candles from Polygon
    - BullBrain v2 (48 features) per symbol
    - Allocation %, gain %, position value
    - Grok (XAI) for natural language answers
    - Lightweight price/volume-based sentiment

    Works for both:
    - Predefined questions (question_id)
    - Custom free-form questions (question)
    """

    # 1) Basic validation
    if not req.positions or req.total_value <= 0:
        return {
            "answer": (
                "I need at least one holding with a non-zero portfolio value "
                "to analyze. Please add positions to your portfolio and try again."
            ),
            "used_llm": False,
            "analysis": {},
        }

    # 2) Resolve question (chips or custom text)
    q_map = {
        "overview": "Give a concise overview of how my portfolio is doing.",
        "risk_exposure": "Explain my overall risk exposure and concentration.",
        "overweight": "Which stocks are overweight or underweight, and what should I consider doing?",
        "worst": "Which positions need my urgent attention and why?",
        "ai_suggestions": "Give me AI-driven suggestions on how to optimize this portfolio.",
    }

    base_question = (req.question or "").strip()
    if not base_question and req.question_id:
        base_question = q_map.get(
            req.question_id,
            "Give a clear, practical analysis of this portfolio.",
        )

    if not base_question:
        base_question = "Give a clear, practical analysis of this portfolio."

    question_id = req.question_id or ""

    # 3) Sort positions by allocation (focus on biggest ones first)
    positions_sorted = sorted(
        req.positions, key=lambda p: p.allocation_pct, reverse=True
    )

    # Limit how many symbols we send to BullBrain (performance)
    max_symbols_for_model = 10
    focus_positions = positions_sorted[:max_symbols_for_model]

    per_symbol_analysis = []
    bullbrain_failures = []

    # 4) Per-symbol BullBrain v2 analysis
    for pos in focus_positions:
        symbol = pos.symbol.upper()

        try:
            candles = fetch_daily_candles(symbol)
            if not candles:
                bullbrain_failures.append(symbol)
                continue

            features_vec, feature_dict, last_close = compute_bullbrain_features(
                candles
            )
            if features_vec is None:
                bullbrain_failures.append(symbol)
                continue

            out = bullbrain_infer(features_vec)
            prob_up = float(out.get("probability_up") or out.get("raw_output") or 0.5)
            signal = out.get("signal") or "NEUTRAL"

            vol = feature_dict.get("volatility_5d", 0.02)
            # Approx expected move (5d-ish) on a -1..+1 scale
            expected_move = vol * (prob_up * 2 - 1)
            # Convert to %
            expected_move_pct = round(expected_move * 100, 2)

            # Confidence bucket
            if prob_up >= 0.66:
                confidence = "High"
            elif prob_up >= 0.55:
                confidence = "Moderate"
            else:
                confidence = "Low"

            # Risk bucket from volatility
            if vol < 0.015:
                risk = "Low"
            elif vol < 0.035:
                risk = "Medium"
            else:
                risk = "High"

            # Simple pattern from SMAs
            sma5 = feature_dict.get("sma5", 0)
            sma20 = feature_dict.get("sma20", 0)
            if sma5 > sma20:
                pattern = "Short-term momentum"
            elif sma5 < sma20:
                pattern = "Reversal risk"
            else:
                pattern = "Sideways consolidation"

            # Over/under-weight (vs equal-weight baseline)
            equal_weight = 100.0 / max(len(req.positions), 1)
            allocation = pos.allocation_pct
            if allocation > equal_weight * 1.8:
                weight_flag = "Strongly Overweight"
            elif allocation > equal_weight * 1.2:
                weight_flag = "Overweight"
            elif allocation < equal_weight * 0.5:
                weight_flag = "Underweight"
            else:
                weight_flag = "Balanced"

            per_symbol_analysis.append(
                {
                    "symbol": symbol,
                    "allocation_pct": round(pos.allocation_pct, 2),
                    "gain_pct": round(pos.gain_pct, 2),
                    "unrealized_gain": round(pos.gain, 2),
                    "today_pl": round(pos.today, 2),
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "price": pos.price,
                    "bullbrain": {
                        "signal": signal,
                        "prob_up": round(prob_up * 100, 1),  # %
                        "expected_move_pct": expected_move_pct,
                        "risk": risk,
                        "confidence": confidence,
                        "pattern": pattern,
                    },
                    "weight_flag": weight_flag,
                }
            )

        except Exception as e:
            print(f"Astra BullBrain error for {symbol}:", e)
            bullbrain_failures.append(symbol)
            continue

    if not per_symbol_analysis:
        # If BullBrain failed for everything, we at least return basic metrics.
        return {
            "answer": (
                "I could not compute AI signals for your holdings at this time. "
                "Please try again later. Your portfolio still has a positive value, "
                "but BullBrain analysis is temporarily unavailable."
            ),
            "used_llm": False,
            "analysis": {
                "total_value": req.total_value,
                "total_gain": req.total_gain,
                "today_gain": req.today_gain,
                "positions": [p.symbol for p in req.positions],
                "bullbrain_unavailable": bullbrain_failures,
            },
        }

    # 5) Portfolio-level aggregates

    # Overall return %
    overall_return_pct = (
        (req.total_gain / max(req.total_value - req.total_gain, 1)) * 100.0
        if req.total_value > 0
        else 0.0
    )

    # Weighted average bullish probability (by allocation)
    total_alloc_for_weight = sum(p["allocation_pct"] for p in per_symbol_analysis) or 1
    weighted_prob_up = (
        sum(
            p["allocation_pct"] * p["bullbrain"]["prob_up"]
            for p in per_symbol_analysis
        )
        / total_alloc_for_weight
    )

    # Top contributors / laggards
    sorted_by_gain = sorted(
        per_symbol_analysis, key=lambda p: p["unrealized_gain"], reverse=True
    )
    top_gainer = sorted_by_gain[0] if sorted_by_gain else None
    worst_loser = sorted_by_gain[-1] if sorted_by_gain else None

    # Risk exposure heuristic: max allocation and high-vol names
    max_alloc = max(p["allocation_pct"] for p in per_symbol_analysis)
    high_risk_names = [
        p
        for p in per_symbol_analysis
        if p["bullbrain"]["risk"] == "High" and p["allocation_pct"] >= 10
    ]

    if max_alloc > 40:
        portfolio_risk_label = "High (concentrated in a single position)"
    elif max_alloc > 25:
        portfolio_risk_label = "Moderate to High (few large positions)"
    else:
        portfolio_risk_label = "Balanced to Moderate"

    # 6) Hybrid sentiment selection
    portfolio_symbols = [p["symbol"] for p in per_symbol_analysis]
    sentiment_targets: List[str] = []

    # a) For predefined questions, pick a small, meaningful set of symbols
    if question_id == "overview":
        # focus on largest holding + top gainer + worst loser
        if per_symbol_analysis:
            sentiment_targets.append(per_symbol_analysis[0]["symbol"])
        if top_gainer:
            sentiment_targets.append(top_gainer["symbol"])
        if worst_loser:
            sentiment_targets.append(worst_loser["symbol"])

    elif question_id == "risk_exposure":
        # high risk concentration names
        sentiment_targets.extend([p["symbol"] for p in high_risk_names])

    elif question_id == "overweight":
        overweight = [
            p for p in per_symbol_analysis if "Overweight" in p["weight_flag"]
        ]
        sentiment_targets.extend([p["symbol"] for p in overweight[:4]])

    elif question_id == "worst":
        if worst_loser:
            sentiment_targets.append(worst_loser["symbol"])

    elif question_id == "ai_suggestions":
        overweight = [
            p for p in per_symbol_analysis if "Overweight" in p["weight_flag"]
        ]
        if top_gainer:
            sentiment_targets.append(top_gainer["symbol"])
        if worst_loser:
            sentiment_targets.append(worst_loser["symbol"])
        sentiment_targets.extend([p["symbol"] for p in overweight[:3]])

    # b) For custom questions, detect mentioned tickers
    else:
        upper_q = base_question.upper()
        mentioned = []
        for sym in portfolio_symbols:
            if re.search(rf"\b{re.escape(sym)}\b", upper_q):
                mentioned.append(sym)

        if mentioned:
            sentiment_targets.extend(mentioned[:5])

    # remove duplicates while preserving order
    seen = set()
    uniq_targets = []
    for s in sentiment_targets:
        if s and s not in seen:
            seen.add(s)
            uniq_targets.append(s)

    sentiment_map: Dict[str, Any] = {}
    for sym in uniq_targets:
        sentiment_map[sym] = astra_symbol_sentiment(sym)

    # 7) Build analysis object
    analysis_obj = {
        "question": base_question,
        "question_id": question_id,
        "total_value": round(req.total_value, 2),
        "total_gain": round(req.total_gain, 2),
        "today_gain": round(req.today_gain, 2),
        "overall_return_pct": round(overall_return_pct, 2),
        "weighted_bull_prob_pct": round(weighted_prob_up, 1),
        "portfolio_risk_label": portfolio_risk_label,
        "top_gainer": top_gainer,
        "worst_loser": worst_loser,
        "high_risk_concentrations": high_risk_names,
        "per_symbol": per_symbol_analysis,
        "bullbrain_failed_symbols": bullbrain_failures,
        "sentiment": sentiment_map,
    }

    # 8) Build LLM prompt for Grok (Astra)
    system_prompt = (
        "You are Astra, an expert AI portfolio analyst inside a mobile app "
        "called BullSignalsAI. The user is a retail investor.\n"
        "You must be clear, calm, and practical.\n"
        "Use real numbers from the JSON (values, percentages, tickers, shares).\n"
        "Do not invent numbers. If a value is missing, simply say it is not available.\n"
        "Do NOT use markdown, bullet symbols, asterisks, or headings.\n"
        "Write in plain text with short paragraphs.\n"
        "Tailor your answer to the specific question. Do not repeat the same template every time.\n"
        "Avoid financial jargon. Do not say buy or sell; instead say things like "
        "consider reducing exposure or consider increasing exposure.\n"
    )

    user_prompt = (
        f"User question: {base_question}\n\n"
        "Here is their portfolio and AI analysis as JSON. "
        "This includes BullBrain v2 outputs and a small sentiment block for some symbols.\n\n"
        f"{json.dumps(analysis_obj, indent=2)}\n\n"
        "Now answer the user's question using this data only.\n"
        "Use the following style:\n"
        "- If the question is about the whole portfolio (overview, risk, suggestions), "
        "talk about total value, total gain or loss, overall return percent, "
        "and mention 2–3 key tickers with their allocation and gain or loss.\n"
        "- If the question is about a specific stock, focus on that ticker's allocation, "
        "gain or loss, BullBrain signal, probability_up and expected_move_pct, "
        "plus the sentiment summary for that ticker if available.\n"
        "- For overweight or underweight questions, mention which tickers are flagged as overweight "
        "or underweight and what the user could consider doing.\n"
        "- Keep the answer concise. Usually 1 to 3 short paragraphs are enough.\n"
        "- Always close with one short sentence reminding the user that this is AI-driven insight, "
        "not personal financial advice.\n"
    )

    llm_answer = astra_llm_answer(system_prompt, user_prompt)
    used_llm = llm_answer is not None

    # 9) Fallback answer if Grok fails
    if not llm_answer:
        lines = []
        lines.append(
            f"Your portfolio is currently valued at about ${req.total_value:,.2f} "
            f"with an overall return of approximately {overall_return_pct:+.2f} percent."
        )
        lines.append(
            f"Across your largest positions, BullBrain sees an average bullish probability "
            f"of around {weighted_prob_up:.1f} percent."
        )
        lines.append(f"Risk level is classified as {portfolio_risk_label}.")

        if top_gainer:
            lines.append(
                f"Your main positive contributor right now is {top_gainer['symbol']}, "
                f"with an unrealized gain of about ${top_gainer['unrealized_gain']:,.2f} "
                f"and a change of roughly {top_gainer['gain_pct']:+.1f} percent."
            )
        if worst_loser:
            lines.append(
                f"The biggest drag is {worst_loser['symbol']}, "
                f"at about ${worst_loser['unrealized_gain']:,.2f} "
                f"and {worst_loser['gain_pct']:+.1f} percent."
            )

        if sentiment_map:
            # Use first sentiment as example
            s_sym, s_val = next(iter(sentiment_map.items()))
            if s_val.get("summary"):
                lines.append(f"For example, {s_sym}: {s_val['summary']}")

        lines.append(
            "Treat this as AI-supported analysis to guide your thinking, not as personal financial advice."
        )

        llm_answer = " ".join(lines)

    # 10) Return structured response
    return {
        "answer": llm_answer,
        "used_llm": used_llm,
        "analysis": analysis_obj,
    }



# ---------------------------------------------------------
# Helpers for Market Pulse
# ---------------------------------------------------------

def _clean_text_py(s: str) -> str:
    if not s:
        return ""
    # Basic cleaning similar to frontend
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _analyze_headline_sentiment_py(headlines):
    bullish_words = [
        "gain", "gains", "rise", "rises", "soar", "soars",
        "beat", "beats", "growth", "surge", "surges",
        "optimism", "rebound", "rebounds", "strong",
        "rally", "record high", "expands", "up", "advance",
        "higher", "jumps", "spikes"
    ]

    bearish_words = [
        "drop", "drops", "fall", "falls", "slip", "plunge", "plunges",
        "loss", "losses", "slowdown", "decline", "declines",
        "cut", "cuts", "layoff", "layoffs", "weak", "selloff",
        "tumbles", "down", "pressure", "warning", "downgrade",
        "guidance cut"
    ]

    results = []
    for h in headlines:
        title = _clean_text_py(h or "")
        lower = title.lower()

        tag = "⚖️"  # default neutral
        if any(w in lower for w in bullish_words):
            tag = "📈"
        if any(w in lower for w in bearish_words):
            tag = "📉"

        results.append({"title": title, "tag": tag})

    return results


FALLBACK_SENTENCES = {
    "bullish": [
        "Markets show improving breadth across key sectors.",
        "Investor risk appetite firms during the early session.",
        "Equities strengthen as buying momentum builds.",
        "Positive flows support upside stability.",
        "Growth stocks continue their leadership trend.",
    ],
    "neutral": [
        "Markets remain steady as traders await key catalysts.",
        "Equities trade sideways amid balanced sentiment.",
        "Mixed sector rotation keeps indexes stable.",
        "Traders monitor macro signals for direction.",
        "Volatility holds near average levels.",
    ],
    "bearish": [
        "Market participants show caution amid uncertainty.",
        "Risk-off flows build as volatility edges higher.",
        "Selling pressure emerges in selective sectors.",
        "Equities pull back as momentum cools.",
        "Weakness appears across multiple asset groups.",
    ],
}


def _ensure_five(items, key):
    arr = list(items)
    if len(arr) >= 5:
        return arr[:5]
    needed = 5 - len(arr)
    return arr + FALLBACK_SENTENCES[key][:needed]


def _compute_risk_from_stats(sp500_change: float, vix: float, fg_value: int) -> str:
    # Same logic as before
    if vix < 15 and fg_value > 60:
        return "Low Risk"
    if vix > 20 or fg_value < 30:
        return "High Risk"
    return "Moderate Risk"


def _get_market_overview_quick():
    """
    Lightweight market overview:
    - SPY as proxy for S&P 500
    - VIX index
    - Synthetic Fear & Greed from VIX
    """
    overview = {
        "sp500_change": 0.0,
        "vix": 15.0,
        "fearGreed": {"value": 50, "label": "Neutral"},
        "risk_level": "Moderate Risk",
    }

    try:
        # --- S&P proxy via SPY ETF ---
        spy_quote = backend_fetch_quote("SPY") or {}
        change_pct = spy_quote.get("changePct")
        if change_pct is None:
            price = spy_quote.get("price")
            prev = spy_quote.get("prevClose")
            if price and prev and prev != 0:
                change_pct = ((price - prev) / prev) * 100.0
            else:
                change_pct = 0.0
        overview["sp500_change"] = float(change_pct)

    except Exception as e:
        print("SPY quote error:", e)

    try:
        # --- VIX index ---
        vix_quote = backend_fetch_quote("^VIX") or backend_fetch_quote("VIX") or {}
        vix_price = vix_quote.get("price") or vix_quote.get("close") or 15.0
        overview["vix"] = float(vix_price)
    except Exception as e:
        print("VIX quote error:", e)

    # --- Synthetic Fear & Greed from VIX ---
    vix_val = overview["vix"]
    fg_value = 50
    fg_label = "Neutral"

    if vix_val < 14:
        fg_value = 70
        fg_label = "Greed"
    elif vix_val < 18:
        fg_value = 55
        fg_label = "Slight Greed"
    elif vix_val < 22:
        fg_value = 45
        fg_label = "Slight Fear"
    elif vix_val < 28:
        fg_value = 35
        fg_label = "Fear"
    else:
        fg_value = 25
        fg_label = "Extreme Fear"

    overview["fearGreed"] = {"value": fg_value, "label": fg_label}
    overview["risk_level"] = _compute_risk_from_stats(
        overview["sp500_change"], vix_val, fg_value
    )

    return overview

@app.get("/market-pulse")
def market_pulse():
    import pytz
    eastern = pytz.timezone("America/New_York")
    utc = pytz.utc

    try:
        # ----------------------------------------------------
        # 1) ALWAYS FETCH FRESH NEWS
        # ----------------------------------------------------
        news_resp = market_news()
        raw_news = news_resp.get("data", []) if isinstance(news_resp, dict) else []

        if not isinstance(raw_news, list):
            raw_news = []

        cleaned = []

        # ----------------------------------------------------
        # 2) Convert UTC → Eastern time
        # ----------------------------------------------------
        for n in raw_news:
            try:
                # Parse published date (UTC ISO)
                dt_utc = datetime.datetime.fromisoformat(
                    n["pubDate"].replace("Z", "")
                ).replace(tzinfo=utc)

                # Convert → Eastern
                dt_et = dt_utc.astimezone(eastern)

                n["pubDateET"] = dt_et.isoformat()
                n["pubDateObj"] = dt_et

                cleaned.append(n)

            except:
                continue

        # ----------------------------------------------------
        # 3) SORT LATEST → FIRST using ET
        # ----------------------------------------------------
        cleaned.sort(key=lambda x: x["pubDateObj"], reverse=True)

        # ----------------------------------------------------
        # 4) SENTIMENT — from latest 80 items
        # ----------------------------------------------------
        titles = [n.get("title", "") for n in cleaned[:80] if n.get("title")]

        analyzed = _analyze_headline_sentiment_py(titles)

        bullish_raw = [a["title"] for a in analyzed if a["tag"] == "📈"]
        bearish_raw = [a["title"] for a in analyzed if a["tag"] == "📉"]
        neutral_raw = [a["title"] for a in analyzed if a["tag"] == "⚖️"]

        # Fallback keyword method (adds variation)
        for n in cleaned[:80]:
            t = n.get("title", "").lower()
            if not t:
                continue

            if any(w in t for w in ["rises", "beats", "up", "strong", "record"]):
                bullish_raw.append(n["title"])
            elif any(w in t for w in ["falls", "drops", "down", "weak", "cuts"]):
                bearish_raw.append(n["title"])
            else:
                neutral_raw.append(n["title"])

        # Unique
        bullish_raw = list(set(bullish_raw))
        bearish_raw = list(set(bearish_raw))
        neutral_raw = list(set(neutral_raw))

        # Top 5 each
        bullish = bullish_raw[:5]
        neutral = neutral_raw[:5]
        bearish = bearish_raw[:5]

        # Numeric summary
        highlights_numeric = {
            "bull": len(bullish_raw),
            "bear": len(bearish_raw),
            "neutral": len(neutral_raw),
        }

        # ----------------------------------------------------
        # 5) MARKET OVERVIEW
        # ----------------------------------------------------
        overview = _get_market_overview_quick()

        # ----------------------------------------------------
        # 6) GROUP NEWS BY DATE (ET)
        # ----------------------------------------------------
        grouped = {"today": [], "yesterday": [], "week": [], "older": []}

        now_et = datetime.datetime.now(eastern)
        today_et = now_et.date()
        yesterday_et = today_et - datetime.timedelta(days=1)
        week_ago_et = today_et - datetime.timedelta(days=7)

        for n in cleaned:
            d = n["pubDateObj"].date()

            if d == today_et:
                grouped["today"].append(n)
            elif d == yesterday_et:
                grouped["yesterday"].append(n)
            elif d >= week_ago_et:
                grouped["week"].append(n)
            else:
                grouped["older"].append(n)

        # ----------------------------------------------------
        # 7) Sort grouped lists by ET timestamp
        # ----------------------------------------------------
        for key in grouped:
            grouped[key].sort(key=lambda x: x["pubDateObj"], reverse=True)

        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------
        return {
            "market_overview": overview,
            "highlights_numeric": highlights_numeric,
            "highlights_grouped": {
                "bullish": bullish,
                "neutral": neutral,
                "bearish": bearish,
            },
            "news_grouped": grouped,
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print("market_pulse error:", e)
        fb = _get_market_overview_quick()
        return {
            "market_overview": fb,
            "highlights_numeric": {"bull": 0, "bear": 0, "neutral": 0},
            "highlights_grouped": {
                "bullish": [],
                "neutral": [],
                "bearish": [],
            },
            "news_grouped": {"today": [], "yesterday": [], "week": [], "older": []},
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

def bullbrain_infer_single(symbol: str):
    try:
        candles = fetch_daily_candles(symbol)
        if not candles:
            return None

        features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
        return bullbrain_infer(features_vec)

    except Exception as e:
        print("bullbrain_infer_single error:", symbol, e)
        return None



@app.get("/debug-bullbrain/{symbol}")
def debug_bullbrain(symbol: str):
    try:
        sym = symbol.upper()
        candles = fetch_daily_candles(sym)

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
# Read market AI cache (global Firestore)
# ---------------------------------------------------------
def read_market_cache(doc_id):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # safe init
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        doc_ref = db.collection("bullsignals_ai").document(doc_id)
        snap = doc_ref.get()

        if not snap.exists:
            return None

        data = snap.to_dict()
        updated = data.get("updated_at")
        if not updated:
            return None

        # Check cache age
        age_minutes = (datetime.datetime.utcnow() -
                       datetime.datetime.fromisoformat(updated.replace("Z",""))
                       ).total_seconds() / 60

        if age_minutes <= 5:
            print(f"💾 Using cached {doc_id} ({age_minutes:.1f} min old)")
            return data

        print(f"⏳ Cache expired ({age_minutes:.1f} min). Will recompute.")
        return None

    except Exception as e:
        print("read_market_cache error:", e)
        return None



# ---------------------------------------------------------
# /market-hotlist — Top 15 strongest bullish tickers
# ---------------------------------------------------------
@app.get("/market-hotlist")
def market_hotlist():
    from symbols_clean import REAL_TICKERS
    import datetime

    # 1) Try returning cached first
    cache = read_market_cache("market_hotlist")
    if cache:
        return {
            "count": cache.get("count", 0),
            "hotlist": cache.get("hotlist", [])
        }

    results = []

    # 2) Full scan (ALL SP500 tickers)
    for sym in REAL_TICKERS:
        try:
            infer = bullbrain_infer_single(sym)
            if not infer:
                continue

            prob_up = float(infer.get("probability_up", 0))
            prob_down = float(infer.get("probability_down", 0))

            results.append({
                "symbol": sym,
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "signal": infer.get("signal"),
                "confidence": infer.get("confidence"),
            })

        except Exception as e:
            print(f"hotlist error {sym}:", e)

    # 3) Sort strongest bullish
    results.sort(key=lambda x: x["prob_up"], reverse=True)
    top15 = results[:15]

    # 4) Save to Firestore
    try:
        cache_data = {
            "count": len(top15),
            "hotlist": top15,
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        save_to_firestore_market_cache("market_hotlist", cache_data)
    except Exception as e:
        print("🔥 Cache save failed (hotlist):", e)

    return {"count": len(top15), "hotlist": top15}



# ---------------------------------------------------------
# /market-bearwatch — Top 15 strongest bearish tickers
# ---------------------------------------------------------
@app.get("/market-bearwatch")
def market_bearwatch():
    from symbols_clean import REAL_TICKERS
    import datetime

    # 1) Use cached version if available
    cache = read_market_cache("market_bearwatch")
    if cache:
        return {
            "count": cache.get("count", 0),
            "bearwatch": cache.get("bearwatch", [])
        }

    results = []

    # 2) FULL SCAN of SP500
    for sym in REAL_TICKERS:
        try:
            infer = bullbrain_infer_single(sym)
            if not infer:
                continue

            prob_up = float(infer.get("probability_up", 0))
            prob_down = float(infer.get("probability_down", 0))

            results.append({
                "symbol": sym,
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "signal": infer.get("signal"),
                "confidence": infer.get("confidence"),
            })

        except Exception as e:
            print(f"bearwatch error {sym}:", e)

    # 3) Sort strongest bearish
    results.sort(key=lambda x: x["prob_down"], reverse=True)
    top15 = results[:15]

    # 4) Save to Firestore
    try:
        cache_data = {
            "count": len(top15),
            "bearwatch": top15,
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        save_to_firestore_market_cache("market_bearwatch", cache_data)
    except Exception as e:
        print("🔥 Cache save failed (bearwatch):", e)

    return {"count": len(top15), "bearwatch": top15}
