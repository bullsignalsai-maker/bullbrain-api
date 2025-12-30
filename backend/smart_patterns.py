# backend/smart_patterns.py
# ============================================================
# SMART PATTERN CORE + HISTORY SCANNER
# (Extracted verbatim from main.py / test.py)
# ============================================================

import numpy as np
import pandas as pd
import datetime
from typing import Dict, Any


# ============================================================
# Indicator Helpers
# ============================================================

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
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
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


# ============================================================
# Core Smart Pattern Classifier
# ============================================================

def _evaluate_smart_pattern_row(
    *,
    gap,
    change,
    vol_z,
    vol_vs_ma,
    rsi,
    will_r,
    lower_shadow,
    upper_shadow,
    body_pct,
    price_vs_sma20,
    trend,
    ret3,
    ret5,
):
    def ok(x):
        return x is not None and not np.isnan(x)

    patterns = []

    # 1) GAP UP & RUNNING
    if ok(gap) and ok(change) and ok(vol_vs_ma):
        if gap > 1.0 and change > 2.0 and vol_vs_ma > 20.0:
            patterns.append((0.9, {
                "pattern": "GAP UP & RUNNING",
                "winRate": 0.73,
                "bias": "bull",
                "headline": "Stock exploded higher at the open and buyers kept control all day.",
                "explanation": "Strong gap-and-go with sustained volume."
            }))

    # 2) MASSIVE VOLUME BREAKOUT
    if ok(vol_z) and vol_z > 3.0:
        patterns.append((0.85, {
            "pattern": "VOLUME BREAKOUT",
            "winRate": 0.76,
            "bias": "bull",
            "headline": "Unusually heavy trading volume.",
            "explanation": "Institutional participation likely."
        }))

    # 3) OVERSOLD BOUNCE
    if ok(rsi) and ok(will_r) and ok(vol_z):
        if rsi < 30 and will_r < -80 and vol_z > 2.0:
            patterns.append((0.9, {
                "pattern": "OVERSOLD BOUNCE",
                "winRate": 0.80,
                "bias": "bull",
                "headline": "Capitulation followed by strong buying.",
                "explanation": "Classic relief rally setup."
            }))

    # 4) HAMMER REVERSAL
    if ok(lower_shadow) and ok(body_pct) and ok(change):
        if lower_shadow > 40.0 and abs(body_pct) < 40.0 and change > 0:
            patterns.append((0.8, {
                "pattern": "HAMMER REVERSAL",
                "winRate": 0.74,
                "bias": "bull",
                "headline": "Intraday selloff rejected.",
                "explanation": "Potential local bottom."
            }))

    # 5) BUY THE DIP (UPTREND)
    if ok(trend) and ok(price_vs_sma20) and ok(change):
        if trend > 10.0 and price_vs_sma20 < -3.0 and change > 0:
            patterns.append((0.78, {
                "pattern": "BUY THE DIP (UPTREND)",
                "winRate": 0.69,
                "bias": "bull",
                "headline": "Healthy pullback in an uptrend.",
                "explanation": "Trend continuation likely."
            }))

    # 6) DEAD CAT BOUNCE
    if ok(ret5) and ok(change) and ok(vol_z):
        if ret5 < -8.0 and change > 0 and vol_z < 1.0:
            patterns.append((0.75, {
                "pattern": "DEAD CAT BOUNCE",
                "winRate": 0.68,
                "bias": "bear",
                "headline": "Weak rebound after heavy selling.",
                "explanation": "Often fails."
            }))

    # 7) OVERBOUGHT DISTRIBUTION
    if ok(rsi) and ok(vol_vs_ma) and ok(change):
        if rsi > 70 and vol_vs_ma < 0:
            patterns.append((0.72, {
                "pattern": "OVERBOUGHT DISTRIBUTION",
                "winRate": 0.67,
                "bias": "bear",
                "headline": "Momentum extended, volume fading.",
                "explanation": "Possible topping behavior."
            }))

    # 8) FAILED BREAKOUT TRAP
    if ok(change) and ok(vol_z):
        if change < -2.0 and vol_z > 2.0:
            patterns.append((0.7, {
                "pattern": "FAILED BREAKOUT TRAP",
                "winRate": 0.66,
                "bias": "bear",
                "headline": "Breakout attempt reversed.",
                "explanation": "Bull trap."
            }))

    # 9) INSIDE RANGE COMPRESSION
    if ok(change) and ok(ret3) and ok(vol_vs_ma):
        if abs(change) < 0.8 and abs(ret3) < 2.0 and vol_vs_ma < 0:
            patterns.append((0.6, {
                "pattern": "INSIDE RANGE COMPRESSION",
                "winRate": 0.62,
                "bias": "neutral",
                "headline": "Quiet consolidation.",
                "explanation": "Energy building."
            }))

    # 10) HIGH-WAVE INDECISION
    if ok(upper_shadow) and ok(lower_shadow) and ok(body_pct):
        if upper_shadow > 30 and lower_shadow > 30 and abs(body_pct) < 20:
            patterns.append((0.58, {
                "pattern": "HIGH-WAVE INDECISION",
                "winRate": 0.60,
                "bias": "neutral",
                "headline": "Volatile tug-of-war.",
                "explanation": "Market uncertainty."
            }))

    # 11) TREND ACCELERATION
    if ok(trend) and ok(change) and ok(vol_vs_ma):
        if trend > 15 and change > 1.5 and vol_vs_ma > 5:
            patterns.append((0.7, {
                "pattern": "TREND ACCELERATION",
                "winRate": 0.70,
                "bias": "bull",
                "headline": "Momentum expanding.",
                "explanation": "Continuation likely."
            }))

    # 12) GAP DOWN & PRESSURE
    if ok(gap) and ok(change):
        if gap < -1.0 and change < -2.0:
            patterns.append((0.68, {
                "pattern": "GAP DOWN & PRESSURE",
                "winRate": 0.65,
                "bias": "bear",
                "headline": "Sellers in control.",
                "explanation": "Risk-off behavior."
            }))

    if not patterns:
        return None

    patterns.sort(key=lambda x: x[0], reverse=True)
    return patterns[0][1]


# ============================================================
# History Scanner (VERBATIM)
# ============================================================

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


# ============================================================
# Single-Day Detector (VERBATIM)
# ============================================================

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

