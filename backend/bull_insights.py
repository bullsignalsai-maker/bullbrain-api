# backend/bull_insights.py
# ------------------------------------------------------
# BullBrain Insights Generator (Backend)
# Goal:
#   - Produce long, plain-English explanations (2+ sentences)
#   - Make narratives feel unique per symbol
#   - Keep the same function signature + output keys
#   - Avoid "Trend:" / "Momentum:" prefixes in UI text
#   - Pattern is context, not a signal
# ------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Optional, List
import math
import hashlib


# ======================================================
# Utilities
# ======================================================

def _num(v: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
            return float(v)
    except Exception:
        pass
    return fallback


def _seeded_pick(options: List[str], seed_key: str) -> str:
    """Deterministic pick per seed (symbol + time bucket) to keep text stable."""
    if not options:
        return ""
    h = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(options)
    return options[idx]


def _conf_tier(conf: float) -> str:
    if conf >= 80:
        return "high"
    if conf >= 65:
        return "moderate"
    return "low"


def _safe_pct(v: Optional[float], digits: int = 0) -> Optional[str]:
    if v is None:
        return None
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except Exception:
        return None


def _clamp_sentence(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


# ======================================================
# Long, Plain-English Indicator Narratives
# ======================================================

def _build_trend_text(
    *,
    sma5: Optional[float],
    sma20: Optional[float],
    seed_key: str,
) -> str:
    if sma5 is None or sma20 is None:
        return _clamp_sentence(
            _seeded_pick(
                [
                    "Trend information is limited right now because the moving-average inputs were not available",
                    "The trend picture is incomplete in this scan, so it’s better to rely on confirmation from the next update",
                ],
                seed_key + ":trend:missing",
            )
            + " This is not a problem, but it does reduce how strongly we can describe direction."
        )

    if sma5 > sma20:
        a = _seeded_pick(
            [
                "Price is holding above its short-term baseline, which generally supports a bullish lean",
                "The short-term trend is constructive because the faster average is above the slower average",
                "Recent price action is behaving better than the longer short-term mean, which is usually a positive sign",
            ],
            seed_key + ":trend:up",
        )
        b = _seeded_pick(
            [
                "That said, trend alone is not a trade plan; the best setups are when momentum and participation also confirm the move",
                "This is a supportive backdrop, but the model still waits for alignment across momentum and volume before calling a clean edge",
            ],
            seed_key + ":trend:up2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    if sma5 < sma20:
        a = _seeded_pick(
            [
                "Price is trading below its short-term averages, which typically signals weakness or unfinished consolidation",
                "The short-term trend is under pressure because the faster average is below the slower average",
                "Recent price action is sitting under a key baseline, which often means buyers need to prove themselves again",
            ],
            seed_key + ":trend:down",
        )
        b = _seeded_pick(
            [
                "In this environment, upside attempts can fail quickly unless momentum improves and volume supports the move",
                "This does not guarantee further downside, but it does mean bullish setups usually need clearer confirmation to be reliable",
            ],
            seed_key + ":trend:down2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    return _clamp_sentence(
        "Price is hovering near its short-term averages, which is common during consolidation phases."
    ) + " " + _clamp_sentence(
        "When averages are tightly aligned, the market often needs a catalyst or stronger momentum signal before a clean direction emerges."
    )


def _build_momentum_text(
    *,
    rsi14: Optional[float],
    macd: Optional[float],
    macd_signal: Optional[float],
    seed_key: str,
) -> str:
    parts: List[str] = []

    if rsi14 is not None:
        if rsi14 < 30:
            parts.append(
                _seeded_pick(
                    [
                        "Momentum is stretched to the downside, which can lead to a bounce, but oversold conditions alone are not a buy signal",
                        "Momentum looks oversold, and that sometimes precedes a rebound; the safer approach is to wait for confirmation instead of guessing the bottom",
                    ],
                    seed_key + ":mom:oversold",
                )
            )
        elif rsi14 > 70:
            parts.append(
                _seeded_pick(
                    [
                        "Momentum is extended after a strong run, and overbought readings often raise the risk of pullbacks or choppy digestion",
                        "Momentum appears overheated, which does not mean an immediate drop, but it does mean upside can become harder to sustain without new demand",
                    ],
                    seed_key + ":mom:overbought",
                )
            )
        else:
            parts.append(
                _seeded_pick(
                    [
                        "Momentum is in a normal range, which usually means the next direction depends more on trend structure and participation",
                        "Momentum is not extreme, so the market may be waiting for a clearer catalyst or stronger directional pressure",
                    ],
                    seed_key + ":mom:normal",
                )
            )

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            parts.append(
                _seeded_pick(
                    [
                        "The MACD structure is improving, which suggests bullish pressure is building, even if it is not yet decisive",
                        "MACD is leaning positive, which is often seen when a recovery phase starts to form",
                    ],
                    seed_key + ":mom:macdpos",
                )
            )
        elif macd < macd_signal:
            parts.append(
                _seeded_pick(
                    [
                        "The MACD structure is weakening, which often appears during fading rallies or slow grind-down phases",
                        "MACD remains under its signal line, suggesting downside pressure is still present in the momentum profile",
                    ],
                    seed_key + ":mom:macdneg",
                )
            )

    if not parts:
        return _clamp_sentence(
            "Momentum signals are not fully available in this scan, so it’s best to rely more heavily on trend and volume confirmation."
        ) + " " + _clamp_sentence(
            "This usually resolves on the next refresh as the feature set stabilizes."
        )

    # Ensure 2 sentences minimum
    if len(parts) == 1:
        parts.append(
            _seeded_pick(
                [
                    "The practical takeaway is to treat this as a “wait for confirmation” environment unless other indicators strongly agree",
                    "In plain terms: the momentum picture is informative, but not strong enough by itself to justify an aggressive action",
                ],
                seed_key + ":mom:tail",
            )
        )
    return _clamp_sentence(parts[0]) + " " + _clamp_sentence(parts[1])


def _build_volume_text(
    *,
    vol_z: Optional[float],
    seed_key: str,
) -> str:
    if vol_z is None:
        return _clamp_sentence(
            "Volume confirmation is not available in this update, so participation strength cannot be judged reliably."
        ) + " " + _clamp_sentence(
            "When volume is unknown, the system avoids overconfident claims because conviction is harder to measure."
        )

    if vol_z > 2:
        a = _seeded_pick(
            [
                "Trading activity is unusually strong, which often means institutions are actively involved and the move is more meaningful",
                "Volume is spiking well above normal levels, suggesting high participation rather than a low-liquidity drift",
            ],
            seed_key + ":vol:spike",
        )
        b = _seeded_pick(
            [
                "High participation can support follow-through, but it matters whether the volume is aligned with direction and momentum",
                "This is a constructive input for reliability, although the model still requires trend and momentum to agree before labeling it a clean edge",
            ],
            seed_key + ":vol:spike2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    if vol_z < -1:
        a = _seeded_pick(
            [
                "Trading participation is lighter than usual, which increases the chance of false starts and whipsaws",
                "Volume is below normal, which can make signals less dependable because price can move on thinner activity",
            ],
            seed_key + ":vol:thin",
        )
        b = _seeded_pick(
            [
                "In thin participation regimes, it’s common to see moves reverse quickly, so waiting for confirmation is usually the safer choice",
                "When volume is weak, the system becomes more conservative because the “story” behind the move is less convincing",
            ],
            seed_key + ":vol:thin2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    a = _seeded_pick(
        [
            "Volume is roughly in line with normal levels, which suggests the move is not being driven by a strong participation surge",
            "Trading activity is near average, meaning we don’t see an obvious participation tailwind in either direction",
        ],
        seed_key + ":vol:normal",
    )
    b = _seeded_pick(
        [
            "When volume is neutral, the model relies more on the quality of trend and momentum alignment to form a confident view",
            "This is a “normal participation” environment, so direction depends more on whether trend and momentum agree clearly",
        ],
        seed_key + ":vol:normal2",
    )
    return _clamp_sentence(a) + " " + _clamp_sentence(b)


def _build_volatility_text(
    *,
    intraday_range_pct: Optional[float],
    seed_key: str,
) -> str:
    if intraday_range_pct is None:
        return _clamp_sentence(
            "Volatility details are not available in this scan, so risk conditions cannot be measured precisely."
        ) + " " + _clamp_sentence(
            "When volatility is unknown, it’s better to avoid tight conclusions and focus on directional confirmation."
        )

    if intraday_range_pct > 4:
        a = _seeded_pick(
            [
                "Price swings are elevated, which increases risk and makes entries harder to time cleanly",
                "Volatility is high, so even a correct directional call can experience sharp pullbacks along the way",
            ],
            seed_key + ":vola:high",
        )
        b = _seeded_pick(
            [
                "In high-volatility regimes, the safer approach is to size smaller or wait for calmer structure before acting",
                "This environment rewards patience because noise can look like signal, especially around key levels",
            ],
            seed_key + ":vola:high2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    if intraday_range_pct < 1.5:
        a = _seeded_pick(
            [
                "Price action is relatively compressed, which often happens before a breakout or a larger directional move",
                "Volatility is muted, suggesting the market is coiling rather than trending aggressively",
            ],
            seed_key + ":vola:low",
        )
        b = _seeded_pick(
            [
                "In quieter regimes, confirmation becomes important because the first move can be a head fake",
                "This can be a constructive setup if momentum improves, but it still needs a trigger to become actionable",
            ],
            seed_key + ":vola:low2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    a = _seeded_pick(
        [
            "Volatility is moderate, which usually makes signals easier to interpret than extremely noisy conditions",
            "Price swings are present but not extreme, which is generally a healthier environment for reading trend and momentum",
        ],
        seed_key + ":vola:mid",
    )
    b = _seeded_pick(
        [
            "With volatility in a reasonable range, the quality of trend and participation becomes the main driver for conviction",
            "This tends to be an environment where aligned signals have a better chance of follow-through",
        ],
        seed_key + ":vola:mid2",
    )
    return _clamp_sentence(a) + " " + _clamp_sentence(b)


# ======================================================
# Pattern Context (kept calm, not “signal-like”)
# ======================================================

def _build_pattern_text(
    *,
    patt_name: Optional[str],
    patt_bias: Optional[str],
    win_rate: Optional[float],
    avg_ret: Optional[float],
    samples: Optional[int],
    seed_key: str,
) -> Optional[str]:
    if not patt_name:
        return None

    wr = None
    if isinstance(win_rate, (int, float)):
        wr = float(win_rate)

    avg = None
    if isinstance(avg_ret, (int, float)):
        avg = float(avg_ret)

    s = None
    if isinstance(samples, int) and samples > 0:
        s = samples

    bias_clause = ""
    if patt_bias in ("bull", "bear", "neutral"):
        if patt_bias == "bull":
            bias_clause = "It has historically leaned bullish in similar situations"
        elif patt_bias == "bear":
            bias_clause = "It has historically leaned bearish in similar situations"
        else:
            bias_clause = "It has historically behaved more mixed or balanced"

    stats_clause = ""
    if wr is not None:
        stats_clause = f"Over the next 5 days, it shows about {wr:.0%} favorable outcomes"
        if avg is not None:
            stats_clause += f" with an average move near {avg:+.2f}%"
        if s is not None:
            stats_clause += f" across {s} past samples"
        stats_clause += "."

    if not bias_clause and not stats_clause:
        return _clamp_sentence(
            f"A chart pattern is present ({patt_name.lower()}), but there is not enough history here to treat it as strong evidence."
        ) + " " + _clamp_sentence(
            "It’s best viewed as context rather than a decision by itself."
        )

    # Ensure 2 sentences minimum
    first = _clamp_sentence(f"A chart pattern is present ({patt_name.lower()}).")
    second = _clamp_sentence(bias_clause) if bias_clause else ""
    third = _clamp_sentence(stats_clause) if stats_clause else ""

    # If only one additional sentence exists, add a calm disclaimer
    add = []
    if second:
        add.append(second)
    if third:
        add.append(third)

    if len(add) == 0:
        add.append(_clamp_sentence("It should be treated as context, not a standalone trading signal."))
    elif len(add) == 1:
        add.append(_clamp_sentence("Patterns can fail, so alignment with trend, momentum, and volume matters more than the name of the pattern."))

    return first + " " + " ".join(add)


# ======================================================
# Signal Text (Long, friendly, not repetitive)
# ======================================================

def _build_signal_text(
    *,
    signal: str,
    confidence: float,
    seed_key: str,
) -> str:
    tier = _conf_tier(confidence)

    if signal == "BUY":
        a = _seeded_pick(
            [
                "The model is leaning bullish here, meaning upside conditions look more favorable than downside conditions",
                "The model sees a constructive setup where upside probability appears more attractive than the downside risk",
            ],
            seed_key + ":sig:buy",
        )
        b = _seeded_pick(
            [
                f"Confidence is {tier}, so this should still be managed with risk controls rather than treated as a guarantee",
                f"This comes with {tier} confidence, so disciplined sizing and a clear invalidation level are still important",
            ],
            seed_key + ":sig:buy2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    if signal == "SELL":
        a = _seeded_pick(
            [
                "The model is leaning bearish, which means downside risk is still the more dominant outcome in this setup",
                "The model is defensive here, suggesting weakness is more likely to persist than reverse immediately",
            ],
            seed_key + ":sig:sell",
        )
        b = _seeded_pick(
            [
                f"Confidence is {tier}, so the practical move is to avoid chasing strength and to respect volatility and risk",
                f"This is a {tier}-confidence environment, so patience and confirmation matter if you plan to trade around the move",
            ],
            seed_key + ":sig:sell2",
        )
        return _clamp_sentence(a) + " " + _clamp_sentence(b)

    # HOLD
    a = _seeded_pick(
        [
            "The model is not seeing a clean directional edge right now, which is why it prefers a wait-and-confirm posture",
            "The setup is currently mixed, so the model avoids forcing a direction when the evidence is not strong enough",
            "This is a low-edge environment for a directional call, so patience is usually the better trade than guessing",
        ],
        seed_key + ":sig:hold",
    )
    b = _seeded_pick(
        [
            f"Confidence is {tier}, and that usually means the market needs either stronger trend confirmation or clearer momentum pressure before acting",
            f"With {tier} confidence, the best next step is to wait for alignment rather than taking a trade that depends on luck",
        ],
        seed_key + ":sig:hold2",
    )
    return _clamp_sentence(a) + " " + _clamp_sentence(b)


# ======================================================
# Main Generator
# ======================================================

def generate_bull_insights(
    *,
    symbol: str,
    features: Dict[str, Any],
    bullbrain: Dict[str, Any],
    technical: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
    pattern: Optional[Dict[str, Any]] = None,
    pattern_history: Optional[Dict[str, Any]] = None,
    seed_key: Optional[str] = None,
    vol_z_corrected: Optional[float] = None,
) -> Dict[str, Any]:

    # keep params for compatibility
    technical = technical or {}
    decision = decision or {}
    pattern = pattern or {}
    pattern_history = pattern_history or {}

    signal = str(decision.get("finalSignal") or bullbrain.get("signal") or "HOLD").upper()
    confidence = float(_num(bullbrain.get("confidence"), 0.0) or 0.0)

    # Deterministic seed: if you pass a time bucket from cron, text changes only when that changes
    seed_key = seed_key or f"{symbol}:{technical.get('updated_at') or ''}"

    # --- Extract features (don’t remove)
    sma5 = _num(features.get("sma5"))
    sma20 = _num(features.get("sma20"))
    rsi14 = _num(features.get("rsi14"))
    macd = _num(features.get("macd"))
    macd_signal = _num(features.get("macd_signal"))
    # volume_zscore_20 from features is inflated ~6.66x (see
    # bullbrain_gate_ladder_audit memory). Prefer the corrected value.
    vol_z = _num(vol_z_corrected) if vol_z_corrected is not None else _num(features.get("volume_zscore_20"))
    intraday_range_pct = _num(features.get("intraday_range_pct"))

    # --- Pattern context
    patt_name = pattern.get("pattern") or pattern.get("patternLabel")
    patt_bias = pattern.get("bias") or pattern.get("patternBias")

    days5 = (pattern_history.get("forwardReturns") or {}).get("days5") or {}
    wr = days5.get("winRate")
    avg = days5.get("avg")
    cnt = days5.get("count")

    # ==================================================
    # Build long narratives (each 2+ sentences)
    # ==================================================

    trend_text = _build_trend_text(sma5=sma5, sma20=sma20, seed_key=seed_key)
    momentum_text = _build_momentum_text(rsi14=rsi14, macd=macd, macd_signal=macd_signal, seed_key=seed_key)
    volume_text = _build_volume_text(vol_z=vol_z, seed_key=seed_key)
    volatility_text = _build_volatility_text(intraday_range_pct=intraday_range_pct, seed_key=seed_key)

    pattern_text = _build_pattern_text(
        patt_name=patt_name,
        patt_bias=patt_bias,
        win_rate=wr if isinstance(wr, (int, float)) else None,
        avg_ret=avg if isinstance(avg, (int, float)) else None,
        samples=cnt if isinstance(cnt, int) else None,
        seed_key=seed_key,
    )

    signal_text = _build_signal_text(signal=signal, confidence=confidence, seed_key=seed_key)

    # ==================================================
    # Primary oneliner becomes a readable paragraph
    # (Still returned in the same key: oneLiner)
    # ==================================================
    # Make it feel unique by picking a different "lead" emphasis per symbol.
    lead_choices = [
        ("trend", trend_text),
        ("momentum", momentum_text),
        ("volume", volume_text),
        ("volatility", volatility_text),
    ]
    lead = _seeded_pick([k for k, _ in lead_choices], seed_key + ":lead")
    lead_map = {k: v for k, v in lead_choices}
    lead_text = lead_map.get(lead) or trend_text

    # oneLiner: long, explainable, plain English (2-3 sentences)
    one_liner = (
        _clamp_sentence(lead_text)
        + " "
        + _clamp_sentence(signal_text)
    )
    if pattern_text:
        one_liner += " " + _clamp_sentence(pattern_text)

    # summaryLine: a second long explanation, built from the remaining pieces
    # Ensure it does NOT repeat lead_text verbatim.
    remaining = [trend_text, momentum_text, volume_text, volatility_text]
    remaining = [t for t in remaining if t and t != lead_text]

    # Pick 2 additional components deterministically
    if remaining:
        r1 = remaining[0]
        r2 = remaining[1] if len(remaining) > 1 else ""
    else:
        r1 = "Signals are present, but the evidence is not complete in this scan."
        r2 = "Waiting for alignment usually improves reliability."

    summary_line = _clamp_sentence(r1) + " " + _clamp_sentence(r2)
    if pattern_text:
        summary_line += " " + _clamp_sentence("The pattern information above is context and should not be treated as the signal by itself.")

    combined = " ".join(
        _clamp_sentence(x)
        for x in [one_liner, summary_line]
        if x
    ).strip()

    return {
        "oneLiner": one_liner,
        "whySignal": signal_text,  # optional for UI usage, long sentence
        "summaryLine": summary_line,

        # Each of these is now long/plain English (not "Trend:" labels)
        "trendSummary": trend_text,
        "momentumSummary": momentum_text,
        "volumeSummary": volume_text,
        "volatilitySummary": volatility_text,

        "combinedTechnicalSummary": combined,
    }
