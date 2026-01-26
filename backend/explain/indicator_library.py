# backend/explain/indicator_library.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

"""
Indicator Library:
- Defines WHAT each indicator means
- Provides state-specific narration templates
- Generates many high-quality variants deterministically

You get:
- short (1 line)
- medium (2–3 lines)
- institutional (professional trading desk style)
"""


# ----------------------------
# Formatting helpers
# ----------------------------
def fmt_num(x: Any, decimals: int = 2) -> str:
    try:
        if x is None:
            return "n/a"
        v = float(x)
        return f"{v:.{decimals}f}"
    except Exception:
        return "n/a"


def fmt_pct(x: Any, decimals: int = 2) -> str:
    try:
        if x is None:
            return "n/a"
        v = float(x)
        return f"{v:.{decimals}f}%"
    except Exception:
        return "n/a"


def fmt_int(x: Any) -> str:
    try:
        if x is None:
            return "n/a"
        return f"{int(float(x)):,}"
    except Exception:
        return "n/a"


def stable_choice(items: list[str], seed: str) -> str:
    if not items:
        return ""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(items)
    return items[idx]


# ----------------------------
# Variant builders (this is how we get "many versions")
# ----------------------------
def combine(*parts: list[str]) -> list[str]:
    """
    Cartesian combine of short phrase banks, capped to a reasonable set.
    """
    out = [""]
    for bank in parts:
        nxt = []
        for a in out:
            for b in bank:
                s = (a + " " + b).strip()
                nxt.append(s)
        out = nxt
        if len(out) > 120:  # cap
            out = out[:120]
    return out


def variants_short(core: str, cues: list[str]) -> list[str]:
    # ~10+ variants
    lead = [
        core,
        f"{core} {stable_choice(cues, core)}",
        f"{stable_choice(cues, core+'a')} {core.lower()}",
    ]
    # expand with structured phrase banks
    bank1 = [core]
    bank2 = cues[:]
    bank3 = ["", "Risk is higher until confirmation.", "Watch follow-through."]
    raw = combine(bank1, bank2, bank3)
    # de-dup
    uniq = []
    seen = set()
    for s in lead + raw:
        s2 = " ".join(s.split())
        if s2 and s2 not in seen:
            seen.add(s2)
            uniq.append(s2)
        if len(uniq) >= 18:
            break
    return uniq


def variants_institutional(headline: str, detail_points: list[str]) -> list[str]:
    openers = [
        "Desk read:",
        "Tape read:",
        "Market structure:",
        "Institutional view:",
        "Setup:",
        "Flow read:",
    ]
    frames = [
        "The signal is best treated as context until structure improves.",
        "Edge is limited without better alignment across momentum and participation.",
        "Prefer confirmation (trend + volume) before sizing risk.",
        "In this regime, false starts are common; prioritize risk control.",
        "Focus on levels and participation rather than headlines.",
    ]
    # Build 10–20 variants
    bank = combine(openers, [headline], frames)
    # Append 1–2 bullet-like points inline
    enriched = []
    for s in bank[:40]:
        p1 = stable_choice(detail_points, s + "p1")
        p2 = stable_choice(detail_points, s + "p2")
        enriched.append(f"{s} {p1} {p2}".strip())
        if len(enriched) >= 18:
            break
    # de-dup
    uniq = []
    seen = set()
    for s in enriched:
        s2 = " ".join(s.split())
        if s2 not in seen:
            seen.add(s2)
            uniq.append(s2)
    return uniq


# ----------------------------
# Library definition
# ----------------------------
# Each indicator defines:
# - label
# - unit (pct/num/int)
# - states: state -> {short[], medium[], institutional[]}
#
# For compactness, medium is derived from short+institutional in narrative_engine if missing.
# But we still provide medium phrase banks for key indicators.


INDICATOR_LIBRARY: dict[str, dict] = {}

def _add(
    name: str,
    label: str,
    unit: str,
    meaning: str,
    states: dict[str, dict[str, list[str]]],
) -> None:
    INDICATOR_LIBRARY[name] = {
        "label": label,
        "unit": unit,
        "meaning": meaning,
        "states": states,
    }


# ----------------------------
# STATE BANKS (shared)
# ----------------------------
CUES_CONFIRM = [
    "Needs confirmation from trend and volume.",
    "Wait for confirmation.",
    "Confirmation improves signal quality.",
    "Treat as context, not a trigger.",
    "Let price prove it.",
    "Don’t front-run; confirm first.",
]

CUES_RISK = [
    "Whipsaw risk is elevated.",
    "Risk control matters here.",
    "Sizing should be conservative.",
    "Avoid forcing trades.",
    "Patience beats prediction.",
]

# ----------------------------
# Core indicators with rich templates
# ----------------------------

# RSI
_add(
    "rsi14",
    "RSI (14)",
    "num",
    "Measures momentum speed; extremes often precede consolidation or mean reversion.",
    states={
        "OVERSOLD_EXTREME": {
            "short": variants_short("RSI is deeply oversold.", CUES_CONFIRM + CUES_RISK),
            "institutional": variants_institutional(
                "Momentum is at an exhaustion extreme.",
                [
                    "Oversold can bounce, but bounces fail frequently in weak trends.",
                    "Look for stabilization and higher lows before sizing long risk.",
                    "If volume stays thin, moves can reverse quickly.",
                    "In downtrends, oversold can stay oversold—avoid catching knives.",
                ],
            ),
        },
        "OVERSOLD": {
            "short": variants_short("RSI is oversold.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Momentum is stretched to the downside.",
                [
                    "Relief rallies are possible, but trend alignment is required.",
                    "Prefer confirmation: reclaim key averages or improve MACD structure.",
                    "Oversold is a condition, not a signal.",
                ],
            ),
        },
        "BEARISH_WEAK": {
            "short": variants_short("RSI is below neutral, momentum is soft.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Momentum tilts bearish but not capitulated.",
                [
                    "Weak momentum often leads to choppy downside drift.",
                    "If trend is also weak, rallies tend to fade.",
                    "Volume confirmation is key for any reversal thesis.",
                ],
            ),
        },
        "NEUTRAL": {
            "short": variants_short("RSI is neutral.", ["Momentum is balanced.", "No momentum extreme.", "Direction depends on structure."]),
            "institutional": variants_institutional(
                "Momentum is balanced; structure will decide.",
                [
                    "Neutral momentum shifts focus to trend, support/resistance, and participation.",
                    "Breaks from range matter more than oscillator readings here.",
                ],
            ),
        },
        "BULLISH_WEAK": {
            "short": variants_short("RSI is above neutral, momentum is improving.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Momentum is constructive but not euphoric.",
                [
                    "Best outcomes occur when trend and volume confirm.",
                    "Watch for continuation after minor pullbacks.",
                ],
            ),
        },
        "OVERBOUGHT": {
            "short": variants_short("RSI is overbought.", ["Risk of pullback rises.", "Chasing becomes lower quality.", "Wait for reset."]),
            "institutional": variants_institutional(
                "Momentum is extended; odds of consolidation rise.",
                [
                    "Overbought can persist in strong trends—use structure to judge.",
                    "If volume fades, extension risk increases.",
                ],
            ),
        },
        "OVERBOUGHT_EXTREME": {
            "short": variants_short("RSI is extremely overbought.", ["Extension risk is high.", "Pullbacks can be sharp.", "Avoid late entries."]),
            "institutional": variants_institutional(
                "Momentum is at a frothy extreme.",
                [
                    "Late-cycle entries have poor risk/reward unless trend is accelerating.",
                    "Prefer pullback setups or break-and-hold behavior.",
                ],
            ),
        },
        "UNKNOWN": {"short": ["RSI unavailable."], "institutional": ["Momentum oscillator unavailable."]},
    },
)

# MACD histogram
_add(
    "macd_hist",
    "MACD Histogram",
    "num",
    "Captures momentum acceleration/deceleration; positive supports upside continuation.",
    states={
        "POS_STRONG": {
            "short": variants_short("MACD momentum is strongly bullish.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Momentum acceleration favors upside continuation.",
                [
                    "Strong positive histogram suggests trend continuation if structure holds.",
                    "Watch for pullback-on-light-volume entries rather than chasing peaks.",
                ],
            ),
        },
        "POS_MILD": {
            "short": variants_short("MACD momentum is mildly bullish.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Momentum is improving but not decisive.",
                [
                    "Upside continuation is possible with participation.",
                    "If trend is sideways, expect range behavior despite positive momentum.",
                ],
            ),
        },
        "NEUTRAL": {
            "short": variants_short("MACD momentum is flat.", ["No acceleration.", "Market is in balance.", "Wait for expansion."]),
            "institutional": variants_institutional(
                "Momentum is not expanding; risk/reward relies on structure.",
                [
                    "Flat histogram is common in consolidations.",
                    "A breakout needs volume to sustain.",
                ],
            ),
        },
        "NEG_MILD": {
            "short": variants_short("MACD momentum is mildly bearish.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Momentum is softening; rallies may fade.",
                [
                    "Mild negative histogram often precedes range-to-down transitions.",
                    "Watch whether support holds; failure invites downside continuation.",
                ],
            ),
        },
        "NEG_STRONG": {
            "short": variants_short("MACD momentum is strongly bearish.", CUES_CONFIRM + CUES_RISK),
            "institutional": variants_institutional(
                "Downside momentum acceleration is present.",
                [
                    "Bearish histogram strengthens the case for downside pressure.",
                    "Counter-trend bounces are lower quality unless structure shifts.",
                ],
            ),
        },
        "UNKNOWN": {"short": ["MACD histogram unavailable."], "institutional": ["Momentum acceleration unavailable."]},
    },
)

# price vs SMA20
_add(
    "price_vs_sma20_pct",
    "Price vs SMA20",
    "pct",
    "Measures where price sits relative to the short-term trend baseline (20-day).",
    states={
        "BELOW_STRONG": {
            "short": variants_short("Price is well below the 20-day trend.", CUES_CONFIRM + CUES_RISK),
            "institutional": variants_institutional(
                "Structure is below the short-term trend baseline.",
                [
                    "Below-SMA regimes often punish early dip-buys without momentum turn.",
                    "Reclaiming the 20-day with volume improves the long thesis.",
                ],
            ),
        },
        "BELOW_MILD": {
            "short": variants_short("Price is below the 20-day trend.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Structure is soft versus the 20-day baseline.",
                [
                    "Rallies can fail unless momentum improves.",
                    "Watch for stabilization and reclaim attempts.",
                ],
            ),
        },
        "NEAR": {
            "short": variants_short("Price is near its 20-day trend line.", ["Neutral positioning.", "Trend balance zone.", "Watch for break."]),
            "institutional": variants_institutional(
                "Price is near trend; direction depends on breakout or breakdown.",
                [
                    "Near-SMA zones often chop; confirm with momentum and volume.",
                    "Treat moves away from the average as the actionable information.",
                ],
            ),
        },
        "ABOVE_MILD": {
            "short": variants_short("Price is above the 20-day trend.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Structure is constructive versus the 20-day baseline.",
                [
                    "Above-SMA regimes support trend continuation if momentum stays firm.",
                    "Pullbacks that hold the average are higher quality.",
                ],
            ),
        },
        "ABOVE_STRONG": {
            "short": variants_short("Price is extended above the 20-day trend.", ["Extension risk rises.", "Pullback odds increase.", "Avoid chasing."]),
            "institutional": variants_institutional(
                "Price is extended above trend; risk/reward becomes less favorable.",
                [
                    "Extended states can persist in strong trends, but entries require discipline.",
                    "Prefer pullback/retest structure rather than late breakout chasing.",
                ],
            ),
        },
        "UNKNOWN": {"short": ["Trend position unavailable."], "institutional": ["Price vs SMA20 unavailable."]},
    },
)

# Volume Z-score
_add(
    "volume_zscore_20",
    "Volume Z-Score (20)",
    "num",
    "Measures how unusual today’s volume is versus 20-day norms; proxies participation strength.",
    states={
        "VOLUME_VERY_LOW": {
            "short": variants_short("Volume is extremely low versus normal.", CUES_RISK),
            "institutional": variants_institutional(
                "Participation is very weak; price moves are less trustworthy.",
                [
                    "Thin participation increases reversal/whipsaw risk.",
                    "Wait for volume confirmation before directional sizing.",
                ],
            ),
        },
        "VOLUME_LOW": {
            "short": variants_short("Volume is below normal.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Participation is below baseline; signals weaken.",
                [
                    "Low volume reduces follow-through probability.",
                    "Better trades appear when participation expands in the direction of the move.",
                ],
            ),
        },
        "VOLUME_NORMAL": {
            "short": variants_short("Volume is near normal.", ["Participation is average.", "No strong flow signal.", "Structure matters more."]),
            "institutional": variants_institutional(
                "Participation is in-line; structure drives outcome.",
                [
                    "Average volume means you rely on trend/momentum alignment.",
                    "Volume expansion is the confirmation trigger.",
                ],
            ),
        },
        "VOLUME_HIGH": {
            "short": variants_short("Volume is elevated.", CUES_CONFIRM),
            "institutional": variants_institutional(
                "Participation is above baseline; follow-through odds improve.",
                [
                    "Elevated volume supports continuation when aligned with trend.",
                    "Watch if elevated volume is buying or selling (candle + trend context).",
                ],
            ),
        },
        "VOLUME_SPIKE": {
            "short": variants_short("Volume spike detected.", ["Institutions are active.", "Expect volatility.", "Watch follow-through."]),
            "institutional": variants_institutional(
                "Abnormal volume suggests large players are active.",
                [
                    "Volume spikes often precede range expansion.",
                    "Confirm direction via structure and next-day behavior.",
                ],
            ),
        },
        "UNKNOWN": {"short": ["Volume Z-score unavailable."], "institutional": ["Participation metric unavailable."]},
    },
)

# Volatility 20d
_add(
    "volatility_20d",
    "Volatility (20d)",
    "pct",
    "Short-term realized volatility; higher vol increases noise and reduces signal reliability.",
    states={
        "LOW": {
            "short": variants_short("Volatility is low.", ["Market is calm.", "Moves may be compressed.", "Breakouts need catalysts."]),
            "institutional": variants_institutional(
                "Low-vol regime: compression and coiling behavior is common.",
                [
                    "Low vol can precede expansion; watch for volume + structure triggers.",
                    "Signals are cleaner when breakouts are confirmed with participation.",
                ],
            ),
        },
        "NORMAL": {
            "short": variants_short("Volatility is normal.", ["Noise is manageable.", "Signals are interpretable.", "Structure matters."]),
            "institutional": variants_institutional(
                "Normal-vol regime supports typical trend/mean-reversion behaviors.",
                [
                    "Use structure and momentum alignment for sizing decisions.",
                    "Avoid overreacting to single candles.",
                ],
            ),
        },
        "ELEVATED": {
            "short": variants_short("Volatility is elevated.", CUES_RISK),
            "institutional": variants_institutional(
                "Elevated vol increases noise and widens error bars.",
                [
                    "Prefer smaller size and stronger confirmation.",
                    "False breaks become more frequent in elevated vol regimes.",
                ],
            ),
        },
        "HIGH": {
            "short": variants_short("Volatility is high.", CUES_RISK),
            "institutional": variants_institutional(
                "High-vol regime: signals degrade; whipsaws rise.",
                [
                    "Prioritize risk control and confirmation.",
                    "In high vol, mean-reversion can dominate unless trend is exceptional.",
                ],
            ),
        },
        "UNKNOWN": {"short": ["Volatility unavailable."], "institutional": ["Volatility metric unavailable."]},
    },
)

# ----------------------------
# Bulk coverage for the remaining indicators
# (Still many variants per state, generated via the same banks)
# ----------------------------

def _generic_known(name: str, label: str, unit: str, meaning: str) -> None:
    _add(
        name, label, unit, meaning,
        states={
            "KNOWN": {
                "short": variants_short(f"{label} is available.", ["Used as a reference level.", "Supports the snapshot context.", "Provides baseline context."]),
                "institutional": variants_institutional(
                    f"{label} provides baseline context.",
                    [
                        "Use in conjunction with momentum and structure.",
                        "Baseline levels help frame trend and mean-reversion.",
                        "This metric is descriptive, not a standalone signal.",
                    ],
                ),
            },
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} is missing; interpretation is limited."]},
        }
    )

def _generic_return(name: str, label: str) -> None:
    _add(
        name, label, "pct",
        "Simple return over the period; shows directional pressure and recent behavior.",
        states={
            "DOWN_STRONG": {"short": variants_short(f"{label} is strongly negative.", CUES_RISK), "institutional": variants_institutional("Recent returns show strong downside pressure.", ["Strong negative returns can trigger forced selling.", "Bounces need confirmation.", "Trend alignment matters most here."])},
            "DOWN_MILD": {"short": variants_short(f"{label} is mildly negative.", CUES_CONFIRM), "institutional": variants_institutional("Recent returns lean down but not capitulated.", ["Chop risk is common.", "Watch support behavior.", "Volume confirms direction."])},
            "FLAT": {"short": variants_short(f"{label} is flat.", ["No clear drift.", "Range behavior likely.", "Wait for break."]), "institutional": variants_institutional("Returns are flat; range behavior dominates.", ["Use structure levels as triggers.", "Momentum/volume determine breakout quality."])},
            "UP_MILD": {"short": variants_short(f"{label} is mildly positive.", CUES_CONFIRM), "institutional": variants_institutional("Returns lean up; continuation possible with confirmation.", ["Prefer pullbacks that hold support.", "Watch participation."])},
            "UP_STRONG": {"short": variants_short(f"{label} is strongly positive.", ["Strong upside pressure.", "Extension risk can rise.", "Confirm continuation."]), "institutional": variants_institutional("Recent returns show strong upside pressure.", ["Strong returns can trend, but late entries worsen R/R.", "Prefer structured setups over chasing."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; short-term pressure is unclear."]},
        }
    )

def _generic_pctdist(name: str, label: str, meaning: str) -> None:
    _add(
        name, label, "pct", meaning,
        states={
            "BELOW_STRONG": {"short": variants_short(f"{label} is far below reference.", CUES_RISK), "institutional": variants_institutional("Positioning is materially below reference.", ["Below-reference regimes punish weak reversals.", "Reclaim attempts matter."])},
            "BELOW_MILD": {"short": variants_short(f"{label} is below reference.", CUES_CONFIRM), "institutional": variants_institutional("Positioning is below reference.", ["Structure is soft; confirmation needed.", "Watch reclaim attempts."])},
            "NEAR": {"short": variants_short(f"{label} is near reference.", ["Balance zone.", "Chop is common.", "Wait for expansion."]), "institutional": variants_institutional("Near-reference zones often chop.", ["Break-and-hold behavior is the trigger.", "Participation confirms moves."])},
            "ABOVE_MILD": {"short": variants_short(f"{label} is above reference.", CUES_CONFIRM), "institutional": variants_institutional("Positioning is above reference; structure is constructive.", ["Pullbacks that hold reference are higher quality.", "Momentum/volume confirm continuation."])},
            "ABOVE_STRONG": {"short": variants_short(f"{label} is far above reference.", ["Extension risk rises.", "Pullback odds increase.", "Avoid chasing."]), "institutional": variants_institutional("Positioning is extended; risk/reward deteriorates.", ["Prefer pullback/retest entries.", "Late breakouts can fail without volume."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; position cannot be assessed."]},
        }
    )

def _generic_signed(name: str, label: str, meaning: str) -> None:
    _add(
        name, label, "num", meaning,
        states={
            "POS_STRONG": {"short": variants_short(f"{label} is strongly positive.", CUES_CONFIRM), "institutional": variants_institutional("The metric is strongly positive, supporting upside bias.", ["Sustained positives help trend continuation.", "Confirm with structure and participation."])},
            "POS_MILD": {"short": variants_short(f"{label} is mildly positive.", CUES_CONFIRM), "institutional": variants_institutional("The metric is positive but not decisive.", ["Constructive, but needs structure confirmation.", "Sideways regimes can still chop."])},
            "NEUTRAL": {"short": variants_short(f"{label} is near neutral.", ["Balance state.", "No clear edge.", "Wait for expansion."]), "institutional": variants_institutional("Neutral readings shift focus to structure.", ["Use trend/levels for decisions.", "Confirm with volume."])},
            "NEG_MILD": {"short": variants_short(f"{label} is mildly negative.", CUES_CONFIRM), "institutional": variants_institutional("The metric is negative, mild downside bias.", ["Rallies may fade.", "Watch support and participation."])},
            "NEG_STRONG": {"short": variants_short(f"{label} is strongly negative.", CUES_RISK), "institutional": variants_institutional("The metric is strongly negative, supporting downside pressure.", ["Counter-trend moves are lower quality.", "Wait for structure shift."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; interpretation limited."]},
        }
    )

def _generic_volvs(name: str, label: str) -> None:
    _add(
        name, label, "pct",
        "Volume relative to moving average; measures participation vs baseline.",
        states={
            "BELOW_AVG_STRONG": {"short": variants_short(f"{label} is far below average.", CUES_RISK), "institutional": variants_institutional("Participation is materially below baseline.", ["Low participation reduces follow-through.", "Signals degrade in thin tape."])},
            "BELOW_AVG": {"short": variants_short(f"{label} is below average.", CUES_CONFIRM), "institutional": variants_institutional("Participation is below baseline.", ["Wait for expansion before directional sizing.", "Thin tape can whipsaw."])},
            "NEAR_AVG": {"short": variants_short(f"{label} is near average.", ["Participation is normal.", "No flow edge.", "Structure decides."]), "institutional": variants_institutional("Participation is in-line; structure matters.", ["Breakouts need volume to sustain.", "Momentum helps when participation is average."])},
            "ABOVE_AVG": {"short": variants_short(f"{label} is above average.", CUES_CONFIRM), "institutional": variants_institutional("Participation is above baseline.", ["Follow-through odds improve when aligned.", "Watch candle + trend context."])},
            "ABOVE_AVG_STRONG": {"short": variants_short(f"{label} is far above average.", ["Strong participation.", "Expect expansion.", "Confirm direction."]), "institutional": variants_institutional("Strong participation suggests active large players.", ["Volume shocks often precede range expansion.", "Confirm via next-session behavior."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; participation unclear."]},
        }
    )

def _generic_volatility(name: str, label: str) -> None:
    _add(
        name, label, "pct",
        "Realized volatility over the period; higher values increase noise and whipsaw risk.",
        states={
            "LOW": {"short": variants_short(f"{label} is low.", ["Compressed tape.", "Coiling behavior.", "Wait for expansion."]), "institutional": variants_institutional("Low-vol compression; watch for breakout triggers.", ["Volume confirms expansion.", "Structure is more reliable in low vol."])},
            "NORMAL": {"short": variants_short(f"{label} is normal.", ["Noise is manageable.", "Typical behavior.", "Structure matters."]), "institutional": variants_institutional("Normal vol supports standard trend/mean-reversion playbooks.", ["Use alignment for sizing.", "Avoid overreacting to one bar."])},
            "ELEVATED": {"short": variants_short(f"{label} is elevated.", CUES_RISK), "institutional": variants_institutional("Elevated vol increases noise; demand stronger confirmation.", ["False breaks increase.", "Reduce size / widen stops if trading."])},
            "HIGH": {"short": variants_short(f"{label} is high.", CUES_RISK), "institutional": variants_institutional("High-vol regime degrades signals; prioritize risk control.", ["Whipsaws are common.", "Wait for stabilization."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; volatility regime unclear."]},
        }
    )

def _generic_wick(name: str, label: str) -> None:
    _add(
        name, label, "pct",
        "Wick size as percent; large wicks indicate rejection and intraday tug-of-war.",
        states={
            "TINY": {"short": variants_short(f"{label} is tiny.", ["Little rejection.", "Cleaner close.", "Trend reads clearer."]), "institutional": variants_institutional("Small wick suggests cleaner directional control.", ["Less intraday rejection.", "Interpret with body + trend."])},
            "SMALL": {"short": variants_short(f"{label} is small.", ["Some rejection.", "Still orderly.", "Watch continuation."]), "institutional": variants_institutional("Modest wick suggests mild rejection.", ["Not decisive alone.", "Use with trend/momentum."])},
            "MEDIUM": {"short": variants_short(f"{label} is medium.", ["Noticeable rejection.", "Chop risk rises.", "Wait for clarity."]), "institutional": variants_institutional("Meaningful wick indicates rejection and uncertainty.", ["Higher chop risk.", "Confirm direction next bar."])},
            "LARGE": {"short": variants_short(f"{label} is large.", CUES_RISK), "institutional": variants_institutional("Large wick implies strong rejection; uncertainty elevated.", ["Whipsaw risk is higher.", "Treat as context until structure clarifies."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; candle anatomy incomplete."]},
        }
    )

def _generic_gap(name: str, label: str) -> None:
    _add(
        name, label, "pct",
        "Open vs prior close; gaps often reflect new information and repositioning.",
        states={
            "GAP_DOWN_BIG": {"short": variants_short("Large gap down.", CUES_RISK), "institutional": variants_institutional("Large downside gap suggests risk-off repricing.", ["Follow-through matters: gap-and-fade vs gap-and-go.", "Watch early levels and volume."])},
            "GAP_DOWN": {"short": variants_short("Gap down.", CUES_CONFIRM), "institutional": variants_institutional("Downside gap; buyers must prove demand.", ["Gaps can fill or extend—watch structure.", "Volume confirms conviction."])},
            "FLAT": {"short": variants_short("No meaningful gap.", ["Open was orderly.", "No shock.", "Trend drives."]), "institutional": variants_institutional("Orderly open; less overnight shock.", ["Intraday structure is more important."])},
            "GAP_UP": {"short": variants_short("Gap up.", CUES_CONFIRM), "institutional": variants_institutional("Upside gap; confirms demand only with follow-through.", ["Watch if gap holds (strength) or fades (supply).", "Volume confirms."])},
            "GAP_UP_BIG": {"short": variants_short("Large gap up.", ["Expect expansion.", "Volatility can rise.", "Avoid chasing."]), "institutional": variants_institutional("Large upside gap reflects repricing; late entries can be punished.", ["Prefer gap-hold structures.", "Look for measured pullbacks."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; open dynamics unclear."]},
        }
    )

def _generic_body(name: str, label: str) -> None:
    _add(
        name, label, "pct",
        "Candle body; shows directional control in the session.",
        states={
            "DOJI": {"short": variants_short("Doji-like body: indecision.", ["Tug-of-war.", "Wait for confirmation.", "Chop risk."]), "institutional": variants_institutional("Indecision candle; direction unresolved.", ["Treat as pause signal.", "Confirm next-session direction."])},
            "BULL_BODY": {"short": variants_short("Bullish candle body.", CUES_CONFIRM), "institutional": variants_institutional("Buyers had session control, but confirmation matters.", ["Look for follow-through and support holds.", "Volume strengthens the read."])},
            "BULL_BODY_STRONG": {"short": variants_short("Strong bullish candle body.", ["Demand dominated.", "Continuation possible.", "Watch follow-through."]), "institutional": variants_institutional("Strong close suggests demand control.", ["Best if aligned with trend + participation.", "Avoid late chasing; prefer pullbacks."])},
            "BEAR_BODY": {"short": variants_short("Bearish candle body.", CUES_CONFIRM), "institutional": variants_institutional("Sellers controlled the session; watch support.", ["Follow-through confirms downside.", "Beware snap-back rallies in oversold tape."])},
            "BEAR_BODY_STRONG": {"short": variants_short("Strong bearish candle body.", CUES_RISK), "institutional": variants_institutional("Strong downside close suggests supply control.", ["Downside continuation possible if support fails.", "Countertrend longs need proof."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; candle control unclear."]},
        }
    )

def _generic_trend_strength(name: str, label: str) -> None:
    _add(
        name, label, "num",
        "Trend strength proxy; higher absolute values imply more directional persistence.",
        states={
            "DOWN_STRONG": {"short": variants_short("Trend is strongly down.", CUES_RISK), "institutional": variants_institutional("Trend is decisively bearish.", ["Rallies tend to be sold.", "Wait for structural reversal signals."])},
            "DOWN_MILD": {"short": variants_short("Trend is mildly down.", CUES_CONFIRM), "institutional": variants_institutional("Trend leans bearish; chop-to-down is common.", ["Support behavior matters.", "Momentum shift improves odds."])},
            "SIDEWAYS": {"short": variants_short("Trend is sideways.", ["Range regime.", "Mean reversion common.", "Breakouts need volume."]), "institutional": variants_institutional("Sideways regime; range strategies dominate.", ["Wait for break-and-hold to switch playbook.", "Volume confirms breakouts."])},
            "UP_MILD": {"short": variants_short("Trend is mildly up.", CUES_CONFIRM), "institutional": variants_institutional("Trend leans bullish; pullbacks can be bought with confirmation.", ["Hold of averages supports continuation.", "Participation matters."])},
            "UP_STRONG": {"short": variants_short("Trend is strongly up.", ["Trend is persistent.", "Continuation bias.", "Avoid fading."]), "institutional": variants_institutional("Trend is decisively bullish.", ["Pullbacks are higher quality entries.", "Manage extension risk."])},
            "UNKNOWN": {"short": [f"{label} unavailable."], "institutional": [f"{label} missing; trend regime unclear."]},
        }
    )


# ----------------------------
# Build remaining indicator entries
# ----------------------------

# Price/OHLC/MA values (KNOWN/UNKNOWN)
_generic_known("adj_close", "Adjusted Close", "num", "Adjusted close price; reference price level.")
_generic_known("close", "Close", "num", "Close price; last traded reference for the session/day.")
_generic_known("high", "High", "num", "Session high; helps define range and resistance.")
_generic_known("low", "Low", "num", "Session low; helps define range and support.")
_generic_known("open", "Open", "num", "Session open; reflects overnight repricing.")
_add(
    "volume",
    "Volume",
    "int",
    "Shares traded; primary participation metric.",
    states={
        "KNOWN": {
            "short": variants_short("Volume is available.", ["Use to confirm moves.", "Participation matters.", "Helps validate breakouts."]),
            "institutional": variants_institutional("Volume provides participation context.", ["Volume confirms conviction.", "Thin volume increases noise."]),
        },
        "UNKNOWN": {"short": ["Volume unavailable."], "institutional": ["Volume missing; participation read is limited."]},
    },
)

# Returns
_generic_return("return_1d", "Return (1d)")
_generic_return("return_5d", "Return (5d)")
_generic_return("return_10d", "Return (10d)")

# Volatility
_generic_volatility("volatility_5d", "Volatility (5d)")
# volatility_20d already defined richly above
_generic_volatility("volatility_60d", "Volatility (60d)")

# Moving averages
_generic_known("sma5", "SMA (5)", "num", "Short moving average; micro-trend baseline.")
_generic_known("sma10", "SMA (10)", "num", "Short moving average; near-term baseline.")
_generic_known("sma20", "SMA (20)", "num", "Key short-term baseline; trend reference.")
_generic_known("sma50", "SMA (50)", "num", "Medium-term baseline; trend filter.")
_generic_known("sma200", "SMA (200)", "num", "Long-term trend baseline; regime anchor.")

# MA ratios
_generic_pctdist("sma5_sma20_pct", "SMA5 vs SMA20", "Short-term trend relative to baseline.")
_generic_pctdist("sma20_sma50_pct", "SMA20 vs SMA50", "Medium-term trend relative to baseline.")

# MACD line + signal
_generic_signed("macd", "MACD", "MACD line; direction and strength of momentum.")
_generic_signed("macd_signal", "MACD Signal", "Signal line; momentum trend baseline.")

# EMA lines
_generic_known("ema12", "EMA (12)", "num", "Short EMA; momentum-sensitive average.")
_generic_known("ema26", "EMA (26)", "num", "Long EMA; momentum-sensitive average.")
_generic_pctdist("ema_ratio", "EMA12/EMA26 Ratio", "Ratio above 1 supports bullish momentum; below 1 supports bearish momentum.")

# Oscillators
# williams_r_14, stoch_k_14, stoch_d_3 already mapped in states; narrations:
_add(
    "williams_r_14",
    "Williams %R (14)",
    "num",
    "Momentum oscillator; closer to -100 implies oversold, closer to 0 implies overbought.",
    states={
        "OVERSOLD_EXTREME": {"short": variants_short("Williams %R is deeply oversold.", CUES_CONFIRM), "institutional": variants_institutional("Momentum is at an oversold extreme.", ["Oversold can bounce, but trend alignment is required.", "Use structure confirmation."])},
        "OVERSOLD": {"short": variants_short("Williams %R is oversold.", CUES_CONFIRM), "institutional": variants_institutional("Momentum is stretched to the downside.", ["Treat as condition; confirm with structure."])},
        "BEARISH_WEAK": {"short": variants_short("Williams %R leans bearish.", CUES_CONFIRM), "institutional": variants_institutional("Momentum leans down but not exhausted.", ["Chop risk is common in this zone."])},
        "NEUTRAL": {"short": variants_short("Williams %R is neutral.", ["Momentum is balanced.", "No extreme.", "Structure decides."]), "institutional": variants_institutional("Balanced oscillator; structure drives outcomes.", ["Watch break-and-hold."])},
        "BULLISH_WEAK": {"short": variants_short("Williams %R leans bullish.", CUES_CONFIRM), "institutional": variants_institutional("Momentum leans up but not extended.", ["Trend alignment improves continuation odds."])},
        "OVERBOUGHT": {"short": variants_short("Williams %R is overbought.", ["Extension risk rises.", "Pullback odds rise.", "Avoid chasing."]), "institutional": variants_institutional("Extension condition; odds of consolidation rise.", ["Overbought can persist in strong trends—use structure."])},
        "OVERBOUGHT_EXTREME": {"short": variants_short("Williams %R is extremely overbought.", CUES_RISK), "institutional": variants_institutional("Frothy momentum; late entries degrade.", ["Prefer pullback setups."])},
        "UNKNOWN": {"short": ["Williams %R unavailable."], "institutional": ["Oscillator unavailable."]},
    }
)

_add(
    "stoch_k_14",
    "Stochastic %K (14)",
    "num",
    "Momentum oscillator; high values indicate strength/extension, low values indicate weakness/oversold.",
    states={
        "OVERSOLD_EXTREME": {"short": variants_short("Stoch %K is deeply oversold.", CUES_CONFIRM), "institutional": variants_institutional("Momentum is washed out.", ["Oversold needs structure confirmation.", "Avoid catching knives in downtrends."])},
        "OVERSOLD": {"short": variants_short("Stoch %K is oversold.", CUES_CONFIRM), "institutional": variants_institutional("Momentum is stretched lower.", ["Bounces require confirmation from trend/volume."])},
        "BEARISH_WEAK": {"short": variants_short("Stoch %K leans bearish.", CUES_CONFIRM), "institutional": variants_institutional("Momentum is soft; rallies can fade.", ["Support behavior matters."])},
        "NEUTRAL": {"short": variants_short("Stoch %K is neutral.", ["No extreme.", "Balance state.", "Wait for expansion."]), "institutional": variants_institutional("Balanced oscillator; structure leads.", ["Breakouts need participation."])},
        "BULLISH_WEAK": {"short": variants_short("Stoch %K leans bullish.", CUES_CONFIRM), "institutional": variants_institutional("Constructive momentum; continuation possible.", ["Trend alignment improves."])},
        "OVERBOUGHT": {"short": variants_short("Stoch %K is overbought.", ["Extension risk rises.", "Pullback odds rise.", "Avoid chasing."]), "institutional": variants_institutional("Extension condition; expect consolidation risk.", ["Overbought can persist in strong trends—use structure."])},
        "OVERBOUGHT_EXTREME": {"short": variants_short("Stoch %K is extremely overbought.", CUES_RISK), "institutional": variants_institutional("Frothy momentum; late entries degrade.", ["Prefer pullbacks / retests."])},
        "UNKNOWN": {"short": ["Stoch %K unavailable."], "institutional": ["Oscillator unavailable."]},
    }
)

_add(
    "stoch_d_3",
    "Stochastic %D (3)",
    "num",
    "Smoothed stochastic signal; confirms %K behavior and momentum shifts.",
    states=INDICATOR_LIBRARY["stoch_k_14"]["states"],  # same states/phrasing is acceptable
)

# Volume features
_generic_return("volume_change_1d", "Volume Change (1d)")
_generic_known("volume_ma5", "Volume MA (5)", "int", "5-day volume baseline.")
_generic_known("volume_ma20", "Volume MA (20)", "int", "20-day volume baseline.")
_generic_volvs("volume_vs_ma5_pct", "Volume vs MA5")
_generic_volvs("volume_vs_ma20_pct", "Volume vs MA20")
# volume_zscore_20 already defined
_generic_known("obv", "OBV", "int", "On-Balance Volume; cumulative volume flow proxy.")
_add(
    "obv_slope_10",
    "OBV Slope (10)",
    "num",
    "OBV trend; rising suggests accumulation, falling suggests distribution.",
    states={
        "RISING": {"short": variants_short("OBV slope is rising: accumulation.", CUES_CONFIRM), "institutional": variants_institutional("Flow proxy indicates accumulation.", ["Prefer longs when structure confirms.", "Confirm with trend/momentum."])},
        "FALLING": {"short": variants_short("OBV slope is falling: distribution.", CUES_CONFIRM), "institutional": variants_institutional("Flow proxy indicates distribution.", ["Rallies may fade in distribution.", "Downside pressure can persist."])},
        "FLAT": {"short": variants_short("OBV slope is flat: no clear flow edge.", ["Flow is balanced.", "No strong accumulation/distribution.", "Structure decides."]), "institutional": variants_institutional("Flow is balanced; structure matters more.", ["Wait for participation shift."])},
        "UNKNOWN": {"short": ["OBV slope unavailable."], "institutional": ["Flow metric unavailable."]},
    }
)

# Range / ATR / candle
_add(
    "intraday_range_pct",
    "Intraday Range",
    "pct",
    "High-low range as percent; wider ranges imply higher intraday uncertainty.",
    states={
        "TIGHT": {"short": variants_short("Intraday range is tight.", ["Orderly tape.", "Compression.", "Breakout needs trigger."]), "institutional": variants_institutional("Tight range suggests compression.", ["Watch for expansion with volume.", "Structure becomes key."])},
        "NORMAL": {"short": variants_short("Intraday range is normal.", ["Typical noise.", "Manageable volatility.", "Structure matters."]), "institutional": variants_institutional("Normal range supports standard playbooks.", ["Use trend/momentum alignment."])},
        "WIDE": {"short": variants_short("Intraday range is wide.", CUES_RISK), "institutional": variants_institutional("Wide range increases noise and stop risk.", ["Prefer confirmation and smaller size."])},
        "VERY_WIDE": {"short": variants_short("Intraday range is very wide.", CUES_RISK), "institutional": variants_institutional("Very wide range implies instability.", ["Signals degrade; prioritize risk control."])},
        "UNKNOWN": {"short": ["Intraday range unavailable."], "institutional": ["Range metric unavailable."]},
    }
)

_generic_known("true_range", "True Range", "num", "True range; captures gap-inclusive daily movement.")
_add(
    "atr14",
    "ATR (14)",
    "num",
    "Average True Range; expected daily movement proxy used for sizing risk.",
    states={
        "ATR_LOW": {"short": variants_short("ATR is low: smaller daily movement.", ["Tighter moves.", "Less noise.", "Breakouts need catalysts."]), "institutional": variants_institutional("Lower ATR implies smaller expected movement.", ["Targets/stops can be tighter.", "Breakouts need participation."])},
        "ATR_NORMAL": {"short": variants_short("ATR is normal.", ["Typical movement.", "Manageable noise.", "Structure matters."]), "institutional": variants_institutional("ATR is in normal regime.", ["Use standard risk sizing; confirm with structure."])},
        "ATR_ELEVATED": {"short": variants_short("ATR is elevated: larger daily movement.", CUES_RISK), "institutional": variants_institutional("Elevated ATR widens expected movement.", ["Reduce size or widen stops if trading.", "Signals require stronger confirmation."])},
        "ATR_HIGH": {"short": variants_short("ATR is high: very large movement.", CUES_RISK), "institutional": variants_institutional("High ATR implies unstable movement and wider error bars.", ["Whipsaw risk rises; trade smaller or wait."])},
        "UNKNOWN": {"short": ["ATR unavailable."], "institutional": ["ATR missing; sizing context limited."]},
    }
)

_generic_wick("upper_shadow_pct", "Upper Wick")
_generic_wick("lower_shadow_pct", "Lower Wick")
_generic_body("body_pct", "Candle Body")
_generic_gap("gap_pct", "Gap %")

_generic_pctdist("distance_from_20d_high", "Distance from 20D High", "How far price sits from its 20-day high; gauges drawdown/extension.")
_generic_pctdist("distance_from_20d_low", "Distance from 20D Low", "How far price sits from its 20-day low; gauges bounce/extension off lows.")

_generic_trend_strength("trend_strength_20", "Trend Strength (20)",)

# ----------------------------
# Decision-layer “indicators” (18)
# ----------------------------
_add(
    "prob_up",
    "Model Prob Up",
    "num",
    "Probability of upward outcome (0..1) from model blend.",
    states={
        "LOW": {"short": variants_short("Upside probability is low.", CUES_CONFIRM), "institutional": variants_institutional("Prob-up is low; downside bias dominates.", ["Treat long risk cautiously.", "Wait for structure shift."])},
        "LEAN_DOWN": {"short": variants_short("Upside probability leans down.", CUES_CONFIRM), "institutional": variants_institutional("Probabilities lean bearish but not extreme.", ["Edge is limited; confirmation matters."])},
        "NEUTRAL": {"short": variants_short("Upside probability is neutral.", ["Balanced odds.", "No edge.", "Structure matters."]), "institutional": variants_institutional("Probabilities are near 50/50; edge is low.", ["Let price/volume decide."])},
        "LEAN_UP": {"short": variants_short("Upside probability leans up.", CUES_CONFIRM), "institutional": variants_institutional("Probabilities lean bullish; confirm with structure.", ["Trend/volume alignment improves outcomes."])},
        "HIGH": {"short": variants_short("Upside probability is high.", ["Bullish odds.", "Continuation possible.", "Watch confirmation."]), "institutional": variants_institutional("Prob-up is high; bullish odds dominate if structure holds.", ["Avoid late chasing; prefer structured entries."])},
        "UNKNOWN": {"short": ["Prob-up unavailable."], "institutional": ["Probability unavailable."]},
    }
)

_add(
    "prob_down",
    "Model Prob Down",
    "num",
    "Probability of downward outcome (0..1).",
    states=INDICATOR_LIBRARY["prob_up"]["states"],  # same state system applies
)

_add(
    "confidence",
    "Signal Confidence",
    "pct",
    "Confidence score (0..100); reflects strength of edge after model output.",
    states={
        "LOW": {"short": variants_short("Confidence is low: edge is weak.", CUES_RISK), "institutional": variants_institutional("Low confidence implies low edge.", ["Avoid aggressive positioning.", "Wait for alignment."])},
        "MODERATE": {"short": variants_short("Confidence is moderate.", CUES_CONFIRM), "institutional": variants_institutional("Moderate confidence suggests mixed evidence.", ["Selective setups only.", "Confirm with structure."])},
        "HIGH": {"short": variants_short("Confidence is high.", ["Better edge.", "Still needs execution discipline.", "Watch levels."]), "institutional": variants_institutional("High confidence supports the signal, contingent on structure.", ["Use levels + participation for entries."])},
        "VERY_HIGH": {"short": variants_short("Confidence is very high.", ["Strong edge.", "Execution still matters.", "Manage risk."]), "institutional": variants_institutional("Very high confidence supports strong bias.", ["Avoid over-sizing; manage tail risk."])},
        "UNKNOWN": {"short": ["Confidence unavailable."], "institutional": ["Confidence metric unavailable."]},
    }
)

_add(
    "hybridProbUp",
    "Hybrid Prob Up",
    "num",
    "Hybrid probability after blending components (0..1).",
    states=INDICATOR_LIBRARY["prob_up"]["states"],
)
_add(
    "hybridProbDown",
    "Hybrid Prob Down",
    "num",
    "Hybrid downside probability (0..1).",
    states=INDICATOR_LIBRARY["prob_up"]["states"],
)

_add(
    "bias_strength",
    "Bias Strength",
    "num",
    "Directional bias strength (0..100) from decision block.",
    states={
        "WEAK": {"short": variants_short("Bias strength is weak.", ["Edge is small.", "Expect chop.", "Confirmation needed."]), "institutional": variants_institutional("Weak bias implies low edge.", ["Structure and participation matter most here."])},
        "MODERATE": {"short": variants_short("Bias strength is moderate.", CUES_CONFIRM), "institutional": variants_institutional("Moderate bias; confirm before sizing.", ["Focus on levels and follow-through."])},
        "STRONG": {"short": variants_short("Bias strength is strong.", ["Clear lean.", "Execution matters.", "Watch levels."]), "institutional": variants_institutional("Strong bias supports directional thesis if structure holds.", ["Avoid late chasing; prefer pullbacks/retests."])},
        "VERY_STRONG": {"short": variants_short("Bias strength is very strong.", ["Directional conviction is high.", "Manage risk.", "Expect expansion."]), "institutional": variants_institutional("Very strong bias; conditions favor direction.", ["Tail risk still exists; risk control remains essential."])},
        "UNKNOWN": {"short": ["Bias strength unavailable."], "institutional": ["Bias strength missing."]},
    }
)

_add(
    "bias_label",
    "Bias Label",
    "str",
    "Directional bias label from decision block.",
    states={
        "BULLISH": {"short": variants_short("Bias label is bullish.", CUES_CONFIRM), "institutional": variants_institutional("Bias is bullish.", ["Confirm with trend + participation for higher quality entries."])},
        "BEARISH": {"short": variants_short("Bias label is bearish.", CUES_CONFIRM), "institutional": variants_institutional("Bias is bearish.", ["Rallies may fade unless structure shifts."])},
        "NEUTRAL": {"short": variants_short("Bias label is neutral.", ["Low edge.", "Range risk.", "Wait for break."]), "institutional": variants_institutional("Bias is neutral; edge is low.", ["Let structure decide direction."])},
        "UNKNOWN": {"short": ["Bias label unavailable."], "institutional": ["Bias label missing."]},
    }
)

_add(
    "finalSignal",
    "Final Signal",
    "str",
    "Final decision output: BUY/SELL/HOLD.",
    states={
        "BUY": {"short": variants_short("Signal is BUY.", ["Bullish setup.", "Confirm entry level.", "Manage risk."]), "institutional": variants_institutional("Directional bias is BUY.", ["Prefer structured entries and risk-defined execution.", "Confirm participation."])},
        "SELL": {"short": variants_short("Signal is SELL.", ["Bearish setup.", "Confirm breakdown.", "Manage risk."]), "institutional": variants_institutional("Directional bias is SELL.", ["Downside continuation improves if support fails.", "Watch participation."])},
        "HOLD": {"short": variants_short("Signal is HOLD.", ["Edge is mixed.", "Wait for alignment.", "Avoid forcing trades."]), "institutional": variants_institutional("HOLD reflects mixed evidence and insufficient edge.", ["Wait for alignment across structure, momentum, and participation."])},
        "UNKNOWN": {"short": ["Signal unavailable."], "institutional": ["Signal missing."]},
    }
)

_add(
    "liquidity_quality",
    "Liquidity Quality",
    "str",
    "Liquidity classification used by your decision ladder.",
    states={
        "GOOD": {"short": variants_short("Liquidity is good.", ["Execution is cleaner.", "Slippage lower.", "Signals are usable."]), "institutional": variants_institutional("Liquidity is acceptable for signal execution.", ["Slippage risk is reduced versus thin tape."])},
        "THIN": {"short": variants_short("Liquidity is thin.", CUES_RISK), "institutional": variants_institutional("Thin liquidity degrades signal quality.", ["Reduce size; demand confirmation."])},
        "POOR": {"short": variants_short("Liquidity is poor.", CUES_RISK), "institutional": variants_institutional("Poor liquidity forces HOLD to reduce slippage/noise.", ["Wait for liquidity improvement."])},
        "UNKNOWN": {"short": ["Liquidity unavailable."], "institutional": ["Liquidity classification missing."]},
    }
)

_add(
    "market_regime",
    "Market Regime",
    "str",
    "Regime classification: TRENDING/RANGING/HIGH_VOL/UNKNOWN.",
    states={
        "TRENDING": {"short": variants_short("Regime is trending.", ["Trend strategies work better.", "Pullbacks are actionable.", "Use structure."]), "institutional": variants_institutional("Trending regime supports continuation trades.", ["Prefer pullbacks that hold trend baselines."])},
        "RANGING": {"short": variants_short("Regime is ranging.", ["Mean reversion common.", "Breakouts need volume.", "Chop risk."]), "institutional": variants_institutional("Ranging regime supports mean-reversion.", ["Demand strong confirmation for breakouts."])},
        "HIGH_VOL": {"short": variants_short("Regime is high volatility.", CUES_RISK), "institutional": variants_institutional("High-vol regime degrades signals.", ["Reduce size; wait for stabilization."])},
        "UNKNOWN": {"short": ["Regime unknown."], "institutional": ["Regime classification unavailable."]},
    }
)

_add(
    "feature_consensus",
    "Feature Consensus",
    "num",
    "Consensus score across trend/momentum/volume votes.",
    states={
        "BULLISH": {"short": variants_short("Indicators show bullish consensus.", CUES_CONFIRM), "institutional": variants_institutional("Consensus is bullish.", ["Best when aligned with structure and participation."])},
        "MIXED": {"short": variants_short("Indicators are mixed; consensus is weak.", CUES_RISK), "institutional": variants_institutional("Mixed consensus implies low edge.", ["HOLD is often optimal in mixed regimes."])},
        "BEARISH": {"short": variants_short("Indicators show bearish consensus.", CUES_CONFIRM), "institutional": variants_institutional("Consensus is bearish.", ["Downside continuation improves if support fails."])},
        "UNKNOWN": {"short": ["Consensus unavailable."], "institutional": ["Consensus metric missing."]},
    }
)

_add(
    "directional_pressure",
    "Directional Pressure",
    "num",
    "Pressure score from returns, MACD histogram, OBV slope.",
    states={
        "BULLISH": {"short": variants_short("Directional pressure is bullish.", CUES_CONFIRM), "institutional": variants_institutional("Pressure favors upside continuation.", ["Confirm with structure/volume."])},
        "MIXED": {"short": variants_short("Directional pressure is mixed.", CUES_RISK), "institutional": variants_institutional("Pressure is mixed; follow-through is uncertain.", ["Wait for clarity."])},
        "BEARISH": {"short": variants_short("Directional pressure is bearish.", CUES_CONFIRM), "institutional": variants_institutional("Pressure favors downside continuation.", ["Watch support; breakdown confirms."])},
        "UNKNOWN": {"short": ["Pressure unavailable."], "institutional": ["Pressure metric missing."]},
    }
)

_add(
    "fragility_index",
    "Fragility Index",
    "num",
    "Fragility score; high fragility implies instability/whipsaw risk.",
    states={
        "LOW": {"short": variants_short("Setup fragility is low.", ["Cleaner behavior.", "Less whipsaw.", "Signals are more reliable."]), "institutional": variants_institutional("Low fragility supports cleaner execution.", ["Still confirm structure."])},
        "MODERATE": {"short": variants_short("Setup fragility is moderate.", CUES_CONFIRM), "institutional": variants_institutional("Moderate fragility; demand confirmation.", ["Reduce size or wait for clarity."])},
        "ELEVATED": {"short": variants_short("Setup fragility is elevated.", CUES_RISK), "institutional": variants_institutional("Elevated fragility increases whipsaw risk.", ["Prefer HOLD or smaller sizing."])},
        "HIGH": {"short": variants_short("Setup fragility is high.", CUES_RISK), "institutional": variants_institutional("High fragility degrades signals.", ["HOLD is typically optimal until structure stabilizes."])},
        "UNKNOWN": {"short": ["Fragility unavailable."], "institutional": ["Fragility metric missing."]},
    }
)

_add(
    "expected_value",
    "Expected Value",
    "num",
    "Expected value score derived from win rate, average return, and fragility penalty.",
    states={
        "NEGATIVE": {"short": variants_short("Expected value is negative.", CUES_RISK), "institutional": variants_institutional("EV is non-positive; edge is unfavorable.", ["Wait for better expectancy setups."])},
        "LOW": {"short": variants_short("Expected value is low.", CUES_CONFIRM), "institutional": variants_institutional("EV is marginal; require strong confirmation.", ["Avoid forcing trades."])},
        "GOOD": {"short": variants_short("Expected value is good.", ["Edge is favorable.", "Execution matters.", "Use structure."]), "institutional": variants_institutional("EV supports the thesis if structure holds.", ["Prefer risk-defined execution."])},
        "HIGH": {"short": variants_short("Expected value is high.", ["Strong edge.", "Still manage risk.", "Avoid over-sizing."]), "institutional": variants_institutional("High EV supports conviction, conditioned on regime/structure.", ["Risk control remains essential."])},
        "UNKNOWN": {"short": ["EV unavailable."], "institutional": ["EV metric missing."]},
    }
)

_add(
    "rarity",
    "Signal Rarity",
    "num",
    "Rarity of the pattern/signal occurrence versus total days; rarer can imply higher selectivity.",
    states={
        "RARE": {"short": variants_short("Setup is rare.", ["Selective signal.", "Potentially higher quality.", "Confirm."]), "institutional": variants_institutional("Rare setups can carry higher edge when confirmed.", ["Confirm structure/volume; avoid forcing."])},
        "SELECTIVE": {"short": variants_short("Setup is selective.", CUES_CONFIRM), "institutional": variants_institutional("Selectivity supports quality, but confirmation is required.", ["Use structure."])},
        "COMMON": {"short": variants_short("Setup is common.", CUES_RISK), "institutional": variants_institutional("Common setups often carry lower edge.", ["Require stronger confirmation and tighter risk rules."])},
        "UNKNOWN": {"short": ["Rarity unavailable."], "institutional": ["Rarity metric missing."]},
    }
)

_add(
    "pattern_winRate_5d",
    "Pattern Win Rate (5d)",
    "num",
    "Historical win rate for the detected pattern over 5 days.",
    states={
        "GOOD": {"short": variants_short("Pattern win rate is strong.", ["Supports edge.", "Still confirm.", "Use structure."]), "institutional": variants_institutional("Historical win rate meets quality thresholds.", ["Do not treat as a signal alone; confirm structure."])},
        "OK": {"short": variants_short("Pattern win rate is moderate.", CUES_CONFIRM), "institutional": variants_institutional("Win rate is acceptable but not dominant.", ["Use as context; confirmation required."])},
        "WEAK": {"short": variants_short("Pattern win rate is weak.", CUES_RISK), "institutional": variants_institutional("Pattern edge is weak historically.", ["Treat pattern as context only."])},
        "UNKNOWN": {"short": ["Pattern win rate unavailable."], "institutional": ["Pattern stats missing."]},
    }
)

_add(
    "pattern_avg_5d",
    "Pattern Avg Return (5d)",
    "num",
    "Historical average forward return for the pattern over 5 days.",
    states={
        "POSITIVE": {"short": variants_short("Pattern average return is positive.", CUES_CONFIRM), "institutional": variants_institutional("Average forward return is positive; edge is constructive.", ["Confirm with structure; beware regime shifts."])},
        "FLAT": {"short": variants_short("Pattern average return is flat.", CUES_RISK), "institutional": variants_institutional("Average return is near flat; edge is limited.", ["Treat as context."])},
        "NEGATIVE": {"short": variants_short("Pattern average return is negative.", CUES_RISK), "institutional": variants_institutional("Average return is negative; edge is unfavorable.", ["Treat as context; avoid biasing entries."])},
        "UNKNOWN": {"short": ["Pattern avg return unavailable."], "institutional": ["Pattern stats missing."]},
    }
)

_add(
    "pattern_samples_5d",
    "Pattern Samples (5d)",
    "num",
    "Number of historical samples; higher counts are more reliable.",
    states={
        "HIGH": {"short": variants_short("Pattern sample size is high.", ["Stats are more reliable.", "Still context.", "Confirm."]), "institutional": variants_institutional("Large sample size improves reliability of stats.", ["Still not a signal alone."])},
        "MEDIUM": {"short": variants_short("Pattern sample size is moderate.", CUES_CONFIRM), "institutional": variants_institutional("Moderate sample size; treat as supportive context.", ["Prefer confirmation."])},
        "LOW": {"short": variants_short("Pattern sample size is low.", CUES_RISK), "institutional": variants_institutional("Small sample size reduces reliability.", ["Do not overweight pattern stats."])},
        "UNKNOWN": {"short": ["Pattern sample size unavailable."], "institutional": ["Pattern stats missing."]},
    }
)
