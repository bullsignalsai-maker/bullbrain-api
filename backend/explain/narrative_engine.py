# backend/explain/narrative_engine.py
# ============================================================
# BullSignalsAI — Narrative Engine
#
# Purpose:
# - Deterministically generate institutional-quality narration
# - Reusable across cron + API
# - Zero randomness
# - Zero Firestore access
#
# Dependencies:
# - indicator_states.py   → provides indicator -> STATE
# - indicator_templates.py → provides indicator -> state -> text[]
# - screen_specs.py        → defines screen layouts
#
# ============================================================

from typing import Dict, List, Optional

from backend.explain.indicator_templates import INDICATOR_TEMPLATES
from backend.explain.screen_specs import SCREEN_SPECS


# ------------------------------------------------------------
# Deterministic template selector
# ------------------------------------------------------------
def _select_template(
    indicator: str,
    state: str,
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Deterministically select a narration template.

    Rules:
    - No randomness
    - Same (indicator, state, seed) → same output
    - If seed is None → always choose first template
    """

    state_map = INDICATOR_TEMPLATES.get(indicator)
    if not state_map:
        return None

    templates = state_map.get(state)
    if not templates:
        return None

    if seed is None:
        return templates[0]

    idx = abs(hash(f"{indicator}:{state}:{seed}")) % len(templates)
    return templates[idx]


# ------------------------------------------------------------
# Narrate a single indicator
# ------------------------------------------------------------
def narrate_indicator(
    indicator: str,
    state: str,
    *,
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Generate narration for one indicator-state pair.
    """

    if not indicator or not state:
        return None

    return _select_template(
        indicator=indicator,
        state=state,
        seed=seed
    )


# ------------------------------------------------------------
# Build narration for a screen (sectioned)
# ------------------------------------------------------------
def build_screen_narrative(
    *,
    screen: str,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Build narration grouped by screen sections.

    Output shape:
    {
      "momentum": [str, str],
      "trend": [str],
      "volume": [str],
      ...
    }
    """

    spec = SCREEN_SPECS.get(screen)
    if not spec:
        return {}

    output: Dict[str, List[str]] = {}

    for section, indicators in spec.items():
        lines: List[str] = []

        for indicator in indicators:
            state = indicator_states.get(indicator)
            if not state:
                continue

            text = narrate_indicator(
                indicator=indicator,
                state=state,
                seed=seed
            )
            if text:
                lines.append(text)

        if lines:
            output[section] = lines

    return output


# ------------------------------------------------------------
# Build concise summary (signal pill / header)
# ------------------------------------------------------------
def build_summary_narrative(
    *,
    indicators: List[str],
    indicator_states: Dict[str, str],
    seed: Optional[int] = None,
    max_sentences: int = 3
) -> str:
    """
    Build short professional summary.
    Used near BUY / SELL / HOLD signal.
    """

    lines: List[str] = []

    for indicator in indicators:
        state = indicator_states.get(indicator)
        if not state:
            continue

        text = narrate_indicator(
            indicator=indicator,
            state=state,
            seed=seed
        )
        if text:
            lines.append(text)

        if len(lines) >= max_sentences:
            break

    return " ".join(lines)


# ------------------------------------------------------------
# Probability / bias narration
# ------------------------------------------------------------
def build_probability_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Dedicated narration for probability bar / bias meter.
    """

    state = indicator_states.get("hybrid_prob_up")
    if not state:
        return None

    return narrate_indicator(
        indicator="hybrid_prob_up",
        state=state,
        seed=seed
    )


# ------------------------------------------------------------
# Trade idea narration
# ------------------------------------------------------------
def build_trade_idea_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Builds a 1–2 sentence trade idea explanation.
    """

    priority_indicators = [
        "trend_strength_20",
        "rsi14",
        "macd_hist",
        "volatility_20d",
        "volume_vs_ma20_pct"
    ]

    lines: List[str] = []

    for indicator in priority_indicators:
        state = indicator_states.get(indicator)
        if not state:
            continue

        text = narrate_indicator(
            indicator=indicator,
            state=state,
            seed=seed
        )
        if text:
            lines.append(text)

        if len(lines) >= 2:
            break

    if not lines:
        return None

    return " ".join(lines)


# ------------------------------------------------------------
# Full Stock Detail narration bundle
# ------------------------------------------------------------
def build_full_narrative_bundle(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Dict[str, object]:
    """
    High-level orchestrator for Stock Detail screen.
    """

    return {
        "summary": build_summary_narrative(
            indicators=[
                "trend_strength_20",
                "rsi14",
                "macd_hist"
            ],
            indicator_states=indicator_states,
            seed=seed,
            max_sentences=2
        ),
        "sections": build_screen_narrative(
            screen="STOCK_DETAIL",
            indicator_states=indicator_states,
            seed=seed
        ),
        "probability": build_probability_narrative(
            indicator_states=indicator_states,
            seed=seed
        ),
        "tradeIdea": build_trade_idea_narrative(
            indicator_states=indicator_states,
            seed=seed
        )
    }
