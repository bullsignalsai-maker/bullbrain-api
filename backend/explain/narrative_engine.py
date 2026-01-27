from typing import Dict, List, Optional

from backend.explain.indicator_templates import INDICATOR_TEMPLATES
from backend.explain.screen_specs import SCREEN_SPECS


def _dedupe_lines(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for line in lines:
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


# ------------------------------------------------------------
# Deterministic template selector
# ------------------------------------------------------------
def _select_template(
    indicator: str,
    state: str,
    seed: Optional[int] = None
) -> Optional[str]:

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


def narrate_indicator(
    indicator: str,
    state: str,
    *,
    seed: Optional[int] = None
) -> Optional[str]:

    if not indicator or not state:
        return None

    return _select_template(indicator, state, seed)


# ------------------------------------------------------------
# Screen-level narration
# ------------------------------------------------------------
def build_screen_narrative(
    *,
    screen: str,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Dict[str, List[str]]:

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

            text = narrate_indicator(indicator, state, seed=seed)
            if text:
                lines.append(text)

        lines = _dedupe_lines(lines)
        if lines:
            output[section] = lines

    return output


# ------------------------------------------------------------
# High-level summary (HEADER / WATCHLIST)
# ------------------------------------------------------------
def build_summary_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None,
    max_sentences: int = 2
) -> str:
    """
    Causal explanation — why this signal exists.
    """

    priority = [
        "momentum_composite",
        "probability_composite",
    ]

    lines: List[str] = []

    for indicator in priority:
        state = indicator_states.get(indicator)
        if not state:
            continue

        text = narrate_indicator(indicator, state, seed=seed)
        if text:
            lines.append(text)

    lines = _dedupe_lines(lines)
    return " ".join(lines[:max_sentences])


# ------------------------------------------------------------
# Signal justification (HOLD / blocker only)
# ------------------------------------------------------------
def build_signal_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:

    blocker = indicator_states.get("action_blocker")
    if not blocker or blocker == "NO_BLOCKER":
        return None

    return narrate_indicator("action_blocker", blocker, seed=seed)


# ------------------------------------------------------------
# Probability narration (single source of truth)
# ------------------------------------------------------------
def build_probability_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:

    state = indicator_states.get("probability_composite")
    if not state:
        return None

    return narrate_indicator("probability_composite", state, seed=seed)


# ------------------------------------------------------------
# Trade posture narration
# ------------------------------------------------------------
def build_trade_idea_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:

    priority = [
        "trend_strength_20",
        "volatility_composite",
        "liquidity_quality",
    ]

    lines: List[str] = []

    for indicator in priority:
        state = indicator_states.get(indicator)
        if not state:
            continue

        text = narrate_indicator(indicator, state, seed=seed)
        if text:
            lines.append(text)

    lines = _dedupe_lines(lines)
    return " ".join(lines[:2]) if lines else None


# ------------------------------------------------------------
# Pattern explanation (supporting, never dominant)
# ------------------------------------------------------------
def build_pattern_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:

    indicators = [
        "pattern_edge_5d",
        "pattern_winrate_5d",
        "pattern_occurrences",
    ]

    lines: List[str] = []

    for ind in indicators:
        state = indicator_states.get(ind)
        if not state:
            continue

        text = narrate_indicator(ind, state, seed=seed)
        if text:
            lines.append(text)

    lines = _dedupe_lines(lines)
    return " ".join(lines[:2]) if lines else None


# ------------------------------------------------------------
# Full bundle (Stock Detail)
# ------------------------------------------------------------
def build_full_narrative_bundle(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Dict[str, object]:

    return {
        "summary": build_summary_narrative(
            indicator_states=indicator_states,
            seed=seed,
            max_sentences=2
        ),
        "signal": build_signal_narrative(
            indicator_states=indicator_states,
            seed=seed
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
        ),
        "pattern": build_pattern_narrative(
            indicator_states=indicator_states,
            seed=seed
        ),
    }
