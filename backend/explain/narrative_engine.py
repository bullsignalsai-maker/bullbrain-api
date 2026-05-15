from typing import Any, Dict, List, Optional

from backend.explain.indicator_templates import INDICATOR_TEMPLATES
from backend.explain.screen_specs import SCREEN_SPECS
from backend.explain.indicator_library import INDICATOR_LIBRARY


def _dedupe_lines(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for line in lines:
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _select_template(
    indicator: str,
    state: str,
    *,
    seed: Optional[int] = None
) -> Optional[str]:
    """Select template with fallback to rich library"""
    # Primary: indicator_templates.py
    state_map = INDICATOR_TEMPLATES.get(indicator)
    if state_map and state in state_map:
        templates = state_map.get(state, [])
        if templates:
            if seed is None:
                return templates[0]
            idx = abs(hash(f"{indicator}:{state}:{seed}")) % len(templates)
            return templates[idx]

    # Fallback: rich library
    lib = INDICATOR_LIBRARY.get(indicator)
    if lib and "states" in lib:
        state_data = lib["states"].get(state)
        if state_data and "short" in state_data:
            templates = state_data["short"]
            if templates:
                idx = abs(hash(f"{indicator}:{state}:{seed or 0}")) % len(templates)
                return templates[idx]

    return None


def narrate_indicator(
    indicator: str,
    state: str,
    *,
    seed: Optional[int] = None
) -> Optional[str]:
    if not indicator or not state:
        return None
    return _select_template(indicator, state, seed=seed)


# ================================================================
# NEW: Homescreen-Optimized Narrative (Phase 2)
# ================================================================
def build_homescreen_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> str:
    """Short, high-signal summary optimized for HomeScreen"""
    priority = [
        "momentum_composite",
        "probability_composite",
        "pattern_edge_5d",
        "trend_strength_20",
    ]

    lines: List[str] = []
    for ind in priority:
        state = indicator_states.get(ind)
        if not state or state == "UNKNOWN":
            continue
        text = narrate_indicator(ind, state, seed=seed)
        if text:
            lines.append(text)

    result = " ".join(lines[:2]).strip()
    return result if result else "Market conditions reflect balanced conviction with moderate edge."


# ================================================================
# Improved High-level Summary
# ================================================================
def build_summary_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None,
    max_sentences: int = 2
) -> str:
    priority = ["momentum_composite", "probability_composite", "trend_strength_20"]

    lines: List[str] = []
    for indicator in priority:
        state = indicator_states.get(indicator)
        if not state or state == "UNKNOWN":
            continue
        text = narrate_indicator(indicator, state, seed=seed)
        if text:
            lines.append(text)

    lines = _dedupe_lines(lines)
    return " ".join(lines[:max_sentences])


# ================================================================
# Other Functions (Kept + Minor Polish)
# ================================================================
def build_signal_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:
    blocker = indicator_states.get("action_blocker")
    if not blocker or blocker == "NO_BLOCKER":
        return None
    return narrate_indicator("action_blocker", blocker, seed=seed)


def build_probability_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:
    state = indicator_states.get("probability_composite")
    if not state:
        return None
    return narrate_indicator("probability_composite", state, seed=seed)


def build_trade_idea_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:
    priority = ["trend_strength_20", "volatility_composite", "liquidity_quality"]
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


def build_pattern_narrative(
    *,
    indicator_states: Dict[str, str],
    seed: Optional[int] = None
) -> Optional[str]:
    indicators = ["pattern_edge_5d", "pattern_winrate_5d"]
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


# ================================================================
# Full Bundle (Stock Detail + Homescreen)
# ================================================================
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
        "homescreen": build_homescreen_narrative(          # ← New for Phase 2
            indicator_states=indicator_states,
            seed=seed
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