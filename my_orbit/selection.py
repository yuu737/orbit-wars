"""Action selection for V8."""
from __future__ import annotations

from typing import Any, Dict, List

from .world import World
from .candidates import Candidate


def select_actions(world: World, candidates: List[Candidate], max_actions: int = 6) -> List[Dict[str, Any]]:
    """Greedy non-overlapping selection.

    Output format is intentionally generic. my_v8_main.py will adapt it to the
    exact Orbit Wars move format after the local observation/action format is fixed.
    """
    actions: List[Dict[str, Any]] = []
    used_sources: set[int] = set()
    for c in candidates:
        if len(actions) >= max_actions:
            break
        if c.source_id in used_sources:
            continue
        used_sources.add(c.source_id)
        actions.append({
            "source": c.source_id,
            "target": c.target_id,
            "ships": int(max(1, c.ships)),
            "kind": c.kind,
            "score": c.base_score,
        })
    return actions
