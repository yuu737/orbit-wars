"""V8 entry bot using the new my_orbit package.

This file is intentionally thin.  The real logic lives in ../my_orbit/.
Current V8.0 is a scaffold bot: it parses the world, projects a short future,
generates candidates, scores them, and converts the selected candidates into
Orbit Wars launch actions.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List

# bots/my_v8_main.py から orbit-wars/my_orbit を読めるようにする。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from my_orbit.world import World, parse_world
from my_orbit.projection import project_world
from my_orbit.candidates import generate_candidates
from my_orbit.scoring import score_candidates
from my_orbit.selection import select_actions


def _planet_by_id(world: World) -> Dict[int, Any]:
    return {int(p.id): p for p in world.planets}


def _angle_to_target(world: World, source_id: int, target_id: int) -> float:
    """Return launch angle in radians from source planet to target planet.

    V8.0 uses direct aiming.  Future versions should replace this with
    intercept aiming for moving planets.
    """
    planets = _planet_by_id(world)
    src = planets.get(int(source_id))
    tgt = planets.get(int(target_id))
    if src is None or tgt is None:
        return 0.0
    return math.atan2(float(tgt.y) - float(src.y), float(tgt.x) - float(src.x))


def _to_orbit_wars_actions(world: World, selected: List[Dict[str, Any]]) -> List[List[Any]]:
    """Convert my_orbit generic actions to Kaggle Orbit Wars move rows.

    Existing local bots use rows like [source_planet_id, angle, ships].
    """
    rows: List[List[Any]] = []
    source_ships = {int(p.id): float(p.ships) for p in world.my_planets}
    spent: Dict[int, int] = {}

    for action in selected:
        try:
            src = int(action["source"])
            tgt = int(action["target"])
            ships = int(action["ships"])
        except Exception:
            continue

        if ships <= 0:
            continue
        available = int(max(0.0, source_ships.get(src, 0.0))) - spent.get(src, 0)
        if available <= 0:
            continue
        ships = min(ships, available)
        if ships <= 0:
            continue

        angle = _angle_to_target(world, src, tgt)
        rows.append([src, angle, ships])
        spent[src] = spent.get(src, 0) + ships

    return rows


def agent(obs: Any) -> List[List[Any]]:
    """Kaggle agent entry point."""
    try:
        world = parse_world(obs)
        if not world.my_planets or not world.planets:
            return []

        projection = project_world(world, horizon=20)
        candidates = generate_candidates(world, projection, max_targets_per_source=8)
        scored = score_candidates(world, projection, candidates)
        selected = select_actions(world, scored, max_actions=6)
        return _to_orbit_wars_actions(world, selected)
    except Exception:
        # For Kaggle safety: never crash the agent.  A future debug build can log
        # exceptions locally, but submitted/evaluated bots should return no move.
        return []
