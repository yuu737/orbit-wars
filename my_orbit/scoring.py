"""Candidate scoring for V8."""
from __future__ import annotations

from typing import Iterable, List

from .world import World
from .projection import Projection
from .candidates import Candidate


def score_candidate(world: World, projection: Projection, cand: Candidate) -> float:
    target = next((p for p in world.planets if p.id == cand.target_id), None)
    if target is None:
        return float("-inf")

    value = target.production * 120.0 - target.ships * 2.0
    if target.owner >= 0 and target.owner != world.player_id:
        value += target.production * 80.0
    value -= cand.eta * 3.0
    value -= cand.ships * 0.8
    return value


def score_candidates(world: World, projection: Projection, candidates: Iterable[Candidate]) -> List[Candidate]:
    out: List[Candidate] = []
    for c in candidates:
        c.base_score = score_candidate(world, projection, c)
        out.append(c)
    out.sort(key=lambda c: c.base_score, reverse=True)
    return out
