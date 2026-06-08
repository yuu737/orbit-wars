"""Candidate generation for V8.

Candidate objects are deliberately simple so we can inspect and log them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math

from .world import Planet, World, distance
from .projection import Projection
from .safety import safe_drain
from .capture import capture_floor


@dataclass
class Candidate:
    source_id: int
    target_id: int
    ships: float
    eta: float
    kind: str
    base_score: float = 0.0


def estimate_eta(source: Planet, target: Planet, ships: float) -> float:
    # Placeholder speed model. V8.1 should replace this with real game speed.
    speed = max(1.0, math.sqrt(max(1.0, ships)))
    return distance(source, target) / speed


def generate_candidates(world: World, projection: Projection, max_targets_per_source: int = 8) -> List[Candidate]:
    candidates: List[Candidate] = []
    targets = world.neutral_planets + world.enemy_planets
    for src in world.my_planets:
        budget = safe_drain(src, world, projection)
        if budget < 1.0:
            continue
        ranked_targets = sorted(
            targets,
            key=lambda t: (-(t.production * 20.0 - t.ships), distance(src, t)),
        )[:max_targets_per_source]
        for tgt in ranked_targets:
            eta = estimate_eta(src, tgt, budget)
            need = capture_floor(tgt, world, projection, eta=int(math.ceil(eta)))
            if budget >= need:
                kind = "attack" if tgt.owner >= 0 else "expand"
                candidates.append(Candidate(src.id, tgt.id, min(budget, need + 2.0), eta, kind))
    return candidates
