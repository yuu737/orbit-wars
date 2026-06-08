"""Short-horizon board projection.

V8.0 starts with a conservative projection: planet production over time.
Fleet-arrival simulation will be added after the parser is verified against
local Orbit Wars observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .world import World


@dataclass
class PlanetProjection:
    planet_id: int
    owners: List[int]
    ships: List[float]


@dataclass
class Projection:
    horizon: int
    planets: Dict[int, PlanetProjection]

    def owner_at(self, planet_id: int, turn: int) -> int:
        p = self.planets[planet_id]
        return p.owners[min(max(turn, 0), self.horizon)]

    def ships_at(self, planet_id: int, turn: int) -> float:
        p = self.planets[planet_id]
        return p.ships[min(max(turn, 0), self.horizon)]


def project_world(world: World, horizon: int = 20) -> Projection:
    """Project owner/ships for each planet.

    Current version:
    - owned planets gain production every turn
    - neutral planets do not grow
    - fleet arrivals are not applied yet

    This is intentionally simple so V8.0 can match V7 behavior first.
    """
    horizon = max(0, int(horizon))
    planets: Dict[int, PlanetProjection] = {}
    for p in world.planets:
        owners: List[int] = []
        ships: List[float] = []
        for t in range(horizon + 1):
            owners.append(p.owner)
            grown = p.ships + (p.production * t if p.owner >= 0 else 0.0)
            ships.append(max(0.0, float(grown)))
        planets[p.id] = PlanetProjection(planet_id=p.id, owners=owners, ships=ships)
    return Projection(horizon=horizon, planets=planets)
