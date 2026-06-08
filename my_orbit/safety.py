"""Safety calculations such as safe_drain.

These functions are small and explicit first. Later versions can account for
incoming enemy fleets and moving planets.
"""
from __future__ import annotations

from .world import Planet, World
from .projection import Projection


def reserve_for_planet(planet: Planet, world: World, projection: Projection | None = None) -> float:
    """Return minimum ships to leave on a source planet.

    Initial heuristic:
    - home/high-production planets keep more reserve
    - low production planets can drain more aggressively
    """
    base = 3.0
    prod_reserve = max(0.0, planet.production) * 1.5
    high_value = 6.0 if planet.production >= 4 else 0.0
    return base + prod_reserve + high_value


def safe_drain(planet: Planet, world: World, projection: Projection | None = None) -> float:
    """Maximum ships that can be launched without emptying the source."""
    reserve = reserve_for_planet(planet, world, projection)
    return max(0.0, planet.ships - reserve)
