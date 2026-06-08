"""Capture calculations.

V8.0 provides a simple capture floor. V8.1+ should replace this with arrival-time
projection that accounts for production and incoming fleets.
"""
from __future__ import annotations

from .world import Planet, World
from .projection import Projection


def capture_floor(target: Planet, world: World, projection: Projection | None = None, eta: int = 0) -> float:
    """Ships needed to capture or reinforce a target at arrival time."""
    if projection is not None and target.id in projection.planets:
        defender_ships = projection.ships_at(target.id, int(max(0, eta)))
        owner = projection.owner_at(target.id, int(max(0, eta)))
    else:
        defender_ships = target.ships
        owner = target.owner

    if owner == world.player_id:
        return 0.0
    return max(1.0, defender_ships + 1.0)
