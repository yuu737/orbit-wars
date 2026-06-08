"""V8 world/geometry helpers split from V6.9.

Exact-behavior extraction layer; do not tune logic here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

MAX_SPEED = 6.0

CENTER_X = 50.0
CENTER_Y = 50.0

SUN_RADIUS = 10.0
SUN_MARGIN = 1.0

END_STEP = 500


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 18
    defense_horizon: int = 18
    max_sources: int = 10
    max_targets: int = 12
    max_actions: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: int = 4
    regroup_enabled: bool = True
    regroup_distance: float = 7.0
    regroup_threshold: float = 9.0
    reserve_margin: int = 2


CONFIG_2P = PlannerConfig(
    roi_threshold=2.2,
    defense_horizon=26,
)


CONFIG_4P = PlannerConfig(
    horizon=13,
    defense_horizon=16,
    max_sources=6,
    max_targets=10,
    max_actions=5,
    roi_threshold=2.2,
    regroup_distance=6.0,
    regroup_threshold=11.0,
    reserve_margin=3,
)


@dataclass
class Projection:
    owner_by_id: dict[int, list[int]]
    ships_by_id: dict[int, list[int]]
    incoming_by_id: dict[int, list[dict[int, int]]]
    first_loss_turn_by_id: dict[int, int | None]


@dataclass
class Candidate:
    kind: str
    source_id: int
    target_id: int
    angle: float
    ships: int
    eta: float
    score: float


@dataclass
class MultiCandidate:
    kind: str
    target_id: int
    orders: list[Candidate]
    eta: float
    score: float


def fleet_speed(ships):
    ships = max(1, int(ships))
    if ships == 1:
        return 1.0
    scaled = math.log(ships) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * (scaled ** 1.5)


def distance_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def distance(a, b):
    return distance_xy(a.x, a.y, b.x, b.y)


def angle_diff(a, b):
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def infer_num_players(raw_initial_planets, raw_planets, raw_fleets, player):
    owners = {int(player)}

    for raw in raw_initial_planets:
        owner = int(raw[1])
        if owner >= 0:
            owners.add(owner)

    for raw in raw_planets:
        owner = int(raw[1])
        if owner >= 0:
            owners.add(owner)

    for raw in raw_fleets:
        owner = int(raw[1])
        if owner >= 0:
            owners.add(owner)

    return 2 if len(owners) <= 2 else 4


def is_rotating(initial_planet):
    orbital_radius = distance_xy(
        initial_planet.x,
        initial_planet.y,
        CENTER_X,
        CENTER_Y,
    )
    return orbital_radius + initial_planet.radius < 50.0


def predicted_planet_position(
    planet,
    initial_planet,
    step,
    angular_velocity,
    comet_ids,
):
    if planet.id in comet_ids or initial_planet is None or not is_rotating(initial_planet):
        return planet.x, planet.y

    dx = initial_planet.x - CENTER_X
    dy = initial_planet.y - CENTER_Y
    radius = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) + angular_velocity * step

    return (
        CENTER_X + radius * math.cos(angle),
        CENTER_Y + radius * math.sin(angle),
    )


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay

    if dx == 0.0 and dy == 0.0:
        return distance_xy(px, py, ax, ay)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    return distance_xy(px, py, ax + t * dx, ay + t * dy)


def crosses_sun(source, target_x, target_y):
    return (
        point_to_segment_distance(
            CENTER_X,
            CENTER_Y,
            source.x,
            source.y,
            target_x,
            target_y,
        )
        <= SUN_RADIUS + SUN_MARGIN
    )


def estimate_arrival_turns(source, target_x, target_y, ships):
    return distance_xy(source.x, source.y, target_x, target_y) / fleet_speed(ships)


def predict_intercept_position(
    source,
    target,
    initial_planet,
    current_step,
    angular_velocity,
    comet_ids,
    ships,
):
    target_x, target_y = target.x, target.y
    arrival_turns = estimate_arrival_turns(source, target_x, target_y, ships)

    for _ in range(3):
        future_step = current_step + arrival_turns
        target_x, target_y = predicted_planet_position(
            target,
            initial_planet,
            future_step,
            angular_velocity,
            comet_ids,
        )
        arrival_turns = estimate_arrival_turns(
            source,
            target_x,
            target_y,
            ships,
        )

    return target_x, target_y, arrival_turns


def validate_intercept_window(
    source,
    target,
    initial_planet,
    current_step,
    angular_velocity,
    comet_ids,
    ships,
    target_x,
    target_y,
    eta,
):
    speed = fleet_speed(max(1, ships))
    direction = math.atan2(target_y - source.y, target_x - source.x)
    best_gap = float("inf")

    for dt in (-1.5, -0.75, 0.0, 0.75, 1.5):
        probe_eta = max(0.0, eta + dt)

        fx = source.x + math.cos(direction) * speed * probe_eta
        fy = source.y + math.sin(direction) * speed * probe_eta

        px, py = predicted_planet_position(
            target,
            initial_planet,
            current_step + probe_eta,
            angular_velocity,
            comet_ids,
        )

        best_gap = min(best_gap, distance_xy(fx, fy, px, py) - target.radius)

    return best_gap <= 0.85


def attack_ambiguity_penalty(
    source,
    target,
    planets,
    initial_planets,
    current_step,
    angular_velocity,
    comet_ids,
    target_x,
    target_y,
):
    target_heading = math.atan2(target_y - source.y, target_x - source.x)
    target_dist = distance_xy(source.x, source.y, target_x, target_y)
    worst = 0.0

    for other in planets:
        if other.id in (source.id, target.id):
            continue

        ox, oy = predicted_planet_position(
            other,
            initial_planets.get(other.id),
            current_step,
            angular_velocity,
            comet_ids,
        )

        other_dist = distance_xy(source.x, source.y, ox, oy)
        if other_dist >= target_dist - 0.5:
            continue

        other_heading = math.atan2(oy - source.y, ox - source.x)
        heading_gap = angle_diff(target_heading, other_heading)
        lane_width = math.atan2(other.radius + 0.35, max(1.0, other_dist))
        overlap = lane_width - heading_gap

        if overlap <= 0.0:
            continue

        penalty = overlap * 210.0 + max(0.0, target_dist - other_dist) * 0.45

        if other.owner == -1:
            penalty *= 0.9
        elif other.owner == source.owner:
            penalty *= 0.7
        else:
            penalty *= 1.15

        worst = max(worst, penalty)

    return worst


def fleet_points_toward_planet(fleet, planet):
    heading = math.atan2(planet.y - fleet.y, planet.x - fleet.x)
    dist = distance_xy(fleet.x, fleet.y, planet.x, planet.y)
    tolerance = 0.22 + min(0.30, planet.radius / max(8.0, dist))

    return angle_diff(fleet.angle, heading) <= tolerance


def fleet_eta_to_planet(fleet, planet):
    return distance_xy(fleet.x, fleet.y, planet.x, planet.y) / fleet_speed(fleet.ships)