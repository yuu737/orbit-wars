"""
Orbit Wars - Value Expansion Agent

This version upgrades the starter bot with three simple ideas:
  1. Prefer planets with better production payoff, not just nearest distance.
  2. Keep a minimum garrison on owned planets instead of launching everything.
  3. Avoid overcommitting multiple planets onto the same target.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


MAX_SPEED = 6.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0


def fleet_speed(ships):
    ships = max(1, ships)
    if ships == 1:
        return 1.0

    scaled = math.log(ships) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * (scaled ** 1.5)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def distance_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def ships_needed_to_capture(target):
    return target.ships + 1


def reserve_ships(planet):
    # Keep more ships on stronger planets so we do not instantly lose them after expanding.
    return max(6, planet.production * 3)


def dynamic_reserve_ships(planet, current_step):
    if current_step < 80:
        return max(3, planet.production * 2)
    if current_step < 160:
        return max(5, planet.production * 2)
    return reserve_ships(planet)


def is_rotating(planet):
    orbital_radius = distance_xy(planet.x, planet.y, CENTER_X, CENTER_Y)
    return orbital_radius + planet.radius < 50.0


def predicted_planet_position(planet, initial_planet, step, angular_velocity, comet_ids):
    if planet.id in comet_ids or initial_planet is None or not is_rotating(initial_planet):
        return planet.x, planet.y

    dx = initial_planet.x - CENTER_X
    dy = initial_planet.y - CENTER_Y
    angle = math.atan2(dy, dx) + angular_velocity * step
    radius = math.hypot(dx, dy)
    return (
        CENTER_X + radius * math.cos(angle),
        CENTER_Y + radius * math.sin(angle),
    )


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return distance_xy(px, py, ax, ay)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return distance_xy(px, py, closest_x, closest_y)


def crosses_sun(source, target_x, target_y):
    return point_to_segment_distance(CENTER_X, CENTER_Y, source.x, source.y, target_x, target_y) <= (
        SUN_RADIUS + SUN_MARGIN
    )


def estimate_arrival_turns(source, target_x, target_y, ships):
    dist = distance_xy(source.x, source.y, target_x, target_y)
    return dist / fleet_speed(ships)


def predict_intercept_position(source, target, initial_planet, current_step, angular_velocity, comet_ids, ships):
    target_x, target_y = target.x, target.y
    arrival_turns = estimate_arrival_turns(source, target_x, target_y, ships)

    for _ in range(3):
        future_step = current_step + arrival_turns
        target_x, target_y = predicted_planet_position(
            target, initial_planet, future_step, angular_velocity, comet_ids
        )
        arrival_turns = estimate_arrival_turns(source, target_x, target_y, ships)

    return target_x, target_y, arrival_turns


def target_score(source, target, committed_ships, current_step, angular_velocity, initial_planet, comet_ids):
    needed = ships_needed_to_capture(target) - committed_ships
    if needed <= 0:
        return float("-inf")

    target_x, target_y, travel_time = predict_intercept_position(
        source, target, initial_planet, current_step, angular_velocity, comet_ids, needed
    )
    dist = distance_xy(source.x, source.y, target_x, target_y)

    value = target.production * 18.0
    capture_cost = needed * 1.6
    time_cost = travel_time * 2.5
    enemy_penalty = 14.0 if target.owner != -1 else 0.0
    sun_penalty = 40.0 if crosses_sun(source, target_x, target_y) else 0.0

    if target.owner == -1 and current_step < 80:
        value += 18.0
        if target.production >= 4:
            value += 30.0
        time_cost *= 0.8

    if target.owner == -1 and current_step < 140 and dist < 28:
        value += 12.0

    return value - capture_cost - time_cost - enemy_penalty - sun_penalty


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_initial_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else obs.angular_velocity
    current_step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    comet_ids = set(obs.get("comet_planet_ids", [])) if isinstance(obs, dict) else set(obs.comet_planet_ids)
    planets = [Planet(*p) for p in raw_planets]
    initial_planets = {planet.id: planet for planet in (Planet(*p) for p in raw_initial_planets)}

    my_planets = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    target_commits = {}
    moves = []

    if not my_planets or not targets:
        return moves

    for mine in sorted(my_planets, key=lambda planet: planet.ships, reverse=True):
        available = mine.ships - dynamic_reserve_ships(mine, current_step)
        if available <= 0:
            continue

        best_target = None
        best_needed = None
        best_score = float("-inf")

        for target in targets:
            committed = target_commits.get(target.id, 0)
            needed = ships_needed_to_capture(target) - committed
            if needed <= 0 or needed > available:
                continue

            initial_target = initial_planets.get(target.id)
            score = target_score(
                mine,
                target,
                committed,
                current_step,
                angular_velocity,
                initial_target,
                comet_ids,
            )
            if score > best_score:
                best_score = score
                best_target = target
                best_needed = needed

        early_expand = current_step < 80 and best_target is not None and best_target.owner == -1
        if best_target is None or (best_score <= 0 and not early_expand):
            continue

        target_x, target_y, _ = predict_intercept_position(
            mine,
            best_target,
            initial_planets.get(best_target.id),
            current_step,
            angular_velocity,
            comet_ids,
            best_needed,
        )
        if crosses_sun(mine, target_x, target_y):
            continue

        angle = math.atan2(target_y - mine.y, target_x - mine.x)
        moves.append([mine.id, angle, int(best_needed)])
        target_commits[best_target.id] = target_commits.get(best_target.id, 0) + best_needed

    return moves
