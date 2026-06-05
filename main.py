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


def fleet_speed(ships):
    ships = max(1, ships)
    if ships == 1:
        return 1.0

    scaled = math.log(ships) / math.log(1000)
    return 1.0 + (MAX_SPEED - 1.0) * (scaled ** 1.5)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def ships_needed_to_capture(target):
    return target.ships + 1


def reserve_ships(planet):
    # Keep more ships on stronger planets so we do not instantly lose them after expanding.
    return max(6, planet.production * 3)


def target_score(source, target, committed_ships):
    needed = ships_needed_to_capture(target) - committed_ships
    if needed <= 0:
        return float("-inf")

    dist = distance(source, target)
    travel_time = dist / fleet_speed(needed)

    value = target.production * 18.0
    capture_cost = needed * 1.6
    time_cost = travel_time * 2.5
    enemy_penalty = 14.0 if target.owner != -1 else 0.0

    return value - capture_cost - time_cost - enemy_penalty


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    target_commits = {}
    moves = []

    if not my_planets or not targets:
        return moves

    for mine in sorted(my_planets, key=lambda planet: planet.ships, reverse=True):
        available = mine.ships - reserve_ships(mine)
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

            score = target_score(mine, target, committed)
            if score > best_score:
                best_score = score
                best_target = target
                best_needed = needed

        if best_target is None or best_score <= 0:
            continue

        angle = math.atan2(best_target.y - mine.y, best_target.x - mine.x)
        moves.append([mine.id, angle, int(best_needed)])
        target_commits[best_target.id] = target_commits.get(best_target.id, 0) + best_needed

    return moves
