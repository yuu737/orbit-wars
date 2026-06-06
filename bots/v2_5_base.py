"""
Orbit Wars - Aggressive Balanced Agent

Design goals:
- Keep the proven V2.2 fast neutral expansion.
- Avoid the V2.4 problem: defense must not slow the opening.
- Add light threat awareness after the opening only.
- Prefer cheap high-production planets and safe shots that do not cross the sun.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


MAX_SPEED = 6.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
END_STEP = 500


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


def ships_needed_to_capture(target):
    return int(target.ships) + 1


def is_rotating(initial_planet):
    orbital_radius = distance_xy(initial_planet.x, initial_planet.y, CENTER_X, CENTER_Y)
    return orbital_radius + initial_planet.radius < 50.0


def predicted_planet_position(planet, initial_planet, step, angular_velocity, comet_ids):
    if planet.id in comet_ids or initial_planet is None or not is_rotating(initial_planet):
        return planet.x, planet.y

    dx = initial_planet.x - CENTER_X
    dy = initial_planet.y - CENTER_Y
    radius = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) + angular_velocity * step
    return CENTER_X + radius * math.cos(angle), CENTER_Y + radius * math.sin(angle)


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return distance_xy(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return distance_xy(px, py, ax + t * dx, ay + t * dy)


def crosses_sun(source, target_x, target_y):
    return point_to_segment_distance(
        CENTER_X, CENTER_Y, source.x, source.y, target_x, target_y
    ) <= SUN_RADIUS + SUN_MARGIN


def estimate_arrival_turns(source, target_x, target_y, ships):
    return distance_xy(source.x, source.y, target_x, target_y) / fleet_speed(ships)


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


def fleet_points_toward_planet(fleet, planet):
    heading = math.atan2(planet.y - fleet.y, planet.x - fleet.x)
    dist = distance_xy(fleet.x, fleet.y, planet.x, planet.y)
    # Wide enough to catch real threats, narrow enough to avoid random nearby fleets.
    tolerance = 0.22 + min(0.30, planet.radius / max(8.0, dist))
    return angle_diff(fleet.angle, heading) <= tolerance


def estimate_incoming_to_planet(planet, fleets, player):
    friendly = 0
    enemy = 0
    soon_enemy = 0

    for fleet in fleets:
        if not fleet_points_toward_planet(fleet, planet):
            continue

        dist = distance_xy(fleet.x, fleet.y, planet.x, planet.y)
        turns = dist / fleet_speed(fleet.ships)
        if turns > 42:
            continue

        ships = int(fleet.ships)
        if fleet.owner == player:
            friendly += ships
        else:
            enemy += ships
            if turns <= 18:
                soon_enemy += ships

    return friendly, enemy, soon_enemy


def base_reserve(planet, current_step):
    # V2.2-style aggressive reserve. This is intentionally low in the opening.
    if current_step < 90:
        return max(2, planet.production)
    if current_step < 160:
        return max(5, planet.production * 2)
    if current_step > 430:
        return max(10, planet.production * 4)
    return max(6, planet.production * 3)


def reserve_for_planet(planet, current_step, fleets, player):
    reserve = base_reserve(planet, current_step)

    # Do not let defense ruin the opening. Only protect against very nearby, obvious attacks.
    if current_step < 90:
        friendly, enemy, soon_enemy = estimate_incoming_to_planet(planet, fleets, player)
        if soon_enemy > 0:
            reserve = max(reserve, min(planet.ships, soon_enemy + 2 - friendly))
        return reserve

    friendly, enemy, soon_enemy = estimate_incoming_to_planet(planet, fleets, player)
    pressure = max(soon_enemy, int(enemy * 0.6))
    if pressure > 0:
        reserve = max(reserve, pressure + planet.production * 2 + 2 - friendly)

    return max(0, reserve)


def opening_target_score(source, target, current_step, angular_velocity, initial_planet, comet_ids, committed_ships):
    if target.owner != -1 or target.id in comet_ids:
        return float("-inf"), None

    needed = ships_needed_to_capture(target) - committed_ships
    if needed <= 0:
        return float("-inf"), None

    target_x, target_y, travel_time = predict_intercept_position(
        source, target, initial_planet, current_step, angular_velocity, comet_ids, needed
    )
    if crosses_sun(source, target_x, target_y):
        return float("-inf"), None

    dist = distance_xy(source.x, source.y, target_x, target_y)
    if dist > 40:
        return float("-inf"), None

    score = 0.0
    score += target.production * 34.0
    score -= needed * 2.45
    score -= travel_time * 3.1

    if target.production >= 4:
        score += 38.0
    if target.ships <= 12:
        score += 20.0
    if target.ships <= 7:
        score += 14.0
    if dist <= 22:
        score += 17.0
    if dist <= 14:
        score += 10.0

    # A very cheap low-production planet is still useful early if it is close.
    if target.production <= 2 and target.ships > 18:
        score -= 18.0

    return score, (target_x, target_y, needed)


def target_score(source, target, committed_ships, current_step, angular_velocity, initial_planet, comet_ids, player):
    needed = ships_needed_to_capture(target) - committed_ships
    if needed <= 0:
        return float("-inf")

    target_x, target_y, travel_time = predict_intercept_position(
        source, target, initial_planet, current_step, angular_velocity, comet_ids, needed
    )
    if crosses_sun(source, target_x, target_y):
        return float("-inf")

    dist = distance_xy(source.x, source.y, target_x, target_y)
    remaining = END_STEP - current_step

    value = target.production * 18.5
    capture_cost = needed * 1.55
    time_cost = travel_time * 2.55

    if target.owner == -1:
        if current_step < 120:
            value += 18.0
            time_cost *= 0.82
        if current_step < 170 and dist < 30:
            value += 12.0
        if target.production >= 4:
            value += 22.0
        if target.ships <= 10:
            value += 12.0
    else:
        # Enemy attacks should be selective. Bad enemy attacks are worse than taking neutrals.
        value += target.production * 8.0
        capture_cost += 8.0
        if current_step < 120:
            capture_cost += 26.0
        if target.ships <= 14:
            value += 12.0
        if dist < 24:
            value += 10.0

    if target.id in comet_ids:
        # Comets disappear and are unstable; only take very cheap nearby ones.
        value -= 20.0
        if dist < 18 and target.ships <= 8:
            value += 16.0

    # Late game: do not send far fleets that may not pay back.
    if remaining < travel_time + 25:
        value -= 35.0
    if remaining < 80 and target.owner == -1:
        value -= 16.0

    return value - capture_cost - time_cost


def reinforcement_score(source, ally, deficit):
    dist = distance(source, ally)
    score = deficit * 2.0 + ally.production * 12.0 - dist * 1.15
    if ally.production >= 4:
        score += 12.0
    return score


def find_reinforcement_target(source, my_planets, fleets, player, current_step, available):
    if current_step < 90:
        return None, 0

    best = None
    best_send = 0
    best_score = float("-inf")

    for ally in my_planets:
        if ally.id == source.id:
            continue
        friendly, enemy, soon_enemy = estimate_incoming_to_planet(ally, fleets, player)
        desired = max(base_reserve(ally, current_step), ally.production * 4 + 4)
        deficit = desired + max(soon_enemy, int(enemy * 0.5)) - (ally.ships + friendly)
        if deficit <= 2:
            continue

        send = min(available, int(deficit + 2))
        if send < 4:
            continue

        score = reinforcement_score(source, ally, deficit)
        if score > best_score:
            best = ally
            best_send = send
            best_score = score

    if best is not None and best_score > 16.0:
        return best, best_send
    return None, 0


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_initial_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else obs.initial_planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else obs.angular_velocity
    current_step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    comet_ids = set(obs.get("comet_planet_ids", [])) if isinstance(obs, dict) else set(obs.comet_planet_ids)

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    initial_planets = {p.id: p for p in (Planet(*p) for p in raw_initial_planets)}

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    moves = []
    target_commits = {}
    launched_from = {}

    if not my_planets or not targets:
        return moves

    # Strong planets act first. This keeps weak newly captured planets from over-launching.
    ordered_my_planets = sorted(my_planets, key=lambda p: (p.ships, p.production), reverse=True)

    for mine in ordered_my_planets:
        already_launched = launched_from.get(mine.id, 0)
        reserve = reserve_for_planet(mine, current_step, fleets, player)
        available = int(mine.ships - already_launched - reserve)
        if available <= 0:
            continue

        # Minimal reinforcement after the opening only.
        ally, send = find_reinforcement_target(mine, my_planets, fleets, player, current_step, available)
        if ally is not None and send > 0:
            angle = math.atan2(ally.y - mine.y, ally.x - mine.x)
            moves.append([mine.id, angle, int(send)])
            launched_from[mine.id] = launched_from.get(mine.id, 0) + int(send)
            available -= int(send)
            if available <= 0:
                continue

        # Opening: aggressively claim nearby neutral planets.
        if current_step < 92:
            best_target = None
            best_data = None
            best_score = float("-inf")

            for target in targets:
                committed = target_commits.get(target.id, 0)
                score, data = opening_target_score(
                    mine,
                    target,
                    current_step,
                    angular_velocity,
                    initial_planets.get(target.id),
                    comet_ids,
                    committed,
                )
                if data is None:
                    continue
                _, _, needed = data
                if needed > available:
                    continue
                if score > best_score:
                    best_score = score
                    best_target = target
                    best_data = data

            if best_target is not None and best_score > -10.0:
                target_x, target_y, needed = best_data
                angle = math.atan2(target_y - mine.y, target_x - mine.x)
                moves.append([mine.id, angle, int(needed)])
                launched_from[mine.id] = launched_from.get(mine.id, 0) + int(needed)
                target_commits[best_target.id] = target_commits.get(best_target.id, 0) + int(needed)
                continue

        # Normal mode: pick the best neutral/enemy target.
        best_target = None
        best_needed = 0
        best_score = float("-inf")

        for target in targets:
            committed = target_commits.get(target.id, 0)
            needed = ships_needed_to_capture(target) - committed
            if needed <= 0 or needed > available:
                continue

            score = target_score(
                mine,
                target,
                committed,
                current_step,
                angular_velocity,
                initial_planets.get(target.id),
                comet_ids,
                player,
            )
            if score > best_score:
                best_score = score
                best_target = target
                best_needed = needed

        early_expand = current_step < 85 and best_target is not None and best_target.owner == -1
        if best_target is None or (best_score <= 0.0 and not early_expand):
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
        launched_from[mine.id] = launched_from.get(mine.id, 0) + int(best_needed)
        target_commits[best_target.id] = target_commits.get(best_target.id, 0) + int(best_needed)

    return moves
