"""
Orbit Wars - V3.0 Influence Map Agent

Design goals:
- Keep the proven V2.2 fast neutral expansion.
- Avoid the V2.4 problem: defense must not slow the opening.
- Use guarded midgame pressure without weakening the opening.
- Push harder for production-4/5 neutral planets to reduce seed-specific economic losses.
- Keep V2.7/V2.9 speed and selective high-production recapture.
- Add a lightweight influence map for midgame target choice.
- Avoid V2.8-style global defense slowdown.
- Prefer targets that can be supported and avoid unsupported midgame overreach.
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

    # Small late-midgame buffer for valuable planets.
    # Starts after the expansion window so it does not break the V2.2-style opening.
    if current_step >= 180 and planet.production >= 4:
        reserve += 3

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
    # V2.7: production is nonlinear in the opening.
    # The previous version could waste early ships on cheap low-production planets
    # and then lose the map economy on seeds like 10/18/27.
    score += target.production * 28.0
    score += (target.production * target.production) * 5.5
    score -= needed * 2.35
    score -= travel_time * 3.0

    if target.production >= 5:
        score += 72.0
        if dist <= 34:
            score += 24.0
    elif target.production >= 4:
        score += 48.0
        if dist <= 34:
            score += 14.0

    if target.ships <= 12:
        score += 20.0
    if target.ships <= 7:
        score += 14.0
    if dist <= 22:
        score += 17.0
    if dist <= 14:
        score += 10.0

    # Cheap low-production planets are useful, but should not dominate the opening
    # when high-production neutrals are reachable.
    if target.production <= 1:
        score -= 20.0
        if current_step < 120 and dist > 18:
            score -= 18.0
    elif target.production <= 2 and target.ships > 18:
        score -= 18.0

    return score, (target_x, target_y, needed)


def target_score(source, target, committed_ships, current_step, angular_velocity, initial_planet, comet_ids, player, planets, fleets):
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
        # V2.7: keep expanding toward production-4/5 neutrals in midgame.
        if target.production >= 5:
            value += 46.0
            if current_step < 240 and dist < 42:
                value += 22.0
        elif target.production >= 4:
            value += 30.0
            if current_step < 220 and dist < 36:
                value += 12.0
        elif target.production <= 1 and current_step < 160:
            value -= 14.0
        if target.ships <= 10:
            value += 12.0
    else:
        # Enemy attacks should be selective. Bad enemy attacks are worse than taking neutrals.
        value += target.production * 8.0
        capture_cost += 8.0
        # Do not rush enemy planets before neutral expansion has paid off.
        if current_step < 140:
            capture_cost += 38.0
        elif current_step < 210:
            capture_cost += 12.0
        if dist > 38:
            capture_cost += 18.0
        # V2.9: selective high-production recapture pressure.
        # Do not globally increase reserves; instead punish valuable enemy planets
        # when they are close enough and affordable. This targets the real-match
        # failure mode where captured production-3/4/5 planets were not held/retaken.
        if target.production >= 5:
            if current_step >= 105:
                value += 34.0
            if dist < 36:
                value += 18.0
            if target.ships <= 35:
                value += 14.0
            if target.ships <= 18:
                value += 14.0
        elif target.production >= 4:
            if current_step >= 120:
                value += 22.0
            if dist < 32:
                value += 12.0
            if target.ships <= 26:
                value += 10.0

        if target.ships <= 14:
            value += 12.0
        if target.ships <= 8 and dist < 26:
            value += 12.0
        if dist < 24:
            value += 10.0

        # Still avoid low-value enemy planets; V2.9 is not a general rush bot.
        if target.production <= 2 and current_step < 260:
            capture_cost += 10.0

    # V3.0: influence map for midgame support/overreach control.
    value += influence_adjustment(
        source, target, target_x, target_y, planets, fleets, player, current_step
    )

    if target.id in comet_ids:
        # Comets disappear and are unstable; only take very cheap nearby ones.
        value -= 20.0
        if dist < 18 and target.ships <= 8:
            value += 16.0

    # Late game: do not send far fleets that may not pay back.
    if remaining < travel_time + 25:
        value -= 35.0
    if remaining < 100 and target.owner == -1:
        if target.production >= 4 and dist < 28:
            value += 8.0
        else:
            value -= 24.0
    if remaining < 70 and target.owner != -1 and target.production >= 4 and dist < 30:
        value += 18.0

    return value - capture_cost - time_cost



def local_influence_at(x, y, planets, fleets, player):
    """Lightweight midgame influence map.

    It estimates whether a target area is supported by our nearby economy/fleets
    or dominated by enemy ships. This is deliberately cheap: O(planets+fleets)
    per candidate, no simulation tree.
    """
    friendly = 0.0
    enemy = 0.0

    for p in planets:
        if p.owner < 0:
            continue
        d = max(4.0, distance_xy(x, y, p.x, p.y))
        # Nearby production matters more than far stored ships.
        weight = 1.0 / (1.0 + d / 18.0)
        value = p.ships * 0.20 + p.production * 8.5
        if p.owner == player:
            friendly += value * weight
        else:
            enemy += value * weight

    for f in fleets:
        d = max(3.0, distance_xy(x, y, f.x, f.y))
        weight = 1.0 / (1.0 + d / 20.0)
        value = f.ships * 0.42
        if f.owner == player:
            friendly += value * weight
        else:
            enemy += value * weight

    return friendly, enemy


def influence_adjustment(source, target, target_x, target_y, planets, fleets, player, current_step):
    """Score adjustment for the weak midgame phase.

    V2.9 already knows which planets are valuable. This only asks:
    "Can we hold or trade this area after taking it?"
    """
    if current_step < 92 or current_step > 360:
        return 0.0

    friendly, enemy = local_influence_at(target_x, target_y, planets, fleets, player)
    net = friendly - enemy
    adj = 0.0

    if target.owner == -1:
        # Neutral targets inside our influence are good chain-expansion targets.
        if net > 18.0:
            adj += min(22.0, net * 0.22)
        # Avoid wasting midgame ships on neutrals already controlled by enemy area.
        elif net < -24.0:
            penalty = min(30.0, (-net) * 0.20)
            if target.production >= 4:
                penalty *= 0.55
            adj -= penalty
    else:
        # Enemy high-production planets are still worth contesting, but only if
        # the area is not completely unsupported.
        if target.production >= 4:
            if net > -18.0:
                adj += 16.0
            if net > 14.0:
                adj += min(18.0, net * 0.18)
            if net < -45.0 and target.ships > 24:
                adj -= 18.0
        else:
            if net < -18.0:
                adj -= min(26.0, (-net) * 0.20)
            if net > 22.0 and target.ships <= 18:
                adj += 10.0

    # Production hubs should be used to snowball nearby high-value targets.
    if source.production >= 4 and current_step >= 110 and target.production >= 3:
        src_friend, src_enemy = local_influence_at(source.x, source.y, planets, fleets, player)
        if src_friend + 10.0 > src_enemy:
            adj += 7.0

    return adj


def reinforcement_score(source, ally, deficit):
    dist = distance(source, ally)
    score = deficit * 2.0 + ally.production * 12.0 - dist * 1.15
    if ally.production >= 4:
        score += 12.0
    return score


def find_reinforcement_target(source, my_planets, fleets, player, current_step, available):
    # Reinforcement is expensive: delay it until the economy is established.
    if current_step < 140:
        return None, 0

    best = None
    best_send = 0
    best_score = float("-inf")

    for ally in my_planets:
        if ally.id == source.id:
            continue
        friendly, enemy, soon_enemy = estimate_incoming_to_planet(ally, fleets, player)
        desired = max(base_reserve(ally, current_step), ally.production * 3 + 4)
        # Only react to concrete incoming pressure, not vague distant fleets.
        deficit = desired + max(soon_enemy, int(enemy * 0.35)) - (ally.ships + friendly)
        if deficit <= 2:
            continue

        send = min(available // 2, int(deficit + 2))
        if send < 5:
            continue

        score = reinforcement_score(source, ally, deficit)
        if score > best_score:
            best = ally
            best_send = send
            best_score = score

    if best is not None and best_score > 32.0:
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
                planets,
                fleets,
            )
            if score > best_score:
                best_score = score
                best_target = target
                best_needed = needed

        early_expand = current_step < 85 and best_target is not None and best_target.owner == -1
        min_score = 0.0
        if current_step >= 180 and best_target is not None and best_target.owner != -1:
            # V2.9: allow production-4/5 enemy planets to be retaken,
            # but keep the bar high for ordinary enemy planets.
            min_score = -2.0 if best_target.production >= 4 else 8.0
        if current_step >= 390 and best_target is not None and best_target.owner == -1:
            min_score = 8.0
        if best_target is None or (best_score <= min_score and not early_expand):
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
