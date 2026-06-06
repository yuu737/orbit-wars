"""
Orbit Wars - V3.2 Berserker Rush Agent

Design goals:
- Keep the proven V2.2 fast neutral expansion.
- Avoid the V2.4 problem: defense must not slow the opening.
- Use guarded midgame pressure without weakening the opening.
- Push harder for production-4/5 neutral planets to reduce seed-specific economic losses.
- Keep V2.7/V2.9 high-production economy pressure.
- Experimental berserker mode: lower reserves and attack enemy high-production planets earlier.
- Ignore low-value expansion more aggressively to force short/mid-game advantage.
- Prefer safe shots that do not cross the sun.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


MAX_SPEED = 6.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
END_STEP = 500
RESERVE_EVENT_HORIZON = 28
OPENING_LOOKAHEAD_HORIZON = 12


MODE_2P = {
    "opening_turn": 92,
    "opening_distance_cap": 46.0,
    "opening_accept_threshold": -24.0,
    "base_min_score": -3.0,
    "enemy_min_score_turn": 120,
    "high_prod_enemy_min_score": -10.0,
    "low_prod_enemy_min_score": 10.0,
    "reinforcement_turn": 210,
    "reinforcement_threshold": 48.0,
    "event_horizon": 28,
    "high_prod_hold_turn": 220,
    "high_prod_hold_bonus": 1,
}

MODE_4P = {
    "opening_turn": 88,
    "opening_distance_cap": 40.0,
    "opening_accept_threshold": -8.0,
    "base_min_score": 2.0,
    "enemy_min_score_turn": 180,
    "high_prod_enemy_min_score": 6.0,
    "low_prod_enemy_min_score": 14.0,
    "reinforcement_turn": 150,
    "reinforcement_threshold": 28.0,
    "event_horizon": 18,
    "high_prod_hold_turn": 170,
    "high_prod_hold_bonus": 3,
}


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


def summarize_player_power(planets, fleets):
    stats = {}

    def ensure(owner):
        return stats.setdefault(
            int(owner),
            {
                "planets": 0,
                "production": 0,
                "planet_ships": 0,
                "fleet_ships": 0,
            },
        )

    for planet in planets:
        owner = int(planet.owner)
        if owner < 0:
            continue
        entry = ensure(owner)
        entry["planets"] += 1
        entry["production"] += int(planet.production)
        entry["planet_ships"] += int(planet.ships)

    for fleet in fleets:
        owner = int(fleet.owner)
        if owner < 0:
            continue
        entry = ensure(owner)
        entry["fleet_ships"] += int(fleet.ships)

    for entry in stats.values():
        entry["power"] = (
            entry["production"] * 16
            + entry["planets"] * 10
            + entry["planet_ships"]
            + int(entry["fleet_ships"] * 0.7)
        )

    return stats


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


def estimate_enemy_race_eta(target, planets, player):
    best_eta = None

    for planet in planets:
        if planet.owner in (-1, player):
            continue
        if int(planet.ships) < ships_needed_to_capture(target):
            continue

        eta = distance(planet, target) / fleet_speed(int(planet.ships))
        if best_eta is None or eta < best_eta:
            best_eta = eta

    return best_eta


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


def estimate_incoming_to_planet(planet, fleets, player, horizon):
    friendly = 0
    enemy = 0
    soon_enemy = 0

    for fleet in fleets:
        if not fleet_points_toward_planet(fleet, planet):
            continue

        dist = distance_xy(fleet.x, fleet.y, planet.x, planet.y)
        turns = dist / fleet_speed(fleet.ships)
        if turns > horizon:
            continue

        ships = int(fleet.ships)
        if fleet.owner == player:
            friendly += ships
        else:
            enemy += ships
            if turns <= 18:
                soon_enemy += ships

    return friendly, enemy, soon_enemy


def projected_reserve_requirement(planet, fleets, player, event_horizon):
    events = []

    for fleet in fleets:
        if not fleet_points_toward_planet(fleet, planet):
            continue

        turns = distance_xy(fleet.x, fleet.y, planet.x, planet.y) / fleet_speed(fleet.ships)
        if turns > event_horizon:
            continue

        ships = int(fleet.ships)
        if fleet.owner == player:
            events.append((turns, ships))
        else:
            events.append((turns, -ships))

    if not events:
        return 0

    events.sort(key=lambda item: item[0])
    balance = int(planet.ships)
    min_balance = balance
    last_turn = 0.0
    growth = int(planet.production)

    for turns, delta in events:
        balance += growth * int(max(0.0, turns - last_turn))
        balance += delta
        min_balance = min(min_balance, balance)
        last_turn = turns

    if min_balance >= 0:
        excess = min_balance
        return max(0, int(planet.ships) - excess)

    return int(planet.ships)


def base_reserve(planet, current_step, is_2p):
    # V3.2: berserker reserves. Keep only enough ships to avoid trivial losses.
    # This intentionally trades holding power for tempo and early snowball.
    if current_step < 90:
        return max(1, planet.production // 2 + 1) if is_2p else max(2, planet.production)
    if current_step < 160:
        return max(3, planet.production + 1) if is_2p else max(5, planet.production * 2)
    if current_step > 430:
        return max(7, planet.production * 3) if is_2p else max(10, planet.production * 4)
    return max(4, planet.production * 2) if is_2p else max(6, planet.production * 3)


def reserve_for_planet(planet, current_step, fleets, player, mode, is_2p):
    reserve = max(
        base_reserve(planet, current_step, is_2p),
        projected_reserve_requirement(planet, fleets, player, mode["event_horizon"]),
    )

    if current_step >= mode["high_prod_hold_turn"] and planet.production >= 5:
        reserve += mode["high_prod_hold_bonus"]

    # Do not let defense ruin the opening. Only protect against very nearby, obvious attacks.
    if current_step < 90:
        friendly, enemy, soon_enemy = estimate_incoming_to_planet(planet, fleets, player, mode["event_horizon"])
        if soon_enemy > 0:
            reserve = max(reserve, min(planet.ships, soon_enemy + 2 - friendly))
        return reserve

    friendly, enemy, soon_enemy = estimate_incoming_to_planet(planet, fleets, player, mode["event_horizon"])
    pressure = max(soon_enemy, int(enemy * 0.6))
    if pressure > 0:
        reserve = max(reserve, pressure + planet.production * 2 + 2 - friendly)

    return max(0, reserve)


def opening_future_bonus(target, travel_time, enemy_race_eta, current_step, is_2p):
    remaining = max(0.0, OPENING_LOOKAHEAD_HORIZON - travel_time)
    bonus = target.production * remaining * (6.5 if is_2p else 5.0)

    if target.production >= 4:
        bonus += 12.0 if is_2p else 8.0

    if target.ships <= 8:
        bonus += 8.0

    if enemy_race_eta is not None:
        if travel_time < enemy_race_eta:
            bonus += (enemy_race_eta - travel_time) * 4.0
        else:
            bonus -= (travel_time - enemy_race_eta) * 3.0

    if current_step < 50 and travel_time <= 8:
        bonus += 6.0

    return bonus


def opening_target_score(
    source,
    target,
    current_step,
    angular_velocity,
    initial_planet,
    comet_ids,
    committed_ships,
    enemy_race_eta,
    mode,
):
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
    if dist > mode["opening_distance_cap"]:
        return float("-inf"), None

    if enemy_race_eta is not None and travel_time > enemy_race_eta + 3.0:
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
        score += 88.0
        if dist <= 38:
            score += 30.0
    elif target.production >= 4:
        score += 58.0
        if dist <= 36:
            score += 18.0

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
        score -= 34.0
        if current_step < 150 and dist > 14:
            score -= 26.0
    elif target.production <= 2 and target.ships > 14:
        score -= 24.0

    if enemy_race_eta is not None:
        score -= max(0.0, travel_time - enemy_race_eta) * 3.5

    score += opening_future_bonus(target, travel_time, enemy_race_eta, current_step, mode is MODE_2P)

    return score, (target_x, target_y, needed)


def target_score(
    source,
    target,
    committed_ships,
    current_step,
    angular_velocity,
    initial_planet,
    comet_ids,
    player,
    enemy_race_eta,
    is_2p,
    player_power,
):
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
        if is_2p:
            if target.production >= 5:
                value += 56.0
                if current_step < 260 and dist < 46:
                    value += 28.0
            elif target.production >= 4:
                value += 38.0
                if current_step < 240 and dist < 40:
                    value += 16.0
            elif target.production <= 1 and current_step < 220:
                value -= 26.0
        else:
            if target.production >= 5:
                value += 42.0
                if current_step < 220 and dist < 38:
                    value += 18.0
            elif target.production >= 4:
                value += 26.0
                if current_step < 200 and dist < 34:
                    value += 10.0
            elif target.production <= 1 and current_step < 180:
                value -= 12.0
        if target.ships <= 10:
            value += 12.0
        if enemy_race_eta is not None:
            if travel_time > enemy_race_eta + 4.0:
                value -= 40.0
            value -= max(0.0, travel_time - enemy_race_eta) * 2.5
    else:
        # Enemy attacks should be selective. Bad enemy attacks are worse than taking neutrals.
        value += target.production * 8.0
        capture_cost += 8.0
        # Do not rush enemy planets before neutral expansion has paid off.
        if is_2p:
            if current_step < 120:
                capture_cost += 30.0 if target.production < 4 else 8.0
            elif current_step < 210:
                capture_cost += 10.0 if target.production < 4 else 0.0
        else:
            if current_step < 150:
                capture_cost += 40.0 if target.production < 4 else 18.0
            elif current_step < 240:
                capture_cost += 16.0 if target.production < 4 else 6.0
        if dist > 38:
            capture_cost += 18.0
        # V2.9: selective high-production recapture pressure.
        # Do not globally increase reserves; instead punish valuable enemy planets
        # when they are close enough and affordable. This targets the real-match
        # failure mode where captured production-3/4/5 planets were not held/retaken.
        if target.production >= 5:
            if current_step >= 75:
                value += 52.0
            if dist < 42:
                value += 24.0
            if target.ships <= 45:
                value += 16.0
            if target.ships <= 22:
                value += 18.0
        elif target.production >= 4:
            if current_step >= 90:
                value += 34.0
            if dist < 38:
                value += 16.0
            if target.ships <= 32:
                value += 12.0

        if target.ships <= 14:
            value += 12.0
        if target.ships <= 8 and dist < 26:
            value += 12.0
        if dist < 24:
            value += 10.0

        if not is_2p:
            enemy_entries = [entry for owner, entry in player_power.items() if owner != player]
            owner_power = player_power.get(target.owner, {}).get("power", 0)
            owner_production = player_power.get(target.owner, {}).get("production", 0)
            if enemy_entries:
                weakest_enemy_power = min(entry["power"] for entry in enemy_entries)
                strongest_enemy_power = max(entry["power"] for entry in enemy_entries)
                power_span = strongest_enemy_power - weakest_enemy_power

                if owner_power <= weakest_enemy_power + 18:
                    value += 14.0
                    if target.production >= 3:
                        value += 10.0
                    if target.ships <= 18:
                        value += 8.0

                if owner_power >= strongest_enemy_power - 18:
                    # In 4P, bashing the leader is only worth it when the planet is
                    # cheap or strategically valuable; otherwise it often feeds the table.
                    if target.production >= 5 and target.ships <= 22 and dist < 28:
                        value += 10.0
                    elif power_span >= 35:
                        capture_cost += 16.0
                        capture_cost += max(0.0, target.ships - 16) * 0.45
                        if target.production <= 3:
                            capture_cost += 8.0
                    else:
                        capture_cost += 6.0

                if owner_production <= 5 and target.production >= 4 and target.ships <= 20:
                    value += 8.0

        # Still avoid low-value enemy planets; V2.9 is not a general rush bot.
        if target.production <= 2:
            capture_cost += 18.0 if is_2p and current_step < 300 else 12.0

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


def reinforcement_score(source, ally, deficit):
    dist = distance(source, ally)
    score = deficit * 2.0 + ally.production * 12.0 - dist * 1.15
    if ally.production >= 4:
        score += 12.0
    return score


def find_reinforcement_target(source, my_planets, fleets, player, current_step, available, mode, is_2p):
    # Reinforcement is expensive: delay it until the economy is established.
    if current_step < mode["reinforcement_turn"]:
        return None, 0

    best = None
    best_send = 0
    best_score = float("-inf")

    for ally in my_planets:
        if ally.id == source.id:
            continue
        friendly, enemy, soon_enemy = estimate_incoming_to_planet(ally, fleets, player, mode["event_horizon"])
        desired = max(base_reserve(ally, current_step, is_2p), ally.production * 3 + 4)
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

    if best is not None and best_score > mode["reinforcement_threshold"]:
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
    num_players = infer_num_players(raw_initial_planets, raw_planets, raw_fleets, player)
    is_2p = num_players == 2
    mode = MODE_2P if is_2p else MODE_4P
    player_power = summarize_player_power(planets, fleets)
    enemy_race_eta_by_target = {
        target.id: estimate_enemy_race_eta(target, planets, player)
        for target in planets
        if target.owner == -1
    }

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
        reserve = reserve_for_planet(mine, current_step, fleets, player, mode, is_2p)
        available = int(mine.ships - already_launched - reserve)
        if available <= 0:
            continue

        # Minimal reinforcement after the opening only.
        ally, send = find_reinforcement_target(
            mine, my_planets, fleets, player, current_step, available, mode, is_2p
        )
        if ally is not None and send > 0:
            angle = math.atan2(ally.y - mine.y, ally.x - mine.x)
            moves.append([mine.id, angle, int(send)])
            launched_from[mine.id] = launched_from.get(mine.id, 0) + int(send)
            available -= int(send)
            if available <= 0:
                continue

        # Opening: aggressively claim nearby neutral planets.
        if current_step < mode["opening_turn"]:
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
                    enemy_race_eta_by_target.get(target.id),
                    mode,
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

            if best_target is not None and best_score > mode["opening_accept_threshold"]:
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
                enemy_race_eta_by_target.get(target.id),
                is_2p,
                player_power,
            )
            if score > best_score:
                best_score = score
                best_target = target
                best_needed = needed

        early_expand = current_step < 85 and best_target is not None and best_target.owner == -1
        min_score = mode["base_min_score"]
        if current_step >= mode["enemy_min_score_turn"] and best_target is not None and best_target.owner != -1:
            min_score = (
                mode["high_prod_enemy_min_score"]
                if best_target.production >= 4
                else mode["low_prod_enemy_min_score"]
            )
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
