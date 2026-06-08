def player_power(planets, fleets):
    stats = {}
    for planet in planets:
        if planet.owner < 0:
            continue
        entry = stats.setdefault(int(planet.owner), {"production": 0, "ships": 0, "planets": 0})
        entry["production"] += int(planet.production)
        entry["ships"] += int(planet.ships)
        entry["planets"] += 1

    for fleet in fleets:
        if fleet.owner < 0:
            continue
        entry = stats.setdefault(int(fleet.owner), {"production": 0, "ships": 0, "planets": 0})
        entry["ships"] += int(fleet.ships * 0.7)

    for entry in stats.values():
        entry["power"] = entry["production"] * 16 + entry["ships"] + entry["planets"] * 10

    return stats


def enemy_pressure(planet, planets, fleets, player, horizon, distance, fleet_speed, fleet_eta_to_planet):
    pressure = 0.0

    for enemy in planets:
        if enemy.owner in (-1, player):
            continue
        eta = distance(enemy, planet) / fleet_speed(max(1, int(enemy.ships)))
        if eta <= horizon:
            pressure += int(enemy.ships) * max(0.0, 1.0 - eta / max(1.0, horizon))
            pressure += int(enemy.production) * 3.0

    for fleet in fleets:
        if fleet.owner in (-1, player):
            continue
        eta = fleet_eta_to_planet(fleet, planet)
        if eta <= horizon:
            pressure += int(fleet.ships) * 0.7 * max(0.0, 1.0 - eta / max(1.0, horizon))

    return pressure


def orbital_ring_value(planet, distance_xy, CENTER_X, CENTER_Y):
    radius = distance_xy(planet.x, planet.y, CENTER_X, CENTER_Y)
    outer_bonus = max(0.0, radius - 22.0)
    return outer_bonus * 1.8 + planet.production * 4.0


def friendly_support_count(target, my_planets, max_dist, distance):
    return sum(
        1
        for friend in my_planets
        if friend.id != target.id and distance(friend, target) <= max_dist
    )


def selective_hold_target(target, my_planets, current_step, is_2p, friendly_support_count):
    if current_step > (165 if is_2p else 120):
        return False
    if target.production < (5 if is_2p else 5):
        return False
    support = friendly_support_count(target, my_planets, 16.5 if is_2p else 12.0)
    if support < 1:
        return False
    if target.owner == -1 and current_step > (115 if is_2p else 85):
        return False
    return True


def target_shortlist(my_planets, targets, planets, config, distance):
    ranked = []

    for target in targets:
        nearest = min(distance(source, target) for source in my_planets)
        value = target.production * 65.0 - int(target.ships) * 1.4 - nearest * 2.2

        if target.owner != -1:
            value += target.production * 28.0
        if target.production <= 1:
            value -= 45.0

        ranked.append((value, target))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [target for _, target in ranked[: config.max_targets]]

def frontier_gate_score(
    source,
    target,
    my_planets,
    planets,
    fleets,
    player,
    current_step,
    is_2p,
    enemy_pressure,
    friendly_support_count,
    orbital_ring_value,
):
    target_pressure = enemy_pressure(target, planets, fleets, player, 18 if is_2p else 12)
    source_pressure = enemy_pressure(source, planets, fleets, player, 18 if is_2p else 12)
    support = friendly_support_count(target, my_planets, 16.5 if is_2p else 12.0)
    ring_gain = orbital_ring_value(target) - orbital_ring_value(source)

    score = 0.0
    score += ring_gain * 1.1
    score += max(0.0, target_pressure - source_pressure) * 0.55
    score += target.production * (8.0 if is_2p else 6.0)
    score += support * 10.0

    if current_step < 75:
        score += 8.0
    if target.production >= 5:
        score += 12.0
    if support == 0:
        score -= 30.0
    if ring_gain < 4.0:
        score -= 20.0

    return score

def score_candidate(
    candidate,
    target,
    projection,
    planets,
    fleets,
    player,
    current_step,
    is_2p,
    powers,
    END_STEP,
    enemy_pressure,
    selective_hold_target,
    capture_floor,
    retake_risk_after_capture,
):
    remaining = max(0.0, END_STEP - current_step - candidate.eta)
    horizon_payoff = min(75.0 if is_2p else 50.0, remaining)
    prod_value = target.production * horizon_payoff * (0.55 if is_2p else 0.36)
    pressure = enemy_pressure(target, planets, fleets, player, 24 if is_2p else 16)
    risk_cost = max(0.0, pressure - candidate.ships) * (0.28 if is_2p else 0.18)
    ship_cost = candidate.ships * (1.15 if candidate.kind == "attack" else 0.72)
    time_cost = candidate.eta * (2.8 if candidate.kind == "attack" else 1.4)

    if candidate.kind == "defense":
        saved = target.production * 45.0 + int(target.ships) * 1.2
        return saved - ship_cost - time_cost

    denial = 0.0
    if target.owner != -1:
        owner_stats = powers.get(int(target.owner), {})
        my_stats = powers.get(int(player), {})
        denial += target.production * (22.0 if is_2p else 13.0)
        if owner_stats.get("power", 0) > my_stats.get("power", 0):
            denial += min(34.0, (owner_stats.get("power", 0) - my_stats.get("power", 0)) * 0.08)

    if target.owner == -1 and current_step > 260 and target.production <= 2:
        prod_value -= 35.0
    if remaining < candidate.eta + 25 and target.owner == -1:
        prod_value -= 25.0

    score = prod_value + denial - risk_cost - ship_cost - time_cost

    if candidate.kind == "attack" and selective_hold_target(
        target,
        [p for p in planets if p.owner == player],
        current_step,
        is_2p,
    ):
        capture_need = capture_floor(target, projection, candidate.eta, player)
        post_capture = max(1, int(candidate.ships) - int(capture_need) + 1)
        if not retake_risk_after_capture(
            target,
            planets,
            player,
            candidate.eta,
            post_capture,
            is_2p,
        ):
            score += 16.0 + target.production * 3.0

    if target.production >= 5:
        score += 34.0
    elif target.production >= 4:
        score += 16.0

    return score

def light_neutral_overexpand_penalty(
    source,
    target,
    my_planets,
    planets,
    fleets,
    player,
    current_step,
    is_2p,
    distance,
    fleet_speed,
    fleet_eta_to_planet,
):
    if target.owner != -1:
        return 0.0
    if current_step >= 140:
        return 0.0

    target_pressure = enemy_pressure(
        target,
        planets,
        fleets,
        player,
        16 if is_2p else 10,
        distance,
        fleet_speed,
        fleet_eta_to_planet,
    )

    support_dist = min(
        (distance(friend, target) for friend in my_planets if friend.id != source.id),
        default=99.0,
    )

    penalty = 0.0
    if target.production <= 2:
        penalty += max(0.0, support_dist - (15.0 if is_2p else 11.0)) * 2.6
        penalty += max(0.0, target_pressure - int(target.ships) - 2) * 0.32
    elif target.production == 3 and target_pressure > int(target.ships) + 6:
        penalty += (target_pressure - int(target.ships) - 6) * 0.18

    return penalty
