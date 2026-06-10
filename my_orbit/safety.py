def base_reserve(planet, current_step, is_2p):
    if current_step < 90:
        return max(1, planet.production // 2 + 1) if is_2p else max(2, planet.production)
    if current_step < 160:
        return max(3, planet.production + 1) if is_2p else max(5, planet.production * 2)
    if current_step > 430:
        return max(7, planet.production * 3) if is_2p else max(10, planet.production * 4)
    return max(4, planet.production * 2) if is_2p else max(6, planet.production * 3)


def frontline_reserve_bonus(planet, planets, player, current_step, is_2p, distance):
    if current_step < 85:
        return 0

    bonus = 0
    for enemy in planets:
        if enemy.owner in (-1, player):
            continue

        dist = distance(enemy, planet)
        if dist > (28.0 if is_2p else 22.0):
            continue

        if int(enemy.ships) < int(planet.ships) + int(planet.production) * 2:
            continue

        local = int(planet.production) + 2
        if planet.production >= 4:
            local += 3
        if dist < 18:
            local += 2

        bonus = max(bonus, local)

    return min(bonus, 10 if is_2p else 7)


def safe_drain(source, projection, planets, player, config, current_step, is_2p, base_reserve, frontline_reserve_bonus):
    owner_traj = projection.owner_by_id[source.id]
    ships_traj = projection.ships_by_id[source.id]
    held_slack = []

    for turn in range(1, min(config.defense_horizon, len(owner_traj) - 1) + 1):
        if owner_traj[turn] == player and ships_traj[turn] > 0:
            held_slack.append(int(ships_traj[turn]))

    if held_slack:
        drain = min(int(source.ships), min(held_slack))
    else:
        drain = int(source.ships)

    # Remove static heuristics to emulate hairate2
    # reserve = base_reserve(source, current_step, is_2p)
    # reserve += frontline_reserve_bonus(source, planets, player, current_step, is_2p)
    reserve = config.reserve_margin

    return max(0, drain - reserve)