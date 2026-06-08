def capture_floor(target, projection, eta, player, overhead=1):
    turn = max(1, min(len(projection.ships_by_id[target.id]) - 1, int(__import__("math").ceil(eta))))
    owner = projection.owner_by_id[target.id][turn]
    ships = projection.ships_by_id[target.id][turn]
    if owner == player:
        return 1
    return max(1, int(__import__("math").ceil(ships + overhead)))


def predicted_enemy_capture_surplus(target, enemy_fleet, enemy_eta):
    garrison = int(target.ships)
    if target.owner >= 0:
        garrison += int(target.production) * int(max(0.0, enemy_eta))
    surplus = int(enemy_fleet.ships) - garrison
    return surplus if surplus > 0 else 0


def retake_risk_after_capture(target, planets, player, arrival_turn, post_capture_ships, is_2p, distance, fleet_speed):
    horizon = 18 if is_2p else 12
    margin = 5 if is_2p else 7

    for enemy in planets:
        if enemy.owner in (-1, player):
            continue

        eta = distance(enemy, target) / fleet_speed(max(1, int(enemy.ships)))
        delay = eta - arrival_turn
        if delay <= 0.0 or delay > horizon:
            continue

        hold = int(post_capture_ships) + int(target.production) * int(delay)
        if int(enemy.ships) >= hold + margin:
            return True

    return False


def capture_holds_after_counter_snipe(target, planets, player, arrival_turn, ships_sent, needed, is_2p, distance, fleet_speed):
    surplus = max(0, int(ships_sent) - int(needed) + 1)
    horizon = 24 if is_2p else 14
    margin = 4 if is_2p else 7

    for enemy in planets:
        if enemy.owner in (-1, player):
            continue

        enemy_eta = distance(enemy, target) / fleet_speed(max(1, int(enemy.ships)))
        delay = enemy_eta - arrival_turn
        if delay <= 0.0 or delay > horizon:
            continue

        projected_hold = surplus + int(target.production) * int(delay)
        if int(enemy.ships) >= projected_hold + margin:
            return False

    return True