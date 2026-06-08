import math
from my_orbit.world import distance
from my_orbit.scoring import (
    enemy_pressure,
    selective_hold_target,
    friendly_support_count,
    score_candidate,
    light_neutral_overexpand_penalty,
    orbital_ring_value,
    frontier_gate_score,
)

def build_attack_candidates(
    sources,
    my_planets,
    targets,
    planets,
    fleets,
    projection,
    budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    is_2p,
    powers,
    Candidate,
    predict_intercept_position,
    crosses_sun,
    capture_floor,
    retake_risk_after_capture,
    attack_ambiguity_penalty,
    distance,
    fleet_speed,
    fleet_eta_to_planet,
):
    candidates = []

    def ep(planet, planets_, fleets_, player_, horizon):
        return enemy_pressure(
            planet,
            planets_,
            fleets_,
            player_,
            horizon,
            distance,
            fleet_speed,
            fleet_eta_to_planet,
        )

    def fst(target_, my_planets_, current_step_, is_2p_):
        return selective_hold_target(
            target_,
            my_planets_,
            current_step_,
            is_2p_,
            lambda t, mp, md: friendly_support_count(t, mp, md, distance),
        )

    for source in sources:
        budget = budgets.get(source.id, 0)
        if budget < config.min_ships_to_launch:
            continue

        for target in targets:
            if target.id in comet_ids:
                continue

            target_x, target_y, eta = predict_intercept_position(
                source,
                target,
                initial_planets.get(target.id),
                current_step,
                angular_velocity,
                comet_ids,
                budget,
            )

            if eta > config.horizon or crosses_sun(source, target_x, target_y):
                continue

            needed = capture_floor(target, projection, eta, player)
            if needed > budget or needed < config.min_ships_to_launch:
                continue

            if target.owner == -1 and current_step < 120:
                target_pressure = ep(
                    target,
                    planets,
                    fleets,
                    player,
                    16 if is_2p else 10,
                )
                if target.production <= 2 and target_pressure > needed + 4:
                    continue

            send = int(needed)

            if fst(target, my_planets, current_step, is_2p):
                hold_margin = 2 if target.owner == -1 else 3
                hold_margin += max(0, int(target.production) - 5)
                padded = int(needed + hold_margin)

                if padded <= budget:
                    post_capture = max(1, padded - int(needed) + 1)
                    if not retake_risk_after_capture(
                        target,
                        planets,
                        player,
                        eta,
                        post_capture,
                        is_2p,
                    ):
                        send = padded

            angle = math.atan2(target_y - source.y, target_x - source.x)
            cand = Candidate(
                "attack",
                source.id,
                target.id,
                angle,
                int(send),
                eta,
                0.0,
            )

            cand.score = score_candidate(
                cand,
                target,
                projection,
                planets,
                fleets,
                player,
                current_step,
                is_2p,
                powers,
                500,
                ep,
                fst,
                capture_floor,
                retake_risk_after_capture,
            )

            cand.score -= light_neutral_overexpand_penalty(
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
            )

            cand.score -= attack_ambiguity_penalty(
                source,
                target,
                planets,
                initial_planets,
                current_step,
                angular_velocity,
                comet_ids,
                target_x,
                target_y,
            )

            candidates.append(cand)

    return candidates


def build_counter_snipe_candidates(
    sources,
    planets,
    fleets,
    budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    is_2p,
    Candidate,
    fleet_points_toward_planet,
    fleet_eta_to_planet,
    predicted_enemy_capture_surplus,
    capture_holds_after_counter_snipe,
    predict_intercept_position,
    crosses_sun,
):
    if current_step < (35 if is_2p else 55):
        return []

    candidates = []
    max_delay = 14 if is_2p else 9
    max_cost = 34 if is_2p else 24
    enemy_eta_cap = 24 if is_2p else 18

    neutral_targets = [
        planet for planet in planets
        if planet.owner == -1 and planet.id not in comet_ids and planet.production >= 3
    ]

    for source in sources:
        budget = budgets.get(source.id, 0)
        if budget < config.min_ships_to_launch:
            continue

        for target in neutral_targets:
            for fleet in fleets:
                if fleet.owner in (-1, player):
                    continue
                if not fleet_points_toward_planet(fleet, target):
                    continue

                enemy_eta = fleet_eta_to_planet(fleet, target)
                if enemy_eta > enemy_eta_cap:
                    continue

                enemy_surplus = predicted_enemy_capture_surplus(target, fleet, enemy_eta)
                if enemy_surplus <= 0:
                    continue

                probe_ships = max(1, enemy_surplus + 1)
                target_x, target_y, travel_time = predict_intercept_position(
                    source,
                    target,
                    initial_planets.get(target.id),
                    current_step,
                    angular_velocity,
                    comet_ids,
                    probe_ships,
                )
                delay = travel_time - enemy_eta
                if delay < 1.0 or delay > max_delay:
                    continue
                if crosses_sun(source, target_x, target_y):
                    continue

                needed = enemy_surplus + int(target.production) * int(delay) + 1
                needed += 1 if is_2p else 2
                if needed > budget or needed > max_cost or needed < config.min_ships_to_launch:
                    continue

                target_x, target_y, travel_time = predict_intercept_position(
                    source,
                    target,
                    initial_planets.get(target.id),
                    current_step,
                    angular_velocity,
                    comet_ids,
                    needed,
                )
                delay = travel_time - enemy_eta
                if delay < 1.0 or delay > max_delay or crosses_sun(source, target_x, target_y):
                    continue

                if not capture_holds_after_counter_snipe(
                    target, planets, player, travel_time, needed, needed, is_2p
                ):
                    continue

                score = (
                    target.production * 42.0
                    + max(0.0, max_delay - delay) * 4.0
                    - needed * 2.0
                    - travel_time * 1.7
                )
                if target.production >= 5:
                    score += 45.0
                elif target.production >= 4:
                    score += 22.0
                if current_step > 220 and target.production <= 3:
                    score -= 20.0

                angle = math.atan2(target_y - source.y, target_x - source.x)
                candidates.append(
                    Candidate("counter_snipe", source.id, target.id, angle, int(needed), travel_time, score)
                )

    return candidates


def build_defense_candidates(
    sources,
    my_planets,
    projection,
    budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    Candidate,
    predict_intercept_position,
    validate_intercept_window,
    end_step,
):
    candidates = []
    for target in my_planets:
        loss_turn = projection.first_loss_turn_by_id.get(target.id)
        if loss_turn is None or loss_turn > config.horizon:
            continue
        if target.production < 4 and loss_turn > 8:
            continue
        previous_turn = max(0, loss_turn - 1)
        incoming = projection.incoming_by_id[target.id][loss_turn]
        enemy_incoming = sum(
            ships for owner, ships in incoming.items()
            if owner not in (-1, player)
        )
        friendly_incoming = incoming.get(player, 0)
        pre_loss_ships = projection.ships_by_id[target.id][previous_turn]
        produced = int(target.production) if target.owner >= 0 else 0
        estimated_shortfall = max(0, enemy_incoming - friendly_incoming - pre_loss_ships - produced)
        need = max(4, estimated_shortfall + int(target.production) * 2 + 3)
        if target.production < 4:
            need = min(need, int(target.production * 3 + 7))
        for source in sources:
            if source.id == target.id:
                continue
            budget = budgets.get(source.id, 0)
            if budget < need:
                continue
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, need
            )
            if eta > loss_turn + 1.25:
                continue
            if not validate_intercept_window(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, need, tx, ty, eta
            ):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            remaining = max(0, end_step - current_step - loss_turn)
            saved = target.production * 58.0 + remaining * target.production * 0.18 + int(target.ships) * 0.6
            if target.production >= 4:
                saved += 45.0
            if loss_turn <= 8:
                saved += 28.0
            score = saved - need * 0.95 - eta * 2.2
            candidates.append(Candidate("defense", source.id, target.id, angle, int(need), eta, score))
    return candidates


def build_urgent_hold_candidates(
    my_planets,
    projection,
    budgets,
    config,
    player,
    current_step,
    is_2p,
    angular_velocity,
    initial_planets,
    comet_ids,
    Candidate,
    predict_intercept_position,
    validate_intercept_window,
):
    candidates = []
    for target in my_planets:
        loss_turn = projection.first_loss_turn_by_id.get(target.id)
        if loss_turn is None:
            continue
        if loss_turn > (7 if is_2p else 5):
            continue
        if target.production < (3 if is_2p else 4) and loss_turn > 4:
            continue
        previous_turn = max(0, loss_turn - 1)
        incoming = projection.incoming_by_id[target.id][loss_turn]
        enemy_incoming = sum(
            ships for owner, ships in incoming.items()
            if owner not in (-1, player)
        )
        friendly_incoming = incoming.get(player, 0)
        pre_loss_ships = projection.ships_by_id[target.id][previous_turn]
        produced = int(target.production) if target.owner >= 0 else 0
        shortfall = max(0, enemy_incoming - friendly_incoming - pre_loss_ships - produced)
        if shortfall <= 0:
            continue
        desired = max(config.min_ships_to_launch, shortfall + int(target.production) + 2)
        for source in my_planets:
            if source.id == target.id:
                continue
            safe_budget = budgets.get(source.id, 0)
            reserve_break_budget = int(source.ships * (0.34 if loss_turn > 3 else 0.48))
            budget = max(safe_budget, reserve_break_budget)
            if budget < config.min_ships_to_launch:
                continue
            send = min(int(budget), int(desired))
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send
            )
            if eta > loss_turn + (1.0 if is_2p else 0.6):
                continue
            if not validate_intercept_window(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send, tx, ty, eta
            ):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            urgency = max(0.0, (10 if is_2p else 7) - loss_turn)
            score = (
                target.production * (72.0 if is_2p else 56.0)
                + shortfall * 3.5
                + urgency * 12.0
                - send * 1.2
                - eta * 3.2
            )
            if safe_budget < send:
                score -= (send - safe_budget) * 0.55
            if target.production >= 4:
                score += 32.0
            candidates.append(Candidate("urgent_hold", source.id, target.id, angle, int(send), eta, score))
    return candidates


def build_frontline_relay_candidates(
    my_planets,
    planets,
    fleets,
    budgets,
    config,
    player,
    current_step,
    is_2p,
    angular_velocity,
    initial_planets,
    comet_ids,
    Candidate,
    enemy_pressure,
    orbital_ring_value,
    friendly_support_count,
    frontier_gate_score,
    distance,
    predict_intercept_position,
    validate_intercept_window,
):
    if current_step > (105 if is_2p else 72):
        return []
    pressures = {
        planet.id: enemy_pressure(planet, planets, fleets, player, 18 if is_2p else 12)
        for planet in my_planets
    }
    candidates = []
    for source in my_planets:
        budget = budgets.get(source.id, 0)
        if budget < max(6, config.min_ships_to_launch):
            continue
        if source.production >= (5 if is_2p else 6):
            continue
        source_pressure = pressures.get(source.id, 0.0)
        for target in my_planets:
            if target.id == source.id:
                continue
            if target.production < (4 if is_2p else 5):
                continue
            dist = distance(source, target)
            if dist > (16.0 if is_2p else 11.5):
                continue
            target_pressure = pressures.get(target.id, 0.0)
            gap = target_pressure - source_pressure
            if gap < (24.0 if is_2p else 16.0):
                continue
            if orbital_ring_value(target) <= orbital_ring_value(source) + 4.0:
                continue
            support = friendly_support_count(target, my_planets, 16.5 if is_2p else 12.0)
            if support < 1:
                continue
            gate = frontier_gate_score(source, target, my_planets, planets, fleets, player, current_step, is_2p)
            if gate < (78.0 if is_2p else 62.0):
                continue
            send_cap = min(int(budget), max(8 if is_2p else 6, int(source.ships * (0.18 if is_2p else 0.15))))
            send = min(send_cap, max(config.min_ships_to_launch, int(gap * 0.18)))
            if send < max(5, config.min_ships_to_launch):
                continue
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send
            )
            if eta > (7.0 if is_2p else 5.0):
                continue
            if not validate_intercept_window(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send, tx, ty, eta
            ):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            score = (
                gate
                + target.production * (12.0 if is_2p else 10.0)
                + gap * 0.85
                - send * 1.45
                - eta * 2.1
            )
            if current_step < 75:
                score += 4.0
            candidates.append(Candidate("frontline_relay", source.id, target.id, angle, int(send), eta, score))
    return candidates


def build_regroup_candidates(
    my_planets,
    planets,
    fleets,
    budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    Candidate,
    enemy_pressure,
    distance,
    predict_intercept_position,
    validate_intercept_window,
):
    if not config.regroup_enabled:
        return []
    pressures = {
        planet.id: enemy_pressure(planet, planets, fleets, player, config.horizon)
        for planet in my_planets
    }
    candidates = []
    for source in my_planets:
        budget = budgets.get(source.id, 0)
        if budget < config.min_ships_to_launch:
            continue
        source_pressure = pressures.get(source.id, 0.0)
        for target in my_planets:
            if target.id == source.id:
                continue
            dist = distance(source, target)
            if dist > config.regroup_distance:
                continue
            gap = pressures.get(target.id, 0.0) - source_pressure
            if gap < config.regroup_threshold:
                continue
            send = min(budget, max(config.min_ships_to_launch, int(gap * 0.35)))
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send
            )
            if not validate_intercept_window(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send, tx, ty, eta
            ):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            score = gap - send * 0.45 - eta
            candidates.append(Candidate("regroup", source.id, target.id, angle, int(send), eta, score))
    return candidates