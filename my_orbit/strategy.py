import math
from my_orbit.models import Candidate
from my_orbit.selection import selected_ids

def projected_attacker_surplus_at_loss(target, projection, loss_turn, player):
    arrivals = projection.incoming_by_id[target.id][loss_turn]
    best_enemy = 0
    own_arrival = arrivals.get(int(player), 0)
    for owner, ships in arrivals.items():
        if int(owner) != int(player):
            best_enemy = max(best_enemy, int(ships))
    projected_garrison = int(target.ships) + max(0, loss_turn - 1) * int(target.production) + own_arrival
    return max(0, best_enemy - projected_garrison)


def build_shadow_emergency_defense(
    sources,
    my_planets,
    projection,
    remaining_budgets,
    config,
    player,
    current_step,
    is_2p,
    selected,
    END_STEP,
    distance,
    fleet_speed,
):
    _, _, defended_targets = selected_ids(selected)
    candidates = []
    for target in my_planets:
        if target.id in defended_targets:
            continue
        loss_turn = projection.first_loss_turn_by_id.get(target.id)
        if loss_turn is None or loss_turn > min(config.horizon, 8 if is_2p else 6):
            continue
        surplus = projected_attacker_surplus_at_loss(target, projection, loss_turn, player)
        # Keep this layer narrow: only rescue important planets or very near losses.
        if target.production < (5 if is_2p else 6) and loss_turn > 3:
            continue
        need = max(config.min_ships_to_launch, surplus + 2, int(target.production) + 2)
        need = min(need, max(4, int(target.production * 2 + 4)))
        for source in sources:
            if source.id == target.id:
                continue
            budget = remaining_budgets.get(source.id, 0)
            if budget < need:
                continue
            eta = distance(source, target) / fleet_speed(need)
            if eta > loss_turn + 0.15:
                continue
            angle = math.atan2(target.y - source.y, target.x - source.x)
            saved = target.production * (52.0 if is_2p else 38.0) + max(0, END_STEP - current_step - loss_turn) * target.production * 0.08
            score = saved - need * 1.35 - eta * 3.0
            candidates.append(Candidate("shadow_defense", source.id, target.id, angle, int(need), eta, score))
    return candidates


def build_shadow_opening_expansion(
    sources,
    my_planets,
    planets,
    fleets,
    projection,
    remaining_budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    is_2p,
    selected,
    predict_intercept_position,
    crosses_sun,
    capture_floor,
    retake_risk_after_capture,
    enemy_pressure,
):
    if current_step > (55 if is_2p else 42):
        return []
    used_sources, taken_targets, _ = selected_ids(selected)
    # Do not over-expand if the baseline already made several commitments, but
    # allow one extra expansion when the baseline is otherwise too passive.
    if len([cand for cand in selected if cand.kind in ("attack", "counter_snipe")]) >= 2:
        return []

    candidates = []
    neutral_targets = [
        p for p in planets
        if p.owner == -1 and p.id not in comet_ids and p.id not in taken_targets and p.production >= (4 if is_2p else 5)
    ]
    for source in sources:
        budget = remaining_budgets.get(source.id, 0)
        if budget < config.min_ships_to_launch:
            continue
        for target in neutral_targets:
            # First estimate with available budget, then recompute with needed ships.
            tx, ty, eta_probe = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, budget
            )
            if eta_probe > (10.0 if is_2p else 7.5) or crosses_sun(source, tx, ty):
                continue
            needed = capture_floor(target, projection, eta_probe, player, overhead=2)
            if target.production >= 5:
                needed += 1
            if needed > budget or needed < config.min_ships_to_launch:
                continue
            if needed > min(14, int(budget * (0.58 if is_2p else 0.45))):
                continue
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, needed
            )
            if eta > (10.0 if is_2p else 7.5) or crosses_sun(source, tx, ty):
                continue
            post_capture = max(1, int(needed) - int(projection.ships_by_id[target.id][min(len(projection.ships_by_id[target.id])-1, int(math.ceil(eta)))]) )
            if retake_risk_after_capture(target, planets, player, eta, post_capture, is_2p):
                continue
            pressure = enemy_pressure(target, planets, fleets, player, 16 if is_2p else 10)
            if pressure > needed + (3 if is_2p else 2):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            score = target.production * 50.0 - needed * 3.1 - eta * 4.8 - pressure * 0.7
            if target.production >= 5:
                score += 24.0
            candidates.append(Candidate("shadow_expand", source.id, target.id, angle, int(needed), eta, score))
    return candidates


def build_shadow_finish_attack(
    sources,
    my_planets,
    planets,
    fleets,
    projection,
    remaining_budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    is_2p,
    powers,
    selected,
    predict_intercept_position,
    crosses_sun,
    capture_floor,
    retake_risk_after_capture,
):
    # Disabled in V4.26: V4.25 gained placement wins but lost average score,
    # and this was the only shadow action that could spend large fleets in
    # already-volatile midgame positions.
    return []

    # Only late/midgame, only when we have a material advantage, and only against
    # high-production enemy planets.  This avoids changing the baseline opening.
    if current_step < (135 if is_2p else 100) or current_step > 420:
        return []
    my_stats = powers.get(int(player), {})
    enemy_stats = [v for k, v in powers.items() if int(k) != int(player)]
    if not enemy_stats:
        return []
    strongest_enemy = max(enemy_stats, key=lambda e: e.get("power", 0))
    if my_stats.get("power", 0) < strongest_enemy.get("power", 0) * (1.08 if is_2p else 1.18):
        return []

    used_sources, taken_targets, _ = selected_ids(selected)
    candidates = []
    targets = [
        p for p in planets
        if p.owner not in (-1, player) and p.id not in comet_ids and p.id not in taken_targets and p.production >= (5 if is_2p else 6)
    ]
    for source in sources:
        if source.id in used_sources:
            continue
        budget = remaining_budgets.get(source.id, 0)
        if budget < max(8, config.min_ships_to_launch):
            continue
        for target in targets:
            tx, ty, eta_probe = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, budget
            )
            if eta_probe > (13.0 if is_2p else 8.0) or crosses_sun(source, tx, ty):
                continue
            needed = capture_floor(target, projection, eta_probe, player, overhead=3)
            hold_margin = 2 + max(0, int(target.production) - 4)
            send = needed + hold_margin
            if send > budget or send > int(budget * (0.62 if is_2p else 0.50)):
                continue
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send
            )
            if eta > (13.0 if is_2p else 8.0) or crosses_sun(source, tx, ty):
                continue
            if retake_risk_after_capture(target, planets, player, eta, hold_margin, is_2p):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            score = target.production * 70.0 - send * 2.0 - eta * 3.4
            candidates.append(Candidate("shadow_finish", source.id, target.id, angle, int(send), eta, score))
    return candidates


def is_behind_for_breakout(powers, player, is_2p, current_step, has_large_idle=False):
    if not is_2p:
        return False
    my_stats = powers.get(int(player), {})
    enemies = [stats for owner, stats in powers.items() if int(owner) != int(player)]
    if not enemies:
        return False
    strongest = max(enemies, key=lambda stats: stats.get("power", 0))
    my_power = my_stats.get("power", 0)
    enemy_power = strongest.get("power", 0)
    my_production = my_stats.get("production", 0)
    enemy_production = strongest.get("production", 0)
    if enemy_power <= 0:
        return False
    if current_step >= 220 and enemy_production > 0 and my_production < enemy_production * 0.95:
        return True
    if current_step >= 260 and has_large_idle and enemy_production > my_production:
        return True
    if current_step >= 170 and enemy_production > 0 and my_production + 6 < enemy_production * 0.85:
        return True
    if my_power < enemy_power * 0.78:
        return True
    return my_production + 4 < enemy_production * 0.80


def build_desperation_breakout(
    sources,
    planets,
    projection,
    remaining_budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    is_2p,
    powers,
    selected,
    predict_intercept_position,
    crosses_sun,
    capture_floor,
):
    if current_step < 135 or current_step > 485:
        return []
    has_large_idle = any(int(source.ships) >= 95 for source in sources)
    if not is_behind_for_breakout(powers, player, is_2p, current_step, has_large_idle):
        return []
    attack_count = sum(1 for cand in selected if cand.kind in ("attack", "counter_snipe", "shadow_finish", "breakout"))
    if attack_count >= 1 and current_step < 240:
        return []

    _, taken_targets, _ = selected_ids(selected)
    targets = [
        planet for planet in planets
        if planet.owner not in (-1, player)
        and planet.id not in comet_ids
        and planet.id not in taken_targets
        and planet.production >= (2 if current_step >= 280 else 3)
    ]
    if not targets:
        return []

    candidates = []
    for source in sources:
        reserve_break_budget = 0
        if int(source.ships) >= 60:
            reserve_break_budget = int(source.ships * (0.50 if current_step < 300 else 0.68))
        budget = max(remaining_budgets.get(source.id, 0), reserve_break_budget)
        if budget < max(18, config.min_ships_to_launch * 3):
            continue
        for target in targets:
            probe_ships = max(12, min(budget, int(budget * 0.55)))
            tx, ty, eta_probe = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, probe_ships
            )
            if eta_probe > 48.0 or crosses_sun(source, tx, ty):
                continue
            needed = capture_floor(target, projection, eta_probe, player, overhead=3)
            can_capture = needed <= budget
            if can_capture:
                send = min(budget, needed + max(6, min(28, int(target.production * 5))))
            else:
                # When we are far behind, a large idle stack should still apply
                # pressure instead of waiting forever for a perfect capture.
                if target.production < 3 or budget < 24:
                    continue
                send = max(20, min(budget, int(budget * (0.72 if current_step >= 260 else 0.55))))

            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, send
            )
            if eta > 48.0 or crosses_sun(source, tx, ty):
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            score = (
                target.production * (86.0 if can_capture else 58.0)
                + min(60, int(target.ships)) * 0.42
                - send * (0.42 if can_capture else 0.18)
                - eta * 1.6
            )
            if can_capture:
                score += 42.0
            candidates.append(Candidate("breakout", source.id, target.id, angle, int(send), eta, score))
    return candidates


def append_breakout(selected, candidates, remaining_budgets):
    used_sources, taken_targets, _ = selected_ids(selected)
    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if cand.score <= 20.0:
            break
        if cand.source_id in used_sources:
            continue
        if cand.target_id in taken_targets:
            continue
        selected.append(cand)
        remaining_budgets[cand.source_id] = max(0, remaining_budgets.get(cand.source_id, 0) - cand.ships)
        break
    return selected


def greedy_append_shadow(selected, candidates, remaining_budgets, max_extra, score_floor=95.0):
    if max_extra <= 0:
        return selected
    used_sources, taken_targets, defended_targets = selected_ids(selected)
    extra = 0
    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if extra >= max_extra:
            break
        if cand.score <= score_floor:
            break
        if cand.ships > remaining_budgets.get(cand.source_id, 0):
            continue
        # V4.30: a source used by the baseline may still spend its remaining
        # safe budget.  The remaining_budgets check above protects the reserve.
        if cand.kind != "shadow_defense" and cand.target_id in taken_targets:
            continue
        if cand.kind == "shadow_defense" and cand.target_id in defended_targets:
            continue
        selected.append(cand)
        remaining_budgets[cand.source_id] = max(0, remaining_budgets.get(cand.source_id, 0) - cand.ships)
        used_sources.add(cand.source_id)
        if cand.kind != "shadow_defense":
            taken_targets.add(cand.target_id)
        else:
            defended_targets.add(cand.target_id)
        extra += 1
    return selected


def shadow_score_floor(selected, powers, player, current_step, is_2p):
    my_stats = powers.get(int(player), {})
    enemy_power = max(
        (stats.get("power", 0) for owner, stats in powers.items() if int(owner) != int(player)),
        default=0,
    )
    my_power = my_stats.get("power", 0)
    attacks = len([cand for cand in selected if cand.kind in ("attack", "counter_snipe")])
    if current_step < (75 if is_2p else 55) and attacks == 0:
        return 54.0
    if enemy_power > 0 and my_power < enemy_power * (0.88 if is_2p else 0.82):
        return 62.0
    return 88.0


def shadow_extra_limit(selected, powers, player, current_step, is_2p):
    attacks = len([cand for cand in selected if cand.kind in ("attack", "counter_snipe")])
    if current_step < (75 if is_2p else 55) and attacks == 0:
        return 2
    my_stats = powers.get(int(player), {})
    enemy_power = max(
        (stats.get("power", 0) for owner, stats in powers.items() if int(owner) != int(player)),
        default=0,
    )
    if enemy_power > 0 and my_stats.get("power", 0) < enemy_power * 0.82:
        return 2
    return 1
