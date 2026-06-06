"""
Orbit Wars - V4.6 Opening/Defense Guard branch

Experimental wrapper around V4.1. It keeps the planner architecture intact but
patches two planner decisions:
- Opening score shaping to avoid low-production expansion traps.
- Early projected-loss defense sizing for valuable owned planets.
"""

import math

from bots import main_v4_1_counter_snipe as base


base.CONFIG_2P = base.PlannerConfig(
    roi_threshold=2.0,
    max_targets=13,
    reserve_margin=2,
)
base.CONFIG_4P = base.PlannerConfig(
    horizon=13,
    max_sources=6,
    max_targets=10,
    max_actions=5,
    roi_threshold=1.8,
    regroup_distance=6.0,
    regroup_threshold=11.0,
    reserve_margin=3,
)


_original_score_candidate = base.score_candidate


def guarded_score_candidate(
    candidate,
    target,
    projection,
    planets,
    fleets,
    player,
    current_step,
    is_2p,
    powers,
):
    score = _original_score_candidate(
        candidate,
        target,
        projection,
        planets,
        fleets,
        player,
        current_step,
        is_2p,
        powers,
    )
    if candidate.kind != "attack":
        return score

    my_prod = sum(int(planet.production) for planet in planets if planet.owner == player)
    my_planets = sum(1 for planet in planets if planet.owner == player)

    if current_step <= 80:
        if target.owner == -1:
            if target.production >= 5:
                score += 65.0
            elif target.production >= 4:
                score += 36.0
            elif target.production <= 1:
                score -= 95.0
            elif target.production <= 2 and candidate.eta > 7.0:
                score -= 48.0

            if my_prod < 8 and target.production >= 4:
                score += 55.0
            if my_prod < 8 and target.production <= 2:
                score -= 70.0

        elif target.owner != player:
            if my_prod < 10 and target.production < 5:
                score -= 85.0
            if my_planets <= 2 and target.production < 5:
                score -= 45.0

    if current_step <= 45 and target.owner != -1 and target.production < 5:
        score -= 55.0

    return score


def projected_defense_need(target, projection, loss_turn):
    ships_by_turn = projection.ships_by_id.get(target.id, [])
    projected_ships = 0
    if 0 <= loss_turn < len(ships_by_turn):
        projected_ships = int(ships_by_turn[loss_turn])
    base_need = max(4, -projected_ships + int(target.production) * 2 + 5)
    if target.production >= 5:
        base_need += 5
    elif target.production >= 4:
        base_need += 3
    if loss_turn <= 5:
        base_need += 4
    return int(base_need)


def guarded_build_defense_candidates(sources, my_planets, projection, budgets, config, player, current_step):
    candidates = []
    for target in my_planets:
        loss_turn = projection.first_loss_turn_by_id.get(target.id)
        if loss_turn is None or loss_turn > config.horizon:
            continue
        if current_step < 90 and target.production <= 2 and loss_turn > 7:
            continue

        need = projected_defense_need(target, projection, int(loss_turn))
        for source in sources:
            if source.id == target.id:
                continue
            budget = budgets.get(source.id, 0)
            if budget < need:
                continue
            eta = base.distance(source, target) / base.fleet_speed(need)
            if eta > loss_turn + 0.75:
                continue
            angle = math.atan2(target.y - source.y, target.x - source.x)
            urgency = max(0.0, config.horizon - float(loss_turn))
            saved = target.production * 62.0 + urgency * 18.0 + int(target.ships) * 0.6
            if current_step < 100 and target.production >= 4:
                saved += 55.0
            score = saved - need * 0.75 - eta * 2.2
            candidates.append(base.Candidate("defense", source.id, target.id, angle, need, eta, score))
    return candidates


base.score_candidate = guarded_score_candidate
base.build_defense_candidates = guarded_build_defense_candidates


def agent(obs):
    return base.agent(obs)
