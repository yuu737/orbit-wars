"""
Orbit Wars - V4 Submit Candidate

Self-owned planner-style branch inspired by the structure of strong public bots,
but implemented independently:
- Project short-horizon planet ownership/garrisons.
- Compute safe source drain from the projection.
- Build attack/defense/regroup candidates.
- Add reactive counter-snipe candidates against enemy neutral captures.
- Select non-conflicting candidates greedily.

Experimental V6.3 settings:
- Keep the V6.1 frontier-gated unified action-pool structure.
- Add narrow hold-aware attack sizing only for the highest-value captures.
- Tune the branch first on the focused hairate benchmark before broad evaluation.
"""

import math
from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


MAX_SPEED = 6.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 1.0
END_STEP = 500


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 18
    defense_horizon: int = 18
    max_sources: int = 10
    max_targets: int = 12
    max_actions: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: int = 4
    regroup_enabled: bool = True
    regroup_distance: float = 7.0
    regroup_threshold: float = 9.0
    reserve_margin: int = 2


CONFIG_2P = PlannerConfig(roi_threshold=2.2, defense_horizon=26)
CONFIG_4P = PlannerConfig(
    horizon=13,
    defense_horizon=16,
    max_sources=6,
    max_targets=10,
    max_actions=5,
    roi_threshold=2.2,
    regroup_distance=6.0,
    regroup_threshold=11.0,
    reserve_margin=3,
)


@dataclass
class Projection:
    owner_by_id: dict[int, list[int]]
    ships_by_id: dict[int, list[int]]
    incoming_by_id: dict[int, list[dict[int, int]]]
    first_loss_turn_by_id: dict[int, int | None]


@dataclass
class Candidate:
    kind: str
    source_id: int
    target_id: int
    angle: float
    ships: int
    eta: float
    score: float


@dataclass
class MultiCandidate:
    kind: str
    target_id: int
    orders: list[Candidate]
    eta: float
    score: float


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


def validate_intercept_window(source, target, initial_planet, current_step, angular_velocity, comet_ids, ships, target_x, target_y, eta):
    speed = fleet_speed(max(1, ships))
    direction = math.atan2(target_y - source.y, target_x - source.x)
    best_gap = float("inf")
    for dt in (-1.5, -0.75, 0.0, 0.75, 1.5):
        probe_eta = max(0.0, eta + dt)
        fx = source.x + math.cos(direction) * speed * probe_eta
        fy = source.y + math.sin(direction) * speed * probe_eta
        px, py = predicted_planet_position(
            target,
            initial_planet,
            current_step + probe_eta,
            angular_velocity,
            comet_ids,
        )
        best_gap = min(best_gap, distance_xy(fx, fy, px, py) - target.radius)
    return best_gap <= 0.85


def attack_ambiguity_penalty(
    source,
    target,
    planets,
    initial_planets,
    current_step,
    angular_velocity,
    comet_ids,
    target_x,
    target_y,
):
    target_heading = math.atan2(target_y - source.y, target_x - source.x)
    target_dist = distance_xy(source.x, source.y, target_x, target_y)
    worst = 0.0
    for other in planets:
        if other.id in (source.id, target.id):
            continue
        ox, oy = predicted_planet_position(
            other,
            initial_planets.get(other.id),
            current_step,
            angular_velocity,
            comet_ids,
        )
        other_dist = distance_xy(source.x, source.y, ox, oy)
        if other_dist >= target_dist - 0.5:
            continue
        other_heading = math.atan2(oy - source.y, ox - source.x)
        heading_gap = angle_diff(target_heading, other_heading)
        lane_width = math.atan2(other.radius + 0.35, max(1.0, other_dist))
        overlap = lane_width - heading_gap
        if overlap <= 0.0:
            continue
        penalty = overlap * 210.0 + max(0.0, target_dist - other_dist) * 0.45
        if other.owner == -1:
            penalty *= 0.9
        elif other.owner == source.owner:
            penalty *= 0.7
        else:
            penalty *= 1.15
        worst = max(worst, penalty)
    return worst


def fleet_points_toward_planet(fleet, planet):
    heading = math.atan2(planet.y - fleet.y, planet.x - fleet.x)
    dist = distance_xy(fleet.x, fleet.y, planet.x, planet.y)
    tolerance = 0.22 + min(0.30, planet.radius / max(8.0, dist))
    return angle_diff(fleet.angle, heading) <= tolerance


def fleet_eta_to_planet(fleet, planet):
    return distance_xy(fleet.x, fleet.y, planet.x, planet.y) / fleet_speed(fleet.ships)


def resolve_planet_combat(owner, garrison, arrivals):
    arrivals = {int(k): int(v) for k, v in arrivals.items() if int(v) > 0}
    if not arrivals:
        return int(owner), max(0, int(garrison))

    ranked = sorted(arrivals.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = ranked[0]
    second_ships = ranked[1][1] if len(ranked) > 1 else 0
    survivor = top_ships - second_ships
    if survivor <= 0:
        return int(owner), max(0, int(garrison))

    if top_owner == owner:
        return int(owner), max(0, int(garrison) + survivor)

    if survivor > garrison:
        return int(top_owner), int(survivor - garrison)
    return int(owner), int(garrison - survivor)


def project_planet_states(planets, fleets, player, horizon):
    owner_by_id = {}
    ships_by_id = {}
    incoming_by_id = {}
    first_loss_turn_by_id = {}

    for planet in planets:
        owner_by_id[planet.id] = [int(planet.owner)] + [int(planet.owner)] * horizon
        ships_by_id[planet.id] = [int(planet.ships)] + [int(planet.ships)] * horizon
        incoming_by_id[planet.id] = [dict() for _ in range(horizon + 1)]
        first_loss_turn_by_id[planet.id] = None

    planet_by_id = {planet.id: planet for planet in planets}
    for fleet in fleets:
        best_planet = None
        best_eta = None
        for planet in planets:
            if not fleet_points_toward_planet(fleet, planet):
                continue
            eta = fleet_eta_to_planet(fleet, planet)
            if eta <= horizon and (best_eta is None or eta < best_eta):
                best_planet = planet
                best_eta = eta

        if best_planet is None or best_eta is None:
            continue

        turn = max(1, min(horizon, int(math.ceil(best_eta))))
        arrivals = incoming_by_id[best_planet.id][turn]
        arrivals[int(fleet.owner)] = arrivals.get(int(fleet.owner), 0) + int(fleet.ships)

    for planet_id, planet in planet_by_id.items():
        owner = int(planet.owner)
        ships = int(planet.ships)
        for turn in range(1, horizon + 1):
            if owner >= 0:
                ships += int(planet.production)

            owner, ships = resolve_planet_combat(owner, ships, incoming_by_id[planet_id][turn])
            owner_by_id[planet_id][turn] = owner
            ships_by_id[planet_id][turn] = ships
            if first_loss_turn_by_id[planet_id] is None and planet.owner == player and owner != player:
                first_loss_turn_by_id[planet_id] = turn

    return Projection(owner_by_id, ships_by_id, incoming_by_id, first_loss_turn_by_id)


def base_reserve(planet, current_step, is_2p):
    if current_step < 90:
        return max(1, planet.production // 2 + 1) if is_2p else max(2, planet.production)
    if current_step < 160:
        return max(3, planet.production + 1) if is_2p else max(5, planet.production * 2)
    if current_step > 430:
        return max(7, planet.production * 3) if is_2p else max(10, planet.production * 4)
    return max(4, planet.production * 2) if is_2p else max(6, planet.production * 3)


def frontline_reserve_bonus(planet, planets, player, current_step, is_2p):
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


def safe_drain(source, projection, planets, player, config, current_step, is_2p):
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

    reserve = base_reserve(source, current_step, is_2p)
    reserve += frontline_reserve_bonus(source, planets, player, current_step, is_2p)
    reserve += config.reserve_margin
    return max(0, min(int(source.ships) - reserve, drain))


def capture_floor(target, projection, eta, player, overhead=1):
    turn = max(1, min(len(projection.ships_by_id[target.id]) - 1, int(math.ceil(eta))))
    owner = projection.owner_by_id[target.id][turn]
    ships = projection.ships_by_id[target.id][turn]
    if owner == player:
        return 1
    return max(1, int(math.ceil(ships + overhead)))


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


def enemy_pressure(planet, planets, fleets, player, horizon):
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


def orbital_ring_value(planet):
    radius = distance_xy(planet.x, planet.y, CENTER_X, CENTER_Y)
    outer_bonus = max(0.0, radius - 22.0)
    return outer_bonus * 1.8 + planet.production * 4.0


def friendly_support_count(target, my_planets, max_dist):
    return sum(1 for friend in my_planets if friend.id != target.id and distance(friend, target) <= max_dist)


def selective_hold_target(target, my_planets, current_step, is_2p):
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


def frontier_gate_score(source, target, my_planets, planets, fleets, player, current_step, is_2p):
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


def opening_target_bias(source, target, my_planets, initial_planets, current_step, is_2p):
    if current_step > (22 if is_2p else 14):
        return 0.0
    if target.owner != -1:
        return 0.0

    initial = initial_planets.get(target.id)
    support_dist = min(
        (distance(friend, target) for friend in my_planets if friend.id != source.id),
        default=99.0,
    )
    bias = 0.0
    if target.production >= 5:
        bias += 70.0
    elif target.production >= 4:
        bias += 34.0
    elif target.production <= 2:
        bias -= 26.0
    if initial is not None and not is_rotating(initial):
        bias += 18.0
    if support_dist <= (14.0 if is_2p else 10.0):
        bias += 16.0
    elif support_dist > (18.0 if is_2p else 13.0):
        bias -= 18.0
    if orbital_ring_value(target) >= orbital_ring_value(source) + 3.0:
        bias += 12.0
    return bias


def target_shortlist(my_planets, targets, planets, config, initial_planets, current_step, is_2p):
    ranked = []
    for target in targets:
        source = min(my_planets, key=lambda s: distance(s, target))
        nearest = distance(source, target)
        value = target.production * 65.0 - int(target.ships) * 1.4 - nearest * 2.2
        if target.owner != -1:
            value += target.production * 28.0
        if target.production <= 1:
            value -= 45.0
        value += opening_target_bias(source, target, my_planets, initial_planets, current_step, is_2p)
        ranked.append((value, target))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [target for _, target in ranked[: config.max_targets]]


def score_candidate(candidate, target, projection, planets, fleets, player, current_step, is_2p, powers):
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
    if candidate.kind == "attack" and selective_hold_target(target, [p for p in planets if p.owner == player], current_step, is_2p):
        capture_need = capture_floor(target, projection, candidate.eta, player)
        post_capture = max(1, int(candidate.ships) - int(capture_need) + 1)
        if not retake_risk_after_capture(target, planets, player, candidate.eta, post_capture, is_2p):
            score += 16.0 + target.production * 3.0
    if target.production >= 5:
        score += 34.0
    elif target.production >= 4:
        score += 16.0
    return score


def light_neutral_overexpand_penalty(source, target, my_planets, planets, fleets, player, current_step, is_2p):
    if target.owner != -1:
        return 0.0
    if current_step >= 140:
        return 0.0

    target_pressure = enemy_pressure(target, planets, fleets, player, 16 if is_2p else 10)
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
):
    candidates = []
    for source in sources:
        budget = budgets.get(source.id, 0)
        if budget < config.min_ships_to_launch:
            continue
        for target in targets:
            if target.id in comet_ids:
                continue
            target_x, target_y, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, budget
            )
            if eta > config.horizon or crosses_sun(source, target_x, target_y):
                continue
            needed = capture_floor(target, projection, eta, player)
            if needed > budget or needed < config.min_ships_to_launch:
                continue
            if target.owner == -1 and current_step < 120:
                target_pressure = enemy_pressure(target, planets, fleets, player, 16 if is_2p else 10)
                if target.production <= 2 and target_pressure > needed + 4:
                    continue
            send = int(needed)
            if selective_hold_target(target, my_planets, current_step, is_2p):
                hold_margin = 2 if target.owner == -1 else 3
                hold_margin += max(0, int(target.production) - 5)
                padded = int(needed + hold_margin)
                if padded <= budget:
                    post_capture = max(1, padded - int(needed) + 1)
                    if not retake_risk_after_capture(target, planets, player, eta, post_capture, is_2p):
                        send = padded
            angle = math.atan2(target_y - source.y, target_x - source.x)
            cand = Candidate("attack", source.id, target.id, angle, int(send), eta, 0.0)
            cand.score = score_candidate(cand, target, projection, planets, fleets, player, current_step, is_2p, powers)
            cand.score -= light_neutral_overexpand_penalty(
                source, target, my_planets, planets, fleets, player, current_step, is_2p
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


def predicted_enemy_capture_surplus(target, enemy_fleet, enemy_eta):
    garrison = int(target.ships)
    if target.owner >= 0:
        garrison += int(target.production) * int(max(0.0, enemy_eta))
    surplus = int(enemy_fleet.ships) - garrison
    return surplus if surplus > 0 else 0


def capture_holds_after_counter_snipe(target, planets, player, arrival_turn, ships_sent, needed, is_2p):
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
    sources, my_planets, projection, budgets, config, player, current_step, angular_velocity, initial_planets, comet_ids
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
            remaining = max(0, END_STEP - current_step - loss_turn)
            saved = target.production * 58.0 + remaining * target.production * 0.18 + int(target.ships) * 0.6
            if target.production >= 4:
                saved += 45.0
            if loss_turn <= 8:
                saved += 28.0
            score = saved - need * 0.95 - eta * 2.2
            candidates.append(Candidate("defense", source.id, target.id, angle, int(need), eta, score))
    return candidates


def build_urgent_hold_candidates(
    my_planets, projection, budgets, config, player, current_step, is_2p, angular_velocity, initial_planets, comet_ids
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
    my_planets, planets, fleets, budgets, config, player, current_step, is_2p, angular_velocity, initial_planets, comet_ids
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


def build_multisource_capture_candidates(
    sources,
    targets,
    projection,
    budgets,
    config,
    player,
    current_step,
    angular_velocity,
    initial_planets,
    comet_ids,
    is_2p,
    selected,
):
    if not is_2p or current_step < 95 or current_step > 285:
        return []

    used_sources, taken_targets, _ = selected_ids(selected)
    if sum(1 for cand in selected if cand.kind in ("attack", "counter_snipe")) >= 2:
        return []
    candidates = []
    valuable_targets = [
        target for target in targets
        if target.id not in comet_ids
        and target.id not in taken_targets
        and target.production >= 5
    ]

    for target in valuable_targets:
        parts = []
        total_send = 0
        arrival_eta = 0.0
        for source in sorted(sources, key=lambda item: distance(item, target)):
            if source.id in used_sources:
                continue
            budget = budgets.get(source.id, 0)
            if budget < config.min_ships_to_launch:
                continue
            probe = min(budget, max(config.min_ships_to_launch, int(budget * 0.58)))
            tx, ty, eta = predict_intercept_position(
                source, target, initial_planets.get(target.id), current_step, angular_velocity, comet_ids, probe
            )
            if eta > 18.0 or crosses_sun(source, tx, ty):
                continue
            parts.append((eta, source, tx, ty, budget))

        if len(parts) < 2:
            continue
        parts.sort(key=lambda item: item[0])
        planned = []
        for eta, source, tx, ty, budget in parts[:3]:
            needed_at_eta = capture_floor(target, projection, eta, player, overhead=3)
            remaining_need = max(0, needed_at_eta + max(4, int(target.production * 2)) - total_send)
            if remaining_need <= 0:
                break
            send = min(budget, max(config.min_ships_to_launch, min(remaining_need, int(budget * 0.55))))
            if send < config.min_ships_to_launch:
                continue
            angle = math.atan2(ty - source.y, tx - source.x)
            planned.append(Candidate("multi_capture", source.id, target.id, angle, int(send), eta, 0.0))
            total_send += int(send)
            arrival_eta = max(arrival_eta, eta)

        if len(planned) < 2:
            continue
        needed_final = capture_floor(target, projection, arrival_eta, player, overhead=4)
        if total_send < needed_final + max(3, int(target.production)):
            continue
        total_budget = sum(budgets.get(order.source_id, 0) for order in planned)
        if total_send > max(18, int(total_budget * 0.62)):
            continue
        score = (
            target.production * 78.0
            + (36.0 if target.owner != -1 else 18.0)
            - total_send * 1.05
            - arrival_eta * 3.5
        )
        if current_step > 190 and target.owner == -1:
            score -= 35.0
        candidates.append(MultiCandidate("multi_capture", target.id, planned, arrival_eta, score))
    return candidates


def append_multisource_capture(selected, multi_candidates, budgets):
    used_sources, taken_targets, _ = selected_ids(selected)
    for multi in sorted(multi_candidates, key=lambda item: item.score, reverse=True):
        if multi.score <= 35.0:
            break
        if multi.target_id in taken_targets:
            continue
        if any(order.source_id in used_sources for order in multi.orders):
            continue
        if any(order.ships > budgets.get(order.source_id, 0) for order in multi.orders):
            continue
        for order in multi.orders:
            selected.append(order)
            budgets[order.source_id] = max(0, budgets.get(order.source_id, 0) - order.ships)
        break
    return selected


def greedy_select(candidates, budgets, config):
    selected = []
    target_taken = set()
    defended_targets = set()
    used_sources = set()

    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(selected) >= config.max_actions:
            break
        if cand.score <= config.roi_threshold:
            break
        if cand.ships > budgets.get(cand.source_id, 0):
            continue
        if cand.kind != "regroup" and cand.target_id in target_taken:
            continue
        if cand.source_id in defended_targets:
            continue
        if cand.kind in ("defense", "urgent_hold") and cand.target_id in used_sources:
            continue

        selected.append(cand)
        budgets[cand.source_id] = max(0, budgets.get(cand.source_id, 0) - cand.ships)
        used_sources.add(cand.source_id)
        if cand.kind != "regroup":
            target_taken.add(cand.target_id)
        if cand.kind in ("defense", "urgent_hold"):
            defended_targets.add(cand.target_id)

    return selected


def greedy_select_limited(candidates, budgets, max_actions, roi_threshold):
    selected = []
    target_taken = set()
    defended_targets = set()
    used_sources = set()

    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(selected) >= max_actions:
            break
        if cand.score <= roi_threshold:
            break
        if cand.ships > budgets.get(cand.source_id, 0):
            continue
        if cand.kind != "regroup" and cand.target_id in target_taken:
            continue
        if cand.source_id in defended_targets:
            continue
        if cand.kind in ("defense", "urgent_hold") and cand.target_id in used_sources:
            continue

        selected.append(cand)
        budgets[cand.source_id] = max(0, budgets.get(cand.source_id, 0) - cand.ships)
        used_sources.add(cand.source_id)
        if cand.kind != "regroup":
            target_taken.add(cand.target_id)
        if cand.kind in ("defense", "urgent_hold"):
            defended_targets.add(cand.target_id)

    return selected


def build_regroup_candidates(my_planets, planets, fleets, budgets, config, player, current_step, angular_velocity, initial_planets, comet_ids):
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



# ---------------------------------------------------------------------------
# V4.26 score-guarded shadow layer
#
# This layer is intentionally action-preserving: the original planner's selected
# candidates are never rescored, removed, or replaced.  It only spends budget
# that remains after the baseline planner and baseline regroup have already
# chosen their moves.  The goal is to get closer to an upper-compatible upgrade
# while avoiding the regressions caused by changing the core scoring balance.
# ---------------------------------------------------------------------------


def selected_ids(selected):
    used_sources = {cand.source_id for cand in selected}
    taken_targets = {cand.target_id for cand in selected if cand.kind != "regroup"}
    defended_targets = {cand.target_id for cand in selected if cand.kind == "defense"}
    return used_sources, taken_targets, defended_targets


def projected_attacker_surplus_at_loss(target, projection, loss_turn, player):
    arrivals = projection.incoming_by_id[target.id][loss_turn]
    best_enemy = 0
    own_arrival = arrivals.get(int(player), 0)
    for owner, ships in arrivals.items():
        if int(owner) != int(player):
            best_enemy = max(best_enemy, int(ships))
    projected_garrison = int(target.ships) + max(0, loss_turn - 1) * int(target.production) + own_arrival
    return max(0, best_enemy - projected_garrison)


def retake_risk_after_capture(target, planets, player, arrival_turn, post_capture_ships, is_2p):
    # Conservative local test: reject captures that an enemy planet can cheaply
    # retake soon after our arrival.  This is deliberately harsher than the
    # baseline attack scorer and is only used for extra shadow actions.
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
    if not planets:
        return []

    num_players = infer_num_players(raw_initial_planets, raw_planets, raw_fleets, player)
    is_2p = num_players == 2
    config = CONFIG_2P if is_2p else CONFIG_4P

    my_planets = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    if not my_planets:
        return []

    projection = project_planet_states(planets, fleets, player, max(config.horizon, config.defense_horizon))
    powers = player_power(planets, fleets)
    sources = sorted(
        my_planets,
        key=lambda p: (int(p.ships), int(p.production)),
        reverse=True,
    )[: config.max_sources]
    budgets = {
        source.id: safe_drain(source, projection, planets, player, config, current_step, is_2p)
        for source in sources
    }

    shortlisted_targets = target_shortlist(
        my_planets,
        targets,
        planets,
        config,
        initial_planets,
        current_step,
        is_2p,
    )
    reserve_budgets = {
        source.id: max(
            budgets.get(source.id, 0),
            int(source.ships * (0.28 if current_step >= 55 else 0.34))
        )
        for source in my_planets
    }
    unified_budgets = {
        source.id: max(budgets.get(source.id, 0), reserve_budgets.get(source.id, 0))
        for source in my_planets
    }

    planner_candidates = []
    planner_candidates.extend(
        build_frontline_relay_candidates(
            my_planets,
            planets,
            fleets,
            reserve_budgets,
            config,
            player,
            current_step,
            is_2p,
            angular_velocity,
            initial_planets,
            comet_ids,
        )
    )
    planner_candidates.extend(
        build_urgent_hold_candidates(
            my_planets,
            projection,
            reserve_budgets,
            config,
            player,
            current_step,
            is_2p,
            angular_velocity,
            initial_planets,
            comet_ids,
        )
    )
    planner_candidates.extend(
        build_defense_candidates(
            sources, my_planets, projection, budgets, config, player, current_step, angular_velocity, initial_planets, comet_ids
        )
    )
    planner_candidates.extend(
        build_counter_snipe_candidates(
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
        )
    )
    planner_candidates.extend(
        build_attack_candidates(
            sources,
            my_planets,
            shortlisted_targets,
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
        )
    )

    selected = greedy_select(planner_candidates, dict(unified_budgets), config)
    spent = {source.id: 0 for source in sources}
    for cand in selected:
        spent[cand.source_id] = spent.get(cand.source_id, 0) + cand.ships

    multisource_budgets = {
        source.id: max(0, budgets.get(source.id, 0) - spent.get(source.id, 0))
        for source in sources
    }
    selected = append_multisource_capture(
        selected,
        build_multisource_capture_candidates(
            sources,
            shortlisted_targets,
            projection,
            multisource_budgets,
            config,
            player,
            current_step,
            angular_velocity,
            initial_planets,
            comet_ids,
            is_2p,
            selected,
        ),
        multisource_budgets,
    )
    spent = {source.id: 0 for source in sources}
    for cand in selected:
        spent[cand.source_id] = spent.get(cand.source_id, 0) + cand.ships

    regroup_budgets = {
        source.id: max(0, budgets.get(source.id, 0) - spent.get(source.id, 0))
        for source in sources
    }
    regroup_candidates = build_regroup_candidates(
        my_planets, planets, fleets, regroup_budgets, config, player, current_step, angular_velocity, initial_planets, comet_ids
    )
    if regroup_candidates:
        selected.extend(greedy_select(regroup_candidates, regroup_budgets, config))

    # Conservative shadow layer.  Recompute remaining budget after the full
    # baseline plan, then append at most one high-confidence action.  Baseline
    # actions are intentionally left untouched.
    final_spent = {source.id: 0 for source in sources}
    for cand in selected:
        final_spent[cand.source_id] = final_spent.get(cand.source_id, 0) + cand.ships
    shadow_budgets = {
        source.id: max(0, budgets.get(source.id, 0) - final_spent.get(source.id, 0))
        for source in sources
    }
    shadow_candidates = []
    shadow_candidates.extend(
        build_shadow_emergency_defense(
            sources,
            my_planets,
            projection,
            shadow_budgets,
            config,
            player,
            current_step,
            is_2p,
            selected,
        )
    )
    shadow_candidates.extend(
        build_shadow_opening_expansion(
            sources,
            my_planets,
            planets,
            fleets,
            projection,
            shadow_budgets,
            config,
            player,
            current_step,
            angular_velocity,
            initial_planets,
            comet_ids,
            is_2p,
            selected,
        )
    )
    shadow_candidates.extend(
        build_shadow_finish_attack(
            sources,
            my_planets,
            planets,
            fleets,
            projection,
            shadow_budgets,
            config,
            player,
            current_step,
            angular_velocity,
            initial_planets,
            comet_ids,
            is_2p,
            powers,
            selected,
        )
    )
    selected = greedy_append_shadow(
        selected,
        shadow_candidates,
        shadow_budgets,
        max_extra=shadow_extra_limit(selected, powers, player, current_step, is_2p),
        score_floor=shadow_score_floor(selected, powers, player, current_step, is_2p),
    )

    breakout_spent = {source.id: 0 for source in sources}
    for cand in selected:
        breakout_spent[cand.source_id] = breakout_spent.get(cand.source_id, 0) + cand.ships
    breakout_budgets = {
        source.id: max(0, budgets.get(source.id, 0) - breakout_spent.get(source.id, 0))
        for source in sources
    }
    selected = append_breakout(
        selected,
        build_desperation_breakout(
            sources,
            planets,
            projection,
            breakout_budgets,
            config,
            player,
            current_step,
            angular_velocity,
            initial_planets,
            comet_ids,
            is_2p,
            powers,
            selected,
        ),
        breakout_budgets,
    )

    return [[cand.source_id, cand.angle, int(cand.ships)] for cand in selected if cand.ships > 0]
