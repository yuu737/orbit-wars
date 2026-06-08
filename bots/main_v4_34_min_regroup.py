"""
Orbit Wars - V4 Submit Candidate

Self-owned planner-style branch inspired by the structure of strong public bots,
but implemented independently:
- Project short-horizon planet ownership/garrisons.
- Compute safe source drain from the projection.
- Build attack/defense/regroup candidates.
- Add reactive counter-snipe candidates against enemy neutral captures.
- Select non-conflicting candidates greedily.

Submission candidate settings:
- Based on V4.1 counter-snipe planner.
- Uses the safer V4.5 `roi_threshold=2.2` parameter branch.
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
    max_sources: int = 10
    max_targets: int = 12
    max_actions: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: int = 4
    regroup_enabled: bool = True
    regroup_distance: float = 7.0
    regroup_threshold: float = 9.0
    reserve_margin: int = 2


CONFIG_2P = PlannerConfig(roi_threshold=2.2)
CONFIG_4P = PlannerConfig(
    horizon=13,
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

    for turn in range(1, min(config.horizon, len(owner_traj) - 1) + 1):
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


def target_shortlist(my_planets, targets, planets, config):
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
    if target.production >= 5:
        score += 34.0
    elif target.production >= 4:
        score += 16.0
    return score


def build_attack_candidates(
    sources,
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
            angle = math.atan2(target_y - source.y, target_x - source.x)
            cand = Candidate("attack", source.id, target.id, angle, int(needed), eta, 0.0)
            cand.score = score_candidate(cand, target, projection, planets, fleets, player, current_step, is_2p, powers)
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


def build_defense_candidates(sources, my_planets, projection, budgets, config, player, current_step):
    candidates = []
    for target in my_planets:
        loss_turn = projection.first_loss_turn_by_id.get(target.id)
        if loss_turn is None or loss_turn > config.horizon:
            continue
        need = max(4, int(target.production * 2 + 4))
        for source in sources:
            if source.id == target.id:
                continue
            budget = budgets.get(source.id, 0)
            if budget < need:
                continue
            eta = distance(source, target) / fleet_speed(need)
            if eta > loss_turn + 0.75:
                continue
            angle = math.atan2(target.y - source.y, target.x - source.x)
            saved = target.production * 45.0 + max(0, END_STEP - current_step - loss_turn) * target.production * 0.12
            score = saved - need * 0.8 - eta * 2.0
            candidates.append(Candidate("defense", source.id, target.id, angle, int(need), eta, score))
    return candidates


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
        if cand.kind == "defense" and cand.target_id in used_sources:
            continue

        selected.append(cand)
        budgets[cand.source_id] = max(0, budgets.get(cand.source_id, 0) - cand.ships)
        used_sources.add(cand.source_id)
        if cand.kind != "regroup":
            target_taken.add(cand.target_id)
        if cand.kind == "defense":
            defended_targets.add(cand.target_id)

    return selected


def build_regroup_candidates(my_planets, planets, fleets, budgets, config, player):
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
            eta = dist / fleet_speed(send)
            angle = math.atan2(target.y - source.y, target.x - source.x)
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


def build_shadow_micro_regroup(
    sources,
    my_planets,
    planets,
    fleets,
    remaining_budgets,
    config,
    player,
    current_step,
    is_2p,
    selected,
):
    if current_step < (55 if is_2p else 42) or current_step > (90 if is_2p else 70):
        return []
    used_sources, taken_targets, defended_targets = selected_ids(selected)
    pressures = {
        planet.id: enemy_pressure(planet, planets, fleets, player, 20 if is_2p else 14)
        for planet in my_planets
    }
    candidates = []
    for target in my_planets:
        if target.id in taken_targets or target.id in defended_targets:
            continue
        if target.production < (4 if is_2p else 5):
            continue
        target_pressure = pressures.get(target.id, 0.0)
        desired = int(target.production * (3.5 if is_2p else 4.0) + target_pressure * 0.06)
        gap = desired - int(target.ships)
        if gap < config.min_ships_to_launch + 4:
            continue
        for source in sources:
            if source.id == target.id or source.id in used_sources:
                continue
            budget = remaining_budgets.get(source.id, 0)
            if budget < config.min_ships_to_launch:
                continue
            source_pressure = pressures.get(source.id, 0.0)
            if source_pressure + (10 if is_2p else 7) >= target_pressure:
                continue
            dist = distance(source, target)
            if dist > (10.0 if is_2p else 7.0):
                continue
            if crosses_sun(source, target.x, target.y):
                continue
            send = min(config.min_ships_to_launch, budget, max(config.min_ships_to_launch, gap // 3))
            if send < config.min_ships_to_launch:
                continue
            eta = dist / fleet_speed(send)
            score = (
                target.production * 20.0
                + min(gap, 16) * 1.2
                + (target_pressure - source_pressure) * 0.6
                - send * 1.2
                - eta * 2.3
            )
            angle = math.atan2(target.y - source.y, target.x - source.x)
            candidates.append(Candidate("shadow_micro_regroup", source.id, target.id, angle, int(send), eta, score))
    return candidates


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

    projection = project_planet_states(planets, fleets, player, config.horizon)
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

    shortlisted_targets = target_shortlist(my_planets, targets, planets, config)
    candidates = []
    candidates.extend(build_defense_candidates(sources, my_planets, projection, budgets, config, player, current_step))
    candidates.extend(
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
    candidates.extend(
        build_attack_candidates(
            sources,
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

    selected = greedy_select(candidates, dict(budgets), config)
    spent = {source.id: 0 for source in sources}
    for cand in selected:
        spent[cand.source_id] = spent.get(cand.source_id, 0) + cand.ships

    regroup_budgets = {
        source.id: max(0, budgets.get(source.id, 0) - spent.get(source.id, 0))
        for source in sources
    }
    regroup_candidates = build_regroup_candidates(my_planets, planets, fleets, regroup_budgets, config, player)
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
    shadow_candidates.extend(
        build_shadow_micro_regroup(
            sources,
            my_planets,
            planets,
            fleets,
            shadow_budgets,
            config,
            player,
            current_step,
            is_2p,
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

    return [[cand.source_id, cand.angle, int(cand.ships)] for cand in selected if cand.ships > 0]
