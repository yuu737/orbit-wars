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

import my_orbit.world as _w
import my_orbit.projection as _proj
import my_orbit.safety as _safe
import my_orbit.capture as _cap
import my_orbit.models as _mod
import my_orbit.scoring as _score
import my_orbit.selection as _sel
import my_orbit.strategy as _strat
import my_orbit.candidates as _cand

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


Candidate = _mod.Candidate
MultiCandidate = _mod.MultiCandidate


def fleet_speed(ships):
    return _w.fleet_speed(ships)


def distance_xy(ax, ay, bx, by):
    return _w.distance_xy(ax, ay, bx, by)

def distance(a, b):
    return _w.distance(a, b)

def angle_diff(a, b):
    return _w.angle_diff(a, b)


def infer_num_players(raw_initial_planets, raw_planets, raw_fleets, player):
    return _w.infer_num_players(raw_initial_planets, raw_planets, raw_fleets, player)


def is_rotating(initial_planet):
    return _w.is_rotating(initial_planet)


def predicted_planet_position(planet, initial_planet, step, angular_velocity, comet_ids):
    return _w.predicted_planet_position(
        planet, initial_planet, step, angular_velocity, comet_ids
    )
def point_to_segment_distance(px, py, ax, ay, bx, by):
    return _w.point_to_segment_distance(px, py, ax, ay, bx, by)

def crosses_sun(source, target_x, target_y):
    return _w.crosses_sun(source, target_x, target_y)

def estimate_arrival_turns(source, target_x, target_y, ships):
    return _w.estimate_arrival_turns(source, target_x, target_y, ships)

def predict_intercept_position(
    source,
    target,
    initial_planet,
    current_step,
    angular_velocity,
    comet_ids,
    ships,
):
    return _w.predict_intercept_position(
        source,
        target,
        initial_planet,
        current_step,
        angular_velocity,
        comet_ids,
        ships,
    )


def validate_intercept_window(
    source,
    target,
    initial_planet,
    current_step,
    angular_velocity,
    comet_ids,
    ships,
    target_x,
    target_y,
    eta,
):
    return _w.validate_intercept_window(
        source,
        target,
        initial_planet,
        current_step,
        angular_velocity,
        comet_ids,
        ships,
        target_x,
        target_y,
        eta,
    )

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
    return _w.attack_ambiguity_penalty(
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


def fleet_points_toward_planet(fleet, planet):
    return _w.fleet_points_toward_planet(fleet, planet)


def fleet_eta_to_planet(fleet, planet):
    return _w.fleet_eta_to_planet(fleet, planet)

def resolve_planet_combat(owner, garrison, arrivals):
    return _proj.resolve_planet_combat(owner, garrison, arrivals)


def project_planet_states(planets, fleets, player, horizon):
    return _proj.project_planet_states(
        planets,
        fleets,
        player,
        horizon,
        Projection,
        fleet_points_toward_planet,
        fleet_eta_to_planet,
    )


def base_reserve(planet, current_step, is_2p):
    return _safe.base_reserve(planet, current_step, is_2p)

def frontline_reserve_bonus(planet, planets, player, current_step, is_2p):
    return _safe.frontline_reserve_bonus(
        planet,
        planets,
        player,
        current_step,
        is_2p,
        distance,
    )


def safe_drain(source, projection, planets, player, config, current_step, is_2p):
    return _safe.safe_drain(
        source,
        projection,
        planets,
        player,
        config,
        current_step,
        is_2p,
        base_reserve,
        frontline_reserve_bonus,
    )


def capture_floor(target, projection, eta, player, overhead=1):
    return _cap.capture_floor(target, projection, eta, player, overhead)



def player_power(planets, fleets):
    return _score.player_power(planets, fleets)



def enemy_pressure(planet, planets, fleets, player, horizon):
    return _score.enemy_pressure(
        planet,
        planets,
        fleets,
        player,
        horizon,
        distance,
        fleet_speed,
        fleet_eta_to_planet,
    )


def orbital_ring_value(planet):
    return _score.orbital_ring_value(
        planet,
        distance_xy,
        CENTER_X,
        CENTER_Y,
    )


def friendly_support_count(target, my_planets, max_dist):
    return _score.friendly_support_count(
        target,
        my_planets,
        max_dist,
        distance,
    )


def selective_hold_target(target, my_planets, current_step, is_2p):
    return _score.selective_hold_target(
        target,
        my_planets,
        current_step,
        is_2p,
        friendly_support_count,
    )


def frontier_gate_score(source, target, my_planets, planets, fleets, player, current_step, is_2p):
    return _score.frontier_gate_score(
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
    )


def target_shortlist(my_planets, targets, planets, config):
    return _score.target_shortlist(
        my_planets,
        targets,
        planets,
        config,
        distance,
    )

def score_candidate(candidate, target, projection, planets, fleets, player, current_step, is_2p, powers):
    return _score.score_candidate(
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
    )


def light_neutral_overexpand_penalty(source, target, my_planets, planets, fleets, player, current_step, is_2p):
    return _score.light_neutral_overexpand_penalty(
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
    return _cand.build_attack_candidates(
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
    )


def predicted_enemy_capture_surplus(target, enemy_fleet, enemy_eta):
    return _cap.predicted_enemy_capture_surplus(target, enemy_fleet, enemy_eta)


def capture_holds_after_counter_snipe(target, planets, player, arrival_turn, ships_sent, needed, is_2p):
    return _cap.capture_holds_after_counter_snipe(
        target,
        planets,
        player,
        arrival_turn,
        ships_sent,
        needed,
        is_2p,
        distance,
        fleet_speed,
    )

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
    return _cand.build_counter_snipe_candidates(
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
    )


def build_defense_candidates(
    sources, my_planets, projection, budgets, config, player, current_step, angular_velocity, initial_planets, comet_ids
):
    return _cand.build_defense_candidates(
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
        END_STEP,
    )


def build_urgent_hold_candidates(
    my_planets, projection, budgets, config, player, current_step, is_2p, angular_velocity, initial_planets, comet_ids
):
    return _cand.build_urgent_hold_candidates(
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
    )


def build_frontline_relay_candidates(
    my_planets, planets, fleets, budgets, config, player, current_step, is_2p, angular_velocity, initial_planets, comet_ids
):
    return _cand.build_frontline_relay_candidates(
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
    )


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
    return _sel.build_multisource_capture_candidates(
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
        distance,
        predict_intercept_position,
        crosses_sun,
        capture_floor,
        Candidate,
        MultiCandidate,
    )




def append_multisource_capture(selected, multi_candidates, budgets):
    return _sel.append_multisource_capture(selected, multi_candidates, budgets)


def greedy_select(candidates, budgets, config):
    return _sel.greedy_select(candidates, budgets, config)


def greedy_select_limited(candidates, budgets, max_actions, roi_threshold):
    return _sel.greedy_select_limited(candidates, budgets, max_actions, roi_threshold)


def build_regroup_candidates(my_planets, planets, fleets, budgets, config, player, current_step, angular_velocity, initial_planets, comet_ids):
    return _cand.build_regroup_candidates(
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
    )



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
    return _sel.selected_ids(selected)


def projected_attacker_surplus_at_loss(target, projection, loss_turn, player):
    return _strat.projected_attacker_surplus_at_loss(target, projection, loss_turn, player)


def retake_risk_after_capture(target, planets, player, arrival_turn, post_capture_ships, is_2p):
    return _cap.retake_risk_after_capture(
        target,
        planets,
        player,
        arrival_turn,
        post_capture_ships,
        is_2p,
        distance,
        fleet_speed,
    )


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
    return _strat.build_shadow_emergency_defense(
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
    )


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
    return _strat.build_shadow_opening_expansion(
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
    )


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
    return _strat.build_shadow_finish_attack(
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
    )


def is_behind_for_breakout(powers, player, is_2p, current_step, has_large_idle=False):
    return _strat.is_behind_for_breakout(powers, player, is_2p, current_step, has_large_idle)


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
    return _strat.build_desperation_breakout(
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
    )


def append_breakout(selected, candidates, remaining_budgets):
    return _strat.append_breakout(selected, candidates, remaining_budgets)


def greedy_append_shadow(selected, candidates, remaining_budgets, max_extra, score_floor=95.0):
    return _strat.greedy_append_shadow(selected, candidates, remaining_budgets, max_extra, score_floor)


def shadow_score_floor(selected, powers, player, current_step, is_2p):
    return _strat.shadow_score_floor(selected, powers, player, current_step, is_2p)


def shadow_extra_limit(selected, powers, player, current_step, is_2p):
    return _strat.shadow_extra_limit(selected, powers, player, current_step, is_2p)


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

    shortlisted_targets = target_shortlist(my_planets, targets, planets, config)
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
