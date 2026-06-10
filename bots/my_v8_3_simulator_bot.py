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
import my_orbit.candidates as _cand
import my_orbit.lite_planner as _lite

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
    max_actions: int = 8
    roi_threshold: float = 0.1
    min_ships_to_launch: int = 4
    regroup_enabled: bool = True
    regroup_distance: float = 7.0
    regroup_threshold: float = 0.25
    reserve_margin: int = 2


CONFIG_2P = PlannerConfig(roi_threshold=0.1, defense_horizon=26)
CONFIG_4P = PlannerConfig(
    horizon=13,
    defense_horizon=16,
    max_sources=6,
    max_targets=10,
    max_actions=5,
    roi_threshold=0.1,
    regroup_distance=6.0,
    regroup_threshold=0.25,
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
    return _safe.frontline_reserve_bonus(planet, planets, player, current_step, is_2p, distance)
def safe_drain(source, projection, planets, player, config, current_step, is_2p):
    return _safe.safe_drain(
        source, projection, planets, player, config, current_step, is_2p,
        base_reserve, frontline_reserve_bonus
    )


def capture_floor(target, projection, eta, player, overhead=1):
    return _cap.capture_floor(target, projection, eta, player, overhead)


def agent(obs):
    try:
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
        sources = sorted(
            my_planets,
            key=lambda p: (int(p.ships), int(p.production)),
            reverse=True,
        )[: config.max_sources]
        budgets = {
            source.id: safe_drain(source, projection, planets, player, config, current_step, is_2p)
            for source in sources
        }

        lite_candidates = _lite.build_lite_candidates(
            my_planets,
            planets,
            budgets,
            config,
            player,
            current_step,
            angular_velocity,
            initial_planets,
            comet_ids,
            is_2p,
            projection,
            capture_floor,
            predict_intercept_position,
        )
        
        # DEBUG
        if lite_candidates:
            with open("v8_3_debug.log", "a") as f:
                f.write(f"Step {current_step}: {len(lite_candidates)} candidates generated. Best score: {max(c.score for c in lite_candidates)}\n")
            
        selected = _sel.greedy_select(lite_candidates, dict(budgets), config)
        
        spent = {source.id: 0 for source in sources}
        for cand in selected:
            spent[cand.source_id] = spent.get(cand.source_id, 0) + cand.ships

        regroup_budgets = {
            source.id: max(0, budgets.get(source.id, 0) - spent.get(source.id, 0))
            for source in sources
        }
        regroup_candidates = _cand.build_regroup_candidates(
            my_planets, planets, fleets, regroup_budgets, config, player, current_step, angular_velocity, initial_planets, comet_ids,
            _mod.Candidate,
            lambda p, ps, fs, plr, hz: _score.enemy_pressure(p, ps, fs, plr, hz, distance, fleet_speed, fleet_eta_to_planet),
            distance,
            predict_intercept_position,
            validate_intercept_window,
        )
        if regroup_candidates:
            selected.extend(_sel.greedy_select(regroup_candidates, regroup_budgets, config))

        return [[cand.source_id, cand.angle, int(cand.ships)] for cand in selected if cand.ships > 0]
    except Exception as e:
        import traceback
        with open("agent_crash.log", "w") as f:
            traceback.print_exc(file=f)
        raise
