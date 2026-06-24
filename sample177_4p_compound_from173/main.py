from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from dataclasses import dataclass

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

# ============================================================================
# Phase 0 orbit_lite SPLIT (sample168). Shared/unchanged infra is imported from
# orbit_lite_4p (identical to orbit_lite_2p in Phase 0). The PLANNER modules
# (planner_core + garrison_launch) are dispatched per game by _bind_planner():
# 2P -> orbit_lite_2p, 4P -> orbit_lite_4p. Phase 0 keeps both packages byte-
# identical, so this is behaviour-preserving; Phase 2/3 change orbit_lite_2p /
# orbit_lite_4p independently. (No cross-package isinstance exists, so duck-typed
# objects flow safely between shared infra and the per-mode planner.)
# ============================================================================
from orbit_lite_4p.geometry import fleet_speed
from orbit_lite_4p.intercept_aim import intercept_angle
from orbit_lite_4p.movement import MovementConfig, PlanetMovement
from orbit_lite_4p.movement_step import (
    LaunchEntries,
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite_4p.obs import parse_obs
from orbit_lite_4p.distance_cache import build_distance_cache
from orbit_lite_4p.adapter import single_obs_to_tensor, sparse_action_row_to_moves

import orbit_lite_2p.planner_core as _pc2
import orbit_lite_4p.planner_core as _pc4
import orbit_lite_2p.garrison_launch as _gl2
import orbit_lite_4p.garrison_launch as _gl4

# Names rebound as module globals per game (so existing call sites stay unchanged).
_OL_PLANNER_NAMES = (
    "_candidate_indices", "_empty_entries", "_greedy_select", "_stable_argmax",
    "_plan_regroup", "build_target_shortlist", "capture_floor", "empty_action_row",
    "entries_to_sparse_payload", "largest_initial_player_count", "make_launch_set",
    "reachable_mask", "reinforcement_timing_factor", "safe_drain", "score_candidates",
)
_OL_GL_NAMES = ("LaunchSet", "sparse_launch_flow_delta")


def _bind_planner(player_count: int) -> None:
    """Bind the per-mode planner_core + garrison_launch names into module globals.
    One process == one game == one player_count, so this is set once per game."""
    g = globals()
    pc = _pc4 if int(player_count) >= 4 else _pc2
    gl = _gl4 if int(player_count) >= 4 else _gl2
    for _n in _OL_PLANNER_NAMES:
        g[_n] = getattr(pc, _n)
    for _n in _OL_GL_NAMES:
        g[_n] = getattr(gl, _n)


_bind_planner(4)  # default bind (4P) at import; rebound per game in tensor_action

TOTAL_STEPS = 500


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs."""

    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 7
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    reinforce_size_beta: float = 2.2
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    enable_regroup: bool = True
    max_regroup_time: float = 6.0
    regroup_pressure_delta_min: float = 0.35
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 5
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    terminal_phase_turns: int = 40
    terminal_roi_threshold: float = 1.0
    terminal_max_waves_per_turn: int = 8
    terminal_enable_regroup: bool = False
    # --- exp41: evaluate several commit fractions per (src, tgt) ------------
    size_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0)
    # --- 4P third mode: build a lane anchor before scattering ---------------
    enable_lane_anchor_4p: bool = False
    lane_anchor_turn_limit: int = 115
    lane_anchor_eta_cap: float = 18.0
    lane_anchor_max_extra_targets: int = 6
    lane_anchor_angle_width: float = 1.18
    lane_anchor_bonus: float = 0.46
    lane_anchor_prod_bonus: float = 0.045
    lane_anchor_max_bonus: float = 0.78
    # --- 4P domain race: build a thick outer domain instead of small skirmish ---
    enable_domain_race_4p: bool = False
    domain_race_turn_limit: int = 135
    domain_race_eta_cap: float = 20.0
    domain_race_angle_width: float = 1.08
    domain_race_max_extra_targets: int = 7
    domain_race_bonus: float = 0.54
    domain_race_prod_bonus: float = 0.06
    domain_race_outer_bonus: float = 0.22
    domain_race_cheap_penalty: float = 0.24
    domain_anchor_hold_base: float = 34.0
    domain_anchor_hold_prod: float = 11.0
    domain_anchor_drain_penalty: float = 0.64
    # --- 4P winner path: commit to one outer lane and grow thick anchors ------
    enable_winner_path_4p: bool = False
    winner_path_turn_limit: int = 125
    winner_path_max_extra_targets: int = 8
    winner_path_angle_width: float = 0.92
    winner_path_bonus: float = 0.82
    winner_path_prod_bonus: float = 0.075
    winner_path_outer_bonus: float = 0.30
    winner_path_offlane_cheap_penalty: float = 0.34
    winner_anchor_hold_base: float = 46.0
    winner_anchor_hold_prod: float = 12.5
    winner_anchor_drain_penalty: float = 0.78
    # --- 4P enemy domain block: deny a rival's safe outer cluster -------------
    enable_enemy_domain_block_4p: bool = False
    enemy_block_start_turn: int = 35
    enemy_block_turn_limit: int = 115
    enemy_block_eta_cap: float = 22.0
    enemy_block_max_extra_targets: int = 6
    enemy_block_cluster_radius: float = 40.0
    enemy_block_bonus: float = 0.58
    enemy_block_prod_bonus: float = 0.065
    enemy_block_outer_bonus: float = 0.20
    enemy_block_direct_enemy_penalty: float = 0.42
    # --- sample57 top director: hard 4P anti-brawl + reserved safe growth ----
    enable_top_director_4p: bool = False
    top_director_turn_limit: int = 135
    top_director_central_radius: float = 29.0
    top_director_central_brawl_penalty: float = 1.45
    top_director_nonleader_enemy_penalty: float = 0.78
    top_director_safe_neutral_bonus: float = 0.72
    top_director_outer_bonus: float = 0.22
    top_director_drain_penalty: float = 0.34
    top_director_reserve_start: int = 18
    top_director_reserve_limit: int = 110
    top_director_reserve_roi: float = -0.45
    # --- sample69: use orbit_lite future arrivals as a lightweight response gate
    enable_orbit_response_gate_4p: bool = False
    orbit_response_start: int = 110
    orbit_response_ramp_end: int = 80
    orbit_response_ramp_min: float = 0.35
    orbit_response_turn_limit: int = 170
    orbit_response_window: int = 7
    orbit_response_enemy_frac: float = 0.24
    orbit_response_margin: float = 8.0
    orbit_response_recapture_penalty: float = 0.42
    orbit_response_bad_war_penalty: float = 0.60
    orbit_response_safe_bonus: float = 0.18
    orbit_response_defense_bonus: float = 0.30
    # --- sample96: global control map + observed enemy fleet intent ----------
    enable_influence_map_4p: bool = False
    influence_radius: float = 34.0
    influence_safe_bonus: float = 0.16
    influence_danger_penalty: float = 0.24
    influence_enemy_deep_penalty: float = 0.18
    enable_fleet_intent_4p: bool = False
    intent_horizon: float = 24.0
    intent_angle_width: float = 0.55
    intent_pressure_scale: float = 0.004
    intent_target_penalty: float = 0.30
    intent_defense_bonus: float = 0.24
    # --- sample98: Wave2 phase1, safe post-greedy coordinated follow-ups ----
    enable_coord_followup_4p: bool = False
    coord_start_turn: int = 60
    coord_turn_limit: int = 150
    coord_max_extra: int = 1
    coord_eta_delta: float = 4.0
    coord_min_leftover: float = 24.0
    coord_source_reserve_base: float = 14.0
    coord_source_reserve_prod: float = 3.5
    coord_send_frac: float = 0.22
    coord_send_cap: float = 22.0
    coord_min_target_prod: float = 4.0
    coord_one_ply_gate_4p: bool = False
    coord_one_ply_min_target_prod: float = 4.0
    coord_one_ply_top_sources: int = 5
    coord_one_ply_min_net: float = 5.0
    coord_one_ply_hard_min_net: float = -35.0
    coord_one_ply_response_ratio_cap: float = 1.20
    # --- sample104: Wave3 lightweight mini-rollout score refinement ----------
    enable_mini_rollout_4p: bool = False
    mini_rollout_start: int = 20
    mini_rollout_turn_limit: int = 160
    mini_rollout_top_k: int = 24
    mini_rollout_window: int = 10
    mini_rollout_hold_bonus: float = 0.22
    mini_rollout_fail_penalty: float = 0.55
    mini_rollout_source_strip_penalty: float = 0.22
    mini_rollout_nonleader_enemy_penalty: float = 0.18
    # --- sample109: true one-ply candidate response ------------------------
    enable_true_one_ply_4p: bool = False
    true_one_ply_start: int = 25
    true_one_ply_turn_limit: int = 175
    true_one_ply_top_k: int = 36
    true_one_ply_enemy_frac: float = 0.78
    true_one_ply_enemy_min_ships: float = 4.0
    true_one_ply_enemy_eta_cap: float = 18.0
    true_one_ply_enemy_reserve_base: float = 5.0
    true_one_ply_enemy_reserve_prod: float = 1.6
    true_one_ply_base_weight: float = 0.45
    true_one_ply_net_scale: float = 0.085
    true_one_ply_good_bonus: float = 0.95
    true_one_ply_bad_penalty: float = 1.55
    true_one_ply_bad_net: float = -12.0
    true_one_ply_good_net: float = 18.0
    true_one_ply_hard_bad_net: float = -42.0
    # --- sample157: 4P third-party source-stripping guard ---
    # Real-replay diagnosis (replays_my_4p): 63% of our planet losses are THIRD-PARTY
    # steals -- we drain a planet to attack opponent A and a DIFFERENT opponent B takes
    # the bared planet. This guard, before committing, measures each owned planet's
    # garrison after the cumulative drain of the selected waves against the reachable
    # force of the STRONGEST SINGLE opponent (latent garrison within horizon + in-flight),
    # and DROPS the marginal waves that bare a *holdable* planet below that threat. Hard
    # source-side constraint (removes bad commits only), threat-triggered, savable-gated,
    # drop-only (no forcing / no score bonus / no blanket reserve) -> §1.1 compliant.
    # Subsumes 110's would-be threat_reserve (covers in-flight AND latent). 4P ONLY.
    enable_thirdparty_guard_4p: bool = False
    tp_guard_horizon: float = 10.0
    tp_guard_window: int = 10
    tp_guard_margin: float = 3.0
    tp_guard_vuln_weight: float = 1.0
    tp_guard_drop_sweeps: int = 3
    tp_guard_drop_deadband: float = 0.0
    tp_guard_start: int = 20
    tp_guard_limit: int = 250
    # --- sample170: 4P state-dependent expansion drive ---
    # When we are behind in owned production / planet count, push durable-looking
    # outer neutral captures before the normal 4P safety filters run. This changes
    # the planner's candidate preference, while the existing response/third-party
    # guards still get the final veto on over-stripping sources.
    enable_expansion_drive_4p: bool = False
    expansion_drive_start: int = 8
    expansion_drive_limit: int = 150
    expansion_drive_low_planets: int = 6
    expansion_drive_base_bonus: float = 0.40
    expansion_drive_lag_bonus: float = 0.85
    expansion_drive_prod_bonus: float = 0.16
    expansion_drive_outer_bonus: float = 0.32
    expansion_drive_cheap_bonus: float = 0.22
    expansion_drive_enemy_penalty: float = 0.35
    expansion_drive_center_penalty: float = 0.42
    expansion_drive_pressure_penalty: float = 0.020
    expansion_drive_strip_penalty: float = 0.75
    expansion_drive_min_prod: float = 1.0
    expansion_campaign_radius: float = 16.0
    expansion_campaign_prod_bonus: float = 0.045
    expansion_campaign_cheap_bonus: float = 0.030
    expansion_campaign_risk_penalty: float = 0.006
    # --- sample177: production-COMPOUND drive (Phase 3 deepen) ---
    # The lag-gated drive above fizzles in the competitive zone (need~0), so we plateau at
    # 5-8 planets while one rival snowballs to 16-23 (competition replays). This adds a
    # need-INDEPENDENT, production-weighted pull toward durable neutrals that fires exactly
    # when we are NOT behind (1-need), to keep compounding production past the plateau. The
    # existing center/pressure/strip penalties stay, so it does not over-extend (sample174).
    expansion_drive_compound_bonus: float = 0.0


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def _apply_phase_config(config: ProducerLiteConfig, step: int) -> ProducerLiteConfig:
    if int(step) >= TOTAL_STEPS - int(config.terminal_phase_turns):
        return dataclasses.replace(
            config,
            roi_threshold=float(config.terminal_roi_threshold),
            max_waves_per_turn=int(config.terminal_max_waves_per_turn),
            enable_regroup=bool(config.terminal_enable_regroup),
        )
    return config


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


def _obs_step(obs_tensors: dict) -> int:
    return int(obs_tensors["step"].reshape(-1)[0].item())


def _append_lane_anchor_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    obs,
    obs_tensors: dict,
    cache,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Expose high-upside early lane anchors for hard 4P board types."""
    if int(player_count) < 4 or not bool(config.enable_lane_anchor_4p):
        return target_idx, target_exists, None
    if _obs_step(obs_tensors) > int(config.lane_anchor_turn_limit):
        return target_idx, target_exists, None
    if not bool(source_mask.any()):
        return target_idx, target_exists, None

    P = int(obs.P)
    if P <= 0:
        return target_idx, target_exists, None

    device = obs.device
    dtype = obs.ships.dtype
    d0 = cache.cross_dist[0].to(dtype)
    speed = fleet_speed(obs.ships.to(dtype)).clamp(min=1e-6)
    eta_all = d0 / speed.view(P, 1)
    eta_from_me = torch.where(
        source_mask.view(P, 1),
        eta_all,
        torch.full_like(eta_all, float("inf")),
    ).amin(dim=0)

    enemy_source = obs.is_enemy & obs.alive & (obs.ships >= 4.0)
    if bool(enemy_source.any()):
        enemy_eta_all = d0 / speed.view(P, 1)
        eta_from_enemy = torch.where(
            enemy_source.view(P, 1),
            enemy_eta_all,
            torch.full_like(enemy_eta_all, float("inf")),
        ).amin(dim=0)
    else:
        eta_from_enemy = torch.full((P,), float("inf"), dtype=dtype, device=device)

    planets = obs_tensors["planets"].to(dtype)
    x = planets[:, 2]
    y = planets[:, 3]
    angle = torch.atan2(y - 50.0, x - 50.0)
    owned_idx = torch.nonzero(source_mask, as_tuple=False).flatten()
    base = int(owned_idx[torch.argmax(obs.ships[owned_idx])].item())
    base_angle = angle[base]
    angle_delta = torch.abs((angle - base_angle + math.pi) % (2.0 * math.pi) - math.pi)

    reachable = eta_from_me <= float(config.lane_anchor_eta_cap)
    useful = (prod >= 3.0) | ((prod >= 2.0) & (obs.ships <= 34.0)) | (obs.ships >= 28.0)
    not_too_late = eta_from_me <= (eta_from_enemy + 5.0)
    same_lane = angle_delta <= float(config.lane_anchor_angle_width)
    target_mask = obs.is_neutral & obs.alive & reachable & useful & not_too_late & same_lane

    all_targets = torch.nonzero(target_mask, as_tuple=False).flatten()
    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)
    extra_mask = target_mask & ~existing
    if not bool(extra_mask.any()):
        return target_idx, target_exists, all_targets if all_targets.numel() else None

    target_score = (
        prod.to(dtype) * 7.0
        + obs.ships.to(dtype) * 0.035
        + (float(config.lane_anchor_eta_cap) - eta_from_me).clamp(min=0.0) * 0.18
        + (eta_from_enemy - eta_from_me).clamp(min=-2.0, max=8.0) * 0.09
        - angle_delta * 0.55
    )
    target_score = torch.where(extra_mask, target_score, torch.full_like(target_score, float("-inf")))
    extra_idx, extra_exists = _candidate_indices(
        target_score,
        torch.isfinite(target_score),
        max(1, min(int(config.lane_anchor_max_extra_targets), P)),
    )
    extras = extra_idx[extra_exists]
    if extras.numel() == 0:
        return target_idx, target_exists, all_targets if all_targets.numel() else None

    extra_exists_full = torch.ones(extras.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extras], dim=0), torch.cat([target_exists, extra_exists_full], dim=0), all_targets


def _apply_lane_anchor_bonus(
    *,
    score: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    prod: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    lane_targets: Tensor | None,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_lane_anchor_4p):
        return score
    if lane_targets is None or lane_targets.numel() == 0:
        return score
    P = int(prod.shape[0])
    target_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    target_mask[lane_targets.clamp(0, P - 1)] = True
    tgt = cand_tgt_slot.clamp(0, P - 1)
    selected = target_mask[tgt] & cand_active.any(dim=-1)
    bonus = (
        torch.full_like(score, float(config.lane_anchor_bonus))
        + (prod[tgt].to(score.dtype) * float(config.lane_anchor_prod_bonus)).clamp(
            max=float(config.lane_anchor_max_bonus) - float(config.lane_anchor_bonus)
        )
    )
    return score + torch.where(selected, bonus, torch.zeros_like(score))


def _append_domain_race_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    obs,
    obs_tensors: dict,
    cache,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Expose connector/outer targets that form a winning 4P domain."""
    if int(player_count) < 4 or not bool(config.enable_domain_race_4p):
        return target_idx, target_exists, None
    if _obs_step(obs_tensors) > int(config.domain_race_turn_limit):
        return target_idx, target_exists, None
    if not bool(source_mask.any()):
        return target_idx, target_exists, None

    P = int(obs.P)
    if P <= 0:
        return target_idx, target_exists, None
    device = obs.device
    dtype = obs.ships.dtype

    d0 = cache.cross_dist[0].to(dtype)
    speed = fleet_speed(obs.ships.to(dtype)).clamp(min=1e-6)
    eta_all = d0 / speed.view(P, 1)
    eta_from_me = torch.where(
        source_mask.view(P, 1),
        eta_all,
        torch.full_like(eta_all, float("inf")),
    ).amin(dim=0)

    enemy_source = obs.is_enemy & obs.alive & (obs.ships >= 4.0)
    if bool(enemy_source.any()):
        eta_from_enemy = torch.where(
            enemy_source.view(P, 1),
            eta_all,
            torch.full_like(eta_all, float("inf")),
        ).amin(dim=0)
    else:
        eta_from_enemy = torch.full((P,), float("inf"), dtype=dtype, device=device)

    planets = obs_tensors["planets"].to(dtype)
    x = planets[:, 2]
    y = planets[:, 3]
    dx = x - 50.0
    dy = y - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)
    angle = torch.atan2(dy, dx)
    owned_idx = torch.nonzero(source_mask, as_tuple=False).flatten()
    base = int(owned_idx[torch.argmax(obs.ships[owned_idx])].item())
    base_angle = angle[base]
    angle_delta = torch.abs((angle - base_angle + math.pi) % (2.0 * math.pi) - math.pi)

    reachable = eta_from_me <= float(config.domain_race_eta_cap)
    same_lane = angle_delta <= float(config.domain_race_angle_width)
    contest_ok = eta_from_me <= (eta_from_enemy + 6.0)
    connector = (radius >= 22.0) & (radius <= 43.0) & ((prod >= 2.0) | (obs.ships >= 20.0))
    outer_anchor = (radius >= 35.0) & ((prod >= 3.0) | (obs.ships >= 58.0))
    target_mask = (
        obs.is_neutral & obs.alive & reachable & same_lane & contest_ok
        & (connector | outer_anchor)
    )
    domain_targets = torch.nonzero(target_mask, as_tuple=False).flatten()

    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)
    extra_mask = target_mask & ~existing
    if not bool(extra_mask.any()):
        return target_idx, target_exists, domain_targets if domain_targets.numel() else None

    target_score = (
        prod.to(dtype) * 8.5
        + obs.ships.to(dtype) * 0.045
        + radius.clamp(max=65.0) * 0.055
        + (float(config.domain_race_eta_cap) - eta_from_me).clamp(min=0.0) * 0.16
        + (eta_from_enemy - eta_from_me).clamp(min=-3.0, max=8.0) * 0.07
        - angle_delta * 0.70
    )
    target_score = torch.where(extra_mask, target_score, torch.full_like(target_score, float("-inf")))
    extra_idx, extra_exists = _candidate_indices(
        target_score,
        torch.isfinite(target_score),
        max(1, min(int(config.domain_race_max_extra_targets), P)),
    )
    extras = extra_idx[extra_exists]
    if extras.numel() == 0:
        return target_idx, target_exists, domain_targets if domain_targets.numel() else None

    extra_exists_full = torch.ones(extras.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extras], dim=0), torch.cat([target_exists, extra_exists_full], dim=0), domain_targets


def _apply_domain_race_adjustment(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    domain_targets: Tensor | None,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_domain_race_4p):
        return score
    if _obs_step(obs_tensors) > int(config.domain_race_turn_limit):
        return score
    P = int(prod.shape[0])
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    active = cand_active.any(dim=-1)

    domain_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    if domain_targets is not None and domain_targets.numel() > 0:
        domain_mask[domain_targets.clamp(0, P - 1)] = True

    planets = obs_tensors["planets"].to(score.dtype)
    dx = planets[:, 2] - 50.0
    dy = planets[:, 3] - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)

    selected_domain = domain_mask[tgt] & active
    domain_bonus = (
        torch.full_like(score, float(config.domain_race_bonus))
        + (prod[tgt].to(score.dtype) * float(config.domain_race_prod_bonus)).clamp(max=0.38)
        + torch.where(radius[tgt] >= 36.0, torch.full_like(score, float(config.domain_race_outer_bonus)), torch.zeros_like(score))
    )

    cheap_small = (
        active
        & obs.is_neutral[tgt]
        & (prod[tgt] <= 1.0)
        & (obs.ships[tgt].to(score.dtype) <= 28.0)
        & (~domain_mask[tgt])
    )
    penalty = torch.where(cheap_small, torch.full_like(score, float(config.domain_race_cheap_penalty)), torch.zeros_like(score))

    src_radius = radius[src]
    src_prod = prod[src].to(score.dtype)
    src_ships = obs.ships[src].to(score.dtype)
    src_anchor = obs.owned[src] & (src_radius >= 32.0) & (src_prod >= 2.0)
    hold = float(config.domain_anchor_hold_base) + src_prod * float(config.domain_anchor_hold_prod)
    after = src_ships - send
    hold_short = (hold - after).clamp(min=0.0)
    drain_penalty = (
        (hold_short / 24.0).clamp(max=1.5)
        * float(config.domain_anchor_drain_penalty)
    )
    drain_penalty = torch.where(src_anchor & active, drain_penalty, torch.zeros_like(score))

    return score + torch.where(selected_domain, domain_bonus, torch.zeros_like(score)) - penalty - drain_penalty


def _append_enemy_domain_block_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    obs,
    obs_tensors: dict,
    cache,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Add neutral border targets that interrupt a rival's safe outer cluster."""
    if int(player_count) < 4 or not bool(config.enable_enemy_domain_block_4p):
        return target_idx, target_exists, None
    step = _obs_step(obs_tensors)
    if step < int(config.enemy_block_start_turn) or step > int(config.enemy_block_turn_limit):
        return target_idx, target_exists, None
    if not bool(source_mask.any()):
        return target_idx, target_exists, None

    P = int(obs.P)
    if P <= 0:
        return target_idx, target_exists, None
    device = obs.device
    dtype = obs.ships.dtype
    planets = obs_tensors["planets"].to(dtype)
    x = planets[:, 2]
    y = planets[:, 3]
    dx = x - 50.0
    dy = y - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)

    d0 = cache.cross_dist[0].to(dtype)
    speed = fleet_speed(obs.ships.to(dtype)).clamp(min=1e-6)
    eta_all = d0 / speed.view(P, 1)
    eta_from_me = torch.where(
        source_mask.view(P, 1),
        eta_all,
        torch.full_like(eta_all, float("inf")),
    ).amin(dim=0)

    owners = obs.owner_abs.to(torch.long)
    enemy_owned = obs.is_enemy & obs.alive & (owners >= 0)
    enemy_source = enemy_owned & (obs.ships >= 4.0)
    if not bool(enemy_owned.any()) or not bool(enemy_source.any()):
        return target_idx, target_exists, None

    # Score each enemy's already-safe outer foothold. This detects the blue/orange
    # pattern from replays: a rival owns a quiet outer domain and is about to snowball.
    threat_by_owner = torch.zeros(max(int(player_count), 1), dtype=dtype, device=device)
    for owner in range(int(player_count)):
        if owner == int(obs.player_id):
            continue
        owned = enemy_owned & (owners == owner)
        outer = owned & (radius >= 30.0)
        if not bool(outer.any()):
            continue
        threat = (
            prod[outer].to(dtype).sum() * 4.5
            + obs.ships[outer].to(dtype).sum() * 0.055
            + torch.count_nonzero(outer).to(dtype) * 2.0
        )
        threat_by_owner[owner] = threat
    leader = int(torch.argmax(threat_by_owner).item())
    if float(threat_by_owner[leader].item()) < 18.0:
        return target_idx, target_exists, None

    leader_sources = enemy_source & (owners == leader)
    eta_from_leader = torch.where(
        leader_sources.view(P, 1),
        eta_all,
        torch.full_like(eta_all, float("inf")),
    ).amin(dim=0)

    leader_outer = enemy_owned & (owners == leader) & (radius >= 30.0)
    near_leader_outer = torch.zeros(P, dtype=torch.bool, device=device)
    leader_idx = torch.nonzero(leader_outer, as_tuple=False).flatten()
    for src in leader_idx.tolist():
        near_leader_outer |= d0[src] <= float(config.enemy_block_cluster_radius)

    useful = (prod >= 2.0) | (obs.ships >= 24.0)
    reachable = eta_from_me <= float(config.enemy_block_eta_cap)
    leader_can_contest = eta_from_leader <= (eta_from_me + 8.0)
    target_mask = (
        obs.is_neutral
        & obs.alive
        & (radius >= 26.0)
        & near_leader_outer
        & useful
        & reachable
        & leader_can_contest
    )
    block_targets = torch.nonzero(target_mask, as_tuple=False).flatten()

    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)
    extra_mask = target_mask & ~existing
    if not bool(extra_mask.any()):
        return target_idx, target_exists, block_targets if block_targets.numel() else None

    target_score = (
        prod.to(dtype) * 9.0
        + obs.ships.to(dtype) * 0.05
        + (float(config.enemy_block_eta_cap) - eta_from_me).clamp(min=0.0) * 0.14
        + (eta_from_leader - eta_from_me).clamp(min=-4.0, max=8.0) * 0.08
        + radius.clamp(max=65.0) * 0.04
    )
    target_score = torch.where(extra_mask, target_score, torch.full_like(target_score, float("-inf")))
    extra_idx, extra_exists = _candidate_indices(
        target_score,
        torch.isfinite(target_score),
        max(1, min(int(config.enemy_block_max_extra_targets), P)),
    )
    extras = extra_idx[extra_exists]
    if extras.numel() == 0:
        return target_idx, target_exists, block_targets if block_targets.numel() else None
    extra_exists_full = torch.ones(extras.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extras], dim=0), torch.cat([target_exists, extra_exists_full], dim=0), block_targets


def _apply_enemy_domain_block_adjustment(
    *,
    score: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    block_targets: Tensor | None,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_enemy_domain_block_4p):
        return score
    step = _obs_step(obs_tensors)
    if step < int(config.enemy_block_start_turn) or step > int(config.enemy_block_turn_limit):
        return score
    P = int(prod.shape[0])
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    active = cand_active.any(dim=-1)

    block_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    if block_targets is not None and block_targets.numel() > 0:
        block_mask[block_targets.clamp(0, P - 1)] = True

    planets = obs_tensors["planets"].to(score.dtype)
    radius = torch.sqrt((planets[:, 2] - 50.0) ** 2 + (planets[:, 3] - 50.0) ** 2)
    selected = block_mask[tgt] & active
    bonus = (
        torch.full_like(score, float(config.enemy_block_bonus))
        + (prod[tgt].to(score.dtype) * float(config.enemy_block_prod_bonus)).clamp(max=0.42)
        + torch.where(radius[tgt] >= 32.0, torch.full_like(score, float(config.enemy_block_outer_bonus)), torch.zeros_like(score))
    )
    direct_enemy = active & obs.is_enemy[tgt] & (~block_mask[tgt])
    penalty = torch.where(
        direct_enemy,
        torch.full_like(score, float(config.enemy_block_direct_enemy_penalty)),
        torch.zeros_like(score),
    )
    return score + torch.where(selected, bonus, torch.zeros_like(score)) - penalty


def _make_winner_path_plan(
    *,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
) -> dict | None:
    """Choose one early 4P lane: connector -> thick outer anchors."""
    if not bool(source_mask.any()):
        return None
    P = int(obs.P)
    if P <= 0:
        return None

    dtype = obs.ships.dtype
    planets = obs_tensors["planets"].to(dtype)
    x = planets[:, 2]
    y = planets[:, 3]
    dx = x - 50.0
    dy = y - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)
    angle = torch.atan2(dy, dx)

    owned_idx = torch.nonzero(source_mask, as_tuple=False).flatten()
    base = int(owned_idx[torch.argmax(obs.ships[owned_idx])].item())
    bx = float(x[base].item())
    by = float(y[base].item())
    base_angle = float(angle[base].item())

    neutral_idx = torch.nonzero(obs.is_neutral & obs.alive, as_tuple=False).flatten().tolist()
    if not neutral_idx:
        return None

    best: tuple[float, int, list[int], list[int], float] | None = None
    for anchor in neutral_idx:
        ar = float(radius[anchor].item())
        ap = float(prod[anchor].item())
        ships = float(obs.ships[anchor].item())
        if ar < 32.0:
            continue
        if ap < 3.0 and ships < 45.0:
            continue
        ax = float(x[anchor].item())
        ay = float(y[anchor].item())
        dist_from_base = _dist_xy(bx, by, ax, ay)
        if dist_from_base > 78.0:
            continue
        lane_angle = float(angle[anchor].item())
        if _angle_diff(base_angle, lane_angle) > 1.35:
            continue

        connectors: list[int] = []
        supports: list[int] = []
        for j in neutral_idx:
            if j == anchor:
                continue
            jr = float(radius[j].item())
            jp = float(prod[j].item())
            js = float(obs.ships[j].item())
            ja = float(angle[j].item())
            if _angle_diff(lane_angle, ja) > float(config.winner_path_angle_width):
                continue
            d_anchor = _dist_xy(float(x[j].item()), float(y[j].item()), ax, ay)
            d_base = _dist_xy(float(x[j].item()), float(y[j].item()), bx, by)
            if 16.0 <= d_base <= dist_from_base + 6.0 and 18.0 <= jr <= ar + 8.0:
                if jp >= 2.0 or 16.0 <= js <= 36.0:
                    connectors.append(j)
            if d_anchor <= 48.0 and (jp >= 2.0 or js >= 18.0):
                supports.append(j)

        support_prod = sum(float(prod[j].item()) for j in supports)
        connector_value = sum(float(prod[j].item()) * 1.8 + max(0.0, 32.0 - float(obs.ships[j].item())) * 0.035 for j in connectors)
        score = (
            ap * 10.0
            + ships * 0.07
            + ar * 0.08
            + support_prod * 1.75
            + connector_value
            + len(supports) * 1.15
            - dist_from_base * 0.18
        )
        if best is None or score > best[0]:
            best = (score, anchor, connectors, supports, lane_angle)

    if best is None or best[0] < 19.0:
        return None
    _, anchor, connectors, supports, lane_angle = best
    path_targets = [anchor]
    path_targets.extend(sorted(connectors, key=lambda j: _dist_xy(bx, by, float(x[j].item()), float(y[j].item())))[:4])
    path_targets.extend(sorted(supports, key=lambda j: float(prod[j].item()) * 10.0 + float(obs.ships[j].item()), reverse=True)[:5])
    seen: set[int] = set()
    unique_targets = []
    for t in path_targets:
        if t not in seen:
            unique_targets.append(int(t))
            seen.add(int(t))
    return {
        "anchor": int(anchor),
        "targets": unique_targets,
        "angle": float(lane_angle),
        "created_step": _obs_step(obs_tensors),
        "score": float(best[0]),
    }


def _append_winner_path_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    memory,
) -> tuple[Tensor, Tensor, dict | None]:
    if int(player_count) < 4 or not bool(config.enable_winner_path_4p):
        return target_idx, target_exists, None
    if _obs_step(obs_tensors) > int(config.winner_path_turn_limit):
        return target_idx, target_exists, getattr(memory, "winner_path", None)
    if getattr(memory, "winner_path", None) is None or _obs_step(obs_tensors) <= 1:
        memory.winner_path = _make_winner_path_plan(
            obs=obs, obs_tensors=obs_tensors, prod=prod, source_mask=source_mask, config=config,
        )
    plan = getattr(memory, "winner_path", None)
    if not plan:
        return target_idx, target_exists, None

    P = int(obs.P)
    device = obs.device
    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)

    extras = []
    for t in plan.get("targets", [])[: int(config.winner_path_max_extra_targets)]:
        if 0 <= int(t) < P and bool(obs.alive[int(t)].item()) and not bool(existing[int(t)].item()):
            extras.append(int(t))
    if not extras:
        return target_idx, target_exists, plan
    extra_idx = torch.tensor(extras, dtype=target_idx.dtype, device=target_idx.device)
    extra_exists = torch.ones(extra_idx.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extra_idx], dim=0), torch.cat([target_exists, extra_exists], dim=0), plan


def _apply_winner_path_adjustment(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    winner_path: dict | None,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_winner_path_4p) or not winner_path:
        return score
    if _obs_step(obs_tensors) > int(config.winner_path_turn_limit):
        return score
    P = int(prod.shape[0])
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    active = cand_active.any(dim=-1)

    targets = set(int(t) for t in winner_path.get("targets", []))
    target_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    for t in targets:
        if 0 <= t < P:
            target_mask[t] = True

    planets = obs_tensors["planets"].to(score.dtype)
    dx = planets[:, 2] - 50.0
    dy = planets[:, 3] - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)
    angle = torch.atan2(dy, dx)
    lane_angle = float(winner_path.get("angle", 0.0))
    angle_delta = torch.abs((angle - lane_angle + math.pi) % (2.0 * math.pi) - math.pi)

    selected = target_mask[tgt] & active
    bonus = (
        torch.full_like(score, float(config.winner_path_bonus))
        + (prod[tgt].to(score.dtype) * float(config.winner_path_prod_bonus)).clamp(max=0.44)
        + torch.where(radius[tgt] >= 34.0, torch.full_like(score, float(config.winner_path_outer_bonus)), torch.zeros_like(score))
    )

    offlane_cheap = (
        active
        & obs.is_neutral[tgt]
        & (~target_mask[tgt])
        & (angle_delta[tgt] > float(config.winner_path_angle_width))
        & (prod[tgt] <= 2.0)
        & (obs.ships[tgt].to(score.dtype) <= 30.0)
    )
    offlane_penalty = torch.where(
        offlane_cheap,
        torch.full_like(score, float(config.winner_path_offlane_cheap_penalty)),
        torch.zeros_like(score),
    )

    anchor = int(winner_path.get("anchor", -1))
    src_anchor = (src == anchor) & obs.owned[src]
    src_prod = prod[src].to(score.dtype)
    src_ships = obs.ships[src].to(score.dtype)
    hold = float(config.winner_anchor_hold_base) + src_prod * float(config.winner_anchor_hold_prod)
    after = src_ships - send
    hold_short = (hold - after).clamp(min=0.0)
    drain_penalty = (
        (hold_short / 28.0).clamp(max=1.6)
        * float(config.winner_anchor_drain_penalty)
    )
    drain_penalty = torch.where(src_anchor & active, drain_penalty, torch.zeros_like(score))
    return score + torch.where(selected, bonus, torch.zeros_like(score)) - offlane_penalty - drain_penalty


def _leader_owner_by_power(obs, prod: Tensor, *, player_count: int) -> int:
    owners = obs.owner_abs.to(torch.long)
    owner_count = max(int(player_count), int(owners[owners >= 0].max().item()) + 1 if bool((owners >= 0).any()) else 1)
    power = torch.full((owner_count,), float("-inf"), dtype=prod.dtype, device=prod.device)
    for owner in range(owner_count):
        if owner == int(obs.player_id):
            continue
        mask = obs.alive & (owners == int(owner))
        if bool(mask.any()):
            power[owner] = obs.ships[mask].to(prod.dtype).sum() + prod[mask].to(prod.dtype).sum() * 13.0
    return int(torch.argmax(power).item()) if bool(torch.isfinite(power).any()) else -1


def _top_director_neighborhood(*, obs, prod: Tensor, cache, tgt: Tensor, eta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    P = int(obs.P)
    dtype = prod.dtype
    device = prod.device
    d0 = cache.cross_dist[0].to(dtype)
    owned = obs.owned & obs.alive
    enemy = obs.is_enemy & obs.alive
    support_window = eta.to(dtype).clamp(min=1.0) + 6.0
    my_eta = torch.where(
        owned.view(P, 1),
        d0[:, tgt],
        torch.full((P, tgt.shape[0]), float("inf"), dtype=dtype, device=device),
    )
    my_w = (1.0 - my_eta / support_window.view(1, -1)).clamp(min=0.0)
    my_support = (
        obs.ships.to(dtype).view(P, 1) * my_w
        + prod.to(dtype).view(P, 1) * my_w * 2.2
    ).sum(dim=0)
    if bool(enemy.any()):
        enemy_speed = fleet_speed(obs.ships.to(dtype).clamp(min=1e-6))
        enemy_eta = torch.where(
            enemy.view(P, 1),
            d0[:, tgt] / enemy_speed.view(P, 1).clamp(min=1e-6),
            torch.full((P, tgt.shape[0]), float("inf"), dtype=dtype, device=device),
        )
        enemy_w = (1.0 - enemy_eta / support_window.view(1, -1)).clamp(min=0.0)
        enemy_pressure = (
            obs.ships.to(dtype).view(P, 1) * enemy_w
            + prod.to(dtype).view(P, 1) * enemy_w * 2.8
        ).sum(dim=0)
        enemy_reach_count = ((enemy_eta <= support_window.view(1, -1)) & enemy.view(P, 1)).sum(dim=0)
    else:
        enemy_pressure = torch.zeros_like(my_support)
        enemy_reach_count = torch.zeros(tgt.shape[0], dtype=torch.long, device=device)
    return my_support, enemy_pressure, enemy_reach_count


def _apply_top_director_adjustment(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_top_director_4p):
        return score
    step = _obs_step(obs_tensors)
    if step > int(config.top_director_turn_limit):
        return score
    P = int(obs.P)
    if P <= 0:
        return score

    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    eta = cand_eta[:, 0].to(score.dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1)
    radius = torch.sqrt((obs.x.to(score.dtype) - 50.0) ** 2 + (obs.y.to(score.dtype) - 50.0) ** 2)
    my_support, enemy_pressure, enemy_reach_count = _top_director_neighborhood(
        obs=obs, prod=prod, cache=cache, tgt=tgt, eta=eta,
    )
    central_brawl = (
        active
        & (radius[tgt] <= float(config.top_director_central_radius))
        & (enemy_reach_count >= 2)
        & (my_support < enemy_pressure * 1.08)
    )
    neutral = active & obs.is_neutral[tgt]
    cheap_or_good = (obs.ships[tgt].to(score.dtype) <= 30.0) | (prod[tgt].to(score.dtype) >= 3.0)
    safe_neutral = neutral & cheap_or_good & (~central_brawl) & (my_support + send >= enemy_pressure * 0.80)

    leader = _leader_owner_by_power(obs, prod, player_count=player_count)
    owners = obs.owner_abs.to(torch.long)
    nonleader_enemy = (
        active
        & obs.is_enemy[tgt]
        & (owners[tgt] >= 0)
        & (leader >= 0)
        & (owners[tgt] != int(leader))
        & (step < 130)
        & (my_support < enemy_pressure * 1.25 + 8.0)
    )
    src_after = obs.ships[src].to(score.dtype) - send
    source_hold = 16.0 + prod[src].to(score.dtype) * 4.2
    drain_risk = active & obs.owned[src] & (src_after < source_hold) & (step < 120)

    bonus = torch.where(
        safe_neutral,
        torch.full_like(score, float(config.top_director_safe_neutral_bonus))
        + (prod[tgt].to(score.dtype) * 0.055).clamp(max=0.26)
        + torch.where(radius[tgt] >= 31.0, torch.full_like(score, float(config.top_director_outer_bonus)), torch.zeros_like(score)),
        torch.zeros_like(score),
    )
    penalty = torch.zeros_like(score)
    penalty = penalty + torch.where(central_brawl, torch.full_like(score, float(config.top_director_central_brawl_penalty)), torch.zeros_like(score))
    penalty = penalty + torch.where(nonleader_enemy, torch.full_like(score, float(config.top_director_nonleader_enemy_penalty)), torch.zeros_like(score))
    penalty = penalty + torch.where(
        drain_risk,
        ((source_hold - src_after).clamp(min=0.0) / 30.0).clamp(max=1.5) * float(config.top_director_drain_penalty),
        torch.zeros_like(score),
    )
    return score + bonus - penalty


def _top_director_reserved_safe_mask(
    *,
    score: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
) -> tuple[Tensor, Tensor]:
    if int(player_count) < 4 or not bool(config.enable_top_director_4p):
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))
    step = _obs_step(obs_tensors)
    if step < int(config.top_director_reserve_start) or step > int(config.top_director_reserve_limit):
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))
    P = int(obs.P)
    if P <= 0:
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    eta = cand_eta[:, 0].to(score.dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1)
    my_support, enemy_pressure, enemy_reach_count = _top_director_neighborhood(
        obs=obs, prod=prod, cache=cache, tgt=tgt, eta=eta,
    )
    radius = torch.sqrt((obs.x.to(score.dtype) - 50.0) ** 2 + (obs.y.to(score.dtype) - 50.0) ** 2)
    central_brawl = (radius[tgt] <= float(config.top_director_central_radius)) & (enemy_reach_count >= 2) & (my_support < enemy_pressure)
    cheap_or_good = (obs.ships[tgt].to(score.dtype) <= 28.0) | (prod[tgt].to(score.dtype) >= 3.0)
    safe = (
        active
        & obs.is_neutral[tgt]
        & cheap_or_good
        & (~central_brawl)
        & (my_support + send + prod[tgt].to(score.dtype) * 3.0 >= enemy_pressure * 0.82)
    )
    reserve_score = (
        prod[tgt].to(score.dtype) * 38.0
        - send * 2.1
        - eta * 1.6
        + my_support * 0.12
        - enemy_pressure * 0.22
        + torch.where(radius[tgt] >= 31.0, torch.full_like(score, 5.0), torch.zeros_like(score))
    ) * 0.035
    return safe, torch.where(safe, reserve_score, torch.full_like(score, float("-inf")))


def _apply_orbit_response_gate(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    cache,
    garrison_status,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_orbit_response_gate_4p):
        return score
    step = _obs_step(obs_tensors)
    if step < int(config.orbit_response_start):
        return score
    if step > int(config.orbit_response_turn_limit):
        return score
    if getattr(garrison_status, "arrivals_by_owner", None) is None:
        return score
    P = int(obs.P)
    H = int(garrison_status.owner.shape[-1]) - 1
    if P <= 0 or H <= 0:
        return score

    dtype = score.dtype
    device = score.device
    pid = int(obs.player_id)
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1) & torch.isfinite(score)

    k_arr = torch.ceil(eta).to(torch.long).clamp(min=1, max=H)
    rows = torch.arange(score.shape[0], device=device)
    owner_at = garrison_status.owner[tgt, k_arr].to(torch.long)
    ships_at = garrison_status.ships[tgt, k_arr].to(dtype)
    prod_t = prod[tgt].to(dtype)

    enemy_or_neutral = owner_at != pid
    arrival_margin = torch.where(enemy_or_neutral, send - ships_at - 1.0, send + ships_at)

    end_k = (k_arr + int(config.orbit_response_window)).clamp(max=H)
    steps_h = torch.arange(1, H + 1, device=device).view(1, H)
    window = (steps_h > k_arr.view(-1, 1)) & (steps_h <= end_k.view(-1, 1))
    arr = garrison_status.arrivals_by_owner[tgt, 1:, :].to(dtype)
    owner_ids = torch.arange(int(player_count), device=device).view(1, 1, -1)
    enemy_arr = torch.where(owner_ids != pid, arr, torch.zeros_like(arr)).sum(dim=2)
    my_arr = arr[:, :, pid]
    enemy_window = torch.where(window, enemy_arr, torch.zeros_like(enemy_arr)).sum(dim=1)
    my_window = torch.where(window, my_arr, torch.zeros_like(my_arr)).sum(dim=1)

    d0 = cache.cross_dist[0].to(dtype)
    enemy_mask = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)
    enemy_speed = fleet_speed(obs.ships.to(dtype).clamp(min=1.0))
    enemy_eta = d0[:, tgt].transpose(0, 1) / enemy_speed.view(1, P).clamp(min=1e-6)
    response_horizon = eta + float(config.orbit_response_window)
    local_enemy = torch.where(
        (enemy_eta <= response_horizon.view(-1, 1)) & enemy_mask.view(1, P),
        obs.ships.to(dtype).view(1, P) * float(config.orbit_response_enemy_frac),
        torch.zeros_like(enemy_eta),
    ).sum(dim=1)

    future_hold = arrival_margin + my_window + prod_t * float(config.orbit_response_window) - enemy_window - local_enemy
    response_bad = active & (future_hold < -float(config.orbit_response_margin))

    target_owner_now = obs.owner_abs.to(torch.long)[tgt]
    leader = _leader_owner_by_power(obs, prod, player_count=player_count)
    enemy_target = active & obs.is_enemy[tgt]
    neutral_target = active & obs.is_neutral[tgt]
    own_target = active & obs.owned[tgt]
    nonleader_enemy = (
        enemy_target
        & (leader >= 0)
        & (target_owner_now >= 0)
        & (target_owner_now != int(leader))
        & (step < 135)
    )

    safe_future = active & (neutral_target | enemy_target) & (~response_bad) & (future_hold > float(config.orbit_response_margin) * 1.25)
    defense_help = own_target & ((owner_at != pid) | (future_hold < float(config.orbit_response_margin)))
    src_after = obs.ships[src].to(dtype) - send
    source_stripped = src_after < (10.0 + prod[src].to(dtype) * 3.4)

    bonus = torch.zeros_like(score)
    bonus = bonus + torch.where(
        safe_future & neutral_target,
        torch.full_like(score, float(config.orbit_response_safe_bonus)),
        torch.zeros_like(score),
    )
    bonus = bonus + torch.where(
        defense_help,
        torch.full_like(score, float(config.orbit_response_defense_bonus)),
        torch.zeros_like(score),
    )

    penalty = torch.zeros_like(score)
    penalty = penalty + torch.where(
        response_bad & enemy_target,
        torch.full_like(score, float(config.orbit_response_recapture_penalty)),
        torch.zeros_like(score),
    )
    penalty = penalty + torch.where(
        response_bad & nonleader_enemy,
        torch.full_like(score, float(config.orbit_response_bad_war_penalty)),
        torch.zeros_like(score),
    )
    penalty = penalty + torch.where(
        response_bad & source_stripped & enemy_target,
        torch.full_like(score, float(config.orbit_response_recapture_penalty) * 0.75),
        torch.zeros_like(score),
    )
    ramp_end = max(int(config.orbit_response_ramp_end), int(config.orbit_response_start) + 1)
    if step >= ramp_end:
        response_strength = 1.0
    else:
        span = float(max(1, ramp_end - int(config.orbit_response_start)))
        progress = float(max(0, step - int(config.orbit_response_start))) / span
        response_strength = float(config.orbit_response_ramp_min) + (
            1.0 - float(config.orbit_response_ramp_min)
        ) * progress
    response_strength = max(0.0, min(1.0, response_strength))
    return score + (bonus - penalty) * float(response_strength)


def _control_map(obs, cache, prod: Tensor, config: ProducerLiteConfig) -> Tensor:
    P = int(obs.P)
    if P <= 0:
        return obs.ships.new_zeros((0,))
    dtype = obs.ships.dtype
    d0 = cache.cross_dist[0].to(dtype)
    radius = max(float(config.influence_radius), 1.0)
    decay = torch.exp(-(d0 / radius).clamp(min=0.0, max=8.0))
    mass = (obs.ships.to(dtype) + prod.to(dtype) * 13.0).clamp(min=0.0)
    my_mass = torch.where(obs.owned & obs.alive, mass, torch.zeros_like(mass))
    enemy_mass = torch.where(obs.is_enemy & obs.alive, mass, torch.zeros_like(mass))
    my_inf = (my_mass.view(P, 1) * decay).sum(dim=0)
    enemy_inf = (enemy_mass.view(P, 1) * decay).sum(dim=0)
    return (my_inf - enemy_inf) / (my_inf + enemy_inf + 1.0)


def _fleet_intent_pressure(obs, config: ProducerLiteConfig) -> Tensor:
    P = int(obs.P)
    if P <= 0 or int(getattr(obs, "F", 0)) <= 0:
        return obs.ships.new_zeros((P,))
    dtype = obs.ships.dtype
    device = obs.device
    enemy_fleet = (
        obs.f_alive
        & (obs.f_owner >= 0.0)
        & (obs.f_owner != float(obs.player_id))
        & (obs.f_ships > 0.0)
    )
    if not bool(enemy_fleet.any()):
        return obs.ships.new_zeros((P,))

    fx = obs.f_x[enemy_fleet].to(dtype)
    fy = obs.f_y[enemy_fleet].to(dtype)
    fa = obs.f_angle[enemy_fleet].to(dtype)
    fs = obs.f_ships[enemy_fleet].to(dtype).clamp(min=1.0)
    dx = obs.x.to(dtype).view(1, P) - fx.view(-1, 1)
    dy = obs.y.to(dtype).view(1, P) - fy.view(-1, 1)
    dist = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
    target_angle = torch.atan2(dy, dx)
    diff = torch.atan2(torch.sin(target_angle - fa.view(-1, 1)), torch.cos(target_angle - fa.view(-1, 1))).abs()
    speed = fleet_speed(fs).to(dtype).view(-1, 1).clamp(min=1e-6)
    eta = dist / speed
    horizon = float(config.intent_horizon)
    forward = (diff <= float(config.intent_angle_width)) & (eta <= horizon)
    angle_weight = (1.0 - diff / max(float(config.intent_angle_width), 1e-3)).clamp(min=0.0)
    eta_weight = (1.0 - eta / max(horizon, 1.0)).clamp(min=0.0)
    contrib = torch.where(
        forward,
        fs.view(-1, 1) * angle_weight * eta_weight,
        torch.zeros((fs.shape[0], P), dtype=dtype, device=device),
    )
    return contrib.sum(dim=0)


def _apply_influence_intent_adjustment(
    *,
    score: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    prod: Tensor,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4:
        return score
    if not bool(config.enable_influence_map_4p) and not bool(config.enable_fleet_intent_4p):
        return score
    P = int(obs.P)
    if P <= 0:
        return score

    dtype = score.dtype
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    active = cand_active.any(dim=-1) & torch.isfinite(score)
    neutral_target = active & obs.is_neutral[tgt]
    enemy_target = active & obs.is_enemy[tgt]
    own_target = active & obs.owned[tgt]

    adjust = torch.zeros_like(score)
    if bool(config.enable_influence_map_4p):
        control = _control_map(obs, cache, prod, config).to(dtype)
        c = control[tgt]
        safe = c.clamp(min=0.0)
        danger = (-c).clamp(min=0.0)
        adjust = adjust + torch.where(
            neutral_target & (c > 0.12),
            safe.clamp(max=1.0) * float(config.influence_safe_bonus),
            torch.zeros_like(score),
        )
        adjust = adjust - torch.where(
            neutral_target & (c < -0.18),
            danger.clamp(max=1.0) * float(config.influence_danger_penalty),
            torch.zeros_like(score),
        )
        adjust = adjust - torch.where(
            enemy_target & (c < -0.35),
            danger.clamp(max=1.0) * float(config.influence_enemy_deep_penalty),
            torch.zeros_like(score),
        )
        adjust = adjust + torch.where(
            own_target & (c < -0.10),
            danger.clamp(max=1.0) * (float(config.influence_safe_bonus) * 0.75),
            torch.zeros_like(score),
        )

    if bool(config.enable_fleet_intent_4p):
        pressure = _fleet_intent_pressure(obs, config).to(dtype)
        if pressure.numel() > 0:
            p = (pressure[tgt] * float(config.intent_pressure_scale)).clamp(max=1.0)
            adjust = adjust - torch.where(
                (neutral_target | enemy_target) & (p > 0.0),
                p * float(config.intent_target_penalty),
                torch.zeros_like(score),
            )
            adjust = adjust + torch.where(
                own_target & (p > 0.0),
                p * float(config.intent_defense_bonus),
                torch.zeros_like(score),
            )

    return score + adjust


def _apply_expansion_drive_4p(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    prod: Tensor,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_expansion_drive_4p):
        return score
    step = int(float(obs.step.reshape(-1)[0].item()))
    if step < int(config.expansion_drive_start) or step > int(config.expansion_drive_limit):
        return score
    P = int(obs.P)
    if P <= 0 or score.numel() == 0:
        return score

    device = score.device
    dtype = score.dtype
    pid = int(obs.player_id)
    active = cand_active.any(dim=-1) & torch.isfinite(score)
    if not bool(active.any()):
        return score

    owners = obs.owner_abs.to(torch.long)
    alive = obs.alive
    owned_mask = obs.owned & alive
    my_prod = prod[owned_mask].to(dtype).sum()
    my_planets = owned_mask.to(dtype).sum()
    opp_prod_vals = []
    for owner in range(int(player_count)):
        if owner == pid:
            continue
        mask = alive & (owners == int(owner))
        opp_prod_vals.append(prod[mask].to(dtype).sum())
    if opp_prod_vals:
        opp_prod_max = torch.stack(opp_prod_vals).max()
    else:
        opp_prod_max = torch.zeros((), dtype=dtype, device=device)

    prod_lag = ((opp_prod_max - my_prod) / (opp_prod_max.abs() + 1.0)).clamp(min=0.0, max=1.0)
    planet_lag = (
        (torch.tensor(float(config.expansion_drive_low_planets), dtype=dtype, device=device) - my_planets)
        / max(float(config.expansion_drive_low_planets), 1.0)
    ).clamp(min=0.0, max=1.0)
    need = torch.maximum(prod_lag, planet_lag)

    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    target_prod = prod[tgt].to(dtype)
    target_ships = obs.ships[tgt].to(dtype)
    neutral_target = active & obs.is_neutral[tgt] & (target_prod >= float(config.expansion_drive_min_prod))
    enemy_target = active & obs.is_enemy[tgt]
    attack_target = neutral_target | enemy_target

    dx = obs.x[tgt].to(dtype) - 50.0
    dy = obs.y[tgt].to(dtype) - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)
    outer = ((radius - 24.0) / 34.0).clamp(min=0.0, max=1.0)
    cheap = ((28.0 - target_ships) / 28.0).clamp(min=0.0, max=1.0)
    center = radius < 25.0

    enemy_pressure = cheap_enemy_pressure(
        obs, cache, horizon=max(10.0, float(config.horizon) + 3.0), player_id=pid
    ).to(dtype)
    pressure = enemy_pressure[tgt]
    d0 = cache.cross_dist[0].to(dtype)
    dist_from_target = d0[tgt]
    planet_idx = torch.arange(P, device=device).view(1, P)
    neighbor_neutral = (
        obs.is_neutral.view(1, P)
        & alive.view(1, P)
        & (planet_idx != tgt.view(-1, 1))
        & (dist_from_target <= float(config.expansion_campaign_radius))
        & (prod.view(1, P).to(dtype) >= 1.0)
    )
    neighbor_decay = (1.0 - dist_from_target / max(float(config.expansion_campaign_radius), 1.0)).clamp(min=0.0)
    neighbor_prod = torch.where(
        neighbor_neutral,
        prod.view(1, P).to(dtype) * neighbor_decay,
        torch.zeros((score.shape[0], P), dtype=dtype, device=device),
    ).sum(dim=1)
    neighbor_cheap = torch.where(
        neighbor_neutral & (obs.ships.view(1, P).to(dtype) <= 24.0),
        neighbor_decay,
        torch.zeros((score.shape[0], P), dtype=dtype, device=device),
    ).sum(dim=1)
    neighbor_risk = torch.where(
        neighbor_neutral,
        enemy_pressure.view(1, P).to(dtype) * neighbor_decay,
        torch.zeros((score.shape[0], P), dtype=dtype, device=device),
    ).sum(dim=1)

    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    primary_send = cand_send[:, 0].to(dtype)
    source_after = obs.ships[src].to(dtype) - primary_send
    source_reserve = 8.0 + prod[src].to(dtype) * 2.0
    stripped = active & obs.owned[src] & (source_after < source_reserve)

    neutral_bonus = (
        float(config.expansion_drive_base_bonus) * (0.35 + need)
        + target_prod * float(config.expansion_drive_prod_bonus)
        + outer * float(config.expansion_drive_outer_bonus)
        + cheap * float(config.expansion_drive_cheap_bonus)
        + need * float(config.expansion_drive_lag_bonus)
        + neighbor_prod * float(config.expansion_campaign_prod_bonus) * (0.35 + need)
        + neighbor_cheap * float(config.expansion_campaign_cheap_bonus)
        + target_prod * float(config.expansion_drive_compound_bonus) * (1.0 - need)
    )
    adjust = torch.zeros_like(score)
    adjust = adjust + torch.where(neutral_target, neutral_bonus, torch.zeros_like(score))
    adjust = adjust - torch.where(
        enemy_target,
        torch.full_like(score, float(config.expansion_drive_enemy_penalty)) * (1.0 - need * 0.35),
        torch.zeros_like(score),
    )
    adjust = adjust - torch.where(
        attack_target & center,
        torch.full_like(score, float(config.expansion_drive_center_penalty)),
        torch.zeros_like(score),
    )
    adjust = adjust - torch.where(
        attack_target,
        pressure.clamp(min=0.0) * float(config.expansion_drive_pressure_penalty),
        torch.zeros_like(score),
    )
    adjust = adjust - torch.where(
        neutral_target,
        neighbor_risk.clamp(min=0.0) * float(config.expansion_campaign_risk_penalty),
        torch.zeros_like(score),
    )
    adjust = adjust - torch.where(
        stripped & attack_target,
        torch.full_like(score, float(config.expansion_drive_strip_penalty)),
        torch.zeros_like(score),
    )
    return score + adjust


def _plan_coord_followups(
    *,
    movement: PlanetMovement,
    obs,
    prod: Tensor,
    alive_by_step: Tensor,
    cache,
    garrison_status,
    wave_entries: LaunchEntries,
    leftover: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> tuple[LaunchEntries, Tensor]:
    if int(player_count) < 4 or not bool(config.enable_coord_followup_4p):
        return _empty_entries(obs.device, obs.ships.dtype), leftover
    step = int(float(obs.step.reshape(-1)[0].item()))
    if step < int(config.coord_start_turn) or step > int(config.coord_turn_limit):
        return _empty_entries(obs.device, obs.ships.dtype), leftover
    if wave_entries.width <= 0 or not bool(wave_entries.valid.any()):
        return _empty_entries(obs.device, obs.ships.dtype), leftover

    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    owners = obs.owner_abs.to(torch.long)
    valid_idx = torch.where(wave_entries.valid)[0]
    if valid_idx.numel() == 0:
        return _empty_entries(device, dtype), leftover

    used_src = torch.zeros(P, dtype=torch.bool, device=device)
    used_src[wave_entries.source_slots[wave_entries.valid].clamp(0, max(P - 1, 0))] = True
    src_base_mask = obs.owned & obs.alive & (leftover >= float(config.coord_min_leftover))
    if not bool(src_base_mask.any()):
        return _empty_entries(device, dtype), leftover

    target_items: list[tuple[float, int, float, float]] = []
    for t_raw in torch.unique(wave_entries.target_slots[valid_idx].clamp(0, max(P - 1, 0))).tolist():
        t = int(t_raw)
        if t < 0 or t >= P or not bool(obs.alive[t].item()) or bool(obs.owned[t].item()):
            continue
        sent_mask = wave_entries.valid & (wave_entries.target_slots.clamp(0, max(P - 1, 0)) == t)
        if not bool(sent_mask.any()):
            continue
        sent = float(wave_entries.ships[sent_mask].sum().item())
        eta0 = float(wave_entries.eta[sent_mask].min().item())
        target_prod = float(prod[t].item())
        is_enemy = int(owners[t].item()) >= 0 and int(owners[t].item()) != int(obs.player_id)
        if target_prod < float(config.coord_one_ply_min_target_prod):
            continue
        if (not is_enemy) and target_prod < float(config.coord_min_target_prod):
            continue
        target_value = target_prod * 10.0 + (18.0 if is_enemy else 0.0) + float(obs.ships[t].item()) * 0.04
        target_items.append((target_value, t, sent, eta0))
    target_items.sort(reverse=True)
    if not target_items:
        return _empty_entries(device, dtype), leftover

    out_src: list[Tensor] = []
    out_tgt: list[Tensor] = []
    out_ships: list[Tensor] = []
    out_angle: list[Tensor] = []
    out_eta: list[Tensor] = []
    out_valid: list[Tensor] = []
    budget = leftover.clone()
    added = 0

    for _value, tgt, sent, eta0 in target_items:
        if added >= int(config.coord_max_extra):
            break
        src_mask = src_base_mask & (~used_src) & (torch.arange(P, device=device) != int(tgt))
        if not bool(src_mask.any()):
            break
        src_idx = torch.where(src_mask)[0]
        reserve = float(config.coord_source_reserve_base) + prod[src_idx].to(dtype) * float(config.coord_source_reserve_prod)
        room = (budget[src_idx].to(dtype) - reserve).clamp(min=0.0)
        send = torch.minimum(
            room,
            torch.full_like(room, min(float(config.coord_send_cap), max(5.0, sent * float(config.coord_send_frac)))),
        ).floor()
        viable_send = send >= 4.0
        if not bool(viable_send.any()):
            continue
        src_idx = src_idx[viable_send]
        send = send[viable_send]
        room = room[viable_send]
        tgt_vec = torch.full_like(src_idx, int(tgt))
        aim = intercept_angle(
            movement,
            source_slots=src_idx,
            target_slots=tgt_vec,
            fleet_sizes=send,
        )
        eta = aim["eta"].to(dtype)
        viable = aim["viable"] & torch.isfinite(eta) & (eta <= float(config.horizon))
        viable = viable & ((eta - float(eta0)).abs() <= float(config.coord_eta_delta))
        if not bool(viable.any()):
            continue
        prod_t = float(prod[tgt].item())
        score = (
            send.to(dtype) * 0.10
            + room.to(dtype) * 0.02
            - (eta - float(eta0)).abs() * 0.80
            - eta * 0.08
            + prod_t * 0.65
        )
        score = torch.where(viable, score, torch.full_like(score, float("-inf")))
        order = torch.argsort(score, descending=True)
        picked = None
        max_trials = min(int(config.coord_one_ply_top_sources), int(order.numel()))
        for rank_pos in range(max_trials):
            best = int(order[rank_pos].item())
            if not bool(torch.isfinite(score[best]).item()):
                continue
            if _coord_one_ply_pass(
                obs=obs,
                prod=prod,
                alive_by_step=alive_by_step,
                cache=cache,
                garrison_status=garrison_status,
                wave_entries=wave_entries,
                target_slot=int(tgt),
                coord_src=src_idx[best],
                coord_ships=send[best],
                coord_eta=eta[best],
                config=config,
                player_count=player_count,
            ):
                picked = best
                break
        if picked is None:
            continue
        best = int(picked)
        src = src_idx[best]
        ships = send[best]
        angle = aim["angle"][best].to(dtype)
        eta_best = eta[best]

        out_src.append(src.reshape(1).to(torch.long))
        out_tgt.append(torch.tensor([int(tgt)], dtype=torch.long, device=device))
        out_ships.append(ships.reshape(1).to(dtype))
        out_angle.append(angle.reshape(1).to(dtype))
        out_eta.append(eta_best.reshape(1).to(dtype))
        out_valid.append(torch.ones(1, dtype=torch.bool, device=device))
        budget[src] = (budget[src] - ships).clamp(min=0.0)
        used_src[src] = True
        added += 1

    if added <= 0:
        return _empty_entries(device, dtype), leftover
    return LaunchEntries(
        source_slots=torch.cat(out_src, dim=0),
        target_slots=torch.cat(out_tgt, dim=0),
        ships=torch.cat(out_ships, dim=0),
        angle=torch.cat(out_angle, dim=0),
        eta=torch.cat(out_eta, dim=0),
        valid=torch.cat(out_valid, dim=0),
    ), budget


def _apply_mini_rollout_adjustment(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    prod: Tensor,
    cache,
    garrison_status,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_mini_rollout_4p):
        return score
    step = int(float(obs.step.reshape(-1)[0].item()))
    if step < int(config.mini_rollout_start) or step > int(config.mini_rollout_turn_limit):
        return score
    if getattr(garrison_status, "arrivals_by_owner", None) is None:
        return score
    active = cand_active.any(dim=-1) & torch.isfinite(score)
    if not bool(active.any()):
        return score

    P = int(obs.P)
    H = int(garrison_status.owner.shape[-1]) - 1
    if P <= 0 or H <= 0:
        return score
    device = score.device
    dtype = score.dtype
    pid = int(obs.player_id)

    top_k = max(1, min(int(config.mini_rollout_top_k), int(score.numel())))
    masked = torch.where(active, score, torch.full_like(score, float("-inf")))
    top_vals, top_idx = torch.topk(masked, k=top_k)
    keep = torch.isfinite(top_vals)
    if not bool(keep.any()):
        return score
    idx = top_idx[keep]

    tgt = cand_tgt_slot[idx].clamp(0, max(P - 1, 0))
    src = cand_src[idx, 0].clamp(0, max(P - 1, 0))
    send = cand_send[idx, 0].to(dtype)
    eta = cand_eta[idx, 0].to(dtype).clamp(min=1.0)
    k_arr = torch.ceil(eta).to(torch.long).clamp(min=1, max=H)
    owner_at = garrison_status.owner[tgt, k_arr].to(torch.long)
    ships_at = garrison_status.ships[tgt, k_arr].to(dtype)
    prod_t = prod[tgt].to(dtype)

    enemy_or_neutral = owner_at != pid
    arrival_margin = torch.where(enemy_or_neutral, send - ships_at - 1.0, send + ships_at)

    window = int(config.mini_rollout_window)
    end_k = (k_arr + window).clamp(max=H)
    steps_h = torch.arange(1, H + 1, device=device).view(1, H)
    in_window = (steps_h > k_arr.view(-1, 1)) & (steps_h <= end_k.view(-1, 1))
    arr = garrison_status.arrivals_by_owner[tgt, 1:, :].to(dtype)
    owner_ids = torch.arange(int(player_count), device=device).view(1, 1, -1)
    enemy_arr = torch.where(owner_ids != pid, arr, torch.zeros_like(arr)).sum(dim=2)
    my_arr = arr[:, :, pid]
    enemy_window = torch.where(in_window, enemy_arr, torch.zeros_like(enemy_arr)).sum(dim=1)
    my_window = torch.where(in_window, my_arr, torch.zeros_like(my_arr)).sum(dim=1)

    d0 = cache.cross_dist[0].to(dtype)
    enemy_mask = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)
    enemy_speed = fleet_speed(obs.ships.to(dtype).clamp(min=1.0))
    enemy_eta = d0[:, tgt].transpose(0, 1) / enemy_speed.view(1, P).clamp(min=1e-6)
    response_horizon = eta + float(window)
    local_enemy = torch.where(
        (enemy_eta <= response_horizon.view(-1, 1)) & enemy_mask.view(1, P),
        obs.ships.to(dtype).view(1, P) * 0.18,
        torch.zeros_like(enemy_eta),
    ).sum(dim=1)

    future_hold = arrival_margin + my_window + prod_t * float(window) - enemy_window - local_enemy
    target_owner_now = obs.owner_abs.to(torch.long)[tgt]
    enemy_target = obs.is_enemy[tgt]
    neutral_target = obs.is_neutral[tgt]
    own_target = obs.owned[tgt]
    leader = _leader_owner_by_power(obs, prod, player_count=player_count)
    nonleader_enemy = (
        enemy_target
        & (leader >= 0)
        & (target_owner_now >= 0)
        & (target_owner_now != int(leader))
        & (step < 135)
    )

    src_after = obs.ships[src].to(dtype) - send
    src_reserve = 10.0 + prod[src].to(dtype) * 3.25
    source_stripped = src_after < src_reserve
    hold_good = future_hold > (8.0 + prod_t * 1.5)
    hold_bad = future_hold < -6.0

    adjust = torch.zeros_like(score)
    local = torch.zeros(idx.shape[0], dtype=dtype, device=device)
    local = local + torch.where(
        (neutral_target | enemy_target) & hold_good,
        torch.full_like(local, float(config.mini_rollout_hold_bonus)),
        torch.zeros_like(local),
    )
    local = local + torch.where(
        own_target & (future_hold < 10.0),
        torch.full_like(local, float(config.mini_rollout_hold_bonus) * 0.75),
        torch.zeros_like(local),
    )
    local = local - torch.where(
        (neutral_target | enemy_target) & hold_bad,
        torch.full_like(local, float(config.mini_rollout_fail_penalty)),
        torch.zeros_like(local),
    )
    local = local - torch.where(
        source_stripped & (enemy_target | neutral_target),
        torch.full_like(local, float(config.mini_rollout_source_strip_penalty)),
        torch.zeros_like(local),
    )
    local = local - torch.where(
        nonleader_enemy & (~hold_good),
        torch.full_like(local, float(config.mini_rollout_nonleader_enemy_penalty)),
        torch.zeros_like(local),
    )
    adjust.scatter_add_(0, idx, local)
    return score + adjust


def _build_enemy_best_response_launches(
    *,
    tgt: Tensor,
    obs,
    prod: Tensor,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
    dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Approximate each enemy's best immediate response to our candidate target.

    Each enemy gets at most one launch: the owned source with the strongest
    ship-pressure after travel time, aimed at the candidate target.
    """
    device = tgt.device
    N = int(tgt.numel())
    P = int(obs.P)
    pid = int(obs.player_id)
    if N <= 0 or P <= 0:
        empty_long = torch.zeros(N, 0, dtype=torch.long, device=device)
        empty_float = torch.zeros(N, 0, dtype=dtype, device=device)
        empty_bool = torch.zeros(N, 0, dtype=torch.bool, device=device)
        return empty_long, empty_long, empty_float, empty_float, empty_long, empty_bool

    d0 = cache.cross_dist[0].to(dtype)
    owners = obs.owner_abs.to(torch.long)
    ships_now = obs.ships.to(dtype)
    prod_now = prod.to(dtype)
    prod_t = prod_now[tgt.clamp(0, max(P - 1, 0))]

    src_cols = []
    tgt_cols = []
    ship_cols = []
    eta_cols = []
    owner_cols = []
    valid_cols = []
    neg_inf = torch.full((N, 1), float("-inf"), dtype=dtype, device=device)

    for enemy in range(int(player_count)):
        if enemy == pid:
            continue
        source_mask = obs.alive & (owners == int(enemy))
        if not bool(source_mask.any()):
            src_cols.append(torch.zeros(N, 1, dtype=torch.long, device=device))
            tgt_cols.append(tgt.view(N, 1).to(torch.long))
            ship_cols.append(torch.zeros(N, 1, dtype=dtype, device=device))
            eta_cols.append(torch.ones(N, 1, dtype=dtype, device=device))
            owner_cols.append(torch.full((N, 1), int(enemy), dtype=torch.long, device=device))
            valid_cols.append(torch.zeros(N, 1, dtype=torch.bool, device=device))
            continue

        e_idx = torch.nonzero(source_mask, as_tuple=False).flatten().to(torch.long)
        e_ships = ships_now[e_idx]
        e_prod = prod_now[e_idx]
        reserve = float(config.true_one_ply_enemy_reserve_base) + e_prod * float(config.true_one_ply_enemy_reserve_prod)
        send = torch.minimum(
            (e_ships * float(config.true_one_ply_enemy_frac)).floor(),
            (e_ships - reserve).floor().clamp(min=0.0),
        )
        speed = fleet_speed(send.clamp(min=1.0)).clamp(min=1e-6)
        eta = d0[e_idx][:, tgt.clamp(0, max(P - 1, 0))].transpose(0, 1) / speed.view(1, -1)
        not_same = e_idx.view(1, -1) != tgt.view(N, 1)
        valid = (
            (send.view(1, -1) >= float(config.true_one_ply_enemy_min_ships))
            & (eta <= float(config.true_one_ply_enemy_eta_cap))
            & not_same
        )
        pressure = send.view(1, -1) - eta * 0.65 + prod_t.view(N, 1) * 1.4
        pressure = torch.where(valid, pressure, torch.full_like(pressure, float("-inf")))
        best_pressure, best = torch.max(pressure, dim=1, keepdim=True)
        has = torch.isfinite(best_pressure)
        best_src = e_idx[best.squeeze(1)].view(N, 1)
        best_send = send[best.squeeze(1)].view(N, 1)
        best_eta = eta.gather(1, best).clamp(min=1.0)

        src_cols.append(torch.where(has, best_src, torch.zeros_like(best_src)))
        tgt_cols.append(tgt.view(N, 1).to(torch.long))
        ship_cols.append(torch.where(has, best_send, torch.zeros_like(best_send)))
        eta_cols.append(torch.where(has, best_eta, torch.ones_like(best_eta)))
        owner_cols.append(torch.full((N, 1), int(enemy), dtype=torch.long, device=device))
        valid_cols.append(has.to(torch.bool))

    return (
        torch.cat(src_cols, dim=1) if src_cols else torch.zeros(N, 0, dtype=torch.long, device=device),
        torch.cat(tgt_cols, dim=1) if tgt_cols else torch.zeros(N, 0, dtype=torch.long, device=device),
        torch.cat(ship_cols, dim=1) if ship_cols else torch.zeros(N, 0, dtype=dtype, device=device),
        torch.cat(eta_cols, dim=1) if eta_cols else torch.zeros(N, 0, dtype=dtype, device=device),
        torch.cat(owner_cols, dim=1) if owner_cols else torch.zeros(N, 0, dtype=torch.long, device=device),
        torch.cat(valid_cols, dim=1) if valid_cols else torch.zeros(N, 0, dtype=torch.bool, device=device),
    )


def _coord_one_ply_pass(
    *,
    obs,
    prod: Tensor,
    alive_by_step: Tensor,
    cache,
    garrison_status,
    wave_entries: LaunchEntries,
    target_slot: int,
    coord_src: Tensor,
    coord_ships: Tensor,
    coord_eta: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> bool:
    if int(player_count) < 4 or not bool(config.coord_one_ply_gate_4p):
        return True
    P = int(obs.P)
    if P <= 0 or target_slot < 0 or target_slot >= P:
        return False
    if float(prod[target_slot].item()) < float(config.coord_one_ply_min_target_prod):
        return False

    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    tgt = torch.tensor([int(target_slot)], dtype=torch.long, device=device)
    sent_mask = wave_entries.valid & (wave_entries.target_slots.clamp(0, max(P - 1, 0)) == int(target_slot))
    if not bool(sent_mask.any()):
        return False

    primary_src = wave_entries.source_slots[sent_mask].clamp(0, max(P - 1, 0)).to(torch.long).view(1, -1)
    primary_tgt = torch.full_like(primary_src, int(target_slot))
    primary_ships = wave_entries.ships[sent_mask].to(dtype).view(1, -1)
    primary_eta = wave_entries.eta[sent_mask].to(dtype).clamp(min=1.0).view(1, -1)
    primary_owner = torch.full_like(primary_src, pid, dtype=torch.long)
    primary_valid = torch.ones_like(primary_src, dtype=torch.bool)

    extra_src = coord_src.reshape(1, 1).clamp(0, max(P - 1, 0)).to(torch.long)
    extra_tgt = torch.full_like(extra_src, int(target_slot))
    extra_ships = coord_ships.reshape(1, 1).to(dtype)
    extra_eta = coord_eta.reshape(1, 1).to(dtype).clamp(min=1.0)
    extra_owner = torch.full_like(extra_src, pid, dtype=torch.long)
    extra_valid = extra_ships > 0

    resp_src, resp_tgt, resp_send, resp_eta, resp_owner, resp_valid = _build_enemy_best_response_launches(
        tgt=tgt,
        obs=obs,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
        dtype=dtype,
    )
    launches = LaunchSet(
        source_slots=torch.cat([primary_src, extra_src, resp_src], dim=1),
        target_slots=torch.cat([primary_tgt, extra_tgt, resp_tgt], dim=1),
        ships=torch.cat([primary_ships, extra_ships, resp_send], dim=1).to(dtype),
        eta=torch.cat([primary_eta, extra_eta, resp_eta], dim=1).to(dtype).clamp(min=1.0),
        owner=torch.cat([primary_owner, extra_owner, resp_owner], dim=1).to(torch.long),
        valid=torch.cat([primary_valid, extra_valid, resp_valid], dim=1).to(torch.bool),
    )
    try:
        diff = sparse_launch_flow_delta(
            garrison_status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=int(player_count),
            launches=launches,
            player_id=pid,
        )
    except Exception:
        return False

    net = diff.net_ship_delta.to(dtype)
    if net.ndim != 2 or int(net.shape[0]) < 1:
        return False
    weights = torch.full((int(player_count),), 0.78, dtype=dtype, device=device)
    if 0 <= pid < int(weights.numel()):
        weights[pid] = 0.0
    leader = _leader_owner_by_power(obs, prod, player_count=player_count)
    if 0 <= int(leader) < int(weights.numel()) and int(leader) != pid:
        weights[int(leader)] = 1.35
    me = net[0, pid] if 0 <= pid < int(net.shape[1]) else torch.tensor(0.0, dtype=dtype, device=device)
    opp = (net[0, : int(player_count)] * weights).sum()
    one_ply_net = float((me - opp).item())
    response_mass = float(resp_send.masked_fill(~resp_valid, 0.0).sum().item()) if resp_send.numel() else 0.0
    committed = float((primary_ships.sum() + extra_ships.sum()).item())
    response_ratio = response_mass / max(1.0, committed)
    if one_ply_net < float(config.coord_one_ply_hard_min_net):
        return False
    if response_ratio > float(config.coord_one_ply_response_ratio_cap) and one_ply_net < 8.0:
        return False
    return one_ply_net >= float(config.coord_one_ply_min_net)


def _apply_true_one_ply_rescore(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    prod: Tensor,
    alive_by_step: Tensor,
    cache,
    garrison_status,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_true_one_ply_4p):
        return score
    step = int(float(obs.step.reshape(-1)[0].item()))
    if step < int(config.true_one_ply_start) or step > int(config.true_one_ply_turn_limit):
        return score
    active = cand_active.any(dim=-1) & torch.isfinite(score)
    if not bool(active.any()):
        return score

    P = int(obs.P)
    if P <= 0:
        return score
    device = score.device
    dtype = score.dtype
    pid = int(obs.player_id)

    top_k = max(1, min(int(config.true_one_ply_top_k), int(score.numel())))
    masked = torch.where(active, score, torch.full_like(score, float("-inf")))
    top_vals, top_idx = torch.topk(masked, k=top_k)
    keep = torch.isfinite(top_vals)
    if not bool(keep.any()):
        return score
    idx = top_idx[keep]
    N = int(idx.numel())
    L = int(cand_src.shape[1])

    tgt = cand_tgt_slot[idx].clamp(0, max(P - 1, 0)).to(torch.long)
    own_src = cand_src[idx].clamp(0, max(P - 1, 0)).to(torch.long)
    own_tgt = tgt.view(N, 1).expand(N, L)
    own_send = cand_send[idx].to(dtype)
    own_eta = cand_eta[idx].to(dtype).clamp(min=1.0)
    own_valid = cand_active[idx].to(torch.bool) & (own_send > 0)
    own_owner = torch.full((N, L), pid, dtype=torch.long, device=device)

    resp_src, resp_tgt, resp_send, resp_eta, resp_owner, resp_valid = _build_enemy_best_response_launches(
        tgt=tgt,
        obs=obs,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
        dtype=dtype,
    )
    launches = LaunchSet(
        source_slots=torch.cat([own_src, resp_src], dim=1).to(torch.long),
        target_slots=torch.cat([own_tgt, resp_tgt], dim=1).to(torch.long),
        ships=torch.cat([own_send, resp_send], dim=1).to(dtype),
        eta=torch.cat([own_eta, resp_eta], dim=1).to(dtype).clamp(min=1.0),
        owner=torch.cat([own_owner, resp_owner], dim=1).to(torch.long),
        valid=torch.cat([own_valid, resp_valid], dim=1).to(torch.bool),
    )
    try:
        diff = sparse_launch_flow_delta(
            garrison_status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=int(player_count),
            launches=launches,
            player_id=pid,
        )
    except Exception:
        return score

    net = diff.net_ship_delta.to(dtype)
    if net.ndim != 2 or int(net.shape[0]) != N:
        return score

    weights = torch.full((int(player_count),), 0.78, dtype=dtype, device=device)
    if 0 <= pid < int(weights.numel()):
        weights[pid] = 0.0
    leader = _leader_owner_by_power(obs, prod, player_count=player_count)
    if 0 <= int(leader) < int(weights.numel()) and int(leader) != pid:
        weights[int(leader)] = 1.35
    me = net[:, pid] if 0 <= pid < int(net.shape[1]) else torch.zeros(N, dtype=dtype, device=device)
    opp = (net[:, : int(player_count)] * weights.view(1, -1)).sum(dim=1)
    one_ply_net = me - opp

    prod_t = prod[tgt].to(dtype)
    target_owner = obs.owner_abs.to(torch.long)[tgt]
    target_enemy = target_owner >= 0
    target_not_mine = target_owner != pid
    response_mass = resp_send.masked_fill(~resp_valid, 0.0).sum(dim=1) if resp_send.numel() else torch.zeros(N, dtype=dtype, device=device)
    own_mass = own_send.masked_fill(~own_valid, 0.0).sum(dim=1)
    response_ratio = response_mass / own_mass.clamp(min=1.0)

    local = score[idx] * float(config.true_one_ply_base_weight) + one_ply_net * float(config.true_one_ply_net_scale)
    local = local + torch.where(
        target_not_mine & (prod_t >= 3.0),
        prod_t * 0.12,
        torch.zeros_like(local),
    )
    local = local + torch.where(
        one_ply_net >= float(config.true_one_ply_good_net),
        torch.full_like(local, float(config.true_one_ply_good_bonus)),
        torch.zeros_like(local),
    )
    local = local - torch.where(
        one_ply_net <= float(config.true_one_ply_bad_net),
        torch.full_like(local, float(config.true_one_ply_bad_penalty)),
        torch.zeros_like(local),
    )
    local = local - torch.where(
        (target_not_mine & (prod_t >= 3.0) & (response_ratio > 0.85) & (one_ply_net < 4.0)),
        torch.full_like(local, float(config.true_one_ply_bad_penalty) * 0.85),
        torch.zeros_like(local),
    )
    local = local + torch.where(
        (target_not_mine & (prod_t >= 3.0) & (response_ratio < 0.45) & (one_ply_net > 8.0)),
        torch.full_like(local, float(config.true_one_ply_good_bonus) * 0.85),
        torch.zeros_like(local),
    )

    hard_bad = (
        (one_ply_net <= float(config.true_one_ply_hard_bad_net))
        | (target_enemy & (response_ratio > 1.25) & (one_ply_net < -12.0))
    )
    local = torch.where(hard_bad, torch.full_like(local, float("-inf")), local)

    out = score.clone()
    out[idx] = local
    return out


def _tier_candidates(
    *,
    movement: PlanetMovement,
    source_idx: Tensor,
    source_exists: Tensor,
    target_idx: Tensor,
    target_exists: Tensor,
    target_is_mine: Tensor,
    drain: Tensor,
    floor: Tensor,
    eta_cap: Tensor,
    size_mult: float,
    S: int,
    T: int,
    pid: int,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    device,
    dtype,
):
    """Build scored candidates for one fleet-size fraction."""
    sizes = (drain.view(S, 1) * float(size_mult)).floor().clamp(min=1.0).expand(S, T)
    K = int(floor.shape[-1])

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )

    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    return cand_src, cand_send, cand_angle, cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, score


def _thirdparty_threat_4p(
    obs, cache, garrison_status, *, horizon, window, player_count, pid, dtype, device,
):
    """Per owned planet P, the MAX over single opponents of reachable force onto P:
    latent enemy garrison within ``horizon`` (distance/speed-decayed) + in-flight enemy
    arrivals within ``window``. This is the *strongest single third party* that could
    take P if we bare it -- the mechanism behind 63% of our 4P planet losses. The MAX
    (not the sum) is deliberate: a planet is stolen by one neighbour, not all combined."""
    P = int(obs.P)
    if P == 0:
        return torch.zeros(0, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                  # [Q, P] distances
    ships = obs.ships.to(dtype)
    owners = obs.owner_abs.to(torch.long)
    alive = obs.alive
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)   # per-source Q reach
    eye = torch.eye(P, device=device, dtype=torch.bool)
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                      # [Q, P]
    base_valid = alive.view(P, 1) & ~eye
    contrib = torch.where(base_valid, ships.view(P, 1) * decay, torch.zeros_like(decay))  # [Q, P]

    arr = getattr(garrison_status, "arrivals_by_owner", None)
    W = 0
    if arr is not None:
        W = max(0, min(int(window), int(arr.shape[-2]) - 1))

    tp = torch.zeros(P, dtype=dtype, device=device)
    for e in range(int(player_count)):
        if e == pid:
            continue
        mask_e = (owners == e) & alive
        latent_e = (contrib * mask_e.view(P, 1)).sum(dim=0)            # [P] onto each target
        inflight_e = torch.zeros(P, dtype=dtype, device=device)
        if arr is not None and W > 0 and e < int(arr.shape[-1]):
            inflight_e = arr[:, 1:W + 1, e].to(dtype).sum(dim=1)
        tp = torch.maximum(tp, latent_e + inflight_e)
    return tp


def _global_select_4p(
    *, P, W, device, dtype, score, cand_src, cand_send, cand_angle, cand_eta,
    cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, source_budget,
    target_exists, roi_threshold, tp_threat, ships, owned,
    vuln_weight, margin, drop_sweeps, drop_deadband,
):
    """4P-only global-select (§2 Idea A): greedy initial solution -- byte-identical to
    ``_greedy_select`` -- then a third-party source-vulnerability drop sweep.

    Phase 1 reproduces ``_greedy_select`` exactly while recording each wave's score.
    Phase 2 drops a wave when its cumulative source drain leaves a *holdable* owned
    planet short against the strongest single opponent's reachable force (``tp_threat``)
    and the marginal vulnerability cost ``vuln_weight * shortfall`` exceeds the wave's
    competitive score. Removes launches only (no add/force/rescore); savable-gated
    (a planet is only defended when ``ships >= tp_threat + margin`` at full garrison).
    With no holdable planet bared the sweep is a no-op == ``_greedy_select``."""
    L = int(cand_src.shape[1])
    budget = source_budget.clone()
    target_taken = ~target_exists.clone()
    defended = torch.zeros(P, dtype=torch.bool, device=device)
    used_src = torch.zeros(P, dtype=torch.bool, device=device)

    w_src = torch.zeros(W, L, dtype=torch.long, device=device)
    w_send = torch.zeros(W, L, dtype=dtype, device=device)
    w_angle = torch.zeros(W, L, dtype=dtype, device=device)
    w_eta = torch.ones(W, L, dtype=dtype, device=device)
    w_tgt = torch.zeros(W, L, dtype=torch.long, device=device)
    w_active = torch.zeros(W, L, dtype=torch.bool, device=device)
    w_score = torch.full((W,), float("-inf"), dtype=dtype, device=device)
    w_fired = torch.zeros(W, dtype=torch.bool, device=device)

    # ---- Phase 1: greedy initial solution (mirror of _greedy_select) ----
    for w in range(W):
        taken_cand = target_taken[cand_tgt_short]
        budget_at = budget[cand_src]
        can_fund = ((cand_send <= budget_at) | ~cand_active).all(dim=-1)
        tgt_used_as_src = used_src[cand_tgt_slot]
        contrib_defended = (defended[cand_src] & cand_active).any(dim=-1)
        mask = torch.isfinite(score) & ~taken_cand & can_fund & ~tgt_used_as_src & ~contrib_defended
        masked = torch.where(mask, score, torch.full_like(score, float("-inf")))
        best_c = _stable_argmax(masked)
        best_score = masked[best_c]
        fired = bool(torch.isfinite(best_score) & (best_score > roi_threshold))
        if not fired:
            break
        sel_src = cand_src[best_c]
        sel_send = cand_send[best_c]
        sel_active = cand_active[best_c]
        w_src[w] = sel_src
        w_send[w] = torch.where(sel_active, sel_send, torch.zeros_like(sel_send))
        w_angle[w] = cand_angle[best_c]
        w_eta[w] = cand_eta[best_c]
        w_tgt[w] = cand_tgt_slot[best_c]
        w_active[w] = sel_active
        w_score[w] = best_score
        w_fired[w] = True
        debit = torch.zeros_like(budget)
        debit.scatter_add_(0, sel_src, torch.where(sel_active, sel_send, torch.zeros_like(sel_send)))
        budget = (budget - debit).clamp(min=0.0)
        target_taken[cand_tgt_short[best_c]] = True
        src_mark = torch.zeros(P, dtype=torch.long, device=device)
        src_mark.scatter_add_(0, sel_src, sel_active.to(torch.long))
        used_src = used_src | (src_mark > 0)
        sel_tgt = cand_tgt_slot[best_c]
        sel_is_def = bool(cand_is_def[best_c])
        defended[sel_tgt] = defended[sel_tgt] | sel_is_def

    # ---- Phase 2: third-party source-vulnerability drop sweep ----
    need = tp_threat.to(dtype) + float(margin)
    holdable = owned & (ships >= need)            # only defend planets savable at full garrison
    lam = float(vuln_weight)
    deadband = float(drop_deadband)

    def _committed(active_send):
        c = torch.zeros(P, dtype=dtype, device=device)
        c.scatter_add_(0, w_src.reshape(W * L), active_send.reshape(W * L))
        return c

    def _penalty_sum(committed):
        short = (need - (ships - committed)).clamp(min=0.0)
        return torch.where(holdable, short, torch.zeros_like(short)).sum()

    kept = w_fired.clone()
    if lam > 0.0 and bool(holdable.any()) and bool(w_fired.any()):
        for _ in range(max(1, int(drop_sweeps))):
            order = torch.argsort(torch.where(kept, w_score, torch.full_like(w_score, float("inf"))))
            dropped_any = False
            for wi in order.tolist():
                if not bool(kept[wi]):
                    continue
                send_now = torch.where((kept & w_fired).view(W, 1) & w_active, w_send, torch.zeros_like(w_send))
                pen_before = _penalty_sum(_committed(send_now))
                kept2 = kept.clone()
                kept2[wi] = False
                send_drop = torch.where((kept2 & w_fired).view(W, 1) & w_active, w_send, torch.zeros_like(w_send))
                pen_after = _penalty_sum(_committed(send_drop))
                gain = lam * float(pen_before - pen_after) - float(w_score[wi])
                if gain > deadband:
                    kept[wi] = False
                    dropped_any = True
            if not dropped_any:
                break

    keep_wl = (kept & w_fired).view(W, 1)
    w_active = w_active & keep_wl
    w_send = torch.where(w_active, w_send, torch.zeros_like(w_send))
    leftover = (source_budget - _committed(w_send)).clamp(min=0.0)

    WL = W * L
    entries = LaunchEntries(
        source_slots=w_src.reshape(WL),
        target_slots=w_tgt.reshape(WL),
        ships=torch.where(w_active, w_send, torch.zeros_like(w_send)).reshape(WL),
        angle=torch.where(w_active, w_angle, torch.zeros_like(w_angle)).reshape(WL),
        eta=torch.where(w_active, w_eta, torch.ones_like(w_eta)).reshape(WL),
        valid=w_active.reshape(WL),
    )
    return entries, leftover


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    memory=None,
):
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    target_idx, target_exists, lane_targets = _append_lane_anchor_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        prod=prod,
        source_mask=source_mask,
        config=config,
        player_count=player_count,
    )
    target_idx, target_exists, domain_targets = _append_domain_race_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        prod=prod,
        source_mask=source_mask,
        config=config,
        player_count=player_count,
    )
    target_idx, target_exists, block_targets = _append_enemy_domain_block_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        prod=prod,
        source_mask=source_mask,
        config=config,
        player_count=player_count,
    )
    target_idx, target_exists, winner_path = _append_winner_path_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        source_mask=source_mask,
        config=config,
        player_count=player_count,
        memory=memory,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
        if beta > 0.0 or bool(config.enable_regroup) else None
    )
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange,
            eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid, reinforcement=reinforcement,
    )

    tier_parts = [
        _tier_candidates(
            movement=movement,
            source_idx=source_idx,
            source_exists=source_exists,
            target_idx=target_idx,
            target_exists=target_exists,
            target_is_mine=target_is_mine,
            drain=drain,
            floor=floor,
            eta_cap=eta_cap,
            size_mult=float(mult),
            S=S,
            T=T,
            pid=pid,
            garrison_status=garrison_status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            device=device,
            dtype=dtype,
        )
        for mult in config.size_multipliers
    ]

    cand_src = torch.cat([p[0] for p in tier_parts], dim=0)
    cand_send = torch.cat([p[1] for p in tier_parts], dim=0)
    cand_angle = torch.cat([p[2] for p in tier_parts], dim=0)
    cand_eta = torch.cat([p[3] for p in tier_parts], dim=0)
    cand_active = torch.cat([p[4] for p in tier_parts], dim=0)
    cand_tgt_slot = torch.cat([p[5] for p in tier_parts], dim=0)
    cand_tgt_short = torch.cat([p[6] for p in tier_parts], dim=0)
    cand_is_def = torch.cat([p[7] for p in tier_parts], dim=0)
    score = torch.cat([p[8] for p in tier_parts], dim=0)
    score = _apply_lane_anchor_bonus(
        score=score,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        prod=prod,
        config=config,
        player_count=player_count,
        lane_targets=lane_targets,
    )
    score = _apply_domain_race_adjustment(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        config=config,
        player_count=player_count,
        domain_targets=domain_targets,
    )
    score = _apply_enemy_domain_block_adjustment(
        score=score,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        config=config,
        player_count=player_count,
        block_targets=block_targets,
    )
    score = _apply_winner_path_adjustment(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        config=config,
        player_count=player_count,
        winner_path=winner_path,
    )
    score = _apply_top_director_adjustment(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
    )
    score = _apply_influence_intent_adjustment(
        score=score,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
    )
    score = _apply_expansion_drive_4p(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
    )
    score = _apply_orbit_response_gate(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        cache=cache,
        garrison_status=garrison_status,
        config=config,
        player_count=player_count,
    )
    score = _apply_mini_rollout_adjustment(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        prod=prod,
        cache=cache,
        garrison_status=garrison_status,
        config=config,
        player_count=player_count,
    )
    score = _apply_true_one_ply_rescore(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        prod=prod,
        alive_by_step=alive_by_step,
        cache=cache,
        garrison_status=garrison_status,
        config=config,
        player_count=player_count,
    )

    source_budget = obs.ships.to(dtype).clone()
    target_exists_for_normal = target_exists.clone()
    reserve_entries = _empty_entries(device, dtype)
    reserve_mask, reserve_score = _top_director_reserved_safe_mask(
        score=score,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
    )
    if bool(reserve_mask.any()):
        reserve_entries, source_budget = _greedy_select(
            P=P, W=1, device=device, dtype=dtype, score=reserve_score,
            cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
            cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_is_def=cand_is_def, source_budget=source_budget,
            target_exists=target_exists_for_normal, roi_threshold=float(config.top_director_reserve_roi),
        )
        if bool(reserve_entries.valid.any()):
            taken = reserve_entries.target_slots[reserve_entries.valid].clamp(0, max(P - 1, 0))
            if taken.numel() > 0:
                taken_mask = torch.zeros(P, dtype=torch.bool, device=device)
                taken_mask.scatter_(0, taken, True)
                target_exists_for_normal = target_exists_for_normal & (~taken_mask[target_idx.clamp(0, P - 1)])
    _tp_step = int(float(obs.step.reshape(-1)[0].item()))
    if (int(player_count) >= 4 and bool(config.enable_thirdparty_guard_4p)
            and int(config.tp_guard_start) <= _tp_step <= int(config.tp_guard_limit)):
        # 4P-only third-party source-stripping guard (sample157). 2P / flag-off / out of
        # the turn window fall through to the unchanged _greedy_select below.
        tp_threat = _thirdparty_threat_4p(
            obs, cache, garrison_status,
            horizon=float(config.tp_guard_horizon), window=int(config.tp_guard_window),
            player_count=player_count, pid=pid, dtype=dtype, device=device,
        )
        wave_entries, leftover = _global_select_4p(
            P=P, W=W, device=device, dtype=dtype, score=score,
            cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
            cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_is_def=cand_is_def, source_budget=source_budget,
            target_exists=target_exists_for_normal, roi_threshold=float(config.roi_threshold),
            tp_threat=tp_threat, ships=obs.ships.to(dtype), owned=(obs.owned & obs.alive),
            vuln_weight=float(config.tp_guard_vuln_weight), margin=float(config.tp_guard_margin),
            drop_sweeps=int(config.tp_guard_drop_sweeps), drop_deadband=float(config.tp_guard_drop_deadband),
        )
    else:
        wave_entries, leftover = _greedy_select(
            P=P, W=W, device=device, dtype=dtype, score=score,
            cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
            cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_is_def=cand_is_def, source_budget=source_budget,
            target_exists=target_exists_for_normal, roi_threshold=float(config.roi_threshold),
        )
    coord_entries, leftover = _plan_coord_followups(
        movement=movement,
        obs=obs,
        prod=prod,
        alive_by_step=alive_by_step,
        cache=cache,
        garrison_status=garrison_status,
        wave_entries=wave_entries,
        leftover=leftover,
        config=config,
        player_count=player_count,
    )

    if not bool(config.enable_regroup):
        return concat_launch_entries([reserve_entries, wave_entries, coord_entries])
    if enemy_mass is None:
        enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([reserve_entries, wave_entries, coord_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
        memory=memory,
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.25,
    max_regroup_targets_per_source=8,
    enable_orbit_response_gate_4p=True,
    orbit_response_start=15,
    orbit_response_ramp_end=80,
    orbit_response_ramp_min=0.35,
    enable_influence_map_4p=True,
    enable_fleet_intent_4p=True,
    enable_mini_rollout_4p=True,
    enable_true_one_ply_4p=True,
    enable_coord_followup_4p=True,
    enable_expansion_drive_4p=True,
    expansion_drive_start=8,
    expansion_drive_limit=150,
    expansion_drive_low_planets=6,
    expansion_drive_base_bonus=0.40,
    expansion_drive_lag_bonus=0.85,
    expansion_drive_prod_bonus=0.16,
    expansion_drive_outer_bonus=0.32,
    expansion_drive_cheap_bonus=0.22,
    expansion_drive_enemy_penalty=0.35,
    expansion_drive_center_penalty=0.42,
    expansion_drive_pressure_penalty=0.020,
    expansion_drive_strip_penalty=0.75,
    expansion_drive_min_prod=1.0,
    expansion_campaign_radius=16.0,
    expansion_campaign_prod_bonus=0.045,
    expansion_campaign_cheap_bonus=0.030,
    expansion_campaign_risk_penalty=0.006,
    expansion_drive_compound_bonus=0.10,
    coord_start_turn=18,
    coord_turn_limit=165,
    coord_max_extra=2,
    coord_eta_delta=5.0,
    coord_min_leftover=16.0,
    coord_source_reserve_base=9.0,
    coord_source_reserve_prod=2.2,
    coord_send_frac=0.36,
    coord_send_cap=42.0,
    coord_min_target_prod=4.0,
    coord_one_ply_gate_4p=True,
    coord_one_ply_min_target_prod=4.0,
    coord_one_ply_top_sources=6,
    coord_one_ply_min_net=5.0,
    coord_one_ply_hard_min_net=-35.0,
    coord_one_ply_response_ratio_cap=1.20,
    # sample157: third-party source-stripping guard (4P). Baked in CODE, not params.json
    # (params.json is ignored under the eval harness: __file__ is unset so _HERE falls
    # back to cwd and params.json is never found). Defaults off in the dataclass so 2P is
    # untouched; enabled here for all 4P modes.
    enable_thirdparty_guard_4p=True,
    tp_guard_horizon=10.0,
    tp_guard_window=10,
    tp_guard_margin=3.0,
    tp_guard_vuln_weight=1.0,
    tp_guard_drop_sweeps=3,
    tp_guard_drop_deadband=0.0,
    tp_guard_start=20,
    tp_guard_limit=250,
)



# ---------------------------------------------------------------------------
# Search params injected by search/generate_candidates.py
# ---------------------------------------------------------------------------

import json as _ow_json


def _ow_load_strategy_params() -> dict:
    path = os.path.join(_HERE, "params.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = _ow_json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _ow_apply_config_overrides(config: ProducerLiteConfig, values: dict | None) -> ProducerLiteConfig:
    if not values:
        return config
    allowed = {field.name for field in dataclasses.fields(ProducerLiteConfig)}
    cleaned = {}
    for key, value in values.items():
        if key not in allowed:
            continue
        if key == "size_multipliers":
            value = tuple(float(x) for x in value)
        cleaned[key] = value
    return dataclasses.replace(config, **cleaned)


_OW_STRATEGY_PARAMS = _ow_load_strategy_params()
CONFIG_2P = _ow_apply_config_overrides(ProducerLiteConfig(), _OW_STRATEGY_PARAMS.get("config_2p"))
CONFIG_4P = _ow_apply_config_overrides(CONFIG_4P, _OW_STRATEGY_PARAMS.get("config_4p"))


# Single-file 4P selector:
# - s8_burst keeps sample8's multi-size 4P behavior.
# - s7_stable uses the same orbit_lite engine but sample7-like one-size behavior.
# This avoids dynamic folder imports, which changed behavior in previous tests.
CONFIG_4P_S8_BURST = dataclasses.replace(
    CONFIG_4P,
    enable_coord_followup_4p=True,
    coord_start_turn=20,
    coord_turn_limit=150,
    coord_max_extra=2,
    coord_eta_delta=4.0,
    coord_min_leftover=18.0,
    coord_source_reserve_base=10.0,
    coord_source_reserve_prod=3.0,
    coord_send_frac=0.32,
    coord_send_cap=34.0,
    coord_min_target_prod=3.0,
)
CONFIG_4P_S7_STABLE = dataclasses.replace(
    CONFIG_4P,
    horizon=13,
    max_sources_per_lane=6,
    max_offensive_targets=12,
    max_defensive_targets=2,
    max_waves_per_turn=6,
    roi_threshold=1.5,
    min_ships_to_launch=4.0,
    reinforce_size_beta=2.2,
    reinforce_eta_free=3.0,
    reinforce_eta_scale=12.0,
    enable_regroup=True,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.25,
    max_regroup_sources_per_lane=6,
    max_regroup_targets_per_source=8,
    terminal_phase_turns=0,
    size_multipliers=(1.0,),
)

CONFIG_4P_LANE_ANCHOR = dataclasses.replace(
    CONFIG_4P,
    horizon=14,
    max_sources_per_lane=7,
    max_offensive_targets=14,
    max_defensive_targets=2,
    max_waves_per_turn=7,
    roi_threshold=1.42,
    reinforce_size_beta=2.2,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.25,
    max_regroup_sources_per_lane=6,
    max_regroup_targets_per_source=8,
    terminal_phase_turns=0,
    size_multipliers=(0.75, 1.0),
    enable_lane_anchor_4p=True,
)

CONFIG_4P_DOMAIN_RACE = dataclasses.replace(
    CONFIG_4P,
    horizon=14,
    max_sources_per_lane=7,
    max_offensive_targets=14,
    max_defensive_targets=2,
    max_waves_per_turn=7,
    roi_threshold=1.44,
    min_ships_to_launch=4.0,
    reinforce_size_beta=2.0,
    reinforce_eta_free=3.0,
    reinforce_eta_scale=12.0,
    enable_regroup=True,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.25,
    max_regroup_sources_per_lane=6,
    max_regroup_targets_per_source=8,
    terminal_phase_turns=0,
    size_multipliers=(0.75, 1.0),
    enable_domain_race_4p=True,
)

CONFIG_4P_WINNER_PATH = dataclasses.replace(
    CONFIG_4P,
    horizon=15,
    max_sources_per_lane=7,
    max_offensive_targets=15,
    max_defensive_targets=2,
    max_waves_per_turn=7,
    roi_threshold=1.50,
    min_ships_to_launch=4.0,
    reinforce_size_beta=2.2,
    reinforce_eta_free=3.0,
    reinforce_eta_scale=12.0,
    enable_regroup=True,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.25,
    max_regroup_sources_per_lane=6,
    max_regroup_targets_per_source=8,
    terminal_phase_turns=0,
    size_multipliers=(0.75, 1.0),
    enable_winner_path_4p=True,
    winner_path_turn_limit=118,
    winner_path_bonus=0.58,
    winner_path_prod_bonus=0.055,
    winner_path_outer_bonus=0.18,
    winner_path_offlane_cheap_penalty=0.22,
    winner_anchor_hold_base=38.0,
    winner_anchor_hold_prod=10.5,
    winner_anchor_drain_penalty=0.55,
)

CONFIG_4P_ENEMY_DOMAIN_BLOCK = dataclasses.replace(
    CONFIG_4P,
    horizon=14,
    max_sources_per_lane=7,
    max_offensive_targets=14,
    max_defensive_targets=2,
    max_waves_per_turn=7,
    roi_threshold=1.46,
    min_ships_to_launch=4.0,
    reinforce_size_beta=2.1,
    reinforce_eta_free=3.0,
    reinforce_eta_scale=12.0,
    enable_regroup=True,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.25,
    max_regroup_sources_per_lane=6,
    max_regroup_targets_per_source=8,
    terminal_phase_turns=0,
    size_multipliers=(0.75, 1.0),
    enable_enemy_domain_block_4p=True,
)

CONFIG_4P_TOP_DIRECTOR = dataclasses.replace(
    CONFIG_4P_S8_BURST,
    enable_top_director_4p=True,
    enable_coord_followup_4p=True,
)


def _load_oracle_rules() -> dict:
    path = os.path.join(_HERE, "oracle_rules.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


_ORACLE_RULES = _load_oracle_rules()


def _dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _angle_from_center_xy(x: float, y: float) -> float:
    return math.atan2(y - 50.0, x - 50.0)


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _initial_board_features(obs_tensors: dict) -> dict[str, float]:
    obs = parse_obs(obs_tensors)
    player = int(obs.player_id)
    alive = [i for i in range(int(obs.P)) if bool(obs.alive[i].item())]
    owned = [i for i in alive if int(round(float(obs.owner_abs[i].item()))) == player]
    if not owned:
        return {"fallback": 1.0}

    start = max(owned, key=lambda i: float(obs.ships[i].item()))
    sx = float(obs.x[start].item())
    sy = float(obs.y[start].item())
    start_angle = _angle_from_center_xy(sx, sy)

    neutrals = [i for i in alive if int(round(float(obs.owner_abs[i].item()))) < 0]
    enemies = [
        i for i in alive
        if int(round(float(obs.owner_abs[i].item()))) >= 0
        and int(round(float(obs.owner_abs[i].item()))) != player
    ]

    def band(max_dist: float) -> dict[str, float]:
        items = [
            i for i in neutrals
            if _dist_xy(sx, sy, float(obs.x[i].item()), float(obs.y[i].item())) <= max_dist
        ]
        return {
            f"n{int(max_dist)}_count": float(len(items)),
            f"n{int(max_dist)}_prod": float(sum(float(obs.prod[i].item()) for i in items)),
            f"n{int(max_dist)}_ships": float(sum(float(obs.ships[i].item()) for i in items)),
            f"n{int(max_dist)}_high_prod": float(sum(1 for i in items if float(obs.prod[i].item()) >= 3.0)),
            f"n{int(max_dist)}_cheap": float(sum(1 for i in items if float(obs.ships[i].item()) <= 20.0)),
        }

    f: dict[str, float] = {
        "planet_count": float(len(alive)),
        "enemy_dist": min(
            (
                _dist_xy(sx, sy, float(obs.x[i].item()), float(obs.y[i].item()))
                for i in enemies
            ),
            default=999.0,
        ),
    }
    for d in (25.0, 45.0, 65.0):
        f.update(band(d))

    best_chain = 0.0
    for mid in neutrals:
        mx = float(obs.x[mid].item())
        my = float(obs.y[mid].item())
        md = _dist_xy(sx, sy, mx, my)
        mid_prod = float(obs.prod[mid].item())
        mid_ships = float(obs.ships[mid].item())
        if not (8.0 <= md <= 48.0):
            continue
        if mid_prod < 2.0 and not (15.0 <= mid_ships <= 40.0):
            continue
        mid_angle = _angle_from_center_xy(mx, my)
        if _angle_diff(start_angle, mid_angle) > 1.15:
            continue
        for outer in neutrals:
            if outer == mid:
                continue
            ox = float(obs.x[outer].item())
            oy = float(obs.y[outer].item())
            od = _dist_xy(mx, my, ox, oy)
            if not (18.0 <= od <= 80.0):
                continue
            if _angle_diff(mid_angle, _angle_from_center_xy(ox, oy)) > 0.85:
                continue
            score = (
                mid_prod * 5.5
                + max(0.0, 42.0 - mid_ships) * 0.12
                + float(obs.prod[outer].item()) * 7.0
                + float(obs.ships[outer].item()) * 0.08
                - md * 0.12
                - od * 0.08
            )
            best_chain = max(best_chain, score)
    f["chain_score"] = best_chain

    best_outer_anchor = 0.0
    best_support_density = 0.0
    for anchor in neutrals:
        ax = float(obs.x[anchor].item())
        ay = float(obs.y[anchor].item())
        ar = _dist_xy(ax, ay, 50.0, 50.0)
        ad = _dist_xy(sx, sy, ax, ay)
        ap = float(obs.prod[anchor].item())
        ash = float(obs.ships[anchor].item())
        aa = _angle_from_center_xy(ax, ay)
        if ar < 30.0 or ad > 78.0:
            continue
        if ap < 3.0 and ash < 45.0:
            continue
        if _angle_diff(start_angle, aa) > 1.35:
            continue
        support = [
            j for j in neutrals
            if j != anchor
            and _angle_diff(aa, _angle_from_center_xy(float(obs.x[j].item()), float(obs.y[j].item()))) <= 0.95
            and _dist_xy(ax, ay, float(obs.x[j].item()), float(obs.y[j].item())) <= 50.0
            and (float(obs.prod[j].item()) >= 2.0 or float(obs.ships[j].item()) >= 18.0)
        ]
        support_prod = sum(float(obs.prod[j].item()) for j in support)
        support_density = support_prod + len(support) * 1.5
        score = (
            ap * 10.0
            + ash * 0.055
            + ar * 0.055
            + support_prod * 1.65
            + len(support) * 1.25
            - ad * 0.16
        )
        if score > best_outer_anchor:
            best_outer_anchor = score
            best_support_density = support_density
    f["outer_anchor_score"] = best_outer_anchor
    f["support_density"] = best_support_density

    enemy_cluster_risk = 0.0
    for enemy in enemies:
        ex = float(obs.x[enemy].item())
        ey = float(obs.y[enemy].item())
        nearby = [
            j for j in neutrals
            if _dist_xy(ex, ey, float(obs.x[j].item()), float(obs.y[j].item())) <= 46.0
            and _dist_xy(float(obs.x[j].item()), float(obs.y[j].item()), 50.0, 50.0) >= 28.0
        ]
        if not nearby:
            continue
        risk = (
            sum(float(obs.prod[j].item()) for j in nearby) * 3.9
            + sum(1 for j in nearby if float(obs.prod[j].item()) >= 3.0) * 3.4
            + sum(max(0.0, 34.0 - float(obs.ships[j].item())) for j in nearby) * 0.045
            + sum(float(obs.ships[j].item()) for j in nearby) * 0.018
        )
        enemy_cluster_risk = max(enemy_cluster_risk, risk)
    f["enemy_cluster_risk"] = enemy_cluster_risk
    return f


def _choose_4p_mode(obs_tensors: dict) -> str:
    f = _initial_board_features(obs_tensors)
    if f.get("fallback"):
        return "s7_stable"

    oracle_mode = _oracle_rule_mode(f)
    if oracle_mode in ("s7_stable", "s8_burst"):
        return oracle_mode

    near_prod = f["n25_prod"]
    mid_prod = f["n45_prod"]
    mid_cheap = f["n45_cheap"]
    high65 = f["n65_high_prod"]
    enemy_dist = f["enemy_dist"]
    chain = f["chain_score"]
    planets = f["planet_count"]
    outer_anchor = f.get("outer_anchor_score", 0.0)
    support_density = f.get("support_density", 0.0)
    enemy_cluster_risk = f.get("enemy_cluster_risk", 0.0)

    # Loss-gate promotions found by forced-mode probes:
    # - 9874600002-type close high-prod cluster wins with winner_outer_domain.
    # - 12000002-type sparse outer route wins when we deny the rival domain first.
    if (
        enemy_cluster_risk >= 100.0
        and enemy_dist >= 38.0
        and enemy_dist <= 48.0
        and high65 >= 9.0
        and near_prod >= 8.0
        and near_prod <= 11.0
        and mid_prod >= 20.0
        and mid_prod <= 24.0
        and mid_cheap <= 2.5
        and chain >= 54.0
        and planets <= 28.0
    ):
        return "winner_outer_domain"
    if (
        enemy_dist >= 62.0
        and enemy_dist <= 72.0
        and near_prod >= 8.0
        and near_prod <= 11.0
        and mid_prod >= 18.0
        and mid_prod <= 22.0
        and mid_cheap >= 3.5
        and mid_cheap <= 4.5
        and high65 >= 5.0
        and high65 <= 7.0
        and chain >= 60.0
        and outer_anchor >= 75.0
        and support_density >= 18.0
        and support_density <= 24.0
        and planets <= 22.0
    ):
        return "winner_outer_domain"
    if (
        enemy_dist >= 62.0
        and enemy_dist <= 72.0
        and near_prod >= 8.0
        and near_prod <= 10.5
        and mid_prod >= 14.0
        and mid_prod <= 18.0
        and mid_cheap <= 1.5
        and high65 >= 5.0
        and high65 <= 7.0
        and chain >= 48.0
        and chain <= 54.0
        and support_density <= 16.0
        and planets <= 22.0
    ):
        return "enemy_domain_block"

    # Dense high-prod starts are closer to the public top-bot pattern:
    # win one strong route early instead of scattering over every cheap planet.
    if (
        enemy_dist >= 62.0
        and near_prod >= 14.0
        and mid_prod <= 22.0
        and mid_cheap <= 2.5
        and high65 >= 6.0
        and chain >= 44.0
    ):
        return "winner_outer_domain"

    # Sparse boards with one or two expensive anchors need a committed
    # high-value route, not the broad sample8 burst.
    if (
        enemy_dist >= 62.0
        and near_prod >= 8.0
        and near_prod <= 10.5
        and mid_prod <= 18.0
        and mid_cheap <= 1.5
        and high65 >= 6.0
        and chain >= 48.0
    ):
        return "winner_outer_domain"

    # Very rich chain-cluster openings often belong to sample8's burst script:
    # it reaches many high-prod targets before the lane-anchor script overcommits.
    if (
        chain >= 61.0
        and enemy_dist >= 64.0
        and near_prod <= 9.5
        and mid_prod >= 28.0
        and mid_cheap <= 3.5
        and high65 >= 11.0
        and support_density >= 35.0
        and enemy_cluster_risk >= 120.0
    ):
        return "top_director"

    if (
        chain >= 61.0
        and enemy_dist >= 64.0
        and near_prod <= 9.5
        and mid_prod >= 28.0
        and mid_cheap <= 3.5
        and high65 >= 11.0
    ):
        return "s8_burst"

    if (
        chain >= 56.0
        and enemy_dist >= 62.0
        and near_prod <= 11.0
        and mid_prod <= 30.0
        and high65 >= 5.0
    ):
        return "lane_anchor"
    if (
        chain >= 54.0
        and enemy_dist <= 52.0
        and near_prod >= 12.0
        and mid_prod >= 22.0
        and mid_prod <= 28.5
        and mid_cheap <= 2.5
        and high65 >= 10.0
    ):
        return "lane_anchor"

    if (
        enemy_cluster_risk >= 72.0
        and enemy_dist >= 48.0
        and high65 >= 8.0
        and near_prod <= 13.0
        and mid_prod <= 26.0
        and mid_cheap <= 3.5
        and not (outer_anchor >= 46.0 and support_density >= 9.0)
    ):
        return "enemy_domain_block"

    if (
        enemy_dist >= 62.0
        and near_prod <= 11.0
        and mid_prod >= 14.0
        and mid_prod <= 27.0
        and mid_cheap >= 2.0
        and high65 >= 6.0
        and chain >= 55.0
        and outer_anchor >= 46.0
        and support_density >= 9.0
    ):
        return "winner_outer_domain"
    if (
        enemy_dist >= 68.0
        and near_prod <= 6.0
        and mid_prod >= 18.0
        and mid_prod <= 25.0
        and mid_cheap >= 5.0
        and high65 >= 7.0
        and chain >= 50.0
        and outer_anchor >= 44.0
        and support_density >= 8.0
    ):
        return "winner_outer_domain"

    burst_score = (
        high65 * 1.35
        + near_prod * 0.22
        + mid_prod * 0.12
        + chain * 0.035
        - mid_cheap * 0.85
        - max(0.0, 56.0 - enemy_dist) * 0.08
        - max(0.0, near_prod - 14.0) * 0.55
        - max(0.0, mid_prod - 29.0) * 0.35
    )

    if (
        burst_score >= 7.8
        and mid_cheap <= 4.5
        and high65 >= 6.0
        and near_prod <= 14.0
        and mid_prod <= 29.0
        and not (planets <= 28.0 and high65 >= 10.0)
    ):
        return "s8_burst"
    if (
        enemy_dist >= 70.0
        and near_prod <= 14.0
        and mid_prod <= 22.0
        and (
            (chain >= 60.0 and high65 >= 5.0 and mid_cheap <= 4.0)
            or (high65 >= 10.0 and mid_cheap <= 3.5)
            or (high65 >= 5.0 and mid_cheap <= 3.0 and mid_prod <= 16.0 and chain <= 45.0)
        )
    ):
        return "s8_burst"
    if (
        enemy_cluster_risk >= 64.0
        and enemy_dist >= 48.0
        and high65 >= 8.0
        and near_prod <= 13.0
        and mid_prod >= 16.0
        and mid_prod <= 27.0
        and mid_cheap <= 3.5
        and not (near_prod >= 16.0 and enemy_dist <= 52.0)
        and not (outer_anchor >= 46.0 and support_density >= 9.0)
    ):
        return "enemy_domain_block"
    return "s7_stable"


def _oracle_rule_mode(features: dict[str, float]) -> str | None:
    rules = _ORACLE_RULES.get("rules")
    if not isinstance(rules, list):
        return None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        mode = str(rule.get("mode", ""))
        if mode not in ("s7_stable", "s8_burst"):
            continue
        conditions = rule.get("conditions", {})
        if not isinstance(conditions, dict):
            continue
        ok = True
        for key, bounds in conditions.items():
            if key not in features or not isinstance(bounds, dict):
                ok = False
                break
            value = float(features[key])
            if "min" in bounds and value < float(bounds["min"]):
                ok = False
                break
            if "max" in bounds and value > float(bounds["max"]):
                ok = False
                break
        if ok:
            return mode
    default_mode = _ORACLE_RULES.get("default_mode")
    if default_mode in ("s7_stable", "s8_burst"):
        return str(default_mode)
    return None


def _config_for(player_count: int, mode: str | None = None) -> ProducerLiteConfig:
    if int(player_count) < 4:
        return CONFIG_2P
    if mode == "lane_anchor":
        return CONFIG_4P_LANE_ANCHOR
    if mode == "winner_outer_domain":
        return CONFIG_4P_WINNER_PATH
    if mode == "enemy_domain_block":
        return CONFIG_4P_ENEMY_DOMAIN_BLOCK
    if mode == "top_director":
        return CONFIG_4P_TOP_DIRECTOR
    if mode == "s8_burst":
        return CONFIG_4P_S8_BURST
    return CONFIG_4P_S7_STABLE


def _owner_power_snapshot(obs, prod: Tensor, *, player_count: int) -> tuple[Tensor, Tensor, Tensor]:
    dtype = obs.ships.dtype
    device = obs.device
    power = torch.zeros(int(player_count), dtype=dtype, device=device)
    production = torch.zeros(int(player_count), dtype=dtype, device=device)
    planets = torch.zeros(int(player_count), dtype=dtype, device=device)
    owners = obs.owner_abs.to(torch.long)
    for owner in range(int(player_count)):
        mask = obs.alive & (owners == int(owner))
        if bool(mask.any()):
            production[owner] = prod[mask].to(dtype).sum()
            planets[owner] = mask.to(dtype).sum()
            power[owner] = obs.ships[mask].to(dtype).sum() + production[owner] * 13.0
    if int(getattr(obs, "F", 0)) > 0:
        for owner in range(int(player_count)):
            fmask = obs.f_alive & (obs.f_owner.to(torch.long) == int(owner))
            if bool(fmask.any()):
                power[owner] = power[owner] + obs.f_ships[fmask].to(dtype).sum()
    return power, production, planets


def _maybe_reconsider_4p_mode(obs_tensors: dict, current_mode: str | None) -> str | None:
    if current_mode is None:
        return current_mode
    step = int(obs_tensors["step"].reshape(-1)[0].item())
    if step < 60:
        return current_mode
    if current_mode not in ("winner_outer_domain", "enemy_domain_block", "top_director", "s8_burst"):
        return current_mode

    obs = parse_obs(obs_tensors)
    pid = int(obs.player_id)
    player_count = 4
    prod = obs.prod
    power, production, planets = _owner_power_snapshot(obs, prod, player_count=player_count)
    my_power = float(power[pid].item()) if pid < int(power.numel()) else 0.0
    my_prod = float(production[pid].item()) if pid < int(production.numel()) else 0.0
    my_planets = float(planets[pid].item()) if pid < int(planets.numel()) else 0.0
    enemy_power = [
        float(power[o].item())
        for o in range(player_count)
        if o != pid and o < int(power.numel())
    ]
    best_enemy = max(enemy_power, default=0.0)

    # If a burst/path/block opening failed to produce territory by midgame,
    # stop paying for that script's early-game bias and fall back to stable play.
    if step >= 85 and my_planets <= 2.0 and my_prod <= 5.0:
        return "s7_stable"
    if step >= 100 and best_enemy > 0.0 and my_power < best_enemy * 0.52:
        return "s7_stable"

    # s8_burst is useful as an opening, but past the opening phase it should
    # not keep dragging decisions toward broad, high-variance expansion.
    if current_mode == "s8_burst" and step >= 90:
        return "s7_stable"
    return current_mode


def _apply_dynamic_4p_config(
    config: ProducerLiteConfig,
    obs_tensors: dict,
    *,
    player_count: int,
) -> ProducerLiteConfig:
    if int(player_count) < 4:
        return config
    step = int(obs_tensors["step"].reshape(-1)[0].item())
    if step < 60:
        return config
    obs = parse_obs(obs_tensors)
    pid = int(obs.player_id)
    power, production, _planets = _owner_power_snapshot(obs, obs.prod, player_count=int(player_count))
    my_power = float(power[pid].item()) if pid < int(power.numel()) else 0.0
    enemy_power = [
        float(power[o].item())
        for o in range(int(player_count))
        if o != pid and o < int(power.numel())
    ]
    best_enemy = max(enemy_power, default=0.0)
    if best_enemy <= 0.0:
        return config

    if my_power >= best_enemy * 1.18:
        return dataclasses.replace(
            config,
            roi_threshold=float(config.roi_threshold) + 0.10,
            enable_regroup=True,
            orbit_response_turn_limit=max(int(config.orbit_response_turn_limit), 220),
        )
    if my_power <= best_enemy * 0.62:
        return dataclasses.replace(
            config,
            roi_threshold=max(1.30, float(config.roi_threshold) - 0.08),
            max_waves_per_turn=max(int(config.max_waves_per_turn), 7),
            enable_regroup=True,
        )
    return config


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.strategy_mode: str | None = None
        self.winner_path: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.strategy_mode = None
        self.winner_path = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.strategy_mode = None
            mem.winner_path = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        current_player = int(obs_tensors["player"].reshape(-1)[0].item())
        min_count = current_player + 1
        mem.cached_player_count = 4 if max(int(mem.cached_player_count), min_count) > 2 else 2
        _bind_planner(int(mem.cached_player_count))  # split: 2P->orbit_lite_2p, 4P->orbit_lite_4p
        if int(mem.cached_player_count) >= 4 and mem.strategy_mode is None:
            mem.strategy_mode = _choose_4p_mode(obs_tensors)
        if int(mem.cached_player_count) >= 4 and mem.strategy_mode is not None:
            new_mode = _maybe_reconsider_4p_mode(obs_tensors, mem.strategy_mode)
            if new_mode != mem.strategy_mode:
                mem.strategy_mode = new_mode
                mem.winner_path = None
        if int(mem.cached_player_count) < 4:
            mem.strategy_mode = None
        base = _config_for(mem.cached_player_count, mem.strategy_mode)
        step = int(obs_tensors["step"].reshape(-1)[0].item())
        config = _apply_phase_config(base, step)
        config = _apply_dynamic_4p_config(config, obs_tensors, player_count=int(mem.cached_player_count))
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
