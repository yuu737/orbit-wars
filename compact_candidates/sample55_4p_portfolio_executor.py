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

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

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
    # --- 4P domain factory: build a connected outer production base ----------
    enable_domain_factory_4p: bool = False
    domain_factory_turn_limit: int = 175
    domain_factory_project_turn: int = 72
    domain_factory_max_extra_targets: int = 10
    domain_factory_angle_width: float = 1.18
    domain_factory_core_bonus: float = 0.64
    domain_factory_support_bonus: float = 0.38
    domain_factory_expansion_bonus: float = 0.78
    domain_factory_prod_bonus: float = 0.07
    domain_factory_outer_bonus: float = 0.24
    domain_factory_offdomain_cheap_penalty: float = 0.18
    domain_factory_hold_base: float = 34.0
    domain_factory_hold_prod: float = 7.5
    domain_factory_support_hold_base: float = 14.0
    domain_factory_support_hold_prod: float = 4.5
    domain_factory_drain_penalty: float = 0.40
    # --- 4P outer-lane sequence: choose a lane at opening and build a factory
    enable_outer_lane_sequence_4p: bool = False
    outer_lane_turn_limit: int = 190
    outer_lane_claim_turn: int = 55
    outer_lane_anchor_turn: int = 115
    outer_lane_angle_width: float = 1.05
    outer_lane_max_extra_targets: int = 12
    outer_lane_claim_bonus: float = 0.46
    outer_lane_anchor_bonus: float = 0.40
    outer_lane_support_bonus: float = 0.26
    outer_lane_expansion_bonus: float = 0.52
    outer_lane_prod_bonus: float = 0.04
    outer_lane_outer_bonus: float = 0.13
    outer_lane_offlane_penalty: float = 0.10
    outer_lane_hold_base: float = 18.0
    outer_lane_hold_prod: float = 4.2
    outer_lane_drain_penalty: float = 0.16
    outer_lane_budget_waves: int = 1
    outer_lane_project_budget_waves: int = 2
    outer_lane_budget_roi_discount: float = 0.08
    outer_lane_rollout_horizon: int = 60
    outer_lane_rollout_candidates: int = 4
    outer_lane_queue_width: int = 2
    outer_lane_budget_start_turn: int = 55
    # --- 4P sector factory: support-first outer lane selection ---------------
    sector_min_support_before_anchor: int = 2
    sector_support_first_bonus: float = 3.8
    sector_safe_support_bonus: float = 2.2
    sector_anchor_rush_penalty: float = 8.0
    sector_enemy_overlap_penalty: float = 0.32
    sector_support_radius: float = 62.0
    # --- 4P queue executor: add capture-floor sized candidates for lane queue
    enable_outer_lane_queue_executor_4p: bool = True
    outer_lane_executor_start_turn: int = 18
    outer_lane_executor_turn_limit: int = 88
    outer_lane_executor_targets: int = 2
    outer_lane_executor_size_overhead: float = 1.0
    outer_lane_executor_bonus: float = 1.15
    outer_lane_executor_roi_discount: float = 0.34
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
    # --- 4P low-conflict expansion: grow by cheap nearby neutrals ------------
    enable_low_conflict_expansion_4p: bool = False
    low_conflict_start_turn: int = 18
    low_conflict_turn_limit: int = 115
    low_conflict_cheap_bonus: float = 0.42
    low_conflict_prod_bonus: float = 0.055
    low_conflict_friend_bonus: float = 0.20
    low_conflict_enemy_target_penalty: float = 0.55
    low_conflict_contest_penalty: float = 0.26
    low_conflict_drain_penalty: float = 0.18
    low_conflict_nonleader_front_penalty: float = 0.48
    low_conflict_leader_boundary_bonus: float = 0.24
    low_conflict_safe_score_scale: float = 0.010
    low_conflict_unsafe_neutral_penalty: float = 0.85
    # --- 4P sample55 portfolio executor: reserve script slots before greedy ---
    enable_portfolio_executor_4p: bool = False
    portfolio_start_turn: int = 25
    portfolio_turn_limit: int = 120
    portfolio_home_loss_turn: int = 6
    portfolio_home_high_prod_loss_turn: int = 10
    portfolio_home_min_prod: float = 2.5
    portfolio_abandon_loss_turn: int = 5
    portfolio_abandon_high_prod_loss_turn: int = 8
    portfolio_abandon_max_prod: float = 2.5
    portfolio_abandon_max_ship: float = 24.0
    portfolio_abandon_send_frac: float = 0.55
    portfolio_regroup_anchor_count: int = 2
    portfolio_regroup_anchor_min_prod: float = 2.5


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
    cand_eta: Tensor,
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
    eta = cand_eta[:, 0].to(score.dtype).clamp(min=1.0)
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


def _apply_low_conflict_expansion_adjustment(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_low_conflict_expansion_4p):
        return score
    step = _obs_step(obs_tensors)
    if step < int(config.low_conflict_start_turn) or step > int(config.low_conflict_turn_limit):
        return score
    P = int(prod.shape[0])
    if P <= 0:
        return score

    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    active = cand_active.any(dim=-1)

    d0 = cache.cross_dist[0].to(score.dtype)
    owned = obs.owned & obs.alive
    enemy = obs.is_enemy & obs.alive
    if bool(owned.any()):
        dist_from_owned = torch.where(
            owned.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).amin(dim=0)
        friend_count = ((d0 <= 42.0) & owned.view(P, 1)).sum(dim=0).to(score.dtype)
    else:
        dist_from_owned = torch.full((P,), float("inf"), dtype=score.dtype, device=score.device)
        friend_count = torch.zeros(P, dtype=score.dtype, device=score.device)

    if bool(enemy.any()):
        enemy_speed = fleet_speed(obs.ships.to(score.dtype).clamp(min=1e-6))
        eta_enemy_all = torch.where(
            enemy.view(P, 1),
            d0 / enemy_speed.view(P, 1).clamp(min=1e-6),
            torch.full_like(d0, float("inf")),
        )
        eta_from_enemy = eta_enemy_all.amin(dim=0)
    else:
        eta_enemy_all = torch.full((P, P), float("inf"), dtype=score.dtype, device=score.device)
        eta_from_enemy = torch.full((P,), float("inf"), dtype=score.dtype, device=score.device)

    owners = obs.owner_abs.to(torch.long)
    enemy_owner_count = max(int(player_count), int(owners[owners >= 0].max().item()) + 1 if bool((owners >= 0).any()) else 1)
    owner_power = torch.full((enemy_owner_count,), float("-inf"), dtype=score.dtype, device=score.device)
    for owner in range(enemy_owner_count):
        if owner == int(obs.player_id):
            continue
        owned_by = obs.alive & (owners == owner)
        if bool(owned_by.any()):
            owner_power[owner] = prod[owned_by].to(score.dtype).sum() * 14.0 + obs.ships[owned_by].to(score.dtype).sum()
    leader_owner = int(torch.argmax(owner_power).item()) if bool(torch.isfinite(owner_power).any()) else -1
    nearest_enemy_owner = torch.full((P,), -1, dtype=torch.long, device=score.device)
    if bool(enemy.any()):
        nearest_enemy_idx = torch.where(
            enemy.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).argmin(dim=0)
        nearest_enemy_owner = owners[nearest_enemy_idx].to(torch.long)

    neutral = obs.is_neutral[tgt] & active
    support_window = eta + 6.0
    my_eta_to_tgt = torch.where(
        owned.view(P, 1),
        d0[:, tgt],
        torch.full((P, tgt.shape[0]), float("inf"), dtype=score.dtype, device=score.device),
    )
    my_reach_weight = (1.0 - my_eta_to_tgt / support_window.view(1, -1)).clamp(min=0.0)
    my_support = (
        (obs.ships.to(score.dtype).view(P, 1) * my_reach_weight).sum(dim=0) * 0.35
        + (prod.to(score.dtype).view(P, 1) * my_reach_weight).sum(dim=0) * 2.0
    )
    enemy_eta_to_tgt = eta_enemy_all[:, tgt]
    enemy_reach_weight = (1.0 - enemy_eta_to_tgt / support_window.view(1, -1)).clamp(min=0.0)
    enemy_reach = (
        (obs.ships.to(score.dtype).view(P, 1) * enemy_reach_weight).sum(dim=0)
        + (prod.to(score.dtype).view(P, 1) * enemy_reach_weight).sum(dim=0) * 2.5
    )
    post_capture = (send - obs.ships[tgt].to(score.dtype) + 1.0).clamp(min=1.0)
    hold_power = post_capture + prod[tgt].to(score.dtype) * 4.0 + my_support
    safe_neutral = neutral & (hold_power >= enemy_reach * 0.95)
    safe_score_raw = (
        prod[tgt].to(score.dtype) * 42.0
        - send * 2.2
        - eta * 2.0
        + my_support * 0.18
        - enemy_reach * 0.28
    )
    safe_score_bonus = torch.where(
        safe_neutral,
        (safe_score_raw * float(config.low_conflict_safe_score_scale)).clamp(min=-0.40, max=0.45),
        torch.zeros_like(score),
    )
    unsafe_neutral_penalty = torch.where(
        neutral & (~safe_neutral) & (obs.ships[tgt].to(score.dtype) <= 26.0),
        torch.full_like(score, float(config.low_conflict_unsafe_neutral_penalty)),
        torch.zeros_like(score),
    )
    cheap_neutral = neutral & (
        (obs.ships[tgt].to(score.dtype) <= 12.0)
        | ((obs.ships[tgt].to(score.dtype) <= 26.0) & (prod[tgt].to(score.dtype) >= 3.0))
    )
    local_neutral = neutral & (dist_from_owned[tgt] <= 46.0)
    friend_bonus = friend_count[tgt].clamp(max=3.0) * float(config.low_conflict_friend_bonus)
    neutral_bonus = torch.where(
        cheap_neutral | local_neutral,
        torch.full_like(score, float(config.low_conflict_cheap_bonus))
        + (prod[tgt].to(score.dtype) * float(config.low_conflict_prod_bonus)).clamp(max=0.28)
        + friend_bonus,
        torch.zeros_like(score),
    )

    enemy_target_penalty = torch.where(
        active & obs.is_enemy[tgt],
        torch.full_like(score, float(config.low_conflict_enemy_target_penalty)),
        torch.zeros_like(score),
    )
    contested_neutral = (
        neutral
        & (eta_from_enemy[tgt] <= 10.0)
        & (dist_from_owned[tgt] > 32.0)
        & (obs.ships[tgt].to(score.dtype) > 12.0)
    )
    contest_penalty = torch.where(
        contested_neutral,
        torch.full_like(score, float(config.low_conflict_contest_penalty)),
        torch.zeros_like(score),
    )
    nonleader_front = (
        active
        & (nearest_enemy_owner[tgt] >= 0)
        & (nearest_enemy_owner[tgt] != int(leader_owner))
        & (eta_from_enemy[tgt] <= 16.0)
        & (obs.ships[tgt].to(score.dtype) > 12.0)
    )
    nonleader_penalty = torch.where(
        nonleader_front,
        torch.full_like(score, float(config.low_conflict_nonleader_front_penalty)),
        torch.zeros_like(score),
    )
    leader_boundary = (
        neutral
        & (nearest_enemy_owner[tgt] == int(leader_owner))
        & (eta_from_enemy[tgt] <= 18.0)
        & (dist_from_owned[tgt] <= 54.0)
        & (obs.ships[tgt].to(score.dtype) <= 30.0)
    )
    leader_bonus = torch.where(
        leader_boundary,
        torch.full_like(score, float(config.low_conflict_leader_boundary_bonus)),
        torch.zeros_like(score),
    )

    src_prod = prod[src].to(score.dtype)
    hold = torch.where(
        src_prod >= 3.0,
        torch.full_like(score, 18.0) + src_prod * 4.0,
        torch.full_like(score, 10.0) + src_prod * 3.0,
    )
    after = obs.ships[src].to(score.dtype) - send
    drain_penalty = ((hold - after).clamp(min=0.0) / 32.0).clamp(max=1.4) * float(config.low_conflict_drain_penalty)
    drain_penalty = torch.where(active & owned[src], drain_penalty, torch.zeros_like(score))

    return score + neutral_bonus + safe_score_bonus + leader_bonus - enemy_target_penalty - contest_penalty - nonleader_penalty - unsafe_neutral_penalty - drain_penalty


def _low_conflict_reserved_safe_mask(
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
    config: ProducerLiteConfig,
    player_count: int,
    current_step: int,
) -> tuple[Tensor, Tensor]:
    if (
        int(player_count) < 4
        or not bool(config.enable_low_conflict_expansion_4p)
        or int(current_step) < 25
        or int(current_step) > min(120, int(config.low_conflict_turn_limit))
    ):
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))
    P = int(prod.shape[0])
    if P <= 0:
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))

    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    eta = cand_eta[:, 0].to(score.dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1)
    neutral = active & obs.is_neutral[tgt]
    if not bool(neutral.any()):
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))

    d0 = cache.cross_dist[0].to(score.dtype)
    owned = obs.owned & obs.alive
    enemy = obs.is_enemy & obs.alive
    support_window = eta + 6.0
    my_eta = torch.where(
        owned.view(P, 1),
        d0[:, tgt],
        torch.full((P, tgt.shape[0]), float("inf"), dtype=score.dtype, device=score.device),
    )
    my_w = (1.0 - my_eta / support_window.view(1, -1)).clamp(min=0.0)
    my_support = (
        (obs.ships.to(score.dtype).view(P, 1) * my_w).sum(dim=0) * 0.35
        + (prod.to(score.dtype).view(P, 1) * my_w).sum(dim=0) * 2.0
    )
    if bool(enemy.any()):
        enemy_speed = fleet_speed(obs.ships.to(score.dtype).clamp(min=1e-6))
        enemy_eta = torch.where(
            enemy.view(P, 1),
            d0[:, tgt] / enemy_speed.view(P, 1).clamp(min=1e-6),
            torch.full((P, tgt.shape[0]), float("inf"), dtype=score.dtype, device=score.device),
        )
        enemy_w = (1.0 - enemy_eta / support_window.view(1, -1)).clamp(min=0.0)
        enemy_reach = (
            (obs.ships.to(score.dtype).view(P, 1) * enemy_w).sum(dim=0)
            + (prod.to(score.dtype).view(P, 1) * enemy_w).sum(dim=0) * 2.5
        )
        enemy_reach_count = ((enemy_eta <= support_window.view(1, -1)) & enemy.view(P, 1)).sum(dim=0)
    else:
        enemy_reach = torch.zeros_like(score)
        enemy_reach_count = torch.zeros_like(tgt)

    post_capture = (send - obs.ships[tgt].to(score.dtype) + 1.0).clamp(min=1.0)
    hold_power = post_capture + prod[tgt].to(score.dtype) * 4.0 + my_support
    radius = torch.sqrt((obs.x.to(score.dtype) - 50.0) ** 2 + (obs.y.to(score.dtype) - 50.0) ** 2)
    central_brawl = (radius[tgt] < 28.0) & (enemy_reach_count >= 2) & (my_support < enemy_reach)
    cheap_or_good = (obs.ships[tgt].to(score.dtype) <= 26.0) | (prod[tgt].to(score.dtype) >= 3.0)
    safe = neutral & cheap_or_good & (~central_brawl) & (hold_power >= enemy_reach * 0.82)
    reserve_score = (
        prod[tgt].to(score.dtype) * 42.0
        - send * 2.2
        - eta * 2.0
        + my_support * 0.18
        - enemy_reach * 0.28
        + torch.where(radius[tgt] >= 30.0, torch.full_like(score, 4.0), torch.zeros_like(score))
    )
    reserve_score = reserve_score * 0.03
    return safe, torch.where(safe, reserve_score, torch.full_like(score, float("-inf")))


@dataclass(frozen=True)
class MidgameFeatures:
    step: int
    player_count: int
    leader_id: int
    my_power: float
    leader_power: float
    my_production: float
    leader_production: float
    my_planet_count: int
    cheap_safe_neutral_count: int
    losing_front_count: int


def _portfolio_enabled(config: ProducerLiteConfig, *, player_count: int, current_step: int) -> bool:
    return (
        int(player_count) >= 4
        and bool(config.enable_portfolio_executor_4p)
        and int(current_step) >= int(config.portfolio_start_turn)
        and int(current_step) <= int(config.portfolio_turn_limit)
    )


def _portfolio_owner_power(obs, prod: Tensor, *, player_count: int) -> Tensor:
    owners = obs.owner_abs.to(torch.long)
    owner_count = max(int(player_count), int(owners[owners >= 0].max().item()) + 1 if bool((owners >= 0).any()) else 1)
    power = torch.zeros(owner_count, dtype=prod.dtype, device=prod.device)
    for owner in range(owner_count):
        mask = obs.alive & (owners == int(owner))
        if bool(mask.any()):
            power[owner] = obs.ships[mask].to(prod.dtype).sum() + prod[mask].to(prod.dtype).sum() * 12.0
    return power


def _portfolio_loss_turns(garrison_status, *, player_id: int, horizon: int, P: int, device, dtype) -> Tensor:
    limit = max(1, min(int(horizon), int(garrison_status.owner.shape[-1]) - 1))
    if limit <= 0:
        return torch.full((P,), float("inf"), dtype=dtype, device=device)
    future_owner = garrison_status.owner[:, 1: limit + 1].to(torch.long)
    lost = future_owner != int(player_id)
    steps = torch.arange(1, limit + 1, dtype=dtype, device=device).view(1, limit).expand(P, limit)
    first = torch.where(lost, steps, torch.full_like(steps, float("inf"))).min(dim=1).values
    return first


def _portfolio_features(
    *,
    obs,
    prod: Tensor,
    cache,
    garrison_status,
    config: ProducerLiteConfig,
    player_count: int,
    current_step: int,
) -> MidgameFeatures:
    P = int(obs.P)
    device = obs.device
    dtype = prod.dtype
    pid = int(obs.player_id)
    power = _portfolio_owner_power(obs, prod, player_count=player_count)
    if 0 <= pid < int(power.shape[0]):
        my_power = float(power[pid].item())
        power_for_leader = power.clone()
        power_for_leader[pid] = float("-inf")
    else:
        my_power = 0.0
        power_for_leader = power
    leader_id = int(torch.argmax(power_for_leader).item()) if bool(torch.isfinite(power_for_leader).any()) else -1
    leader_power = float(power[leader_id].item()) if 0 <= leader_id < int(power.shape[0]) else 0.0

    owner = obs.owner_abs.to(torch.long)
    mine = obs.owned & obs.alive
    leader_mask = obs.alive & (owner == int(leader_id)) if leader_id >= 0 else torch.zeros(P, dtype=torch.bool, device=device)
    my_prod = float(prod[mine].sum().item()) if bool(mine.any()) else 0.0
    leader_prod = float(prod[leader_mask].sum().item()) if bool(leader_mask.any()) else 0.0

    d0 = cache.cross_dist[0].to(dtype)
    neutral = obs.is_neutral & obs.alive
    cheap_neutral = neutral & ((obs.ships.to(dtype) <= 26.0) | (prod.to(dtype) >= 3.0))
    if bool(mine.any()) and bool(cheap_neutral.any()):
        my_dist = torch.where(mine.view(P, 1), d0, torch.full((P, P), float("inf"), dtype=dtype, device=device)).min(dim=0).values
        enemy = obs.is_enemy & obs.alive
        enemy_dist = torch.where(enemy.view(P, 1), d0, torch.full((P, P), float("inf"), dtype=dtype, device=device)).min(dim=0).values
        safe_neutral = cheap_neutral & (my_dist <= 22.0) & ((enemy_dist - my_dist) >= -3.0)
        cheap_safe_count = int(safe_neutral.sum().item())
    else:
        cheap_safe_count = 0

    loss_turn = _portfolio_loss_turns(
        garrison_status, player_id=pid, horizon=int(config.portfolio_abandon_high_prod_loss_turn),
        P=P, device=device, dtype=dtype,
    )
    losing_front = mine & torch.isfinite(loss_turn) & (loss_turn <= float(config.portfolio_abandon_high_prod_loss_turn))
    return MidgameFeatures(
        step=int(current_step),
        player_count=int(player_count),
        leader_id=int(leader_id),
        my_power=my_power,
        leader_power=leader_power,
        my_production=my_prod,
        leader_production=leader_prod,
        my_planet_count=int(mine.sum().item()),
        cheap_safe_neutral_count=cheap_safe_count,
        losing_front_count=int(losing_front.sum().item()),
    )


def _append_portfolio_regroup_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    obs,
    prod: Tensor,
    garrison_status,
    config: ProducerLiteConfig,
    player_count: int,
    current_step: int,
) -> tuple[Tensor, Tensor]:
    if not _portfolio_enabled(config, player_count=player_count, current_step=current_step):
        return target_idx, target_exists
    P = int(obs.P)
    device = obs.device
    dtype = prod.dtype
    if P <= 0:
        return target_idx, target_exists
    loss_turn = _portfolio_loss_turns(
        garrison_status, player_id=int(obs.player_id), horizon=int(config.portfolio_home_high_prod_loss_turn),
        P=P, device=device, dtype=dtype,
    )
    mine = obs.owned & obs.alive
    safe_anchor = (
        mine
        & (prod.to(dtype) >= float(config.portfolio_regroup_anchor_min_prod))
        & ((~torch.isfinite(loss_turn)) | (loss_turn > float(config.portfolio_home_high_prod_loss_turn)))
    )
    if not bool(safe_anchor.any()):
        return target_idx, target_exists
    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)
    scores = prod.to(dtype) * 25.0 + obs.ships.to(dtype) - torch.sqrt((obs.x.to(dtype) - 50.0) ** 2 + (obs.y.to(dtype) - 50.0) ** 2) * 0.12
    scores = torch.where(safe_anchor & (~existing), scores, torch.full_like(scores, float("-inf")))
    extras: list[int] = []
    for _ in range(max(0, int(config.portfolio_regroup_anchor_count))):
        best = int(torch.argmax(scores).item())
        if not bool(torch.isfinite(scores[best])):
            break
        extras.append(best)
        scores[best] = float("-inf")
    if not extras:
        return target_idx, target_exists
    extra_idx = torch.tensor(extras, dtype=target_idx.dtype, device=target_idx.device)
    extra_exists = torch.ones(extra_idx.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extra_idx], dim=0), torch.cat([target_exists, extra_exists], dim=0)


def _portfolio_home_defense_mask(
    *,
    score: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    cand_is_def: Tensor,
    obs,
    prod: Tensor,
    garrison_status,
    config: ProducerLiteConfig,
    features: MidgameFeatures,
) -> tuple[Tensor, Tensor]:
    P = int(obs.P)
    device = score.device
    dtype = score.dtype
    loss_turn = _portfolio_loss_turns(
        garrison_status, player_id=int(obs.player_id), horizon=int(config.portfolio_home_high_prod_loss_turn),
        P=P, device=device, dtype=dtype,
    )
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1)
    target_prod = prod[tgt].to(dtype)
    urgent = loss_turn[tgt] <= float(config.portfolio_home_loss_turn)
    high_prod_urgent = (target_prod >= float(config.portfolio_home_min_prod)) & (loss_turn[tgt] <= float(config.portfolio_home_high_prod_loss_turn))
    mask = active & cand_is_def & obs.owned[tgt] & (urgent | high_prod_urgent)
    reserve_score = (
        1.0
        + target_prod * 0.22
        + (float(config.portfolio_home_high_prod_loss_turn) - loss_turn[tgt]).clamp(min=0.0) * 0.09
        - send * 0.010
        - eta * 0.015
        + torch.where(
            torch.tensor(features.leader_power > features.my_power + 30.0, dtype=torch.bool, device=device),
            torch.full_like(score, 0.03),
            torch.zeros_like(score),
        )
    )
    return mask, torch.where(mask, reserve_score, torch.full_like(score, float("-inf")))


def _portfolio_abandon_regroup_mask(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    cand_is_def: Tensor,
    obs,
    prod: Tensor,
    garrison_status,
    config: ProducerLiteConfig,
) -> tuple[Tensor, Tensor]:
    P = int(obs.P)
    device = score.device
    dtype = score.dtype
    loss_turn = _portfolio_loss_turns(
        garrison_status, player_id=int(obs.player_id), horizon=int(config.portfolio_abandon_high_prod_loss_turn),
        P=P, device=device, dtype=dtype,
    )
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1)
    low_value_doomed = (
        obs.owned[src]
        & torch.isfinite(loss_turn[src])
        & (
            (loss_turn[src] <= float(config.portfolio_abandon_loss_turn))
            | ((prod[src].to(dtype) <= float(config.portfolio_abandon_max_prod)) & (loss_turn[src] <= float(config.portfolio_abandon_high_prod_loss_turn)))
        )
        & (obs.ships[src].to(dtype) <= float(config.portfolio_abandon_max_ship))
    )
    safe_dest = obs.owned[tgt] & (
        (~torch.isfinite(loss_turn[tgt]))
        | (loss_turn[tgt] > float(config.portfolio_home_high_prod_loss_turn))
    )
    max_send = (obs.ships[src].to(dtype) * float(config.portfolio_abandon_send_frac)).clamp(min=float(config.min_ships_to_launch))
    mask = active & cand_is_def & low_value_doomed & safe_dest & (src != tgt) & (send <= max_send)
    reserve_score = (
        0.65
        + prod[tgt].to(dtype) * 0.10
        + obs.ships[src].to(dtype).clamp(max=30.0) * 0.006
        - prod[src].to(dtype) * 0.08
        - eta * 0.018
        - send * 0.006
    )
    return mask, torch.where(mask, reserve_score, torch.full_like(score, float("-inf")))


def _select_one_portfolio_slot(
    *,
    P: int,
    device,
    dtype,
    slot_score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_angle: Tensor,
    cand_eta: Tensor,
    cand_active: Tensor,
    cand_tgt_slot: Tensor,
    cand_tgt_short: Tensor,
    cand_is_def: Tensor,
    source_budget: Tensor,
    target_exists: Tensor,
    target_idx: Tensor,
    target_exists_for_normal: Tensor,
    roi_threshold: float,
) -> tuple:
    entries, source_budget = _greedy_select(
        P=P, W=1, device=device, dtype=dtype, score=slot_score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists,
        roi_threshold=float(roi_threshold),
    )
    if bool(entries.valid.any()):
        taken = entries.target_slots[entries.valid].clamp(0, max(P - 1, 0))
        if taken.numel() > 0:
            taken_mask = torch.zeros(P, dtype=torch.bool, device=device)
            taken_mask.scatter_(0, taken, True)
            target_exists_for_normal = target_exists_for_normal & (~taken_mask[target_idx.clamp(0, P - 1)])
    return entries, source_budget, target_exists_for_normal


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


def _make_domain_factory_plan(
    *,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    preferred_plan: dict | None = None,
) -> dict | None:
    """Choose an outer production domain: anchor, supports, connectors, expansion."""
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
    enemy_idx = torch.nonzero(
        obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(obs.player_id)),
        as_tuple=False,
    ).flatten().tolist()
    if not neutral_idx:
        return None

    preferred_anchor = -1
    preferred_targets: list[int] = []
    if preferred_plan:
        preferred_anchor = int(preferred_plan.get("anchor", -1))
        preferred_targets = [int(t) for t in preferred_plan.get("targets", [])]

    anchor_scan = [preferred_anchor] if preferred_anchor in neutral_idx else neutral_idx
    best: tuple[float, int, list[int], list[int], list[int], float] | None = None
    for anchor in anchor_scan:
        ax = float(x[anchor].item())
        ay = float(y[anchor].item())
        ar = float(radius[anchor].item())
        ap = float(prod[anchor].item())
        ships = float(obs.ships[anchor].item())
        dist_from_base = _dist_xy(bx, by, ax, ay)
        lane_angle = float(angle[anchor].item())
        if not (28.0 <= ar <= 72.0):
            continue
        if dist_from_base > 82.0:
            continue
        if ap < 3.0 and ships < 38.0:
            continue
        if _angle_diff(base_angle, lane_angle) > 1.45:
            continue

        nearest_enemy = min(
            (
                _dist_xy(float(x[e].item()), float(y[e].item()), ax, ay)
                for e in enemy_idx
            ),
            default=999.0,
        )
        contest_penalty = max(0.0, dist_from_base - nearest_enemy + 8.0) * 0.12

        connectors: list[int] = []
        supports: list[int] = []
        expansions: list[int] = []
        for j in neutral_idx:
            if j == anchor:
                continue
            jx = float(x[j].item())
            jy = float(y[j].item())
            jr = float(radius[j].item())
            jp = float(prod[j].item())
            js = float(obs.ships[j].item())
            ja = float(angle[j].item())
            lane_delta = _angle_diff(lane_angle, ja)
            d_anchor = _dist_xy(jx, jy, ax, ay)
            d_base = _dist_xy(jx, jy, bx, by)
            if lane_delta <= float(config.domain_factory_angle_width):
                if 12.0 <= d_base <= dist_from_base + 8.0 and 12.0 <= d_anchor <= 58.0:
                    if jp >= 2.0 or 12.0 <= js <= 40.0:
                        connectors.append(j)
                if d_anchor <= 55.0 and (jp >= 2.0 or 14.0 <= js <= 46.0):
                    supports.append(j)
            if lane_delta <= float(config.domain_factory_angle_width) + 0.28:
                if 24.0 <= d_anchor <= 88.0 and jr >= ar - 5.0 and (jp >= 3.0 or js >= 45.0):
                    expansions.append(j)

        support_prod = sum(float(prod[j].item()) for j in supports)
        connector_value = sum(
            float(prod[j].item()) * 2.1 + max(0.0, 34.0 - float(obs.ships[j].item())) * 0.045
            for j in connectors
        )
        expansion_prod = sum(float(prod[j].item()) for j in expansions)
        expansion_big = sum(1 for j in expansions if float(prod[j].item()) >= 3.0 or float(obs.ships[j].item()) >= 54.0)
        score = (
            ap * 12.5
            + ships * 0.07
            + ar * 0.08
            + support_prod * 2.35
            + len(supports) * 1.45
            + connector_value
            + expansion_prod * 1.15
            + expansion_big * 2.8
            - dist_from_base * 0.20
            - contest_penalty
        )
        if best is None or score > best[0]:
            best = (score, anchor, connectors, supports, expansions, lane_angle)

    if best is None or best[0] < (24.0 if preferred_anchor in neutral_idx else 32.0):
        return None

    score, anchor, connectors, supports, expansions, lane_angle = best
    connectors_sorted = sorted(
        connectors,
        key=lambda j: _dist_xy(bx, by, float(x[j].item()), float(y[j].item())),
    )[:5]
    supports_sorted = sorted(
        supports,
        key=lambda j: float(prod[j].item()) * 10.0 + float(obs.ships[j].item()) * 0.12,
        reverse=True,
    )[:7]
    expansions_sorted = sorted(
        expansions,
        key=lambda j: float(prod[j].item()) * 11.0 + float(obs.ships[j].item()) * 0.08 + float(radius[j].item()) * 0.04,
        reverse=True,
    )[:6]

    core = [int(anchor)]
    core.extend(int(t) for t in preferred_targets if int(t) != int(anchor))
    core.extend(int(t) for t in connectors_sorted)
    core.extend(int(t) for t in supports_sorted)
    seen: set[int] = set()
    unique_core: list[int] = []
    for t in core:
        if t not in seen:
            unique_core.append(t)
            seen.add(t)
    unique_expansion: list[int] = []
    for t in expansions_sorted:
        if t not in seen:
            unique_expansion.append(int(t))
            seen.add(int(t))

    return {
        "mode": "domain_factory",
        "anchor": int(anchor),
        "core": unique_core,
        "support": [int(t) for t in supports_sorted],
        "connector": [int(t) for t in connectors_sorted],
        "expansion": unique_expansion,
        "targets": unique_core + unique_expansion,
        "angle": float(lane_angle),
        "created_step": _obs_step(obs_tensors),
        "score": float(score),
    }


def _domain_factory_mature(
    *,
    obs,
    prod: Tensor,
    config: ProducerLiteConfig,
    plan: dict | None,
    step: int,
) -> bool:
    if not plan:
        return False
    P = int(prod.shape[0])
    anchor = int(plan.get("anchor", -1))
    if 0 <= anchor < P and bool(obs.owned[anchor].item()):
        hold = float(config.domain_factory_hold_base) + float(prod[anchor].item()) * float(config.domain_factory_hold_prod)
        if float(obs.ships[anchor].item()) >= hold * 0.78:
            return True

    core = [int(t) for t in plan.get("core", []) if 0 <= int(t) < P]
    owned_core = [t for t in core if bool(obs.owned[t].item())]
    if not owned_core:
        return False
    total_prod = sum(float(prod[t].item()) for t in owned_core)
    total_ships = sum(float(obs.ships[t].item()) for t in owned_core)
    if total_ships >= 70.0 + total_prod * 8.0:
        return True
    return int(step) >= int(config.domain_factory_project_turn) + 30 and total_ships >= 45.0 + total_prod * 5.5


def _append_domain_factory_targets(
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
    preferred_plan: dict | None = None,
) -> tuple[Tensor, Tensor, dict | None]:
    if int(player_count) < 4 or not bool(config.enable_domain_factory_4p):
        return target_idx, target_exists, None
    step = _obs_step(obs_tensors)
    if step > int(config.domain_factory_turn_limit):
        return target_idx, target_exists, getattr(memory, "domain_plan", None)
    if getattr(memory, "domain_plan", None) is None or step <= 1:
        memory.domain_plan = _make_domain_factory_plan(
            obs=obs,
            obs_tensors=obs_tensors,
            prod=prod,
            source_mask=source_mask,
            config=config,
            preferred_plan=preferred_plan,
        )
    plan = getattr(memory, "domain_plan", None)
    if not plan:
        return target_idx, target_exists, None
    if step < 62:
        return target_idx, target_exists, None

    mature = _domain_factory_mature(obs=obs, prod=prod, config=config, plan=plan, step=step)
    core_targets = [int(t) for t in plan.get("core", [])]
    expansion_targets = [int(t) for t in plan.get("expansion", [])]
    if step < 55:
        plan_targets = core_targets
    elif mature or step >= int(config.domain_factory_project_turn):
        plan_targets = core_targets + expansion_targets
    else:
        plan_targets = core_targets

    P = int(obs.P)
    device = obs.device
    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)

    extras: list[int] = []
    for t in plan_targets[: int(config.domain_factory_max_extra_targets)]:
        if 0 <= int(t) < P and bool(obs.alive[int(t)].item()) and not bool(existing[int(t)].item()):
            extras.append(int(t))
    if not extras:
        return target_idx, target_exists, plan
    extra_idx = torch.tensor(extras, dtype=target_idx.dtype, device=target_idx.device)
    extra_exists = torch.ones(extra_idx.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extra_idx], dim=0), torch.cat([target_exists, extra_exists], dim=0), plan


def _apply_domain_factory_adjustment(
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
    domain_plan: dict | None,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_domain_factory_4p) or not domain_plan:
        return score
    step = _obs_step(obs_tensors)
    if step > int(config.domain_factory_turn_limit):
        return score
    if step < 62:
        return score

    P = int(prod.shape[0])
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    active = cand_active.any(dim=-1)

    core_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    support_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    expansion_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    for t in domain_plan.get("core", []):
        if 0 <= int(t) < P:
            core_mask[int(t)] = True
    for t in domain_plan.get("support", []):
        if 0 <= int(t) < P:
            support_mask[int(t)] = True
    for t in domain_plan.get("expansion", []):
        if 0 <= int(t) < P:
            expansion_mask[int(t)] = True

    project_ok = _domain_factory_mature(obs=obs, prod=prod, config=config, plan=domain_plan, step=step) or step >= int(config.domain_factory_project_turn)
    planets = obs_tensors["planets"].to(score.dtype)
    dx = planets[:, 2] - 50.0
    dy = planets[:, 3] - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)
    angle = torch.atan2(dy, dx)
    domain_angle = float(domain_plan.get("angle", 0.0))
    angle_delta = torch.abs((angle - domain_angle + math.pi) % (2.0 * math.pi) - math.pi)

    selected_core = core_mask[tgt] & active
    selected_support = support_mask[tgt] & active
    selected_expansion = expansion_mask[tgt] & active & bool(project_ok)
    target_bonus = torch.zeros_like(score)
    target_bonus = target_bonus + torch.where(
        selected_core,
        torch.full_like(score, float(config.domain_factory_core_bonus)),
        torch.zeros_like(score),
    )
    target_bonus = target_bonus + torch.where(
        selected_support,
        torch.full_like(score, float(config.domain_factory_support_bonus)),
        torch.zeros_like(score),
    )
    target_bonus = target_bonus + torch.where(
        selected_expansion,
        torch.full_like(score, float(config.domain_factory_expansion_bonus)),
        torch.zeros_like(score),
    )
    selected_any = selected_core | selected_support | selected_expansion
    target_bonus = target_bonus + torch.where(
        selected_any,
        (prod[tgt].to(score.dtype) * float(config.domain_factory_prod_bonus)).clamp(max=0.42)
        + torch.where(radius[tgt] >= 32.0, torch.full_like(score, float(config.domain_factory_outer_bonus)), torch.zeros_like(score)),
        torch.zeros_like(score),
    )

    domain_mask = core_mask | support_mask | expansion_mask
    offdomain_cheap = (
        active
        & obs.is_neutral[tgt]
        & (~domain_mask[tgt])
        & (angle_delta[tgt] > float(config.domain_factory_angle_width))
        & (prod[tgt] <= 2.0)
        & (obs.ships[tgt].to(score.dtype) <= 30.0)
        & (step < int(config.domain_factory_project_turn))
    )
    offdomain_penalty = torch.where(
        offdomain_cheap,
        torch.full_like(score, float(config.domain_factory_offdomain_cheap_penalty)),
        torch.zeros_like(score),
    )

    src_domain = (core_mask[src] | support_mask[src]) & obs.owned[src] & active
    src_prod = prod[src].to(score.dtype)
    src_ships = obs.ships[src].to(score.dtype)
    anchor = int(domain_plan.get("anchor", -1))
    anchor_src = src == anchor
    hold = torch.where(
        anchor_src,
        torch.full_like(score, float(config.domain_factory_hold_base)) + src_prod * float(config.domain_factory_hold_prod),
        torch.full_like(score, float(config.domain_factory_support_hold_base)) + src_prod * float(config.domain_factory_support_hold_prod),
    )
    after = src_ships - send
    hold_short = (hold - after).clamp(min=0.0)
    target_in_domain = domain_mask[tgt]
    relief = torch.where(target_in_domain, torch.full_like(score, 0.45), torch.ones_like(score))
    if project_ok:
        relief = torch.where(expansion_mask[tgt], torch.full_like(score, 0.55), relief)
    drain_penalty = (
        (hold_short / 30.0).clamp(max=1.7)
        * float(config.domain_factory_drain_penalty)
        * relief
    )
    drain_penalty = torch.where(src_domain, drain_penalty, torch.zeros_like(score))
    return score + target_bonus - offdomain_penalty - drain_penalty


def _outer_lane_phase(step: int, config: ProducerLiteConfig) -> str:
    if int(step) < int(config.outer_lane_claim_turn):
        return "claim"
    if int(step) < int(config.outer_lane_anchor_turn):
        return "anchor"
    if int(step) <= int(config.outer_lane_turn_limit):
        return "project"
    return "free"


def _outer_lane_rollout_score(
    *,
    obs,
    prod: Tensor,
    x: Tensor,
    y: Tensor,
    radius: Tensor,
    base: int,
    anchor: int,
    claim: list[int],
    support: list[int],
    expansion: list[int],
    enemy_idx: list[int],
    base_score: float,
    horizon: int,
) -> tuple[float, list[int]]:
    """Cheap deterministic rollout of a lane action sequence.

    This is not full physics. It estimates whether the first few lane captures
    create a production factory quickly enough before enemies contest the lane.
    The exact scorer still decides real launch sizes later.
    """
    bx = float(x[base].item())
    by = float(y[base].item())
    ax = float(x[anchor].item())
    ay = float(y[anchor].item())
    available = max(8.0, float(obs.ships[base].item()) * 0.92)
    prod_income = max(0.5, float(prod[base].item()))
    time = 0.0
    owned_prod = float(prod[base].item())
    owned_ships = float(obs.ships[base].item())
    score = float(base_score)
    queue: list[int] = []

    def travel(src: int, tgt: int, ships: float) -> float:
        sx = float(x[src].item())
        sy = float(y[src].item())
        tx = float(x[tgt].item())
        ty = float(y[tgt].item())
        speed = max(0.18, float(fleet_speed(torch.tensor([max(1.0, ships)], dtype=obs.ships.dtype, device=obs.device))[0].item()))
        return _dist_xy(sx, sy, tx, ty) / speed

    def enemy_eta(tgt: int) -> float:
        tx = float(x[tgt].item())
        ty = float(y[tgt].item())
        best = 999.0
        for e in enemy_idx:
            ships = max(1.0, float(obs.ships[e].item()))
            speed = max(0.18, float(fleet_speed(torch.tensor([ships], dtype=obs.ships.dtype, device=obs.device))[0].item()))
            best = min(best, _dist_xy(float(x[e].item()), float(y[e].item()), tx, ty) / speed)
        return best

    seq: list[int] = []
    seen: set[int] = set()
    for t in claim + support + [anchor] + expansion:
        ti = int(t)
        if ti == base or ti in seen:
            continue
        seen.add(ti)
        seq.append(ti)

    current_src = base
    for rank, tgt in enumerate(seq[:8]):
        need = max(3.0, float(obs.ships[tgt].item()) + 2.0)
        launch_size = min(max(need, available * 0.52), available)
        eta = travel(current_src, tgt, launch_size)
        arrive = time + eta
        if arrive > float(horizon):
            score -= 2.0 + 0.04 * arrive
            continue
        e_eta = enemy_eta(tgt)
        contest = max(0.0, arrive - e_eta + 1.5)
        available = max(0.0, available - launch_size)
        # Between captures, our lane factory is producing. This is the part the
        # one-turn scorer could not see: early prod compounds into later force.
        available += prod_income * max(0.0, eta * 0.42)
        time = arrive
        p = float(prod[tgt].item())
        s = float(obs.ships[tgt].item())
        r = float(radius[tgt].item())
        owned_prod += p
        owned_ships += max(0.0, launch_size - s)
        prod_income += p
        current_src = tgt
        queue.append(tgt)

        role = 1.0
        if tgt == anchor:
            role = 1.35
        elif tgt in expansion:
            role = 1.25
        elif tgt in support:
            role = 1.15
        quick = max(0.0, float(horizon) - arrive) / float(max(1, horizon))
        score += role * (p * 4.8 + max(0.0, 46.0 - s) * 0.05 + r * 0.035) * (0.45 + quick)
        score -= contest * 1.45
        score -= rank * 0.18

    # Prefer lanes that become thick enough to launch from, not just lanes that
    # collect many tiny planets.
    score += owned_prod * 2.2
    score += min(owned_ships, 160.0) * 0.035
    score += max(0.0, float(radius[anchor].item()) - 24.0) * 0.035
    score -= max(0.0, _dist_xy(bx, by, ax, ay) - 62.0) * 0.09
    return score, queue


def _make_outer_lane_sequence_plan(
    *,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
) -> dict | None:
    """Pick one outer lane and keep pursuing it as an opening/midgame script."""
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
    enemy_idx = torch.nonzero(
        obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(obs.player_id)),
        as_tuple=False,
    ).flatten().tolist()
    if not neutral_idx:
        return None

    plans: list[tuple[float, float, int, list[int], list[int], list[int], list[int], float, list[int]]] = []
    for anchor in neutral_idx:
        ax = float(x[anchor].item())
        ay = float(y[anchor].item())
        ar = float(radius[anchor].item())
        ap = float(prod[anchor].item())
        ships = float(obs.ships[anchor].item())
        lane_angle = float(angle[anchor].item())
        dist_base = _dist_xy(bx, by, ax, ay)
        if dist_base > 92.0:
            continue
        if ar < 24.0:
            continue
        if ap < 2.0 and ships > 42.0:
            continue
        if _angle_diff(base_angle, lane_angle) > 1.70:
            continue

        nearest_enemy = min(
            (
                _dist_xy(float(x[e].item()), float(y[e].item()), ax, ay)
                for e in enemy_idx
            ),
            default=999.0,
        )
        contest_penalty = max(0.0, dist_base - nearest_enemy + 6.0) * 0.16

        claim: list[int] = []
        support: list[int] = []
        connector: list[int] = []
        expansion: list[int] = []
        for j in neutral_idx:
            if j == anchor:
                continue
            jx = float(x[j].item())
            jy = float(y[j].item())
            jr = float(radius[j].item())
            jp = float(prod[j].item())
            js = float(obs.ships[j].item())
            ja = float(angle[j].item())
            lane_delta = _angle_diff(lane_angle, ja)
            d_base = _dist_xy(bx, by, jx, jy)
            d_anchor = _dist_xy(ax, ay, jx, jy)
            if lane_delta <= float(config.outer_lane_angle_width) + 0.22:
                if d_base <= dist_base + 12.0 and d_anchor <= 62.0 and (jp >= 1.0 or js <= 34.0):
                    claim.append(j)
                if 10.0 <= d_anchor <= 60.0 and (jp >= 2.0 or 12.0 <= js <= 45.0):
                    support.append(j)
                if 10.0 <= d_base <= dist_base + 18.0 and d_anchor <= 72.0 and js <= 38.0:
                    connector.append(j)
            if lane_delta <= float(config.outer_lane_angle_width) + 0.45:
                if 26.0 <= d_anchor <= 98.0 and jr >= ar - 7.0 and (jp >= 3.0 or js >= 45.0):
                    expansion.append(j)

        support_before_anchor = [
            j for j in support
            if _dist_xy(bx, by, float(x[j].item()), float(y[j].item())) <= dist_base + 10.0
            and _dist_xy(ax, ay, float(x[j].item()), float(y[j].item())) <= float(config.sector_support_radius)
            and (float(obs.ships[j].item()) <= 46.0 or float(prod[j].item()) >= 2.0)
        ]
        cheap_support = [
            j for j in support_before_anchor
            if float(obs.ships[j].item()) <= 36.0 and float(prod[j].item()) >= 1.0
        ]
        support_enemy_margin = 0.0
        for j in support_before_anchor:
            jx = float(x[j].item())
            jy = float(y[j].item())
            my_d = _dist_xy(bx, by, jx, jy)
            enemy_d = min(
                (_dist_xy(float(x[e].item()), float(y[e].item()), jx, jy) for e in enemy_idx),
                default=999.0,
            )
            support_enemy_margin += max(-8.0, min(12.0, enemy_d - my_d))

        claim_prod = sum(float(prod[j].item()) for j in claim)
        support_prod = sum(float(prod[j].item()) for j in support_before_anchor)
        expansion_prod = sum(float(prod[j].item()) for j in expansion)
        cheap_claim = sum(1 for j in claim if float(obs.ships[j].item()) <= 34.0)
        big_expansion = sum(1 for j in expansion if float(prod[j].item()) >= 3.0 or float(obs.ships[j].item()) >= 50.0)
        heavy_anchor = ships >= 48.0 or ap >= 4.0 or ar >= 45.0
        support_count = len(support_before_anchor)
        support_first_bonus = (
            support_count * float(config.sector_support_first_bonus)
            + len(cheap_support) * float(config.sector_safe_support_bonus)
            + max(0.0, support_enemy_margin) * 0.12
        )
        anchor_rush_penalty = 0.0
        if heavy_anchor and support_count < int(config.sector_min_support_before_anchor):
            anchor_rush_penalty = float(config.sector_anchor_rush_penalty) * (int(config.sector_min_support_before_anchor) - support_count + 1)
        enemy_overlap_penalty = max(0.0, 12.0 - (nearest_enemy - dist_base)) * float(config.sector_enemy_overlap_penalty)
        score = (
            ap * 6.4
            + ships * 0.030
            + ar * 0.11
            + claim_prod * 1.55
            + cheap_claim * 0.85
            + support_prod * 3.65
            + support_count * 1.80
            + support_first_bonus
            + expansion_prod * 1.45
            + big_expansion * 2.6
            + len(connector) * 0.65
            - dist_base * 0.19
            - contest_penalty
            - enemy_overlap_penalty
            - anchor_rush_penalty
        )
        if score > 14.0:
            rollout, raw_queue = _outer_lane_rollout_score(
                obs=obs,
                prod=prod,
                x=x,
                y=y,
                radius=radius,
                base=base,
                anchor=anchor,
                claim=claim,
                support=support_before_anchor,
                expansion=expansion,
                enemy_idx=enemy_idx,
                base_score=score,
                horizon=int(config.outer_lane_rollout_horizon),
            )
            plans.append((rollout, score, anchor, claim, support_before_anchor, connector, expansion, lane_angle, raw_queue))

    if not plans:
        return None

    plans.sort(key=lambda p: (p[0], p[1]), reverse=True)
    plans = plans[: max(1, int(config.outer_lane_rollout_candidates))]
    rollout_score, score, anchor, claim, support, connector, expansion, lane_angle, raw_queue = plans[0]
    if rollout_score < 24.0:
        return None
    claim_sorted = sorted(
        [j for j in claim if float(obs.ships[j].item()) <= 40.0 or float(prod[j].item()) >= 2.0],
        key=lambda j: (
            _dist_xy(bx, by, float(x[j].item()), float(y[j].item())),
            -float(prod[j].item()),
        ),
    )[:7]
    if not claim_sorted and (ships <= 45.0 or ap >= 3.0):
        claim_sorted = [anchor]
    support_sorted = sorted(
        support,
        key=lambda j: float(prod[j].item()) * 10.0 + float(obs.ships[j].item()) * 0.10,
        reverse=True,
    )[:7]
    connector_sorted = sorted(
        connector,
        key=lambda j: _dist_xy(bx, by, float(x[j].item()), float(y[j].item())),
    )[:5]
    expansion_sorted = sorted(
        expansion,
        key=lambda j: float(prod[j].item()) * 12.0 + float(obs.ships[j].item()) * 0.08 + float(radius[j].item()) * 0.05,
        reverse=True,
    )[:7]

    seen: set[int] = set()
    unique_claim: list[int] = []
    for t in claim_sorted:
        if int(t) not in seen:
            unique_claim.append(int(t))
            seen.add(int(t))
    unique_support: list[int] = []
    for t in connector_sorted + support_sorted:
        if int(t) not in seen:
            unique_support.append(int(t))
            seen.add(int(t))
    unique_expansion: list[int] = []
    for t in expansion_sorted:
        if int(t) not in seen:
            unique_expansion.append(int(t))
            seen.add(int(t))

    return {
        "mode": "outer_lane_sequence",
        "anchor": int(anchor),
        "claim": unique_claim,
        "support": unique_support,
        "expansion": unique_expansion,
        "queue": [int(t) for t in raw_queue[:8] if 0 <= int(t) < P],
        "targets": unique_claim + unique_support + [int(anchor)] + unique_expansion,
        "angle": float(lane_angle),
        "created_step": _obs_step(obs_tensors),
        "score": float(score),
        "rollout_score": float(rollout_score),
        "alternatives": [
            {
                "anchor": int(p[2]),
                "rollout_score": float(p[0]),
                "base_score": float(p[1]),
            }
            for p in plans[:3]
        ],
    }


def _outer_lane_sequence_mature(*, obs, prod: Tensor, config: ProducerLiteConfig, plan: dict | None) -> bool:
    if not plan:
        return False
    P = int(prod.shape[0])
    core = [int(t) for t in plan.get("claim", []) + plan.get("support", []) if 0 <= int(t) < P]
    owned_core = [t for t in core if bool(obs.owned[t].item())]
    if not owned_core:
        return False
    total_prod = sum(float(prod[t].item()) for t in owned_core)
    total_ships = sum(float(obs.ships[t].item()) for t in owned_core)
    return total_ships >= 58.0 + total_prod * 6.0


def _append_outer_lane_sequence_targets(
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
    if int(player_count) < 4 or not bool(config.enable_outer_lane_sequence_4p):
        return target_idx, target_exists, None
    step = _obs_step(obs_tensors)
    if step > int(config.outer_lane_turn_limit):
        return target_idx, target_exists, getattr(memory, "outer_lane_plan", None)
    if getattr(memory, "outer_lane_plan", None) is None or step <= 1:
        memory.outer_lane_plan = _make_outer_lane_sequence_plan(
            obs=obs,
            obs_tensors=obs_tensors,
            prod=prod,
            source_mask=source_mask,
            config=config,
        )
    plan = getattr(memory, "outer_lane_plan", None)
    if not plan:
        return target_idx, target_exists, None
    if step < 34:
        return target_idx, target_exists, plan

    phase = _outer_lane_phase(step, config)
    mature = _outer_lane_sequence_mature(obs=obs, prod=prod, config=config, plan=plan)
    queued = [
        int(t) for t in plan.get("queue", [])
        if 0 <= int(t) < int(obs.P) and bool(obs.alive[int(t)].item()) and not bool(obs.owned[int(t)].item())
    ]
    anchor = int(plan.get("anchor", -1))
    anchor_targets = [anchor] if 0 <= anchor < int(obs.P) else []
    if phase == "claim":
        early_support = [
            int(t) for t in plan.get("support", [])
            if 0 <= int(t) < int(obs.P)
            and (float(obs.ships[int(t)].item()) <= 42.0 or float(prod[int(t)].item()) >= 2.5)
        ][:2]
        plan_targets = queued[: int(config.outer_lane_queue_width)] + [int(t) for t in plan.get("claim", []) + early_support]
    elif phase == "anchor":
        plan_targets = queued[: int(config.outer_lane_queue_width)] + [int(t) for t in plan.get("claim", []) + plan.get("support", []) + anchor_targets]
    elif phase == "project" and mature:
        plan_targets = queued[: int(config.outer_lane_queue_width) + 1] + [int(t) for t in plan.get("claim", []) + plan.get("support", []) + anchor_targets + plan.get("expansion", [])]
    elif phase == "project":
        plan_targets = queued[: int(config.outer_lane_queue_width)] + [int(t) for t in plan.get("claim", []) + plan.get("support", []) + anchor_targets]
    else:
        return target_idx, target_exists, plan

    P = int(obs.P)
    device = obs.device
    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)

    extras: list[int] = []
    for t in plan_targets[: int(config.outer_lane_max_extra_targets)]:
        if 0 <= int(t) < P and bool(obs.alive[int(t)].item()) and not bool(existing[int(t)].item()):
            extras.append(int(t))
    if not extras:
        return target_idx, target_exists, plan
    extra_idx = torch.tensor(extras, dtype=target_idx.dtype, device=target_idx.device)
    extra_exists = torch.ones(extra_idx.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extra_idx], dim=0), torch.cat([target_exists, extra_exists], dim=0), plan


def _apply_outer_lane_sequence_adjustment(
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
    outer_lane_plan: dict | None,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_outer_lane_sequence_4p) or not outer_lane_plan:
        return score
    step = _obs_step(obs_tensors)
    if step > int(config.outer_lane_turn_limit):
        return score
    if step < 34:
        return score

    P = int(prod.shape[0])
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    active = cand_active.any(dim=-1)

    claim_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    support_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    anchor_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    expansion_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    for t in outer_lane_plan.get("claim", []):
        if 0 <= int(t) < P:
            claim_mask[int(t)] = True
    for t in outer_lane_plan.get("support", []):
        if 0 <= int(t) < P:
            support_mask[int(t)] = True
    anchor = int(outer_lane_plan.get("anchor", -1))
    if 0 <= anchor < P:
        anchor_mask[anchor] = True
    for t in outer_lane_plan.get("expansion", []):
        if 0 <= int(t) < P:
            expansion_mask[int(t)] = True

    phase = _outer_lane_phase(step, config)
    mature = _outer_lane_sequence_mature(obs=obs, prod=prod, config=config, plan=outer_lane_plan)
    project_ok = phase == "project" and mature
    selected_claim = claim_mask[tgt] & active
    selected_support = support_mask[tgt] & active & (phase != "claim")
    selected_anchor = anchor_mask[tgt] & active & (phase != "claim")
    selected_expansion = expansion_mask[tgt] & active & bool(project_ok)

    planets = obs_tensors["planets"].to(score.dtype)
    dx = planets[:, 2] - 50.0
    dy = planets[:, 3] - 50.0
    radius = torch.sqrt(dx * dx + dy * dy)
    angle = torch.atan2(dy, dx)
    lane_angle = float(outer_lane_plan.get("angle", 0.0))
    angle_delta = torch.abs((angle - lane_angle + math.pi) % (2.0 * math.pi) - math.pi)

    bonus = torch.zeros_like(score)
    bonus = bonus + torch.where(selected_claim, torch.full_like(score, float(config.outer_lane_claim_bonus)), torch.zeros_like(score))
    bonus = bonus + torch.where(selected_support, torch.full_like(score, float(config.outer_lane_support_bonus)), torch.zeros_like(score))
    bonus = bonus + torch.where(selected_anchor, torch.full_like(score, float(config.outer_lane_anchor_bonus)), torch.zeros_like(score))
    bonus = bonus + torch.where(selected_expansion, torch.full_like(score, float(config.outer_lane_expansion_bonus)), torch.zeros_like(score))
    selected_any = selected_claim | selected_support | selected_anchor | selected_expansion
    bonus = bonus + torch.where(
        selected_any,
        (prod[tgt].to(score.dtype) * float(config.outer_lane_prod_bonus)).clamp(max=0.38)
        + torch.where(radius[tgt] >= 30.0, torch.full_like(score, float(config.outer_lane_outer_bonus)), torch.zeros_like(score)),
        torch.zeros_like(score),
    )

    lane_mask = claim_mask | support_mask | anchor_mask | expansion_mask
    offlane_cheap = (
        active
        & obs.is_neutral[tgt]
        & (~lane_mask[tgt])
        & (angle_delta[tgt] > float(config.outer_lane_angle_width) + 0.35)
        & (prod[tgt] <= 2.0)
        & (obs.ships[tgt].to(score.dtype) <= 32.0)
        & (step >= 28)
        & (step < int(config.outer_lane_anchor_turn))
    )
    offlane_penalty = torch.where(
        offlane_cheap,
        torch.full_like(score, float(config.outer_lane_offlane_penalty)),
        torch.zeros_like(score),
    )

    src_core = (claim_mask[src] | support_mask[src] | anchor_mask[src]) & obs.owned[src] & active & (step >= int(config.outer_lane_claim_turn))
    src_prod = prod[src].to(score.dtype)
    src_ships = obs.ships[src].to(score.dtype)
    hold = torch.full_like(score, float(config.outer_lane_hold_base)) + src_prod * float(config.outer_lane_hold_prod)
    after = src_ships - send
    hold_short = (hold - after).clamp(min=0.0)
    target_in_lane = lane_mask[tgt]
    relief = torch.where(target_in_lane, torch.full_like(score, 0.55), torch.ones_like(score))
    drain_penalty = (
        (hold_short / 30.0).clamp(max=1.5)
        * float(config.outer_lane_drain_penalty)
        * relief
    )
    drain_penalty = torch.where(src_core, drain_penalty, torch.zeros_like(score))
    return score + bonus - offlane_penalty - drain_penalty


def _outer_lane_candidate_mask(
    *,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    config: ProducerLiteConfig,
    outer_lane_plan: dict | None,
) -> Tensor:
    if not outer_lane_plan:
        return torch.zeros(cand_tgt_slot.shape[0], dtype=torch.bool, device=cand_tgt_slot.device)
    step = _obs_step(obs_tensors)
    if step < 34 or step > int(config.outer_lane_turn_limit):
        return torch.zeros(cand_tgt_slot.shape[0], dtype=torch.bool, device=cand_tgt_slot.device)
    P = int(prod.shape[0])
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    active = cand_active.any(dim=-1)

    claim_mask = torch.zeros(P, dtype=torch.bool, device=cand_tgt_slot.device)
    support_mask = torch.zeros(P, dtype=torch.bool, device=cand_tgt_slot.device)
    anchor_mask = torch.zeros(P, dtype=torch.bool, device=cand_tgt_slot.device)
    expansion_mask = torch.zeros(P, dtype=torch.bool, device=cand_tgt_slot.device)
    for t in outer_lane_plan.get("claim", []):
        if 0 <= int(t) < P:
            claim_mask[int(t)] = True
    for t in outer_lane_plan.get("support", []):
        if 0 <= int(t) < P:
            support_mask[int(t)] = True
    anchor = int(outer_lane_plan.get("anchor", -1))
    if 0 <= anchor < P:
        anchor_mask[anchor] = True
    for t in outer_lane_plan.get("expansion", []):
        if 0 <= int(t) < P:
            expansion_mask[int(t)] = True

    phase = _outer_lane_phase(step, config)
    mature = _outer_lane_sequence_mature(obs=obs, prod=prod, config=config, plan=outer_lane_plan)
    queue_targets = [
        int(t) for t in outer_lane_plan.get("queue", [])
        if 0 <= int(t) < P and bool(obs.alive[int(t)].item()) and not bool(obs.owned[int(t)].item())
    ][: max(1, int(config.outer_lane_queue_width))]
    if queue_targets:
        queue_mask = torch.zeros(P, dtype=torch.bool, device=cand_tgt_slot.device)
        for t in queue_targets:
            queue_mask[int(t)] = True
        return active & queue_mask[tgt] & (~obs.owned[tgt])

    if phase == "claim":
        lane_mask = claim_mask | support_mask
    elif phase == "anchor":
        lane_mask = claim_mask | support_mask | anchor_mask
    elif phase == "project" and mature:
        lane_mask = claim_mask | support_mask | anchor_mask | expansion_mask
    else:
        lane_mask = claim_mask | support_mask | anchor_mask
    # Reserve budget for capturing/building the lane, not for reinforcing owned
    # planets. Reinforcement can still happen in the normal greedy pass.
    return active & lane_mask[tgt] & (~obs.owned[tgt])


def _outer_lane_queue_executor_candidates(
    *,
    movement: PlanetMovement,
    obs,
    source_idx: Tensor,
    source_exists: Tensor,
    target_idx: Tensor,
    target_exists: Tensor,
    target_is_mine: Tensor,
    drain: Tensor,
    floor: Tensor,
    eta_cap: Tensor,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    outer_lane_plan: dict | None,
    current_step: int,
    pid: int,
    device,
    dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None:
    """Add explicit capture-floor sized candidates for the first lane queue targets."""
    if (
        int(player_count) < 4
        or not bool(config.enable_outer_lane_sequence_4p)
        or not bool(config.enable_outer_lane_queue_executor_4p)
        or not outer_lane_plan
        or int(current_step) < int(config.outer_lane_executor_start_turn)
        or int(current_step) > int(config.outer_lane_executor_turn_limit)
    ):
        return None
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    if S <= 0 or T <= 0 or int(floor.shape[-1]) <= 0:
        return None

    P = int(obs.P)
    queue: list[int] = []
    for t in outer_lane_plan.get("queue", []):
        ti = int(t)
        if 0 <= ti < P and bool(obs.alive[ti].item()) and not bool(obs.owned[ti].item()) and ti not in queue:
            queue.append(ti)
        if len(queue) >= int(config.outer_lane_executor_targets):
            break
    if not queue:
        for group in ("claim", "support"):
            for t in outer_lane_plan.get(group, []):
                ti = int(t)
                if 0 <= ti < P and bool(obs.alive[ti].item()) and not bool(obs.owned[ti].item()) and ti not in queue:
                    queue.append(ti)
                if len(queue) >= int(config.outer_lane_executor_targets):
                    break
            if len(queue) >= int(config.outer_lane_executor_targets):
                break
    if not queue:
        return None

    q_mask = torch.zeros(T, dtype=torch.bool, device=device)
    for ti in queue:
        q_mask = q_mask | (target_idx.clamp(0, max(P - 1, 0)) == int(ti))
    if not bool(q_mask.any()):
        return None

    K = int(floor.shape[-1])
    # First pass: use a mid-horizon floor to estimate ETA, then gather the exact
    # floor at that ETA and recompute aim with the resulting launch size.
    mid_k = min(max(1, K), 4) - 1
    base_need_t = (floor[:, mid_k] + float(config.outer_lane_executor_size_overhead)).clamp(min=float(config.min_ships_to_launch))
    sizes0 = base_need_t.view(1, T).expand(S, T)
    aim0 = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes0,
        active=source_exists.view(S, 1) & target_exists.view(1, T) & q_mask.view(1, T),
    )
    eta0 = aim0["eta"].clamp(min=1.0)
    k_arr = (eta0.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
    floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    sizes = (floor_at_arr + float(config.outer_lane_executor_size_overhead)).ceil().clamp(min=float(config.min_ships_to_launch))
    sizes = torch.minimum(sizes, drain.view(S, 1).floor().clamp(min=0.0))

    active_base = source_exists.view(S, 1) & target_exists.view(1, T) & q_mask.view(1, T)
    reachable = reachable_mask(
        movement,
        source_idx=source_idx,
        target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1),
        eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active_base & reachable,
    )
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    k_arr2 = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
    floor_at_arr2 = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr2.unsqueeze(-1)).squeeze(-1)
    clears_floor = sizes >= floor_at_arr2
    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        active_base
        & reachable
        & viable
        & clears_floor
        & src_neq_tgt
        & (sizes >= float(config.min_ships_to_launch))
    )
    C = S * T
    L = 1
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = aim["angle"].reshape(C, L)
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
        garrison_status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launches,
        player_id=pid,
    )
    score = torch.where(
        cand_valid,
        score + float(config.outer_lane_executor_bonus),
        torch.full_like(score, float("-inf")),
    )
    return cand_src, cand_send, cand_angle, cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, score


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
    current_step = _obs_step(obs_tensors)

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
    target_idx, target_exists, outer_lane_plan = _append_outer_lane_sequence_targets(
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
    target_idx, target_exists, domain_plan = _append_domain_factory_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        source_mask=source_mask,
        config=config,
        player_count=player_count,
        memory=memory,
        preferred_plan=winner_path,
    )
    target_idx, target_exists = _append_portfolio_regroup_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        prod=prod,
        garrison_status=garrison_status,
        config=config,
        player_count=player_count,
        current_step=current_step,
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
    queue_executor_part = _outer_lane_queue_executor_candidates(
        movement=movement,
        obs=obs,
        source_idx=source_idx,
        source_exists=source_exists,
        target_idx=target_idx,
        target_exists=target_exists,
        target_is_mine=target_is_mine,
        drain=drain,
        floor=floor,
        eta_cap=eta_cap,
        garrison_status=garrison_status,
        prod=prod,
        alive_by_step=alive_by_step,
        config=config,
        player_count=player_count,
        outer_lane_plan=outer_lane_plan,
        current_step=current_step,
        pid=pid,
        device=device,
        dtype=dtype,
    )
    if queue_executor_part is not None:
        tier_parts.append(queue_executor_part)

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
        cand_eta=cand_eta,
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
    score = _apply_domain_factory_adjustment(
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
        domain_plan=domain_plan,
    )
    score = _apply_outer_lane_sequence_adjustment(
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
        outer_lane_plan=outer_lane_plan,
    )
    score = _apply_low_conflict_expansion_adjustment(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
    )

    source_budget = obs.ships.to(dtype).clone()
    target_exists_for_normal = target_exists.clone()
    portfolio_home_entries = _empty_entries(device, dtype)
    portfolio_abandon_entries = _empty_entries(device, dtype)
    portfolio_features = None
    if _portfolio_enabled(config, player_count=player_count, current_step=current_step):
        portfolio_features = _portfolio_features(
            obs=obs,
            prod=prod,
            cache=cache,
            garrison_status=garrison_status,
            config=config,
            player_count=player_count,
            current_step=current_step,
        )
        home_mask, home_score = _portfolio_home_defense_mask(
            score=score,
            cand_send=cand_send,
            cand_eta=cand_eta,
            cand_tgt_slot=cand_tgt_slot,
            cand_active=cand_active,
            cand_is_def=cand_is_def,
            obs=obs,
            prod=prod,
            garrison_status=garrison_status,
            config=config,
            features=portfolio_features,
        )
        if bool(home_mask.any()):
            portfolio_home_entries, source_budget, target_exists_for_normal = _select_one_portfolio_slot(
                P=P, device=device, dtype=dtype, slot_score=home_score,
                cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
                cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
                cand_is_def=cand_is_def, source_budget=source_budget,
                target_exists=target_exists_for_normal, target_idx=target_idx,
                target_exists_for_normal=target_exists_for_normal,
                roi_threshold=-0.25,
            )
        abandon_mask, abandon_score = _portfolio_abandon_regroup_mask(
            score=score,
            cand_src=cand_src,
            cand_send=cand_send,
            cand_eta=cand_eta,
            cand_tgt_slot=cand_tgt_slot,
            cand_active=cand_active,
            cand_is_def=cand_is_def,
            obs=obs,
            prod=prod,
            garrison_status=garrison_status,
            config=config,
        )
        if bool(abandon_mask.any()):
            portfolio_abandon_entries, source_budget, target_exists_for_normal = _select_one_portfolio_slot(
                P=P, device=device, dtype=dtype, slot_score=abandon_score,
                cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
                cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
                cand_is_def=cand_is_def, source_budget=source_budget,
                target_exists=target_exists_for_normal, target_idx=target_idx,
                target_exists_for_normal=target_exists_for_normal,
                roi_threshold=-0.10,
            )
    reserve_entries = _empty_entries(device, dtype)
    reserve_mask, reserve_score = _low_conflict_reserved_safe_mask(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        prod=prod,
        cache=cache,
        config=config,
        player_count=player_count,
        current_step=current_step,
    )
    if portfolio_features is not None and portfolio_features.cheap_safe_neutral_count <= 0:
        reserve_mask = torch.zeros_like(reserve_mask)
        reserve_score = torch.full_like(reserve_score, float("-inf"))
    if bool(reserve_mask.any()):
        reserve_entries, source_budget = _greedy_select(
            P=P, W=1, device=device, dtype=dtype, score=reserve_score,
            cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
            cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_is_def=cand_is_def, source_budget=source_budget,
            target_exists=target_exists_for_normal,
            roi_threshold=-1.0,
        )
        if bool(reserve_entries.valid.any()):
            taken = reserve_entries.target_slots[reserve_entries.valid].clamp(0, max(P - 1, 0))
            if taken.numel() > 0:
                taken_mask = torch.zeros(P, dtype=torch.bool, device=device)
                taken_mask.scatter_(0, taken, True)
                target_exists_for_normal = target_exists_for_normal & (~taken_mask[target_idx.clamp(0, P - 1)])
    lane_entries = _empty_entries(device, dtype)
    lane_phase = _outer_lane_phase(_obs_step(obs_tensors), config)
    mode = getattr(memory, "strategy_mode", None)
    lane_budget_enabled = (
        mode in ("winner_outer_domain", "domain_factory", "lane_anchor")
        and current_step >= int(config.outer_lane_budget_start_turn)
    )
    lane_waves = (
        int(config.outer_lane_project_budget_waves)
        if lane_phase == "project"
        else int(config.outer_lane_budget_waves)
    )
    lane_mask = _outer_lane_candidate_mask(
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        config=config,
        outer_lane_plan=outer_lane_plan,
    )
    if lane_budget_enabled and lane_waves > 0 and bool(lane_mask.any()):
        lane_score = torch.where(lane_mask, score, torch.full_like(score, float("-inf")))
        lane_entries, source_budget = _greedy_select(
            P=P, W=max(1, min(lane_waves, W)), device=device, dtype=dtype, score=lane_score,
            cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
            cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
            cand_is_def=cand_is_def, source_budget=source_budget,
            target_exists=target_exists,
            roi_threshold=float(config.roi_threshold) - float(config.outer_lane_budget_roi_discount),
        )
        if bool(lane_entries.valid.any()):
            taken = lane_entries.target_slots[lane_entries.valid].clamp(0, max(P - 1, 0))
            if taken.numel() > 0:
                taken_mask = torch.zeros(P, dtype=torch.bool, device=device)
                taken_mask.scatter_(0, taken, True)
                target_exists_for_normal = target_exists_for_normal & (~taken_mask[target_idx.clamp(0, P - 1)])

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists_for_normal, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return concat_launch_entries([portfolio_home_entries, portfolio_abandon_entries, reserve_entries, lane_entries, wave_entries])
    if enemy_mass is None:
        enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([portfolio_home_entries, portfolio_abandon_entries, reserve_entries, lane_entries, wave_entries, regroup_entries])


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
    enable_outer_lane_sequence_4p=True,
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
CONFIG_4P = dataclasses.replace(CONFIG_4P, enable_portfolio_executor_4p=True)
CONFIG_4P = _ow_apply_config_overrides(CONFIG_4P, _OW_STRATEGY_PARAMS.get("config_4p"))


# Single-file 4P selector:
# - s8_burst keeps sample8's multi-size 4P behavior.
# - s7_stable uses the same orbit_lite engine but sample7-like one-size behavior.
# This avoids dynamic folder imports, which changed behavior in previous tests.
CONFIG_4P_S8_BURST = dataclasses.replace(
    CONFIG_4P,
    enable_domain_factory_4p=True,
    domain_factory_turn_limit=150,
    domain_factory_project_turn=82,
    domain_factory_max_extra_targets=8,
    domain_factory_angle_width=1.05,
    domain_factory_core_bonus=0.22,
    domain_factory_support_bonus=0.14,
    domain_factory_expansion_bonus=0.30,
    domain_factory_prod_bonus=0.028,
    domain_factory_outer_bonus=0.08,
    domain_factory_offdomain_cheap_penalty=0.0,
    domain_factory_hold_base=14.0,
    domain_factory_hold_prod=3.2,
    domain_factory_support_hold_base=6.0,
    domain_factory_support_hold_prod=1.8,
    domain_factory_drain_penalty=0.04,
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
    enable_domain_factory_4p=True,
    domain_factory_turn_limit=155,
    domain_factory_project_turn=84,
    domain_factory_max_extra_targets=8,
    domain_factory_angle_width=1.05,
    domain_factory_core_bonus=0.24,
    domain_factory_support_bonus=0.14,
    domain_factory_expansion_bonus=0.32,
    domain_factory_prod_bonus=0.030,
    domain_factory_outer_bonus=0.08,
    domain_factory_offdomain_cheap_penalty=0.0,
    domain_factory_hold_base=14.0,
    domain_factory_hold_prod=3.4,
    domain_factory_support_hold_base=6.0,
    domain_factory_support_hold_prod=1.8,
    domain_factory_drain_penalty=0.04,
)

CONFIG_4P_LOW_CONFLICT = dataclasses.replace(
    CONFIG_4P_S7_STABLE,
    enable_low_conflict_expansion_4p=True,
    low_conflict_turn_limit=80,
    low_conflict_cheap_bonus=0.58,
    low_conflict_enemy_target_penalty=0.72,
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
    enable_domain_factory_4p=True,
    domain_factory_turn_limit=155,
    domain_factory_project_turn=82,
    domain_factory_max_extra_targets=8,
    domain_factory_angle_width=1.10,
    domain_factory_core_bonus=0.24,
    domain_factory_support_bonus=0.14,
    domain_factory_expansion_bonus=0.32,
    domain_factory_prod_bonus=0.030,
    domain_factory_outer_bonus=0.08,
    domain_factory_offdomain_cheap_penalty=0.0,
    domain_factory_hold_base=14.0,
    domain_factory_hold_prod=3.2,
    domain_factory_support_hold_base=6.0,
    domain_factory_support_hold_prod=1.8,
    domain_factory_drain_penalty=0.04,
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
    enable_domain_factory_4p=True,
    domain_factory_turn_limit=165,
    domain_factory_project_turn=86,
    domain_factory_max_extra_targets=9,
    domain_factory_angle_width=1.02,
    domain_factory_core_bonus=0.34,
    domain_factory_support_bonus=0.20,
    domain_factory_expansion_bonus=0.44,
    domain_factory_prod_bonus=0.04,
    domain_factory_outer_bonus=0.12,
    domain_factory_offdomain_cheap_penalty=0.02,
    domain_factory_hold_base=20.0,
    domain_factory_hold_prod=4.5,
    domain_factory_support_hold_base=8.0,
    domain_factory_support_hold_prod=2.5,
    domain_factory_drain_penalty=0.08,
)

CONFIG_4P_DOMAIN_FACTORY = dataclasses.replace(
    CONFIG_4P,
    horizon=15,
    max_sources_per_lane=8,
    max_offensive_targets=16,
    max_defensive_targets=2,
    max_waves_per_turn=8,
    roi_threshold=1.48,
    min_ships_to_launch=4.0,
    reinforce_size_beta=2.25,
    reinforce_eta_free=3.0,
    reinforce_eta_scale=12.0,
    enable_regroup=True,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.22,
    max_regroup_sources_per_lane=7,
    max_regroup_targets_per_source=8,
    terminal_phase_turns=0,
    size_multipliers=(0.75, 1.0),
    enable_domain_factory_4p=True,
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

    # Close, rich outer-cluster boards can look attractive to the burst script,
    # but thin scattering lets a nearby rival turn our home sector into a kill box.
    if (
        enemy_cluster_risk >= 60.0
        and enemy_dist <= 55.0
        and near_prod <= 8.0
        and mid_prod >= 18.0
        and mid_prod <= 24.0
        and mid_cheap <= 4.5
        and high65 >= 8.0
        and chain >= 48.0
        and outer_anchor >= 50.0
        and support_density >= 10.0
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
        return "domain_factory"
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
        return "domain_factory"

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
    if (
        enemy_cluster_risk >= 58.0
        and enemy_dist >= 48.0
        and enemy_dist <= 58.0
        and near_prod <= 7.0
        and mid_prod <= 16.0
        and mid_cheap <= 3.0
        and high65 <= 4.0
        and chain <= 45.0
        and outer_anchor <= 55.0
        and support_density <= 12.0
    ):
        return "low_conflict_expansion"
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
    if mode == "domain_factory":
        return CONFIG_4P_DOMAIN_FACTORY
    if mode == "winner_outer_domain":
        return CONFIG_4P_WINNER_PATH
    if mode == "enemy_domain_block":
        return CONFIG_4P_ENEMY_DOMAIN_BLOCK
    if mode == "low_conflict_expansion":
        return CONFIG_4P_LOW_CONFLICT
    if mode == "s8_burst":
        return CONFIG_4P_S8_BURST
    return CONFIG_4P_S7_STABLE


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.strategy_mode: str | None = None
        self.winner_path: dict | None = None
        self.domain_plan: dict | None = None
        self.outer_lane_plan: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.strategy_mode = None
        self.winner_path = None
        self.domain_plan = None
        self.outer_lane_plan = None


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
            mem.domain_plan = None
            mem.outer_lane_plan = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        current_player = int(obs_tensors["player"].reshape(-1)[0].item())
        min_count = current_player + 1
        mem.cached_player_count = 4 if max(int(mem.cached_player_count), min_count) > 2 else 2
        if int(mem.cached_player_count) >= 4 and mem.strategy_mode is None:
            mem.strategy_mode = _choose_4p_mode(obs_tensors)
        if int(mem.cached_player_count) < 4:
            mem.strategy_mode = None
        step = int(obs_tensors["step"].reshape(-1)[0].item())
        # Keep the proven sample-style opening. The dedicated domain_factory
        # strategy is powerful only after we already have a few planets; using
        # it from step 0 over-committed and missed early captures.
        if int(mem.cached_player_count) >= 4 and mem.strategy_mode == "domain_factory" and step < 62:
            base = CONFIG_4P_S7_STABLE
        else:
            base = _config_for(mem.cached_player_count, mem.strategy_mode)
        config = _apply_phase_config(base, step)
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
