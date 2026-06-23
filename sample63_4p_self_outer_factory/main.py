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
    # --- sample63 self outer factory: build our own safe outer production base
    enable_self_outer_factory_4p: bool = False
    self_outer_turn_limit: int = 130
    self_outer_anchor_max_dist: float = 78.0
    self_outer_support_radius: float = 20.0
    self_outer_max_extra_targets: int = 7
    self_outer_anchor_bonus: float = 1.05
    self_outer_support_bonus: float = 0.72
    self_outer_prod_bonus: float = 0.085
    self_outer_outer_bonus: float = 0.24
    self_outer_offlane_penalty: float = 1.35
    self_outer_center_penalty: float = 1.10
    self_outer_pressure_penalty: float = 1.65
    self_outer_anchor_hold_base: float = 34.0
    self_outer_anchor_hold_prod: float = 10.0
    self_outer_near_hold_base: float = 12.0
    self_outer_near_hold_prod: float = 4.0
    self_outer_drain_penalty: float = 0.72
    self_outer_reserve_start: int = 10
    self_outer_reserve_limit: int = 105
    self_outer_reserve_roi: float = -0.40


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


def _make_self_outer_factory_plan(
    *,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    source_mask: Tensor,
    cache,
    config: ProducerLiteConfig,
) -> dict | None:
    if not bool(source_mask.any()):
        return None
    P = int(obs.P)
    if P <= 0:
        return None
    dtype = obs.ships.dtype
    device = obs.device
    planets = obs_tensors["planets"].to(dtype)
    x = planets[:, 2]
    y = planets[:, 3]
    radius = torch.sqrt((x - 50.0) ** 2 + (y - 50.0) ** 2)
    d0 = cache.cross_dist[0].to(dtype)

    owned_idx = torch.nonzero(source_mask, as_tuple=False).flatten()
    base = int(owned_idx[torch.argmax(obs.ships[owned_idx])].item())
    base_angle = float(torch.atan2(y[base] - 50.0, x[base] - 50.0).item())
    speed = fleet_speed(obs.ships.to(dtype).clamp(min=1e-6))
    my_eta = torch.where(
        source_mask.view(P, 1),
        d0 / speed.view(P, 1).clamp(min=1e-6),
        torch.full((P, P), float("inf"), dtype=dtype, device=device),
    ).amin(dim=0)
    enemy_sources = obs.is_enemy & obs.alive & (obs.ships >= 4.0)
    enemy_eta = torch.where(
        enemy_sources.view(P, 1),
        d0 / speed.view(P, 1).clamp(min=1e-6),
        torch.full((P, P), float("inf"), dtype=dtype, device=device),
    )
    enemy_min_eta = enemy_eta.amin(dim=0)
    enemy_reach_count = ((enemy_eta <= (my_eta.view(1, P) + 8.0)) & enemy_sources.view(P, 1)).sum(dim=0)

    neutral_idx = torch.nonzero(obs.is_neutral & obs.alive, as_tuple=False).flatten().tolist()
    best: tuple[float, int, list[int], float] | None = None
    for anchor in neutral_idx:
        ar = float(radius[anchor].item())
        ap = float(prod[anchor].item())
        ships = float(obs.ships[anchor].item())
        if ar < 31.0 or ap < 2.5:
            continue
        if float(my_eta[anchor].item()) > float(config.self_outer_anchor_max_dist):
            continue
        aa = float(torch.atan2(y[anchor] - 50.0, x[anchor] - 50.0).item())
        if _angle_diff(base_angle, aa) > 1.45:
            continue
        if ships > 58.0 and ap < 4.0:
            continue

        supports: list[int] = []
        support_potential = 0.0
        for j in neutral_idx:
            if j == anchor:
                continue
            d_anchor = float(d0[anchor, j].item())
            if d_anchor > float(config.self_outer_support_radius):
                continue
            jp = float(prod[j].item())
            js = float(obs.ships[j].item())
            if jp < 2.0 or js > 30.0:
                continue
            if int(enemy_reach_count[j].item()) > 1:
                continue
            supports.append(int(j))
            support_potential += jp * 8.0 + max(0.0, 22.0 - js) * 0.25

        enemy_pressure = max(0.0, 20.0 - float(enemy_min_eta[anchor].item())) * 2.0 + float(enemy_reach_count[anchor].item()) * 9.0
        center_penalty = max(0.0, 32.0 - ar) * 3.0
        score = (
            ap * 80.0
            + ar * 2.5
            + support_potential * 0.35
            - ships * 1.3
            - enemy_pressure * 0.6
            - center_penalty
        )
        if best is None or score > best[0]:
            best = (score, int(anchor), supports, aa)
    if best is None or best[0] < 120.0:
        return None
    _, anchor, supports, angle = best
    ordered_supports = sorted(
        supports,
        key=lambda j: float(prod[j].item()) * 20.0 - float(obs.ships[j].item()) - float(d0[anchor, j].item()) * 0.3,
        reverse=True,
    )[:4]
    return {
        "anchor": int(anchor),
        "support": [int(t) for t in ordered_supports],
        "targets": [int(anchor)] + [int(t) for t in ordered_supports],
        "angle": float(angle),
        "created_step": _obs_step(obs_tensors),
        "score": float(best[0]),
    }


def _append_self_outer_factory_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    source_mask: Tensor,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
    memory,
) -> tuple[Tensor, Tensor, dict | None]:
    if int(player_count) < 4 or not bool(config.enable_self_outer_factory_4p):
        return target_idx, target_exists, None
    step = _obs_step(obs_tensors)
    if step > int(config.self_outer_turn_limit):
        return target_idx, target_exists, getattr(memory, "self_outer_factory_plan", None)
    plan = getattr(memory, "self_outer_factory_plan", None)
    anchor_alive = False
    if plan:
        anchor = int(plan.get("anchor", -1))
        anchor_alive = 0 <= anchor < int(obs.P) and bool(obs.alive[anchor].item()) and not bool(obs.is_enemy[anchor].item())
    if plan is None or step <= 10 or not anchor_alive:
        memory.self_outer_factory_plan = _make_self_outer_factory_plan(
            obs=obs, obs_tensors=obs_tensors, prod=prod, source_mask=source_mask, cache=cache, config=config,
        )
    plan = getattr(memory, "self_outer_factory_plan", None)
    if not plan:
        return target_idx, target_exists, None
    P = int(obs.P)
    device = obs.device
    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)
    extras = []
    for t in plan.get("targets", [])[: int(config.self_outer_max_extra_targets)]:
        ti = int(t)
        if 0 <= ti < P and bool(obs.alive[ti].item()) and not bool(existing[ti].item()):
            extras.append(ti)
    if not extras:
        return target_idx, target_exists, plan
    extra_idx = torch.tensor(extras, dtype=target_idx.dtype, device=target_idx.device)
    extra_exists = torch.ones(extra_idx.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extra_idx], dim=0), torch.cat([target_exists, extra_exists], dim=0), plan


def _apply_self_outer_factory_policy(
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
    plan: dict | None,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_self_outer_factory_4p) or not plan:
        return score
    step = _obs_step(obs_tensors)
    if step > int(config.self_outer_turn_limit):
        return score
    P = int(obs.P)
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    active = cand_active.any(dim=-1)
    anchor = int(plan.get("anchor", -1))
    targets = set(int(t) for t in plan.get("targets", []))
    support = set(int(t) for t in plan.get("support", []))
    target_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    support_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    for t in targets:
        if 0 <= t < P:
            target_mask[t] = True
    for t in support:
        if 0 <= t < P:
            support_mask[t] = True
    radius = torch.sqrt((obs.x.to(score.dtype) - 50.0) ** 2 + (obs.y.to(score.dtype) - 50.0) ** 2)
    selected_anchor = active & (tgt == int(anchor))
    selected_support = active & support_mask[tgt]
    selected = selected_anchor | selected_support
    bonus = torch.where(
        selected_anchor,
        torch.full_like(score, float(config.self_outer_anchor_bonus)),
        torch.where(selected_support, torch.full_like(score, float(config.self_outer_support_bonus)), torch.zeros_like(score)),
    )
    bonus = bonus + torch.where(
        selected,
        (prod[tgt].to(score.dtype) * float(config.self_outer_prod_bonus)).clamp(max=0.42)
        + torch.where(radius[tgt] >= 32.0, torch.full_like(score, float(config.self_outer_outer_bonus)), torch.zeros_like(score)),
        torch.zeros_like(score),
    )
    d_to_anchor = cache.cross_dist[0][anchor, tgt].to(score.dtype) if 0 <= anchor < P else torch.full_like(score, 999.0)
    offlane = active & obs.is_neutral[tgt] & (~target_mask[tgt]) & (d_to_anchor > float(config.self_outer_support_radius) + 2.0) & (step < 130)
    center = active & obs.is_neutral[tgt] & (radius[tgt] < 28.0) & (step < 100)
    enemy = obs.is_enemy & obs.alive
    enemy_near = ((cache.cross_dist[0][:, tgt].to(score.dtype) <= 18.0) & enemy.view(P, 1)).sum(dim=0) >= 2
    pressure_bad = active & obs.is_neutral[tgt] & enemy_near & (~target_mask[tgt])
    src_dist = cache.cross_dist[0][anchor, src].to(score.dtype) if 0 <= anchor < P else torch.full_like(score, 999.0)
    hold = torch.where(
        src == int(anchor),
        torch.full_like(score, float(config.self_outer_anchor_hold_base)) + prod[src].to(score.dtype) * float(config.self_outer_anchor_hold_prod),
        torch.where(
            src_dist <= 12.0,
            torch.full_like(score, float(config.self_outer_near_hold_base)) + prod[src].to(score.dtype) * float(config.self_outer_near_hold_prod),
            torch.zeros_like(score),
        ),
    )
    after = obs.ships[src].to(score.dtype) - send
    drain_penalty = ((hold - after).clamp(min=0.0) / 28.0).clamp(max=1.6) * float(config.self_outer_drain_penalty)
    penalty = (
        torch.where(offlane, torch.full_like(score, float(config.self_outer_offlane_penalty)), torch.zeros_like(score))
        + torch.where(center, torch.full_like(score, float(config.self_outer_center_penalty)), torch.zeros_like(score))
        + torch.where(pressure_bad, torch.full_like(score, float(config.self_outer_pressure_penalty)), torch.zeros_like(score))
        + torch.where(active & (hold > 0.0), drain_penalty, torch.zeros_like(score))
    )
    return score + bonus - penalty


def _self_outer_reserved_mask(
    *,
    score: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    prod: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    plan: dict | None,
) -> tuple[Tensor, Tensor]:
    if int(player_count) < 4 or not bool(config.enable_self_outer_factory_4p) or not plan:
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))
    step = _obs_step(obs_tensors)
    if step < int(config.self_outer_reserve_start) or step > int(config.self_outer_reserve_limit):
        return torch.zeros_like(score, dtype=torch.bool), torch.full_like(score, float("-inf"))
    P = int(obs.P)
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(score.dtype)
    eta = cand_eta[:, 0].to(score.dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1)
    targets = torch.zeros(P, dtype=torch.bool, device=score.device)
    for t in plan.get("targets", []):
        if 0 <= int(t) < P:
            targets[int(t)] = True
    valid = active & targets[tgt] & obs.is_neutral[tgt]
    reserve_score = (
        prod[tgt].to(score.dtype) * 36.0
        - send * 1.8
        - eta * 1.4
        + torch.where(tgt == int(plan.get("anchor", -1)), torch.full_like(score, 12.0), torch.zeros_like(score))
    ) * 0.04
    return valid, torch.where(valid, reserve_score, torch.full_like(score, float("-inf")))


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
    target_idx, target_exists, self_outer_plan = _append_self_outer_factory_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        source_mask=source_mask,
        cache=cache,
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
    score = _apply_self_outer_factory_policy(
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
        plan=self_outer_plan,
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

    source_budget = obs.ships.to(dtype).clone()
    target_exists_for_normal = target_exists.clone()
    reserve_entries = _empty_entries(device, dtype)
    reserve_mask, reserve_score = _self_outer_reserved_mask(
        score=score,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        config=config,
        player_count=player_count,
        plan=self_outer_plan,
    )
    if not bool(reserve_mask.any()):
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
    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists_for_normal, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return concat_launch_entries([reserve_entries, wave_entries])
    if enemy_mass is None:
        enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([reserve_entries, wave_entries, regroup_entries])


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
CONFIG_4P_S8_BURST = CONFIG_4P
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
)

CONFIG_4P_SELF_OUTER_FACTORY = dataclasses.replace(
    CONFIG_4P,
    horizon=15,
    max_sources_per_lane=7,
    max_offensive_targets=14,
    max_defensive_targets=2,
    max_waves_per_turn=7,
    roi_threshold=1.42,
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
    enable_self_outer_factory_4p=True,
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
    forced = os.environ.get("OW_FORCE_MODE")
    if forced in {
        "s7_stable",
        "s8_burst",
        "lane_anchor",
        "winner_outer_domain",
        "enemy_domain_block",
        "top_director",
        "self_outer_factory",
    }:
        return forced
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

    # Sparse outer-bank starts need a factory route: take the outer lane and
    # its local support instead of joining the early center/off-lane race.
    if (
        enemy_dist <= 55.0
        and outer_anchor >= 45.0
        and support_density >= 8.0
        and mid_cheap <= 1.5
        and high65 <= 3.0
        and planets <= 28.0
    ):
        return "self_outer_factory"
    if (
        enemy_dist <= 55.0
        and outer_anchor >= 84.0
        and support_density >= 25.0
        and mid_prod >= 22.0
        and mid_cheap >= 5.0
        and high65 >= 7.0
        and (high65 <= 8.0 or enemy_cluster_risk >= 130.0)
        and planets <= 32.0
    ):
        return "self_outer_factory"
    if (
        enemy_dist >= 68.0
        and enemy_dist <= 74.0
        and outer_anchor >= 50.0
        and outer_anchor <= 60.0
        and support_density >= 16.0
        and support_density <= 20.0
        and mid_prod <= 15.0
        and high65 <= 5.0
        and planets >= 30.0
    ):
        return "self_outer_factory"

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
    if mode == "self_outer_factory":
        return CONFIG_4P_SELF_OUTER_FACTORY
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
        self.self_outer_factory_plan: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.strategy_mode = None
        self.winner_path = None
        self.self_outer_factory_plan = None


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
            mem.self_outer_factory_plan = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        current_player = int(obs_tensors["player"].reshape(-1)[0].item())
        min_count = current_player + 1
        mem.cached_player_count = 4 if max(int(mem.cached_player_count), min_count) > 2 else 2
        if int(mem.cached_player_count) >= 4 and mem.strategy_mode is None:
            mem.strategy_mode = _choose_4p_mode(obs_tensors)
        if int(mem.cached_player_count) < 4:
            mem.strategy_mode = None
        base = _config_for(mem.cached_player_count, mem.strategy_mode)
        step = int(obs_tensors["step"].reshape(-1)[0].item())
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
