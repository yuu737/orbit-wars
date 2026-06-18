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
    # --- 4P domain v2: keep a small stateful opening route ------------------
    enable_domain_plan_4p: bool = False
    domain_turn_limit: int = 170
    domain_anchor_eta_cap: float = 20.0
    domain_support_radius: float = 55.0
    domain_expansion_radius: float = 88.0
    domain_max_targets: int = 8
    domain_bonus: float = 0.54
    domain_support_bonus: float = 0.24
    domain_expansion_bonus: float = 0.34
    domain_anchor_hold_base: float = 36.0
    domain_anchor_hold_prod: float = 9.0
    domain_source_penalty: float = 0.72


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


def _make_domain_plan_4p(
    *,
    obs,
    obs_tensors: dict,
    cache,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
) -> dict | None:
    if not bool(config.enable_domain_plan_4p) or not bool(source_mask.any()):
        return None
    P = int(obs.P)
    if P <= 0:
        return None
    device = obs.device
    dtype = obs.ships.dtype
    d0 = cache.cross_dist[0].to(dtype)
    speed = fleet_speed(obs.ships.to(dtype)).clamp(min=1e-6)
    eta_all = d0 / speed.view(P, 1)
    my_eta = torch.where(source_mask.view(P, 1), eta_all, torch.full_like(eta_all, float("inf"))).amin(dim=0)

    planets = obs_tensors["planets"].to(dtype)
    x = planets[:, 2]
    y = planets[:, 3]
    angle = torch.atan2(y - 50.0, x - 50.0)
    center_dist = torch.sqrt((x - 50.0).square() + (y - 50.0).square())
    owned_idx = torch.nonzero(source_mask, as_tuple=False).flatten()
    base = int(owned_idx[torch.argmax(obs.ships[owned_idx])].item())
    base_angle = angle[base]
    angle_delta = torch.abs((angle - base_angle + math.pi) % (2.0 * math.pi) - math.pi)

    neutral = obs.is_neutral & obs.alive
    anchor_mask = (
        neutral
        & (my_eta <= float(config.domain_anchor_eta_cap))
        & (angle_delta <= float(config.lane_anchor_angle_width))
        & ((prod >= 2.0) | (obs.ships >= 22.0))
    )
    if not bool(anchor_mask.any()):
        return None

    local = neutral.view(1, P) & (d0 <= float(config.domain_support_radius))
    support_prod = torch.where(local, prod.view(1, P).to(dtype), torch.zeros(P, P, dtype=dtype, device=device)).sum(dim=1)
    support_count = local.sum(dim=1).to(dtype)
    expansion = neutral.view(1, P) & (d0 <= float(config.domain_expansion_radius)) & (
        (prod.view(1, P) >= 3.0) | (obs.ships.view(1, P) >= 38.0)
    )
    expansion_prod = torch.where(expansion, prod.view(1, P).to(dtype), torch.zeros(P, P, dtype=dtype, device=device)).sum(dim=1)
    expansion_count = expansion.sum(dim=1).to(dtype)

    anchor_score = (
        prod.to(dtype) * 8.0
        + obs.ships.to(dtype) * 0.05
        + support_prod * 1.35
        + support_count * 0.85
        + expansion_prod * 0.95
        + expansion_count * 1.1
        + (float(config.domain_anchor_eta_cap) - my_eta).clamp(min=0.0) * 0.22
        - angle_delta * 1.15
        - center_dist * 0.015
    )
    anchor_score = torch.where(anchor_mask, anchor_score, torch.full_like(anchor_score, float("-inf")))
    anchor = int(torch.argmax(anchor_score).item())
    if not torch.isfinite(anchor_score[anchor]):
        return None

    support_score = torch.where(
        local[anchor],
        prod.to(dtype) * 6.0 + obs.ships.to(dtype) * 0.04 - d0[anchor] * 0.045,
        torch.full((P,), float("-inf"), dtype=dtype, device=device),
    )
    route_cap = max(1, min(int(config.domain_max_targets), P))
    support_idx, support_exists = _candidate_indices(support_score, torch.isfinite(support_score), route_cap)
    support = [int(x) for x in support_idx[support_exists].detach().cpu().tolist()]
    if anchor not in support:
        support.insert(0, anchor)

    support_mask = torch.zeros(P, dtype=torch.bool, device=device)
    if support:
        support_mask[torch.tensor(support, dtype=torch.long, device=device).clamp(0, P - 1)] = True
    expansion_mask = neutral & ~support_mask & (d0[anchor] <= float(config.domain_expansion_radius)) & (
        (prod >= 3.0) | (obs.ships >= 38.0)
    )
    expansion_score = torch.where(
        expansion_mask,
        prod.to(dtype) * 7.0 + obs.ships.to(dtype) * 0.05 - d0[anchor] * 0.035,
        torch.full((P,), float("-inf"), dtype=dtype, device=device),
    )
    expansion_list: list[int] = []
    if bool(torch.isfinite(expansion_score).any()):
        exp_idx, exp_exists = _candidate_indices(expansion_score, torch.isfinite(expansion_score), min(4, P))
        expansion_list = [int(x) for x in exp_idx[exp_exists].detach().cpu().tolist()]

    return {
        "anchor": anchor,
        "support": list(dict.fromkeys(support)),
        "expansion": list(dict.fromkeys(expansion_list)),
        "created_step": _obs_step(obs_tensors),
        "confidence": float(anchor_score[anchor].item()),
    }


def _domain_plan_targets(
    *,
    memory,
    obs,
    obs_tensors: dict,
    cache,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> tuple[Tensor | None, Tensor | None, int | None, str, bool]:
    if int(player_count) < 4 or not bool(config.enable_domain_plan_4p):
        return None, None, None, "free", False
    step = _obs_step(obs_tensors)
    if step > int(config.domain_turn_limit):
        return None, None, None, "free", False
    if getattr(memory, "domain_plan", None) is None or step <= 1:
        memory.domain_plan = _make_domain_plan_4p(
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            prod=prod,
            source_mask=source_mask,
            config=config,
        )
    plan = getattr(memory, "domain_plan", None)
    if not plan:
        return None, None, None, "free", False

    P = int(obs.P)
    anchor = int(plan.get("anchor", -1))
    if anchor < 0 or anchor >= P or not bool(obs.alive[anchor]):
        return None, None, None, "free", False

    anchor_owned = bool(obs.owned[anchor])
    anchor_hold = float(config.domain_anchor_hold_base) + float(config.domain_anchor_hold_prod) * float(prod[anchor].item())
    anchor_mature = anchor_owned and float(obs.ships[anchor].item()) >= anchor_hold
    if step < 45:
        phase = "claim"
    elif step < 105:
        phase = "consolidate"
    else:
        phase = "project"

    support = [int(t) for t in plan.get("support", []) if 0 <= int(t) < P and bool(obs.alive[int(t)])]
    expansion = [int(t) for t in plan.get("expansion", []) if 0 <= int(t) < P and bool(obs.alive[int(t)])]
    if not anchor_owned:
        active = [anchor] + support[:4]
    elif not anchor_mature or phase == "consolidate":
        active = support[:6]
    else:
        active = expansion[:4] + support[:4]

    active = [t for t in dict.fromkeys(active) if 0 <= t < P and bool(obs.alive[t])]
    if not active:
        return None, None, anchor, phase, anchor_mature
    device = obs.device
    targets = torch.tensor(active, dtype=torch.long, device=device)
    support_targets = torch.tensor(support, dtype=torch.long, device=device) if support else None
    return targets, support_targets, anchor, phase, anchor_mature


def _append_domain_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    domain_targets: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if domain_targets is None or domain_targets.numel() == 0:
        return target_idx, target_exists
    Pmax = int(max(int(target_idx.max().item()) if target_idx.numel() else 0, int(domain_targets.max().item()))) + 1
    existing = torch.zeros(max(Pmax, 1), dtype=torch.bool, device=target_idx.device)
    if target_idx.numel() > 0:
        existing[target_idx[target_exists].clamp(0, existing.shape[0] - 1)] = True
    extras = domain_targets[~existing[domain_targets.clamp(0, existing.shape[0] - 1)]]
    if extras.numel() == 0:
        return target_idx, target_exists
    extra_exists = torch.ones(extras.shape[0], dtype=torch.bool, device=target_exists.device)
    return torch.cat([target_idx, extras], dim=0), torch.cat([target_exists, extra_exists], dim=0)


def _apply_domain_adjustment(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    prod: Tensor,
    config: ProducerLiteConfig,
    domain_targets: Tensor | None,
    support_targets: Tensor | None,
    anchor_slot: int | None,
    phase: str,
    anchor_mature: bool,
) -> Tensor:
    if domain_targets is None or domain_targets.numel() == 0:
        return score
    P = int(obs.P)
    target_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
    target_mask[domain_targets.clamp(0, P - 1)] = True
    tgt = cand_tgt_slot.clamp(0, P - 1)
    selected = target_mask[tgt] & cand_active.any(dim=-1)
    phase_bonus = 0.0
    if phase == "claim":
        phase_bonus = float(config.domain_bonus)
    elif phase == "consolidate":
        phase_bonus = float(config.domain_support_bonus)
    elif phase == "project":
        phase_bonus = float(config.domain_expansion_bonus)
    score = score + torch.where(
        selected,
        torch.full_like(score, phase_bonus) + (prod[tgt].to(score.dtype) * 0.025).clamp(max=0.25),
        torch.zeros_like(score),
    )

    if support_targets is not None and support_targets.numel() > 0:
        support_mask = torch.zeros(P, dtype=torch.bool, device=score.device)
        support_mask[support_targets.clamp(0, P - 1)] = True
        support_sel = support_mask[tgt] & cand_active.any(dim=-1)
        score = score + torch.where(support_sel, torch.full_like(score, 0.12), torch.zeros_like(score))

    if anchor_slot is not None and 0 <= int(anchor_slot) < P and bool(obs.owned[int(anchor_slot)]) and not anchor_mature:
        src_is_anchor = (cand_src.clamp(0, P - 1) == int(anchor_slot)).any(dim=-1)
        tgt_is_anchor = tgt == int(anchor_slot)
        penalty = src_is_anchor & ~tgt_is_anchor & cand_active.any(dim=-1)
        score = score - torch.where(penalty, torch.full_like(score, float(config.domain_source_penalty)), torch.zeros_like(score))
    return score


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
    memory,
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
    domain_targets, domain_support, domain_anchor, domain_phase, domain_mature = _domain_plan_targets(
        memory=memory,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        prod=prod,
        source_mask=source_mask,
        config=config,
        player_count=player_count,
    )
    target_idx, target_exists = _append_domain_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        domain_targets=domain_targets,
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
    score = _apply_domain_adjustment(
        score=score,
        cand_src=cand_src,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        prod=prod,
        config=config,
        domain_targets=domain_targets,
        support_targets=domain_support,
        anchor_slot=domain_anchor,
        phase=domain_phase,
        anchor_mature=domain_mature,
    )

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    if enemy_mass is None:
        enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


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

CONFIG_4P_DOMAIN_LANE_ANCHOR = dataclasses.replace(
    CONFIG_4P_LANE_ANCHOR,
    enable_domain_plan_4p=True,
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
        return "domain_lane_anchor"

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
    if mode == "domain_lane_anchor":
        return CONFIG_4P_DOMAIN_LANE_ANCHOR
    if mode == "lane_anchor":
        return CONFIG_4P_LANE_ANCHOR
    if mode == "s8_burst":
        return CONFIG_4P_S8_BURST
    return CONFIG_4P_S7_STABLE


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.strategy_mode: str | None = None
        self.domain_plan: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.strategy_mode = None
        self.domain_plan = None


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
            mem.domain_plan = None
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
