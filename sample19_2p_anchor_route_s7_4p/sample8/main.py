from __future__ import annotations

import dataclasses
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
    # --- 2P anchor route planner -------------------------------------------
    anchor_route_enabled: bool = False
    anchor_turns: int = 125
    anchor_select_turns: int = 16
    anchor_near_dist: float = 430.0
    anchor_link_dist: float = 280.0
    anchor_expand_dist: float = 640.0
    anchor_extra_targets: int = 8
    anchor_min_prod: float = 2.0
    anchor_min_ships: float = 28.0
    anchor_min_score: float = 24.0
    anchor_hold_prod_mult: float = 8.0
    anchor_hold_base: float = 28.0
    anchor_expand_ready_mult: float = 0.85
    anchor_big_prod: float = 3.0
    anchor_big_ships: float = 45.0
    anchor_score_bonus: float = 12.0
    anchor_local_bonus: float = 7.0
    anchor_expand_bonus: float = 10.0
    anchor_regroup_weight: float = 2.2
    anchor_regroup_radius: float = 340.0


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


def _merge_extra_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    extra_idx: Tensor,
    P: int,
) -> tuple[Tensor, Tensor]:
    if int(extra_idx.numel()) == 0:
        return target_idx, target_exists
    base = target_idx[target_exists]
    seen: set[int] = set()
    merged: list[int] = []
    for value in base.detach().cpu().tolist() + extra_idx.detach().cpu().tolist():
        ivalue = int(value)
        if 0 <= ivalue < int(P) and ivalue not in seen:
            seen.add(ivalue)
            merged.append(ivalue)
    if not merged:
        return target_idx, target_exists
    out = torch.tensor(merged, dtype=target_idx.dtype, device=target_idx.device)
    return out, torch.ones(int(out.numel()), dtype=torch.bool, device=target_idx.device)


def _select_anchor_plan(
    *,
    obs,
    cache,
    prod: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    step: int,
) -> tuple[int, tuple[int, ...], float]:
    if (
        not bool(config.anchor_route_enabled)
        or int(player_count) != 2
        or int(step) > int(config.anchor_select_turns)
        or int(obs.P) <= 0
        or not bool((obs.owned & obs.alive).any())
    ):
        return -1, (), 0.0

    P = int(obs.P)
    dtype = obs.ships.dtype
    device = obs.device
    d0 = cache.cross_dist[0].to(dtype)
    own_mask = obs.owned & obs.alive
    enemy_mask = obs.is_enemy & obs.alive
    own_dist = torch.where(own_mask.view(P, 1), d0, torch.full_like(d0, float("inf"))).amin(dim=0)
    if bool(enemy_mask.any()):
        enemy_dist = torch.where(enemy_mask.view(P, 1), d0, torch.full_like(d0, float("inf"))).amin(dim=0)
        enemy_adv = (own_dist - enemy_dist).clamp(min=0.0)
    else:
        enemy_adv = torch.zeros(P, dtype=dtype, device=device)

    valuable = (prod.to(dtype) >= float(config.anchor_min_prod)) | (
        obs.ships.to(dtype) >= float(config.anchor_min_ships)
    )
    anchor_mask = (
        obs.alive
        & obs.is_neutral
        & valuable
        & (own_dist <= float(config.anchor_near_dist))
        & (enemy_adv <= 60.0)
    )
    if not bool(anchor_mask.any()):
        return -1, (), 0.0

    anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).view(-1)
    anchor_to_all = d0[anchor_idx, :]
    linked = anchor_to_all <= float(config.anchor_link_dist)
    local_neutral = linked & obs.is_neutral.view(1, P) & obs.alive.view(1, P)
    local_value = torch.where(
        local_neutral,
        prod.to(dtype).view(1, P) * 3.0 + obs.ships.to(dtype).view(1, P) * 0.04,
        torch.zeros_like(anchor_to_all),
    ).sum(dim=1)
    score = (
        prod[anchor_idx].to(dtype) * 12.0
        + obs.ships[anchor_idx].to(dtype) * 0.055
        + local_value
        - own_dist[anchor_idx].clamp(max=float(config.anchor_near_dist)) * 0.018
        - enemy_adv[anchor_idx] * 0.070
    )
    best_pos = int(torch.argmax(score).item())
    best_score = float(score[best_pos].item())
    if best_score < float(config.anchor_min_score):
        return -1, (), best_score

    anchor_slot = int(anchor_idx[best_pos].item())
    route_pref = torch.where(
        local_neutral[best_pos],
        prod.to(dtype) * 8.0
        + obs.ships.to(dtype) * 0.05
        - d0[anchor_slot, :].to(dtype).clamp(max=float(config.anchor_link_dist)) * 0.012,
        torch.full((P,), float("-inf"), dtype=dtype, device=device),
    )
    cap = max(1, min(int(config.anchor_extra_targets), P))
    route_idx, route_exists = _candidate_indices(route_pref, torch.isfinite(route_pref), cap)
    route = tuple(int(v) for v in route_idx[route_exists].detach().cpu().tolist())
    return anchor_slot, route, best_score


def _anchor_source_mask(
    *,
    obs,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    step: int,
    anchor_slot: int,
) -> Tensor:
    if (
        not bool(config.anchor_route_enabled)
        or int(player_count) != 2
        or int(step) > int(config.anchor_turns)
        or int(anchor_slot) < 0
        or int(anchor_slot) >= int(obs.P)
        or not bool(obs.owned[int(anchor_slot)])
    ):
        return source_mask
    hold_floor = obs.prod[int(anchor_slot)] * float(config.anchor_hold_prod_mult) + float(config.anchor_hold_base)
    if float(obs.ships[int(anchor_slot)].item()) >= float(hold_floor.item()):
        return source_mask
    guarded = source_mask.clone()
    guarded[int(anchor_slot)] = False
    return guarded if bool(guarded.any()) else source_mask


def _add_anchor_route_targets(
    *,
    obs,
    cache,
    prod: Tensor,
    target_idx: Tensor,
    target_exists: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    step: int,
    anchor_slot: int,
    route_targets: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    P = int(obs.P)
    if (
        not bool(config.anchor_route_enabled)
        or int(player_count) != 2
        or int(step) > int(config.anchor_turns)
        or int(anchor_slot) < 0
        or int(anchor_slot) >= P
    ):
        return target_idx, target_exists

    dtype = obs.ships.dtype
    device = obs.device
    d_anchor = cache.cross_dist[0].to(dtype)[int(anchor_slot), :]
    pref = torch.full((P,), float("-inf"), dtype=dtype, device=device)
    if bool(obs.is_neutral[int(anchor_slot)]):
        pref[int(anchor_slot)] = 120.0 + prod[int(anchor_slot)].to(dtype) * 10.0

    local = (
        obs.is_neutral
        & obs.alive
        & (d_anchor <= float(config.anchor_link_dist))
        & ((prod.to(dtype) >= 1.0) | (obs.ships.to(dtype) >= 8.0))
    )
    local_score = (
        prod.to(dtype) * 8.0
        + obs.ships.to(dtype) * 0.045
        - d_anchor.clamp(max=float(config.anchor_link_dist)) * 0.012
        + 18.0
    )
    pref = torch.where(local, torch.maximum(pref, local_score), pref)

    if route_targets:
        route_tensor = torch.tensor([v for v in route_targets if 0 <= int(v) < P], dtype=torch.long, device=device)
        if int(route_tensor.numel()) > 0:
            pref[route_tensor] = torch.maximum(pref[route_tensor], local_score[route_tensor] + 8.0)

    anchor_owned = bool(obs.owned[int(anchor_slot)])
    hold_floor = prod[int(anchor_slot)].to(dtype) * float(config.anchor_hold_prod_mult) + float(config.anchor_hold_base)
    anchor_ready = anchor_owned and float(obs.ships[int(anchor_slot)].item()) >= float((hold_floor * float(config.anchor_expand_ready_mult)).item())
    if anchor_ready or int(step) >= 60:
        expand = (
            (obs.is_neutral | obs.is_enemy)
            & obs.alive
            & (d_anchor <= float(config.anchor_expand_dist))
            & ((prod.to(dtype) >= float(config.anchor_big_prod)) | (obs.ships.to(dtype) >= float(config.anchor_big_ships)))
        )
        expand_score = (
            prod.to(dtype) * 9.0
            + obs.ships.to(dtype) * 0.04
            - d_anchor.clamp(max=float(config.anchor_expand_dist)) * 0.010
            + 20.0
        )
        pref = torch.where(expand, torch.maximum(pref, expand_score), pref)

    valid = torch.isfinite(pref)
    if not bool(valid.any()):
        return target_idx, target_exists
    cap = max(1, min(int(config.anchor_extra_targets), P))
    extra_idx, extra_exists = _candidate_indices(pref, valid, cap)
    return _merge_extra_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        extra_idx=extra_idx[extra_exists],
        P=P,
    )


def _anchor_candidate_bonus(
    *,
    obs,
    cache,
    prod: Tensor,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    step: int,
    anchor_slot: int,
    route_targets: tuple[int, ...],
) -> Tensor:
    if (
        not bool(config.anchor_route_enabled)
        or int(player_count) != 2
        or int(step) > int(config.anchor_turns)
        or int(anchor_slot) < 0
        or int(anchor_slot) >= int(obs.P)
    ):
        return torch.zeros_like(cand_tgt_short, dtype=prod.dtype)
    P = int(obs.P)
    dtype = prod.dtype
    device = prod.device
    d_anchor = cache.cross_dist[0].to(dtype)[int(anchor_slot), :]
    tgt = target_idx[cand_tgt_short].clamp(0, P - 1)
    bonus_by_target = torch.zeros(P, dtype=dtype, device=device)
    if bool(obs.is_neutral[int(anchor_slot)]):
        bonus_by_target[int(anchor_slot)] = float(config.anchor_score_bonus)
    local = obs.is_neutral & obs.alive & (d_anchor <= float(config.anchor_link_dist))
    bonus_by_target = torch.where(
        local,
        torch.maximum(bonus_by_target, torch.full_like(bonus_by_target, float(config.anchor_local_bonus)) + prod.to(dtype) * 1.5),
        bonus_by_target,
    )
    if route_targets:
        route_tensor = torch.tensor([v for v in route_targets if 0 <= int(v) < P], dtype=torch.long, device=device)
        if int(route_tensor.numel()) > 0:
            bonus_by_target[route_tensor] = torch.maximum(
                bonus_by_target[route_tensor],
                torch.full((int(route_tensor.numel()),), float(config.anchor_local_bonus) + 5.0, dtype=dtype, device=device),
            )
    anchor_owned = bool(obs.owned[int(anchor_slot)])
    hold_floor = prod[int(anchor_slot)].to(dtype) * float(config.anchor_hold_prod_mult) + float(config.anchor_hold_base)
    anchor_ready = anchor_owned and float(obs.ships[int(anchor_slot)].item()) >= float((hold_floor * float(config.anchor_expand_ready_mult)).item())
    if anchor_ready or int(step) >= 60:
        expand = (
            (obs.is_neutral | obs.is_enemy)
            & obs.alive
            & (d_anchor <= float(config.anchor_expand_dist))
            & ((prod.to(dtype) >= float(config.anchor_big_prod)) | (obs.ships.to(dtype) >= float(config.anchor_big_ships)))
        )
        bonus_by_target = torch.where(
            expand,
            torch.maximum(bonus_by_target, torch.full_like(bonus_by_target, float(config.anchor_expand_bonus)) + prod.to(dtype)),
            bonus_by_target,
        )
    return bonus_by_target[tgt]


def _anchor_regroup_pressure(
    *,
    obs,
    cache,
    pressure: Tensor | None,
    config: ProducerLiteConfig,
    player_count: int,
    step: int,
    anchor_slot: int,
) -> Tensor | None:
    if pressure is None:
        return None
    P = int(obs.P)
    if (
        not bool(config.anchor_route_enabled)
        or int(player_count) != 2
        or int(step) > int(config.anchor_turns)
        or int(anchor_slot) < 0
        or int(anchor_slot) >= P
        or not bool(obs.owned[int(anchor_slot)])
    ):
        return pressure
    dtype = pressure.dtype
    d_anchor = cache.cross_dist[0].to(dtype)[int(anchor_slot), :]
    cluster = obs.owned & obs.alive & (d_anchor <= float(config.anchor_regroup_radius))
    hold_floor = obs.prod.to(dtype) * float(config.anchor_hold_prod_mult) + float(config.anchor_hold_base)
    thin = (hold_floor - obs.ships.to(dtype)).clamp(min=0.0)
    bonus = torch.zeros_like(pressure)
    bonus[int(anchor_slot)] = float(config.anchor_regroup_weight) * (thin[int(anchor_slot)] + 20.0)
    bonus = torch.where(cluster, torch.maximum(bonus, thin * 0.30 + obs.prod.to(dtype) * 3.0), bonus)
    return pressure + bonus


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
    step = int(obs.step.reshape(-1)[0].item())

    anchor_slot = int(getattr(memory, "anchor_slot", -1)) if memory is not None else -1
    route_targets = tuple(getattr(memory, "anchor_route_targets", ()) or ())
    if memory is not None and bool(config.anchor_route_enabled) and int(player_count) == 2:
        if step == 0 or (anchor_slot < 0 and step <= int(config.anchor_select_turns)):
            anchor_slot, route_targets, anchor_score = _select_anchor_plan(
                obs=obs,
                cache=cache,
                prod=prod,
                config=config,
                player_count=int(player_count),
                step=step,
            )
            memory.anchor_slot = int(anchor_slot)
            memory.anchor_route_targets = tuple(route_targets)
            memory.anchor_selected_step = int(step) if anchor_slot >= 0 else -1
            memory.anchor_score = float(anchor_score)

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    source_mask = _anchor_source_mask(
        obs=obs,
        source_mask=source_mask,
        config=config,
        player_count=int(player_count),
        step=step,
        anchor_slot=anchor_slot,
    )
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    target_idx, target_exists = _add_anchor_route_targets(
        obs=obs,
        cache=cache,
        prod=prod,
        target_idx=target_idx,
        target_exists=target_exists,
        config=config,
        player_count=int(player_count),
        step=step,
        anchor_slot=anchor_slot,
        route_targets=route_targets,
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
    score = score + _anchor_candidate_bonus(
        obs=obs,
        cache=cache,
        prod=prod,
        target_idx=target_idx,
        cand_tgt_short=cand_tgt_short,
        config=config,
        player_count=int(player_count),
        step=step,
        anchor_slot=anchor_slot,
        route_targets=route_targets,
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
    enemy_mass = _anchor_regroup_pressure(
        obs=obs,
        cache=cache,
        pressure=enemy_mass,
        config=config,
        player_count=int(player_count),
        step=step,
        anchor_slot=anchor_slot,
    )
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


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else CONFIG_2P


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.anchor_slot: int = -1
        self.anchor_route_targets: tuple[int, ...] = ()
        self.anchor_selected_step: int = -1
        self.anchor_score: float = 0.0

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.anchor_slot = -1
        self.anchor_route_targets = ()
        self.anchor_selected_step = -1
        self.anchor_score = 0.0


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        current_player = int(obs_tensors["player"].reshape(-1)[0].item())
        min_count = current_player + 1
        mem.cached_player_count = 4 if max(int(mem.cached_player_count), min_count) > 2 else 2
        base = _config_for(mem.cached_player_count)
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
