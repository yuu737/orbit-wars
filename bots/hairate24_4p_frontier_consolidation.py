from __future__ import annotations

import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import hairate14_response_search as _h14
from orbit_lite.movement_step import LaunchEntries
from orbit_lite.planner_core import is_comet_planet


_ORIGINAL_PLAN_LITE_WAVES = _h14.plan_lite_waves

FRONTIER_PROD_WEIGHT = 1.4
THIN_ANCHOR_WEIGHT = 2.2
ENEMY_FRONTIER_WEIGHT = 0.8
ISOLATION_WEIGHT = 0.7
MIN_ANCHOR_PROD = 2.0
ANCHOR_HORIZON = 30.0
FRIEND_LINK_DISTANCE = 34.0
FRONTIER_DISTANCE = 44.0
EXTRA_CONSOLIDATION_START = 120
EXTRA_CONSOLIDATION_MAX_TIME = 6.0
EXTRA_CONSOLIDATION_FRACTION = 0.30
EXTRA_CONSOLIDATION_MIN_SCORE = 45.0


def _frontier_pressure(obs, obs_tensors: dict, cache, *, horizon: float, player_id: int):
    base = _h14.cheap_enemy_pressure(obs, cache, horizon=float(horizon), player_id=int(player_id))
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return base

    step = int(obs_tensors.get("step", torch.zeros(1, device=device)).reshape(-1)[0].item())
    if step < 120:
        return base

    owned = obs.owned & obs.alive
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    anchor = owned & (prod >= float(MIN_ANCHOR_PROD))
    value = prod * float(ANCHOR_HORIZON)

    thin_threshold = prod * 8.0 + 12.0
    thin_factor = ((thin_threshold - ships).clamp(min=0.0) / thin_threshold.clamp(min=1.0))
    prod_score = value * float(FRONTIER_PROD_WEIGHT)
    thin_score = value * thin_factor * float(THIN_ANCHOR_WEIGHT)

    d0 = cache.cross_dist[0].to(dtype)
    non_mine = obs.alive & ~owned
    if bool(non_mine.any()):
        masked = torch.where(
            non_mine.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        )
        nearest_non_mine = masked.amin(dim=0)
        frontier_factor = (1.0 - nearest_non_mine / float(FRONTIER_DISTANCE)).clamp(min=0.0, max=1.0)
    else:
        frontier_factor = torch.zeros(P, dtype=dtype, device=device)
    frontier_score = value * frontier_factor * float(ENEMY_FRONTIER_WEIGHT)

    friend_near = (d0 <= float(FRIEND_LINK_DISTANCE)) & owned.view(P, 1) & owned.view(1, P)
    friend_near = friend_near & ~torch.eye(P, dtype=torch.bool, device=device)
    friend_count = friend_near.sum(dim=0).to(dtype)
    isolation_factor = (1.0 / (1.0 + friend_count)).clamp(max=1.0)
    isolation_score = value * isolation_factor * float(ISOLATION_WEIGHT)

    frontier = prod_score + thin_score + frontier_score + isolation_score
    frontier = torch.where(anchor, frontier, torch.zeros_like(frontier))

    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        frontier = torch.where(comet, frontier * 0.25, frontier)

    remaining = max(0, _h14.TOTAL_STEPS - step)
    if remaining < 90:
        frontier = frontier * max(0.20, float(remaining) / 90.0)

    # Kept for diagnostics/experiments. The production branch uses the safer
    # one-shot consolidation layer below because pressure replacement was too
    # invasive in early tests.
    return base + frontier * 0.05


def _debit_entries(leftover, entries):
    if int(entries.valid.numel()) == 0 or not bool(entries.valid.any()):
        return leftover
    debit = torch.zeros_like(leftover)
    debit.scatter_add_(
        0,
        entries.source_slots[entries.valid].clamp(0, int(leftover.shape[0]) - 1),
        entries.ships[entries.valid],
    )
    return (leftover - debit).clamp(min=0.0)


def _plan_extra_frontier_consolidation(
    *,
    movement,
    obs,
    obs_tensors: dict,
    garrison_status,
    leftover,
    H: int,
    config,
    device,
    dtype,
):
    step = int(obs_tensors.get("step", torch.zeros(1, device=device)).reshape(-1)[0].item())
    if step < int(EXTRA_CONSOLIDATION_START) or int(obs.P) == 0:
        return _h14._empty_entries(device, dtype), leftover

    P = int(obs.P)
    owned = obs.owned & obs.alive
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    threshold = prod * 8.0 + 12.0
    deficit = (threshold - ships).clamp(min=0.0)

    comet = is_comet_planet(obs_tensors, P, device)
    dst_mask = owned & (prod >= float(MIN_ANCHOR_PROD)) & (deficit >= 8.0)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    if not bool(dst_mask.any()):
        return _h14._empty_entries(device, dtype), leftover

    reserve = torch.maximum(torch.full_like(ships, 12.0), prod * 6.0)
    surplus = (leftover.to(dtype) - reserve).clamp(min=0.0)
    src_mask = owned & (surplus >= float(config.min_ships_to_launch))
    if not bool(src_mask.any()):
        return _h14._empty_entries(device, dtype), leftover

    src_idx, src_exists = _h14._candidate_indices(surplus, src_mask, max(1, min(6, P)))
    dst_score = deficit + prod * 12.0
    dst_idx, dst_exists = _h14._candidate_indices(dst_score, dst_mask, max(1, min(8, P)))
    S = int(src_idx.numel())
    T = int(dst_idx.numel())
    if S == 0 or T == 0:
        return _h14._empty_entries(device, dtype), leftover

    src_surplus = surplus[src_idx.clamp(0, P - 1)].view(S, 1)
    dst_need = (deficit[dst_idx.clamp(0, P - 1)] + prod[dst_idx.clamp(0, P - 1)] * 4.0).view(1, T)
    sizes = torch.minimum(src_surplus * float(EXTRA_CONSOLIDATION_FRACTION), dst_need).floor().clamp(min=0.0)
    active = sizes >= float(config.min_ships_to_launch)
    reachable = _h14.reachable_mask(
        movement,
        source_idx=src_idx,
        target_idx=dst_idx,
        fleet_sizes=sizes.unsqueeze(-1).clamp(min=1.0),
        eta_cap=torch.full((T,), float(EXTRA_CONSOLIDATION_MAX_TIME), dtype=dtype, device=device),
    ).squeeze(-1)
    aim = _h14.intercept_angle(
        movement,
        src_idx.unsqueeze(1),
        dst_idx.unsqueeze(0),
        sizes.clamp(min=1.0),
        active=active & reachable,
    )
    eta = aim["eta"]
    viable = aim["viable"] & active & reachable & (eta <= float(EXTRA_CONSOLIDATION_MAX_TIME))
    viable = viable & (src_idx.view(S, 1) != dst_idx.view(1, T))

    owner = garrison_status.owner
    H_axis = int(owner.shape[-1])
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)
    dst_owner = owner[dst_idx.clamp(0, P - 1)]
    owner_at_k = dst_owner.unsqueeze(0).expand(S, T, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    viable = viable & (owner_at_k == int(obs.player_id)) & src_exists.view(S, 1) & dst_exists.view(1, T)

    score = (
        deficit[dst_idx.clamp(0, P - 1)].view(1, T)
        + prod[dst_idx.clamp(0, P - 1)].view(1, T) * 10.0
        - eta * 1.5
    )
    score = torch.where(viable, score, torch.full_like(score, float("-inf")))
    flat = score.reshape(-1)
    best = torch.argmax(flat)
    best_score = flat[best]
    if not bool(torch.isfinite(best_score) & (best_score > float(EXTRA_CONSOLIDATION_MIN_SCORE))):
        return _h14._empty_entries(device, dtype), leftover

    s_i = best // T
    t_i = best % T
    src = src_idx[s_i].reshape(1)
    tgt = dst_idx[t_i].reshape(1)
    send = sizes[s_i, t_i].reshape(1)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    entries = LaunchEntries(
        source_slots=src.to(torch.long),
        target_slots=tgt.to(torch.long),
        ships=send,
        angle=aim["angle"][s_i, t_i].reshape(1),
        eta=eta[s_i, t_i].reshape(1),
        valid=valid,
    )
    new_leftover = leftover.clone()
    new_leftover[src] = (new_leftover[src] - send).clamp(min=0.0)
    return entries, new_leftover


def _frontier_plan_lite_waves(
    *,
    movement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod,
    alive_by_step,
    config,
    player_count: int,
):
    if int(player_count) < 4:
        return _ORIGINAL_PLAN_LITE_WAVES(
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            garrison_status=garrison_status,
            prod=prod,
            alive_by_step=alive_by_step,
            config=config,
            player_count=int(player_count),
        )

    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))
    step = int(obs_tensors.get("step", torch.zeros(1, device=device)).reshape(-1)[0].item())

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _h14._empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _h14._candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = _h14.build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _h14._empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = _h14.safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    floor = _h14.capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )

    tier_parts = [
        _h14._tier_candidates(
            movement=movement,
            obs=obs,
            cache=cache,
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
            H=H,
            config=config,
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
    score = _h14._apply_response_search(
        movement=movement,
        obs=obs,
        cache=cache,
        garrison_status=garrison_status,
        prod=prod,
        alive_by_step=alive_by_step,
        config=config,
        player_count=int(player_count),
        step=int(step),
        K_eta=int(K_eta),
        H=int(H),
        cand_src=cand_src,
        cand_send=cand_send,
        cand_angle=cand_angle,
        cand_eta=cand_eta,
        cand_active=cand_active,
        cand_tgt_slot=cand_tgt_slot,
        cand_is_def=cand_is_def,
        score=score,
        device=device,
        dtype=dtype,
    )

    wave_entries, leftover = _h14._greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    pressure = _h14.cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _h14._plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=pressure,
        config=config, H=H,
    )
    leftover_after_regroup = _debit_entries(leftover, regroup_entries)
    extra_entries, _ = _plan_extra_frontier_consolidation(
        movement=movement,
        obs=obs,
        obs_tensors=obs_tensors,
        garrison_status=garrison_status,
        leftover=leftover_after_regroup,
        H=H,
        config=config,
        device=device,
        dtype=dtype,
    )
    return _h14.concat_launch_entries([wave_entries, regroup_entries, extra_entries])


def agent(obs):
    _h14.plan_lite_waves = _frontier_plan_lite_waves
    return _h14.agent(obs)
