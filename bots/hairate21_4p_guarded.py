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
from orbit_lite.geometry import fleet_speed


_ORIGINAL_PLAN_LITE_WAVES = _h14.plan_lite_waves

GUARD_HORIZON = 10.0
GUARD_MIN_SHIPS = 6.0
GUARD_PRESSURE_FRACTION = 0.20
GUARD_MAX_PENALTY = 0.03


def _enemy_pressure_and_eta(obs, cache, *, horizon: float):
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        z = torch.zeros(P, dtype=dtype, device=device)
        return z, torch.full((P,), float("inf"), dtype=dtype, device=device)

    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1.0)).clamp(min=1e-6)
    eta = d0 / speeds.view(P, 1)

    enemy = (
        obs.alive
        & (obs.owner_abs >= 0)
        & (obs.owner_abs != int(obs.player_id))
        & (obs.ships > 0)
    )
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & (eta <= float(horizon))
    decay = (1.0 - eta / float(horizon)).clamp(min=0.0)
    pressure = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay)).sum(dim=0)
    min_eta = torch.where(valid, eta, torch.full_like(eta, float("inf"))).amin(dim=0)
    return pressure, min_eta


def _apply_4p_source_collapse_guard(*, obs, cache, cand_src, cand_send, cand_active, cand_is_def, score):
    if int(obs.P) == 0 or int(score.numel()) == 0:
        return score

    pressure, min_eta = _enemy_pressure_and_eta(obs, cache, horizon=GUARD_HORIZON)
    src = cand_src.clamp(0, int(obs.P) - 1)
    send = torch.where(cand_active, cand_send, torch.zeros_like(cand_send))

    source_before = obs.ships.to(score.dtype)[src]
    source_after = source_before - send
    pressure_at_source = pressure[src].to(score.dtype)
    eta_at_source = min_eta[src].to(score.dtype)
    required = torch.maximum(
        torch.full_like(pressure_at_source, float(GUARD_MIN_SHIPS)),
        pressure_at_source * float(GUARD_PRESSURE_FRACTION),
    )

    # 4P losses often come from opening a source that a nearby third party can hit.
    # Keep this soft: hard vetoes changed too many winning openings in testing.
    unsafe_lane = cand_active & (eta_at_source <= float(GUARD_HORIZON)) & (source_after < required)
    penalized = (~cand_is_def) & unsafe_lane.any(dim=-1)
    if not bool(penalized.any()):
        return score
    shortage = ((required - source_after).clamp(min=0.0) / required.clamp(min=1.0)).amax(dim=-1)
    penalty = (shortage * float(GUARD_MAX_PENALTY)).clamp(max=float(GUARD_MAX_PENALTY))
    guarded = score.clone()
    guarded[penalized] = guarded[penalized] - penalty[penalized]
    return guarded


def _guarded_plan_lite_waves(
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
    score = _apply_4p_source_collapse_guard(
        obs=obs,
        cache=cache,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_active=cand_active,
        cand_is_def=cand_is_def,
        score=score,
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
    enemy_mass = _h14.cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _h14._plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return _h14.concat_launch_entries([wave_entries, regroup_entries])


def agent(obs):
    _h14.plan_lite_waves = _guarded_plan_lite_waves
    return _h14.agent(obs)
