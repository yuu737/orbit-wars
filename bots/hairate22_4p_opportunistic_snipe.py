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
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement_step import LaunchEntries


_ORIGINAL_PLAN_LITE_WAVES = _h14.plan_lite_waves

SNIPE_MAX_BONUS = 0.32
SNIPE_LOW_SHIPS = 24.0
SNIPE_PROD_WEIGHT = 0.030
SNIPE_FLIP_BONUS = 0.12
EXTRA_SNIPE_MAX_SEND_FRACTION = 0.35
EXTRA_SNIPE_MIN_SCORE = 5.2


def _apply_4p_opportunistic_snipe_bonus(
    *,
    obs,
    garrison_status,
    cand_tgt_slot,
    cand_eta,
    cand_is_def,
    score,
):
    if int(obs.P) == 0 or int(score.numel()) == 0:
        return score

    H_axis = int(garrison_status.owner.shape[-1])
    if H_axis <= 1:
        return score

    pid = int(obs.player_id)
    tgt = cand_tgt_slot.clamp(0, int(obs.P) - 1)
    eta = cand_eta[:, 0].ceil().long().clamp(1, H_axis - 1)
    rows = torch.arange(int(score.numel()), dtype=torch.long, device=score.device)

    owner_now = obs.owner_abs[tgt]
    owner_at_eta = garrison_status.owner[tgt, eta].to(score.device)
    ships_at_eta = garrison_status.ships[tgt, eta].to(score.dtype)
    prod = obs.prod[tgt].to(score.dtype)

    not_mine_target = owner_now != float(pid)
    enemy_at_eta = (owner_at_eta >= 0) & (owner_at_eta != pid)
    neutral_becomes_enemy = (owner_now < 0) & enemy_at_eta
    enemy_is_weak = enemy_at_eta & (ships_at_eta <= float(SNIPE_LOW_SHIPS))

    low_ship_factor = ((float(SNIPE_LOW_SHIPS) - ships_at_eta).clamp(min=0.0) / float(SNIPE_LOW_SHIPS))
    prod_factor = (prod * float(SNIPE_PROD_WEIGHT)).clamp(max=0.08)
    bonus = low_ship_factor * float(SNIPE_MAX_BONUS) + prod_factor
    bonus = bonus + neutral_becomes_enemy.to(score.dtype) * float(SNIPE_FLIP_BONUS)
    bonus = bonus.clamp(max=float(SNIPE_MAX_BONUS))

    opportunity = (~cand_is_def) & not_mine_target & enemy_is_weak & torch.isfinite(score)
    if not bool(opportunity.any()):
        return score
    out = score.clone()
    out[rows[opportunity]] = out[rows[opportunity]] + bonus[opportunity]
    return out


def _plan_extra_4p_snipe(
    *,
    movement,
    obs,
    garrison_status,
    leftover,
    wave_entries,
    H: int,
    device,
    dtype,
):
    P = int(obs.P)
    if P == 0:
        return _h14._empty_entries(device, dtype), leftover

    already_targeted = torch.zeros(P, dtype=torch.bool, device=device)
    if int(wave_entries.valid.numel()) > 0 and bool(wave_entries.valid.any()):
        already_targeted[wave_entries.target_slots[wave_entries.valid].clamp(0, P - 1)] = True

    source_mask = obs.owned & obs.alive & (leftover >= 8.0)
    if not bool(source_mask.any()):
        return _h14._empty_entries(device, dtype), leftover

    horizon = max(1, min(int(H), 13))
    owner_future = garrison_status.owner[:, 1 : horizon + 1]
    ships_future = garrison_status.ships[:, 1 : horizon + 1].to(dtype)
    enemy_future = (owner_future >= 0) & (owner_future != int(obs.player_id))
    weak_future = enemy_future & (ships_future <= float(SNIPE_LOW_SHIPS))
    any_weak = weak_future.any(dim=1)

    target_mask = obs.alive & (~obs.owned) & any_weak & (~already_targeted)
    if not bool(target_mask.any()):
        return _h14._empty_entries(device, dtype), leftover

    source_idx = torch.nonzero(source_mask, as_tuple=False).flatten()
    target_idx = torch.nonzero(target_mask, as_tuple=False).flatten()
    S = int(source_idx.numel())
    T = int(target_idx.numel())
    if S == 0 or T == 0:
        return _h14._empty_entries(device, dtype), leftover

    base_send = (leftover[source_idx].to(dtype) * float(EXTRA_SNIPE_MAX_SEND_FRACTION)).floor().clamp(min=4.0)
    sizes = base_send.view(S, 1).expand(S, T)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=torch.ones(S, T, dtype=torch.bool, device=device),
    )
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= float(horizon)) & (source_idx.view(S, 1) != target_idx.view(1, T))
    k = eta.clamp(min=1.0, max=float(horizon)).ceil().long().clamp(1, horizon)
    tgt_grid = target_idx.view(1, T).expand(S, T)
    owner_at_eta = garrison_status.owner[tgt_grid, k]
    ships_at_eta = garrison_status.ships[tgt_grid, k].to(dtype)
    enemy_at_eta = (owner_at_eta >= 0) & (owner_at_eta != int(obs.player_id))
    needed = ships_at_eta + 2.0
    clears = sizes >= needed

    prod = obs.prod[target_idx].to(dtype).view(1, T)
    low_factor = ((float(SNIPE_LOW_SHIPS) - ships_at_eta).clamp(min=0.0) / float(SNIPE_LOW_SHIPS))
    neutral_now = (obs.owner_abs[target_idx].view(1, T) < 0).to(dtype)
    value = prod * 1.15 + low_factor * 3.0 + neutral_now * 1.2 - eta * 0.08
    value = torch.where(viable & enemy_at_eta & clears, value, torch.full_like(value, float("-inf")))
    best = torch.argmax(value.reshape(-1))
    best_score = value.reshape(-1)[best]
    if not bool(torch.isfinite(best_score) & (best_score > float(EXTRA_SNIPE_MIN_SCORE))):
        return _h14._empty_entries(device, dtype), leftover

    s_i = best // T
    t_i = best % T
    src = source_idx[s_i].reshape(1)
    tgt = target_idx[t_i].reshape(1)
    send = sizes[s_i, t_i].reshape(1)
    angle = aim["angle"][s_i, t_i].reshape(1)
    eta_one = eta[s_i, t_i].reshape(1)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    entries = LaunchEntries(
        source_slots=src.to(torch.long),
        target_slots=tgt.to(torch.long),
        ships=send,
        angle=angle,
        eta=eta_one,
        valid=valid,
    )
    new_leftover = leftover.clone()
    new_leftover[src] = (new_leftover[src] - send).clamp(min=0.0)
    return entries, new_leftover


def _snipe_plan_lite_waves(
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
    score = _apply_4p_opportunistic_snipe_bonus(
        obs=obs,
        garrison_status=garrison_status,
        cand_tgt_slot=cand_tgt_slot,
        cand_eta=cand_eta,
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
    extra_snipe_entries, leftover = _plan_extra_4p_snipe(
        movement=movement,
        obs=obs,
        garrison_status=garrison_status,
        leftover=leftover,
        wave_entries=wave_entries,
        H=H,
        device=device,
        dtype=dtype,
    )
    attack_entries = _h14.concat_launch_entries([wave_entries, extra_snipe_entries])

    if not bool(config.enable_regroup):
        return attack_entries
    enemy_mass = _h14.cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _h14._plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return _h14.concat_launch_entries([attack_entries, regroup_entries])


def agent(obs):
    _h14.plan_lite_waves = _snipe_plan_lite_waves
    return _h14.agent(obs)
