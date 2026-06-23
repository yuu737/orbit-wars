
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
from orbit_lite.garrison_launch import _run_exact_recurrence
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
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    terminal_phase_turns: int = 40
    terminal_roi_threshold: float = 1.0
    terminal_max_waves_per_turn: int = 9
    terminal_enable_regroup: bool = False
    # --- exp41: evaluate several commit fractions per (src, tgt) ------------
    size_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0)
    # --- sample68 4P portfolio director ------------------------------------
    enable_4p_portfolio: bool = False
    portfolio_turn_limit: int = 180
    portfolio_hold_window: int = 7
    safe_neutral_bonus: float = 0.62
    defense_bonus: float = 0.95
    leader_attack_bonus: float = 0.36
    finish_weak_enemy_bonus: float = 0.54
    nonleader_war_penalty: float = 0.82
    contested_neutral_penalty: float = 0.62
    center_brawl_penalty: float = 0.58
    source_drain_penalty: float = 0.50
    hold_fail_enemy_penalty: float = 1.15
    hold_fail_neutral_penalty: float = 0.34
    response_frac: float = 0.42
    response_margin: float = 8.0


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


def _owner_power(obs, prod: Tensor, *, player_count: int, dtype, device) -> tuple[Tensor, Tensor]:
    ships = torch.zeros(int(player_count), dtype=dtype, device=device)
    production = torch.zeros(int(player_count), dtype=dtype, device=device)
    owner_long = obs.owner_abs.to(torch.long)
    for owner in range(int(player_count)):
        mask = obs.alive & (owner_long == owner)
        if bool(mask.any()):
            ships[owner] = obs.ships.to(dtype)[mask].sum()
            production[owner] = prod.to(dtype)[mask].sum()
    return ships, production


def _candidate_hold_ok(
    *,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    hold_window: int,
) -> tuple[Tensor, Tensor]:
    P = int(obs.P)
    H = int(garrison_status.owner.shape[-1]) - 1
    C = int(cand_tgt_slot.shape[0])
    device = cand_tgt_slot.device
    dtype = cand_send.dtype
    if P <= 0 or H <= 0 or C <= 0 or garrison_status.arrivals_by_owner is None:
        return cand_active.any(dim=-1), torch.zeros(C, dtype=dtype, device=device)

    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)
    active = cand_active.any(dim=-1)
    k_arr = torch.ceil(eta).to(torch.long).clamp(min=1, max=H)
    h_idx = (k_arr - 1).clamp(min=0, max=H - 1)
    pid = int(obs.player_id)
    A = int(player_count)

    arrivals = garrison_status.arrivals_by_owner[tgt, 1:, :].to(dtype).clone()
    rows = torch.arange(C, device=device)
    valid_launch = active & (send > 0.0)
    rows_v = rows[valid_launch]
    if rows_v.numel() > 0:
        arrivals[rows_v, h_idx[valid_launch], pid] += send[valid_launch]

    init_owner = garrison_status.owner[tgt, 0].to(torch.long)
    init_ships = garrison_status.ships[tgt, 0].to(dtype)
    prod_t = prod[tgt].to(dtype)
    alive_t = alive_by_step[:, tgt].transpose(0, 1).to(torch.bool)
    owner_t, ships_t, _po, _ps = _run_exact_recurrence(
        init_owner=init_owner.unsqueeze(1),
        init_ships=init_ships.unsqueeze(1),
        prod=prod_t.unsqueeze(1),
        alive=alive_t.unsqueeze(1),
        arrivals=arrivals.unsqueeze(1),
    )
    owner_t = owner_t.squeeze(1)
    ships_t = ships_t.squeeze(1)
    owner_at_arrival = owner_t.gather(1, k_arr.view(C, 1)).squeeze(1)
    mine_at_arrival = owner_at_arrival == pid
    steps = torch.arange(H + 1, device=device).view(1, H + 1)
    end_k = (k_arr + int(hold_window)).clamp(max=H)
    window = (steps >= k_arr.view(C, 1)) & (steps <= end_k.view(C, 1))
    lost = ((owner_t != pid) & window).any(dim=1)
    min_ships = torch.where(
        window & (owner_t == pid),
        ships_t,
        torch.full_like(ships_t, float("inf")),
    ).min(dim=1).values
    min_ships = torch.where(torch.isfinite(min_ships), min_ships, torch.zeros_like(min_ships))
    return active & mine_at_arrival & (~lost), min_ships


def _apply_4p_portfolio_score(
    *,
    score: Tensor,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_tgt_slot: Tensor,
    cand_active: Tensor,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    if int(player_count) < 4 or not bool(config.enable_4p_portfolio):
        return score
    step = int(obs_tensors["step"].reshape(-1)[0].item())
    if step > int(config.portfolio_turn_limit):
        return score
    P = int(obs.P)
    if P <= 0:
        return score

    dtype = score.dtype
    device = score.device
    pid = int(obs.player_id)
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype).clamp(min=1.0)
    active = cand_active.any(dim=-1)
    target_owner = obs.owner_abs.to(torch.long)[tgt]

    ship_power, prod_power = _owner_power(obs, prod, player_count=int(player_count), dtype=dtype, device=device)
    total_power = ship_power + prod_power * 24.0
    enemy_power = total_power.clone()
    enemy_power[pid] = -1.0
    leader = int(torch.argmax(enemy_power).item())
    weakest_enemy_power = enemy_power[enemy_power >= 0.0].min() if bool((enemy_power >= 0.0).any()) else torch.tensor(0.0, dtype=dtype, device=device)
    target_power = torch.zeros_like(score)
    for owner in range(int(player_count)):
        target_power = torch.where(target_owner == owner, total_power[owner].expand_as(score), target_power)

    d0 = cache.cross_dist[0].to(dtype)
    enemy_mask = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)
    my_mask = obs.owned & obs.alive
    enemy_reach = d0[:, tgt].transpose(0, 1) / fleet_speed(obs.ships.to(dtype).clamp(min=1.0)).view(1, P).clamp(min=1e-6)
    my_reach = d0[:, tgt].transpose(0, 1) / fleet_speed(obs.ships.to(dtype).clamp(min=1.0)).view(1, P).clamp(min=1e-6)
    enemy_eta_ok = enemy_reach <= (eta + float(config.portfolio_hold_window)).view(-1, 1)
    my_eta_ok = my_reach <= (eta + 4.0).view(-1, 1)
    enemy_response = torch.where(
        enemy_eta_ok & enemy_mask.view(1, P),
        (obs.ships.to(dtype).view(1, P) * float(config.response_frac)),
        torch.zeros_like(enemy_reach),
    ).sum(dim=1)
    my_support = torch.where(
        my_eta_ok & my_mask.view(1, P),
        obs.ships.to(dtype).view(1, P),
        torch.zeros_like(my_reach),
    ).sum(dim=1)
    enemy_reach_count = (enemy_eta_ok & enemy_mask.view(1, P)).sum(dim=1)

    radius = torch.sqrt((obs.x.to(dtype) - 50.0) ** 2 + (obs.y.to(dtype) - 50.0) ** 2)
    source_after = obs.ships[src].to(dtype) - send
    source_floor = 10.0 + prod[src].to(dtype) * 3.5
    source_drain = source_after < source_floor

    enemy_target = active & obs.is_enemy[tgt]
    neutral_target = active & obs.is_neutral[tgt]
    own_target = active & obs.owned[tgt]

    hold_known = active & (enemy_target | neutral_target)
    hold_ok = torch.ones_like(active, dtype=torch.bool)
    min_hold_ships = send + prod[tgt].to(dtype) * float(config.portfolio_hold_window)
    if garrison_status.arrivals_by_owner is not None:
        H = int(garrison_status.arrivals_by_owner.shape[1]) - 1
        if H > 0:
            k_arr = torch.ceil(eta).to(torch.long).clamp(min=1, max=H)
            end_k = (k_arr + int(config.portfolio_hold_window)).clamp(max=H)
            steps_h = torch.arange(1, H + 1, device=device).view(1, H)
            window = (steps_h >= k_arr.view(-1, 1)) & (steps_h <= end_k.view(-1, 1))
            arr = garrison_status.arrivals_by_owner[tgt, 1:, :].to(dtype)
            owner_ids = torch.arange(int(player_count), device=device).view(1, 1, -1)
            enemy_arr = torch.where(owner_ids != pid, arr, torch.zeros_like(arr)).sum(dim=2)
            my_arr = arr[:, :, pid]
            enemy_window = torch.where(window, enemy_arr, torch.zeros_like(enemy_arr)).sum(dim=1)
            my_window = torch.where(window, my_arr, torch.zeros_like(my_arr)).sum(dim=1)
            min_hold_ships = min_hold_ships + my_window - enemy_window
            hold_ok = (~hold_known) | (min_hold_ships >= -float(config.response_margin))
    response_fail = hold_known & hold_ok & ((enemy_response - min_hold_ships) > float(config.response_margin))

    leader_attack = enemy_target & (target_owner == leader)
    finish_weak_enemy = enemy_target & (target_power <= weakest_enemy_power * 1.15 + 70.0) & (target_power <= total_power[pid] * 0.75)
    nonleader_war = enemy_target & (~leader_attack) & (~finish_weak_enemy) & (step < 150)
    safe_neutral = neutral_target & hold_known & hold_ok & (~response_fail) & (enemy_reach_count <= 1) & (radius[tgt] >= 28.0)
    contested_neutral = neutral_target & ((enemy_reach_count >= 2) | response_fail) & (my_support + send < enemy_response * 1.05 + 8.0)
    center_brawl = (radius[tgt] <= 29.0) & (enemy_reach_count >= 2) & (step < 130)
    defend = own_target & (enemy_response > my_support + obs.ships[tgt].to(dtype) * 0.35)

    bonus = torch.zeros_like(score)
    bonus = bonus + torch.where(safe_neutral, torch.full_like(score, float(config.safe_neutral_bonus)) + prod[tgt].to(dtype) * 0.05, torch.zeros_like(score))
    bonus = bonus + torch.where(defend, torch.full_like(score, float(config.defense_bonus)), torch.zeros_like(score))
    bonus = bonus + torch.where(leader_attack, torch.full_like(score, float(config.leader_attack_bonus)), torch.zeros_like(score))
    bonus = bonus + torch.where(finish_weak_enemy, torch.full_like(score, float(config.finish_weak_enemy_bonus)), torch.zeros_like(score))

    penalty = torch.zeros_like(score)
    penalty = penalty + torch.where(nonleader_war & ((hold_known & ~hold_ok) | response_fail), torch.full_like(score, float(config.nonleader_war_penalty)), torch.zeros_like(score))
    penalty = penalty + torch.where(contested_neutral, torch.full_like(score, float(config.contested_neutral_penalty)), torch.zeros_like(score))
    penalty = penalty + torch.where(center_brawl, torch.full_like(score, float(config.center_brawl_penalty)), torch.zeros_like(score))
    penalty = penalty + torch.where(source_drain & (enemy_target | contested_neutral), ((source_floor - source_after).clamp(min=0.0) / 30.0).clamp(max=1.4) * float(config.source_drain_penalty), torch.zeros_like(score))
    penalty = penalty + torch.where(enemy_target & hold_known & (~hold_ok), torch.full_like(score, float(config.hold_fail_enemy_penalty)), torch.zeros_like(score))
    penalty = penalty + torch.where(neutral_target & hold_known & (~hold_ok), torch.full_like(score, float(config.hold_fail_neutral_penalty)), torch.zeros_like(score))

    hard_block = enemy_target & nonleader_war & ((hold_known & ~hold_ok) | response_fail) & (eta >= 4.0)
    return torch.where(hard_block, torch.full_like(score, float("-inf")), score + bonus - penalty)


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
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
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
    score = _apply_4p_portfolio_score(
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_eta=cand_eta,
        cand_tgt_slot=cand_tgt_slot,
        cand_active=cand_active,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        garrison_status=garrison_status,
        prod=prod,
        alive_by_step=alive_by_step,
        config=config,
        player_count=player_count,
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
    horizon=15,
    max_sources_per_lane=8,
    max_offensive_targets=14,
    max_defensive_targets=4,
    max_waves_per_turn=6,
    roi_threshold=1.42,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    enable_4p_portfolio=True,
    portfolio_turn_limit=185,
    portfolio_hold_window=7,
    safe_neutral_bonus=0.56,
    defense_bonus=0.82,
    leader_attack_bonus=0.28,
    finish_weak_enemy_bonus=0.46,
    nonleader_war_penalty=0.66,
    contested_neutral_penalty=0.48,
    center_brawl_penalty=0.44,
    source_drain_penalty=0.38,
    hold_fail_enemy_penalty=0.84,
    hold_fail_neutral_penalty=0.22,
    response_frac=0.34,
    response_margin=10.0,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


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
