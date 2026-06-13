
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
    LaunchEntries,
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
    _stable_argmax,
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


def _tier_candidates(
    *,
    movement: PlanetMovement,
    obs,
    cache,
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
    H: int,
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

    tgt_safe_idx = target_idx.clamp(0, int(obs.P) - 1)
    tgt_prod = prod[tgt_safe_idx].to(dtype)
    needs_risk_check = (
        obs.is_enemy[tgt_safe_idx] | (obs.is_neutral[tgt_safe_idx] & (tgt_prod >= 4.0))
    ) & target_exists
    if bool(needs_risk_check.any()):
        P = int(obs.P)
        enemy_mask = obs.is_enemy & obs.alive
        enemy_ships = torch.where(enemy_mask, obs.ships.to(dtype), torch.zeros(P, dtype=dtype, device=device))
        dist_to_targets = cache.cross_dist[0, :, tgt_safe_idx]
        threat_threshold = 30.0 + tgt_prod.view(1, T) * dist_to_targets
        is_close = dist_to_targets <= float(H) * 1.5
        eye = torch.eye(P, device=device, dtype=torch.bool)
        is_not_target = ~eye[:, tgt_safe_idx]
        is_threat = is_close & (enemy_ships.view(P, 1) > threat_threshold) & is_not_target
        is_risky = needs_risk_check & is_threat.any(dim=0)
    else:
        is_risky = torch.zeros(T, dtype=torch.bool, device=device)

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
        & ~is_risky.view(1, T)
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


def _counter_snipe_pass(
    obs,
    obs_tensors,
    movement,
    garrison_status,
    leftover: Tensor,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Send a small follow-up when a valuable neutral is about to flip enemy."""
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H = int(config.horizon)
    min_send = float(config.min_ships_to_launch)

    owner_traj = garrison_status.owner
    H_axis = int(owner_traj.shape[-1])
    if H_axis <= 1:
        return _empty_entries(device, dtype), leftover

    currently_neutral = obs.is_neutral & obs.alive
    owner_future = owner_traj[:, 1:H_axis]
    is_enemy_future = (owner_future >= 0) & (owner_future != pid)
    neutral_flips = currently_neutral.unsqueeze(-1) & is_enemy_future
    any_flip = neutral_flips.any(dim=-1)

    prod_f = prod.to(dtype)
    cs_mask = any_flip & (prod_f >= 3.0)
    if not bool(cs_mask.any()):
        return _empty_entries(device, dtype), leftover

    first_flip_idx = neutral_flips.to(torch.long).argmax(dim=-1)
    flip_turn = first_flip_idx + 1
    flip_turn = torch.where(any_flip, flip_turn, torch.full_like(flip_turn, H + 1))

    cs_idx = torch.where(cs_mask)[0]
    T_cs = int(cs_idx.shape[0])
    if T_cs == 0:
        return _empty_entries(device, dtype), leftover

    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype), leftover

    S_cap = max(1, min(8, P))
    cs_src_idx, cs_src_exists = _candidate_indices(leftover, src_mask, S_cap)
    S = int(cs_src_idx.shape[0])

    cs_floor = capture_floor(
        garrison_status,
        target_idx=cs_idx,
        k_max=H,
        capture_overhead=1.0,
        player_id=pid,
    )
    K = int(cs_floor.shape[-1])
    if K == 0:
        return _empty_entries(device, dtype), leftover

    flip_k = flip_turn[cs_idx]
    k_range = torch.arange(1, K + 1, device=device).view(1, K)
    after_flip = k_range >= flip_k.view(T_cs, 1)
    floor_masked = torch.where(after_flip, cs_floor, torch.full_like(cs_floor, float("inf")))
    min_floor, _ = floor_masked.min(dim=-1)

    has_opportunity = torch.isfinite(min_floor)
    if not bool(has_opportunity.any()):
        return _empty_entries(device, dtype), leftover

    sniper_size = (min_floor + 2.0).ceil().clamp(min=min_send)

    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    source_ships = leftover[cs_src_idx.clamp(0, P - 1)].to(dtype)
    drain = safe_drain(
        garrison_status,
        source_idx=cs_src_idx,
        source_ships=source_ships,
        H_eff=H_eff,
        player_id=pid,
    )
    sizes = sniper_size.view(1, T_cs).expand(S, T_cs)
    sizes = sizes.clamp(max=drain.view(S, 1)).floor()

    eta_cap = torch.full((T_cs,), float(H), dtype=dtype, device=device)
    active = reachable_mask(
        movement,
        source_idx=cs_src_idx,
        target_idx=cs_idx,
        fleet_sizes=sizes.unsqueeze(-1),
        eta_cap=eta_cap,
    ).squeeze(-1)

    aim = intercept_angle(movement, cs_src_idx.unsqueeze(1), cs_idx.unsqueeze(0), sizes, active=active)
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T_cs))
    after_flip_viable = eta >= flip_k.view(1, T_cs).to(dtype)

    k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
    floor_at_arr = cs_floor.unsqueeze(0).expand(S, T_cs, K).gather(
        -1, k_arr.unsqueeze(-1)
    ).squeeze(-1)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = cs_src_idx.view(S, 1) != cs_idx.view(1, T_cs)
    valid = (
        viable
        & clears_floor
        & after_flip_viable
        & (sizes >= min_send)
        & (sizes >= sniper_size.view(1, T_cs))
        & src_neq_tgt
        & cs_src_exists.view(S, 1)
        & has_opportunity.view(1, T_cs)
    )
    if not bool(valid.any()):
        return _empty_entries(device, dtype), leftover

    C = S * T_cs
    L = 1
    cs_cand_src = cs_src_idx.view(S, 1).expand(S, T_cs).reshape(C, L)
    cs_cand_tgt = cs_idx.view(1, T_cs).expand(S, T_cs).reshape(C)
    cs_cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cs_cand_ang = angle.reshape(C, L)
    cs_cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cs_cand_act = valid.reshape(C, L)
    cs_cand_val = valid.reshape(C)

    launches = make_launch_set(
        source_slots=cs_cand_src,
        target_slots=cs_cand_tgt.unsqueeze(-1).expand(C, L),
        ships=cs_cand_send,
        eta=cs_cand_eta,
        valid=cs_cand_act & cs_cand_val.unsqueeze(-1),
        player_id=pid,
    )
    cs_score = score_candidates(
        garrison_status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launches,
        player_id=pid,
    )
    cs_score = torch.where(cs_cand_val, cs_score, torch.full_like(cs_score, float("-inf")))

    max_cs = 2
    entries_src, entries_tgt, entries_send = [], [], []
    entries_ang, entries_eta = [], []
    cs_leftover = leftover.clone()
    used_targets = set()

    for _ in range(max_cs):
        best_c = int(_stable_argmax(cs_score).item())
        best_val = cs_score[best_c]
        if not (bool(torch.isfinite(best_val)) and float(best_val.item()) > 1.0):
            break

        s_local = best_c // T_cs
        t_local = best_c % T_cs
        src_slot = cs_src_idx[s_local]
        tgt_slot = cs_idx[t_local]
        send_amt = cs_cand_send[best_c, 0]

        tgt_int = int(tgt_slot.item())
        if tgt_int in used_targets or cs_leftover[src_slot] < send_amt:
            cs_score[best_c] = float("-inf")
            continue

        entries_src.append(src_slot)
        entries_tgt.append(tgt_slot)
        entries_send.append(send_amt)
        entries_ang.append(cs_cand_ang[best_c, 0])
        entries_eta.append(cs_cand_eta[best_c, 0])

        cs_leftover[src_slot] = (cs_leftover[src_slot] - send_amt).clamp(min=0.0)
        used_targets.add(tgt_int)

        same_tgt = cs_cand_tgt == tgt_slot
        cs_score = torch.where(same_tgt, torch.full_like(cs_score, float("-inf")), cs_score)

    if not entries_src:
        return _empty_entries(device, dtype), leftover

    return LaunchEntries(
        source_slots=torch.stack(entries_src).to(torch.long),
        target_slots=torch.stack(entries_tgt).to(torch.long),
        ships=torch.stack(entries_send),
        angle=torch.stack(entries_ang),
        eta=torch.stack(entries_eta),
        valid=torch.ones(len(entries_src), dtype=torch.bool, device=device),
    ), cs_leftover


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

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if int(player_count) <= 2:
        cs_entries, leftover = _counter_snipe_pass(
            obs,
            obs_tensors,
            movement,
            garrison_status,
            leftover,
            prod,
            alive_by_step,
            config,
            player_count,
        )
    else:
        cs_entries = _empty_entries(device, dtype)

    if not bool(config.enable_regroup):
        return concat_launch_entries([wave_entries, cs_entries])
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, cs_entries, regroup_entries])


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
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
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
