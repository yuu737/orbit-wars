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


@dataclass(frozen=True)
class ProducerLiteConfig:
    horizon: int = 16
    max_sources_per_lane: int = 10
    max_offensive_targets: int = 10
    max_defensive_targets: int = 6
    max_waves_per_turn: int = 8
    roi_threshold: float = 1.2
    min_ships_to_launch: float = 3.0
    enable_regroup: bool = True
    max_regroup_time: float = 6.0
    regroup_pressure_delta_min: float = 0.20
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=12,
    max_sources_per_lane=8,
    max_offensive_targets=8,
    max_defensive_targets=4,
    max_waves_per_turn=10,
    roi_threshold=1.0,
    min_ships_to_launch=2.0,
    max_regroup_time=5.0,
    max_regroup_targets_per_source=10,
)


def _config_for_state(player_count: int, obs) -> ProducerLiteConfig:
    base = CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()
    if obs is None:
        return base

    alive_planets = obs.alive
    my_ships = obs.ships[obs.owned & alive_planets].sum().item()
    total_ships = obs.ships[alive_planets].sum().item()
    my_share = my_ships / max(total_ships, 1.0)

    if my_share < 0.25:
        return dataclasses.replace(base, roi_threshold=1.0, max_waves_per_turn=10)
    elif my_share > 0.55:
        return dataclasses.replace(base, roi_threshold=2.0, max_defensive_targets=6)
    
    return base


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
        
    pid = int(player_id)
    H = max(float(horizon), 1e-6)
    
    d0 = cache.cross_dist[0].to(dtype)                                   
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          
    reach_dist = (speeds.view(P, 1) * H).clamp(min=1e-6)                 
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)    
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye               
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib_planets = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    pressure = contrib_planets.sum(dim=0)                                
    
    f_alive = obs.f_alive
    if bool(f_alive.any()):
        f_owner = obs.f_owner.to(torch.long)
        f_enemy = f_alive & (f_owner >= 0) & (f_owner != pid)            
        if bool(f_enemy.any()):
            fx = obs.f_x.to(dtype)[f_enemy]                              
            fy = obs.f_y.to(dtype)[f_enemy]
            fs = obs.f_ships.to(dtype)[f_enemy].clamp(min=1e-6)          
            f_speed = fleet_speed(fs)                                    
            f_reach = (f_speed * H).clamp(min=1e-6)                      
            tx = obs.x.to(dtype).view(1, P)                              
            ty = obs.y.to(dtype).view(1, P)
            dxe = fx.view(-1, 1) - tx                                    
            dye = fy.view(-1, 1) - ty
            d_ft = torch.sqrt((dxe * dxe + dye * dye).clamp(min=0.0))    
            decay_f = (1.0 - d_ft / f_reach.view(-1, 1)).clamp(min=0.0)  
            
            if hasattr(obs, 'f_dest_x'):
                fdx = (obs.f_dest_x.to(dtype)[f_enemy] - fx)   
                fdy = (obs.f_dest_y.to(dtype)[f_enemy] - fy)
                f_norm = torch.sqrt(fdx**2 + fdy**2).clamp(min=1e-6)
                fdx, fdy = fdx / f_norm, fdy / f_norm
                to_t_x = tx - fx.view(-1, 1)   
                to_t_y = ty - fy.view(-1, 1)
                dot = fdx.view(-1, 1) * to_t_x + fdy.view(-1, 1) * to_t_y  
                aimed = (dot > 0).to(dtype)
            else:
                aimed = torch.ones(fs.shape[0], P, device=device, dtype=dtype)
            
            tgt_alive = obs.alive.view(1, P)                             
            decay_f = torch.where(tgt_alive, decay_f, torch.zeros_like(decay_f))
            contrib_fleets = fs.view(-1, 1) * decay_f * aimed            
            pressure = pressure + contrib_fleets.sum(dim=0)
            
    return pressure


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
    K = int(floor.shape[-1])

    sizes_max = drain.view(S, 1).expand(S, T)
    floor_t0 = floor[:, 0].clamp(min=1.0)
    sizes_sniper = (floor_t0.view(1, T) + 2.0).expand(S, T)
    sizes_sniper = sizes_sniper.clamp(max=sizes_max)
    sizes = torch.stack([sizes_max, sizes_sniper], dim=-1).view(S, T * 2).floor()

    target_idx_2 = target_idx.unsqueeze(-1).expand(T, 2).reshape(-1)
    target_exists_2 = target_exists.unsqueeze(-1).expand(T, 2).reshape(-1)
    eta_cap_2 = eta_cap.unsqueeze(-1).expand(T, 2).reshape(-1)
    
    if K > 0:
        floor_2 = floor.unsqueeze(1).expand(T, 2, K).reshape(T * 2, K)

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx_2,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap_2,
    ).squeeze(-1)
    
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx_2.unsqueeze(0),
        sizes,
        active=active,
    )
    
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap_2.view(1, T * 2))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor_2.unsqueeze(0).expand(S, T * 2, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T * 2, dtype=dtype, device=device)
        
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx_2.view(1, T * 2)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists_2.view(1, T * 2)
    )

    L = 1
    C = S * T * 2
    cand_src = source_idx.view(S, 1).expand(S, T * 2).reshape(C, L)
    cand_tgt_slot = target_idx_2.view(1, T * 2).expand(S, T * 2).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).unsqueeze(-1).expand(T, 2).reshape(1, T * 2).expand(S, T * 2).reshape(C)
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
            
        obs_for_config = parse_obs(obs_tensors)
        config = _config_for_state(mem.cached_player_count, obs_for_config)
        
        row = run_turn(
            obs_tensors, 
            config=config,
            player_count=int(mem.cached_player_count), 
            memory=mem,
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
