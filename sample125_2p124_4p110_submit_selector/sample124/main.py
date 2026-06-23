
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
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
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- ETA-aware reinforcement risk (capture sizing) -------------------
    # Inflate the capture floor by ``reinforce_size_beta * rho(eta) * C_k`` where
    # C_k is enemy supply reachable to the target during the fleet's flight, so the
    # agent *declines* captures the enemy will reinforce mid-flight instead of
    # sinking its whole garrison into a doomed attack (the flow scorer projects
    # opponents do-nothing, so it can't see reactive reinforcement). ``beta = 0``
    # disables it (bare floor).
    reinforce_size_beta: float = 2.2
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    enable_2p_mid_hold: bool = False
    size_multipliers: tuple[float, ...] = (1.0,)


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources


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


def _apply_2p_midgame_hold_bias(
    *,
    score: Tensor,
    config: ProducerLiteConfig,
    obs,
    prod: Tensor,
    cache,
    cand_src: Tensor,
    cand_send: Tensor,
    cand_eta: Tensor,
    cand_active: Tensor,
    cand_tgt_slot: Tensor,
    cand_is_def: Tensor,
    enemy_mass: Tensor,
    player_count: int,
) -> Tensor:
    if int(player_count) >= 4:
        return score
    if not bool(config.enable_2p_mid_hold):
        return score
    step = int(float(obs.step.reshape(-1)[0].item()))
    if step < 70 or step > 155:
        return score
    center_prod = torch.where(
        obs.alive & (torch.sqrt((obs.x.to(score.dtype) - 50.0) ** 2 + (obs.y.to(score.dtype) - 50.0) ** 2) <= 28.0),
        prod.to(score.dtype),
        torch.zeros_like(prod.to(score.dtype)),
    ).sum()
    if float(center_prod.item()) > 4.0:
        return score
    active = cand_active.any(dim=-1) & torch.isfinite(score)
    if not bool(active.any()):
        return score

    dtype = score.dtype
    P = int(obs.P)
    src = cand_src[:, 0].clamp(0, max(P - 1, 0))
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)

    prod_t = prod[tgt].to(dtype)
    prod_s = prod[src].to(dtype)
    pressure_t = enemy_mass[tgt].to(dtype)
    sx = obs.x[src].to(dtype)
    sy = obs.y[src].to(dtype)
    tx = obs.x[tgt].to(dtype)
    ty = obs.y[tgt].to(dtype)
    src_radius = torch.sqrt((sx - 50.0) ** 2 + (sy - 50.0) ** 2)
    tgt_radius = torch.sqrt((tx - 50.0) ** 2 + (ty - 50.0) ** 2)

    valuable_defense = (
        active
        & cand_is_def
        & (prod_t >= 3.0)
        & (pressure_t >= 14.0)
        & (tgt_radius <= 45.0)
    )
    defense_bonus = (
        0.80
        + prod_t * 0.11
        + pressure_t.clamp(max=90.0) * 0.011
        + (45.0 - tgt_radius).clamp(min=0.0) * 0.014
    )

    src_after = obs.ships[src].to(dtype) - send
    reserve = 16.0 + prod_s * 5.0
    strip_risk = (
        active
        & (~cand_is_def)
        & obs.owned[src]
        & (prod_s >= 3.0)
        & (src_radius <= 46.0)
        & (src_after < reserve)
        & (eta >= 5.0)
    )
    strip_penalty = (
        0.62
        + ((reserve - src_after).clamp(min=0.0) / 21.0).clamp(max=1.35)
        + prod_s * 0.045
    )

    out = score.clone()
    out = out + torch.where(valuable_defense, defense_bonus, torch.zeros_like(score))
    out = out - torch.where(strip_risk, strip_penalty, torch.zeros_like(score))
    return out


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
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
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
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    # Reachable-enemy-mass proxy ([P]) — computed ONCE and reused for BOTH the
    # reinforcement-risk floor margin (below) and the regroup gradient (further
    # down). Its decay distance-scale is the attack reach K_eta.
    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
        if beta > 0.0 or bool(config.enable_regroup) else None
    )

    # ETA-aware reinforcement risk: inflate the capture floor by ``beta * rho(k) *
    # reachable-enemy-mass(target)``. The per-arrival-turn growth comes from the
    # rho(k) timing ramp. Gated by beta > 0.
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]                     # [T]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange, eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )                                                                        # [K_eta]
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)       # [T, K_eta]
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )                                                                            # [T, K]
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
    score = _apply_2p_midgame_hold_bias(
        score=score, config=config, obs=obs, prod=prod, cache=cache,
        cand_src=cand_src, cand_send=cand_send, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot,
        cand_is_def=cand_is_def, enemy_mass=enemy_mass,
        player_count=int(player_count),
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

    # Reuse the enemy-mass proxy already computed above (one [P, P] reduction
    # serves both the reinforcement floor and this regroup gradient).
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
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


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
)

CONFIG_2P_DEFAULT = ProducerLiteConfig()
CONFIG_2P_HOLD = dataclasses.replace(
    ProducerLiteConfig(),
    enable_2p_mid_hold=True,
)
CONFIG_2P_AGGRESSIVE = dataclasses.replace(
    ProducerLiteConfig(),
    max_waves_per_turn=8,
    roi_threshold=1.20,
    min_ships_to_launch=3.0,
    max_offensive_targets=16,
    reinforce_size_beta=1.6,
)
CONFIG_2P_REINFORCE_SAFE = dataclasses.replace(
    ProducerLiteConfig(),
    reinforce_size_beta=3.6,
    reinforce_eta_free=2.0,
    reinforce_eta_scale=9.0,
    roi_threshold=1.65,
    max_defensive_targets=6,
)
CONFIG_2P_S8_MULTI = dataclasses.replace(
    ProducerLiteConfig(),
    max_waves_per_turn=7,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.35,
    max_regroup_targets_per_source=5,
    size_multipliers=(0.5, 0.75, 1.0),
)


def _angle_from_center(x: float, y: float) -> float:
    import math
    return math.atan2(float(y) - 50.0, float(x) - 50.0)


def _angle_diff(a: float, b: float) -> float:
    import math
    d = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def _dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    import math
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def _choose_2p_mode(obs_tensors: dict) -> str:
    planets = obs_tensors["planets"].detach().cpu().tolist()
    player = int(float(obs_tensors["player"].reshape(-1)[0].item()))
    alive = [p for p in planets if len(p) >= 7 and float(p[0]) >= 0.0]
    owned = [p for p in alive if int(round(float(p[1]))) == player]
    enemies = [p for p in alive if int(round(float(p[1]))) >= 0 and int(round(float(p[1]))) != player]
    neutrals = [p for p in alive if int(round(float(p[1]))) < 0]
    if not owned:
        return "default"
    start = max(owned, key=lambda p: float(p[5]))
    sx, sy = float(start[2]), float(start[3])
    sa = _angle_from_center(sx, sy)
    enemy_dist = min((_dist_xy(sx, sy, p[2], p[3]) for p in enemies), default=999.0)

    def band(max_dist: float):
        items = [p for p in neutrals if _dist_xy(sx, sy, p[2], p[3]) <= max_dist]
        return (
            float(len(items)),
            float(sum(float(p[6]) for p in items)),
            float(sum(float(p[5]) for p in items)),
            float(sum(1 for p in items if float(p[5]) <= 20.0)),
        )

    n25_count, n25_prod, n25_ships, _n25_cheap = band(25.0)
    _n45_count, n45_prod, _n45_ships, _n45_cheap = band(45.0)

    best_chain = 0.0
    for mid in neutrals:
        mx, my = float(mid[2]), float(mid[3])
        md = _dist_xy(sx, sy, mx, my)
        mp, msh = float(mid[6]), float(mid[5])
        if not (8.0 <= md <= 48.0):
            continue
        if mp < 2.0 and not (15.0 <= msh <= 40.0):
            continue
        ma = _angle_from_center(mx, my)
        if _angle_diff(sa, ma) > 1.15:
            continue
        for outer in neutrals:
            if outer is mid:
                continue
            ox, oy = float(outer[2]), float(outer[3])
            od = _dist_xy(mx, my, ox, oy)
            if not (18.0 <= od <= 80.0):
                continue
            if _angle_diff(ma, _angle_from_center(ox, oy)) > 0.85:
                continue
            score = mp * 5.5 + max(0.0, 42.0 - msh) * 0.12 + float(outer[6]) * 7.0 + float(outer[5]) * 0.08 - md * 0.12 - od * 0.08
            best_chain = max(best_chain, score)

    best_support = 0.0
    for anchor in neutrals:
        ax, ay = float(anchor[2]), float(anchor[3])
        ar = _dist_xy(ax, ay, 50.0, 50.0)
        ad = _dist_xy(sx, sy, ax, ay)
        ap, ash = float(anchor[6]), float(anchor[5])
        aa = _angle_from_center(ax, ay)
        if ar < 30.0 or ad > 78.0:
            continue
        if ap < 3.0 and ash < 45.0:
            continue
        if _angle_diff(sa, aa) > 1.35:
            continue
        support = [
            p for p in neutrals
            if p is not anchor
            and _angle_diff(aa, _angle_from_center(p[2], p[3])) <= 0.95
            and _dist_xy(ax, ay, p[2], p[3]) <= 50.0
            and (float(p[6]) >= 2.0 or float(p[5]) >= 18.0)
        ]
        support_density = sum(float(p[6]) for p in support) + len(support) * 1.5
        best_support = max(best_support, support_density)

    if (
        enemy_dist >= 95.0
        and n25_count <= 2.0
        and n25_prod >= 9.0
        and n25_ships <= 70.0
        and best_chain >= 60.0
    ):
        return "reinforce"
    if (
        48.0 <= enemy_dist <= 82.0
        and n25_count <= 2.0
        and n25_prod <= 4.0
        and n25_ships <= 55.0
        and n45_prod <= 16.0
    ):
        return "s8multi"
    if (
        65.0 <= enemy_dist <= 85.0
        and n25_count >= 6.0
        and n45_prod >= 24.0
        and _n45_cheap >= 6.0
        and 45.0 <= best_chain <= 56.0
    ):
        return "s8multi"
    if (
        enemy_dist >= 98.0
        and best_support >= 30.0
        and n25_count >= 5.0
        and n25_ships >= 145.0
        and best_chain <= 50.0
        and _n45_cheap >= 5.0
    ):
        return "aggressive"
    if (
        enemy_dist >= 95.0
        and best_support >= 30.0
        and n25_count <= 4.5
        and 70.0 <= n25_ships <= 115.0
        and 18.0 <= n45_prod <= 22.0
        and 45.0 <= best_chain <= 55.0
    ):
        return "hold"
    return "default"


def _config_for(player_count: int, mode_2p: str | None = None) -> ProducerLiteConfig:
    if int(player_count) >= 4:
        return CONFIG_4P
    if mode_2p == "hold":
        return CONFIG_2P_HOLD
    if mode_2p == "aggressive":
        return CONFIG_2P_AGGRESSIVE
    if mode_2p == "reinforce":
        return CONFIG_2P_REINFORCE_SAFE
    if mode_2p == "s8multi":
        return CONFIG_2P_S8_MULTI
    return CONFIG_2P_DEFAULT


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.mode_2p: str | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.mode_2p = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.mode_2p = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        if int(mem.cached_player_count) < 4 and mem.mode_2p is None:
            mem.mode_2p = _choose_2p_mode(obs_tensors)
        config = _config_for(mem.cached_player_count, mem.mode_2p)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
