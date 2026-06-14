
# main_v10.py — hairate2 + Counter-Snipe + Breakout + Adaptive (58.8% proven)
# 4P FFA: uses hairate2 original logic (no modifications)
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
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
try:
    from orbit_lite.opening_2ply import apply_opening_2ply_score
except ModuleNotFoundError:
    def apply_opening_2ply_score(
        *,
        score: Tensor,
        obs,
        cache,
        source_idx: Tensor,
        target_idx: Tensor,
        target_exists: Tensor,
        target_is_mine: Tensor,
        sizes: Tensor,
        eta: Tensor,
        floor_at_arr: Tensor,
        config,
        player_count: int,
        step: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if int(player_count) > 2 or int(step) >= int(getattr(config, "opening_2ply_turns", 45)):
            return score
        P = int(obs.P)
        S = int(source_idx.shape[0])
        T = int(target_idx.shape[0])
        enemy_mask = obs.is_enemy & obs.alive
        if P <= 0 or S == 0 or T == 0 or not bool(enemy_mask.any()):
            return score
        src_safe = source_idx.clamp(0, P - 1)
        tgt_safe = target_idx.clamp(0, P - 1)
        enemy_ships = torch.where(enemy_mask, obs.ships.to(dtype), torch.zeros(P, dtype=dtype, device=device))
        enemy_speed = fleet_speed(enemy_ships.clamp(min=1e-6)).clamp(min=1e-6)
        enemy_active = enemy_mask.to(dtype)
        eta_to_tgt = cache.cross_dist[0, :, tgt_safe].to(dtype) / enemy_speed.view(P, 1)
        eta_to_tgt = torch.where(enemy_mask.view(P, 1), eta_to_tgt, torch.full_like(eta_to_tgt, float("inf")))
        enemy_eta_to_tgt = eta_to_tgt.min(dim=0).values
        eta_to_src = cache.cross_dist[0, :, src_safe].to(dtype) / enemy_speed.view(P, 1)
        eta_to_src = torch.where(enemy_mask.view(P, 1), eta_to_src, torch.full_like(eta_to_src, float("inf")))
        response_window = (eta + float(getattr(config, "opening_2ply_response_turns", 9.0))).clamp(min=1.0)
        tgt_decay = (1.0 - eta_to_tgt.unsqueeze(0) / response_window.unsqueeze(1)).clamp(min=0.0)
        retake_pressure = (enemy_ships.view(1, P, 1) * enemy_active.view(1, P, 1) * tgt_decay).sum(dim=1)
        retake_deficit = (retake_pressure - (sizes - floor_at_arr).clamp(min=0.0)).clamp(min=0.0)
        source_after = obs.ships[src_safe].to(dtype).view(S, 1) - sizes
        source_window = float(getattr(config, "opening_2ply_source_turns", 8.0))
        source_pressure = torch.where(
            eta_to_src <= source_window,
            enemy_ships.view(P, 1) * enemy_active.view(P, 1),
            torch.zeros_like(eta_to_src),
        ).sum(dim=0)
        source_floor = torch.maximum(
            torch.full_like(source_pressure, float(getattr(config, "opening_2ply_source_floor", 5.0))),
            source_pressure * float(getattr(config, "opening_2ply_source_pressure_fraction", 0.22)),
        )
        source_deficit = (source_floor.view(S, 1) - source_after).clamp(min=0.0)
        neutral_t = obs.is_neutral[tgt_safe] & target_exists
        target_is_attack = ~target_is_mine.view(1, T)
        arrival_advantage = enemy_eta_to_tgt.view(1, T) - eta
        target_prod = obs.prod[tgt_safe].to(dtype)
        race_bonus = (
            arrival_advantage.clamp(min=0.0, max=8.0) / 8.0
        ) * target_prod.view(1, T) * float(getattr(config, "opening_2ply_race_weight", 0.10))
        race_allowed = (
            neutral_t.view(1, T)
            & (arrival_advantage >= float(getattr(config, "opening_2ply_min_arrival_advantage", 1.5)))
            & (source_after >= float(getattr(config, "opening_2ply_min_source_after", 5.0)))
            & (score.reshape(S, T) >= float(getattr(config, "roi_threshold", 1.5)) - float(getattr(config, "opening_2ply_roi_margin", 0.35)))
        )
        adjusted = score.reshape(S, T)
        adjusted = adjusted + torch.where(race_allowed, race_bonus, torch.zeros_like(race_bonus))
        adjusted = adjusted - torch.where(target_is_attack, retake_deficit * float(getattr(config, "opening_2ply_retake_weight", 0.14)), torch.zeros_like(retake_deficit))
        adjusted = adjusted - torch.where(target_is_attack, source_deficit * float(getattr(config, "opening_2ply_source_weight", 0.08)), torch.zeros_like(source_deficit))
        return adjusted.reshape_as(score)
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
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    opening_2ply_enabled: bool = True
    opening_2ply_turns: int = 45
    opening_2ply_response_turns: float = 9.0
    opening_2ply_race_weight: float = 0.10
    opening_2ply_retake_weight: float = 0.14
    opening_2ply_source_weight: float = 0.08
    opening_2ply_roi_margin: float = 0.35
    opening_2ply_min_arrival_advantage: float = 1.5
    opening_2ply_min_source_after: float = 5.0
    opening_2ply_source_turns: float = 8.0
    opening_2ply_source_floor: float = 5.0
    opening_2ply_source_pressure_fraction: float = 0.22
    opening_2ply_hard_source_veto: bool = False


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
    """Reachable-enemy-mass proxy per planet — ``[P]``.

    v1-flying-pressure: original baseline + in-flight enemy fleet contribution.
    Pressure on planet ``t`` = sum over enemy sources (planets and fleets) of
    ``ships * decay``, where ``decay = (1 - d / reach_dist).clamp(0, 1)``.

    Source garrison term (unchanged from v1):
      d = cross_dist[0][src, tgt]
      reach_dist = fleet_speed(src.ships) * horizon

    NEW — in-flight enemy fleet term:
      For each alive enemy fleet f (owner != player_id), compute Euclidean
      distance from fleet's current (x,y) to each planet t's current (x,y),
      and apply the same decay structure with the fleet's own speed.
      A fleet that's already 70% of the way to me carries more weight than
      its origin planet, which is the whole point.

    Notes:
      - Approximations still in: ignores target orbital drift over horizon,
        production accrued in flight, per-owner split.
      - Fleet angle is ignored (treated as "could be heading anywhere") —
        upper-bound proxy. Conservative: overstates pressure for fleets
        flying away from t. Acceptable as a regroup gradient (it's a rank).
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    pid = int(player_id)
    H = max(float(horizon), 1e-6)

    # ----- planet-source term (unchanged) ---------------------------------
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt]
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * H).clamp(min=1e-6)                 # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)    # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye               # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib_planets = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    pressure = contrib_planets.sum(dim=0)                                # [P]

    # ----- in-flight fleet term (new) -------------------------------------
    f_alive = obs.f_alive
    if bool(f_alive.any()):
        f_owner = obs.f_owner.to(torch.long)
        f_enemy = f_alive & (f_owner >= 0) & (f_owner != pid)            # [F]
        if bool(f_enemy.any()):
            fx = obs.f_x.to(dtype)[f_enemy]                              # [E]
            fy = obs.f_y.to(dtype)[f_enemy]
            fs = obs.f_ships.to(dtype)[f_enemy].clamp(min=1e-6)          # [E]
            f_speed = fleet_speed(fs)                                    # [E]
            f_reach = (f_speed * H).clamp(min=1e-6)                      # [E]

            tx = obs.x.to(dtype).view(1, P)                              # [1, P]
            ty = obs.y.to(dtype).view(1, P)
            dxe = fx.view(-1, 1) - tx                                    # [E, P]
            dye = fy.view(-1, 1) - ty
            d_ft = torch.sqrt((dxe * dxe + dye * dye).clamp(min=0.0))    # [E, P]
            decay_f = (1.0 - d_ft / f_reach.view(-1, 1)).clamp(min=0.0)  # [E, P]
            tgt_alive = obs.alive.view(1, P)                             # [1, P]
            decay_f = torch.where(tgt_alive, decay_f, torch.zeros_like(decay_f))
            contrib_fleets = fs.view(-1, 1) * decay_f                    # [E, P]
            pressure = pressure + contrib_fleets.sum(dim=0)

    return pressure


# ---------------------------------------------------------------------------
# Phase 1: Counter-Snipe — detect enemy neutral captures, send minimal counter
# ---------------------------------------------------------------------------

def _counter_snipe_pass(
    obs, obs_tensors, movement, garrison_status,
    leftover, prod, alive_by_step, config, player_count,
):
    """Detect neutrals about to be enemy-captured; send sniper-sized counter."""
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H = int(config.horizon)
    min_send = float(config.min_ships_to_launch)

    owner_traj = garrison_status.owner               # [P, H+1]
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
    flip_turn = torch.where(any_flip, flip_turn,
                            torch.full_like(flip_turn, H + 1))

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
        garrison_status, target_idx=cs_idx, k_max=H,
        capture_overhead=1.0, player_id=pid,
    )
    K = int(cs_floor.shape[-1])
    if K == 0:
        return _empty_entries(device, dtype), leftover

    flip_k = flip_turn[cs_idx]
    k_range = torch.arange(1, K + 1, device=device).view(1, K)
    after_flip = k_range >= flip_k.view(T_cs, 1)
    floor_masked = torch.where(after_flip, cs_floor,
                               torch.full_like(cs_floor, float('inf')))
    min_floor, _ = floor_masked.min(dim=-1)

    has_opportunity = torch.isfinite(min_floor)
    if not bool(has_opportunity.any()):
        return _empty_entries(device, dtype), leftover

    sniper_size = (min_floor + 2.0).ceil().clamp(min=min_send)

    sizes = sniper_size.view(1, T_cs).expand(S, T_cs)
    budget_s = leftover[cs_src_idx.clamp(0, P - 1)].view(S, 1)
    sizes = sizes.clamp(max=budget_s).floor()

    eta_cap = torch.full((T_cs,), float(H), dtype=dtype, device=device)

    active = reachable_mask(
        movement, source_idx=cs_src_idx, target_idx=cs_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)

    aim = intercept_angle(
        movement, cs_src_idx.unsqueeze(1), cs_idx.unsqueeze(0),
        sizes, active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T_cs))
    after_flip_viable = eta >= flip_k.view(1, T_cs).to(dtype)

    k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
    floor_at_arr = cs_floor.unsqueeze(0).expand(
        S, T_cs, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = cs_src_idx.view(S, 1) != cs_idx.view(1, T_cs)
    valid = (
        viable & clears_floor & after_flip_viable
        & (sizes >= min_send) & src_neq_tgt
        & cs_src_exists.view(S, 1) & has_opportunity.view(1, T_cs)
    )

    if not bool(valid.any()):
        return _empty_entries(device, dtype), leftover

    C = S * T_cs
    L = 1
    cs_cand_src  = cs_src_idx.view(S, 1).expand(S, T_cs).reshape(C, L)
    cs_cand_tgt  = cs_idx.view(1, T_cs).expand(S, T_cs).reshape(C)
    cs_cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cs_cand_ang  = angle.reshape(C, L)
    cs_cand_eta  = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cs_cand_act  = valid.reshape(C, L)
    cs_cand_val  = valid.reshape(C)

    launches = make_launch_set(
        source_slots=cs_cand_src,
        target_slots=cs_cand_tgt.unsqueeze(-1).expand(C, L),
        ships=cs_cand_send, eta=cs_cand_eta,
        valid=cs_cand_act & cs_cand_val.unsqueeze(-1),
        player_id=pid,
    )
    cs_score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    cs_score = torch.where(cs_cand_val, cs_score,
                           torch.full_like(cs_score, float('-inf')))

    MAX_CS = 2
    entries_src, entries_tgt, entries_send = [], [], []
    entries_ang, entries_eta = [], []
    cs_leftover = leftover.clone()
    used_targets = set()

    for _ in range(MAX_CS):
        best_c = int(_stable_argmax(cs_score).item())
        best_val = cs_score[best_c]
        CS_THRESHOLD = 1.0
        if not (bool(torch.isfinite(best_val))
                and float(best_val.item()) > CS_THRESHOLD):
            break

        s_local = best_c // T_cs
        t_local = best_c % T_cs
        src_slot = cs_src_idx[s_local]
        tgt_slot = cs_idx[t_local]
        send_amt = cs_cand_send[best_c, 0]

        tgt_int = int(tgt_slot.item())
        if tgt_int in used_targets:
            cs_score[best_c] = float('-inf')
            continue
        if cs_leftover[src_slot] < send_amt:
            cs_score[best_c] = float('-inf')
            continue

        entries_src.append(src_slot)
        entries_tgt.append(tgt_slot)
        entries_send.append(send_amt)
        entries_ang.append(cs_cand_ang[best_c, 0])
        entries_eta.append(cs_cand_eta[best_c, 0])

        cs_leftover[src_slot] = (cs_leftover[src_slot] - send_amt).clamp(min=0.0)
        used_targets.add(tgt_int)

        same_tgt = (cs_cand_tgt == tgt_slot)
        cs_score = torch.where(same_tgt,
                               torch.full_like(cs_score, float('-inf')),
                               cs_score)

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


# ---------------------------------------------------------------------------
# Phase 2: Desperation Breakout — break safe_drain when behind
# ---------------------------------------------------------------------------

def _is_behind(obs, prod, step: int, player_count: int) -> bool:
    """Check if we're in a truly desperate position requiring breakout."""
    if step < 250 or player_count < 2:
        return False
    pid = int(obs.player_id)
    alive = obs.alive
    dtype = obs.ships.dtype

    my_mask = obs.owned & alive
    my_prod = prod[my_mask].sum().item() if bool(my_mask.any()) else 0.0
    my_power = obs.ships[my_mask].to(dtype).sum().item() if bool(my_mask.any()) else 0.0

    enemy_mask = obs.is_enemy & alive
    enemy_prod = prod[enemy_mask].sum().item() if bool(enemy_mask.any()) else 0.0
    enemy_power = obs.ships[enemy_mask].to(dtype).sum().item() if bool(enemy_mask.any()) else 0.0

    if enemy_prod <= 0 and enemy_power <= 0:
        return False
    # Only breakout in truly desperate situations
    if my_prod < enemy_prod * 0.70 and my_power < enemy_power * 0.55:
        return True
    return False


def _breakout_pass(
    obs, obs_tensors, movement, garrison_status,
    leftover, prod, alive_by_step, config, player_count,
):
    """Breakout: bypass safe_drain limits to send an aggressive attack."""
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H = int(config.horizon)
    min_send = float(config.min_ships_to_launch)

    BREAKOUT_FRAC = 0.55
    MIN_GARRISON_FOR_BREAKOUT = 50.0

    src_mask = obs.owned & obs.alive & (obs.ships.to(dtype) >= MIN_GARRISON_FOR_BREAKOUT)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype), leftover

    S_cap = max(1, min(6, P))
    bo_src_idx, bo_src_exists = _candidate_indices(obs.ships.to(dtype), src_mask, S_cap)
    S = int(bo_src_idx.shape[0])

    garrison = obs.ships[bo_src_idx.clamp(0, P - 1)].to(dtype)
    bo_budget = (garrison * BREAKOUT_FRAC).floor().clamp(min=min_send)
    already_sent = (garrison - leftover[bo_src_idx.clamp(0, P - 1)]).clamp(min=0.0)
    bo_available = (bo_budget - already_sent).clamp(min=min_send)

    can_send = bo_src_exists & (bo_available >= min_send)
    if not bool(can_send.any()):
        return _empty_entries(device, dtype), leftover

    prod_f = prod.to(dtype)
    tgt_mask = obs.is_enemy & obs.alive & (prod_f >= 2.0)
    if not bool(tgt_mask.any()):
        return _empty_entries(device, dtype), leftover

    T_cap = max(1, min(8, P))
    bo_tgt_idx, bo_tgt_exists = _candidate_indices(prod_f, tgt_mask, T_cap)
    T = int(bo_tgt_idx.shape[0])

    sizes = bo_available.view(S, 1).expand(S, T).floor()
    eta_cap = torch.full((T,), float(H), dtype=dtype, device=device)

    active = reachable_mask(
        movement, source_idx=bo_src_idx, target_idx=bo_tgt_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)

    aim = intercept_angle(
        movement, bo_src_idx.unsqueeze(1), bo_tgt_idx.unsqueeze(0),
        sizes, active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    bo_floor = capture_floor(
        garrison_status, target_idx=bo_tgt_idx, k_max=H,
        capture_overhead=1.0, player_id=pid,
    )
    K = int(bo_floor.shape[-1])

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = bo_floor.unsqueeze(0).expand(
            S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        clears_floor = sizes >= floor_at_arr
    else:
        clears_floor = torch.ones(S, T, dtype=torch.bool, device=device)

    src_neq_tgt = bo_src_idx.view(S, 1) != bo_tgt_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= min_send) & src_neq_tgt
        & can_send.view(S, 1) & bo_tgt_exists.view(1, T)
    )

    if not bool(valid.any()):
        return _empty_entries(device, dtype), leftover

    C = S * T
    L = 1
    bo_cand_src  = bo_src_idx.view(S, 1).expand(S, T).reshape(C, L)
    bo_cand_tgt  = bo_tgt_idx.view(1, T).expand(S, T).reshape(C)
    bo_cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    bo_cand_ang  = angle.reshape(C, L)
    bo_cand_eta  = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    bo_cand_act  = valid.reshape(C, L)
    bo_cand_val  = valid.reshape(C)

    launches = make_launch_set(
        source_slots=bo_cand_src,
        target_slots=bo_cand_tgt.unsqueeze(-1).expand(C, L),
        ships=bo_cand_send, eta=bo_cand_eta,
        valid=bo_cand_act & bo_cand_val.unsqueeze(-1),
        player_id=pid,
    )
    bo_score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    bo_score = torch.where(bo_cand_val, bo_score,
                           torch.full_like(bo_score, float('-inf')))

    best_c = int(_stable_argmax(bo_score).item())
    best_val = bo_score[best_c]

    BO_THRESHOLD = 0.5
    if not (bool(torch.isfinite(best_val))
            and float(best_val.item()) > BO_THRESHOLD):
        return _empty_entries(device, dtype), leftover

    src_slot = bo_src_idx[best_c // T]
    tgt_slot = bo_tgt_idx[best_c % T]
    send_amt = bo_cand_send[best_c, 0]

    bo_leftover = leftover.clone()
    bo_leftover[src_slot] = (bo_leftover[src_slot] - send_amt).clamp(min=0.0)

    return LaunchEntries(
        source_slots=src_slot.unsqueeze(0).to(torch.long),
        target_slots=tgt_slot.unsqueeze(0).to(torch.long),
        ships=send_amt.unsqueeze(0),
        angle=bo_cand_ang[best_c, 0].unsqueeze(0),
        eta=bo_cand_eta[best_c, 0].unsqueeze(0),
        valid=torch.ones(1, dtype=torch.bool, device=device),
    ), bo_leftover


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
    step = int(obs_tensors.get("step", torch.zeros(1, device=device)).reshape(-1)[0].item())

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

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    sizes = drain.view(S, 1).expand(S, T).floor()                                # [S, T]

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),                                                 # [S, 1]
        target_idx.unsqueeze(0),                                                 # [1, T]
        sizes,                                                                    # [S, T]
        active=active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr                                         # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )                                                                            # [S, T]

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
    cand_is_def = target_is_mine[cand_tgt_short]                                  # [C]

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
    )                                                                            # [C]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    score = apply_opening_2ply_score(
        score=score,
        obs=obs,
        cache=cache,
        source_idx=source_idx,
        target_idx=target_idx,
        target_exists=target_exists,
        target_is_mine=target_is_mine,
        sizes=sizes,
        eta=eta,
        floor_at_arr=floor_at_arr,
        config=config,
        player_count=player_count,
        step=step,
        dtype=dtype,
        device=device,
    )

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    # --- V10 enhancements (2P only) ------------------------------------------
    if int(player_count) <= 2:
        # Counter-Snipe pass (Phase 1)
        cs_entries, leftover = _counter_snipe_pass(
            obs, obs_tensors, movement, garrison_status,
            leftover, prod, alive_by_step, config, player_count,
        )
        # Breakout pass (Phase 2) — only when behind
        bo_entries = _empty_entries(device, dtype)
        current_step = int(obs_tensors.get("step", torch.zeros(1)).flatten()[0].item())
        if _is_behind(obs, prod, current_step, int(player_count)):
            bo_entries, leftover = _breakout_pass(
                obs, obs_tensors, movement, garrison_status,
                leftover, prod, alive_by_step, config, player_count,
            )
    else:
        cs_entries = _empty_entries(device, dtype)
        bo_entries = _empty_entries(device, dtype)

    if not bool(config.enable_regroup):
        return concat_launch_entries([wave_entries, cs_entries, bo_entries])
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, cs_entries, bo_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit."""
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

    # Phase 4: Adaptive config (2P only)
    current_step = int(obs_tensors.get("step", torch.zeros(1)).flatten()[0].item())
    config = _config_for(
        player_count, obs=obs, prod=movement.planet_prod, step=current_step,
    )

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


def _config_for(player_count: int, obs=None, prod=None, step: int = 0) -> ProducerLiteConfig:
    """Return config adapted to game state. 4P uses hairate2 original."""
    if int(player_count) >= 4:
        return CONFIG_4P

    base = ProducerLiteConfig()
    if obs is None or prod is None or step < 60:
        return base

    alive = obs.alive
    dtype = obs.ships.dtype
    my_mask = obs.owned & alive
    my_prod = float(prod[my_mask].sum().item()) if bool(my_mask.any()) else 0.0
    enemy_mask = obs.is_enemy & alive
    enemy_prod = float(prod[enemy_mask].sum().item()) if bool(enemy_mask.any()) else 0.0

    if enemy_prod <= 0:
        return base

    prod_ratio = my_prod / max(enemy_prod, 1.0)

    # Only adapt when clearly ahead: avoid small wasteful attacks
    if prod_ratio > 1.20:
        return dataclasses.replace(base, min_ships_to_launch=6.0)
    return base


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
        config = _config_for(mem.cached_player_count)
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
