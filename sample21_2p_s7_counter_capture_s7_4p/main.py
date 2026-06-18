
from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Optional

# Setup logging for debugging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - Critical game/planning parameters
# ============================================================================

# Tensor validation thresholds
NAN_THRESHOLD = 1e-6
INF_THRESHOLD = 1e8
MIN_VIABLE_SHIPS = 0.1

# Performance tuning
DEFAULT_MEMORY_LIMIT_MB = 2048
TENSOR_ALLOCATION_WARNING_SIZE = 1e8

# Timing sentinel
_TURN_TIME_BUDGET_MS = 900  # 900ms per turn for safety

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
    logger.warning("__file__ not available, using working directory: %s", _HERE)

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


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_tensor(tensor: Tensor, name: str = "tensor") -> bool:
    """Check tensor for NaN/Inf values and warn if found.
    
    Args:
        tensor: Tensor to validate
        name: Name for logging
        
    Returns:
        True if valid, False otherwise
    """
    if tensor.numel() == 0:
        return True
    
    has_nan = bool(torch.isnan(tensor).any())
    has_inf = bool(torch.isinf(tensor).any())
    
    if has_nan:
        logger.warning(f"NaN detected in {name}: {torch.isnan(tensor).sum().item()} values")
        return False
    if has_inf:
        logger.warning(f"Inf detected in {name}: {torch.isinf(tensor).sum().item()} values")
        return False
    
    return True


def validate_obs_tensors(obs_tensors: dict[str, Any]) -> bool:
    """Validate observation tensors for shape and NaN/Inf issues.
    
    Args:
        obs_tensors: Tensor observation dict
        
    Returns:
        True if all validations pass
    """
    required_keys = {"planets", "step", "player"}
    if not required_keys.issubset(obs_tensors.keys()):
        logger.error(f"Missing required keys: {required_keys - obs_tensors.keys()}")
        return False
    
    planets = obs_tensors["planets"]
    if planets.dim() != 2 or planets.shape[-1] != 7:
        logger.error(f"Invalid planets shape: {planets.shape}, expected [P, 7]")
        return False
    
    return validate_tensor(planets, "planets")


# ============================================================================
# CONFIG MANAGEMENT
# ============================================================================

@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs for the Producer Lite agent.
    
    Controls planning horizon, shortlist sizes, ROI thresholds, and reinforcement risk modeling.
    Frozen dataclass ensures immutability and cache-friendly hashing.
    """

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
    # --- 2P contested-neutral candidate generation ------------------------
    enable_contested_2p: bool = True
    contested_opening_turns: int = 120
    contested_max_extra_targets: int = 6
    mirror_break_turns: int = 140
    thin_enemy_anchor_bonus: float = 0.45
    
    def validate(self) -> bool:
        """Validate config parameters for consistency.
        
        Returns:
            True if all parameters are valid
        """
        checks = [
            (self.horizon > 0, "horizon must be positive"),
            (self.roi_threshold > 0, "roi_threshold must be positive"),
            (self.min_ships_to_launch > 0, "min_ships_to_launch must be positive"),
            (self.max_waves_per_turn > 0, "max_waves_per_turn must be positive"),
            (self.reinforce_size_beta >= 0, "reinforce_size_beta must be non-negative"),
            (self.reinforce_eta_scale > 0, "reinforce_eta_scale must be positive"),
            (self.max_regroup_time > 0, "max_regroup_time must be positive"),
        ]
        
        for condition, message in checks:
            if not condition:
                logger.error(f"Config validation failed: {message}")
                return False
        
        return True


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon.
    
    Args:
        config: ProducerLiteConfig
        player_count: Number of players in game
        
    Returns:
        MovementConfig for fleet tracking
    """
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
    arithmetic on cached tensors.
    
    Args:
        obs: Parsed observation object with planet/fleet info
        cache: Distance cache object
        horizon: Time horizon in turns for enemy reach calculation
        player_id: ID of the observing player
        
    Returns:
        Tensor of shape [P] with enemy pressure per planet
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
    result = contrib.sum(dim=0)                                           # [P] summed over sources
    
    # Validate result
    if not validate_tensor(result, "enemy_pressure"):
        logger.warning("Enemy pressure tensor had invalid values, clamping to safe range")
        result = result.clamp(min=0.0, max=INF_THRESHOLD)
    
    return result


def _append_contested_2p_targets(
    *,
    target_idx: Tensor,
    target_exists: Tensor,
    obs,
    obs_tensors: dict[str, Any],
    cache,
    garrison_status,
    prod: Tensor,
    source_mask: Tensor,
    config: ProducerLiteConfig,
    K_eta: int,
    H: int,
    player_count: int,
) -> tuple[Tensor, Tensor]:
    """Add early 2P targets that a similar orbit_lite bot is likely to contest.

    This is intentionally candidate-generation only.  The exact scorer and greedy
    selector still decide whether to launch, so we avoid broad score penalties that
    previously broke good sample7/sample8 lines.
    """
    if int(player_count) != 2 or not bool(config.enable_contested_2p):
        return target_idx, target_exists
    step_t = obs_tensors.get("step")
    step = int(step_t.flatten()[0].item()) if isinstance(step_t, Tensor) else int(step_t or 0)
    if step > int(config.contested_opening_turns):
        return target_idx, target_exists

    P = int(obs.P)
    if P <= 0 or int(config.contested_max_extra_targets) <= 0:
        return target_idx, target_exists

    device = obs.device
    dtype = obs.ships.dtype
    neutral = obs.alive & obs.is_neutral
    enemy_src_mask = obs.alive & obs.is_enemy & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(neutral.any()) or not bool(source_mask.any()) or not bool(enemy_src_mask.any()):
        return target_idx, target_exists

    d0 = cache.cross_dist[0].to(dtype)
    speed = fleet_speed(obs.ships.clamp(min=1.0)).to(dtype).clamp(min=1e-6)
    eta_all = d0 / speed.view(P, 1)
    inf = torch.full((P,), float("inf"), dtype=dtype, device=device)
    my_eta = torch.where(source_mask.view(P, 1), eta_all, torch.full_like(eta_all, float("inf"))).amin(dim=0)
    enemy_eta = torch.where(enemy_src_mask.view(P, 1), eta_all, torch.full_like(eta_all, float("inf"))).amin(dim=0)

    # 1) Opening race: same-family bots often value the same near neutral.
    can_race = (
        neutral
        & (my_eta <= float(K_eta))
        & (enemy_eta <= float(K_eta + 6))
        & (my_eta <= enemy_eta + 2.5)
        & ((prod >= 2.0) | (obs.ships <= 25.0))
    )
    race_margin = (enemy_eta - my_eta).clamp(min=-3.0, max=6.0)
    race_score = prod * 5.0 + race_margin * 1.5 - obs.ships * 0.05

    # 2) Observed counter-capture: once an enemy fleet is visible, garrison_status
    # already knows future owner/ship state, but the default shortlist may not
    # include the target. Add those future enemy captures as exact-scored options.
    future_score = torch.full((P,), float("-inf"), dtype=dtype, device=device)
    if H > 0:
        owner_future = garrison_status.owner[:, 1 : H + 1]
        enemy_future = (owner_future >= 0) & (owner_future != int(obs.player_id))
        future_enemy_capture = neutral & enemy_future.any(dim=-1)
        if bool(future_enemy_capture.any()):
            first_turn = _first_true_turn(enemy_future, device=device, dtype=dtype)
            retake_window = (my_eta >= first_turn - 1.0) & (my_eta <= first_turn + 5.0)
            future_mask = future_enemy_capture & retake_window & (my_eta <= float(K_eta)) & (prod >= 2.0)
            future_score = torch.where(
                future_mask,
                prod * 7.0 + (6.0 - (my_eta - first_turn).abs()).clamp(min=0.0) - obs.ships * 0.04,
                future_score,
            )

    existing = torch.zeros(P, dtype=torch.bool, device=device)
    valid_existing = target_idx[target_exists].clamp(0, max(P - 1, 0))
    if valid_existing.numel() > 0:
        existing.scatter_(0, valid_existing, True)

    extra_rank = torch.maximum(torch.where(can_race, race_score, torch.full_like(race_score, float("-inf"))), future_score)
    extra_mask = torch.isfinite(extra_rank) & ~existing
    if not bool(extra_mask.any()):
        return target_idx, target_exists

    extra_idx, extra_exists = _candidate_indices(
        extra_rank,
        extra_mask,
        max(1, min(int(config.contested_max_extra_targets), P)),
    )
    return torch.cat([target_idx, extra_idx], dim=0), torch.cat([target_exists, extra_exists], dim=0)


def _first_true_turn(mask: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """First 1-based true turn per row; inf if none."""
    P, H = int(mask.shape[0]), int(mask.shape[1])
    turns = torch.arange(1, H + 1, device=device, dtype=dtype).view(1, H).expand(P, H)
    inf_turns = torch.full_like(turns, float("inf"))
    return torch.where(mask, turns, inf_turns).amin(dim=-1)


def _apply_2p_mirror_break_bonus(
    *,
    score: Tensor,
    cand_valid: Tensor,
    cand_tgt_slot: Tensor,
    cand_eta: Tensor,
    obs,
    obs_tensors: dict[str, Any],
    prod: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
) -> Tensor:
    """Nudge early attacks into thin enemy anchors common in orbit_lite mirrors."""
    if int(player_count) != 2 or float(config.thin_enemy_anchor_bonus) <= 0.0:
        return score
    step_t = obs_tensors.get("step")
    step = int(step_t.flatten()[0].item()) if isinstance(step_t, Tensor) else int(step_t or 0)
    if step > int(config.mirror_break_turns):
        return score

    P = int(obs.P)
    tgt = cand_tgt_slot.clamp(0, max(P - 1, 0))
    tgt_enemy = obs.is_enemy[tgt]
    tgt_prod = prod[tgt].to(score.dtype)
    tgt_ships = obs.ships[tgt].to(score.dtype)
    eta = cand_eta.squeeze(-1).to(score.dtype) if cand_eta.dim() == 2 else cand_eta.to(score.dtype)

    # Strong orbit_lite-like agents often drain newly captured productive anchors.
    # We only add a small bonus when the target is both productive and thin, and
    # the flight is short enough to be a real punish instead of a slow overreach.
    thin_limit = tgt_prod * 8.0 + 18.0
    thin_anchor = (
        cand_valid
        & tgt_enemy
        & (tgt_prod >= 2.0)
        & (tgt_ships <= thin_limit)
        & (eta <= 12.0)
    )
    bonus = (
        float(config.thin_enemy_anchor_bonus)
        + (0.06 * tgt_prod).clamp(max=0.30)
        + (12.0 - eta).clamp(min=0.0) * 0.015
    ).clamp(max=0.85)
    return score + torch.where(thin_anchor, bonus, torch.zeros_like(score))


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict[str, Any],
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    turn_start_time: Optional[float] = None,
) -> dict[str, Any]:
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    
    Args:
        movement: PlanetMovement tracking object
        obs: Parsed observation
        obs_tensors: Raw tensor observation dict
        cache: Distance cache
        garrison_status: Garrison status projection
        prod: Production tensor
        alive_by_step: Alive planets by time step
        config: ProducerLiteConfig with behavior parameters
        player_count: Number of players in the game
        turn_start_time: Time turn started (for budget checking)
        
    Returns:
        Dict of launch entries with sparse action format
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
    target_idx, target_exists = _append_contested_2p_targets(
        target_idx=target_idx,
        target_exists=target_exists,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        garrison_status=garrison_status,
        prod=prod,
        source_mask=source_mask,
        config=config,
        K_eta=K_eta,
        H=H,
        player_count=player_count,
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
    K = int(floor.shape[-1])

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    # Engine needs integer ship counts; floor (never exceed what's available).
    sizes = drain.view(S, 1).expand(S, T).floor()                                # [S, T]

    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
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

    # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
    # single size must clear the defender it lands on (size >= floor_at_arr). Owned
    # targets have floor 1 (reinforcement), so any positive send clears.
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr                                         # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= MIN_VIABLE_SHIPS) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )                                                                            # [S, T]

    # --- pack one candidate per (source, target); contributor axis L = 1 --------
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
    score = _apply_2p_mirror_break_bonus(
        score=score,
        cand_valid=cand_valid,
        cand_tgt_slot=cand_tgt_slot,
        cand_eta=cand_eta,
        obs=obs,
        obs_tensors=obs_tensors,
        prod=prod,
        config=config,
        player_count=player_count,
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

    # Reuse the enemy-mass proxy already computed above (one [P, P] reduction
    # serves both the reinforcement floor and this regroup gradient).
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict[str, Any], *, config: ProducerLiteConfig, player_count: int, memory: ProducerLiteMemory) -> dict[str, Any]:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    Executes one complete turn of planning:
    1. Validate observation tensors
    2. Parse the observation tensors
    3. Build/update planet movement tracking
    4. Build distance cache
    5. Plan attack waves and regroup movements
    6. Update movement state with new launches
    7. Return sparse action payload
    
    Args:
        obs_tensors: Raw tensor observation dict
        config: ProducerLiteConfig with behavior parameters
        player_count: Number of players in the game
        memory: ProducerLiteMemory object for state persistence
        
    Returns:
        Sparse action row dict ready for conversion to moves
    """
    turn_start_time = time.time()
    
    # Validate observation tensors early
    if not validate_obs_tensors(obs_tensors):
        logger.error("Observation validation failed, returning empty action")
        device = obs_tensors.get("planets", torch.tensor([])).device
        return empty_action_row(device)
    
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
        turn_start_time=turn_start_time,
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
    result = entries_to_sparse_payload(entries, planet_ids=planet_ids)
    
    # Log turn timing
    elapsed_ms = (time.time() - turn_start_time) * 1000
    if elapsed_ms > _TURN_TIME_BUDGET_MS:
        logger.warning(f"Turn exceeded time budget: {elapsed_ms:.1f}ms > {_TURN_TIME_BUDGET_MS}ms")
    
    return result


# Game mode presets with tuned configurations
CONFIG_2P = ProducerLiteConfig()  # Default for 2-player

CONFIG_3P = replace(
    ProducerLiteConfig(),
    horizon=15,
    max_sources_per_lane=8,
    max_offensive_targets=10,
    max_defensive_targets=3,
    max_waves_per_turn=5,
    roi_threshold=1.4,
)

CONFIG_4P = replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    """Get the appropriate config for the given player count.
    
    Selects game-mode-specific config tuning based on player count.
    Higher player count = more chaotic, shorter horizon, fewer targets.
    
    Args:
        player_count: Number of players in the game
        
    Returns:
        ProducerLiteConfig tuned for the game mode
    """
    pc = int(player_count)
    if pc >= 4:
        return CONFIG_4P
    elif pc == 3:
        return CONFIG_3P
    else:
        return CONFIG_2P


class ProducerLiteMemory:
    """Persistent state storage across turns.
    
    Maintains movement tracking, player count cache, last action row,
    and performance metrics for continuity and efficiency.
    """
    
    def __init__(self) -> None:
        self.movement: Optional[PlanetMovement] = None
        self.cached_player_count: Optional[int] = None
        self.last_sparse_action_row: Optional[dict[str, Any]] = None
        self.turns_executed: int = 0
        self.total_turn_time_ms: float = 0.0
        self.launches_sent: int = 0

    def reset(self) -> None:
        """Reset all cached state. Called at game boundaries."""
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.turns_executed = 0
        self.total_turn_time_ms = 0.0
        self.launches_sent = 0
    
    def __repr__(self) -> str:
        avg_turn_time = (
            self.total_turn_time_ms / self.turns_executed 
            if self.turns_executed > 0 else 0.0
        )
        return (f"ProducerLiteMemory(turns={self.turns_executed}, "
                f"avg_turn_ms={avg_turn_time:.1f}, launches={self.launches_sent}, "
                f"movement={'active' if self.movement else 'None'}, "
                f"player_count={self.cached_player_count})")


class ProducerLiteRuntime:
    """Main agent runtime: stateful planner executor.
    
    Manages game state, config selection, and turn execution.
    Handles the stateful aspects (movement tracking, player count inference)
    that persist across the game. Includes monitoring and debugging.
    """
    
    def __init__(self, memory: Optional[ProducerLiteMemory] = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()
        self._last_error_msg: Optional[str] = None

    def reset(self) -> None:
        """Reset memory for a new game. Called between games."""
        self.memory.reset()
        self._last_error_msg = None

    def tensor_action(self, obs_tensors: dict[str, Any]) -> dict[str, Any]:
        """Execute one turn of planning with monitoring.
        
        Args:
            obs_tensors: Raw tensor observation from the game engine
            
        Returns:
            Sparse action row ready for conversion to move list
            
        Raises:
            RuntimeError: If player count inference fails
        """
        mem = self.memory
        turn_start = time.time()
        
        # Reset player count at game start (step == 0)
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        
        # Infer player count once and cache it
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
            if mem.cached_player_count is None:
                raise RuntimeError("Failed to infer player count from observation")
            logger.info(f"Game mode: {mem.cached_player_count}P")
        
        config = _config_for(mem.cached_player_count)
        if not config.validate():
            raise RuntimeError(f"Config validation failed for {mem.cached_player_count}P mode")
        
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        mem.turns_executed += 1
        mem.total_turn_time_ms += (time.time() - turn_start) * 1000
        if row.get("counts", 0) > 0:
            mem.launches_sent += int(row.get("counts", 0))
        
        return row


# Global runtime instance — singleton pattern for agent entry point
_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs: Any) -> list[list[Any]]:
    """Single-observation entry point for local play and Kaggle.
    
    Converts observation dict/object to tensors, runs planning, and
    converts sparse action back to move list format. Includes comprehensive
    error handling with safe fallback behavior.
    
    Args:
        obs: Observation dict or object with player ID, planets, fleets, etc.
        
    Returns:
        Move list: [[from_planet_id, angle, num_ships], ...]
        Returns empty list on any error (safe fallback for robustness).
        
    Raises:
        ValueError: If observation format is fundamentally invalid
    """
    try:
        if obs is None:
            raise ValueError("Observation is None")
        
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        player_id = int(player)
        
        obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
        with torch.no_grad():
            sparse_row = _RUNTIME.tensor_action(obs_tensors)
        
        moves = sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
        if not isinstance(moves, list):
            logger.warning(f"Unexpected return type from sparse_action_row_to_moves: {type(moves)}")
            return []
        
        return moves
        
    except Exception as e:
        logger.error(f"Agent error: {type(e).__name__}: {e}", exc_info=True)
        return []  # Return empty move list on error (safe fallback)
