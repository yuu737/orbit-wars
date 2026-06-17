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
from orbit_lite.planner_core import is_comet_planet


_ORIGINAL_TIER_CANDIDATES = _h14._tier_candidates
_ORIGINAL_BUILD_TARGET_SHORTLIST = _h14.build_target_shortlist
_ORIGINAL_CHEAP_ENEMY_PRESSURE = _h14.cheap_enemy_pressure

ALLY_LINK_DISTANCE = 38.0
ANCHOR_DISTANCE = 52.0
ENEMY_DANGER_DISTANCE = 46.0
TERRITORY_START_TURN = 25
TERRITORY_EXTRA_TARGETS = 6
CONNECT_BONUS = 0.22
ANCHOR_BONUS = 0.18
PROD_CONNECT_SCALE = 0.020
ISOLATION_PENALTY = 0.26
ENEMY_CLUSTER_EDGE_BONUS = 0.14
MAX_ABS_ADJUSTMENT = 0.45
SOURCE_ANCHOR_BONUS = 0.26
LOCAL_EXPANSION_BONUS = 0.18
THIN_FRONTLINE_PENALTY = 0.42
OVEREXTENSION_PENALTY = 0.34
SOURCE_ANCHOR_SHIPS_FACTOR = 9.0
SOURCE_THIN_SHIPS_FACTOR = 6.0
TERRITORY_REGROUP_START = 120
TERRITORY_REGROUP_WEIGHT = 0.065


def _is_4p_obs(obs_tensors: dict) -> bool:
    try:
        return int(_h14.largest_initial_player_count(obs_tensors)) >= 4
    except Exception:
        return False


def _looks_like_4p_runtime(obs) -> bool:
    owner = obs.owner_abs
    alive_owner = owner[(owner >= 0) & obs.alive]
    if int(alive_owner.numel()) == 0:
        return False
    return int(torch.unique(alive_owner.long()).numel()) >= 3


def _territory_regroup_pressure(obs, cache, *, horizon: float, player_id: int):
    base = _ORIGINAL_CHEAP_ENEMY_PRESSURE(obs, cache, horizon=float(horizon), player_id=int(player_id))
    if not _looks_like_4p_runtime(obs) or int(obs.P) == 0:
        return base

    step = int(obs.step.reshape(-1)[0].item())
    if step < int(TERRITORY_REGROUP_START):
        return base

    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    owned = obs.owned & obs.alive
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    d0 = cache.cross_dist[0].to(dtype)

    anchor = owned & (prod >= 2.0)
    threshold = prod * 8.0 + 12.0
    thin = ((threshold - ships).clamp(min=0.0) / threshold.clamp(min=1.0))

    non_mine = obs.alive & ~owned
    if bool(non_mine.any()):
        nearest_non_mine = torch.where(
            non_mine.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).amin(dim=0)
        frontier = (1.0 - nearest_non_mine / float(ENEMY_DANGER_DISTANCE)).clamp(min=0.0, max=1.0)
    else:
        frontier = torch.zeros(P, dtype=dtype, device=device)

    friend_near = (d0 <= float(ALLY_LINK_DISTANCE)) & owned.view(P, 1) & owned.view(1, P)
    friend_near = friend_near & ~torch.eye(P, dtype=torch.bool, device=device)
    friend_count = friend_near.sum(dim=0).to(dtype)
    isolation = (1.0 / (1.0 + friend_count)).clamp(min=0.0, max=1.0)

    graph_pressure = (
        prod * 22.0
        + prod * thin * 42.0
        + prod * frontier * 18.0
        + prod * isolation * 12.0
    )
    graph_pressure = torch.where(anchor, graph_pressure, torch.zeros_like(graph_pressure))

    comet = is_comet_planet({"planets": getattr(obs, "planets", None)}, P, device) if hasattr(obs, "planets") else None
    if comet is not None:
        graph_pressure = torch.where(comet, graph_pressure * 0.25, graph_pressure)

    remaining = max(0, _h14.TOTAL_STEPS - step)
    if remaining < 90:
        graph_pressure = graph_pressure * max(0.25, float(remaining) / 90.0)
    return base + graph_pressure * float(TERRITORY_REGROUP_WEIGHT)


def _territory_adjustment(*, obs, cache, cand_src, cand_send, cand_tgt_slot, cand_eta, cand_is_def, score):
    if int(obs.P) == 0 or int(score.numel()) == 0:
        return score

    P = int(obs.P)
    device = score.device
    dtype = score.dtype
    step = int(obs.step.reshape(-1)[0].item())
    if step < int(TERRITORY_START_TURN):
        return score
    tgt = cand_tgt_slot.clamp(0, P - 1)
    src = cand_src[:, 0].clamp(0, P - 1)
    d0 = cache.cross_dist[0].to(dtype)

    owned = obs.owned & obs.alive
    enemy = obs.is_enemy & obs.alive
    non_mine = obs.alive & ~owned
    prod = obs.prod.to(dtype)

    ally_near = (d0 <= float(ALLY_LINK_DISTANCE)) & owned.view(P, 1)
    ally_count = ally_near.sum(dim=0).to(dtype)
    ally_prod_near = (prod.view(P, 1) * ally_near.to(dtype)).sum(dim=0)
    connected = ally_count[tgt] > 0

    anchor_mask = owned & (prod >= 3.0)
    if bool(anchor_mask.any()):
        anchor_dist = torch.where(
            anchor_mask.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).amin(dim=0)
        anchor_factor = (1.0 - anchor_dist / float(ANCHOR_DISTANCE)).clamp(min=0.0, max=1.0)
    else:
        anchor_factor = torch.zeros(P, dtype=dtype, device=device)

    enemy_near = (d0 <= float(ENEMY_DANGER_DISTANCE)) & enemy.view(P, 1)
    enemy_count = enemy_near.sum(dim=0).to(dtype)
    enemy_prod_near = (prod.view(P, 1) * enemy_near.to(dtype)).sum(dim=0)
    isolated = (~connected) & (enemy_count[tgt] >= 1.0)

    is_attack = ~cand_is_def
    target_prod = prod[tgt]
    eta = cand_eta[:, 0].to(dtype)
    send = cand_send[:, 0].to(dtype)

    source_prod = prod[src]
    source_ships = obs.ships[src].to(dtype)
    source_after = source_ships - send
    source_ally_count = ally_count[src]
    source_enemy_count = enemy_count[src]
    source_anchor = (
        (source_prod >= 2.0)
        & (source_ships >= source_prod * float(SOURCE_ANCHOR_SHIPS_FACTOR) + 10.0)
    ) | (source_ally_count >= 2.0)
    source_thin_frontline = (
        (source_prod >= 1.0)
        & (source_enemy_count >= 1.0)
        & (source_after < source_prod * float(SOURCE_THIN_SHIPS_FACTOR) + 8.0)
    )
    src_to_tgt = d0[src, tgt]
    local_expansion = src_to_tgt <= float(ALLY_LINK_DISTANCE) * 1.35

    connect_bonus = connected.to(dtype) * (
        float(CONNECT_BONUS) + (ally_prod_near[tgt] * float(PROD_CONNECT_SCALE)).clamp(max=0.18)
    )
    anchor_bonus = anchor_factor[tgt] * float(ANCHOR_BONUS)
    source_anchor_bonus = source_anchor.to(dtype) * local_expansion.to(dtype) * float(SOURCE_ANCHOR_BONUS)
    local_expansion_bonus = (
        local_expansion.to(dtype)
        * (target_prod >= 2.0).to(dtype)
        * (0.08 + (target_prod * 0.025).clamp(max=0.10))
        + local_expansion.to(dtype) * float(LOCAL_EXPANSION_BONUS) * connected.to(dtype)
    )
    enemy_edge_bonus = (
        (enemy_prod_near[tgt] > 0).to(dtype)
        * (target_prod >= 2.0).to(dtype)
        * float(ENEMY_CLUSTER_EDGE_BONUS)
    )
    isolation_penalty = isolated.to(dtype) * (
        float(ISOLATION_PENALTY) + (enemy_count[tgt] * 0.04).clamp(max=0.14)
    )
    long_isolated_penalty = isolated.to(dtype) * (eta - 7.0).clamp(min=0.0) * 0.018
    thin_source_penalty = source_thin_frontline.to(dtype) * float(THIN_FRONTLINE_PENALTY)
    overextension_penalty = (
        isolated.to(dtype)
        * (~source_anchor).to(dtype)
        * (eta > 5.0).to(dtype)
        * float(OVEREXTENSION_PENALTY)
    )

    adjustment = (
        connect_bonus
        + anchor_bonus
        + source_anchor_bonus
        + local_expansion_bonus
        + enemy_edge_bonus
        - isolation_penalty
        - long_isolated_penalty
        - thin_source_penalty
        - overextension_penalty
    )
    adjustment = adjustment.clamp(min=-float(MAX_ABS_ADJUSTMENT), max=float(MAX_ABS_ADJUSTMENT))
    adjustment = torch.where(is_attack & non_mine[tgt], adjustment, torch.zeros_like(adjustment))
    return score + torch.where(torch.isfinite(score), adjustment, torch.zeros_like(score))


def _territory_target_score(obs, cache, *, dtype):
    P = int(obs.P)
    device = obs.device
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)

    d0 = cache.cross_dist[0].to(dtype)
    owned = obs.owned & obs.alive
    enemy = obs.is_enemy & obs.alive
    attackable = obs.alive & ~owned
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)

    ally_near = (d0 <= float(ALLY_LINK_DISTANCE)) & owned.view(P, 1)
    ally_count = ally_near.sum(dim=0).to(dtype)
    ally_prod = (prod.view(P, 1) * ally_near.to(dtype)).sum(dim=0)

    anchor_mask = owned & (prod >= 3.0)
    if bool(anchor_mask.any()):
        anchor_dist = torch.where(
            anchor_mask.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).amin(dim=0)
        anchor_factor = (1.0 - anchor_dist / float(ANCHOR_DISTANCE)).clamp(min=0.0, max=1.0)
    else:
        anchor_factor = torch.zeros(P, dtype=dtype, device=device)

    enemy_near = (d0 <= float(ENEMY_DANGER_DISTANCE)) & enemy.view(P, 1)
    enemy_count = enemy_near.sum(dim=0).to(dtype)
    enemy_prod = (prod.view(P, 1) * enemy_near.to(dtype)).sum(dim=0)
    source_anchor_mask = owned & (
        (prod >= 2.0)
        & (ships >= prod * float(SOURCE_ANCHOR_SHIPS_FACTOR) + 10.0)
    )
    if bool(source_anchor_mask.any()):
        source_anchor_dist = torch.where(
            source_anchor_mask.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).amin(dim=0)
        source_anchor_reach = (1.0 - source_anchor_dist / (float(ALLY_LINK_DISTANCE) * 1.55)).clamp(min=0.0, max=1.0)
    else:
        source_anchor_reach = torch.zeros(P, dtype=dtype, device=device)

    connected_value = (ally_count > 0).to(dtype) * (0.8 + ally_prod * 0.05)
    anchor_value = anchor_factor * 1.1
    source_anchor_value = source_anchor_reach * 1.25
    target_value = prod * 0.9 + torch.where(enemy, 0.8, 0.35)
    weak_enemy_value = enemy.to(dtype) * ((34.0 - ships).clamp(min=0.0) / 34.0) * 1.4
    edge_value = (enemy_count > 0).to(dtype) * (enemy_prod * 0.035).clamp(max=1.2)
    isolation_cost = ((ally_count <= 0) & (enemy_count > 0)).to(dtype) * (1.5 + enemy_count * 0.25)

    score = (
        target_value
        + connected_value
        + anchor_value
        + source_anchor_value
        + weak_enemy_value
        + edge_value
        - isolation_cost
    )
    return torch.where(attackable, score, torch.full_like(score, float("-inf")))


def _unique_preserve_order(idx, exists, *, P: int):
    seen = set()
    keep_idx = []
    keep_exists = []
    for i in range(int(idx.numel())):
        ex = bool(exists[i].item()) if int(exists.numel()) > i else False
        value = int(idx[i].item()) if ex else -1
        if ex and 0 <= value < int(P) and value not in seen:
            seen.add(value)
            keep_idx.append(idx[i])
            keep_exists.append(exists[i])
    if not keep_idx:
        device = idx.device
        return (
            torch.zeros(0, dtype=idx.dtype, device=device),
            torch.zeros(0, dtype=exists.dtype, device=device),
        )
    return torch.stack(keep_idx), torch.stack(keep_exists)


def _territory_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    base_idx, base_exists = _ORIGINAL_BUILD_TARGET_SHORTLIST(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not _is_4p_obs(obs_tensors):
        return base_idx, base_exists
    step = int(obs.step.reshape(-1)[0].item())
    if step < int(TERRITORY_START_TURN):
        return base_idx, base_exists

    P = int(obs.P)
    t_score = _territory_target_score(obs, cache, dtype=prod.dtype)
    top_k = max(1, min(int(TERRITORY_EXTRA_TARGETS), P))
    extra_idx = torch.argsort(t_score, descending=True, stable=True)[:top_k]
    extra_exists = torch.isfinite(t_score[extra_idx])
    merged_idx = torch.cat([base_idx, extra_idx.to(base_idx.dtype)], dim=0)
    merged_exists = torch.cat([base_exists, extra_exists.to(base_exists.dtype)], dim=0)
    return _unique_preserve_order(merged_idx, merged_exists, P=P)


def _territory_tier_candidates(**kwargs):
    result = _ORIGINAL_TIER_CANDIDATES(**kwargs)
    player_count = int(kwargs.get("player_count", 2))
    if player_count < 4:
        return result

    (
        cand_src,
        cand_send,
        cand_angle,
        cand_eta,
        cand_active,
        cand_tgt_slot,
        cand_tgt_short,
        cand_is_def,
        score,
    ) = result
    adjusted = _territory_adjustment(
        obs=kwargs["obs"],
        cache=kwargs["cache"],
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_eta=cand_eta,
        cand_is_def=cand_is_def,
        score=score,
    )
    return (
        cand_src,
        cand_send,
        cand_angle,
        cand_eta,
        cand_active,
        cand_tgt_slot,
        cand_tgt_short,
        cand_is_def,
        adjusted,
    )


def agent(obs):
    _h14.build_target_shortlist = _territory_build_target_shortlist
    _h14._tier_candidates = _territory_tier_candidates
    # Regroup-pressure replacement was too invasive in smoke tests: it pulled
    # supply ships away from essential local defenses. Keep the helper above as
    # a research note, but only patch the attack-candidate graph layer for now.
    _h14.cheap_enemy_pressure = _ORIGINAL_CHEAP_ENEMY_PRESSURE
    return _h14.agent(obs)
