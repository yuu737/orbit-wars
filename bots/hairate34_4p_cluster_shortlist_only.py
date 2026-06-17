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


_ORIGINAL_TIER_CANDIDATES = _h14._tier_candidates
_ORIGINAL_BUILD_TARGET_SHORTLIST = _h14.build_target_shortlist

HOME_START_TURN = 0
HOME_PHASE_END = 155
HOME_PHASE_FADE = 90
HOME_EXTRA_ATTACK_TARGETS = 3
HOME_EXTRA_HOLD_TARGETS = 1
HOME_ATTACK_BONUS = 0.24
HOME_HIGH_PROD_BONUS = 0.07
OUTSIDE_ATTACK_PENALTY = 0.18
BORDER_ATTACK_BONUS = 0.12
HOME_DEFENSE_BONUS = 0.42
THIN_PROD_FACTOR = 8.0
THIN_BASE = 12.0
SOURCE_HOME_RESERVE_FACTOR = 7.0
SOURCE_HOME_RESERVE_BASE = 12.0
SOURCE_DRAIN_PENALTY = 0.24
BORDER_CONFIDENCE = 0.14
CLUSTER_END_TURN = 190
CLUSTER_LINK_DISTANCE = 42.0
CLUSTER_CORE_DISTANCE = 58.0
CLUSTER_EXTRA_TARGETS = 5
CLUSTER_CONNECT_BONUS = 0.46
CLUSTER_NEUTRAL_PROD_WEIGHT = 0.20
CLUSTER_FRIEND_PROD_WEIGHT = 0.16
CLUSTER_CORE_BONUS = 0.34
CLUSTER_ISOLATION_PENALTY = 0.54
CLUSTER_SOURCE_BONUS = 0.22
CLUSTER_MAX_ADJUST = 0.85

_LAST_SECTOR_STEP: int | None = None
_LAST_SECTOR_PLAYER: int | None = None
_LAST_HOME_SECTOR = None
_LAST_BORDER_SECTOR = None


def _is_4p_obs(obs_tensors: dict) -> bool:
    try:
        return int(_h14.largest_initial_player_count(obs_tensors)) >= 4
    except Exception:
        return False


def _phase_weight(step: int) -> float:
    if int(step) < int(HOME_START_TURN):
        return 0.0
    if int(step) <= int(HOME_PHASE_END):
        return 1.0
    return max(0.0, 1.0 - (float(step) - float(HOME_PHASE_END)) / float(HOME_PHASE_FADE))


def _compute_home_sector(obs, obs_tensors: dict, cache):
    P = int(obs.P)
    device = obs.device
    if P == 0:
        z = torch.zeros(P, dtype=torch.bool, device=device)
        return z, z

    initial = obs_tensors.get("initial_planets")
    if initial is None:
        z = torch.zeros(P, dtype=torch.bool, device=device)
        return z, z

    owner0 = initial[:, 1].long().to(device)
    home_slots = torch.where(owner0 >= 0)[0]
    if int(home_slots.numel()) == 0:
        z = torch.zeros(P, dtype=torch.bool, device=device)
        return z, z

    d = cache.cross_dist[0, home_slots, :].to(obs.ships.dtype)
    order = torch.argsort(d, dim=0, stable=True)
    nearest_pos = order[0]
    nearest_owner = owner0[home_slots[nearest_pos]]
    home_sector = nearest_owner == int(obs.player_id)

    if int(home_slots.numel()) >= 2:
        first = d.gather(0, order[0:1]).squeeze(0)
        second = d.gather(0, order[1:2]).squeeze(0)
        confidence = ((second - first) / second.clamp(min=1.0)).clamp(min=0.0)
        border = confidence <= float(BORDER_CONFIDENCE)
    else:
        border = torch.zeros(P, dtype=torch.bool, device=device)
    return home_sector, border


def _store_home_sector(obs, obs_tensors: dict, cache):
    global _LAST_SECTOR_STEP, _LAST_SECTOR_PLAYER, _LAST_HOME_SECTOR, _LAST_BORDER_SECTOR
    home, border = _compute_home_sector(obs, obs_tensors, cache)
    _LAST_SECTOR_STEP = int(obs.step.reshape(-1)[0].item())
    _LAST_SECTOR_PLAYER = int(obs.player_id)
    _LAST_HOME_SECTOR = home
    _LAST_BORDER_SECTOR = border
    return home, border


def _current_home_sector(obs):
    step = int(obs.step.reshape(-1)[0].item())
    if _LAST_SECTOR_STEP != step or _LAST_SECTOR_PLAYER != int(obs.player_id):
        return None, None
    return _LAST_HOME_SECTOR, _LAST_BORDER_SECTOR


def _friend_count(obs, cache, *, dtype):
    P = int(obs.P)
    device = obs.device
    owned = obs.owned & obs.alive
    d0 = cache.cross_dist[0].to(dtype)
    near = (d0 <= 38.0) & owned.view(P, 1) & owned.view(1, P)
    near = near & ~torch.eye(P, dtype=torch.bool, device=device)
    return near.sum(dim=0).to(dtype)


def _cluster_features(obs, cache, home_sector, border, *, dtype):
    P = int(obs.P)
    device = obs.device
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    owned = obs.owned & obs.alive
    attackable = obs.alive & ~owned
    d0 = cache.cross_dist[0].to(dtype)

    home_or_border = home_sector | border
    friend_near = (d0 <= float(CLUSTER_LINK_DISTANCE)) & owned.view(P, 1)
    home_neutral_near = (
        (d0 <= float(CLUSTER_LINK_DISTANCE))
        & obs.is_neutral.view(P, 1)
        & home_or_border.view(P, 1)
    )

    friend_count = friend_near.sum(dim=0).to(dtype)
    friend_prod = (friend_near.to(dtype) * prod.view(P, 1)).sum(dim=0)
    neutral_prod = (home_neutral_near.to(dtype) * prod.view(P, 1)).sum(dim=0)

    core_mask = owned & home_sector & (prod >= 2.0) & (ships >= prod * 7.0 + 10.0)
    if bool(core_mask.any()):
        core_dist = torch.where(
            core_mask.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).amin(dim=0)
        core_factor = (1.0 - core_dist / float(CLUSTER_CORE_DISTANCE)).clamp(min=0.0, max=1.0)
    else:
        core_factor = torch.zeros(P, dtype=dtype, device=device)

    home_attack = attackable & home_or_border
    connected = friend_count > 0
    expansion_value = (
        home_attack.to(dtype)
        * (
            prod * 0.34
            + connected.to(dtype) * float(CLUSTER_CONNECT_BONUS)
            + neutral_prod * float(CLUSTER_NEUTRAL_PROD_WEIGHT)
            + friend_prod * float(CLUSTER_FRIEND_PROD_WEIGHT)
            + core_factor * float(CLUSTER_CORE_BONUS)
            - (~connected).to(dtype) * float(CLUSTER_ISOLATION_PENALTY)
        )
    )
    return {
        "friend_count": friend_count,
        "friend_prod": friend_prod,
        "neutral_prod": neutral_prod,
        "core_factor": core_factor,
        "expansion_value": expansion_value,
        "home_or_border": home_or_border,
    }


def _home_shortlist_score(obs, cache, home_sector, border, *, dtype):
    P = int(obs.P)
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    owned = obs.owned & obs.alive
    attackable = obs.alive & ~owned
    friend_count = _friend_count(obs, cache, dtype=dtype)
    cluster = _cluster_features(obs, cache, home_sector, border, dtype=dtype)

    threshold = prod * float(THIN_PROD_FACTOR) + float(THIN_BASE)
    deficit = (threshold - ships).clamp(min=0.0)
    thin = deficit / threshold.clamp(min=1.0)

    attack_score = (
        prod * 1.15
        + home_sector.to(dtype) * (2.2 + prod * 0.55)
        + border.to(dtype) * (0.8 + prod * 0.20)
        + obs.is_enemy.to(dtype) * 0.85
        - (~home_sector & ~border).to(dtype) * 1.15
        + cluster["expansion_value"] * 2.2
    )
    attack_score = torch.where(attackable, attack_score, torch.full((P,), float("-inf"), dtype=dtype, device=obs.device))

    hold_score = (
        prod * 18.0
        + deficit * 1.2
        + thin * prod * 24.0
        + home_sector.to(dtype) * prod * 16.0
        + (friend_count <= 1.0).to(dtype) * prod * 8.0
    )
    hold_mask = owned & home_sector & (prod >= 2.0) & (deficit >= 5.0)
    hold_score = torch.where(hold_mask, hold_score, torch.full_like(hold_score, float("-inf")))
    return attack_score, hold_score


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


def _home_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    base_idx, base_exists = _ORIGINAL_BUILD_TARGET_SHORTLIST(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not _is_4p_obs(obs_tensors):
        return base_idx, base_exists

    home_sector, border = _store_home_sector(obs, obs_tensors, cache)
    step = int(obs.step.reshape(-1)[0].item())
    if _phase_weight(step) <= 0.0:
        return base_idx, base_exists

    P = int(obs.P)
    attack_score, hold_score = _home_shortlist_score(obs, cache, home_sector, border, dtype=prod.dtype)
    a_k = max(1, min(int(max(HOME_EXTRA_ATTACK_TARGETS, CLUSTER_EXTRA_TARGETS)), P))
    h_k = max(1, min(int(HOME_EXTRA_HOLD_TARGETS), P))
    attack_idx = torch.argsort(attack_score, descending=True, stable=True)[:a_k]
    hold_idx = torch.argsort(hold_score, descending=True, stable=True)[:h_k]
    attack_exists = torch.isfinite(attack_score[attack_idx])
    hold_exists = torch.isfinite(hold_score[hold_idx])
    merged_idx = torch.cat([base_idx, attack_idx.to(base_idx.dtype), hold_idx.to(base_idx.dtype)], dim=0)
    merged_exists = torch.cat([base_exists, attack_exists.to(base_exists.dtype), hold_exists.to(base_exists.dtype)], dim=0)
    return _unique_preserve_order(merged_idx, merged_exists, P=P)


def _home_adjustment(*, obs, cache, cand_src, cand_send, cand_tgt_slot, cand_eta, cand_is_def, score):
    if int(obs.P) == 0 or int(score.numel()) == 0:
        return score

    home_sector, border = _current_home_sector(obs)
    if home_sector is None or border is None:
        return score

    step = int(obs.step.reshape(-1)[0].item())
    phase = _phase_weight(step)
    if phase <= 0.0:
        return score

    P = int(obs.P)
    dtype = score.dtype
    src = cand_src[:, 0].clamp(0, P - 1)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)
    friend_count = _friend_count(obs, cache, dtype=dtype)
    cluster = _cluster_features(obs, cache, home_sector, border, dtype=dtype)

    is_attack = ~cand_is_def
    target_home = home_sector[tgt]
    target_border = border[tgt]
    target_outside = ~(target_home | target_border)
    threshold = prod * float(THIN_PROD_FACTOR) + float(THIN_BASE)
    target_deficit = (threshold[tgt] - ships[tgt]).clamp(min=0.0)
    target_thin = target_deficit / threshold[tgt].clamp(min=1.0)
    home_hold = cand_is_def & target_home & (prod[tgt] >= 2.0) & (target_deficit >= 4.0)

    attack_adjust = (
        target_home.to(dtype) * (float(HOME_ATTACK_BONUS) + (prod[tgt] * float(HOME_HIGH_PROD_BONUS)).clamp(max=0.55))
        + target_border.to(dtype) * float(BORDER_ATTACK_BONUS)
        - target_outside.to(dtype) * float(OUTSIDE_ATTACK_PENALTY)
    ) * is_attack.to(dtype) * float(phase)

    defense_adjust = (
        float(HOME_DEFENSE_BONUS)
        + target_thin * 1.2
        + (prod[tgt] * 0.08).clamp(max=0.45)
        - eta * 0.05
    ) * home_hold.to(dtype) * float(phase)

    source_after = ships[src] - send
    source_home_frontier = home_sector[src] & ((friend_count[src] <= 1.0) | border[src])
    reserve = prod[src] * float(SOURCE_HOME_RESERVE_FACTOR) + float(SOURCE_HOME_RESERVE_BASE)
    drain_gap = ((reserve - source_after).clamp(min=0.0) / reserve.clamp(min=1.0)).clamp(max=1.0)
    source_penalty = (
        source_home_frontier.to(dtype)
        * is_attack.to(dtype)
        * drain_gap
        * float(SOURCE_DRAIN_PENALTY)
        * float(phase)
    )

    adjusted = score + attack_adjust + defense_adjust - source_penalty
    return torch.where(torch.isfinite(score), adjusted, score)


def _home_tier_candidates(**kwargs):
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
    adjusted = _home_adjustment(
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
    _h14.build_target_shortlist = _home_build_target_shortlist
    _h14._tier_candidates = _home_tier_candidates
    return _h14.agent(obs)
