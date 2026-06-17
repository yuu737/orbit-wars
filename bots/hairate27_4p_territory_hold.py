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

FRIEND_LINK_DISTANCE = 38.0
FRONTIER_DISTANCE = 48.0
HOLD_START_TURN = 70
HOLD_EXTRA_TARGETS = 3
MIN_HOLD_PROD = 2.0
THIN_FACTOR = 8.0
THIN_BASE = 12.0
DEFENSE_BOOST = 0.95
DEFENSE_ETA_PENALTY = 0.08
FRONTIER_SOURCE_RESERVE_FACTOR = 8.0
FRONTIER_SOURCE_RESERVE_BASE = 14.0
ATTACK_PENALTY_START = 115
ATTACK_DRAIN_PENALTY = 1.25


def _is_4p_obs(obs_tensors: dict) -> bool:
    try:
        return int(_h14.largest_initial_player_count(obs_tensors)) >= 4
    except Exception:
        return False


def _friend_count_and_frontier(obs, cache, *, dtype):
    P = int(obs.P)
    device = obs.device
    owned = obs.owned & obs.alive
    non_mine = obs.alive & ~owned
    d0 = cache.cross_dist[0].to(dtype)

    friend_near = (d0 <= float(FRIEND_LINK_DISTANCE)) & owned.view(P, 1) & owned.view(1, P)
    friend_near = friend_near & ~torch.eye(P, dtype=torch.bool, device=device)
    friend_count = friend_near.sum(dim=0).to(dtype)

    if bool(non_mine.any()):
        nearest_non_mine = torch.where(
            non_mine.view(P, 1),
            d0,
            torch.full_like(d0, float("inf")),
        ).amin(dim=0)
        frontier = nearest_non_mine <= float(FRONTIER_DISTANCE)
    else:
        frontier = torch.zeros(P, dtype=torch.bool, device=device)
    return friend_count, frontier


def _hold_target_score(obs, cache, *, dtype):
    P = int(obs.P)
    device = obs.device
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)

    owned = obs.owned & obs.alive
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    friend_count, frontier = _friend_count_and_frontier(obs, cache, dtype=dtype)

    threshold = prod * float(THIN_FACTOR) + float(THIN_BASE)
    deficit = (threshold - ships).clamp(min=0.0)
    thin_ratio = deficit / threshold.clamp(min=1.0)
    isolated = friend_count <= 1.0

    score = (
        prod * 18.0
        + deficit * 1.35
        + thin_ratio * prod * 28.0
        + frontier.to(dtype) * prod * 18.0
        + isolated.to(dtype) * prod * 12.0
    )
    mask = owned & (prod >= float(MIN_HOLD_PROD)) & (deficit >= 6.0) & (frontier | isolated)
    return torch.where(mask, score, torch.full_like(score, float("-inf")))


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


def _hold_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    base_idx, base_exists = _ORIGINAL_BUILD_TARGET_SHORTLIST(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not _is_4p_obs(obs_tensors):
        return base_idx, base_exists
    step = int(obs.step.reshape(-1)[0].item())
    if step < int(HOLD_START_TURN):
        return base_idx, base_exists

    P = int(obs.P)
    hold_score = _hold_target_score(obs, cache, dtype=prod.dtype)
    top_k = max(1, min(int(HOLD_EXTRA_TARGETS), P))
    extra_idx = torch.argsort(hold_score, descending=True, stable=True)[:top_k]
    extra_exists = torch.isfinite(hold_score[extra_idx])
    merged_idx = torch.cat([base_idx, extra_idx.to(base_idx.dtype)], dim=0)
    merged_exists = torch.cat([base_exists, extra_exists.to(base_exists.dtype)], dim=0)
    return _unique_preserve_order(merged_idx, merged_exists, P=P)


def _hold_adjustment(*, obs, cache, cand_src, cand_send, cand_tgt_slot, cand_eta, cand_is_def, score):
    if int(obs.P) == 0 or int(score.numel()) == 0:
        return score

    P = int(obs.P)
    dtype = score.dtype
    step = int(obs.step.reshape(-1)[0].item())
    if step < int(HOLD_START_TURN):
        return score

    src = cand_src[:, 0].clamp(0, P - 1)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)
    friend_count, frontier = _friend_count_and_frontier(obs, cache, dtype=dtype)

    threshold = prod * float(THIN_FACTOR) + float(THIN_BASE)
    target_deficit = (threshold[tgt] - ships[tgt]).clamp(min=0.0)
    target_thin_ratio = target_deficit / threshold[tgt].clamp(min=1.0)
    target_is_hold = (
        cand_is_def
        & (prod[tgt] >= float(MIN_HOLD_PROD))
        & (target_deficit >= 4.0)
        & (frontier[tgt] | (friend_count[tgt] <= 1.0))
    )

    defense_bonus = (
        float(DEFENSE_BOOST)
        + target_thin_ratio * 1.4
        + (prod[tgt] * 0.08).clamp(max=0.45)
        - eta * float(DEFENSE_ETA_PENALTY)
    )
    defense_bonus = torch.where(target_is_hold, defense_bonus, torch.zeros_like(defense_bonus))

    source_after = ships[src] - send
    source_frontier = frontier[src] | (friend_count[src] <= 1.0)
    source_reserve = prod[src] * float(FRONTIER_SOURCE_RESERVE_FACTOR) + float(FRONTIER_SOURCE_RESERVE_BASE)
    bad_source_drain = (
        (~cand_is_def)
        & (step >= int(ATTACK_PENALTY_START))
        & source_frontier
        & (prod[src] >= 1.0)
        & (source_after < source_reserve)
    )
    drain_gap = ((source_reserve - source_after).clamp(min=0.0) / source_reserve.clamp(min=1.0)).clamp(max=1.0)
    guarded_score = score - bad_source_drain.to(dtype) * drain_gap * float(ATTACK_DRAIN_PENALTY)
    return guarded_score + torch.where(torch.isfinite(guarded_score), defense_bonus, torch.zeros_like(score))


def _hold_tier_candidates(**kwargs):
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
    adjusted = _hold_adjustment(
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
    _h14.build_target_shortlist = _hold_build_target_shortlist
    _h14._tier_candidates = _hold_tier_candidates
    return _h14.agent(obs)
