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
import hairate30_2p_h14_4p_h29 as _base


RING_PHASE_END = 210
RING_PHASE_FADE = 80
RING_EXTRA_TARGETS = 6
RING_CLOSE_BRIDGE_DISTANCE = 44.0
RING_ANCHOR_DISTANCE = 86.0
RING_BRIDGE_BONUS = 1.20
RING_ANCHOR_BONUS = 0.48
RING_BORDER_BONUS = 0.16
RING_SOURCE_RESERVE_FACTOR = 7.5
RING_SOURCE_RESERVE_BASE = 16.0
RING_DRAIN_PENALTY = 0.34


def _ring_phase(step: int) -> float:
    if int(step) <= int(RING_PHASE_END):
        return 1.0
    return max(0.0, 1.0 - (float(step) - float(RING_PHASE_END)) / float(RING_PHASE_FADE))


def _my_start_slot(obs, obs_tensors: dict):
    initial = obs_tensors.get("initial_planets")
    if initial is None:
        return None
    owner0 = initial[:, 1].long().to(obs.device)
    slots = torch.where(owner0 == int(obs.player_id))[0]
    if int(slots.numel()) == 0:
        return None
    return slots[0]


def _ring_masks(obs, obs_tensors: dict, cache, home_sector, border, *, dtype):
    P = int(obs.P)
    device = obs.device
    z = torch.zeros(P, dtype=torch.bool, device=device)
    start_slot = _my_start_slot(obs, obs_tensors)
    if start_slot is None:
        owned_home = obs.owned & home_sector
        if bool(owned_home.any()):
            start_dist = torch.where(
                owned_home.view(P, 1),
                cache.cross_dist[0].to(dtype),
                torch.full((P, P), float("inf"), dtype=dtype, device=device),
            ).amin(dim=0)
        else:
            start_dist = torch.full((P,), float("inf"), dtype=dtype, device=device)
    else:
        start_dist = cache.cross_dist[0, start_slot, :].to(dtype)

    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    attackable = obs.alive & ~obs.owned
    home_or_border = home_sector | border
    # Type B seed family: close high-production bridge planets plus expensive medium ring anchors.
    # Keep the planner active after the bridges are captured; otherwise it never reaches the ring phase.
    bridge_like = (
        home_sector
        & (prod >= 5.0)
        & (start_dist <= float(RING_CLOSE_BRIDGE_DISTANCE))
    )
    anchor_like = (
        home_or_border
        & (ships >= 40.0)
        & (ships <= 60.0)
        & (prod >= 2.0)
        & (start_dist <= float(RING_ANCHOR_DISTANCE))
    )
    bridge = attackable & bridge_like & (ships <= 32.0)
    anchor = attackable & anchor_like
    bridge_count = int(bridge_like.sum().item())
    anchor_count = int(anchor_like.sum().item())
    owned_bridge_count = int((obs.owned & bridge_like).sum().item())
    active = bridge_count >= 2 and anchor_count >= 1
    anchor_ready = owned_bridge_count >= 2

    ring_score = (
        bridge.to(dtype) * (prod * 5.0 + (30.0 - ships).clamp(min=0.0) * 0.20)
        + anchor.to(dtype) * float(anchor_ready) * (prod * 4.0 + (70.0 - start_dist).clamp(min=0.0) * 0.10)
        + border.to(dtype) * anchor.to(dtype) * 3.0
    )
    ring_score = torch.where(active & attackable, ring_score, torch.full_like(ring_score, float("-inf")))
    return {
        "active": bool(active),
        "start_dist": start_dist,
        "bridge": bridge,
        "anchor": anchor,
        "owned_bridge_count": owned_bridge_count,
        "ring_score": ring_score,
    }


def _ring_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    base_idx, base_exists = _base._home_build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not _base._is_4p_obs(obs_tensors):
        return base_idx, base_exists

    home_sector, border = _base._current_home_sector(obs)
    if home_sector is None or border is None:
        return base_idx, base_exists

    step = int(obs.step.reshape(-1)[0].item())
    if _ring_phase(step) <= 0.0:
        return base_idx, base_exists

    masks = _ring_masks(obs, obs_tensors, cache, home_sector, border, dtype=prod.dtype)
    if not masks["active"]:
        return base_idx, base_exists

    P = int(obs.P)
    k = max(1, min(int(RING_EXTRA_TARGETS), P))
    ring_idx = torch.argsort(masks["ring_score"], descending=True, stable=True)[:k]
    ring_exists = torch.isfinite(masks["ring_score"][ring_idx])
    merged_idx = torch.cat([ring_idx.to(base_idx.dtype), base_idx], dim=0)
    merged_exists = torch.cat([ring_exists.to(base_exists.dtype), base_exists], dim=0)
    return _base._unique_preserve_order(merged_idx, merged_exists, P=P)


def _ring_adjustment(*, obs, cache, cand_src, cand_send, cand_tgt_slot, cand_eta, cand_is_def, score):
    if int(obs.P) == 0 or int(score.numel()) == 0:
        return score

    home_sector, border = _base._current_home_sector(obs)
    if home_sector is None or border is None:
        return score

    step = int(obs.step.reshape(-1)[0].item())
    phase = _ring_phase(step)
    if phase <= 0.0:
        return score

    P = int(obs.P)
    dtype = score.dtype
    masks = _ring_masks(obs, {}, cache, home_sector, border, dtype=dtype)
    if not masks["active"]:
        return score

    src = cand_src[:, 0].clamp(0, P - 1)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)
    is_attack = ~cand_is_def

    bridge_tgt = masks["bridge"][tgt]
    anchor_tgt = masks["anchor"][tgt]
    anchor_ready = masks["owned_bridge_count"] >= 2
    border_tgt = border[tgt] & anchor_tgt
    fast_enough = eta <= 80.0

    attack_bonus = (
        bridge_tgt.to(dtype) * (float(RING_BRIDGE_BONUS) + (prod[tgt] * 0.05).clamp(max=0.20))
        + anchor_tgt.to(dtype) * float(anchor_ready) * float(RING_ANCHOR_BONUS)
        + border_tgt.to(dtype) * float(RING_BORDER_BONUS)
        - anchor_tgt.to(dtype) * float(not anchor_ready) * 0.22
        - (eta / 90.0).clamp(max=0.18) * anchor_tgt.to(dtype)
    ) * is_attack.to(dtype) * fast_enough.to(dtype) * float(phase)

    source_after = ships[src] - send
    source_frontier = home_sector[src] | border[src]
    reserve = prod[src] * float(RING_SOURCE_RESERVE_FACTOR) + float(RING_SOURCE_RESERVE_BASE)
    drain_gap = ((reserve - source_after).clamp(min=0.0) / reserve.clamp(min=1.0)).clamp(max=1.0)
    drain_penalty = (
        source_frontier.to(dtype)
        * is_attack.to(dtype)
        * (bridge_tgt | anchor_tgt).to(dtype)
        * drain_gap
        * float(RING_DRAIN_PENALTY)
        * float(phase)
    )

    adjusted = score + attack_bonus - drain_penalty
    return torch.where(torch.isfinite(score), adjusted, score)


def _ring_tier_candidates(**kwargs):
    result = _base._home_tier_candidates(**kwargs)
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
    adjusted = _ring_adjustment(
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
    _h14.build_target_shortlist = _ring_build_target_shortlist
    _h14._tier_candidates = _ring_tier_candidates
    return _h14.agent(obs)
