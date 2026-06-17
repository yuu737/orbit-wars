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
import hairate30_2p_h14_4p_h29 as _stable
import hairate32_4p_home_cluster_planner as _cluster
import hairate38_safe_strategy_selector as _selector
from orbit_lite.geometry import fleet_speed


ROUTE_END_TURN = 145
ROUTE_FADE_TURNS = 45
ROUTE_EXTRA_TARGETS = 6
ROUTE_MIN_POTENTIAL = 5.4
ROUTE_MAX_ADJUST = 1.28
ROUTE_SOURCE_PENALTY = 0.20
ROUTE_ETA_PENALTY = 0.010

ANCHOR_END_TURN = 170
ANCHOR_FADE_TURNS = 70
ANCHOR_EXTRA_TARGETS = 5
ANCHOR_MIN_POTENTIAL = 2.2
ANCHOR_MAX_ANCHORS = 2
ANCHOR_NEAR_DIST = 58.0
ANCHOR_EXPAND_DIST = 112.0
ANCHOR_RESERVE_PROD = 10.0
ANCHOR_RESERVE_BASE = 40.0
ANCHOR_MAX_ADJUST = 2.40
ANCHOR_CAPTURE_BONUS = 2.40
ANCHOR_SOURCE_PENALTY = 0.62
ANCHOR_REINFORCE_BONUS = 0.65
ANCHOR_EXPAND_BONUS = 0.58

_MODE: str | None = None
_MODE_PLAYER: int | None = None
_LAST_ROUTE_STEP: int | None = None
_LAST_ROUTE_PLAYER: int | None = None
_LAST_ROUTE_SCORE = None
_LAST_ROUTE_ETA = None
_LAST_ROUTE_ACTIVE: bool = False
_ROUTE_GAME_ALLOWED: bool | None = None
_ROUTE_GAME_PLAYER: int | None = None
_ANCHOR_GAME_PLAYER: int | None = None
_ANCHOR_GAME_ALLOWED: bool | None = None
_ANCHOR_GAME_IDX = None
_LAST_ANCHOR_STEP: int | None = None
_LAST_ANCHOR_PLAYER: int | None = None
_LAST_ANCHOR_SCORE = None
_LAST_ANCHOR_NEAR_SCORE = None
_LAST_ANCHOR_EXPAND_SCORE = None


def _read(obs, name: str, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _phase(step: int, end_turn: int, fade_turns: int) -> float:
    if int(step) <= int(end_turn):
        return 1.0
    return max(0.0, 1.0 - (float(step) - float(end_turn)) / float(fade_turns))


def _route_phase(step: int) -> float:
    return _phase(step, ROUTE_END_TURN, ROUTE_FADE_TURNS)


def _anchor_phase(step: int) -> float:
    return _phase(step, ANCHOR_END_TURN, ANCHOR_FADE_TURNS)


def _initial_owner_slots(obs_tensors: dict, *, player_id: int, device):
    initial = obs_tensors.get("initial_planets")
    if initial is None:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z
    owner0 = initial[:, 1].long().to(device)
    alive0 = initial[:, 0].to(device) >= 0
    my_start = torch.where(alive0 & (owner0 == int(player_id)))[0]
    enemy_start = torch.where(alive0 & (owner0 >= 0) & (owner0 != int(player_id)))[0]
    return my_start, enemy_start


def _min_current_eta(obs, cache, source_mask, *, dtype):
    P = int(obs.P)
    device = obs.device
    if P <= 0 or not bool(source_mask.any()):
        return torch.full((P,), float("inf"), dtype=dtype, device=device)
    src = torch.where(source_mask)[0]
    source_ships = obs.ships[src].to(dtype).clamp(min=1.0)
    speed = fleet_speed((source_ships * 0.58).clamp(min=1.0)).clamp(min=1e-6)
    eta = cache.cross_dist[0, src, :].to(dtype) / speed.view(-1, 1)
    return eta.amin(dim=0)


def _route_scores(obs, obs_tensors, cache, source_mask, *, dtype):
    P = int(obs.P)
    device = obs.device
    if P <= 0:
        z = torch.full((P,), float("-inf"), dtype=dtype, device=device)
        return z, torch.full((P,), float("inf"), dtype=dtype, device=device), False

    try:
        if int(_h14.largest_initial_player_count(obs_tensors)) < 4:
            z = torch.full((P,), float("-inf"), dtype=dtype, device=device)
            return z, torch.full((P,), float("inf"), dtype=dtype, device=device), False
    except Exception:
        z = torch.full((P,), float("-inf"), dtype=dtype, device=device)
        return z, torch.full((P,), float("inf"), dtype=dtype, device=device), False

    step = int(obs.step.reshape(-1)[0].item())
    phase = _route_phase(step)
    if phase <= 0.0:
        z = torch.full((P,), float("-inf"), dtype=dtype, device=device)
        return z, torch.full((P,), float("inf"), dtype=dtype, device=device), False

    my_start, enemy_start = _initial_owner_slots(obs_tensors, player_id=int(obs.player_id), device=device)
    if int(my_start.numel()) == 0 or int(enemy_start.numel()) == 0:
        owner_now = obs.owner_abs.long()
        my_start = torch.where(obs.alive & (owner_now == int(obs.player_id)))[0]
        enemy_start = torch.where(obs.alive & (owner_now >= 0) & (owner_now != int(obs.player_id)))[0]
    if int(my_start.numel()) == 0 or int(enemy_start.numel()) == 0:
        z = torch.full((P,), float("-inf"), dtype=dtype, device=device)
        return z, torch.full((P,), float("inf"), dtype=dtype, device=device), False

    home_sector, border = _stable._compute_home_sector(obs, obs_tensors, cache)
    home_or_border = home_sector | border
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    d0 = cache.cross_dist[0].to(dtype)

    my_start_dist = d0[my_start, :].amin(dim=0)
    enemy_start_dist = d0[enemy_start, :].amin(dim=0)
    enemy_margin = enemy_start_dist - my_start_dist
    my_eta = _min_current_eta(obs, cache, source_mask, dtype=dtype)

    attackable = obs.alive & ~obs.owned & (obs.is_neutral | obs.is_enemy)
    route_area = attackable & home_or_border & (enemy_margin >= -10.0) & (my_eta <= 92.0)

    near = (
        route_area
        & obs.is_neutral
        & (my_start_dist >= 24.0)
        & (my_start_dist <= 92.0)
        & (ships >= 12.0)
        & (ships <= 38.0)
        & (prod >= 1.0)
    )
    high = (
        route_area
        & (my_start_dist >= 58.0)
        & (my_start_dist <= 158.0)
        & ((prod >= 3.0) | (ships >= 40.0))
        & (ships <= 92.0)
    )
    ring = (
        route_area
        & obs.is_neutral
        & (my_start_dist >= 62.0)
        & (my_start_dist <= 150.0)
        & (ships >= 38.0)
        & (ships <= 62.0)
        & (prod >= 3.0)
        & (enemy_margin >= -4.0)
    )

    route_score = torch.zeros(P, dtype=dtype, device=device)
    chain_value = torch.zeros(P, dtype=dtype, device=device)
    if bool(near.any()) and bool(high.any()):
        start = my_start[0]
        sx = obs.x.to(dtype) - obs.x[start].to(dtype)
        sy = obs.y.to(dtype) - obs.y[start].to(dtype)
        near_idx = torch.where(near)[0]
        high_idx = torch.where(high)[0]
        for n in near_idx.tolist():
            nvx = sx[n]
            nvy = sy[n]
            n_norm = torch.sqrt(nvx * nvx + nvy * nvy).clamp(min=1e-6)
            hx = sx[high_idx]
            hy = sy[high_idx]
            h_norm = torch.sqrt(hx * hx + hy * hy).clamp(min=1e-6)
            cos = (nvx * hx + nvy * hy) / (n_norm * h_norm)
            link_dist = d0[n, high_idx]
            link_ok = (cos >= 0.42) & (link_dist <= 92.0) & (my_start_dist[high_idx] >= my_start_dist[n] - 8.0)
            if not bool(link_ok.any()):
                continue
            linked = high_idx[link_ok]
            outer_value = (prod[linked] * 0.95 + ships[linked].clamp(max=90.0) * 0.045 + enemy_margin[linked].clamp(min=-8.0, max=28.0) * 0.035)
            best_outer = outer_value.max()
            chain_value[n] = torch.maximum(chain_value[n], best_outer * 0.36 + prod[n] * 0.85 + ships[n] * 0.035)
            chain_value[linked] = torch.maximum(
                chain_value[linked],
                prod[linked] * 1.05 + ships[linked].clamp(max=90.0) * 0.050 + prod[n] * 0.24,
            )

    ring_count = ring.sum().to(dtype)
    ring_value = ring.to(dtype) * (prod * 1.15 + ships.clamp(max=70.0) * 0.035 + ring_count.clamp(max=5.0) * 0.28)
    near_value = near.to(dtype) * (prod * 0.95 + ships * 0.032 + enemy_margin.clamp(min=-8.0, max=22.0) * 0.040)
    route_score = torch.maximum(route_score, chain_value)
    route_score = torch.maximum(route_score, ring_value)
    route_score = torch.maximum(route_score, near_value)
    route_score = route_score + ring.to(dtype) * 0.85
    route_score = route_score - my_eta.clamp(max=120.0) * float(ROUTE_ETA_PENALTY)

    route_score = torch.where(route_area, route_score, torch.full_like(route_score, float("-inf")))
    finite = torch.isfinite(route_score)
    if not bool(finite.any()):
        return route_score, my_eta, False

    top = torch.sort(route_score[finite], descending=True).values[:5]
    potential = float(top.clamp(min=0.0).sum().item())
    active = potential >= float(ROUTE_MIN_POTENTIAL)
    if not active:
        route_score = torch.full_like(route_score, float("-inf"))
    return route_score * float(phase), my_eta, active


def _store_route(obs, obs_tensors, cache, source_mask, *, dtype):
    global _LAST_ROUTE_STEP, _LAST_ROUTE_PLAYER, _LAST_ROUTE_SCORE, _LAST_ROUTE_ETA, _LAST_ROUTE_ACTIVE
    global _ROUTE_GAME_ALLOWED, _ROUTE_GAME_PLAYER
    score, eta, active = _route_scores(obs, obs_tensors, cache, source_mask, dtype=dtype)
    _LAST_ROUTE_STEP = int(obs.step.reshape(-1)[0].item())
    _LAST_ROUTE_PLAYER = int(obs.player_id)
    if _LAST_ROUTE_STEP == 0 or _ROUTE_GAME_ALLOWED is None or _ROUTE_GAME_PLAYER != _LAST_ROUTE_PLAYER:
        _ROUTE_GAME_ALLOWED = bool(active)
        _ROUTE_GAME_PLAYER = _LAST_ROUTE_PLAYER
    if not bool(_ROUTE_GAME_ALLOWED):
        score = torch.full_like(score, float("-inf"))
        active = False
    _LAST_ROUTE_SCORE = score
    _LAST_ROUTE_ETA = eta
    _LAST_ROUTE_ACTIVE = bool(active)
    return score, eta, active


def _current_route(obs):
    step = int(obs.step.reshape(-1)[0].item())
    if _LAST_ROUTE_STEP != step or _LAST_ROUTE_PLAYER != int(obs.player_id):
        return None, None, False
    return _LAST_ROUTE_SCORE, _LAST_ROUTE_ETA, bool(_LAST_ROUTE_ACTIVE)


def _anchor_scores(obs, obs_tensors, cache, source_mask, route_score, route_eta, route_active, *, dtype):
    P = int(obs.P)
    device = obs.device
    neg = torch.full((P,), float("-inf"), dtype=dtype, device=device)
    zero = torch.zeros(P, dtype=dtype, device=device)
    if P <= 0:
        return neg, neg, neg, False

    step = int(obs.step.reshape(-1)[0].item())
    phase = _anchor_phase(step)
    if phase <= 0.0:
        return neg, neg, neg, False

    my_start, enemy_start = _initial_owner_slots(obs_tensors, player_id=int(obs.player_id), device=device)
    if int(my_start.numel()) == 0 or int(enemy_start.numel()) == 0:
        return neg, neg, neg, False

    home_sector, border = _stable._compute_home_sector(obs, obs_tensors, cache)
    home_or_border = home_sector | border
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    d0 = cache.cross_dist[0].to(dtype)
    my_start_dist = d0[my_start, :].amin(dim=0)
    enemy_start_dist = d0[enemy_start, :].amin(dim=0)
    enemy_margin = enemy_start_dist - my_start_dist
    my_eta = route_eta if route_eta is not None else _min_current_eta(obs, cache, source_mask, dtype=dtype)

    alive = obs.alive
    reachable_anchor = (my_eta <= 104.0) | obs.owned
    anchor_mask = (
        alive
        & home_or_border
        & (obs.owned | obs.is_neutral)
        & reachable_anchor
        & (enemy_margin >= -12.0)
        & ((prod >= 1.0) | (ships >= 12.0))
        & (my_start_dist >= 24.0)
        & (my_start_dist <= 150.0)
    )

    near_link = (d0 <= float(ANCHOR_NEAR_DIST)) & home_or_border.view(1, P) & alive.view(1, P)
    near_neutral = near_link & obs.is_neutral.view(1, P)
    near_owned = near_link & obs.owned.view(1, P)
    near_prod_sum = (near_neutral.to(dtype) * prod.view(1, P)).sum(dim=1)
    near_mid_count = (near_neutral & (ships.view(1, P) >= 8.0) & (ships.view(1, P) <= 42.0)).sum(dim=1).to(dtype)
    owned_support = near_owned.sum(dim=1).to(dtype)

    outer_link = (d0 <= float(ANCHOR_EXPAND_DIST)) & alive.view(1, P) & home_or_border.view(1, P)
    outer_value = (
        outer_link.to(dtype)
        * (prod.view(1, P) * 0.60 + ships.view(1, P).clamp(max=120.0) * 0.025)
        * (~obs.owned).view(1, P).to(dtype)
    ).amax(dim=1)

    if route_score is None:
        route_part = zero
    else:
        route_part = torch.where(torch.isfinite(route_score), route_score.clamp(min=0.0), zero)
    anchor_score = (
        prod * 1.15
        + ships.clamp(max=95.0) * 0.032
        + route_part * 0.60
        + near_prod_sum * 0.34
        + near_mid_count * 0.55
        + owned_support * 0.18
        + outer_value * 0.50
        + border.to(dtype) * 0.35
        + enemy_margin.clamp(min=-10.0, max=26.0) * 0.020
        - my_eta.clamp(max=130.0) * 0.008
    )
    anchor_score = torch.where(anchor_mask, anchor_score, neg)

    finite = torch.isfinite(anchor_score)
    if not bool(finite.any()):
        return neg, neg, neg, False
    top = torch.sort(anchor_score[finite], descending=True).values[: int(ANCHOR_MAX_ANCHORS)]
    potential = float(top.clamp(min=0.0).sum().item())
    active = potential >= float(ANCHOR_MIN_POTENTIAL)
    if not active:
        return neg, neg, neg, False

    top_idx = torch.argsort(anchor_score, descending=True, stable=True)[: int(ANCHOR_MAX_ANCHORS)]
    top_idx = top_idx[torch.isfinite(anchor_score[top_idx])]
    anchor_near = torch.full((P,), float("-inf"), dtype=dtype, device=device)
    anchor_expand = torch.full((P,), float("-inf"), dtype=dtype, device=device)
    attackable = alive & ~obs.owned
    for rank, a in enumerate(top_idx.tolist()):
        d = d0[int(a)]
        reserve = prod[int(a)] * float(ANCHOR_RESERVE_PROD) + float(ANCHOR_RESERVE_BASE)
        anchor_ready = (obs.owned[int(a)] and float(ships[int(a)].item()) >= float(reserve.item()) * 0.82) or step < 70
        rank_weight = 1.0 - 0.16 * float(rank)
        near_mask = attackable & home_or_border & (d <= float(ANCHOR_NEAR_DIST)) & (d > 0.0)
        near_value = (
            prod * 0.52
            + ships.clamp(max=75.0) * 0.020
            + route_part * 0.30
            + (d <= 38.0).to(dtype) * 0.22
        ) * rank_weight
        anchor_near = torch.maximum(anchor_near, torch.where(near_mask, near_value, torch.full_like(anchor_near, float("-inf"))))

        expand_mask = attackable & home_or_border & (d > float(ANCHOR_NEAR_DIST)) & (d <= float(ANCHOR_EXPAND_DIST)) & ((prod >= 3.0) | (ships >= 45.0))
        expand_value = (
            prod * 0.62
            + ships.clamp(max=120.0) * 0.026
            + route_part * 0.42
            - d * 0.006
        ) * rank_weight
        if not anchor_ready:
            expand_value = expand_value * 0.45
        anchor_expand = torch.maximum(anchor_expand, torch.where(expand_mask, expand_value, torch.full_like(anchor_expand, float("-inf"))))

    return anchor_score * float(phase), anchor_near * float(phase), anchor_expand * float(phase), True


def _store_anchor(obs, obs_tensors, cache, source_mask, route_score, route_eta, route_active, *, dtype):
    global _ANCHOR_GAME_PLAYER, _ANCHOR_GAME_ALLOWED, _ANCHOR_GAME_IDX
    global _LAST_ANCHOR_STEP, _LAST_ANCHOR_PLAYER, _LAST_ANCHOR_SCORE, _LAST_ANCHOR_NEAR_SCORE, _LAST_ANCHOR_EXPAND_SCORE

    anchor_score, near_score, expand_score, active = _anchor_scores(
        obs, obs_tensors, cache, source_mask, route_score, route_eta, route_active, dtype=dtype,
    )
    step = int(obs.step.reshape(-1)[0].item())
    player = int(obs.player_id)
    should_pick_anchor = (
        _ANCHOR_GAME_PLAYER != player
        or _ANCHOR_GAME_ALLOWED is None
        or (not bool(_ANCHOR_GAME_ALLOWED) and step <= 80)
    )
    if should_pick_anchor:
        if active and bool(torch.isfinite(anchor_score).any()):
            top_idx = torch.argsort(anchor_score, descending=True, stable=True)[: int(ANCHOR_MAX_ANCHORS)]
            top_idx = top_idx[torch.isfinite(anchor_score[top_idx])]
            _ANCHOR_GAME_IDX = top_idx.detach().clone()
            _ANCHOR_GAME_ALLOWED = int(top_idx.numel()) > 0
        else:
            if _ANCHOR_GAME_PLAYER != player or _ANCHOR_GAME_ALLOWED is None or step > 80:
                _ANCHOR_GAME_IDX = None
                _ANCHOR_GAME_ALLOWED = False
        _ANCHOR_GAME_PLAYER = player

    if not bool(_ANCHOR_GAME_ALLOWED) or _ANCHOR_GAME_IDX is None:
        anchor_score = torch.full_like(anchor_score, float("-inf"))
        near_score = torch.full_like(near_score, float("-inf"))
        expand_score = torch.full_like(expand_score, float("-inf"))
        active = False
    else:
        keep = torch.zeros_like(anchor_score, dtype=torch.bool)
        keep[_ANCHOR_GAME_IDX.clamp(0, int(obs.P) - 1)] = True
        anchor_score = torch.where(keep, anchor_score, torch.full_like(anchor_score, float("-inf")))
        active = active and bool(torch.isfinite(anchor_score).any())

    _LAST_ANCHOR_STEP = step
    _LAST_ANCHOR_PLAYER = player
    _LAST_ANCHOR_SCORE = anchor_score
    _LAST_ANCHOR_NEAR_SCORE = near_score
    _LAST_ANCHOR_EXPAND_SCORE = expand_score
    if os.environ.get("HAIRATE42_DEBUG") and step <= 3:
        try:
            top = []
            if int(anchor_score.numel()) and bool(torch.isfinite(anchor_score).any()):
                for idx in torch.argsort(anchor_score, descending=True, stable=True)[:4].tolist():
                    if bool(torch.isfinite(anchor_score[idx])):
                        top.append((int(idx), round(float(anchor_score[idx].item()), 3)))
            with open(os.environ.get("HAIRATE42_DEBUG"), "a", encoding="utf-8") as f:
                f.write(f"step={step} player={player} anchor_active={bool(active)} allowed={bool(_ANCHOR_GAME_ALLOWED)} top={top}\n")
        except Exception:
            pass
    return anchor_score, near_score, expand_score, active


def _current_anchor(obs):
    step = int(obs.step.reshape(-1)[0].item())
    if _LAST_ANCHOR_STEP != step or _LAST_ANCHOR_PLAYER != int(obs.player_id):
        return None, None, None, False
    active = (
        _LAST_ANCHOR_SCORE is not None
        and bool(torch.isfinite(_LAST_ANCHOR_SCORE).any())
    )
    return _LAST_ANCHOR_SCORE, _LAST_ANCHOR_NEAR_SCORE, _LAST_ANCHOR_EXPAND_SCORE, active


def _anchor_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    if os.environ.get("HAIRATE42_DEBUG"):
        try:
            step = int(obs.step.reshape(-1)[0].item())
            if step <= 3:
                with open(os.environ.get("HAIRATE42_DEBUG"), "a", encoding="utf-8") as f:
                    f.write(f"shortlist_entry step={step} player={int(obs.player_id)}\n")
        except Exception:
            pass
    base_idx, base_exists = _stable._home_build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    route_score, route_eta, route_active = _store_route(obs, obs_tensors, cache, source_mask, dtype=prod.dtype)
    if route_active and bool(torch.isfinite(route_score).any()):
        P = int(obs.P)
        k = max(1, min(int(ROUTE_EXTRA_TARGETS), P))
        route_idx = torch.argsort(route_score, descending=True, stable=True)[:k]
        route_exists = torch.isfinite(route_score[route_idx])
        base_idx = torch.cat([route_idx.to(base_idx.dtype), base_idx], dim=0)
        base_exists = torch.cat([route_exists.to(base_exists.dtype), base_exists], dim=0)
        base_idx, base_exists = _stable._unique_preserve_order(base_idx, base_exists, P=P)

    anchor_score, near_score, expand_score, anchor_active = _store_anchor(
        obs, obs_tensors, cache, source_mask, route_score, route_eta, route_active, dtype=prod.dtype,
    )
    if not anchor_active:
        return base_idx, base_exists

    P = int(obs.P)
    combined = torch.maximum(
        torch.where(torch.isfinite(anchor_score), anchor_score + 0.65, torch.full_like(anchor_score, float("-inf"))),
        torch.maximum(near_score, expand_score),
    )
    k = max(1, min(int(ANCHOR_EXTRA_TARGETS), P))
    anchor_idx = torch.argsort(combined, descending=True, stable=True)[:k]
    anchor_exists = torch.isfinite(combined[anchor_idx])
    merged_idx = torch.cat([anchor_idx.to(base_idx.dtype), base_idx], dim=0)
    merged_exists = torch.cat([anchor_exists.to(base_exists.dtype), base_exists], dim=0)
    return _stable._unique_preserve_order(merged_idx, merged_exists, P=P)


def _route_adjustment(*, obs, cand_src, cand_send, cand_tgt_slot, cand_eta, cand_is_def, score):
    if int(obs.P) == 0 or int(score.numel()) == 0:
        return score
    route_score, _route_eta, active = _current_route(obs)
    if not active or route_score is None or not bool(torch.isfinite(route_score).any()):
        return score

    P = int(obs.P)
    dtype = score.dtype
    src = cand_src[:, 0].clamp(0, P - 1)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)
    ships = obs.ships.to(dtype)
    prod = obs.prod.to(dtype)

    raw = route_score[tgt].clamp(min=0.0, max=float(ROUTE_MAX_ADJUST))
    timing = (eta <= 85.0).to(dtype)
    is_attack = (~cand_is_def).to(dtype)
    bonus = raw * timing * is_attack

    source_after = ships[src] - send
    reserve = prod[src] * 6.5 + 12.0
    drain_gap = ((reserve - source_after).clamp(min=0.0) / reserve.clamp(min=1.0)).clamp(max=1.0)
    source_penalty = bonus.gt(0.0).to(dtype) * drain_gap * float(ROUTE_SOURCE_PENALTY)

    adjusted = score + bonus - source_penalty
    return torch.where(torch.isfinite(score), adjusted, score)


def _anchor_adjustment(*, obs, cache, cand_src, cand_send, cand_tgt_slot, cand_eta, cand_is_def, score):
    anchor_score, near_score, expand_score, active = _current_anchor(obs)
    if not active or anchor_score is None or int(score.numel()) == 0:
        return score

    P = int(obs.P)
    dtype = score.dtype
    src = cand_src[:, 0].clamp(0, P - 1)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    is_attack = (~cand_is_def).to(dtype)

    anchor_mask = torch.isfinite(anchor_score)
    target_anchor = anchor_mask[tgt]
    target_anchor_score = torch.where(torch.isfinite(anchor_score[tgt]), anchor_score[tgt].clamp(min=0.0), torch.zeros_like(score))
    target_near = torch.where(torch.isfinite(near_score[tgt]), near_score[tgt].clamp(min=0.0), torch.zeros_like(score))
    target_expand = torch.where(torch.isfinite(expand_score[tgt]), expand_score[tgt].clamp(min=0.0), torch.zeros_like(score))
    target_bonus = (
        target_anchor.to(dtype) * target_anchor_score.clamp(max=float(ANCHOR_MAX_ADJUST)) * float(ANCHOR_CAPTURE_BONUS)
        + target_near.clamp(max=float(ANCHOR_MAX_ADJUST))
        + target_expand.clamp(max=float(ANCHOR_MAX_ADJUST)) * float(ANCHOR_EXPAND_BONUS)
    ) * is_attack * (eta <= 108.0).to(dtype)

    reinforce_need = (prod[tgt] * float(ANCHOR_RESERVE_PROD) + float(ANCHOR_RESERVE_BASE) - ships[tgt]).clamp(min=0.0)
    reinforce_bonus = (
        target_anchor.to(dtype)
        * cand_is_def.to(dtype)
        * (reinforce_need / (prod[tgt] * float(ANCHOR_RESERVE_PROD) + float(ANCHOR_RESERVE_BASE)).clamp(min=1.0)).clamp(max=1.0)
        * float(ANCHOR_REINFORCE_BONUS)
    )

    source_after = ships[src] - send
    source_reserve = prod[src] * float(ANCHOR_RESERVE_PROD) + float(ANCHOR_RESERVE_BASE)
    source_anchor = anchor_mask[src]
    source_gap = ((source_reserve - source_after).clamp(min=0.0) / source_reserve.clamp(min=1.0)).clamp(max=1.0)
    source_penalty = source_anchor.to(dtype) * is_attack * source_gap * float(ANCHOR_SOURCE_PENALTY)

    adjusted = score + target_bonus + reinforce_bonus - source_penalty
    return torch.where(torch.isfinite(score), adjusted, score)


def _anchor_tier_candidates(**kwargs):
    result = _stable._home_tier_candidates(**kwargs)
    player_count = int(kwargs.get("player_count", 2))
    if int(player_count) < 4:
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
    adjusted = _route_adjustment(
        obs=kwargs["obs"],
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_eta=cand_eta,
        cand_is_def=cand_is_def,
        score=score,
    )
    adjusted = _anchor_adjustment(
        obs=kwargs["obs"],
        cache=kwargs["cache"],
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_eta=cand_eta,
        cand_is_def=cand_is_def,
        score=adjusted,
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


def _install_mode(mode: str) -> None:
    if mode == "cluster":
        _h14.build_target_shortlist = _cluster._home_build_target_shortlist
        _h14._tier_candidates = _cluster._home_tier_candidates
    else:
        _h14.build_target_shortlist = _anchor_build_target_shortlist
        _h14._tier_candidates = _anchor_tier_candidates


def agent(obs):
    global _MODE, _MODE_PLAYER
    step = int(_read(obs, "step", 0) or 0)
    player = int(_read(obs, "player", 0) or 0)
    if step == 0 or _MODE is None or _MODE_PLAYER != player:
        _MODE = _selector._select_mode(obs)
        _MODE_PLAYER = player
    if os.environ.get("HAIRATE42_DEBUG") and step <= 3:
        try:
            with open(os.environ.get("HAIRATE42_DEBUG"), "a", encoding="utf-8") as f:
                f.write(f"agent_entry step={step} player={player} mode={_MODE}\n")
        except Exception:
            pass
    _install_mode(_MODE)
    return _h14.agent(obs)
