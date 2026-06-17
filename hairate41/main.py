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

_MODE: str | None = None
_MODE_PLAYER: int | None = None
_LAST_ROUTE_STEP: int | None = None
_LAST_ROUTE_PLAYER: int | None = None
_LAST_ROUTE_SCORE = None
_LAST_ROUTE_ETA = None
_LAST_ROUTE_ACTIVE: bool = False
_ROUTE_GAME_ALLOWED: bool | None = None
_ROUTE_GAME_PLAYER: int | None = None


def _read(obs, name: str, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _route_phase(step: int) -> float:
    if int(step) <= int(ROUTE_END_TURN):
        return 1.0
    return max(0.0, 1.0 - (float(step) - float(ROUTE_END_TURN)) / float(ROUTE_FADE_TURNS))


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
    if os.environ.get("HAIRATE41_DEBUG") and (_LAST_ROUTE_STEP <= 2 or bool(active)):
        try:
            finite = score[torch.isfinite(score)]
            potential = float(finite.clamp(min=0.0).sort(descending=True).values[:5].sum().item()) if int(finite.numel()) else 0.0
            top = []
            if int(score.numel()) and bool(torch.isfinite(score).any()):
                for idx in torch.argsort(score, descending=True, stable=True)[:6].tolist():
                    if bool(torch.isfinite(score[idx])):
                        top.append((int(idx), round(float(score[idx].item()), 3), round(float(eta[idx].item()), 1)))
            with open(os.environ.get("HAIRATE41_DEBUG"), "a", encoding="utf-8") as f:
                f.write(f"step={_LAST_ROUTE_STEP} player={_LAST_ROUTE_PLAYER} active={bool(active)} potential={potential:.3f} top={top}\n")
        except Exception:
            pass
    return score, eta, active


def _current_route(obs):
    step = int(obs.step.reshape(-1)[0].item())
    if _LAST_ROUTE_STEP != step or _LAST_ROUTE_PLAYER != int(obs.player_id):
        return None, None, False
    return _LAST_ROUTE_SCORE, _LAST_ROUTE_ETA, bool(_LAST_ROUTE_ACTIVE)


def _route_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    base_idx, base_exists = _stable._home_build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    route_score, _route_eta, active = _store_route(obs, obs_tensors, cache, source_mask, dtype=prod.dtype)
    if not active or not bool(torch.isfinite(route_score).any()):
        return base_idx, base_exists

    P = int(obs.P)
    k = max(1, min(int(ROUTE_EXTRA_TARGETS), P))
    route_idx = torch.argsort(route_score, descending=True, stable=True)[:k]
    route_exists = torch.isfinite(route_score[route_idx])
    merged_idx = torch.cat([route_idx.to(base_idx.dtype), base_idx], dim=0)
    merged_exists = torch.cat([route_exists.to(base_exists.dtype), base_exists], dim=0)
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


def _route_tier_candidates(**kwargs):
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
        _h14.build_target_shortlist = _route_build_target_shortlist
        _h14._tier_candidates = _route_tier_candidates


def agent(obs):
    global _MODE, _MODE_PLAYER
    step = int(_read(obs, "step", 0) or 0)
    player = int(_read(obs, "player", 0) or 0)
    if step == 0 or _MODE is None or _MODE_PLAYER != player:
        _MODE = _selector._select_mode(obs)
        _MODE_PLAYER = player
    _install_mode(_MODE)
    return _h14.agent(obs)
