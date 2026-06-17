from __future__ import annotations

import math
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
from orbit_lite.geometry import fleet_speed


CHAIN_SELECT_THRESHOLD = 17.0
CHAIN_MID_MIN_SHIPS = 20.0
CHAIN_MID_MAX_SHIPS = 35.0
CHAIN_MID_DISTANCE = 78.0
CHAIN_HIGH_DISTANCE = 95.0
CHAIN_ANGLE_COS = 0.35
TYPE_A_HIGH_MIN_SHIPS = 70.0
NEAR_BRIDGE_DISTANCE = 82.0
BORDER_CONFIDENCE = 0.14

ANTI_START_STEP = 80
ANTI_END_STEP = 230
ANTI_EXTRA_TARGETS = 3
ANTI_REACH_ETA = 58.0
LEADER_OWNED_TARGET_BONUS = 0.22
LEADER_NEUTRAL_DENY_BONUS = 0.28
HIGH_PROD_BONUS = 0.08
SOURCE_DRAIN_PENALTY = 0.52
MAX_ANTI_ADJUST = 0.48

_MODE: str | None = None
_MODE_PLAYER: int | None = None


def _read(obs, name: str, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _infer_player_count(obs) -> int:
    explicit = _read(obs, "player_count", None)
    if explicit in (2, 4):
        return int(explicit)
    owners = [int(_read(obs, "player", 0) or 0)]
    for row in _read(obs, "planets", []) or []:
        if len(row) >= 2 and int(row[1]) >= 0:
            owners.append(int(row[1]))
    for row in _read(obs, "fleets", []) or []:
        if len(row) >= 2 and int(row[1]) >= 0:
            owners.append(int(row[1]))
    return 4 if max(owners, default=0) >= 2 else 2


def _dist(a, b) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _vec(a, b) -> tuple[float, float]:
    return float(b[2]) - float(a[2]), float(b[3]) - float(a[3])


def _cos_between(ax: float, ay: float, bx: float, by: float) -> float:
    an = math.hypot(ax, ay)
    bn = math.hypot(bx, by)
    if an <= 1e-6 or bn <= 1e-6:
        return -1.0
    return (ax * bx + ay * by) / (an * bn)


def _initial_geometry_rows(obs):
    planets = [list(p) for p in (_read(obs, "planets", []) or []) if len(p) >= 7 and int(p[0]) >= 0]
    initial = [list(p) for p in (_read(obs, "initial_planets", planets) or []) if len(p) >= 7 and int(p[0]) >= 0]
    init_by_id = {int(p[0]): p for p in initial}
    rows = []
    for p in planets:
        base = list(init_by_id.get(int(p[0]), p))
        base[1] = p[1]
        base[5] = p[5]
        rows.append(base)
    return rows


def _type_a_chain_score(obs, player: int) -> float:
    rows = _initial_geometry_rows(obs)
    starts = [p for p in rows if int(p[1]) >= 0]
    my_starts = [p for p in starts if int(p[1]) == int(player)]
    if not starts or not my_starts:
        return 0.0
    start = my_starts[0]

    mids = []
    highs = []
    for p in rows:
        if int(p[1]) >= 0:
            continue
        ships = float(p[5])
        nearest = sorted((_dist(p, s), int(s[1])) for s in starts)
        nearest_dist, nearest_owner = nearest[0]
        second_dist = nearest[1][0] if len(nearest) > 1 else 999.0
        confidence = (second_dist - nearest_dist) / max(second_dist, 1.0)
        is_homeish = nearest_owner == int(player) or confidence <= float(BORDER_CONFIDENCE)
        if not is_homeish:
            continue
        start_dist = _dist(start, p)
        if float(CHAIN_MID_MIN_SHIPS) <= ships <= float(CHAIN_MID_MAX_SHIPS) and start_dist <= float(CHAIN_MID_DISTANCE):
            mids.append(p)
        if ships >= float(TYPE_A_HIGH_MIN_SHIPS):
            highs.append(p)

    best = 0.0
    for mid in mids:
        sx, sy = _vec(start, mid)
        for high in highs:
            hx, hy = _vec(start, high)
            if _cos_between(sx, sy, hx, hy) < float(CHAIN_ANGLE_COS):
                continue
            high_dist = _dist(mid, high)
            if high_dist > float(CHAIN_HIGH_DISTANCE):
                continue
            value = float(mid[5]) * 0.05 + float(mid[6]) * 0.7 + float(high[5]) * 0.10 + float(high[6]) * 1.0
            best = max(best, value)
    return best


def _type_a_border_chain_score(obs, player: int) -> float:
    rows = _initial_geometry_rows(obs)
    starts = [p for p in rows if int(p[1]) >= 0]
    my_starts = [p for p in starts if int(p[1]) == int(player)]
    if not starts or not my_starts:
        return 0.0
    start = my_starts[0]

    near_bridge_max = 0.0
    border_high_prod = []
    for p in rows:
        if int(p[1]) >= 0:
            continue
        ships = float(p[5])
        prod = float(p[6])
        start_dist = _dist(start, p)
        nearest = sorted((_dist(p, s), int(s[1])) for s in starts)
        nearest_dist, nearest_owner = nearest[0]
        second_dist = nearest[1][0] if len(nearest) > 1 else 999.0
        confidence = (second_dist - nearest_dist) / max(second_dist, 1.0)
        is_homeish = nearest_owner == int(player) or confidence <= float(BORDER_CONFIDENCE)
        if not is_homeish:
            continue
        if start_dist <= float(NEAR_BRIDGE_DISTANCE):
            near_bridge_max = max(near_bridge_max, ships)
        if confidence <= float(BORDER_CONFIDENCE) and 58.0 <= start_dist <= 84.0 and prod >= 5.0 and 18.0 <= ships <= 35.0:
            border_high_prod.append(p)

    if near_bridge_max >= 28.0 and len(border_high_prod) >= 2:
        return near_bridge_max * 0.12 + sum(float(p[6]) * 1.3 for p in border_high_prod[:3])
    return 0.0


def _select_mode(obs) -> str:
    if _infer_player_count(obs) < 4:
        return "stable"
    player = int(_read(obs, "player", 0) or 0)
    if _type_a_chain_score(obs, player) >= float(CHAIN_SELECT_THRESHOLD):
        return "cluster"
    if _type_a_border_chain_score(obs, player) >= 13.0:
        return "cluster"
    return "stable"


def _in_anti_phase(obs, player_count: int) -> bool:
    if int(player_count) < 4:
        return False
    step = int(obs.step.reshape(-1)[0].item())
    return int(ANTI_START_STEP) <= step <= int(ANTI_END_STEP)


def _leader_info(obs, *, player_count: int):
    if int(player_count) < 4:
        return None
    P = int(obs.P)
    if P <= 0:
        return None

    device = obs.device
    dtype = obs.ships.dtype
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    owner = obs.owner_abs.long()
    alive = obs.alive
    pid = int(obs.player_id)

    scores = torch.full((max(4, int(player_count)),), float("-inf"), dtype=dtype, device=device)
    prods = torch.zeros_like(scores)
    for player in range(int(player_count)):
        mask = alive & (owner == int(player))
        owned_prod = prod[mask].sum() if bool(mask.any()) else torch.zeros((), dtype=dtype, device=device)
        owned_ships = ships[mask].sum() if bool(mask.any()) else torch.zeros((), dtype=dtype, device=device)
        high_prod_count = ((prod >= 4.0) & mask).sum().to(dtype)
        high_ship_count = ((ships >= 80.0) & mask).sum().to(dtype)
        scores[player] = owned_prod * 45.0 + owned_ships * 0.35 + high_prod_count * 70.0 + high_ship_count * 30.0
        prods[player] = owned_prod

    my_score = scores[pid]
    my_prod = prods[pid]
    scores[pid] = float("-inf")
    leader_id = int(torch.argmax(scores).item())
    leader_score = scores[leader_id]
    leader_prod = prods[leader_id]
    active = bool((leader_score >= my_score * 1.12).item() or (leader_prod >= my_prod + 4.0).item())
    if not active:
        return None
    return {
        "leader_id": leader_id,
        "leader_score": leader_score,
        "leader_prod": leader_prod,
        "my_score": my_score,
        "my_prod": my_prod,
    }


def _min_eta_from_sources(obs, cache, source_mask, *, dtype):
    P = int(obs.P)
    device = obs.device
    if P <= 0 or not bool(source_mask.any()):
        return torch.full((P,), float("inf"), dtype=dtype, device=device)
    source_slots = torch.where(source_mask)[0]
    source_ships = obs.ships[source_slots].to(dtype).clamp(min=1.0)
    # Use half-drain speed as a conservative reachability proxy.
    speed = fleet_speed((source_ships * 0.5).clamp(min=1.0)).clamp(min=1e-6)
    dist = cache.cross_dist[0, source_slots, :].to(dtype)
    eta = dist / speed.view(-1, 1)
    return eta.amin(dim=0)


def _anti_target_scores(obs, cache, source_mask, *, player_count: int, dtype):
    P = int(obs.P)
    device = obs.device
    info = _leader_info(obs, player_count=player_count)
    if info is None or P <= 0:
        return torch.full((P,), float("-inf"), dtype=dtype, device=device), None

    leader_id = int(info["leader_id"])
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    owner = obs.owner_abs.long()
    alive = obs.alive
    leader_owned = alive & (owner == leader_id)
    leader_sources = leader_owned & (ships >= 1.0)
    attackable = alive & ~obs.owned
    eta_from_me = _min_eta_from_sources(obs, cache, source_mask, dtype=dtype)
    reachable = eta_from_me <= float(ANTI_REACH_ETA)

    if bool(leader_sources.any()):
        leader_dist = torch.where(
            leader_sources.view(P, 1),
            cache.cross_dist[0].to(dtype),
            torch.full((P, P), float("inf"), dtype=dtype, device=device),
        ).amin(dim=0)
    else:
        leader_dist = torch.full((P,), float("inf"), dtype=dtype, device=device)

    owned_target = attackable & leader_owned & (prod >= 3.0)
    neutral_deny = attackable & obs.is_neutral & ((prod >= 3.0) | (ships >= 40.0)) & (leader_dist <= 48.0)
    target_score = (
        owned_target.to(dtype) * (float(LEADER_OWNED_TARGET_BONUS) + prod * float(HIGH_PROD_BONUS) + ships.clamp(max=120.0) * 0.002)
        + neutral_deny.to(dtype) * (float(LEADER_NEUTRAL_DENY_BONUS) + prod * float(HIGH_PROD_BONUS) + (52.0 - leader_dist).clamp(min=0.0) * 0.006)
    )
    target_score = torch.where(attackable & reachable, target_score, torch.full_like(target_score, float("-inf")))
    return target_score, info


def _anti_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    base_idx, base_exists = _stable._home_build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    try:
        player_count = int(_h14.largest_initial_player_count(obs_tensors))
    except Exception:
        player_count = 2
    if not _in_anti_phase(obs, player_count):
        return base_idx, base_exists

    target_score, _info = _anti_target_scores(obs, cache, source_mask, player_count=player_count, dtype=prod.dtype)
    if not bool(torch.isfinite(target_score).any()):
        return base_idx, base_exists

    P = int(obs.P)
    k = max(1, min(int(ANTI_EXTRA_TARGETS), P))
    anti_idx = torch.argsort(target_score, descending=True, stable=True)[:k]
    anti_exists = torch.isfinite(target_score[anti_idx])
    merged_idx = torch.cat([anti_idx.to(base_idx.dtype), base_idx], dim=0)
    merged_exists = torch.cat([anti_exists.to(base_exists.dtype), base_exists], dim=0)
    return _stable._unique_preserve_order(merged_idx, merged_exists, P=P)


def _anti_adjustment(*, obs, cache, cand_src, cand_send, cand_tgt_slot, cand_is_def, score, player_count: int):
    if int(obs.P) == 0 or int(score.numel()) == 0 or not _in_anti_phase(obs, player_count):
        return score

    source_mask = obs.owned & obs.alive & (obs.ships >= 1.0)
    target_score, _info = _anti_target_scores(obs, cache, source_mask, player_count=player_count, dtype=score.dtype)
    if not bool(torch.isfinite(target_score).any()):
        return score

    P = int(obs.P)
    dtype = score.dtype
    src = cand_src[:, 0].clamp(0, P - 1)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    send = cand_send[:, 0].to(dtype)
    is_attack = ~cand_is_def

    anti_bonus = target_score[tgt].clamp(min=0.0, max=float(MAX_ANTI_ADJUST)) * is_attack.to(dtype)
    source_after = ships[src] - send
    reserve = prod[src] * 7.0 + 14.0
    drain_gap = ((reserve - source_after).clamp(min=0.0) / reserve.clamp(min=1.0)).clamp(max=1.0)
    source_penalty = anti_bonus.gt(0.0).to(dtype) * drain_gap * float(SOURCE_DRAIN_PENALTY)

    adjusted = score + anti_bonus - source_penalty
    return torch.where(torch.isfinite(score), adjusted, score)


def _anti_tier_candidates(**kwargs):
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
    adjusted = _anti_adjustment(
        obs=kwargs["obs"],
        cache=kwargs["cache"],
        cand_src=cand_src,
        cand_send=cand_send,
        cand_tgt_slot=cand_tgt_slot,
        cand_is_def=cand_is_def,
        score=score,
        player_count=player_count,
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
        _h14.build_target_shortlist = _anti_build_target_shortlist
        _h14._tier_candidates = _anti_tier_candidates


def agent(obs):
    global _MODE, _MODE_PLAYER
    step = int(_read(obs, "step", 0) or 0)
    player = int(_read(obs, "player", 0) or 0)
    if step == 0 or _MODE is None or _MODE_PLAYER != player:
        _MODE = _select_mode(obs)
        _MODE_PLAYER = player
    _install_mode(_MODE)
    return _h14.agent(obs)
