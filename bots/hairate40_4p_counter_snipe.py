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

SNIPE_START_STEP = 58
SNIPE_END_STEP = 220
SNIPE_EXTRA_TARGETS = 2
SNIPE_MAX_ENEMY_ETA = 68.0
SNIPE_MAX_MY_ETA = 72.0
SNIPE_LINE_MARGIN = 18.0
SNIPE_MIN_TARGET_PROD = 4.0
SNIPE_MIN_TARGET_SHIPS = 45.0
SNIPE_NEUTRAL_BONUS = 0.28
SNIPE_OWN_DEFENSE_BONUS = 0.0
SNIPE_HIGH_PROD_BONUS = 0.085
SNIPE_TIMING_GRACE_BEFORE = 3.0
SNIPE_TIMING_GRACE_AFTER = 12.0
SNIPE_MAX_ADJUST = 0.46
SNIPE_SOURCE_DRAIN_PENALTY = 0.58
RUNAWAY_SNIPE_START_STEP = 32
RUNAWAY_SNIPE_END_STEP = 250
ENABLE_RUNAWAY_BROAD_SNIPE = False

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


def _in_snipe_phase(obs, player_count: int) -> bool:
    if int(player_count) < 4:
        return False
    step = int(obs.step.reshape(-1)[0].item())
    if _leader_runaway_active(obs, int(player_count)):
        return int(RUNAWAY_SNIPE_START_STEP) <= step <= int(RUNAWAY_SNIPE_END_STEP)
    return int(SNIPE_START_STEP) <= step <= int(SNIPE_END_STEP)


def _leader_runaway_active(obs, player_count: int) -> bool:
    if not bool(ENABLE_RUNAWAY_BROAD_SNIPE):
        return False
    if int(player_count) < 4 or int(obs.P) <= 0:
        return False
    step = int(obs.step.reshape(-1)[0].item())
    if step < 105:
        return False

    dtype = obs.ships.dtype
    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    owner = obs.owner_abs.long()
    alive = obs.alive
    pid = int(obs.player_id)
    scores = torch.full((max(4, int(player_count)),), float("-inf"), dtype=dtype, device=obs.device)
    prods = torch.zeros_like(scores)
    for player in range(int(player_count)):
        mask = alive & (owner == int(player))
        owned_prod = prod[mask].sum() if bool(mask.any()) else torch.zeros((), dtype=dtype, device=obs.device)
        owned_ships = ships[mask].sum() if bool(mask.any()) else torch.zeros((), dtype=dtype, device=obs.device)
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
    return bool((leader_score >= my_score * 1.55).item() or (leader_prod >= my_prod + 8.0).item())


def _visible_enemy_fleet_targets(obs, *, dtype, include_owned: bool, min_prod: float, min_ships: float, max_enemy_eta: float):
    """Estimate the first planet each currently visible enemy fleet is aimed at.

    This intentionally stays approximate and cheap. It only feeds a soft overlay:
    if a fleet is clearly travelling along a line toward a planet, we add that
    planet as a candidate and let the exact scorer/greedy selector still decide.
    """
    P = int(obs.P)
    F = int(obs.F)
    device = obs.device
    if P <= 0 or F <= 0:
        z_long = torch.zeros(0, dtype=torch.long, device=device)
        z = torch.zeros(0, dtype=dtype, device=device)
        return z_long, z_long, z, z

    enemy_fleet = obs.f_alive & (obs.f_owner >= 0.0) & (obs.f_owner != float(obs.player_id)) & (obs.f_ships >= 3.0)
    if not bool(enemy_fleet.any()):
        z_long = torch.zeros(0, dtype=torch.long, device=device)
        z = torch.zeros(0, dtype=dtype, device=device)
        return z_long, z_long, z, z

    fleet_slots = torch.where(enemy_fleet)[0]
    fx = obs.f_x[fleet_slots].to(dtype)
    fy = obs.f_y[fleet_slots].to(dtype)
    angle = obs.f_angle[fleet_slots].to(dtype)
    fships = obs.f_ships[fleet_slots].to(dtype)
    fowner = obs.f_owner[fleet_slots].long()

    ux = torch.cos(angle).view(-1, 1)
    uy = torch.sin(angle).view(-1, 1)
    dx = obs.x.to(dtype).view(1, P) - fx.view(-1, 1)
    dy = obs.y.to(dtype).view(1, P) - fy.view(-1, 1)
    along = dx * ux + dy * uy
    dist_sq = dx * dx + dy * dy
    perp_sq = (dist_sq - along * along).clamp(min=0.0)
    radius = obs.r.to(dtype).view(1, P) + float(SNIPE_LINE_MARGIN)
    speed = fleet_speed(fships).clamp(min=1e-6).view(-1, 1)
    eta = along / speed

    alive = obs.alive.view(1, P)
    not_owner = obs.owner_abs.long().view(1, P) != fowner.view(-1, 1)
    owner_mask = obs.is_neutral | (obs.owned if include_owned else torch.zeros_like(obs.owned))
    interesting_owner = owner_mask.view(1, P)
    interesting_value = ((obs.prod >= float(min_prod)) | (obs.ships >= float(min_ships))).view(1, P)
    can_hit = (
        alive
        & not_owner
        & interesting_owner
        & interesting_value
        & (along > 1.0)
        & (eta <= float(max_enemy_eta))
        & (perp_sq <= radius * radius)
    )
    if not bool(can_hit.any()):
        z_long = torch.zeros(0, dtype=torch.long, device=device)
        z = torch.zeros(0, dtype=dtype, device=device)
        return z_long, z_long, z, z

    ranked_along = torch.where(can_hit, along, torch.full_like(along, float("inf")))
    hit_pos = torch.argmin(ranked_along, dim=1)
    has_hit = torch.isfinite(ranked_along.gather(1, hit_pos.view(-1, 1)).squeeze(1))
    if not bool(has_hit.any()):
        z_long = torch.zeros(0, dtype=torch.long, device=device)
        z = torch.zeros(0, dtype=dtype, device=device)
        return z_long, z_long, z, z

    hit_targets = hit_pos[has_hit].long()
    hit_owners = fowner[has_hit].long()
    hit_ships = fships[has_hit]
    hit_eta = eta[has_hit, :].gather(1, hit_targets.view(-1, 1)).squeeze(1)
    return hit_targets, hit_owners, hit_ships, hit_eta


def _counter_snipe_scores(obs, cache, source_mask, *, player_count: int, dtype):
    P = int(obs.P)
    device = obs.device
    if P <= 0 or not _in_snipe_phase(obs, int(player_count)):
        return (
            torch.full((P,), float("-inf"), dtype=dtype, device=device),
            torch.full((P,), float("inf"), dtype=dtype, device=device),
        )

    runaway = _leader_runaway_active(obs, int(player_count))
    hit_targets, _hit_owners, hit_ships, hit_eta = _visible_enemy_fleet_targets(
        obs,
        dtype=dtype,
        include_owned=bool(runaway),
        min_prod=3.0 if runaway else float(SNIPE_MIN_TARGET_PROD),
        min_ships=35.0 if runaway else float(SNIPE_MIN_TARGET_SHIPS),
        max_enemy_eta=76.0 if runaway else float(SNIPE_MAX_ENEMY_ETA),
    )
    if int(hit_targets.numel()) == 0:
        return (
            torch.full((P,), float("-inf"), dtype=dtype, device=device),
            torch.full((P,), float("inf"), dtype=dtype, device=device),
        )

    prod = obs.prod.to(dtype)
    ships = obs.ships.to(dtype)
    score = torch.full((P,), float("-inf"), dtype=dtype, device=device)
    enemy_eta = torch.full((P,), float("inf"), dtype=dtype, device=device)

    for i in range(int(hit_targets.numel())):
        t = int(hit_targets[i].item())
        if t < 0 or t >= P:
            continue
        is_neutral = bool(obs.is_neutral[t].item())
        is_mine = bool(obs.owned[t].item())
        if not (is_neutral or (runaway and is_mine)):
            continue
        enemy_strength = float(hit_ships[i].item())
        target_ships = float(ships[t].item())
        prod_value = float(prod[t].item())
        if is_neutral and enemy_strength < max(4.0, target_ships * 0.55):
            continue

        base = prod[t] * float(SNIPE_HIGH_PROD_BONUS) + ships[t].clamp(max=90.0) * 0.003
        if is_neutral:
            value = base + (0.42 if runaway else float(SNIPE_NEUTRAL_BONUS))
        else:
            pressure = min(enemy_strength / max(target_ships + prod_value * 4.0 + 8.0, 1.0), 1.6)
            value = base + 0.30 * pressure

        score[t] = torch.maximum(score[t], value.to(dtype))
        enemy_eta[t] = torch.minimum(enemy_eta[t], hit_eta[i].to(dtype))

    finite = torch.isfinite(score)
    if not bool(finite.any()):
        return score, enemy_eta

    if bool(source_mask.any()):
        source_slots = torch.where(source_mask)[0]
        source_ships = obs.ships[source_slots].to(dtype).clamp(min=1.0)
        speed = fleet_speed((source_ships * 0.55).clamp(min=1.0)).clamp(min=1e-6)
        my_eta = (cache.cross_dist[0, source_slots, :].to(dtype) / speed.view(-1, 1)).amin(dim=0)
        reachable = my_eta <= (82.0 if runaway else float(SNIPE_MAX_MY_ETA))
        score = torch.where(reachable, score, torch.full_like(score, float("-inf")))
    return score, enemy_eta


def _snipe_build_target_shortlist(obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask):
    base_idx, base_exists = _stable._home_build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    try:
        player_count = int(_h14.largest_initial_player_count(obs_tensors))
    except Exception:
        player_count = 2
    if not _in_snipe_phase(obs, player_count):
        return base_idx, base_exists

    target_score, _enemy_eta = _counter_snipe_scores(obs, cache, source_mask, player_count=player_count, dtype=prod.dtype)
    if not bool(torch.isfinite(target_score).any()):
        return base_idx, base_exists

    P = int(obs.P)
    k = max(1, min(4 if _leader_runaway_active(obs, player_count) else int(SNIPE_EXTRA_TARGETS), P))
    snipe_idx = torch.argsort(target_score, descending=True, stable=True)[:k]
    snipe_exists = torch.isfinite(target_score[snipe_idx])
    merged_idx = torch.cat([snipe_idx.to(base_idx.dtype), base_idx], dim=0)
    merged_exists = torch.cat([snipe_exists.to(base_exists.dtype), base_exists], dim=0)
    return _stable._unique_preserve_order(merged_idx, merged_exists, P=P)


def _snipe_adjustment(*, obs, cache, cand_src, cand_send, cand_eta, cand_tgt_slot, cand_is_def, score, player_count: int):
    if int(obs.P) == 0 or int(score.numel()) == 0 or not _in_snipe_phase(obs, int(player_count)):
        return score

    source_mask = obs.owned & obs.alive & (obs.ships >= 1.0)
    target_score, enemy_eta = _counter_snipe_scores(obs, cache, source_mask, player_count=player_count, dtype=score.dtype)
    if not bool(torch.isfinite(target_score).any()):
        return score

    P = int(obs.P)
    dtype = score.dtype
    src = cand_src[:, 0].clamp(0, P - 1)
    tgt = cand_tgt_slot.clamp(0, P - 1)
    send = cand_send[:, 0].to(dtype)
    eta = cand_eta[:, 0].to(dtype)
    ships = obs.ships.to(dtype)
    prod = obs.prod.to(dtype)

    runaway = _leader_runaway_active(obs, int(player_count))
    raw_bonus = target_score[tgt].clamp(min=0.0, max=0.66 if runaway else float(SNIPE_MAX_ADJUST))
    e_eta = enemy_eta[tgt]
    neutral_timing = (eta >= e_eta - float(SNIPE_TIMING_GRACE_BEFORE)) & (eta <= e_eta + float(SNIPE_TIMING_GRACE_AFTER))
    own_timing = eta <= e_eta + 4.0
    timing_ok = torch.where(obs.owned[tgt] & bool(runaway), own_timing, neutral_timing)
    bonus = raw_bonus * timing_ok.to(dtype)

    # Do not let the overlay turn into a suicide switch; it should only choose
    # among already plausible candidate launches.
    source_after = ships[src] - send
    reserve = prod[src] * 6.5 + 12.0
    drain_gap = ((reserve - source_after).clamp(min=0.0) / reserve.clamp(min=1.0)).clamp(max=1.0)
    source_penalty = bonus.gt(0.0).to(dtype) * drain_gap * (0.42 if runaway else float(SNIPE_SOURCE_DRAIN_PENALTY))

    adjusted = score + bonus - source_penalty
    return torch.where(torch.isfinite(score), adjusted, score)


def _snipe_tier_candidates(**kwargs):
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
    adjusted = _snipe_adjustment(
        obs=kwargs["obs"],
        cache=kwargs["cache"],
        cand_src=cand_src,
        cand_send=cand_send,
        cand_eta=cand_eta,
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
        _h14.build_target_shortlist = _snipe_build_target_shortlist
        _h14._tier_candidates = _snipe_tier_candidates


def agent(obs):
    global _MODE, _MODE_PLAYER
    step = int(_read(obs, "step", 0) or 0)
    player = int(_read(obs, "player", 0) or 0)
    if step == 0 or _MODE is None or _MODE_PLAYER != player:
        _MODE = _select_mode(obs)
        _MODE_PLAYER = player
    _install_mode(_MODE)
    return _h14.agent(obs)
