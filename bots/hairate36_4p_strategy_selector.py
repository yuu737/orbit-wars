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

import hairate30_2p_h14_4p_h29 as _stable
import hairate32_4p_home_cluster_planner as _cluster


CLUSTER_SELECT_THRESHOLD = 19.0
CLUSTER_SELECT_CLOSE_PROD = 8.0
CHAIN_SELECT_THRESHOLD = 11.0
CHAIN_MID_MIN_SHIPS = 20.0
CHAIN_MID_MAX_SHIPS = 35.0
CHAIN_HIGH_MIN_SHIPS = 50.0
CHAIN_MID_DISTANCE = 78.0
CHAIN_HIGH_DISTANCE = 95.0
CHAIN_ANGLE_COS = 0.35
TYPE_A_HIGH_MIN_SHIPS = 70.0
NEAR_BRIDGE_DISTANCE = 82.0
DENSE_RING_MIN_SHIPS = 40.0
DENSE_RING_MAX_SHIPS = 55.0
DENSE_RING_DISTANCE = 115.0
DENSE_RING_ANGLE_COS = 0.15
DENSE_RING_REQUIRED = 2

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

    owners = []
    player = int(_read(obs, "player", 0) or 0)
    owners.append(player)
    for row in _read(obs, "planets", []) or []:
        if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
            owners.append(int(row[1]))
    for row in _read(obs, "fleets", []) or []:
        if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
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
        # Orbit initial rows keep geometry, current step-0 rows keep owner/ships.
        base[1] = p[1]
        base[5] = p[5]
        rows.append(base)
    return rows


def _cluster_potential(obs, player: int) -> float:
    rows = _initial_geometry_rows(obs)
    starts = [p for p in rows if int(p[1]) >= 0]
    if len(starts) < 4:
        return 0.0

    home_neutral_prod = 0.0
    border_neutral_prod = 0.0
    home_high = 0
    border_high = 0
    near_density = 0
    close_prod = 0.0
    close_high = 0

    for p in rows:
        if int(p[1]) >= 0:
            continue
        prod = float(p[6])
        nearest = sorted((_dist(p, s), int(s[1])) for s in starts)
        nearest_dist, nearest_owner = nearest[0]
        second_dist = nearest[1][0] if len(nearest) > 1 else 999.0
        confidence = (second_dist - nearest_dist) / max(second_dist, 1.0)
        is_home = nearest_owner == int(player)
        is_border = confidence <= 0.14
        if not (is_home or is_border):
            continue

        local_neutral_count = 0
        for q in rows:
            if int(q[1]) < 0 and _dist(p, q) <= 42.0:
                local_neutral_count += 1
        near_density = max(near_density, local_neutral_count)

        if is_home:
            home_neutral_prod += prod
            if prod >= 2.0:
                home_high += 1
        else:
            border_neutral_prod += prod
            if prod >= 2.0:
                border_high += 1

        if nearest_dist <= 58.0:
            close_prod += prod
            if prod >= 2.0:
                close_high += 1

    return (
        home_neutral_prod
        + border_neutral_prod * 0.55
        + home_high * 2.0
        + border_high * 0.7
        + near_density * 1.4
        + close_prod * 0.45
        + close_high * 1.0
    )


def _chain_potential(obs, player: int) -> float:
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
        is_homeish = nearest_owner == int(player) or confidence <= 0.14
        if not is_homeish:
            continue
        if float(CHAIN_MID_MIN_SHIPS) <= ships <= float(CHAIN_MID_MAX_SHIPS) and _dist(start, p) <= float(CHAIN_MID_DISTANCE):
            mids.append(p)
        if ships >= float(CHAIN_HIGH_MIN_SHIPS):
            highs.append(p)

    best = 0.0
    for mid in mids:
        sx, sy = _vec(start, mid)
        mid_dist = _dist(start, mid)
        for high in highs:
            hx, hy = _vec(start, high)
            if _cos_between(sx, sy, hx, hy) < float(CHAIN_ANGLE_COS):
                continue
            high_dist = _dist(mid, high)
            if high_dist > float(CHAIN_HIGH_DISTANCE):
                continue
            mid_value = max(0.0, 38.0 - mid_dist) * 0.10 + float(mid[6]) * 0.8 + float(mid[5]) * 0.04
            high_value = float(high[5]) * 0.10 + float(high[6]) * 1.2 + max(0.0, 90.0 - high_dist) * 0.06
            chain_value = mid_value + high_value
            best = max(best, chain_value)
    return best


def _sector_neutral_profile(obs, player: int) -> dict[str, float]:
    rows = _initial_geometry_rows(obs)
    starts = [p for p in rows if int(p[1]) >= 0]
    my_starts = [p for p in starts if int(p[1]) == int(player)]
    if not starts or not my_starts:
        return {
            "near_bridge_max": 0.0,
            "chain_score": 0.0,
            "dense_ring_score": 0.0,
            "border_chain_score": 0.0,
            "dense_core_score": 0.0,
        }
    start = my_starts[0]

    near_bridge_max = 0.0
    close_prod5_count = 0
    near_dense_anchor = 0.0
    mids = []
    highs = []
    ring = []
    border_high_prod = []
    for p in rows:
        if int(p[1]) >= 0:
            continue
        ships = float(p[5])
        prod = float(p[6])
        nearest = sorted((_dist(p, s), int(s[1])) for s in starts)
        nearest_dist, nearest_owner = nearest[0]
        second_dist = nearest[1][0] if len(nearest) > 1 else 999.0
        confidence = (second_dist - nearest_dist) / max(second_dist, 1.0)
        is_homeish = nearest_owner == int(player) or confidence <= 0.14
        if not is_homeish:
            continue

        start_dist = _dist(start, p)
        if start_dist <= float(NEAR_BRIDGE_DISTANCE):
            near_bridge_max = max(near_bridge_max, ships)
        if start_dist <= 42.0 and prod >= 5.0:
            close_prod5_count += 1
        if start_dist <= 42.0 and float(DENSE_RING_MIN_SHIPS) <= ships <= 60.0:
            near_dense_anchor = max(near_dense_anchor, ships)
        if float(CHAIN_MID_MIN_SHIPS) <= ships <= float(CHAIN_MID_MAX_SHIPS) and start_dist <= float(CHAIN_MID_DISTANCE):
            mids.append(p)
        if ships >= float(TYPE_A_HIGH_MIN_SHIPS):
            highs.append(p)
        if float(DENSE_RING_MIN_SHIPS) <= ships <= float(DENSE_RING_MAX_SHIPS) and start_dist <= float(DENSE_RING_DISTANCE):
            ring.append(p)
        if (
            confidence <= 0.14
            and 58.0 <= start_dist <= 84.0
            and prod >= 5.0
            and 18.0 <= ships <= 35.0
        ):
            border_high_prod.append(p)

    chain_score = 0.0
    for mid in mids:
        sx, sy = _vec(start, mid)
        for high in highs:
            hx, hy = _vec(start, high)
            if _cos_between(sx, sy, hx, hy) < float(CHAIN_ANGLE_COS):
                continue
            high_dist = _dist(mid, high)
            if high_dist > float(CHAIN_HIGH_DISTANCE):
                continue
            chain_score = max(
                chain_score,
                float(mid[5]) * 0.05 + float(mid[6]) * 0.7 + float(high[5]) * 0.10 + float(high[6]) * 1.0,
            )

    dense_ring_score = 0.0
    for i, a in enumerate(ring):
        ax, ay = _vec(start, a)
        aligned_count = 1
        ship_sum = float(a[5])
        prod_sum = float(a[6])
        for j, b in enumerate(ring):
            if i == j:
                continue
            bx, by = _vec(start, b)
            if _cos_between(ax, ay, bx, by) >= float(DENSE_RING_ANGLE_COS):
                aligned_count += 1
                ship_sum += float(b[5])
                prod_sum += float(b[6])
        if aligned_count >= int(DENSE_RING_REQUIRED):
            dense_ring_score = max(dense_ring_score, ship_sum * 0.08 + prod_sum * 1.2 + aligned_count * 2.0)

    border_chain_score = 0.0
    if near_bridge_max >= 28.0 and len(border_high_prod) >= 2:
        border_chain_score = near_bridge_max * 0.12 + sum(float(p[6]) * 1.3 for p in border_high_prod[:3])

    dense_core_score = 0.0
    if near_dense_anchor > 0.0 and close_prod5_count >= 2:
        dense_core_score = near_dense_anchor * 0.08 + close_prod5_count * 2.0

    return {
        "near_bridge_max": near_bridge_max,
        "chain_score": chain_score,
        "dense_ring_score": dense_ring_score,
        "border_chain_score": border_chain_score,
        "dense_core_score": dense_core_score,
    }


def _select_mode(obs) -> str:
    if _infer_player_count(obs) < 4:
        return "stable"
    player = int(_read(obs, "player", 0) or 0)
    potential = _cluster_potential(obs, player)
    chain = _chain_potential(obs, player)
    profile = _sector_neutral_profile(obs, player)
    rows = _initial_geometry_rows(obs)
    starts = [p for p in rows if int(p[1]) >= 0]
    close_prod = 0.0
    for p in rows:
        if int(p[1]) >= 0:
            continue
        nearest = min((_dist(p, s), int(s[1])) for s in starts)
        if nearest[1] == player and nearest[0] <= 58.0:
            close_prod += float(p[6])
    if profile["near_bridge_max"] < float(CHAIN_MID_MIN_SHIPS):
        return "stable"
    if profile["chain_score"] >= float(CHAIN_SELECT_THRESHOLD):
        return "cluster"
    if profile["border_chain_score"] >= 13.0:
        return "cluster"
    if profile["dense_core_score"] >= 8.0:
        return "stable"
    if profile["dense_ring_score"] >= 13.0:
        return "cluster"
    if potential >= float(CLUSTER_SELECT_THRESHOLD) and close_prod >= float(CLUSTER_SELECT_CLOSE_PROD) and profile["chain_score"] >= 7.0:
        return "cluster"
    return "stable"


def agent(obs):
    global _MODE, _MODE_PLAYER
    step = int(_read(obs, "step", 0) or 0)
    player = int(_read(obs, "player", 0) or 0)
    if step == 0 or _MODE is None or _MODE_PLAYER != player:
        _MODE = _select_mode(obs)
        _MODE_PLAYER = player
    if _MODE == "cluster":
        return _cluster.agent(obs)
    return _stable.agent(obs)
