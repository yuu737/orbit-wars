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

import hairate14_response_search as _h14
import hairate30_2p_h14_4p_h29 as _stable
import hairate32_4p_home_cluster_planner as _cluster


CHAIN_SELECT_THRESHOLD = 17.0
CHAIN_MID_MIN_SHIPS = 20.0
CHAIN_MID_MAX_SHIPS = 35.0
CHAIN_MID_DISTANCE = 78.0
CHAIN_HIGH_DISTANCE = 95.0
CHAIN_ANGLE_COS = 0.35
TYPE_A_HIGH_MIN_SHIPS = 70.0
NEAR_BRIDGE_DISTANCE = 82.0
BORDER_CONFIDENCE = 0.14

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
    chain_score = _type_a_chain_score(obs, player)
    border_chain_score = _type_a_border_chain_score(obs, player)
    if chain_score >= float(CHAIN_SELECT_THRESHOLD):
        return "cluster"
    if border_chain_score >= 13.0:
        return "cluster"
    return "stable"


def _install_mode(mode: str) -> None:
    if mode == "cluster":
        _h14.build_target_shortlist = _cluster._home_build_target_shortlist
        _h14._tier_candidates = _cluster._home_tier_candidates
    else:
        _h14.build_target_shortlist = _stable._home_build_target_shortlist
        _h14._tier_candidates = _stable._home_tier_candidates


def agent(obs):
    global _MODE, _MODE_PLAYER
    step = int(_read(obs, "step", 0) or 0)
    player = int(_read(obs, "player", 0) or 0)
    if step == 0 or _MODE is None or _MODE_PLAYER != player:
        _MODE = _select_mode(obs)
        _MODE_PLAYER = player
    _install_mode(_MODE)
    return _h14.agent(obs)
