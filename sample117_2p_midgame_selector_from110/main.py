from __future__ import annotations

import importlib.util
import math
import os
import sys


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

_MODE = None
_MODS = {}


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _dist(a, b, c, d):
    return math.hypot(float(a) - float(c), float(b) - float(d))


def _angle_from_center(x, y):
    return math.atan2(float(y) - 50.0, float(x) - 50.0)


def _angle_diff(a, b):
    d = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def _features(obs) -> dict[str, float]:
    planets = _obs_get(obs, "planets", []) or []
    player = int(_obs_get(obs, "player", 0) or 0)
    alive = [p for p in planets if len(p) >= 7]
    owned = [p for p in alive if int(round(float(p[1]))) == player]
    if not owned:
        return {"fallback": 1.0}
    start = max(owned, key=lambda p: float(p[5]))
    sx, sy = float(start[2]), float(start[3])
    sa = _angle_from_center(sx, sy)
    neutrals = [p for p in alive if int(round(float(p[1]))) < 0]
    enemies = [p for p in alive if int(round(float(p[1]))) >= 0 and int(round(float(p[1]))) != player]

    def band(max_dist):
        items = [p for p in neutrals if _dist(sx, sy, p[2], p[3]) <= max_dist]
        return {
            f"n{int(max_dist)}_count": float(len(items)),
            f"n{int(max_dist)}_prod": float(sum(float(p[6]) for p in items)),
            f"n{int(max_dist)}_ships": float(sum(float(p[5]) for p in items)),
            f"n{int(max_dist)}_cheap": float(sum(1 for p in items if float(p[5]) <= 20.0)),
            f"n{int(max_dist)}_high_prod": float(sum(1 for p in items if float(p[6]) >= 3.0)),
        }

    f = {
        "planet_count": float(len(alive)),
        "enemy_dist": min((_dist(sx, sy, p[2], p[3]) for p in enemies), default=999.0),
    }
    for d in (25.0, 45.0, 65.0):
        f.update(band(d))

    best_chain = 0.0
    for mid in neutrals:
        mx, my = float(mid[2]), float(mid[3])
        md = _dist(sx, sy, mx, my)
        mp, msh = float(mid[6]), float(mid[5])
        if not (8.0 <= md <= 48.0):
            continue
        if mp < 2.0 and not (15.0 <= msh <= 40.0):
            continue
        ma = _angle_from_center(mx, my)
        if _angle_diff(sa, ma) > 1.15:
            continue
        for outer in neutrals:
            if outer is mid:
                continue
            ox, oy = float(outer[2]), float(outer[3])
            od = _dist(mx, my, ox, oy)
            if not (18.0 <= od <= 80.0):
                continue
            if _angle_diff(ma, _angle_from_center(ox, oy)) > 0.85:
                continue
            score = mp * 5.5 + max(0.0, 42.0 - msh) * 0.12 + float(outer[6]) * 7.0 + float(outer[5]) * 0.08 - md * 0.12 - od * 0.08
            best_chain = max(best_chain, score)
    f["chain_score"] = best_chain

    best_outer = 0.0
    best_support = 0.0
    for anchor in neutrals:
        ax, ay = float(anchor[2]), float(anchor[3])
        ar = _dist(ax, ay, 50.0, 50.0)
        ad = _dist(sx, sy, ax, ay)
        ap, ash = float(anchor[6]), float(anchor[5])
        aa = _angle_from_center(ax, ay)
        if ar < 30.0 or ad > 78.0:
            continue
        if ap < 3.0 and ash < 45.0:
            continue
        if _angle_diff(sa, aa) > 1.35:
            continue
        support = [
            p for p in neutrals
            if p is not anchor
            and _angle_diff(aa, _angle_from_center(p[2], p[3])) <= 0.95
            and _dist(ax, ay, p[2], p[3]) <= 50.0
            and (float(p[6]) >= 2.0 or float(p[5]) >= 18.0)
        ]
        support_prod = sum(float(p[6]) for p in support)
        support_density = support_prod + len(support) * 1.5
        score = ap * 10.0 + ash * 0.055 + ar * 0.055 + support_prod * 1.65 + len(support) * 1.25 - ad * 0.16
        if score > best_outer:
            best_outer = score
            best_support = support_density
    f["outer_anchor_score"] = best_outer
    f["support_density"] = best_support
    return f


def _choose_mode(obs) -> str:
    f = _features(obs)
    if f.get("fallback"):
        return "s623"
    enemy_dist = f["enemy_dist"]
    support = f["support_density"]
    chain = f["chain_score"]
    n25_count = f["n25_count"]
    n25_prod = f["n25_prod"]
    n45_cheap = f["n45_cheap"]

    # Sparse, far-opening boards with a thick support lane need the light
    # comeback planner; it preserves enough ships for the midgame swing.
    if enemy_dist >= 98.0 and support >= 30.0 and chain <= 52.0:
        return "s115"
    # Very narrow chain openings are better handled by the sample110/sample8
    # planner; the simpler planner under-expands here.
    if chain >= 60.0 or n25_count <= 2.0:
        return "s110"
    # Cheap scattered starts are generally best for the compact 6_22_3 planner.
    if n25_prod <= 8.0 and n45_cheap >= 5.0:
        return "s623"
    return "s623"


def _purge_orbit_lite():
    for name in list(sys.modules):
        if name == "orbit_lite" or name.startswith("orbit_lite."):
            del sys.modules[name]


def _load_mode(mode: str):
    if mode in _MODS:
        return _MODS[mode]
    folder = {"s110": "s110", "s115": "s115", "s623": "s623"}[mode]
    root = os.path.join(_HERE, folder)
    main_path = os.path.join(root, "main.py")
    _purge_orbit_lite()
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(f"_sample117_{mode}", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _MODS[mode] = mod
    return mod


def agent(obs):
    global _MODE
    step = int(_obs_get(obs, "step", 0) or 0)
    players = max(2, int(_obs_get(obs, "player", 0) or 0) + 1)
    planets = _obs_get(obs, "planets", []) or []
    owners = [int(round(float(p[1]))) for p in planets if len(p) >= 2 and int(round(float(p[1]))) >= 0]
    if owners:
        players = max(players, max(owners) + 1)
    if step == 0:
        _MODE = None
    if players >= 4:
        mode = "s110"
    else:
        if _MODE is None:
            _MODE = _choose_mode(obs)
        mode = _MODE
    return _load_mode(mode).agent(obs)
