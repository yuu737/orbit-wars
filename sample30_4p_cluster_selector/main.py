from __future__ import annotations

import importlib.util
import math
import os
import sys
from types import ModuleType
from typing import Any


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

_MOD_2P: ModuleType | None = None
_MOD_SAMPLE7: ModuleType | None = None
_MOD_SAMPLE8: ModuleType | None = None
_MOD_SAMPLE29: ModuleType | None = None
_FIXED_4P_MODE: str | None = None


def _read(obs: Any, name: str, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _infer_player_count(obs: Any) -> int:
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


def _base_dirs() -> list[str]:
    dirs: list[str] = []
    for path in (
        _HERE,
        os.path.dirname(_HERE),
        os.getcwd(),
        os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else "",
        "/kaggle_simulations/agent",
        "/kaggle/working",
    ):
        if path and path not in dirs:
            dirs.append(path)
    return dirs


def _clear_orbit_lite_modules() -> None:
    for name in list(sys.modules):
        if name == "orbit_lite" or name.startswith("orbit_lite."):
            del sys.modules[name]


def _find_agent_folder(folder_name: str) -> str:
    for base in _base_dirs():
        folder = os.path.join(base, folder_name)
        if os.path.exists(os.path.join(folder, "main.py")):
            return folder
    raise FileNotFoundError(f"Could not find {folder_name}/main.py under: {', '.join(_base_dirs())}")


def _load_agent_module(folder_name: str, module_name: str) -> ModuleType:
    folder = _find_agent_folder(folder_name)
    main_path = os.path.join(folder, "main.py")
    old_path = list(sys.path)
    try:
        _clear_orbit_lite_modules()
        sys.path.insert(0, folder)
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {main_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def _sample7() -> ModuleType:
    global _MOD_SAMPLE7
    if _MOD_SAMPLE7 is None:
        _MOD_SAMPLE7 = _load_agent_module("sample7", "_sample30_sample7")
    return _MOD_SAMPLE7


def _sample8() -> ModuleType:
    global _MOD_SAMPLE8
    if _MOD_SAMPLE8 is None:
        _MOD_SAMPLE8 = _load_agent_module("sample8", "_sample30_sample8")
    return _MOD_SAMPLE8


def _sample29() -> ModuleType:
    global _MOD_SAMPLE29
    if _MOD_SAMPLE29 is None:
        _MOD_SAMPLE29 = _load_agent_module("sample29_4p_stateful_domain_planner", "_sample30_sample29")
    return _MOD_SAMPLE29


def _agent_2p() -> ModuleType:
    # Keep the current high-public-score submission shape: 2P = sample8.
    global _MOD_2P
    if _MOD_2P is None:
        _MOD_2P = _load_agent_module("sample8", "_sample30_sample8_2p")
    return _MOD_2P


def _dist(a: list[Any], b: list[Any]) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _angle_from_center(p: list[Any]) -> float:
    return math.atan2(float(p[3]) - 50.0, float(p[2]) - 50.0)


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _board_features(obs: Any) -> dict[str, float]:
    player = int(_read(obs, "player", 0) or 0)
    planets = [list(row) for row in (_read(obs, "planets", []) or []) if len(row) >= 7]
    owned = [p for p in planets if int(p[1]) == player]
    if not owned:
        return {"fallback": 1.0}
    start = max(owned, key=lambda p: float(p[5]))
    neutrals = [p for p in planets if int(p[1]) == -1]
    enemies = [p for p in planets if int(p[1]) not in (-1, player)]

    def band(max_dist: float) -> dict[str, float]:
        items = [p for p in neutrals if _dist(start, p) <= max_dist]
        return {
            f"n{int(max_dist)}_count": float(len(items)),
            f"n{int(max_dist)}_prod": float(sum(float(p[6]) for p in items)),
            f"n{int(max_dist)}_ships": float(sum(float(p[5]) for p in items)),
            f"n{int(max_dist)}_high_prod": float(sum(1 for p in items if float(p[6]) >= 3)),
            f"n{int(max_dist)}_cheap": float(sum(1 for p in items if float(p[5]) <= 20)),
        }

    f: dict[str, float] = {
        "planet_count": float(len(planets)),
        "enemy_dist": min((_dist(start, p) for p in enemies), default=999.0),
    }
    for d in (25.0, 45.0, 65.0):
        f.update(band(d))

    start_angle = _angle_from_center(start)
    best_chain = 0.0
    for mid in neutrals:
        md = _dist(start, mid)
        if not (8.0 <= md <= 48.0):
            continue
        if float(mid[6]) < 2 and not (15 <= float(mid[5]) <= 40):
            continue
        mid_angle = _angle_from_center(mid)
        if _angle_diff(start_angle, mid_angle) > 1.15:
            continue
        for outer in neutrals:
            if int(outer[0]) == int(mid[0]):
                continue
            od = _dist(mid, outer)
            if not (18.0 <= od <= 80.0):
                continue
            if _angle_diff(mid_angle, _angle_from_center(outer)) > 0.85:
                continue
            score = (
                float(mid[6]) * 5.5
                + max(0.0, 42.0 - float(mid[5])) * 0.12
                + float(outer[6]) * 7.0
                + float(outer[5]) * 0.08
                - md * 0.12
                - od * 0.08
            )
            best_chain = max(best_chain, score)
    f["chain_score"] = best_chain
    return f


def _choose_4p_strategy(obs: Any) -> str:
    f = _board_features(obs)
    if f.get("fallback"):
        return "stable"

    near_prod = f["n25_prod"]
    mid_prod = f["n45_prod"]
    mid_cheap = f["n45_cheap"]
    high65 = f["n65_high_prod"]
    enemy_dist = f["enemy_dist"]
    chain = f["chain_score"]
    planets = f["planet_count"]

    # sample8 is strongest when there is enough early value to turn multi-size
    # launches into a fast snowball. Avoid it on cheap-dense maps where prior
    # runs showed early collapse.
    burst_score = (
        high65 * 1.35
        + near_prod * 0.22
        + mid_prod * 0.12
        + chain * 0.035
        - mid_cheap * 0.85
        - max(0.0, 56.0 - enemy_dist) * 0.08
        - max(0.0, near_prod - 14.0) * 0.55
        - max(0.0, mid_prod - 29.0) * 0.35
    )

    # sample29 has only a small signal; use it on balanced/large maps with
    # enough chain potential but without a very dense cheap field.
    domain_score = (
        chain * 0.11
        + high65 * 0.55
        + max(0.0, planets - 28.0) * 0.18
        - abs(mid_prod - 20.0) * 0.07
        - mid_cheap * 0.22
    )

    # Keep burst narrow. It did very well on one fast-snowball family, but was
    # poor on rich long-game maps such as 12000004 and most 56000000 variants.
    if (
        burst_score >= 7.8
        and mid_cheap <= 4.5
        and high65 >= 6.0
        and near_prod <= 14.0
        and mid_prod <= 29.0
        and not (planets <= 28.0 and high65 >= 10.0)
    ):
        return "burst"
    if (
        enemy_dist >= 70.0
        and high65 >= 5.0
        and mid_cheap <= 4.0
        and near_prod <= 14.0
        and mid_prod <= 22.0
    ):
        return "burst"
    if domain_score >= 7.4 and mid_cheap <= 5.5 and 16.0 <= mid_prod <= 28.5:
        return "domain"
    return "stable"


def agent(obs: Any):
    global _FIXED_4P_MODE
    if _infer_player_count(obs) < 4:
        _FIXED_4P_MODE = None
        return _agent_2p().agent(obs)
    step = int(_read(obs, "step", 0) or 0)
    if step <= 0 or _FIXED_4P_MODE is None:
        _FIXED_4P_MODE = _choose_4p_strategy(obs)
    mode = _FIXED_4P_MODE
    if mode == "burst":
        return _sample8().agent(obs)
    if mode == "domain":
        return _sample29().agent(obs)
    return _sample7().agent(obs)
