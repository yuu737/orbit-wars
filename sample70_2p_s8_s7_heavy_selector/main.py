from __future__ import annotations

import importlib.util
import math
import os
import sys
from types import ModuleType


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

_MOD_S8: ModuleType | None = None
_MOD_S7: ModuleType | None = None
_MOD_4P: ModuleType | None = None
_MODE_2P: str | None = None


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


def _clear_orbit_lite_modules() -> None:
    for name in list(sys.modules):
        if name == "orbit_lite" or name.startswith("orbit_lite."):
            del sys.modules[name]


def _candidate_roots():
    roots = []
    for path in (
        _HERE,
        os.getcwd(),
        os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else "",
        "/kaggle_simulations/agent",
        "/kaggle/working",
    ):
        if path and path not in roots:
            roots.append(path)
    return roots


def _load_agent_module(folder_name: str, module_name: str) -> ModuleType:
    folder = ""
    main_path = ""
    for root in _candidate_roots():
        candidate_folder = os.path.join(root, folder_name)
        candidate_main = os.path.join(candidate_folder, "main.py")
        if os.path.exists(candidate_main):
            folder = candidate_folder
            main_path = candidate_main
            break
    if not main_path:
        tried = [os.path.join(root, folder_name, "main.py") for root in _candidate_roots()]
        raise FileNotFoundError("; ".join(tried))
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


def _agent_s8() -> ModuleType:
    global _MOD_S8
    if _MOD_S8 is None:
        _MOD_S8 = _load_agent_module("sample8", "_sample70_s8")
    return _MOD_S8


def _agent_s7() -> ModuleType:
    global _MOD_S7
    if _MOD_S7 is None:
        _MOD_S7 = _load_agent_module("sample7", "_sample70_s7")
    return _MOD_S7


def _agent_4p() -> ModuleType:
    global _MOD_4P
    if _MOD_4P is None:
        _MOD_4P = _load_agent_module("sample69_4p_orbit_lite_response_gate", "_sample70_4p")
    return _MOD_4P


def _dist(a, b) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _choose_2p_mode(obs) -> str:
    planets = list(_read(obs, "planets", []) or [])
    player = int(_read(obs, "player", 0) or 0)
    if not planets:
        return "s8"
    owned = [p for p in planets if len(p) >= 7 and int(p[1]) == player]
    if not owned:
        return "s8"
    home = owned[0]
    neutrals = [p for p in planets if len(p) >= 7 and int(p[1]) == -1]
    if not neutrals:
        return "s8"

    nearest = sorted(neutrals, key=lambda p: _dist(home, p))[:8]
    nearest5 = nearest[:5]
    cheap8 = sum(1 for p in nearest if int(p[5]) <= 15)
    highprod8 = sum(1 for p in nearest if int(p[6]) >= 3)
    heavy5 = sum(1 for p in nearest5 if int(p[5]) >= 24)
    very_heavy8 = sum(1 for p in nearest if int(p[5]) >= 45)
    prod5_near = [p for p in nearest if int(p[6]) >= 5 and _dist(home, p) <= 35.0]
    min_highprod_dist = min((_dist(home, p) for p in nearest if int(p[6]) >= 3), default=999.0)

    # sample8 tends to over-spread here; sample7's concentrated early route wins
    # the heavy neutral race before the opponent snowballs.
    heavy_value_lane = heavy5 >= 3 and cheap8 <= 3 and highprod8 >= 3

    # 1205318666-type: several nearby prod5 targets make sample7's opening route
    # stronger even though the cheap count looks friendly to sample8.
    rich_prod5_lane = (
        len(prod5_near) >= 3
        and cheap8 >= 4
        and highprod8 >= 3
        and min_highprod_dist <= 20.0
        and len(planets) >= 28
    )

    # Extremely heavy nearby anchors with few cheap options also behave more like
    # the sample7-winning set, even on larger boards.
    heavy_anchor_lane = very_heavy8 >= 2 and cheap8 <= 3 and min_highprod_dist <= 25.0

    if heavy_value_lane or rich_prod5_lane or heavy_anchor_lane:
        return "s7"
    return "s8"


def agent(obs):
    global _MODE_2P
    if _infer_player_count(obs) >= 4:
        return _agent_4p().agent(obs)
    step = int(_read(obs, "step", 0) or 0)
    if step == 0:
        _MODE_2P = None
    if _MODE_2P is None:
        _MODE_2P = _choose_2p_mode(obs)
    if _MODE_2P == "s7":
        return _agent_s7().agent(obs)
    return _agent_s8().agent(obs)
