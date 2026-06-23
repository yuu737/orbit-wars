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
_MOD_H5: ModuleType | None = None
_MOD_4P: ModuleType | None = None
_MODE_2P: str | None = None
_ENABLE_H5_RESCUE = True


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


def _load_agent_module(folder_name: str, module_name: str, *, clear_orbit_lite: bool = True) -> ModuleType:
    folder = os.path.join(_HERE, folder_name)
    main_path = os.path.join(folder, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(main_path)
    old_path = list(sys.path)
    try:
        if clear_orbit_lite:
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


def _agent_h5() -> ModuleType:
    global _MOD_H5
    if _MOD_H5 is None:
        # hairate5 is isolated as h5_orbit_lite, so clearing orbit_lite here is
        # unnecessary and can disturb a same-process opponent that imports it.
        _MOD_H5 = _load_agent_module("hairate5", "_sample74_h5", clear_orbit_lite=False)
    return _MOD_H5


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
    prod4_near = [p for p in nearest if int(p[6]) >= 4 and _dist(home, p) <= 35.0]
    min_highprod_dist = min((_dist(home, p) for p in nearest if int(p[6]) >= 3), default=999.0)
    avg_near5_ships = sum(float(p[5]) for p in nearest5) / max(1, len(nearest5))

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

    # sample70 loss-set additions: these boards look friendly to sample8 by
    # cheap count, but hairate2 wins the opening unless we use sample7's route.
    close_prod5_route = len(prod5_near) >= 2 and min_highprod_dist <= 18.0
    low_mass_highprod_trap = (
        cheap8 >= 5
        and highprod8 >= 3
        and min_highprod_dist <= 16.0
        and avg_near5_ships <= 18.5
    )
    mixed_heavy_value_lane = (
        highprod8 >= 3
        and heavy5 >= 2
        and avg_near5_ships >= 20.0
        and len(prod4_near) >= 1
    )

    # Extremely heavy nearby anchors with few cheap options also behave more like
    # the sample7-winning set, even on larger boards.
    heavy_anchor_lane = very_heavy8 >= 2 and cheap8 <= 3 and min_highprod_dist <= 25.0

    # hairate5 is weak as a default, but it wins a few sample71 loss boards.
    # Keep these gates narrow: use it only for opening shapes where its heavier
    # orbit-lite planner beats both sample8's spread and sample7's route.
    h5_low_mass_prod3_cluster = (
        cheap8 >= 6
        and highprod8 >= 5
        and heavy5 >= 2
        and very_heavy8 >= 1
        and len(prod5_near) == 0
        and len(prod4_near) == 0
        and 20.0 <= avg_near5_ships <= 26.0
    )
    h5_far_heavy_prod5_anchor = (
        cheap8 == 4
        and highprod8 == 2
        and heavy5 == 1
        and very_heavy8 >= 2
        and len(prod5_near) == 1
        and min_highprod_dist >= 25.0
        and avg_near5_ships <= 24.0
    )
    h5_compact_double_prod4 = (
        cheap8 <= 2
        and highprod8 == 2
        and heavy5 >= 2
        and very_heavy8 >= 2
        and len(prod4_near) >= 2
        and len(prod5_near) == 0
        and min_highprod_dist <= 18.0
        and avg_near5_ships <= 22.0
    )

    if _ENABLE_H5_RESCUE and (
        h5_low_mass_prod3_cluster or h5_far_heavy_prod5_anchor or h5_compact_double_prod4
    ):
        return "h5"

    if (
        heavy_value_lane
        or rich_prod5_lane
        or heavy_anchor_lane
        or close_prod5_route
        or low_mass_highprod_trap
        or mixed_heavy_value_lane
    ):
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
    if _MODE_2P == "h5":
        return _agent_h5().agent(obs)
    return _agent_s8().agent(obs)
