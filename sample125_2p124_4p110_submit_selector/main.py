from __future__ import annotations

import importlib.util
import os
import sys


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

_MODE: str | None = None
_MOD = None


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _player_count_from_obs(obs) -> int:
    planets = _obs_get(obs, "planets", []) or []
    owners = []
    for planet in planets:
        try:
            owner = int(round(float(planet[1])))
        except Exception:
            continue
        if owner >= 0:
            owners.append(owner)
    if owners:
        return max(owners) + 1
    return max(2, int(_obs_get(obs, "player", 0) or 0) + 1)


def _purge_orbit_lite() -> None:
    for name in list(sys.modules):
        if name == "orbit_lite" or name.startswith("orbit_lite."):
            del sys.modules[name]


def _load_agent(folder: str, module_name: str):
    root = os.path.join(_HERE, folder)
    main_path = os.path.join(root, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(main_path)

    _purge_orbit_lite()
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def agent(obs):
    global _MODE, _MOD
    raw_step = _obs_get(obs, "step", None)
    if raw_step is not None and int(raw_step or 0) == 0:
        _MODE = None
        _MOD = None

    if _MODE is None:
        _MODE = "sample110" if _player_count_from_obs(obs) >= 4 else "sample124"

    if _MOD is None:
        _MOD = _load_agent(_MODE, f"_sample125_{_MODE}")
    return _MOD.agent(obs)
