from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

_MOD_2P: ModuleType | None = None
_MOD_4P: ModuleType | None = None


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


def _base_dirs() -> list[str]:
    dirs = []
    for path in (
        _HERE,
        os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else "",
        os.getcwd(),
        "/kaggle_simulations/agent",
        "/kaggle/working",
    ):
        if path and path not in dirs:
            dirs.append(path)
    return dirs


def _find_agent_folder(folder_name: str) -> str:
    for base in _base_dirs():
        folder = os.path.join(base, folder_name)
        if os.path.exists(os.path.join(folder, "main.py")):
            return folder
    raise FileNotFoundError(
        f"Could not find {folder_name}/main.py under: {', '.join(_base_dirs())}"
    )


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


def _agent_2p() -> ModuleType:
    global _MOD_2P
    if _MOD_2P is None:
        _MOD_2P = _load_agent_module("sample8", "_sample8_2p_main")
    return _MOD_2P


def _agent_4p() -> ModuleType:
    global _MOD_4P
    if _MOD_4P is None:
        _MOD_4P = _load_agent_module("sample7", "_sample7_4p_main")
    return _MOD_4P


def agent(obs):
    if _infer_player_count(obs) >= 4:
        return _agent_4p().agent(obs)
    return _agent_2p().agent(obs)
