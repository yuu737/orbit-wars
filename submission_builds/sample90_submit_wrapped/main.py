from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

_MOD: ModuleType | None = None


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


def _clear_orbit_lite_modules() -> None:
    for name in list(sys.modules):
        if name == "orbit_lite" or name.startswith("orbit_lite."):
            del sys.modules[name]


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


def _agent_module() -> ModuleType:
    global _MOD
    if _MOD is None:
        _MOD = _load_agent_module("sample90_4p_response_gate_step15_from69", "_sample90_submit")
    return _MOD


def agent(obs):
    return _agent_module().agent(obs)
