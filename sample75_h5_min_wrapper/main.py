from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

def _load_h5() -> ModuleType:
    folder = os.path.join(_HERE, "hairate5")
    main_path = os.path.join(folder, "main.py")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, folder)
        spec = importlib.util.spec_from_file_location("main", main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {main_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["main"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


_MOD_H5 = _load_h5()


def agent(obs):
    return _MOD_H5.agent(obs)
