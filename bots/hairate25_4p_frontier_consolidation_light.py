from __future__ import annotations

import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hairate24_4p_frontier_consolidation as _h24


_h24.EXTRA_CONSOLIDATION_START = 150
_h24.EXTRA_CONSOLIDATION_FRACTION = 0.22
_h24.EXTRA_CONSOLIDATION_MIN_SCORE = 60.0


def agent(obs):
    return _h24.agent(obs)
