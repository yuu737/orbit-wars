from __future__ import annotations

import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hairate14_response_search import agent as _agent_2p
from hairate2 import agent as _agent_4p


def _infer_player_count(obs) -> int:
    if isinstance(obs, dict):
        explicit = obs.get("player_count")
        if explicit in (2, 4):
            return int(explicit)

        owners = []
        player = obs.get("player", 0)
        try:
            owners.append(int(player))
        except Exception:
            pass

        for row in obs.get("planets", []):
            if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
                owners.append(int(row[1]))
        for row in obs.get("fleets", []):
            if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
                owners.append(int(row[1]))
        return 4 if max(owners, default=0) >= 2 else 2

    # Fallback for object-like observations. Kaggle Orbit Wars normally uses dicts.
    return 2


def agent(obs):
    if _infer_player_count(obs) >= 4:
        return _agent_4p(obs)
    return _agent_2p(obs)
