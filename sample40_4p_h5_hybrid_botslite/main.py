from __future__ import annotations

import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import h5_agent
import s36_agent


_selected_mode: str | None = None


def _select_mode(obs) -> str:
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    obs_tensors = s36_agent.single_obs_to_tensor(obs, player_id=int(player))
    player_count = s36_agent.largest_initial_player_count(obs_tensors)
    if int(player_count) < 4:
        return "s36"

    features = s36_agent._initial_board_features(obs_tensors)
    if features.get("fallback"):
        return "s36"

    enemy_dist = float(features["enemy_dist"])
    near_prod = float(features["n25_prod"])
    mid_prod = float(features["n45_prod"])
    mid_cheap = float(features["n45_cheap"])
    high65 = float(features["n65_high_prod"])
    chain = float(features["chain_score"])

    # hairate5 is uniquely useful on sparse balanced boards where sample7/8
    # tend to all-in early and die, but it is dangerous on chain-cluster boards.
    if (
        enemy_dist >= 62.0
        and near_prod <= 10.5
        and mid_prod <= 18.5
        and mid_cheap <= 3.5
        and 5.0 <= high65 <= 8.0
        and 44.0 <= chain <= 55.5
    ):
        return "h5"
    return "s36"


def agent(obs):
    global _selected_mode
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    if int(step) == 0 or _selected_mode is None:
        _selected_mode = _select_mode(obs)
        if _selected_mode == "h5":
            h5_agent._RUNTIME.reset()
        else:
            s36_agent._RUNTIME.reset()
    if _selected_mode == "h5":
        return h5_agent.agent(obs)
    return s36_agent.agent(obs)
