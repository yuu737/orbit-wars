"""
Orbit Wars - V4.5 ROI 2.2 parameter branch

Experimental wrapper around the self-owned V4.1 planner. This branch keeps the
planner code unchanged and tests a more selective capture threshold found by
the local parameter sweep.
"""

from bots import main_v4_1_counter_snipe as base


base.CONFIG_2P = base.PlannerConfig(roi_threshold=2.2)
base.CONFIG_4P = base.PlannerConfig(
    horizon=13,
    max_sources=6,
    max_targets=10,
    max_actions=5,
    roi_threshold=2.2,
    regroup_distance=6.0,
    regroup_threshold=11.0,
    reserve_margin=3,
)


def agent(obs):
    return base.agent(obs)
