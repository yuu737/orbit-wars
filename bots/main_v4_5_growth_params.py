"""
Orbit Wars - V4.5 Growth parameter branch

Experimental wrapper around the self-owned V4.1 planner. This branch is more
aggressive about considering targets, matching the best small hairate2 sweep by
score difference so far.
"""

from bots import main_v4_1_counter_snipe as base


base.CONFIG_2P = base.PlannerConfig(
    roi_threshold=1.2,
    max_targets=14,
)
base.CONFIG_4P = base.PlannerConfig(
    horizon=13,
    max_sources=6,
    max_targets=12,
    max_actions=5,
    roi_threshold=1.2,
    regroup_distance=6.0,
    regroup_threshold=11.0,
    reserve_margin=3,
)


def agent(obs):
    return base.agent(obs)
