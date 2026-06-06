"""
Orbit Wars - V4.7 Economy Push branch

Combines the V4.6 opening/defense guards with a wider, more aggressive opening
economy search. This is intentionally experimental: temporary instability is
accepted if it exposes a path toward stronger early production.
"""

from bots import main_v4_6_opening_defense_guard as guarded


base = guarded.base

base.CONFIG_2P = base.PlannerConfig(
    horizon=20,
    max_sources=10,
    max_targets=16,
    max_actions=7,
    roi_threshold=1.15,
    reserve_margin=2,
)
base.CONFIG_4P = base.PlannerConfig(
    horizon=14,
    max_sources=7,
    max_targets=12,
    max_actions=6,
    roi_threshold=1.35,
    regroup_distance=6.0,
    regroup_threshold=11.0,
    reserve_margin=3,
)


def agent(obs):
    return base.agent(obs)
