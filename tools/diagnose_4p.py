from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 4P Orbit Wars games and print standings plus selector features."
    )
    parser.add_argument("--agent", required=True, help="Primary agent main.py path.")
    parser.add_argument("--opponent", action="append", dest="opponents", required=True)
    parser.add_argument("--seed-list", required=True, help="Comma-separated seed list.")
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--feature-agent", help="Agent module to inspect for _choose_4p_mode. Defaults to --agent.")
    parser.add_argument("--show-features", action="store_true", help="Print compact initial selector features.")
    parser.add_argument("--quiet-env-noise", action="store_true", help="Redirect stderr to nul to hide OpenSpiel registry noise.")
    return parser.parse_args()


def compute_total_ships(observation: dict[str, Any], player_index: int) -> int:
    planet_ships = sum(planet[5] for planet in observation["planets"] if planet[1] == player_index)
    fleet_ships = sum(fleet[6] for fleet in observation["fleets"] if fleet[1] == player_index)
    return int(planet_ships + fleet_ships)


def survival_turn(steps: list[list[Any]], player_index: int) -> int:
    last = 0
    for turn, states in enumerate(steps):
        if compute_total_ships(states[player_index].observation, player_index) > 0:
            last = turn
    return last


def build_agents(primary: str, opponents: list[str], player_index: int) -> list[str]:
    if len(opponents) == 1:
        pool = opponents * 3
    elif len(opponents) == 3:
        pool = list(opponents)
    else:
        raise ValueError("--opponent must be supplied once or exactly three times for 4P.")
    agents = list(pool)
    agents.insert(player_index, primary)
    return agents


def label_for_path(path: str) -> str:
    p = Path(path)
    if p.name == "main.py":
        return p.parent.name
    return p.stem


def load_feature_module(path: str):
    module_path = Path(path)
    if module_path.is_dir():
        module_path = module_path / "main.py"
    name = "_diagnose_agent_" + str(abs(hash(str(module_path))))
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old_cwd = os.getcwd()
    try:
        os.chdir(str(module_path.parent))
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def selector_info(module, observation: dict[str, Any], player_id: int) -> tuple[str, dict[str, float]]:
    if not all(hasattr(module, name) for name in ("single_obs_to_tensor", "_choose_4p_mode", "_initial_board_features")):
        return "n/a", {}
    tensors = module.single_obs_to_tensor(observation, player_id=player_id)
    mode = str(module._choose_4p_mode(tensors))
    features = module._initial_board_features(tensors)
    return mode, features


def compact_features(features: dict[str, float]) -> str:
    keys = [
        "enemy_dist",
        "n25_prod",
        "n45_prod",
        "n45_cheap",
        "n65_high_prod",
        "chain_score",
        "outer_anchor_score",
        "support_density",
        "enemy_cluster_risk",
    ]
    parts = []
    for key in keys:
        if key in features:
            parts.append(f"{key}={float(features[key]):.1f}")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    devnull = None
    if args.quiet_env_noise:
        devnull = open(os.devnull, "w", encoding="utf-8")
        os.dup2(devnull.fileno(), 2)
        sys.stderr = devnull
    seeds = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
    if args.player_index < 0 or args.player_index >= 4:
        raise ValueError("--player-index must be 0..3")

    with suppress_output():
        from kaggle_environments import make

    feature_module = load_feature_module(args.feature_agent or args.agent)
    agents = build_agents(args.agent, args.opponents, args.player_index)
    labels = [label_for_path(agent) for agent in agents]

    for seed in seeds:
        with suppress_output():
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)

        initial_obs = env.steps[0][args.player_index].observation
        mode, features = selector_info(feature_module, initial_obs, args.player_index)

        final_scores = [
            compute_total_ships(state.observation, idx)
            for idx, state in enumerate(env.steps[-1])
        ]
        ranked = sorted(range(4), key=lambda idx: (-final_scores[idx], idx))
        place = 1 + sum(score > final_scores[args.player_index] for score in final_scores)
        best_opp = max(final_scores[idx] for idx in range(4) if idx != args.player_index)
        diff = final_scores[args.player_index] - best_opp
        result = "W" if diff > 0 else ("D" if diff == 0 else "L")
        standings = " | ".join(
            f"{rank + 1}:{labels[idx]}(seat{idx},score={final_scores[idx]})"
            for rank, idx in enumerate(ranked)
        )

        print(
            f"seed={seed} result={result} place={place} diff={diff} "
            f"mode={mode} length={len(env.steps)-1} survival={survival_turn(env.steps, args.player_index)}"
        )
        print(f"  standings: {standings}")
        if args.show_features:
            print(f"  features: {compact_features(features)}")


if __name__ == "__main__":
    main()
