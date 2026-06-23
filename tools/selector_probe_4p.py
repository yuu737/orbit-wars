from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast 4P selector probe: no game run, only step-0 mode and features."
    )
    parser.add_argument("--agent", required=True, help="Agent main.py path or folder.")
    parser.add_argument("--seed-list", help="Comma-separated explicit seeds.")
    parser.add_argument("--games", type=int, default=20, help="Used with --seed-start when --seed-list is absent.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--csv", action="store_true", help="Print CSV instead of compact text.")
    parser.add_argument("--quiet-env-noise", action="store_true")
    return parser.parse_args()


def load_agent_module(path: str):
    module_path = Path(path)
    if module_path.is_dir():
        module_path = module_path / "main.py"
    name = "_selector_probe_" + str(abs(hash(str(module_path))))
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


def require_selector_api(module) -> None:
    required = ("single_obs_to_tensor", "_choose_4p_mode", "_initial_board_features")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Agent does not expose selector helpers: {', '.join(missing)}")


def compact_features(features: dict[str, float]) -> dict[str, float]:
    keys = [
        "enemy_dist",
        "n25_count",
        "n25_prod",
        "n25_ships",
        "n25_cheap",
        "n45_prod",
        "n45_ships",
        "n45_cheap",
        "n65_high_prod",
        "chain_score",
        "outer_anchor_score",
        "support_density",
        "enemy_cluster_risk",
        "planet_count",
    ]
    return {key: float(features.get(key, 0.0)) for key in keys}


def main() -> None:
    args = parse_args()
    if args.quiet_env_noise:
        devnull = open(os.devnull, "w", encoding="utf-8")
        os.dup2(devnull.fileno(), 2)
        sys.stderr = devnull
    if args.seed_list:
        seeds = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
    else:
        seeds = [int(args.seed_start) + i for i in range(int(args.games))]

    module = load_agent_module(args.agent)
    require_selector_api(module)

    with suppress_output():
        from kaggle_environments import make

    feature_keys = list(compact_features({}).keys())
    if args.csv:
        print(",".join(["seed", "seat", "mode", *feature_keys]))

    mode_counts: dict[str, int] = {}
    for seed in seeds:
        with suppress_output():
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.reset(4)
        obs = env.steps[0][int(args.player_index)].observation
        tensors = module.single_obs_to_tensor(obs, player_id=int(args.player_index))
        mode = str(module._choose_4p_mode(tensors))
        features = compact_features(module._initial_board_features(tensors))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        if args.csv:
            values = [
                str(seed),
                str(int(args.player_index)),
                mode,
                *[f"{features[key]:.4f}" for key in feature_keys],
            ]
            print(",".join(values))
        else:
            ftxt = " ".join(f"{key}={features[key]:.1f}" for key in feature_keys)
            print(f"seed={seed} seat={int(args.player_index)} mode={mode} {ftxt}")

    if not args.csv:
        counts = " ".join(f"{mode}={count}" for mode, count in sorted(mode_counts.items()))
        print(f"summary total={len(seeds)} {counts}")


if __name__ == "__main__":
    main()
