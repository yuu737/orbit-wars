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
        description="Run 4P eval and print per-seed standings, selector mode, and initial-board features."
    )
    parser.add_argument("--agent", required=True, help="Primary agent main.py or folder.")
    parser.add_argument("--opponent", action="append", dest="opponents", required=True)
    parser.add_argument("--seed-list", help="Comma-separated explicit seeds.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--quiet-env-noise", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_list: str | None, seed_start: int, games: int) -> list[int]:
    if seed_list:
        return [int(s.strip()) for s in seed_list.split(",") if s.strip()]
    return [int(seed_start) + i for i in range(int(games))]


def agent_label(path: str) -> str:
    p = Path(path)
    if p.name.lower() == "main.py":
        return p.parent.name
    if p.suffix:
        return p.stem
    return p.name


def build_agents(primary_agent: str, opponents: list[str], player_index: int, players: int = 4) -> list[str]:
    if len(opponents) == 1:
        pool = opponents * (players - 1)
    elif len(opponents) == players - 1:
        pool = list(opponents)
    else:
        raise ValueError(f"4P requires 1 or 3 opponents; got {len(opponents)}")
    agents = list(pool)
    agents.insert(int(player_index), primary_agent)
    return agents


def build_labels(primary_agent: str, opponents: list[str], player_index: int, players: int = 4) -> list[str]:
    if len(opponents) == 1:
        pool = opponents * (players - 1)
    else:
        pool = list(opponents)
    labels = [agent_label(p) for p in pool]
    labels.insert(int(player_index), agent_label(primary_agent))
    return labels


def compute_total_ships(observation: dict[str, Any], player_index: int) -> int:
    planet_ships = sum(planet[5] for planet in observation["planets"] if planet[1] == player_index)
    fleet_ships = sum(fleet[6] for fleet in observation["fleets"] if fleet[1] == player_index)
    return int(planet_ships + fleet_ships)


def compute_survival_turn(steps: list[list[Any]], player_index: int) -> int:
    last = 0
    for turn, states in enumerate(steps):
        if compute_total_ships(states[player_index].observation, player_index) > 0:
            last = int(turn)
    return last


def load_agent_module(agent_path: str):
    path = Path(agent_path)
    if path.is_dir():
        path = path / "main.py"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    for name in list(sys.modules):
        if name == "orbit_lite" or name.startswith("orbit_lite."):
            del sys.modules[name]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_detail_mode_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_cwd = os.getcwd()
    try:
        os.chdir(str(path.parent))
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def compact_features(features: dict[str, Any]) -> dict[str, float]:
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


def infer_mode_and_features(module, obs: dict[str, Any], player_index: int) -> tuple[str, dict[str, float]]:
    if not all(hasattr(module, name) for name in ("single_obs_to_tensor", "_choose_4p_mode", "_initial_board_features")):
        return "", compact_features({})
    obs_copy = dict(obs)
    obs_copy["player"] = int(player_index)
    tensors = module.single_obs_to_tensor(obs_copy, player_id=int(player_index))
    mode = str(module._choose_4p_mode(tensors))
    features = compact_features(module._initial_board_features(tensors))
    return mode, features


def main() -> None:
    args = parse_args()
    if args.quiet_env_noise:
        devnull = open(os.devnull, "w", encoding="utf-8")
        os.dup2(devnull.fileno(), 2)
        sys.stderr = devnull

    seeds = parse_seed_list(args.seed_list, args.seed_start, args.games)
    with suppress_output():
        from kaggle_environments import make

    module = load_agent_module(args.agent)
    agents = build_agents(args.agent, args.opponents, args.player_index)
    labels = build_labels(args.agent, args.opponents, args.player_index)
    feature_keys = list(compact_features({}).keys())

    if args.csv:
        print(",".join([
            "seed",
            "result",
            "place",
            "diff",
            "score",
            "best_opp",
            "length",
            "survival",
            "mode",
            "standings",
            *feature_keys,
        ]))

    wins = draws = losses = crashes = 0
    for seed in seeds:
        with suppress_output():
            env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
            env.reset(4)
        initial_obs = env.steps[0][int(args.player_index)].observation
        mode, features = infer_mode_and_features(module, initial_obs, int(args.player_index))

        with suppress_output():
            env.run(agents)
        final_states = env.steps[-1]
        scores = [compute_total_ships(state.observation, seat) for seat, state in enumerate(final_states)]
        primary = scores[int(args.player_index)]
        best_opp = max(score for seat, score in enumerate(scores) if seat != int(args.player_index))
        diff = int(primary - best_opp)
        place = 1 + sum(score > primary for score in scores)
        result = "D" if diff == 0 else ("W" if diff > 0 else "L")
        wins += result == "W"
        draws += result == "D"
        losses += result == "L"
        crashes += final_states[int(args.player_index)].status != "DONE"
        length = len(env.steps) - 1
        survival = compute_survival_turn(env.steps, int(args.player_index))
        standings = sorted(
            [(scores[seat], seat, labels[seat]) for seat in range(4)],
            key=lambda item: (-item[0], item[1]),
        )
        standings_text = " | ".join(
            f"{rank}:{name}(seat{seat},score={score})"
            for rank, (score, seat, name) in enumerate(standings, start=1)
        )

        if args.csv:
            print(",".join([
                str(seed),
                result,
                str(place),
                str(diff),
                str(primary),
                str(best_opp),
                str(length),
                str(survival),
                mode,
                '"' + standings_text.replace('"', '""') + '"',
                *[f"{features[key]:.4f}" for key in feature_keys],
            ]))
        else:
            print(
                f"seed={seed} result={result} place={place} diff={diff} "
                f"score={primary} best_opp={best_opp} mode={mode} length={length} survival={survival}"
            )
            print(f"  standings: {standings_text}")
            print(
                "  features: "
                + " ".join(f"{key}={features[key]:.1f}" for key in feature_keys)
            )

    total = len(seeds)
    if not args.csv:
        print(
            f"summary games={total} wins={wins} draws={draws} losses={losses} "
            f"crashes={crashes} win_rate={(wins / total * 100.0 if total else 0.0):.1f}%"
        )


if __name__ == "__main__":
    main()
