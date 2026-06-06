import argparse
import contextlib
import importlib.util
import io
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def suppress_output():
    return contextlib.ExitStack()


@contextlib.contextmanager
def quiet_stdio():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep V4 planner config values with local Orbit Wars games."
    )
    parser.add_argument(
        "--agent",
        default="bots/main_v4_1_counter_snipe.py",
        help="V4 planner module to tune. Default: bots/main_v4_1_counter_snipe.py",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        dest="opponents",
        help="Opponent path/name. Repeat for 4P mixed fields. Default: main.py",
    )
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--both-seats", action="store_true")
    parser.add_argument(
        "--preset",
        choices=("small", "reserve", "roi", "horizon", "wide"),
        default="small",
        help="Parameter grid preset. Default: small",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Optional output path for one JSON object per variant summary.",
    )
    return parser.parse_args()


def load_agent_module(path: str):
    agent_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("sweep_v4_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variant_grid(preset: str) -> list[dict[str, Any]]:
    if preset == "reserve":
        return [
            {"name": "base"},
            {"name": "reserve3", "reserve_margin": 3},
            {"name": "reserve4", "reserve_margin": 4},
            {"name": "reserve5", "reserve_margin": 5},
        ]
    if preset == "roi":
        return [
            {"name": "base"},
            {"name": "roi1.2", "roi_threshold": 1.2},
            {"name": "roi1.8", "roi_threshold": 1.8},
            {"name": "roi2.2", "roi_threshold": 2.2},
        ]
    if preset == "horizon":
        return [
            {"name": "base"},
            {"name": "h16", "horizon": 16},
            {"name": "h20", "horizon": 20},
            {"name": "h22", "horizon": 22},
        ]
    if preset == "wide":
        return [
            {"name": "base"},
            {"name": "guarded", "reserve_margin": 4, "roi_threshold": 1.8},
            {"name": "growth", "reserve_margin": 2, "roi_threshold": 1.2, "max_targets": 14},
            {"name": "patient", "horizon": 22, "roi_threshold": 2.0},
            {"name": "active", "horizon": 16, "roi_threshold": 1.1, "max_actions": 7},
            {"name": "compact4p", "horizon_4p": 11, "max_sources_4p": 5, "max_targets_4p": 8},
        ]
    return [
        {"name": "base"},
        {"name": "guarded", "reserve_margin": 4, "roi_threshold": 1.8},
        {"name": "growth", "roi_threshold": 1.2, "max_targets": 14},
    ]


def apply_variant(module, variant: dict[str, Any]) -> None:
    base2 = module.PlannerConfig()
    base4 = module.PlannerConfig(
        horizon=13,
        max_sources=6,
        max_targets=10,
        max_actions=5,
        regroup_distance=6.0,
        regroup_threshold=11.0,
        reserve_margin=3,
    )

    common_keys = {
        key: value
        for key, value in variant.items()
        if key in base2.__dataclass_fields__ and key != "name"
    }
    four_p_keys = {
        key[:-3]: value
        for key, value in variant.items()
        if key.endswith("_4p") and key[:-3] in base4.__dataclass_fields__
    }

    module.CONFIG_2P = replace(base2, **common_keys)
    module.CONFIG_4P = replace(base4, **common_keys, **four_p_keys)


def compute_total_ships(observation: dict[str, Any], player_index: int) -> int:
    planets = observation["planets"]
    fleets = observation["fleets"]
    return sum(planet[5] for planet in planets if planet[1] == player_index) + sum(
        fleet[6] for fleet in fleets if fleet[1] == player_index
    )


def compute_survival_turn(steps: list[list[Any]], player_index: int) -> int:
    last_turn = 0
    for turn, states in enumerate(steps):
        if compute_total_ships(states[player_index].observation, player_index) > 0:
            last_turn = turn
    return last_turn


def build_agents(primary_agent, opponents: list[str], player_index: int, players: int) -> list[Any]:
    if len(opponents) == 1:
        opponent_pool = opponents * (players - 1)
    elif len(opponents) == players - 1:
        opponent_pool = opponents
    else:
        raise ValueError(
            f"--players {players} needs 1 opponent or {players - 1} opponents; got {len(opponents)}"
        )
    agents = list(opponent_pool)
    agents.insert(player_index, primary_agent)
    return agents


def run_game(make, module, variant, opponents, seed, player_index, players):
    apply_variant(module, variant)
    with quiet_stdio():
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(build_agents(module.agent, opponents, player_index, players))

    final_states = env.steps[-1]
    primary_state = final_states[player_index]
    primary_score = compute_total_ships(primary_state.observation, player_index)
    scores = [
        compute_total_ships(state.observation, index)
        for index, state in enumerate(final_states)
    ]
    best_opp = max(score for index, score in enumerate(scores) if index != player_index)
    return {
        "seed": seed,
        "seat": player_index,
        "reward": primary_state.reward,
        "won": primary_state.reward == 1,
        "score_diff": primary_score - best_opp,
        "primary_score": primary_score,
        "best_opponent_score": best_opp,
        "placement": 1 + sum(score > primary_score for score in scores),
        "game_length": len(env.steps) - 1,
        "survival_turn": compute_survival_turn(env.steps, player_index),
        "status": primary_state.status,
    }


def summarize_variant(variant: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(results)
    return {
        "name": variant["name"],
        "variant": variant,
        "games": games,
        "wins": sum(result["won"] for result in results),
        "win_rate": sum(result["won"] for result in results) / games,
        "avg_score_diff": statistics.fmean(result["score_diff"] for result in results),
        "avg_survival_turn": statistics.fmean(result["survival_turn"] for result in results),
        "avg_placement": statistics.fmean(result["placement"] for result in results),
        "avg_primary_score": statistics.fmean(result["primary_score"] for result in results),
        "avg_best_opponent_score": statistics.fmean(result["best_opponent_score"] for result in results),
    }


def main() -> None:
    args = parse_args()
    opponents = args.opponents or ["main.py"]
    seats = list(range(args.players)) if args.both_seats else [args.player_index]
    module = load_agent_module(args.agent)

    with quiet_stdio():
        from kaggle_environments import make

    summaries = []
    start = time.perf_counter()
    jsonl_handle = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        for variant in variant_grid(args.preset):
            results = []
            for seat in seats:
                for offset in range(args.games):
                    seed = args.seed_start + offset
                    results.append(run_game(make, module, variant, opponents, seed, seat, args.players))
            summary = summarize_variant(variant, results)
            summaries.append(summary)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(summary, sort_keys=True) + "\n")
            print(
                "{name:<10} games={games:<2} win={win_rate:.1%} "
                "diff={avg_score_diff:8.2f} survival={avg_survival_turn:6.2f} "
                "place={avg_placement:.2f}".format(**summary)
            )
    finally:
        if jsonl_handle:
            jsonl_handle.close()

    best = max(
        summaries,
        key=lambda row: (
            row["avg_score_diff"],
            row["avg_survival_turn"],
            -row["avg_placement"],
        ),
    )
    print("")
    print(
        "Best by diff: {name} diff={avg_score_diff:.2f} survival={avg_survival_turn:.2f} "
        "win={win_rate:.1%}".format(**best)
    )
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
