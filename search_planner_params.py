import argparse
import contextlib
import importlib.util
import io
import itertools
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def quiet_stdio():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search PlannerConfig variants on a fixed Orbit Wars benchmark."
    )
    parser.add_argument(
        "--agent",
        default="bots/main_v6_1_frontier_gated.py",
        help="Planner bot module to tune. Default: bots/main_v6_1_frontier_gated.py",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/hairate_fixed_2p.json"),
        help="Benchmark JSON path. Default: benchmarks/hairate_fixed_2p.json",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="sets",
        default=[],
        help=(
            "Parameter candidate list in key=v1,v2 form. Repeat to build a cartesian product. "
            "Example: --set roi_threshold=2.0,2.2 --set reserve_margin=2,3"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of variants to evaluate after expansion. Default: no cap",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="How many top variants to print at the end. Default: 8",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Optional output path for one JSON object per variant summary.",
    )
    return parser.parse_args()


def load_agent_module(path: str):
    agent_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("planner_search_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_scalar(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in lowered for ch in (".", "e")):
            return float(lowered)
        return int(lowered)
    except ValueError:
        return raw.strip()


def parse_variant_sets(entries: list[str]) -> list[dict[str, Any]]:
    dimensions: list[tuple[str, list[Any]]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --set value: {entry}")
        key, raw_values = entry.split("=", 1)
        values = [parse_scalar(part) for part in raw_values.split(",") if part.strip()]
        if not values:
            raise ValueError(f"No values provided for --set {key}")
        dimensions.append((key.strip(), values))

    if not dimensions:
        return [{"name": "base"}]

    variants = [{"name": "base"}]
    for combo in itertools.product(*(values for _, values in dimensions)):
        variant = {}
        name_parts = []
        for (key, _), value in zip(dimensions, combo):
            variant[key] = value
            name_parts.append(f"{key}={value}")
        variant["name"] = " | ".join(name_parts)
        variants.append(variant)
    return variants


def load_benchmark(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"players", "opponents", "seeds"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Benchmark file missing required keys: {sorted(missing)}")
    return data


def apply_variant(module, variant: dict[str, Any], base2, base4) -> None:
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


def run_game(make, module, variant, benchmark, seed, player_index):
    apply_variant(module, variant, module._SEARCH_BASE_CONFIG_2P, module._SEARCH_BASE_CONFIG_4P)
    players = int(benchmark["players"])
    opponents = list(benchmark["opponents"])
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
        "draw": primary_state.reward == 0,
        "score_diff": primary_score - best_opp,
        "primary_score": primary_score,
        "best_opponent_score": best_opp,
        "placement": 1 + sum(score > primary_score for score in scores),
        "game_length": len(env.steps) - 1,
        "survival_turn": compute_survival_turn(env.steps, player_index),
        "status": primary_state.status,
    }


def summarize_variant(variant: dict[str, Any], results: list[dict[str, Any]], benchmark_name: str) -> dict[str, Any]:
    games = len(results)
    return {
        "name": variant["name"],
        "benchmark": benchmark_name,
        "variant": variant,
        "games": games,
        "wins": sum(result["won"] for result in results),
        "draws": sum(result["draw"] for result in results),
        "win_rate": sum(result["won"] for result in results) / games,
        "avg_score_diff": statistics.fmean(result["score_diff"] for result in results),
        "avg_survival_turn": statistics.fmean(result["survival_turn"] for result in results),
        "avg_placement": statistics.fmean(result["placement"] for result in results),
        "avg_primary_score": statistics.fmean(result["primary_score"] for result in results),
        "avg_best_opponent_score": statistics.fmean(result["best_opponent_score"] for result in results),
    }


def main() -> None:
    args = parse_args()
    benchmark = load_benchmark(args.benchmark)
    variants = parse_variant_sets(args.sets)
    if args.limit > 0:
        variants = variants[: args.limit]

    module = load_agent_module(args.agent)
    module._SEARCH_BASE_CONFIG_2P = module.CONFIG_2P
    module._SEARCH_BASE_CONFIG_4P = module.CONFIG_4P
    seats = list(range(int(benchmark["players"]))) if benchmark.get("both_seats", False) else [0]

    with quiet_stdio():
        from kaggle_environments import make

    summaries = []
    start = time.perf_counter()
    jsonl_handle = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        for variant in variants:
            results = []
            for seat in seats:
                for seed in benchmark["seeds"]:
                    results.append(run_game(make, module, variant, benchmark, int(seed), seat))
            summary = summarize_variant(variant, results, benchmark.get("name", args.benchmark.stem))
            summaries.append(summary)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(summary, sort_keys=True) + "\n")
            print(
                "{name:<45} games={games:<3} win={win_rate:>6.1%} diff={avg_score_diff:>8.2f} "
                "survival={avg_survival_turn:>6.2f} place={avg_placement:.2f}".format(**summary)
            )
    finally:
        if jsonl_handle:
            jsonl_handle.close()

    print("")
    ranked = sorted(
        summaries,
        key=lambda row: (
            row["avg_score_diff"],
            row["win_rate"],
            row["avg_survival_turn"],
            -row["avg_placement"],
        ),
        reverse=True,
    )
    for summary in ranked[: args.top]:
        print(
            "top {name} diff={avg_score_diff:.2f} win={win_rate:.1%} "
            "survival={avg_survival_turn:.2f} place={avg_placement:.2f}".format(**summary)
        )
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
