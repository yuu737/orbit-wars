import argparse
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_seed_list(seed_list: str | None, seed_start: int, games: int) -> list[int]:
    if seed_list:
        return [int(part.strip()) for part in seed_list.split(",") if part.strip()]
    return [seed_start + offset for offset in range(games)]


def build_agents(primary_agent: str, opponents: list[str], player_index: int, players: int) -> list[str]:
    if len(opponents) == 1:
        opponent_pool = opponents * (players - 1)
    elif len(opponents) == players - 1:
        opponent_pool = opponents
    else:
        raise ValueError(
            f"--players {players} requires either 1 opponent or exactly {players - 1} opponents; "
            f"got {len(opponents)}"
        )

    agents = list(opponent_pool)
    agents.insert(player_index, primary_agent)
    return agents


def compute_total_ships(observation: dict[str, Any], player_index: int) -> int:
    planet_ships = sum(planet[5] for planet in observation["planets"] if planet[1] == player_index)
    fleet_ships = sum(fleet[6] for fleet in observation["fleets"] if fleet[1] == player_index)
    return planet_ships + fleet_ships


def summarize_env(env: Any, seed: int, player_index: int, players: int, agents: list[str]) -> dict[str, Any]:
    final_states = env.steps[-1]
    scores = [
        compute_total_ships(state.observation, index)
        for index, state in enumerate(final_states)
    ]
    primary_score = scores[player_index]
    best_opponent_score = max(score for index, score in enumerate(scores) if index != player_index)
    placement = 1 + sum(score > primary_score for score in scores)
    return {
        "seed": seed,
        "players": players,
        "player_index": player_index,
        "agents": agents,
        "scores": scores,
        "primary_score": primary_score,
        "best_opponent_score": best_opponent_score,
        "score_diff": primary_score - best_opponent_score,
        "placement": placement,
        "game_length": len(env.steps) - 1,
        "primary_status": final_states[player_index].status,
        "primary_reward": final_states[player_index].reward,
    }


def default_out_dir(prefix: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("research_runs") / f"replays_{prefix}_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Orbit Wars replay HTML files for manual inspection.")
    parser.add_argument("--agent", default="sample7/main.py", help="Primary agent path.")
    parser.add_argument(
        "--opponent",
        action="append",
        dest="opponents",
        help="Opponent agent path. Repeat for 4P opponent pool.",
    )
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed-list", help="Comma-separated explicit seeds. Overrides --seed-start/--games.")
    parser.add_argument("--out-dir", help="Output directory. Default: research_runs/replays_<prefix>_<timestamp>.")
    parser.add_argument("--prefix", default="sample7_pool", help="File prefix for generated replays.")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opponents = args.opponents or ["sample7/main.py", "sample8/main.py", "bots/hairate5.py"]
    seeds = parse_seed_list(args.seed_list, args.seed_start, args.games)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    with suppress_output():
        from kaggle_environments import make

    index: list[dict[str, Any]] = []
    for seed in seeds:
        agents = build_agents(args.agent, opponents, args.player_index, args.players)
        with suppress_output():
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            html = env.render(mode="html", width=args.width, height=args.height)

        summary = summarize_env(env, seed, args.player_index, args.players, agents)
        html_path = out_dir / f"{args.prefix}_seed{seed}.html"
        meta_path = out_dir / f"{args.prefix}_seed{seed}.json"

        html_path.write_text(html, encoding="utf-8")
        meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        summary["html"] = str(html_path)
        summary["meta"] = str(meta_path)
        index.append(summary)
        print(
            f"seed={seed} place={summary['placement']} diff={summary['score_diff']} "
            f"length={summary['game_length']} html={html_path}"
        )

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nDone. Replays saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Rendering writes large HTML files, so this script intentionally runs serially.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
