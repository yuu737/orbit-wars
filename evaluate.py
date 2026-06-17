import argparse
import concurrent.futures
import contextlib
import io
import os
import statistics
import sys
import time
from typing import Any


_MAKE = None
_WORKER_DEVNULL = None


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable Orbit Wars evaluations across multiple seeds."
    )
    parser.add_argument(
        "--agent",
        default="main.py",
        help="Primary agent to evaluate. Default: main.py",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        dest="opponents",
        help=(
            "Opponent agent path or built-in agent name. Repeat to add more opponents. "
            "Default: random"
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games to run. Default: 20",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed to use. Default: 0",
    )
    parser.add_argument(
        "--seed-list",
        help="Comma-separated explicit seeds. Overrides --seed-start/--games when set.",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        choices=(2, 4),
        help="Number of players in each game. Default: 2",
    )
    parser.add_argument(
        "--player-index",
        type=int,
        default=0,
        help="Seat index for the primary agent. Default: 0",
    )
    parser.add_argument(
        "--both-seats",
        action="store_true",
        help="Run each seed once from every valid seat.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Default: 1",
    )
    return parser.parse_args()


def load_environment():
    # Suppress unrelated environment registration noise so evaluation output stays readable.
    with suppress_output():
        from kaggle_environments import make

    return make


def silence_worker_stdio() -> None:
    global _WORKER_DEVNULL
    if _WORKER_DEVNULL is not None:
        return

    _WORKER_DEVNULL = open(os.devnull, "w", encoding="utf-8")
    os.dup2(_WORKER_DEVNULL.fileno(), 1)
    os.dup2(_WORKER_DEVNULL.fileno(), 2)
    sys.stdout = _WORKER_DEVNULL
    sys.stderr = _WORKER_DEVNULL


def init_worker() -> None:
    global _MAKE
    silence_worker_stdio()
    _MAKE = load_environment()


def compute_total_ships(observation: dict[str, Any], player_index: int) -> int:
    planet_ships = sum(planet[5] for planet in observation["planets"] if planet[1] == player_index)
    fleet_ships = sum(fleet[6] for fleet in observation["fleets"] if fleet[1] == player_index)
    return planet_ships + fleet_ships


def compute_survival_turn(steps: list[list[Any]], player_index: int) -> int:
    last_turn_with_assets = 0
    for turn, turn_states in enumerate(steps):
        observation = turn_states[player_index].observation
        if compute_total_ships(observation, player_index) > 0:
            last_turn_with_assets = turn
    return last_turn_with_assets


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


def run_game_with_make(
    make,
    seed: int,
    primary_agent: str,
    opponents: list[str],
    player_index: int,
    players: int,
) -> dict[str, Any]:
    with suppress_output():
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    agents = build_agents(primary_agent, opponents, player_index, players)

    with suppress_output():
        env.run(agents)

    primary_state = env.steps[-1][player_index]

    primary_observation = primary_state.observation
    primary_score = compute_total_ships(primary_observation, player_index)
    all_scores = [compute_total_ships(state.observation, index) for index, state in enumerate(env.steps[-1])]
    best_opponent_score = max(score for index, score in enumerate(all_scores) if index != player_index)
    score_diff = primary_score - best_opponent_score
    reward = primary_state.reward
    game_length = len(env.steps) - 1
    survival_turn = compute_survival_turn(env.steps, player_index)
    crashed = primary_state.status != "DONE"
    placement = 1 + sum(score > primary_score for score in all_scores if score is not None)

    # Kaggle's environment can award the first seat on exact score ties.
    # For bot strength analysis, treat score_diff == 0 as a draw instead.
    won = score_diff > 0
    draw = score_diff == 0
    lost = score_diff < 0

    return {
        "seed": seed,
        "player_index": player_index,
        "players": players,
        "reward": reward,
        "won": won,
        "draw": draw,
        "lost": lost,
        "score_diff": score_diff,
        "primary_score": primary_score,
        "best_opponent_score": best_opponent_score,
        "placement": placement,
        "game_length": game_length,
        "survival_turn": survival_turn,
        "crashed": crashed,
        "status": primary_state.status,
    }


def run_game(task: tuple[int, str, list[str], int, int]) -> dict[str, Any]:
    global _MAKE
    if _MAKE is None:
        _MAKE = load_environment()

    seed, primary_agent, opponents, player_index, players = task
    return run_game_with_make(_MAKE, seed, primary_agent, opponents, player_index, players)


def summarize(results: list[dict[str, Any]]) -> str:
    games = len(results)
    wins = sum(result["won"] for result in results)
    draws = sum(result["draw"] for result in results)
    losses = sum(result["lost"] for result in results)
    crashes = sum(result["crashed"] for result in results)

    avg_score_diff = statistics.fmean(result["score_diff"] for result in results)
    avg_primary_score = statistics.fmean(result["primary_score"] for result in results)
    avg_best_opponent_score = statistics.fmean(result["best_opponent_score"] for result in results)
    avg_game_length = statistics.fmean(result["game_length"] for result in results)
    avg_survival_turn = statistics.fmean(result["survival_turn"] for result in results)
    avg_placement = statistics.fmean(result["placement"] for result in results)
    players = results[0]["players"]

    lines = [
        "Orbit Wars evaluation summary",
        f"Players: {players}",
        f"Games: {games}",
        f"Wins: {wins} ({wins / games:.1%})",
        f"Draws: {draws} ({draws / games:.1%})",
        f"Losses: {losses} ({losses / games:.1%})",
        f"Crash rate: {crashes / games:.1%}",
        f"Average score diff: {avg_score_diff:.2f}",
        f"Average primary score: {avg_primary_score:.2f}",
        f"Average best-opponent score: {avg_best_opponent_score:.2f}",
        f"Average placement: {avg_placement:.2f}",
        f"Average game length: {avg_game_length:.2f}",
        f"Average survival turn: {avg_survival_turn:.2f}",
    ]

    lines.extend(["", "Per-seed results:"])
    for result in results:
        lines.append(
            "  seed={seed:<3} seat={player_index} place={placement} reward={reward:<2} "
            "score={primary_score:<4} best_opp={best_opponent_score:<4} diff={score_diff:<5} "
            "length={game_length:<3} survival={survival_turn:<3} status={status}".format(**result)
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    opponents = args.opponents or ["random"]
    if args.player_index < 0 or args.player_index >= args.players:
        raise ValueError(f"--player-index must be between 0 and {args.players - 1} for --players {args.players}")

    seat_indices = [args.player_index]
    if args.both_seats:
        seat_indices = list(range(args.players))

    if args.seed_list:
        seeds = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
    else:
        seeds = [args.seed_start + offset for offset in range(args.games)]

    tasks = []
    for seat_index in seat_indices:
        for seed in seeds:
            tasks.append((seed, args.agent, opponents, seat_index, args.players))

    start_time = time.perf_counter()
    if args.workers <= 1:
        global _MAKE
        _MAKE = load_environment()
        results = [run_game(task) for task in tasks]
    else:
        max_workers = min(args.workers, len(tasks))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
        ) as executor:
            results = list(executor.map(run_game, tasks))

    elapsed = time.perf_counter() - start_time
    print(summarize(results))
    print("")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Workers: {max(1, min(args.workers, len(tasks)))}")


if __name__ == "__main__":
    main()
