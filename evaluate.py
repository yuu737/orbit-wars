import argparse
import contextlib
import io
import statistics
from typing import Any


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
        default="random",
        help="Opponent agent path or built-in agent name. Default: random",
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
        "--player-index",
        type=int,
        default=0,
        choices=(0, 1),
        help="Seat index for the primary agent. Default: 0",
    )
    return parser.parse_args()


def load_environment():
    # Suppress unrelated environment registration noise so evaluation output stays readable.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make

    return make


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


def run_game(make, seed: int, primary_agent: str, opponent_agent: str, player_index: int) -> dict[str, Any]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    agents = [primary_agent, opponent_agent]
    if player_index == 1:
        agents = [opponent_agent, primary_agent]

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env.run(agents)

    primary_state = env.steps[-1][player_index]
    opponent_state = env.steps[-1][1 - player_index]

    primary_observation = primary_state.observation
    opponent_observation = opponent_state.observation
    primary_score = compute_total_ships(primary_observation, player_index)
    opponent_score = compute_total_ships(opponent_observation, 1 - player_index)
    score_diff = primary_score - opponent_score
    reward = primary_state.reward
    game_length = len(env.steps) - 1
    survival_turn = compute_survival_turn(env.steps, player_index)
    crashed = primary_state.status != "DONE"

    return {
        "seed": seed,
        "reward": reward,
        "won": reward == 1,
        "draw": reward == 0,
        "lost": reward == -1,
        "score_diff": score_diff,
        "game_length": game_length,
        "survival_turn": survival_turn,
        "crashed": crashed,
        "status": primary_state.status,
    }


def summarize(results: list[dict[str, Any]]) -> str:
    games = len(results)
    wins = sum(result["won"] for result in results)
    draws = sum(result["draw"] for result in results)
    losses = sum(result["lost"] for result in results)
    crashes = sum(result["crashed"] for result in results)

    avg_score_diff = statistics.fmean(result["score_diff"] for result in results)
    avg_game_length = statistics.fmean(result["game_length"] for result in results)
    avg_survival_turn = statistics.fmean(result["survival_turn"] for result in results)

    lines = [
        "Orbit Wars evaluation summary",
        f"Games: {games}",
        f"Wins: {wins} ({wins / games:.1%})",
        f"Draws: {draws} ({draws / games:.1%})",
        f"Losses: {losses} ({losses / games:.1%})",
        f"Crash rate: {crashes / games:.1%}",
        f"Average score diff: {avg_score_diff:.2f}",
        f"Average game length: {avg_game_length:.2f}",
        f"Average survival turn: {avg_survival_turn:.2f}",
        "",
        "Per-seed results:",
    ]

    for result in results:
        lines.append(
            "  seed={seed:<3} reward={reward:<2} diff={score_diff:<4} "
            "length={game_length:<3} survival={survival_turn:<3} status={status}".format(**result)
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    make = load_environment()

    results = []
    for offset in range(args.games):
        seed = args.seed_start + offset
        results.append(
            run_game(
                make=make,
                seed=seed,
                primary_agent=args.agent,
                opponent_agent=args.opponent,
                player_index=args.player_index,
            )
        )

    print(summarize(results))


if __name__ == "__main__":
    main()
