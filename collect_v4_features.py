import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make


CHECKPOINTS = (25, 50, 80, 120, 180, 260)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def total_ships(obs: dict[str, Any], player: int) -> int:
    return sum(p[5] for p in obs["planets"] if p[1] == player) + sum(
        f[6] for f in obs["fleets"] if f[1] == player
    )


def stats(obs: dict[str, Any], player: int) -> dict[str, int]:
    owned = [p for p in obs["planets"] if p[1] == player]
    neutral = [p for p in obs["planets"] if p[1] == -1]
    high_owned = [p for p in owned if p[6] >= 4]
    return {
        "planets": len(owned),
        "production": sum(p[6] for p in owned),
        "ships": total_ships(obs, player),
        "high_prod_planets": len(high_owned),
        "high_prod": sum(p[6] for p in high_owned),
        "neutral_planets": len(neutral),
        "neutral_prod": sum(p[6] for p in neutral),
    }


def score_diff(obs: dict[str, Any], player: int, players: int) -> int:
    mine = total_ships(obs, player)
    others = [total_ships(obs, idx) for idx in range(players) if idx != player]
    return mine - max(others)


def classify(reward: int, survival: int, checkpoints: dict[int, dict[str, int]]) -> str:
    if reward == 1:
        return "win"
    early = checkpoints.get(50) or checkpoints.get(25)
    mid = checkpoints.get(120) or checkpoints.get(80)
    if survival <= 80:
        if early and early["planets"] <= 2:
            return "opening_loss"
        return "early_defense_collapse"
    if mid and mid["production"] <= 8:
        return "bad_overexpand_or_low_economy"
    if survival <= 180:
        return "defense_collapse"
    return "late_or_endgame_loss"


def build_agents(agent: str, opponents: list[str], seat: int, players: int) -> list[str]:
    if len(opponents) == 1:
        agents = opponents * players
    elif len(opponents) == players - 1:
        agents = list(opponents)
        agents.insert(seat, opponents[0])
    else:
        raise ValueError(f"Need 1 opponent or {players - 1} opponents; got {len(opponents)}")
    agents[seat] = agent
    return agents


def run_one(agent: str, opponents: list[str], seed: int, seat: int, players: int) -> dict[str, Any]:
    with quiet():
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(build_agents(agent, opponents, seat, players))

    checkpoints = {}
    survival = 0
    for turn, states in enumerate(env.steps):
        obs = states[seat].observation
        if total_ships(obs, seat) > 0:
            survival = turn
        if turn in CHECKPOINTS:
            turn_stats = stats(obs, seat)
            turn_stats["score_diff"] = score_diff(obs, seat, players)
            checkpoints[turn] = turn_stats

    final_state = env.steps[-1][seat]
    final_obs = final_state.observation
    final_stats = stats(final_obs, seat)
    final_stats["score_diff"] = score_diff(final_obs, seat, players)
    label = classify(final_state.reward, survival, checkpoints)
    return {
        "agent": agent,
        "opponents": opponents,
        "seed": seed,
        "seat": seat,
        "players": players,
        "reward": final_state.reward,
        "status": final_state.status,
        "survival": survival,
        "label": label,
        "final": final_stats,
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect V4 outcome features as JSONL.")
    parser.add_argument("--agent", action="append", required=True)
    parser.add_argument("--opponent", action="append", dest="opponents", default=[])
    parser.add_argument("--players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--both-seats", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("v4_features.jsonl"))
    args = parser.parse_args()

    opponents = args.opponents or ["main.py"]
    seats = range(args.players) if args.both_seats else (0,)
    with args.output.open("w", encoding="utf-8") as handle:
        for agent in args.agent:
            for seat in seats:
                for offset in range(args.games):
                    result = run_one(agent, opponents, args.seed_start + offset, seat, args.players)
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
                    print(
                        f"{Path(agent).name} seed={result['seed']} seat={seat} "
                        f"label={result['label']} survival={result['survival']} "
                        f"diff={result['final']['score_diff']}"
                    )


if __name__ == "__main__":
    main()
