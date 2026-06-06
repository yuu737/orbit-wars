import argparse
import contextlib
import io

from kaggle_environments import make


CHECKPOINTS = (25, 50, 80, 120)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def total_ships(obs, player):
    return sum(p[5] for p in obs["planets"] if p[1] == player) + sum(f[6] for f in obs["fleets"] if f[1] == player)


def planet_stats(obs, player):
    owned = [p for p in obs["planets"] if p[1] == player]
    return {
        "planets": len(owned),
        "production": sum(p[6] for p in owned),
        "ships": total_ships(obs, player),
    }


def classify(final_reward, survival, stats_by_turn):
    if final_reward == 1:
        return "win"
    early = stats_by_turn.get(50) or stats_by_turn.get(25)
    mid = stats_by_turn.get(120) or stats_by_turn.get(80)
    if survival <= 80:
        if early and early["planets"] <= 2:
            return "opening_loss"
        return "early_defense_collapse"
    if mid and mid["production"] <= 8:
        return "bad_overexpand_or_low_economy"
    if survival <= 180:
        return "defense_collapse"
    return "late_or_endgame_loss"


def run_one(agent, opponent, seed, players, seat):
    agents = [opponent] * players
    agents[seat] = agent
    with quiet():
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)

    stats_by_turn = {}
    survival = 0
    for turn, states in enumerate(env.steps):
        obs = states[seat].observation
        if total_ships(obs, seat) > 0:
            survival = turn
        if turn in CHECKPOINTS:
            stats_by_turn[turn] = planet_stats(obs, seat)

    final_state = env.steps[-1][seat]
    final_stats = planet_stats(final_state.observation, seat)
    label = classify(final_state.reward, survival, stats_by_turn)
    return {
        "seed": seed,
        "seat": seat,
        "reward": final_state.reward,
        "survival": survival,
        "label": label,
        "final": final_stats,
        "checkpoints": stats_by_turn,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify Orbit Wars loss patterns.")
    parser.add_argument("--agent", default="bots/main_v4_1_counter_snipe.py")
    parser.add_argument("--opponent", default="bots/hairate2.py")
    parser.add_argument("--players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--both-seats", action="store_true")
    args = parser.parse_args()

    seats = range(args.players) if args.both_seats else (0,)
    counts = {}
    for seat in seats:
        for offset in range(args.games):
            result = run_one(args.agent, args.opponent, args.seed_start + offset, args.players, seat)
            counts[result["label"]] = counts.get(result["label"], 0) + 1
            checkpoints = " ".join(
                f"t{turn}:p{stats['planets']}/prod{stats['production']}/s{stats['ships']}"
                for turn, stats in sorted(result["checkpoints"].items())
            )
            print(
                f"seed={result['seed']} seat={seat} reward={result['reward']} "
                f"survival={result['survival']} label={result['label']} {checkpoints}"
            )

    print("summary", " ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
