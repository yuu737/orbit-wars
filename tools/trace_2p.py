from __future__ import annotations

import argparse
import contextlib
import io
from typing import Any


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-turn 2P trace: planets / ships / production per seat.")
    p.add_argument("--agent", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--player-index", type=int, default=0)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--until", type=int, default=300)
    return p.parse_args()


def per_player(obs: dict[str, Any], n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        planets = [pl for pl in obs["planets"] if pl[1] == i]
        ships = sum(pl[5] for pl in planets) + sum(f[6] for f in obs["fleets"] if f[1] == i)
        prod = sum(pl[4] for pl in planets)  # growth/production column
        out.append((len(planets), int(ships), int(prod)))
    return out


def main() -> None:
    args = parse_args()
    with suppress_output():
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
        env.run([args.agent, args.opponent] if args.player_index == 0 else [args.opponent, args.agent])

    me = args.player_index
    n = 2
    print(f"seed={args.seed} me=seat{me}  (planets/ships/prod) per seat; '*'=me")
    for t in range(0, min(args.until, len(env.steps)), args.every):
        obs = env.steps[t][me].observation
        pp = per_player(obs, n)
        cells = []
        for i in range(n):
            tag = "*" if i == me else " "
            cells.append(f"{tag}s{i}:{pp[i][0]:>2}p/{pp[i][1]:>4}sh/{pp[i][2]:>2}pr")
        print(f"  t={t:>3}  " + "   ".join(cells))


if __name__ == "__main__":
    main()
