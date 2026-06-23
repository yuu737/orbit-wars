from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-turn 4P trace: planets and ships per player.")
    p.add_argument("--agent", required=True)
    p.add_argument("--opponent", action="append", dest="opponents", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--player-index", type=int, default=0)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--until", type=int, default=120)
    return p.parse_args()


def per_player(obs: dict[str, Any], n: int) -> list[tuple[int, int]]:
    out = []
    for i in range(n):
        planets = [pl for pl in obs["planets"] if pl[1] == i]
        ships = sum(pl[5] for pl in planets) + sum(f[6] for f in obs["fleets"] if f[1] == i)
        out.append((len(planets), int(ships)))
    return out


def build_agents(primary: str, opponents: list[str], idx: int) -> list[str]:
    pool = opponents * 3 if len(opponents) == 1 else list(opponents)
    a = list(pool)
    a.insert(idx, primary)
    return a


def main() -> None:
    args = parse_args()
    with suppress_output():
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
        env.run(build_agents(args.agent, args.opponents, args.player_index))

    me = args.player_index
    n = 4
    print(f"seed={args.seed} me=seat{me}  (planets,ships) per seat; '*'=me")
    for t in range(0, min(args.until, len(env.steps)), args.every):
        obs = env.steps[t][me].observation
        pp = per_player(obs, n)
        cells = []
        for i in range(n):
            tag = "*" if i == me else " "
            cells.append(f"{tag}s{i}:{pp[i][0]:>2}p/{pp[i][1]:>4}sh")
        print(f"  t={t:>3}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
