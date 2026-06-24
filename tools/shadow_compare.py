"""Form-2 imitation: shadow OUR agent over a top player's replay and diff the actions.

For each turn we feed the EXPERT seat's observation to our agent (so it decides AS that
seat) and compare its launches to what the expert actually did that turn. The goal is to
surface SYSTEMATIC differences (e.g. we reinforce internally where the expert attacks; we
spread where the expert concentrates) that point to a concrete, encodable gap.

Usage:
    python tools/shadow_compare.py <replay.json> [--agent <dir>] [--seat N] [--until T]

--agent: directory of our agent (default sample8). --seat: expert seat to imitate
(default: the winner = highest reward). Heavy-ish (runs the planner per turn).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def import_agent(agent_dir):
    d = os.path.abspath(agent_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    main_path = os.path.join(d, "main.py")
    spec = importlib.util.spec_from_file_location(f"agent_{os.path.basename(d)}", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required so dataclass/typing can resolve the module
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def get_obs(step_entry):
    for agent in step_entry:
        obs = agent.get("observation") if isinstance(agent, dict) else None
        if obs and isinstance(obs.get("planets"), list) and len(obs["planets"]) > 0:
            return obs
    return step_entry[0].get("observation") or {} if step_entry else {}


def seat_obs(step_entry, seat):
    """Observation as seen for a given seat (fall back to the shared full-board obs)."""
    if seat < len(step_entry) and isinstance(step_entry[seat], dict):
        o = step_entry[seat].get("observation")
        if o and isinstance(o.get("planets"), list) and len(o["planets"]) > 0:
            return o
    return get_obs(step_entry)


def planet_xy(obs):
    return {int(p[0]): (float(p[2]), float(p[3])) for p in obs.get("planets", [])}


def planet_owner_prod(obs):
    return {int(p[0]): (int(p[1]), float(p[6])) for p in obs.get("planets", [])}


def infer_target(obs, from_id, angle):
    xy = planet_xy(obs)
    if int(from_id) not in xy:
        return None
    sx, sy = xy[int(from_id)]
    best = None
    for p in obs.get("planets", []):
        pid = int(p[0])
        if pid == int(from_id):
            continue
        a = math.atan2(float(p[3]) - sy, float(p[2]) - sx)
        da = abs(((a - float(angle) + math.pi) % (2 * math.pi)) - math.pi)
        if best is None or da < best[0]:
            best = (da, pid)
    return best[1] if best else None


def categorize(obs, moves, seat):
    """moves: list of [from_planet, angle, ships] -> tally by target type for `seat`."""
    op = planet_owner_prod(obs)
    out = {"n": 0, "ships": 0.0, "to_neutral": 0.0, "to_enemy": 0.0, "to_own": 0.0,
           "sizes": [], "hp_neutral_ships": 0.0}
    for mv in moves:
        if len(mv) < 3:
            continue
        frm, ang, ships = int(mv[0]), float(mv[1]), float(mv[2])
        if ships <= 0:
            continue
        out["n"] += 1
        out["ships"] += ships
        out["sizes"].append(ships)
        tid = infer_target(obs, frm, ang)
        owner, prod = op.get(tid, (-1, 0.0)) if tid is not None else (-1, 0.0)
        if owner < 0:
            out["to_neutral"] += ships
            if prod >= 4:
                out["hp_neutral_ships"] += ships
        elif owner == seat:
            out["to_own"] += ships
        else:
            out["to_enemy"] += ships
    return out


def expert_launches(prev_obs, cur_obs, seat):
    """Expert's launches at turn t = fleets owned by seat present at t+1 but not t."""
    prev_f = {int(f[0]) for f in prev_obs.get("fleets", [])}
    moves = []
    for f in cur_obs.get("fleets", []):
        if int(f[0]) not in prev_f and int(f[1]) == seat:
            moves.append([int(f[5]), float(f[4]), float(f[6])])  # from, angle, ships
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--agent", default="sample8")
    ap.add_argument("--seat", type=int, default=None)
    ap.add_argument("--until", type=int, default=500)
    args = ap.parse_args()

    data = load(args.path)
    steps = data.get("steps", []) or []
    rewards = data.get("rewards")
    info = data.get("info", {}) or {}
    n = min(len(steps), args.until)

    if isinstance(rewards, dict):
        rwd = {int(k): v for k, v in rewards.items()}
    elif isinstance(rewards, list):
        rwd = {i: rewards[i] for i in range(len(rewards))}
    else:
        rwd = {}
    seat = args.seat if args.seat is not None else (max(rwd, key=lambda k: rwd[k]) if rwd else 0)
    tn = info.get("TeamNames")
    name = tn[seat] if tn and seat < len(tn) else None
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    print(f"replay={args.path} seed={info.get('seed')}")
    print(f"shadow agent={args.agent}  imitating seat {seat} (name={name!r}, reward={rwd.get(seat)})")

    mod = import_agent(args.agent)
    import torch  # noqa

    agg_our = {"n": 0, "ships": 0.0, "to_neutral": 0.0, "to_enemy": 0.0, "to_own": 0.0, "hp_neutral_ships": 0.0, "sizes": []}
    agg_exp = {k: (0 if k == "n" else (0.0 if k != "sizes" else [])) for k in agg_our}

    for t in range(n - 1):
        obs = dict(seat_obs(steps[t], seat))
        obs["player"] = seat
        try:
            with torch.no_grad():
                our_moves = mod.agent(obs)
        except Exception as e:
            our_moves = []
        our = categorize(obs, our_moves, seat)
        exp = categorize(obs, expert_launches(get_obs(steps[t]), get_obs(steps[t + 1]), seat), seat)
        for k in ("n", "ships", "to_neutral", "to_enemy", "to_own", "hp_neutral_ships"):
            agg_our[k] += our[k]
            agg_exp[k] += exp[k]
        agg_our["sizes"] += our["sizes"]
        agg_exp["sizes"] += exp["sizes"]

    def avg(s):
        return sum(s) / len(s) if s else 0.0

    print("=" * 64)
    print(f"{'metric':<22}{'OUR(shadow)':>20}{'EXPERT':>20}")
    print("-" * 64)
    for k, lab in [("n", "launches"), ("ships", "ships committed"),
                   ("to_neutral", " -> neutral"), ("hp_neutral_ships", "   -> highprod neut"),
                   ("to_enemy", " -> enemy"), ("to_own", " -> own (reinforce)")]:
        print(f"{lab:<22}{agg_our[k]:>20.0f}{agg_exp[k]:>20.0f}")
    print(f"{'avg launch size':<22}{avg(agg_our['sizes']):>20.1f}{avg(agg_exp['sizes']):>20.1f}")
    # headline ratios
    print("-" * 64)
    so = agg_our["ships"] or 1.0
    se = agg_exp["ships"] or 1.0
    print(f"{'reinforce share %':<22}{100*agg_our['to_own']/so:>20.1f}{100*agg_exp['to_own']/se:>20.1f}")
    print(f"{'enemy share %':<22}{100*agg_our['to_enemy']/so:>20.1f}{100*agg_exp['to_enemy']/se:>20.1f}")
    print(f"{'neutral share %':<22}{100*agg_our['to_neutral']/so:>20.1f}{100*agg_exp['to_neutral']/se:>20.1f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
