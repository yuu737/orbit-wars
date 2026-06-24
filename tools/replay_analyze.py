"""Analyze ONE Orbit Wars replay JSON.

Per spec: confirm structure, aggregate per-player/per-turn stats, print a table of
key turns, and identify the single divergence turn. Evidence only -- every number
comes straight from the replay JSON. No agent code is touched.

Usage:
    python tools/replay_analyze.py <path-to-replay.json> [--me <player_id>]

If --me is omitted, "me" defaults to the loser (lowest final reward); fall back to 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_obs(step_entry):
    """Return the first observation in a step that carries a non-empty planets list."""
    for agent in step_entry:
        obs = agent.get("observation") if isinstance(agent, dict) else None
        if obs and isinstance(obs.get("planets"), list) and len(obs["planets"]) > 0:
            return obs
    # fall back to player 0's observation even if empty
    if step_entry and isinstance(step_entry[0], dict):
        return step_entry[0].get("observation") or {}
    return {}


def per_player(obs, players):
    pl_cnt = {p: 0 for p in players}
    pl_sh = {p: 0.0 for p in players}
    prod = {p: 0.0 for p in players}
    for pl in obs.get("planets", []):
        owner = int(pl[1])
        if owner in pl_cnt:
            pl_cnt[owner] += 1
            pl_sh[owner] += float(pl[5])
            prod[owner] += float(pl[6])
    fl_sh = {p: 0.0 for p in players}
    fl_cnt = {p: 0 for p in players}
    for f in obs.get("fleets", []):
        owner = int(f[1])
        if owner in fl_sh:
            fl_sh[owner] += float(f[6])
            fl_cnt[owner] += 1
    tot = {p: pl_sh[p] + fl_sh[p] for p in players}
    return {
        "planets": pl_cnt, "planet_ships": pl_sh, "fleet_ships": fl_sh,
        "total_ships": tot, "production": prod, "fleets": fl_cnt,
    }


def fleet_ids(obs):
    return {int(f[0]): f for f in obs.get("fleets", [])}


def planet_map(obs):
    return {int(p[0]): p for p in obs.get("planets", [])}


def infer_target(obs, from_id, angle):
    """Best-effort target by angular alignment of the launch heading (radians assumed).

    Returns (ang_err_deg, target_id, target_owner, target_prod, dist) or None. Heuristic
    (ignores orbital lead), so trust the OWNER-FLIP evidence over this for the actual hit.
    """
    pm = planet_map(obs)
    if int(from_id) not in pm:
        return None
    sp = pm[int(from_id)]
    sx, sy = float(sp[2]), float(sp[3])
    best = None
    for p in obs.get("planets", []):
        pid = int(p[0])
        if pid == int(from_id):
            continue
        dx = float(p[2]) - sx
        dy = float(p[3]) - sy
        a = math.atan2(dy, dx)
        da = abs(((a - float(angle) + math.pi) % (2 * math.pi)) - math.pi)
        if best is None or da < best[0]:
            best = (da, pid, int(p[1]), float(p[6]), math.hypot(dx, dy))
    if best is None:
        return None
    return (math.degrees(best[0]), best[1], best[2], best[3], best[4])


def deviation_detail(obs_list, t):
    """Print new launches and owner flips at turn t (vs t-1)."""
    if t <= 0 or t >= len(obs_list):
        return
    prev = obs_list[t - 1]
    cur = obs_list[t]
    # new launches: fleet ids present at t but not t-1
    prev_f = fleet_ids(prev)
    cur_f = fleet_ids(cur)
    new_ids = [fid for fid in cur_f if fid not in prev_f]
    launches = []
    for fid in new_ids:
        f = cur_f[fid]
        owner = int(f[1]); frm = int(f[5]); ships = float(f[6]); ang = float(f[4])
        tgt = infer_target(cur, frm, ang)
        launches.append((owner, frm, ships, tgt))
    # owner flips: planet owner changed vs t-1
    pp = planet_map(prev); cp = planet_map(cur)
    flips = []
    for pid, p in cp.items():
        if pid in pp and int(pp[pid][1]) != int(p[1]):
            flips.append((pid, int(pp[pid][1]), int(p[1]), float(pp[pid][5]), float(p[5]), float(p[6])))
    if not launches and not flips:
        return
    print(f"  -- turn {t} --")
    for owner, frm, ships, tgt in sorted(launches):
        tg = ""
        if tgt:
            ang_err, tid, towner, tprod, dist = tgt
            kind = "neutral" if towner < 0 else ("own" if False else f"P{towner}")
            tg = f" -> ~planet {tid} (owner {towner}/{'neutral' if towner<0 else 'player'}, prod {tprod:.0f}, angErr {ang_err:.0f}deg)"
        print(f"     LAUNCH  player {owner}: {ships:.0f} ships  from planet {frm}{tg}")
    for pid, old, new, sh_b, sh_a, prod in flips:
        print(f"     FLIP    planet {pid}: owner {old} -> {new}   (ships {sh_b:.0f}->{sh_a:.0f}, prod {prod:.0f})")


def all_flips(obs_list):
    """Every owner change across the game: list of (turn, planet, old, new, ships_after, prod)."""
    out = []
    for t in range(1, len(obs_list)):
        pp = planet_map(obs_list[t - 1])
        cp = planet_map(obs_list[t])
        for pid, p in cp.items():
            if pid in pp and int(pp[pid][1]) != int(p[1]):
                out.append((t, pid, int(pp[pid][1]), int(p[1]), float(p[5]), float(p[6])))
    return out


def my_launches_from(obs_list, t, me):
    """Planets `me` launched FROM at turn t. Returns {planet_id: (ships, target_owner)}.

    target_owner is the best-effort owner of the planet the drain launch was aimed at
    (None if unknown). Lets us tell a third-party steal (attacked E1, home taken by E2).
    """
    if t <= 0 or t >= len(obs_list):
        return {}
    prev_f = fleet_ids(obs_list[t - 1])
    cur_f = fleet_ids(obs_list[t])
    out = {}
    for fid, f in cur_f.items():
        if fid not in prev_f and int(f[1]) == int(me):
            frm = int(f[5]); ships = float(f[6])
            tgt = infer_target(obs_list[t], frm, float(f[4]))
            towner = int(tgt[2]) if tgt else None
            if frm not in out or ships > out[frm][0]:
                out[frm] = (ships, towner)
    return out


def report_4p(series, obs_list, players, rwd, me, n_steps, detail_window):
    """4P-specific report: all seats, elimination order, who-killed-me, source-stripping."""
    enemies = [p for p in players if p != me]

    def seat_cell(s, p, tag):
        return f"{tag}s{p}:{s['planets'][p]:>2}p/{s['total_ships'][p]:>5.0f}sh/{s['production'][p]:>3.0f}pr"

    # elimination turn per seat (planets==0 and no ships in flight)
    elim = {}
    for p in players:
        et = None
        for t, s in series:
            if s["planets"][p] == 0 and s["total_ships"][p] == 0:
                et = t
                break
        elim[p] = et
    order = sorted(players, key=lambda p: (elim[p] if elim[p] is not None else n_steps + 1, p))

    print(f"me (analyzed seat): player {me}  reward={rwd.get(me)}  "
          f"elim_turn={elim[me]}  place_by_elim={order.index(me)+1}/{len(players)}")
    print("ELIMINATION ORDER (turn planets->0 & no fleets; None=survived):")
    for rank, p in enumerate(order, 1):
        print(f"  {rank}. player {p}  elim={elim[p]}  reward={rwd.get(p)}")
    print("=" * 70)

    # all-seat key-turn table
    my_elim = elim[me]
    key_turns = [tt for tt in (0, 10, 20, 30, 40, 50, 60, 70, 80) if tt < n_steps]
    for extra in (my_elim, n_steps - 1):
        if extra is not None and extra not in key_turns:
            key_turns.append(extra)
    key_turns = sorted(set(t for t in key_turns if t < n_steps))
    print("KEY-TURN TABLE (all seats; '*' = analyzed seat)")
    for t in key_turns:
        s = series[t][1]
        cells = [seat_cell(s, p, "*" if p == me else " ") for p in players]
        print(f"  t={t:>3}  " + "   ".join(cells))
    print("=" * 70)

    # who took MY planets (FLIPs where old owner == me), grouped by capturing player
    flips = all_flips(obs_list)
    took_me = {}  # new_owner -> list of (turn, planet, prod)
    for (t, pid, old, new, sh_a, prod) in flips:
        if old == me and new != me:
            took_me.setdefault(new, []).append((t, pid, prod))
    print("WHO TOOK MY PLANETS (FLIP old==me):")
    if not took_me:
        print("  (none — I never lost a planet by capture)")
    else:
        for owner in sorted(took_me, key=lambda o: -len(took_me[o])):
            evs = took_me[owner]
            prod_lost = sum(e[2] for e in evs)
            tag = "NEUTRAL" if owner < 0 else f"player {owner}"
            turns = ", ".join(f"t{e[0]}(P{e[1]}/pr{e[2]:.0f})" for e in evs[:12])
            print(f"  {tag}: took {len(evs)} of my planets (prod {prod_lost:.0f})  -> {turns}")
        killers = {o: l for o, l in took_me.items() if o >= 0}
        if killers:
            primary = max(killers, key=lambda o: len(killers[o]))
            print(f"  => PRIMARY KILLER: player {primary} (took {len(took_me[primary])} of my planets)")
    print("=" * 70)

    # SOURCE-STRIPPING: a planet I launched FROM gets captured soon after (<=eta_window)
    eta_window = 10
    print(f"SOURCE-STRIPPING (my planet captured within {eta_window}t after I launched FROM it):")
    hits = 0
    third_party = 0
    prod_lost_ss = 0.0
    for (t, pid, old, new, sh_a, prod) in flips:
        if old != me or new == me:
            continue
        drained_at = None
        drained_ships = 0.0
        drain_target = None
        for k in range(max(1, t - eta_window), t + 1):
            d = my_launches_from(obs_list, k, me)
            if pid in d:
                drained_at = k
                drained_ships, drain_target = d[pid]
        if drained_at is not None:
            hits += 1
            prod_lost_ss += prod
            who = "NEUTRAL" if new < 0 else f"player {new}"
            # third-party steal: captor differs from who I was attacking with that drain
            steal = ""
            if new >= 0 and drain_target is not None and drain_target >= 0 and drain_target != new:
                third_party += 1
                steal = f"  [3RD-PARTY STEAL: I attacked player {drain_target}, lost it to player {new}]"
            tg = f"(attacked player {drain_target})" if drain_target is not None and drain_target >= 0 else "(attacked neutral/unknown)"
            print(f"  planet {pid} (prod {prod:.0f}): launched {drained_ships:.0f} FROM it @t{drained_at} {tg}"
                  f"  -> captured by {who} @t{t} (gap {t-drained_at}t){steal}")
    if hits == 0:
        print("  (none detected)")
    else:
        print(f"  TOTAL: {hits} source-strip losses (prod {prod_lost_ss:.0f}); of these {third_party} were 3RD-PARTY STEALS")
    print("=" * 70)

    # deviation detail around my elimination (or last turn)
    pivot = my_elim if my_elim is not None else n_steps - 1
    w = max(0, int(detail_window))
    lo = max(1, pivot - w)
    hi = min(n_steps - 1, pivot + w)
    print(f"DEVIATION DETAIL around my collapse turn {pivot} (turns {lo}..{hi})")
    print("  (FLIP = actual owner change = solid; LAUNCH target is angle-heuristic)")
    for t in range(lo, hi + 1):
        deviation_detail(obs_list, t)
    print("=" * 70)


def find_seat_by_name(team_names, substr):
    if not team_names or not substr:
        return None
    for i, n in enumerate(team_names):
        if substr in str(n):
            return i
    return None


def find_cjk_seat(team_names):
    """Pick the user's seat. Prefer an exact name marker (the user is 藤田　佑) so that
    games with OTHER Japanese-named players don't mis-select; fall back to any CJK/kana
    seat only if no marker matches.
    """
    if not team_names:
        return None
    # 1) the user's distinctive surname (stable across replays)
    for i, n in enumerate(team_names):
        if "藤田" in str(n):
            return i
    # 2) fall back: any CJK/kana/mojibake seat (single-JP-player games)
    for i, n in enumerate(team_names):
        if any(ord(c) >= 0x3040 or ord(c) == 0xFFFD for c in str(n)):
            return i
    return None


def main():
    # make all output UTF-8 safe so mojibake TeamNames never crash a cp932 console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--me", type=int, default=None, help="seat index to analyze")
    ap.add_argument("--me-name", default=None,
                    help="substring of the TeamName to analyze (auto: mojibake JP name for 4P)")
    ap.add_argument("--detail-window", type=int, default=5,
                    help="turns before/after the divergence to print launches + owner flips")
    args = ap.parse_args()

    data = load(args.path)

    # ---- structure ----
    info = data.get("info", {}) or {}
    rewards = data.get("rewards", None)
    steps = data.get("steps", []) or []
    config = data.get("configuration", {}) or {}

    # players + rewards normalization
    if isinstance(rewards, dict):
        players = sorted(int(k) for k in rewards.keys())
        rwd = {int(k): v for k, v in rewards.items()}
    elif isinstance(rewards, list):
        players = list(range(len(rewards)))
        rwd = {i: rewards[i] for i in range(len(rewards))}
    else:
        # infer from first populated observation owners
        obs0 = get_obs(steps[0]) if steps else {}
        owners = sorted({int(pl[1]) for pl in obs0.get("planets", []) if int(pl[1]) >= 0})
        players = owners or [0, 1]
        rwd = {p: None for p in players}

    n_steps = len(steps)
    print("=" * 70)
    print(f"FILE        : {args.path}")
    print(f"seed        : {info.get('seed')}")
    print(f"TeamNames   : {info.get('TeamNames')}")
    print(f"rewards     : {rwd}")
    print(f"players     : {players}")
    print(f"n_steps     : {n_steps}")
    print(f"config      : shipSpeed={config.get('shipSpeed')} episodeSteps={config.get('episodeSteps')} actTimeout={config.get('actTimeout')}")

    if n_steps == 0 or len(players) < 2:
        print("!! not enough data to analyze")
        return

    # ---- per-turn observations + stats (built once, reused everywhere) ----
    obs_list = [get_obs(steps[t]) for t in range(n_steps)]
    series = [(t, per_player(obs_list[t], players)) for t in range(n_steps)]

    is_4p = len(players) >= 3
    team_names = info.get("TeamNames")

    # me selection: --me > --me-name > (4P) mojibake JP name (= the user) > first-eliminated > loser
    me = None
    me_reason = ""
    if args.me is not None:
        me, me_reason = int(args.me), "--me"
    elif args.me_name:
        me = find_seat_by_name(team_names, args.me_name)
        me_reason = f"--me-name '{args.me_name}'"
    if me is None and is_4p:
        g = find_cjk_seat(team_names)
        if g is not None:
            me, me_reason = g, "CJK TeamName (auto = the user 藤田　佑)"
    if me is None and is_4p:
        elim = {}
        for p in players:
            elim[p] = next((t for t, s in series if s["planets"][p] == 0 and s["total_ships"][p] == 0), None)
        if any(v is not None for v in elim.values()):
            me, me_reason = min((p for p in players if elim[p] is not None), key=lambda p: elim[p]), "first-eliminated"
    if me is None:
        me, me_reason = min(players, key=lambda p: (rwd[p] if rwd[p] is not None else 0.0)), "lowest reward"
    if me not in players:
        me, me_reason = min(players, key=lambda p: (rwd[p] if rwd[p] is not None else 0.0)), "fallback lowest reward"
    me_name = team_names[me] if team_names and me < len(team_names) else None
    print(f"selected me : player {me}  name={me_name!r}  (via {me_reason})")

    if is_4p:
        report_4p(series, obs_list, players, rwd, me, n_steps, args.detail_window)
        print("NOTE: classification tags are heuristic; numbers above are raw from JSON.")
        return

    # ===================== 2P path (unchanged) =====================
    enemies = [p for p in players if p != me]
    enemy = enemies[0] if len(enemies) == 1 else None  # 2P -> single enemy
    print(f"me          : player {me}  (reward={rwd.get(me)})")
    print(f"enemy       : player {enemy if enemy is not None else enemies}  (reward={[rwd.get(e) for e in enemies]})")
    print("=" * 70)

    def en_val(stats, key):
        # 2P: enemy value; >2P: max over enemies (strongest opponent)
        return max(stats[key][e] for e in enemies)

    def row(t, stats):
        mp = stats["planets"][me]; ep = en_val(stats, "planets")
        ms = stats["total_ships"][me]; es = en_val(stats, "total_ships")
        mpr = stats["production"][me]; epr = en_val(stats, "production")
        mf = stats["fleets"][me]; ef = en_val(stats, "fleets")
        adv = "ME" if (mp, ms) > (ep, es) else ("EN" if (mp, ms) < (ep, es) else "=")
        return (f"  t={t:>3}  me:{mp:>2}p/{ms:>5.0f}sh/{mpr:>3.0f}pr/{mf:>2}fl"
                f"   en:{ep:>2}p/{es:>5.0f}sh/{epr:>3.0f}pr/{ef:>2}fl   adv:{adv}")

    # ---- detect special turns ----
    def lead(stats, key):
        return stats["planets" if key == "p" else "production"][me] - en_val(stats, "planets" if key == "p" else "production")

    planet_rev = None
    prod_rev = None
    for i in range(1, n_steps):
        _, s0 = series[i - 1]; _, s1 = series[i]
        if planet_rev is None and lead(s0, "p") >= 0 and lead(s1, "p") < 0:
            planet_rev = i
        if prod_rev is None and lead(s0, "pr") >= 0 and lead(s1, "pr") < 0:
            prod_rev = i

    # my ship sharp-drop turn (largest single-turn % drop in my total ships)
    ship_drop_turn = None; ship_drop_amt = 0.0
    for i in range(1, n_steps):
        prev = series[i - 1][1]["total_ships"][me]
        cur = series[i][1]["total_ships"][me]
        d = cur - prev
        if d < ship_drop_amt:
            ship_drop_amt = d; ship_drop_turn = i

    # my defeat turn (first turn my planets == 0)
    defeat_turn = None
    for t, s in series:
        if s["planets"][me] == 0:
            defeat_turn = t
            break

    # ---- key-turn table ----
    key_turns = [tt for tt in (0, 10, 20, 30, 40, 50, 60) if tt < n_steps]
    for extra in (planet_rev, prod_rev, ship_drop_turn, defeat_turn, n_steps - 1):
        if extra is not None and extra not in key_turns:
            key_turns.append(extra)
    key_turns = sorted(set(key_turns))

    print("KEY-TURN TABLE  (en = strongest opponent if >2P)")
    for t in key_turns:
        print(row(t, series[t][1]))
    print("=" * 70)

    # ---- divergence + signals ----
    print("DIVERGENCE / LOSS SIGNALS (evidence from replay):")
    print(f"  planet-lead reversal turn (me ahead/even -> behind): {planet_rev}")
    if planet_rev is not None:
        a = series[planet_rev - 1][1]; b = series[planet_rev][1]
        print(f"    t{planet_rev-1}: me {a['planets'][me]}p/{a['total_ships'][me]:.0f}sh  en {en_val(a,'planets')}p/{en_val(a,'total_ships'):.0f}sh")
        print(f"    t{planet_rev}: me {b['planets'][me]}p/{b['total_ships'][me]:.0f}sh  en {en_val(b,'planets')}p/{en_val(b,'total_ships'):.0f}sh")
    print(f"  production-lead reversal turn: {prod_rev}")
    print(f"  my biggest single-turn ship drop: turn {ship_drop_turn} ({ship_drop_amt:.0f} ships)")
    print(f"  my defeat turn (planets->0): {defeat_turn}")

    # heuristic signal at divergence: were ships ~even but planets diverged? (production race)
    if planet_rev is not None:
        b = series[planet_rev][1]
        ms = b["total_ships"][me]; es = en_val(b, "total_ships")
        ship_gap_pct = 100.0 * (ms - es) / max(1.0, es)
        print(f"  @divergence ship gap: me {ms:.0f} vs en {es:.0f}  ({ship_gap_pct:+.0f}%)  "
              f"-> {'ships ~even, planets diverged = PRODUCTION-RACE/CHURN' if abs(ship_gap_pct) < 12 else 'ship deficit present'}")
    print("=" * 70)

    # ---- deviation detail: launches + owner flips around the divergence ----
    pivot = planet_rev if planet_rev is not None else (prod_rev if prod_rev is not None else ship_drop_turn)
    if pivot is not None:
        w = max(0, int(args.detail_window))
        lo = max(1, pivot - w)
        hi = min(n_steps - 1, pivot + w)
        print(f"DEVIATION DETAIL around divergence turn {pivot}  (turns {lo}..{hi})")
        print("  (FLIP = a planet actually changed owner that turn = solid evidence;")
        print("   LAUNCH target is angle-heuristic, trust FLIP for the real hit)")
        for t in range(lo, hi + 1):
            deviation_detail(obs_list, t)
        print("=" * 70)

    print("NOTE: classification tags are heuristic; numbers above are raw from JSON.")


if __name__ == "__main__":
    main()
