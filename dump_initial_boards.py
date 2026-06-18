import argparse
import contextlib
import csv
import datetime as dt
import html
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@contextlib.contextmanager
def suppress_fds():
    """Silence C-extension / logging output that bypasses Python's sys.stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = open(os.devnull, "w", encoding="utf-8")
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        devnull.close()


def silence_stdio_for_shutdown() -> None:
    """Hide noisy third-party atexit logs after our useful output is printed."""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = open(os.devnull, "w", encoding="utf-8")
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    sys.stdout = devnull
    sys.stderr = devnull


def parse_seed_list(seed_list: str | None, seed_start: int, games: int) -> list[int]:
    if seed_list:
        return [int(part.strip()) for part in seed_list.split(",") if part.strip()]
    return [seed_start + offset for offset in range(games)]


def default_out_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("research_runs") / f"initial_boards_{stamp}"


def dist(a: list[Any], b: list[Any]) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def angle_from_center(p: list[Any]) -> float:
    return math.atan2(float(p[3]) - 50.0, float(p[2]) - 50.0)


def angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def band_stats(start: list[Any], planets: list[list[Any]], max_dist: float) -> dict[str, float]:
    items = [p for p in planets if int(p[1]) == -1 and dist(start, p) <= max_dist]
    return {
        f"n{int(max_dist)}_count": len(items),
        f"n{int(max_dist)}_prod": sum(float(p[6]) for p in items),
        f"n{int(max_dist)}_ships": sum(float(p[5]) for p in items),
        f"n{int(max_dist)}_high_prod": sum(1 for p in items if float(p[6]) >= 3),
        f"n{int(max_dist)}_big": sum(1 for p in items if float(p[5]) >= 40),
        f"n{int(max_dist)}_cheap": sum(1 for p in items if float(p[5]) <= 20),
    }


def compute_chain_features(start: list[Any], planets: list[list[Any]]) -> dict[str, float]:
    neutrals = [p for p in planets if int(p[1]) == -1]
    start_angle = angle_from_center(start)
    near = [
        p
        for p in neutrals
        if 8.0 <= dist(start, p) <= 48.0
        and (float(p[6]) >= 2 or 15 <= float(p[5]) <= 40)
        and angle_diff(start_angle, angle_from_center(p)) <= 1.15
    ]
    best_score = 0.0
    best_near_id = -1
    best_outer_id = -1
    best_near_ships = 0.0
    best_outer_ships = 0.0
    best_outer_prod = 0.0
    for mid in near:
        mid_angle = angle_from_center(mid)
        for outer in neutrals:
            if int(outer[0]) == int(mid[0]):
                continue
            d_mid_outer = dist(mid, outer)
            if not (18.0 <= d_mid_outer <= 80.0):
                continue
            if angle_diff(mid_angle, angle_from_center(outer)) > 0.85:
                continue
            outer_value = float(outer[6]) * 7.0 + float(outer[5]) * 0.08
            mid_value = float(mid[6]) * 5.5 + max(0.0, 42.0 - float(mid[5])) * 0.12
            distance_penalty = dist(start, mid) * 0.12 + d_mid_outer * 0.08
            score = mid_value + outer_value - distance_penalty
            if score > best_score:
                best_score = score
                best_near_id = int(mid[0])
                best_outer_id = int(outer[0])
                best_near_ships = float(mid[5])
                best_outer_ships = float(outer[5])
                best_outer_prod = float(outer[6])

    return {
        "chain_score": round(best_score, 3),
        "chain_near_id": best_near_id,
        "chain_outer_id": best_outer_id,
        "chain_near_ships": best_near_ships,
        "chain_outer_ships": best_outer_ships,
        "chain_outer_prod": best_outer_prod,
    }


def compute_features(seed: int, player: int, planets: list[list[Any]]) -> dict[str, Any]:
    owned = [p for p in planets if int(p[1]) == player]
    if not owned:
        raise ValueError(f"seed={seed} player={player}: no start planet")
    start = max(owned, key=lambda p: float(p[5]))
    neutrals = [p for p in planets if int(p[1]) == -1]
    enemy_starts = [p for p in planets if int(p[1]) not in (-1, player)]
    nearest = sorted(neutrals, key=lambda p: dist(start, p))

    row: dict[str, Any] = {
        "seed": seed,
        "player": player,
        "planet_count": len(planets),
        "neutral_count": len(neutrals),
        "start_id": int(start[0]),
        "start_x": round(float(start[2]), 3),
        "start_y": round(float(start[3]), 3),
        "start_angle": round(angle_from_center(start), 4),
        "nearest_enemy_start_dist": round(min((dist(start, p) for p in enemy_starts), default=999.0), 3),
    }
    for max_dist in (25, 45, 65, 85):
        row.update(band_stats(start, planets, float(max_dist)))

    for i in range(5):
        if i < len(nearest):
            p = nearest[i]
            row[f"near{i+1}_id"] = int(p[0])
            row[f"near{i+1}_dist"] = round(dist(start, p), 3)
            row[f"near{i+1}_ships"] = float(p[5])
            row[f"near{i+1}_prod"] = float(p[6])
        else:
            row[f"near{i+1}_id"] = -1
            row[f"near{i+1}_dist"] = 999.0
            row[f"near{i+1}_ships"] = 0.0
            row[f"near{i+1}_prod"] = 0.0

    row.update(compute_chain_features(start, planets))
    return row


def owner_color(owner: int) -> str:
    colors = {
        -1: "#55585f",
        0: "#1f9ee7",
        1: "#f28c28",
        2: "#10b981",
        3: "#f5e642",
    }
    return colors.get(owner, "#bbbbbb")


def render_seed_svg(seed: int, planets: list[list[Any]], size: int = 420) -> str:
    parts = [
        f'<svg viewBox="0 0 100 100" width="{size}" height="{size}" '
        'style="background:#050506;border:1px solid #333">'
    ]
    parts.append('<circle cx="50" cy="50" r="12" fill="#f3b400" opacity="0.95"/>')
    for p in planets:
        pid, owner, x, y, radius, ships, prod = p[:7]
        owner = int(owner)
        r = max(1.15, min(3.6, float(radius) * 1.35))
        stroke = "#ffffff" if owner >= 0 else "#777"
        text_color = "#ffffff" if owner != 3 else "#111111"
        parts.append(
            f'<circle cx="{float(x):.3f}" cy="{float(y):.3f}" r="{r:.3f}" '
            f'fill="{owner_color(owner)}" stroke="{stroke}" stroke-width="0.35" opacity="0.95"/>'
        )
        parts.append(
            f'<text x="{float(x):.3f}" y="{float(y)+0.7:.3f}" text-anchor="middle" '
            f'font-size="2.8" font-weight="700" fill="{text_color}">{int(ships)}</text>'
        )
        if float(prod) > 0:
            parts.append(
                f'<text x="{float(x):.3f}" y="{float(y)-r-0.8:.3f}" text-anchor="middle" '
                f'font-size="1.9" fill="#9ef5b2">+{int(prod)}</text>'
            )
        parts.append(
            f'<title>seed={seed} id={int(pid)} owner={owner} ships={int(ships)} prod={int(prod)} '
            f'x={float(x):.2f} y={float(y):.2f}</title>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_html(out_dir: Path, boards: list[dict[str, Any]]) -> None:
    cards = []
    for board in boards:
        seed = board["seed"]
        svg = render_seed_svg(seed, board["planets"])
        cards.append(
            '<section class="card">'
            f'<h2>seed {seed}</h2>'
            f'{svg}'
            '<p class="hint">惑星にマウスを乗せると id / owner / ships / prod が見えます。</p>'
            '</section>'
        )
    body = "\n".join(cards)
    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Orbit Wars Initial Boards</title>
<style>
body {{ margin: 24px; background: #111217; color: #e8e8ee; font-family: Segoe UI, sans-serif; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 18px; }}
.card {{ background: #181a22; padding: 14px; border-radius: 12px; box-shadow: 0 8px 28px #0008; }}
h1 {{ margin-top: 0; }}
h2 {{ margin: 0 0 10px; font-size: 18px; }}
.hint {{ color: #aeb3c2; font-size: 12px; }}
</style>
</head>
<body>
<h1>Orbit Wars Initial Boards</h1>
<div class="grid">
{body}
</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Orbit Wars step-0 boards and terrain features by seed.")
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-list", help="Comma-separated explicit seeds. Overrides --seed-start/--games.")
    parser.add_argument("--out-dir", help="Output directory. Default: research_runs/initial_boards_<timestamp>.")
    parser.add_argument("--no-html", action="store_true", help="Skip writing index.html board preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seed_list, args.seed_start, args.games)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    with suppress_output(), suppress_fds():
        from kaggle_environments import make

    boards: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for seed in seeds:
        with suppress_output(), suppress_fds():
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.reset(args.players)
        obs = env.steps[0][0].observation
        planets = [list(row) for row in obs["planets"]]
        board = {
            "seed": seed,
            "players": args.players,
            "angular_velocity": float(obs.get("angular_velocity", 0.0)),
            "planets": planets,
            "initial_planets": [list(row) for row in obs.get("initial_planets", planets)],
        }
        boards.append(board)
        for player in range(args.players):
            feature_rows.append(compute_features(seed, player, planets))

    raw_path = out_dir / "initial_boards.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for board in boards:
            fh.write(json.dumps(board, ensure_ascii=False) + "\n")

    csv_path = out_dir / "initial_features.csv"
    if feature_rows:
        fieldnames = list(feature_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(feature_rows)

    if not args.no_html:
        write_html(out_dir, boards)

    print(f"Seeds: {len(seeds)}")
    print(f"Raw boards: {raw_path.resolve()}")
    print(f"Features: {csv_path.resolve()}")
    if not args.no_html:
        print(f"Preview HTML: {(out_dir / 'index.html').resolve()}")
    silence_stdio_for_shutdown()


if __name__ == "__main__":
    main()
