from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import html
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dump_initial_boards import (
    compute_features,
    parse_seed_list,
    render_seed_svg,
    silence_stdio_for_shutdown,
    suppress_fds,
    suppress_output,
)


PER_SEED_RE = re.compile(
    r"seed=(?P<seed>-?\d+)\s+"
    r"seat=(?P<seat>-?\d+)\s+"
    r"place=(?P<place>-?\d+)\s+"
    r"reward=(?P<reward>-?\d+)\s+"
    r"score=(?P<score>-?\d+)\s+"
    r"best_opp=(?P<best_opp>-?\d+)\s+"
    r"diff=(?P<diff>-?\d+)\s+"
    r"length=(?P<length>-?\d+)\s+"
    r"survival=(?P<survival>-?\d+)\s+"
    r"status=(?P<status>\S+)"
)


def default_out_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("research_runs") / f"loss_board_analysis_{stamp}"


def parse_eval_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for match in PER_SEED_RE.finditer(text):
        row: dict[str, Any] = {}
        for key, value in match.groupdict().items():
            row[key] = value if key == "status" else int(value)
        row["result"] = result_label(row)
        rows.append(row)
    if not rows:
        raise ValueError(f"No per-seed result rows found in {path}")
    return rows


def result_label(row: dict[str, Any]) -> str:
    # Local Orbit Wars tie-breaks can mark diff=0 as place=1. For strategy
    # analysis we treat those as draws, matching the user's evaluation policy.
    if int(row.get("diff", 0)) == 0:
        return "draw"
    if int(row.get("place", 99)) == 1 and int(row.get("diff", -1)) > 0:
        return "win"
    return "loss"


def load_agent_module(agent_path: str | None):
    if not agent_path:
        return None
    path = Path(agent_path)
    if path.is_dir():
        path = path / "main.py"
    if not path.exists():
        raise FileNotFoundError(path)
    path = path.resolve()

    # Make this import behave like Kaggle loading one self-contained folder.
    for name in list(sys.modules):
        if name == "orbit_lite" or name.startswith("orbit_lite."):
            del sys.modules[name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_loss_probe_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def infer_agent_mode(agent_module, obs: dict[str, Any], player: int) -> str:
    if agent_module is None:
        return ""
    choose = getattr(agent_module, "_choose_4p_mode", None)
    to_tensor = getattr(agent_module, "single_obs_to_tensor", None)
    if choose is None or to_tensor is None:
        return ""
    try:
        obs_copy = dict(obs)
        obs_copy["player"] = int(player)
        obs_tensors = to_tensor(obs_copy, player_id=int(player))
        return str(choose(obs_tensors))
    except Exception as exc:  # keep analysis robust even for experimental bots
        return f"mode_error:{type(exc).__name__}"


def board_bucket(features: dict[str, Any]) -> str:
    enemy = float(features.get("nearest_enemy_start_dist", 999.0))
    mid_prod = float(features.get("n45_prod", 0.0))
    mid_cheap = float(features.get("n45_cheap", 0.0))
    high65 = float(features.get("n65_high_prod", 0.0))
    chain = float(features.get("chain_score", 0.0))
    near_prod = float(features.get("n25_prod", 0.0))
    if enemy <= 55.0 and mid_prod >= 25.0 and chain >= 50.0:
        return "contested-rich"
    if mid_cheap >= 5.0 and mid_prod >= 16.0:
        return "cheap-dense"
    if chain >= 60.0 and high65 >= 5.0:
        return "chain-cluster"
    if high65 >= 8.0 and near_prod <= 14.0:
        return "outer-high"
    if near_prod <= 8.0 and mid_prod <= 16.0:
        return "thin-start"
    return "balanced"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(value)
    except Exception:
        return default


def interesting_reason(row: dict[str, Any], features: dict[str, Any]) -> str:
    result = row.get("result", "")
    if result == "win":
        return "win reference"
    if result == "draw":
        return "diff=0 draw"
    if result == "unknown":
        return f"unscored / {board_bucket(features)}"
    survival = safe_int(row.get("survival"), 999)
    score = safe_int(row.get("score"), 0)
    bucket = board_bucket(features)
    if score == 0 and survival <= 160:
        return f"early collapse / {bucket}"
    if score == 0:
        return f"eliminated / {bucket}"
    if safe_int(row.get("diff"), 0) > -500:
        return f"close loss / {bucket}"
    return f"loss / {bucket}"


def get_initial_board(make_func, seed: int, players: int) -> dict[str, Any]:
    with suppress_output(), suppress_fds():
        env = make_func("orbit_wars", configuration={"seed": int(seed)}, debug=False)
        env.reset(int(players))
    obs = env.steps[0][0].observation
    planets = [list(row) for row in obs["planets"]]
    return {
        "seed": int(seed),
        "players": int(players),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "planets": planets,
        "initial_planets": [list(row) for row in obs.get("initial_planets", planets)],
        "observation": obs,
    }


def write_html(out_dir: Path, rows: list[dict[str, Any]], boards_by_seed: dict[int, dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        seed = int(row["seed"])
        board = boards_by_seed[seed]
        svg = render_seed_svg(seed, board["planets"])
        title = (
            f"seed {seed} | {html.escape(str(row.get('result', '')))} | "
            f"mode {html.escape(str(row.get('agent_mode', '')))} | "
            f"{html.escape(str(row.get('board_bucket', '')))}"
        )
        meta_keys = [
            "place",
            "score",
            "best_opp",
            "diff",
            "length",
            "survival",
            "nearest_enemy_start_dist",
            "n25_prod",
            "n45_prod",
            "n45_cheap",
            "n65_high_prod",
            "chain_score",
            "reason",
        ]
        meta = " / ".join(f"{k}={html.escape(str(row.get(k, '')))}" for k in meta_keys)
        cards.append(
            '<section class="card">'
            f"<h2>{title}</h2>"
            f"<p>{meta}</p>"
            f"{svg}"
            "</section>"
        )

    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Orbit Wars Loss Board Analysis</title>
<style>
body {{ margin: 24px; background: #111217; color: #e8e8ee; font-family: Segoe UI, sans-serif; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }}
.card {{ background: #181a22; padding: 14px; border-radius: 12px; box-shadow: 0 8px 28px #0008; }}
h1 {{ margin-top: 0; }}
h2 {{ margin: 0 0 10px; font-size: 17px; }}
p {{ color: #bcc2d1; font-size: 12px; line-height: 1.45; }}
</style>
</head>
<body>
<h1>Orbit Wars Loss Board Analysis</h1>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Orbit Wars 4P losses using initial-board features and optional agent mode selection."
    )
    parser.add_argument("--eval-log", help="Path to evaluate.py output text. Parses Per-seed results.")
    parser.add_argument("--agent", help="Agent main.py/folder to query for _choose_4p_mode, if available.")
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-list", help="Comma-separated explicit seeds. Used when --eval-log is omitted.")
    parser.add_argument("--include-wins", action="store_true", help="Include wins in HTML/CSV instead of losses+draws only.")
    parser.add_argument("--out-dir", help="Output directory. Default: research_runs/loss_board_analysis_<timestamp>.")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_log:
        result_rows = parse_eval_log(Path(args.eval_log))
    else:
        seeds = parse_seed_list(args.seed_list, args.seed_start, args.games)
        result_rows = [
            {
                "seed": int(seed),
                "seat": 0,
                "place": "",
                "reward": "",
                "score": "",
                "best_opp": "",
                "diff": "",
                "length": "",
                "survival": "",
                "status": "",
                "result": "unknown",
            }
            for seed in seeds
        ]

    selected_rows = result_rows if args.include_wins else [
        row for row in result_rows if row.get("result") != "win"
    ]
    if not selected_rows:
        selected_rows = result_rows

    with suppress_output(), suppress_fds():
        from kaggle_environments import make

    agent_module = load_agent_module(args.agent) if args.agent else None

    boards_by_seed: dict[int, dict[str, Any]] = {}
    analysis_rows: list[dict[str, Any]] = []
    raw_boards: list[dict[str, Any]] = []

    for row in selected_rows:
        seed = int(row["seed"])
        if seed not in boards_by_seed:
            boards_by_seed[seed] = get_initial_board(make, seed, args.players)
            raw_boards.append({
                key: value
                for key, value in boards_by_seed[seed].items()
                if key != "observation"
            })
        board = boards_by_seed[seed]
        player = int(row.get("seat", 0) or 0)
        features = compute_features(seed, player, board["planets"])
        agent_mode = infer_agent_mode(agent_module, board["observation"], player)
        merged = {**row, **features}
        merged["agent_mode"] = agent_mode
        merged["board_bucket"] = board_bucket(features)
        merged["reason"] = interesting_reason(merged, features)
        analysis_rows.append(merged)

    raw_path = out_dir / "loss_initial_boards.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for board in raw_boards:
            fh.write(json.dumps(board, ensure_ascii=False) + "\n")

    csv_path = out_dir / "loss_features.csv"
    if analysis_rows:
        fieldnames: list[str] = []
        for row in analysis_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis_rows)

    summary: dict[str, Any] = {
        "source_eval_log": str(Path(args.eval_log).resolve()) if args.eval_log else None,
        "agent": str(Path(args.agent).resolve()) if args.agent else None,
        "rows": len(analysis_rows),
        "by_result": {},
        "by_mode": {},
        "by_bucket": {},
        "by_reason": {},
    }
    for row in analysis_rows:
        for key, out_key in (
            ("result", "by_result"),
            ("agent_mode", "by_mode"),
            ("board_bucket", "by_bucket"),
            ("reason", "by_reason"),
        ):
            value = str(row.get(key, ""))
            summary[out_key][value] = summary[out_key].get(value, 0) + 1
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_html:
        write_html(out_dir, analysis_rows, boards_by_seed)

    print(f"Rows: {len(analysis_rows)}")
    print(f"Features: {csv_path.resolve()}")
    print(f"Raw boards: {raw_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    if not args.no_html:
        print(f"Preview HTML: {(out_dir / 'index.html').resolve()}")
    silence_stdio_for_shutdown()


if __name__ == "__main__":
    main()
