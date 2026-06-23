from __future__ import annotations

import argparse
import concurrent.futures
import csv
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate


def _parse_agent(value: str) -> tuple[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), path.strip()
    path = value.strip()
    name = Path(path).parent.name or Path(path).stem
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 4P agents across all seats on the same seed set."
    )
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help="Agent as name=path. Repeat to compare multiple agents.",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        dest="opponents",
        required=True,
        help="Opponent agent path. Repeat exactly 3 times, or once to mirror it.",
    )
    parser.add_argument("--seed-list", help="Comma-separated explicit seeds.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", help="Optional output directory.")
    return parser.parse_args()


def _run_task(task: tuple[str, int, str, list[str], int]) -> dict[str, Any]:
    agent_name, seed, agent_path, opponents, seat = task
    result = evaluate.run_game((seed, agent_path, opponents, seat, 4))
    result["agent_name"] = agent_name
    result["agent_path"] = agent_path
    return result


def _summarize(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        games = len(group)
        wins = sum(int(row["won"]) for row in group)
        draws = sum(int(row["draw"]) for row in group)
        losses = sum(int(row["lost"]) for row in group)
        crashes = sum(int(row["crashed"]) for row in group)
        item = {field: value for field, value in zip(key_fields, key)}
        item.update(
            {
                "games": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": wins / games if games else 0.0,
                "avg_diff": statistics.fmean(row["score_diff"] for row in group),
                "avg_place": statistics.fmean(row["placement"] for row in group),
                "avg_survival": statistics.fmean(row["survival_turn"] for row in group),
                "crashes": crashes,
            }
        )
        summary_rows.append(item)
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Agent | Seat | Games | W | D | L | WinRate | AvgDiff | AvgPlace | AvgSurvival |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seat = row.get("player_index", "all")
        lines.append(
            "| {agent_name} | {seat} | {games} | {wins} | {draws} | {losses} | "
            "{win_rate:.1%} | {avg_diff:.1f} | {avg_place:.2f} | {avg_survival:.1f} |".format(
                seat=seat,
                **row,
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    agents = [_parse_agent(value) for value in args.agent]
    opponents = args.opponents
    if len(opponents) not in (1, 3):
        raise ValueError("--opponent must be provided once or exactly three times for 4P.")

    if args.seed_list:
        seeds = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
    else:
        seeds = [int(args.seed_start) + offset for offset in range(int(args.games))]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("research_runs") / f"4p_all_seats_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (agent_name, seed, agent_path, opponents, seat)
        for agent_name, agent_path in agents
        for seed in seeds
        for seat in range(4)
    ]

    start = time.perf_counter()
    if int(args.workers) <= 1:
        evaluate._MAKE = evaluate.load_environment()
        rows = [_run_task(task) for task in tasks]
    else:
        max_workers = min(int(args.workers), len(tasks))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=evaluate.init_worker,
        ) as executor:
            rows = list(executor.map(_run_task, tasks))
    elapsed = time.perf_counter() - start

    field_order = [
        "agent_name",
        "agent_path",
        "seed",
        "player_index",
        "won",
        "draw",
        "lost",
        "score_diff",
        "primary_score",
        "best_opponent_score",
        "placement",
        "game_length",
        "survival_turn",
        "crashed",
        "status",
    ]
    csv_rows = [{field: row.get(field) for field in field_order} for row in rows]
    _write_csv(out_dir / "results.csv", csv_rows)

    by_agent = _summarize(rows, ("agent_name",))
    by_agent_seat = _summarize(rows, ("agent_name", "player_index"))
    _write_csv(out_dir / "summary_by_agent.csv", by_agent)
    _write_csv(out_dir / "summary_by_agent_seat.csv", by_agent_seat)

    summary = [
        "# 4P All-Seat Evaluation",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"Agents: {', '.join(name for name, _ in agents)}",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "## By Agent",
        "",
        _format_summary_table([{**row, "player_index": "all"} for row in by_agent]),
        "",
        "## By Agent And Seat",
        "",
        _format_summary_table(by_agent_seat),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"Rows: {len(rows)}")
    print(f"Results: {out_dir / 'results.csv'}")
    print(f"Summary: {out_dir / 'summary.md'}")
    print("")
    print(_format_summary_table([{**row, "player_index": "all"} for row in by_agent]))


if __name__ == "__main__":
    main()
