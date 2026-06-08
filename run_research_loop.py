import argparse
import contextlib
import datetime as dt
import io
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import audit_shots
import evaluate


DEFAULT_OPPONENTS = {
    "hairate": "bots/hairate.py",
    "main": "main.py",
}


@contextlib.contextmanager
def quiet_stdio():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact Orbit Wars research loop: eval suite, shot audit, and report files."
    )
    parser.add_argument("--agent", required=True, help="Agent path to evaluate.")
    parser.add_argument("--baseline", help="Optional baseline agent path for automatic deltas.")
    parser.add_argument("--name", help="Optional run name. Defaults to the agent stem plus timestamp.")
    parser.add_argument("--outdir", type=Path, default=Path("research_runs"))
    parser.add_argument(
        "--suite",
        choices=("gate", "standard", "broad"),
        default="gate",
        help="gate=fast seeds 0-3, standard=0-9, broad=0-19. Default: gate.",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        dest="opponents",
        help="Opponent label=path or known label. Repeatable. Default: hairate and main.",
    )
    parser.add_argument("--players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--both-seats", action="store_true", default=True)
    parser.add_argument("--no-both-seats", action="store_false", dest="both_seats")
    parser.add_argument("--audit", action="store_true", default=True)
    parser.add_argument("--no-audit", action="store_false", dest="audit")
    parser.add_argument("--audit-opponent", default="hairate")
    parser.add_argument("--audit-games", type=int, default=0, help="Default follows suite size, capped at 10.")
    parser.add_argument(
        "--benchmark",
        action="append",
        type=Path,
        default=[],
        help="Benchmark JSON to run in addition to the suite. Repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Print compact JSON summary to stdout.")
    return parser.parse_args()


def suite_seeds(name: str) -> list[int]:
    if name == "gate":
        return list(range(4))
    if name == "standard":
        return list(range(10))
    return list(range(20))


def resolve_opponents(entries: list[str] | None) -> list[tuple[str, str]]:
    if not entries:
        return list(DEFAULT_OPPONENTS.items())

    resolved = []
    for entry in entries:
        if "=" in entry:
            label, path = entry.split("=", 1)
            resolved.append((label.strip(), path.strip()))
            continue
        if entry in DEFAULT_OPPONENTS:
            resolved.append((entry, DEFAULT_OPPONENTS[entry]))
        else:
            resolved.append((Path(entry).stem, entry))
    return resolved


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(results)
    if games == 0:
        raise ValueError("No results to summarize.")
    return {
        "games": games,
        "wins": sum(result["won"] for result in results),
        "draws": sum(result["draw"] for result in results),
        "losses": sum(result["lost"] for result in results),
        "win_rate": sum(result["won"] for result in results) / games,
        "crashes": sum(result["crashed"] for result in results),
        "crash_rate": sum(result["crashed"] for result in results) / games,
        "avg_score_diff": statistics.fmean(result["score_diff"] for result in results),
        "avg_primary_score": statistics.fmean(result["primary_score"] for result in results),
        "avg_best_opponent_score": statistics.fmean(result["best_opponent_score"] for result in results),
        "avg_survival_turn": statistics.fmean(result["survival_turn"] for result in results),
        "avg_game_length": statistics.fmean(result["game_length"] for result in results),
        "avg_placement": statistics.fmean(result["placement"] for result in results),
    }


def run_eval(make, agent: str, opponents: list[str], players: int, seeds: list[int], both_seats: bool) -> dict[str, Any]:
    seats = list(range(players)) if both_seats else [0]
    results = []
    for seat in seats:
        for seed in seeds:
            results.append(evaluate.run_game_with_make(make, int(seed), agent, opponents, seat, players))
    summary = summarize_results(results)
    summary["seeds"] = seeds
    summary["seats"] = seats
    summary["players"] = players
    return {"summary": summary, "results": results}


def load_benchmark(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_benchmark(make, agent: str, benchmark_path: Path) -> dict[str, Any]:
    benchmark = load_benchmark(benchmark_path)
    players = int(benchmark["players"])
    both_seats = bool(benchmark.get("both_seats", False))
    data = run_eval(make, agent, list(benchmark["opponents"]), players, [int(s) for s in benchmark["seeds"]], both_seats)
    data["benchmark"] = {
        "name": benchmark.get("name", benchmark_path.stem),
        "path": str(benchmark_path),
        "description": benchmark.get("description", ""),
    }
    return data


def audit_summary(agent: str, opponent: str, seeds: list[int], both_seats: bool) -> dict[str, Any]:
    seats = [0, 1] if both_seats else [0]
    totals = Counter()
    for seat in seats:
        for seed in seeds:
            totals.update(audit_shots.audit_one(agent, opponent, int(seed), seat))

    launches = max(1, int(totals["launches"]))
    summary = {
        "opponent": opponent,
        "seeds": seeds,
        "seats": seats,
        "launches": int(totals["launches"]),
        "still_in_flight": int(totals["still_in_flight"]),
        "unmatched_launch_action": int(totals["unmatched_launch_action"]),
    }
    for key in [
        "sun_loss",
        "out_of_bounds",
        "unknown_loss",
        "planet_hit",
        "target_hit_guess",
        "wrong_planet_hit_guess",
    ]:
        value = int(totals.get(key, 0))
        summary[key] = value
        summary[f"{key}_rate"] = value / launches

    breakdowns = {}
    for prefix in [
        "out_kind_", "out_rot_", "out_ship_", "out_prod_",
        "wrong_kind_", "wrong_rot_", "wrong_ship_", "wrong_prod_",
    ]:
        denom = totals["out_of_bounds"] if prefix.startswith("out_") else totals["wrong_planet_hit_guess"]
        rows = {}
        for key, value in sorted((k, v) for k, v in totals.items() if k.startswith(prefix)):
            rows[key.removeprefix(prefix)] = {"count": int(value), "share": int(value) / max(1, int(denom))}
        breakdowns[prefix.removesuffix("_")] = rows
    summary["breakdowns"] = breakdowns
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Research Run: {report['name']}",
        "",
        f"- Agent: `{report['agent']}`",
        f"- Suite: `{report['suite']}`",
        f"- Started: `{report['started_at']}`",
        f"- Elapsed seconds: `{report['elapsed_seconds']:.2f}`",
        "",
        "## Evaluation",
        "",
        "| label | games | wins | win rate | avg diff | crashes | survival |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["evaluations"]:
        summary = row["summary"]
        lines.append(
            "| {label} | {games} | {wins} | {win_rate:.1%} | {avg_score_diff:.2f} | "
            "{crashes} | {avg_survival_turn:.2f} |".format(label=row["label"], **summary)
        )

    if report.get("baseline_evaluations"):
        lines.extend(
            [
                "",
                "## Baseline Delta",
                "",
                f"- Baseline: `{report['baseline_agent']}`",
                "",
                "| label | win delta | diff delta | candidate diff | baseline diff |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        baseline_by_label = {row["label"]: row for row in report["baseline_evaluations"]}
        for row in report["evaluations"]:
            base = baseline_by_label.get(row["label"])
            if not base:
                continue
            current = row["summary"]
            baseline = base["summary"]
            lines.append(
                "| {label} | {win_delta:+.1%} | {diff_delta:+.2f} | {current_diff:.2f} | {baseline_diff:.2f} |".format(
                    label=row["label"],
                    win_delta=current["win_rate"] - baseline["win_rate"],
                    diff_delta=current["avg_score_diff"] - baseline["avg_score_diff"],
                    current_diff=current["avg_score_diff"],
                    baseline_diff=baseline["avg_score_diff"],
                )
            )

    if report.get("benchmarks"):
        lines.extend(["", "## Benchmarks", ""])
        for row in report["benchmarks"]:
            summary = row["summary"]
            bench = row["benchmark"]
            lines.append(
                "- `{name}`: games={games}, wins={wins}, win={win_rate:.1%}, "
                "diff={avg_score_diff:.2f}, survival={avg_survival_turn:.2f}".format(
                    name=bench["name"], **summary
                )
            )

    audit = report.get("audit")
    if audit:
        lines.extend(
            [
                "",
                "## Shot Audit",
                "",
                f"- opponent: `{audit['opponent']}`",
                f"- launches: `{audit['launches']}`",
                f"- sun_loss: `{audit['sun_loss']}` ({audit['sun_loss_rate']:.1%})",
                f"- out_of_bounds: `{audit['out_of_bounds']}` ({audit['out_of_bounds_rate']:.1%})",
                f"- wrong_planet_hit_guess: `{audit['wrong_planet_hit_guess']}` ({audit['wrong_planet_hit_guess_rate']:.1%})",
                f"- target_hit_guess: `{audit['target_hit_guess']}` ({audit['target_hit_guess_rate']:.1%})",
            ]
        )
        for label in ("out_prod", "wrong_kind", "wrong_rot", "wrong_prod"):
            values = audit["breakdowns"].get(label, {})
            if not values:
                continue
            compact = ", ".join(f"{key}={value['count']} ({value['share']:.0%})" for key, value in values.items())
            lines.append(f"- {label}: {compact}")

    lines.extend(["", "## Files", ""])
    for label, path in report["files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.name or f"{Path(args.agent).stem}-{args.suite}-{started}"
    run_dir = args.outdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = suite_seeds(args.suite)
    opponents = resolve_opponents(args.opponents)
    audit_games = args.audit_games or min(10, len(seeds))
    audit_seeds = seeds[:audit_games]
    audit_opponent = dict(DEFAULT_OPPONENTS).get(args.audit_opponent, args.audit_opponent)

    start = time.perf_counter()
    with quiet_stdio():
        make = evaluate.load_environment()

    evaluations = []
    baseline_evaluations = []
    result_rows = []
    for label, opponent in opponents:
        data = run_eval(make, args.agent, [opponent], args.players, seeds, args.both_seats)
        data["label"] = label
        data["opponents"] = [opponent]
        evaluations.append({"label": label, "opponents": [opponent], "summary": data["summary"]})
        for result in data["results"]:
            result_rows.append({"kind": "evaluation", "label": label, **result})
        if args.baseline:
            baseline_data = run_eval(make, args.baseline, [opponent], args.players, seeds, args.both_seats)
            baseline_data["label"] = label
            baseline_evaluations.append(
                {"label": label, "opponents": [opponent], "summary": baseline_data["summary"]}
            )
            for result in baseline_data["results"]:
                result_rows.append({"kind": "baseline", "label": label, **result})

    benchmark_rows = []
    benchmark_results = []
    for benchmark_path in args.benchmark:
        data = run_benchmark(make, args.agent, benchmark_path)
        benchmark_rows.append({"benchmark": data["benchmark"], "summary": data["summary"]})
        for result in data["results"]:
            benchmark_results.append({"kind": "benchmark", "label": data["benchmark"]["name"], **result})

    audit = None
    if args.audit:
        audit = audit_summary(args.agent, audit_opponent, audit_seeds, args.both_seats)

    elapsed = time.perf_counter() - start
    result_path = run_dir / "results.jsonl"
    benchmark_path = run_dir / "benchmark_results.jsonl"
    report_path = run_dir / "summary.md"
    full_path = run_dir / "run.json"

    write_jsonl(result_path, result_rows)
    if benchmark_results:
        write_jsonl(benchmark_path, benchmark_results)

    report = {
        "name": run_name,
        "agent": args.agent,
        "baseline_agent": args.baseline,
        "suite": args.suite,
        "started_at": started,
        "elapsed_seconds": elapsed,
        "evaluations": evaluations,
        "baseline_evaluations": baseline_evaluations,
        "benchmarks": benchmark_rows,
        "audit": audit,
        "files": {
            "summary": str(report_path),
            "run_json": str(full_path),
            "results_jsonl": str(result_path),
        },
    }
    if benchmark_results:
        report["files"]["benchmark_results_jsonl"] = str(benchmark_path)

    report_path.write_text(render_markdown(report), encoding="utf-8")
    full_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(render_markdown(report))
    if args.json:
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
