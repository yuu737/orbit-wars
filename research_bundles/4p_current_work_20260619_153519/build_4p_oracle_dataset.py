from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from dump_initial_boards import compute_features, suppress_fds, suppress_output
from analyze_4p_losses import board_bucket


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


DEFAULT_BLOCKS = [2000000, 5421622, 56000000, 9874600000, 12000000]
DEFAULT_CANDIDATES = [
    ("sample7", "sample7\\main.py"),
    ("sample8", "sample8\\main.py"),
    ("hairate5", "bots\\hairate5.py"),
    ("sample32", "sample32_singlefile_s7_s8_selector\\main.py"),
    ("sample34", "sample34_singlefile_s7_s8_selector_tuned\\main.py"),
]
DEFAULT_POOL_OPPONENTS = ["sample7\\main.py", "sample8\\main.py", "bots\\hairate5.py"]


def default_out_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("research_runs") / f"4p_oracle_dataset_{stamp}"


def parse_blocks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def expand_block_seeds(blocks: list[int], games: int) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for block in blocks:
        for offset in range(int(games)):
            seed = int(block) + offset
            if seed in seen:
                continue
            seen.add(seed)
            seeds.append(seed)
    return seeds


def parse_candidates(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_CANDIDATES
    result: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).parent.name if Path(path).name == "main.py" else Path(path).stem
        result.append((label.strip(), path.strip()))
    return result


def result_label(row: dict[str, Any]) -> str:
    if int(row.get("diff", 0)) == 0:
        return "draw"
    if int(row.get("place", 99)) == 1 and int(row.get("diff", -1)) > 0:
        return "win"
    return "loss"


def parse_eval_stdout(text: str, candidate: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in PER_SEED_RE.finditer(text):
        row: dict[str, Any] = {"candidate": candidate}
        for key, value in match.groupdict().items():
            row[key] = value if key == "status" else int(value)
        row["result"] = result_label(row)
        rows.append(row)
    if not rows:
        raise ValueError(f"No per-seed rows parsed for candidate={candidate}")
    return rows


def log_path_for(out_dir: Path, candidate_label: str, seed_start: int | str) -> Path:
    return out_dir / f"{candidate_label}_{seed_start}.txt"


def try_parse_log(path: Path, candidate_label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return parse_eval_stdout(text, candidate_label)
    except ValueError:
        return []


def run_eval(
    *,
    python_exe: str,
    candidate_label: str,
    candidate_path: str,
    opponents: list[str],
    seed_start: int,
    games: int,
    workers: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    cmd = [
        python_exe,
        "evaluate.py",
        "--players",
        "4",
        "--agent",
        candidate_path,
        "--games",
        str(games),
        "--workers",
        str(workers),
        "--seed-start",
        str(seed_start),
    ]
    for opponent in opponents:
        cmd.extend(["--opponent", opponent])

    proc = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = log_path_for(out_dir, candidate_label, seed_start)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py failed for {candidate_label} block={seed_start}. See {log_path}")
    return parse_eval_stdout(proc.stdout, candidate_label)


def run_eval_seed_list(
    *,
    python_exe: str,
    candidate_label: str,
    candidate_path: str,
    opponents: list[str],
    seeds: list[int],
    workers: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    seed_list = ",".join(str(seed) for seed in seeds)
    cmd = [
        python_exe,
        "evaluate.py",
        "--players",
        "4",
        "--agent",
        candidate_path,
        "--workers",
        str(workers),
        "--seed-list",
        seed_list,
    ]
    for opponent in opponents:
        cmd.extend(["--opponent", opponent])

    proc = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = log_path_for(out_dir, candidate_label, "combined")
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py failed for {candidate_label} combined seed-list. See {log_path}")
    return parse_eval_stdout(proc.stdout, candidate_label)


def make_initial_features(seeds: list[int], players: int) -> dict[int, dict[str, Any]]:
    with suppress_output(), suppress_fds():
        from kaggle_environments import make

    result: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        with suppress_output(), suppress_fds():
            env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
            env.reset(int(players))
        obs = env.steps[0][0].observation
        planets = [list(row) for row in obs["planets"]]
        result[int(seed)] = compute_features(int(seed), 0, planets)
    return result


def choose_best_candidate(rows: list[dict[str, Any]]) -> str:
    def key(row: dict[str, Any]) -> tuple[int, int, int, int]:
        rank = {"win": 2, "draw": 1, "loss": 0}.get(str(row.get("result")), 0)
        return (
            rank,
            int(row.get("score", 0)),
            int(row.get("diff", 0)),
            int(row.get("survival", 0)),
        )

    return str(max(rows, key=key)["candidate"])


def oracle_mode(seed_rows: list[dict[str, Any]]) -> str:
    winners = {str(row["candidate"]) for row in seed_rows if row.get("result") == "win"}
    if "sample8" in winners and "sample7" not in winners:
        return "s8_burst"
    if "sample7" in winners and "sample8" not in winners:
        return "s7_stable"
    if "sample7" in winners and "sample8" in winners:
        return "ambiguous_s7_s8"
    if "sample32" in winners or "sample34" in winners:
        return "selector_only"
    if "hairate5" in winners:
        return "third_needed_hairate5_style"
    return "third_needed"


def summarize(rows_by_seed: dict[int, list[dict[str, Any]]], feature_rows: dict[int, dict[str, Any]]) -> str:
    lines = ["# 4P Oracle Dataset Summary", ""]
    lines.append(f"Seeds: {len(rows_by_seed)}")
    lines.append("")

    by_candidate: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    by_bucket_mode: dict[tuple[str, str], int] = {}
    for seed, seed_rows in rows_by_seed.items():
        best = choose_best_candidate(seed_rows)
        mode = oracle_mode(seed_rows)
        bucket = board_bucket(feature_rows[seed])
        by_candidate[best] = by_candidate.get(best, 0) + 1
        by_mode[mode] = by_mode.get(mode, 0) + 1
        by_bucket_mode[(bucket, mode)] = by_bucket_mode.get((bucket, mode), 0) + 1

    lines.append("## Best Candidate Count")
    for key, value in sorted(by_candidate.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Oracle Mode Count")
    for key, value in sorted(by_mode.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Bucket x Oracle Mode")
    for (bucket, mode), value in sorted(by_bucket_mode.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {bucket} / {mode}: {value}")
    lines.append("")
    return "\n".join(lines)


def build_rule_suggestions(oracle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create conservative range rules for sample35.

    The generated file is intentionally a starting point, not an automatic
    promotion. We only emit s8 rules for buckets where sample8 is the oracle
    answer at least twice.
    """
    keys = [
        "n25_prod",
        "n45_prod",
        "n45_cheap",
        "n65_high_prod",
        "nearest_enemy_start_dist",
        "chain_score",
    ]
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in oracle_rows:
        if row.get("oracle_mode") != "s8_burst":
            continue
        bucket = str(row.get("board_bucket", ""))
        by_bucket.setdefault(bucket, []).append(row)

    rules: list[dict[str, Any]] = []
    for bucket, rows in sorted(by_bucket.items()):
        if len(rows) < 2:
            continue
        conditions: dict[str, dict[str, float]] = {}
        for key in keys:
            values = [float(row[key]) for row in rows if key in row and row[key] != ""]
            if not values:
                continue
            lo = min(values)
            hi = max(values)
            # Small padding avoids one-decimal boundary misses while staying
            # narrow enough to prevent broad over-selection.
            pad = 0.5 if key not in ("chain_score", "nearest_enemy_start_dist") else 2.0
            conditions[key] = {"min": round(lo - pad, 3), "max": round(hi + pad, 3)}
        rules.append({
            "mode": "s8_burst",
            "bucket": bucket,
            "support": len(rows),
            "conditions": conditions,
        })

    return {
        "version": 1,
        "default_mode": "s7_stable",
        "notes": "Generated by build_4p_oracle_dataset.py. Review before copying into a submission folder.",
        "rules": rules,
    }


def collect_existing_logs(
    *,
    out_dir: Path,
    candidates: list[tuple[str, str]],
    blocks: list[int],
    combine_blocks: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    log_keys: list[int | str] = ["combined"] if combine_blocks else list(blocks)
    for seed_start in log_keys:
        for label, _path in candidates:
            for row in try_parse_log(log_path_for(out_dir, label, seed_start), label):
                key = (str(row["candidate"]), int(row["seed"]), int(row.get("seat", 0)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def write_outputs(
    *,
    out_dir: Path,
    all_result_rows: list[dict[str, Any]],
    candidates: list[tuple[str, str]],
    opponents: list[str],
    blocks: list[int],
    games: int,
    workers: int,
) -> None:
    if not all_result_rows:
        print("No completed logs to collect yet.")
        return

    result_path = out_dir / "oracle_results.csv"
    result_fields: list[str] = []
    for row in all_result_rows:
        for key in row:
            if key not in result_fields:
                result_fields.append(key)
    with result_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(all_result_rows)

    seeds = sorted({int(row["seed"]) for row in all_result_rows})
    feature_rows = make_initial_features(seeds, players=4)
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for row in all_result_rows:
        by_seed[int(row["seed"])].append(row)

    oracle_rows: list[dict[str, Any]] = []
    candidate_labels = [label for label, _path in candidates]
    for seed in seeds:
        seed_rows = by_seed[seed]
        features = dict(feature_rows[seed])
        row: dict[str, Any] = {
            **features,
            "completed_candidates": ",".join(sorted({str(r["candidate"]) for r in seed_rows})),
            "completed_candidate_count": len({str(r["candidate"]) for r in seed_rows}),
            "expected_candidate_count": len(candidate_labels),
            "board_bucket": board_bucket(features),
            "best_candidate": choose_best_candidate(seed_rows),
            "oracle_mode": oracle_mode(seed_rows),
            "winning_candidates": ",".join(sorted(str(r["candidate"]) for r in seed_rows if r["result"] == "win")),
            "is_complete_seed": int(len({str(r["candidate"]) for r in seed_rows}) >= len(candidate_labels)),
        }
        for candidate in candidate_labels:
            match = next((r for r in seed_rows if r["candidate"] == candidate), None)
            if match:
                row[f"{candidate}_result"] = match["result"]
                row[f"{candidate}_score"] = match["score"]
                row[f"{candidate}_diff"] = match["diff"]
                row[f"{candidate}_survival"] = match["survival"]
            else:
                row[f"{candidate}_result"] = "missing"
                row[f"{candidate}_score"] = ""
                row[f"{candidate}_diff"] = ""
                row[f"{candidate}_survival"] = ""
        oracle_rows.append(row)

    feature_path = out_dir / "oracle_features.csv"
    feature_fields: list[str] = []
    for row in oracle_rows:
        for key in row:
            if key not in feature_fields:
                feature_fields.append(key)
    with feature_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=feature_fields)
        writer.writeheader()
        writer.writerows(oracle_rows)

    summary = summarize(by_seed, feature_rows)
    completed_seed_count = sum(1 for row in oracle_rows if int(row["is_complete_seed"]) == 1)
    summary += f"\n## Completion\n- Completed candidate logs: {len(all_result_rows)}\n"
    summary += f"- Seeds with all candidates complete: {completed_seed_count}/{len(oracle_rows)}\n"
    summary_path = out_dir / "oracle_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    rules_path = out_dir / "oracle_rules.json"
    rules_path.write_text(
        json.dumps(build_rule_suggestions(oracle_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "blocks": blocks,
        "games": int(games),
        "workers": int(workers),
        "candidates": candidates,
        "opponents": opponents,
        "partial_ok": True,
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Collected logs: {len(all_result_rows)} rows")
    print(f"Results: {result_path.resolve()}")
    print(f"Features: {feature_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Rule suggestions: {rules_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 4P oracle dataset by evaluating candidate bots on identical seeds.")
    parser.add_argument("--python", default=r"C:\tmp\ow\Scripts\python.exe", help="Python executable for evaluate.py.")
    parser.add_argument("--blocks", default=",".join(str(x) for x in DEFAULT_BLOCKS), help="Comma-separated seed-start blocks.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--candidate", action="append", help="Candidate as label=path. Repeatable.")
    parser.add_argument("--opponent", action="append", help="Pool opponent path. Repeatable.")
    parser.add_argument("--out-dir", help="Output directory.")
    parser.add_argument("--collect-only", action="store_true", help="Do not run evaluations; collect existing logs in --out-dir.")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip candidate/block logs already present in --out-dir. Default: on.")
    parser.add_argument("--rerun", action="store_true", help="Ignore existing logs and rerun evaluations.")
    parser.add_argument("--combine-blocks", action="store_true", help="Run one evaluate.py call per candidate using a combined --seed-list for all blocks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_blocks(args.blocks)
    candidates = parse_candidates(args.candidate)
    opponents = args.opponent if args.opponent else DEFAULT_POOL_OPPONENTS

    interrupted = False
    try:
        if not args.collect_only:
            if args.combine_blocks:
                seeds = expand_block_seeds(blocks, int(args.games))
                for label, path in candidates:
                    existing = try_parse_log(log_path_for(out_dir, label, "combined"), label)
                    if existing and not args.rerun:
                        print(f"SKIP existing {label} combined blocks", flush=True)
                        continue
                    print(f"RUN {label} combined blocks seeds={len(seeds)}", flush=True)
                    run_eval_seed_list(
                        python_exe=args.python,
                        candidate_label=label,
                        candidate_path=path,
                        opponents=opponents,
                        seeds=seeds,
                        workers=int(args.workers),
                        out_dir=out_dir,
                    )
            else:
                for seed_start in blocks:
                    for label, path in candidates:
                        existing = try_parse_log(log_path_for(out_dir, label, seed_start), label)
                        if existing and not args.rerun:
                            print(f"SKIP existing {label} block={seed_start}", flush=True)
                            continue
                        print(f"RUN {label} block={seed_start}", flush=True)
                        run_eval(
                            python_exe=args.python,
                            candidate_label=label,
                            candidate_path=path,
                            opponents=opponents,
                            seed_start=seed_start,
                            games=int(args.games),
                            workers=int(args.workers),
                            out_dir=out_dir,
                        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Collecting completed logs...", flush=True)
    finally:
        rows = collect_existing_logs(
            out_dir=out_dir,
            candidates=candidates,
            blocks=blocks,
            combine_blocks=bool(args.combine_blocks),
        )
        write_outputs(
            out_dir=out_dir,
            all_result_rows=rows,
            candidates=candidates,
            opponents=opponents,
            blocks=blocks,
            games=int(args.games),
            workers=int(args.workers),
        )
        if interrupted:
            print("Partial dataset written after interruption.", flush=True)


if __name__ == "__main__":
    main()
