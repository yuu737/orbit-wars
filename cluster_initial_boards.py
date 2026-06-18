import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from dump_initial_boards import compute_features, parse_seed_list, render_seed_svg, suppress_output


_MAKE = None
_WORKER_DEVNULL = None


FEATURE_COLUMNS = [
    "planet_count",
    "neutral_count",
    "nearest_enemy_start_dist",
    "n25_count",
    "n25_prod",
    "n25_ships",
    "n25_high_prod",
    "n25_big",
    "n25_cheap",
    "n45_count",
    "n45_prod",
    "n45_ships",
    "n45_high_prod",
    "n45_big",
    "n45_cheap",
    "n65_count",
    "n65_prod",
    "n65_ships",
    "n65_high_prod",
    "n65_big",
    "n65_cheap",
    "n85_count",
    "n85_prod",
    "n85_ships",
    "n85_high_prod",
    "n85_big",
    "n85_cheap",
    "near1_dist",
    "near1_ships",
    "near1_prod",
    "near2_dist",
    "near2_ships",
    "near2_prod",
    "near3_dist",
    "near3_ships",
    "near3_prod",
    "near4_dist",
    "near4_ships",
    "near4_prod",
    "near5_dist",
    "near5_ships",
    "near5_prod",
    "chain_score",
    "chain_near_ships",
    "chain_outer_ships",
    "chain_outer_prod",
]


def default_out_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("research_runs") / f"initial_board_clusters_{stamp}"


def init_worker() -> None:
    global _MAKE, _WORKER_DEVNULL
    if _WORKER_DEVNULL is None:
        _WORKER_DEVNULL = open(os.devnull, "w", encoding="utf-8")
        os.dup2(_WORKER_DEVNULL.fileno(), 1)
        os.dup2(_WORKER_DEVNULL.fileno(), 2)
    with suppress_output():
        from kaggle_environments import make

    _MAKE = make


def extract_one(seed: int, players: int) -> dict[str, Any]:
    global _MAKE
    if _MAKE is None:
        init_worker()
    with suppress_output():
        env = _MAKE("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(players)
    obs = env.steps[0][0].observation
    planets = [list(row) for row in obs["planets"]]
    return compute_features(seed, 0, planets)


def feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    return (x - mean) / std, mean, std


def kmeans(x: np.ndarray, k: int, *, seed: int, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = x[first]
    min_dist2 = np.sum((x - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(min_dist2.sum())
        if total <= 1e-12:
            centers[c] = x[int(rng.integers(0, n))]
        else:
            idx = int(rng.choice(n, p=min_dist2 / total))
            centers[c] = x[idx]
            min_dist2 = np.minimum(min_dist2, np.sum((x - centers[c]) ** 2, axis=1))

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist2.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = x[mask].mean(axis=0)
            else:
                centers[c] = x[int(rng.integers(0, n))]

    dist2 = ((x - centers[labels]) ** 2).sum(axis=1)
    inertia = float(dist2.sum())
    return labels, centers, inertia


def sample_silhouette(x: np.ndarray, labels: np.ndarray, *, max_points: int = 600, seed: int = 0) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    xs = x[idx]
    ls = labels[idx]
    d = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(idx)):
        same = ls == ls[i]
        same[i] = False
        a = float(d[i, same].mean()) if np.any(same) else 0.0
        b = math.inf
        for c in unique:
            if c == ls[i]:
                continue
            mask = ls == c
            if np.any(mask):
                b = min(b, float(d[i, mask].mean()))
        if math.isfinite(b) and max(a, b) > 1e-12:
            scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else 0.0


def choose_k(results: list[dict[str, Any]]) -> int:
    # Prefer compact but meaningful clusters. Silhouette usually peaks too high on
    # small samples, so use it only as a guide and avoid tiny cluster over-splitting.
    viable = [r for r in results if r["min_cluster_size"] >= max(4, int(0.03 * r["n"]))]
    if not viable:
        viable = results
    return int(max(viable, key=lambda r: (r["silhouette"], -r["k"]))["k"])


def representative_seeds(rows: list[dict[str, Any]], x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> dict[int, int]:
    reps: dict[int, int] = {}
    for c in sorted(set(int(v) for v in labels)):
        idx = np.where(labels == c)[0]
        d = ((x[idx] - centers[c]) ** 2).sum(axis=1)
        rep_idx = int(idx[int(d.argmin())])
        reps[c] = int(rows[rep_idx]["seed"])
    return reps


def cluster_names(rows: list[dict[str, Any]], labels: np.ndarray) -> dict[int, str]:
    names = {}
    overall = {col: np.mean([float(row[col]) for row in rows]) for col in FEATURE_COLUMNS}
    for c in sorted(set(int(v) for v in labels)):
        members = [row for row, label in zip(rows, labels) if int(label) == c]
        avg = {col: np.mean([float(row[col]) for row in members]) for col in FEATURE_COLUMNS}
        tags = []
        if avg["n25_prod"] >= overall["n25_prod"] * 1.2:
            tags.append("near-rich")
        if avg["n45_cheap"] >= overall["n45_cheap"] * 1.25:
            tags.append("cheap-dense")
        if avg["n65_high_prod"] >= overall["n65_high_prod"] * 1.2:
            tags.append("high-prod-field")
        if avg["chain_score"] >= overall["chain_score"] * 1.12:
            tags.append("chain")
        if avg["nearest_enemy_start_dist"] <= overall["nearest_enemy_start_dist"] * 0.9:
            tags.append("close-enemy")
        if avg["n25_prod"] <= overall["n25_prod"] * 0.8 and avg["n65_prod"] >= overall["n65_prod"]:
            tags.append("far-value")
        names[c] = "+".join(tags) if tags else "balanced"
    return names


def load_planets(seed: int, players: int) -> list[list[Any]]:
    with suppress_output():
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(players)
    return [list(row) for row in env.steps[0][0].observation["planets"]]


def write_preview_html(out_dir: Path, reps: dict[int, int], names: dict[int, str], players: int) -> None:
    cards = []
    for cluster_id, seed in sorted(reps.items()):
        svg = render_seed_svg(seed, load_planets(seed, players))
        cards.append(
            '<section class="card">'
            f"<h2>cluster {cluster_id}: {names[cluster_id]}</h2>"
            f"<p>representative seed: <b>{seed}</b></p>"
            f"{svg}"
            "</section>"
        )
    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Orbit Wars Initial Board Clusters</title>
<style>
body {{ margin: 24px; background: #111217; color: #e8e8ee; font-family: Segoe UI, sans-serif; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 18px; }}
.card {{ background: #181a22; padding: 14px; border-radius: 12px; box-shadow: 0 8px 28px #0008; }}
h1 {{ margin-top: 0; }}
h2 {{ margin: 0 0 8px; font-size: 18px; }}
</style>
</head>
<body>
<h1>Orbit Wars Initial Board Clusters</h1>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""
    (out_dir / "cluster_preview.html").write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster many Orbit Wars initial boards by terrain features.")
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-list")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--out-dir", help="Output directory. Default: research_runs/initial_board_clusters_<timestamp>.")
    parser.add_argument("--no-html", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seed_list, args.seed_start, args.games)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=init_worker) as executor:
        rows = list(executor.map(extract_one, seeds, [args.players] * len(seeds), chunksize=8))

    x, mean, std = feature_matrix(rows)
    k_results = []
    labels_by_k = {}
    centers_by_k = {}
    for k in range(max(2, args.k_min), min(args.k_max, len(rows) - 1) + 1):
        labels, centers, inertia = kmeans(x, k, seed=20260619 + k)
        counts = np.bincount(labels, minlength=k)
        sil = sample_silhouette(x, labels, seed=20260619)
        k_results.append(
            {
                "k": k,
                "n": len(rows),
                "inertia": round(inertia, 4),
                "silhouette": round(sil, 4),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
            }
        )
        labels_by_k[k] = labels
        centers_by_k[k] = centers

    best_k = choose_k(k_results)
    labels = labels_by_k[best_k]
    centers = centers_by_k[best_k]
    reps = representative_seeds(rows, x, labels, centers)
    names = cluster_names(rows, labels)

    features_path = out_dir / "board_features.csv"
    with features_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys()) + ["cluster", "cluster_name"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row, label in zip(rows, labels):
            out = dict(row)
            out["cluster"] = int(label)
            out["cluster_name"] = names[int(label)]
            writer.writerow(out)

    k_path = out_dir / "k_summary.csv"
    with k_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(k_results[0].keys()))
        writer.writeheader()
        writer.writerows(k_results)

    cluster_rows = []
    for c in sorted(set(int(v) for v in labels)):
        members = [row for row, label in zip(rows, labels) if int(label) == c]
        cluster_rows.append(
            {
                "cluster": c,
                "name": names[c],
                "count": len(members),
                "pct": round(100.0 * len(members) / len(rows), 2),
                "representative_seed": reps[c],
                "avg_n25_prod": round(np.mean([float(m["n25_prod"]) for m in members]), 3),
                "avg_n45_prod": round(np.mean([float(m["n45_prod"]) for m in members]), 3),
                "avg_n65_high_prod": round(np.mean([float(m["n65_high_prod"]) for m in members]), 3),
                "avg_n45_cheap": round(np.mean([float(m["n45_cheap"]) for m in members]), 3),
                "avg_chain_score": round(np.mean([float(m["chain_score"]) for m in members]), 3),
                "avg_enemy_dist": round(np.mean([float(m["nearest_enemy_start_dist"]) for m in members]), 3),
            }
        )

    clusters_path = out_dir / "clusters.csv"
    with clusters_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cluster_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cluster_rows)

    report_path = out_dir / "report.md"
    lines = [
        "# Initial Board Cluster Report",
        "",
        f"- Seeds: `{len(rows)}`",
        f"- Seed start: `{seeds[0]}`",
        f"- Seed end: `{seeds[-1]}`",
        f"- Chosen k: `{best_k}`",
        "",
        "## k Summary",
        "",
        "| k | silhouette | inertia | min size | max size |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in k_results:
        lines.append(f"| {r['k']} | {r['silhouette']} | {r['inertia']} | {r['min_cluster_size']} | {r['max_cluster_size']} |")
    lines += [
        "",
        "## Clusters",
        "",
        "| cluster | name | count | pct | representative seed | n25 prod | n45 prod | n65 high-prod | n45 cheap | chain | enemy dist |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cluster_rows:
        lines.append(
            f"| {r['cluster']} | {r['name']} | {r['count']} | {r['pct']} | {r['representative_seed']} | "
            f"{r['avg_n25_prod']} | {r['avg_n45_prod']} | {r['avg_n65_high_prod']} | {r['avg_n45_cheap']} | "
            f"{r['avg_chain_score']} | {r['avg_enemy_dist']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.no_html:
        write_preview_html(out_dir, reps, names, args.players)

    print(f"Seeds: {len(rows)}")
    print(f"Chosen k: {best_k}")
    print(f"Report: {report_path.resolve()}")
    print(f"Clusters: {clusters_path.resolve()}")
    print(f"Features: {features_path.resolve()}")
    print(f"k summary: {k_path.resolve()}")
    if not args.no_html:
        print(f"Preview HTML: {(out_dir / 'cluster_preview.html').resolve()}")
    from dump_initial_boards import silence_stdio_for_shutdown

    silence_stdio_for_shutdown()


if __name__ == "__main__":
    main()
