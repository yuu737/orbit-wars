from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPACT_DIR = ROOT / "compact_candidates"


def sync_candidate(source_dir: Path, name: str | None = None, *, sync_orbit: bool = False) -> None:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    candidate = name or source_dir.name

    main_src = source_dir / "main.py"
    if not main_src.exists():
        raise FileNotFoundError(f"Missing main.py in {source_dir}")

    COMPACT_DIR.mkdir(exist_ok=True)
    shutil.copy2(main_src, COMPACT_DIR / f"{candidate}.py")

    for sidecar in ("params.json", "oracle_rules.json", "candidate_metadata.json"):
        src = source_dir / sidecar
        if src.exists():
            stem = sidecar.removesuffix(".json")
            shutil.copy2(src, COMPACT_DIR / f"{candidate}.{stem}.json")

    if sync_orbit:
        orbit_src = source_dir / "orbit_lite"
        if not orbit_src.exists():
            raise FileNotFoundError(f"Missing orbit_lite in {source_dir}")
        orbit_dst = COMPACT_DIR / "orbit_lite"
        if orbit_dst.exists():
            shutil.rmtree(orbit_dst)
        shutil.copytree(orbit_src, orbit_dst, ignore=shutil.ignore_patterns("__pycache__"))

    print(f"Synced {source_dir.name} -> compact candidate {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a folder-style candidate into compact_candidates.")
    parser.add_argument("source_dir", help="Folder containing main.py and optional sidecars.")
    parser.add_argument("--name", help="Compact candidate name. Defaults to source folder name.")
    parser.add_argument("--sync-orbit", action="store_true", help="Replace compact shared orbit_lite from this source.")
    args = parser.parse_args()

    sync_candidate(Path(args.source_dir), args.name, sync_orbit=args.sync_orbit)


if __name__ == "__main__":
    main()
