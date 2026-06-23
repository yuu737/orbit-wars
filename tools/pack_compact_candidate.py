from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPACT_DIR = ROOT / "compact_candidates"
BUILD_DIR = ROOT / "submission_builds"


def copy_optional(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def make_zip(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(src_dir))


def build_candidate(candidate: str, *, zip_only: bool = False) -> tuple[Path, Path]:
    main_src = COMPACT_DIR / f"{candidate}.py"
    if not main_src.exists():
        raise FileNotFoundError(f"Missing compact candidate main: {main_src}")
    orbit_src = COMPACT_DIR / "orbit_lite"
    if not orbit_src.exists():
        raise FileNotFoundError(f"Missing shared orbit_lite: {orbit_src}")

    BUILD_DIR.mkdir(exist_ok=True)
    out_dir = BUILD_DIR / candidate
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(main_src, out_dir / "main.py")
    shutil.copytree(orbit_src, out_dir / "orbit_lite", ignore=shutil.ignore_patterns("__pycache__"))
    copy_optional(COMPACT_DIR / f"{candidate}.params.json", out_dir / "params.json")
    copy_optional(COMPACT_DIR / f"{candidate}.oracle_rules.json", out_dir / "oracle_rules.json")
    copy_optional(COMPACT_DIR / f"{candidate}.candidate_metadata.json", out_dir / "candidate_metadata.json")

    zip_path = BUILD_DIR / f"{candidate}.zip"
    make_zip(out_dir, zip_path)
    if zip_only:
        shutil.rmtree(out_dir)
    return out_dir, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Kaggle submission package from compact_candidates.")
    parser.add_argument("candidate", help="Candidate base name, e.g. sample50_4p_halite_pw_adapted")
    parser.add_argument("--zip-only", action="store_true", help="Remove expanded build folder after creating zip.")
    args = parser.parse_args()

    out_dir, zip_path = build_candidate(args.candidate, zip_only=args.zip_only)
    print(f"Build folder: {out_dir}")
    print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
