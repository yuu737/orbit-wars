"""Build a flat Kaggle submission zip from a sample dir.

Includes main.py + the orbit_lite package(s) the agent actually imports (the split
orbit_lite_2p/orbit_lite_4p if present, else plain orbit_lite). Excludes unused
packages, params.json (ignored at eval; dropped so the code-baked CONFIG always
applies), candidate_metadata.json, and __pycache__.

Usage: python tools/build_submit_zip.py <sample_dir> [out.zip]
"""
import os, re, sys, zipfile


def imported_packages(main_path):
    src = open(main_path, encoding="utf-8").read()
    pkgs = set(re.findall(r"(?:from|import)\s+(orbit_lite(?:_2p|_4p)?)\b", src))
    return pkgs or {"orbit_lite"}


def main():
    src = sys.argv[1].rstrip("/\\")
    out = sys.argv[2] if len(sys.argv) > 2 else f"{os.path.basename(src)}_submit_flat.zip"
    keep_pkgs = imported_packages(os.path.join(src, "main.py"))
    all_pkgs = {d for d in os.listdir(src) if d.startswith("orbit_lite") and os.path.isdir(os.path.join(src, d))}
    drop_pkgs = all_pkgs - keep_pkgs
    if os.path.exists(out):
        os.remove(out)
    n = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d != "__pycache__" and d not in drop_pkgs]
            for f in files:
                if f in ("candidate_metadata.json", "params.json") or f.endswith(".pyc"):
                    continue
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, src))
                n += 1
    print(f"built {out}: {n} files | packages kept={sorted(keep_pkgs)} dropped={sorted(drop_pkgs)}")


if __name__ == "__main__":
    main()
