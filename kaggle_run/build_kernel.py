"""Generate a self-contained Kaggle script kernel for the FlyCore experiments.

The kernel embeds the project source as a base64 zip, so it needs no internet,
no attached datasets, and only Kaggle's preinstalled numpy + g++ toolchain.

Usage:
    python kaggle_run/build_kernel.py --username <kaggle_username>
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = Path(__file__).resolve().parent / "kernel"
KERNEL_SLUG = "flycore-edge-experiments"

INCLUDE = [
    ("flycore", "*.py"),
    ("experiments", "*.py"),
    ("cpp", "*.cpp"),
    ("cpp", "*.hpp"),
]


def bundle_source() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, pattern in INCLUDE:
            for path in sorted((PROJECT_ROOT / folder).glob(pattern)):
                zf.writestr(f"{folder}/{path.name}", path.read_text(encoding="utf-8"))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


KERNEL_TEMPLATE = '''"""FlyCore Edge — full experiment suite (auto-generated, self-contained).

Phases:
  1. LIF vs rule-based baseline across randomized maps
  2. LIF hyperparameter grid search
  3. C++ kernel build + Python/C++ parity + speed benchmark
  4. Best grid config vs default LIF on held-out validation maps
"""
import base64, csv, io, json, subprocess, sys, time, zipfile
from pathlib import Path

T_START = time.time()
BUDGET_S = 6.5 * 3600  # leave headroom under Kaggle's wall-time limit

SRC_B64 = "@@SRC_B64@@"

work = Path(".")
src = work / "src"
src.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_B64))) as zf:
    zf.extractall(src)
sys.path.insert(0, str(src))

from flycore.baseline import BaselineController
from flycore.controller import LIFController
from flycore.maps import generate_map
from flycore.runner import run_episode, summarize

CSV_FIELDS = ["controller", "seed", "success", "steps", "collisions", "path_length",
              "avg_decide_ms", "p95_decide_ms", "avg_vision_ms", "peak_mem_kb", "escapes"]


def write_csv(rows, path, extra_fields=()):
    fields = CSV_FIELDS + list(extra_fields)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def log(msg):
    print(f"[{time.time() - T_START:7.0f}s] {msg}", flush=True)


# ---------------- Phase 1: main comparison sweep ----------------
MAIN_EPISODES = @@MAIN_EPISODES@@
log(f"Phase 1: {MAIN_EPISODES} maps x (lif, baseline)")
rows = []
for i in range(MAIN_EPISODES):
    spec = generate_map(1 + i)
    rows.append(run_episode(LIFController(), spec))
    rows.append(run_episode(BaselineController(), spec))
    if (i + 1) % 25 == 0:
        log(f"  phase 1: map {i + 1}/{MAIN_EPISODES}")
write_csv(rows, "results.csv")
summary = summarize(rows)
Path("results_summary.json").write_text(json.dumps(summary, indent=2))
log("Phase 1 summary: " + json.dumps(summary))

# ---------------- Phase 2: LIF hyperparameter grid search ----------------
GRID_MAPS = @@GRID_MAPS@@
GRID = [
    {"decay": d, "threshold": t, "side_gain": s, "search_gain": g}
    for d in (0.78, 0.82, 0.88)
    for t in (0.70, 0.78)
    for s in (0.60, 0.85)
    for g in (0.002, 0.004)
]
log(f"Phase 2: grid search, {len(GRID)} configs x {GRID_MAPS} maps")
grid_rows = []
grid_results = []
for ci, cfg in enumerate(GRID):
    if time.time() - T_START > BUDGET_S:
        log("  time budget reached; stopping grid early")
        break
    config_rows = []
    for m in range(GRID_MAPS):
        spec = generate_map(5001 + m)
        row = run_episode(LIFController(**cfg), spec, track_memory=False)
        row["config"] = json.dumps(cfg, sort_keys=True)
        config_rows.append(row)
    grid_rows.extend(config_rows)
    n_ok = sum(r["success"] for r in config_rows)
    ok_steps = [r["steps"] for r in config_rows if r["success"]]
    grid_results.append({
        **cfg,
        "success_rate": n_ok / len(config_rows),
        "avg_steps_to_goal": sum(ok_steps) / len(ok_steps) if ok_steps else None,
        "avg_collisions": sum(r["collisions"] for r in config_rows) / len(config_rows),
    })
    log(f"  config {ci + 1}/{len(GRID)} {cfg} -> success {n_ok}/{GRID_MAPS}")
write_csv(grid_rows, "grid.csv", extra_fields=("config",))
grid_results.sort(key=lambda r: (-r["success_rate"],
                                 r["avg_steps_to_goal"] or 1e9))
Path("grid_summary.json").write_text(json.dumps(grid_results, indent=2))
best_cfg = {k: grid_results[0][k] for k in ("decay", "threshold", "side_gain", "search_gain")} \\
    if grid_results else None
log("Phase 2 best config: " + json.dumps(grid_results[0] if grid_results else {}))

# ---------------- Phase 3: C++ kernel build + parity + benchmark ----------------
log("Phase 3: compiling C++ LIF kernel")
try:
    build = subprocess.run(
        ["g++", "-O2", "-shared", "-fPIC",
         "-o", str(src / "cpp" / "libflycore_lif.so"), str(src / "cpp" / "lif_controller.cpp")],
        capture_output=True, text=True)
except FileNotFoundError:
    build = None
if build is None or build.returncode != 0:
    err = "g++ not available" if build is None else build.stderr[:2000]
    log("  g++ failed: " + err)
    Path("bench_cpp.json").write_text(json.dumps({"error": err}))
else:
    import os
    os.environ["FLYCORE_LIF_LIB"] = str(src / "cpp" / "libflycore_lif.so")
    from experiments.bench_cpp import main as bench_main
    bench_main(["--steps", "200000", "--out", "bench_cpp.json"])

# ---------------- Phase 4: best config vs default on held-out maps ----------------
VAL_MAPS = @@VAL_MAPS@@
if best_cfg:
    log(f"Phase 4: default vs best grid config on {VAL_MAPS} held-out maps")
    val_rows = []
    for m in range(VAL_MAPS):
        if time.time() - T_START > BUDGET_S:
            log("  time budget reached; stopping validation early")
            break
        spec = generate_map(9001 + m)
        d = run_episode(LIFController(), spec, track_memory=False)
        d["controller"] = "lif_default"
        b = run_episode(LIFController(**best_cfg), spec, track_memory=False)
        b["controller"] = "lif_best"
        val_rows.extend([d, b])
    val_summary = {"best_config": best_cfg, "summary": summarize(val_rows)}
    write_csv(val_rows, "validation.csv")
    Path("best_vs_default.json").write_text(json.dumps(val_summary, indent=2))
    log("Phase 4 summary: " + json.dumps(val_summary["summary"]))

log("ALL DONE. Outputs: results.csv, results_summary.json, grid.csv, "
    "grid_summary.json, bench_cpp.json, validation.csv, best_vs_default.json")
'''


def build(username: str, main_episodes: int, grid_maps: int, val_maps: int) -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    script = (KERNEL_TEMPLATE
              .replace("@@SRC_B64@@", bundle_source())
              .replace("@@MAIN_EPISODES@@", str(main_episodes))
              .replace("@@GRID_MAPS@@", str(grid_maps))
              .replace("@@VAL_MAPS@@", str(val_maps)))
    code_file = KERNEL_DIR / f"{KERNEL_SLUG}.py"
    code_file.write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "FlyCore Edge Experiments",
        "code_file": f"{KERNEL_SLUG}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return code_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Kaggle username (kernel owner)")
    parser.add_argument("--main-episodes", type=int, default=150)
    parser.add_argument("--grid-maps", type=int, default=25)
    parser.add_argument("--val-maps", type=int, default=100)
    args = parser.parse_args(argv)
    code_file = build(args.username, args.main_episodes, args.grid_maps, args.val_maps)
    print(f"Kernel written: {code_file} ({code_file.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
