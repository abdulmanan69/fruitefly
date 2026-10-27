"""Batch experiment CLI: compare controllers across many randomized maps.

Usage (from the project root):
    python -m experiments.run_experiments --episodes 50
    python -m experiments.run_experiments --episodes 300 --out results/big.csv

Writes one CSV row per (controller, map) episode plus a summary JSON.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flycore.baseline import BaselineController
from flycore.controller import LIFController
from flycore.maps import MAP_KINDS, make_map
from flycore.runner import DEFAULT_MAX_STEPS, run_episode, summarize

CSV_FIELDS = ["controller", "seed", "success", "steps", "collisions", "path_length",
              "avg_decide_ms", "p95_decide_ms", "avg_vision_ms", "peak_mem_kb", "escapes"]

CONTROLLERS = {
    "lif": LIFController,
    "baseline": BaselineController,
}


def run_sweep(episodes: int, controllers: list[str], seed0: int = 1,
              max_steps: int = DEFAULT_MAX_STEPS, kind: str = "open",
              progress=print) -> list[dict]:
    rows = []
    t_start = time.perf_counter()
    for i in range(episodes):
        seed = seed0 + i
        spec = make_map(seed, kind=kind)
        for name in controllers:
            rows.append(run_episode(CONTROLLERS[name](), spec, max_steps=max_steps))
        if progress and (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            progress(f"  map {i + 1}/{episodes} done ({elapsed:.0f}s elapsed)")
    return rows


def write_outputs(rows: list[dict], out_csv: Path) -> dict:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary_path = out_csv.with_name(out_csv.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50,
                        help="number of randomized maps (each run by every controller)")
    parser.add_argument("--controllers", default="lif,baseline",
                        help="comma-separated subset of: " + ",".join(CONTROLLERS))
    parser.add_argument("--seed0", type=int, default=1, help="first map seed")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--kind", choices=MAP_KINDS, default="open",
                        help="map generator: open, maze, or rooms")
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.controllers.split(",") if n.strip()]
    for n in names:
        if n not in CONTROLLERS:
            parser.error(f"unknown controller '{n}'")

    print(f"Running {args.episodes} {args.kind} maps x {names} "
          f"(max {args.max_steps} steps each)...")
    rows = run_sweep(args.episodes, names, seed0=args.seed0,
                     max_steps=args.max_steps, kind=args.kind)
    summary = write_outputs(rows, Path(args.out))

    print(f"\nWrote {len(rows)} episodes -> {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
