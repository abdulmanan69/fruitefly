"""Turn experiment CSVs into figures and a report-ready summary.

Usage:
    python -m experiments.analyze --csv results/local_results.csv
    python -m experiments.analyze --csv results/kaggle/results.csv --outdir results/plots_kaggle
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flycore.runner import summarize

# Fixed identity colors (Paul Tol "bright", CVD-safe blue-yellow pair).
CONTROLLER_COLORS = {
    "lif": "#4477AA",
    "baseline": "#CCBB44",
    "lif_default": "#4477AA",
    "lif_best": "#66CCEE",
    "lif_cpp": "#AA3377",
}
INK = "#333333"
MUTED = "#767676"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key in ("success", "steps", "collisions", "escapes"):
                if key in row and row[key] != "":
                    row[key] = int(float(row[key]))
            for key in ("path_length", "avg_decide_ms", "p95_decide_ms",
                        "avg_vision_ms", "peak_mem_kb"):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def _style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CCCCCC")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)


def _bar(ax, names, values, fmt):
    colors = [CONTROLLER_COLORS.get(n, "#888888") for n in names]
    bars = ax.bar(names, values, color=colors, width=0.55, zorder=3)
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10, color=INK,
                    xytext=(0, 3), textcoords="offset points")


def make_plots(rows: list[dict], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    names = list(summary.keys())
    written = []

    # 1. Goal success rate
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    _bar(ax, names, [summary[n]["success_rate"] * 100 for n in names], "{:.0f}%")
    _style(ax, "Goal success rate", "% of maps solved")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    p = outdir / "success_rate.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # 2. Steps to goal (successful episodes only)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    data = [[r["steps"] for r in rows if r["controller"] == n and r["success"]]
            for n in names]
    boxes = ax.boxplot([d or [0] for d in data], tick_labels=names, patch_artist=True,
                       widths=0.45, medianprops={"color": INK})
    for patch, n in zip(boxes["boxes"], names):
        patch.set_facecolor(CONTROLLER_COLORS.get(n, "#888888"))
        patch.set_alpha(0.85)
    _style(ax, "Steps to reach the goal (successes only)", "simulation steps")
    fig.tight_layout()
    p = outdir / "steps_to_goal.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # 3. Decision latency
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    _bar(ax, names, [summary[n]["avg_decide_ms"] * 1000 for n in names], "{:.1f}")
    _style(ax, "Mean decision latency", "microseconds per decision")
    fig.tight_layout()
    p = outdir / "decision_latency.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # Summary table (markdown)
    md = ["| metric | " + " | ".join(names) + " |",
          "|---" * (len(names) + 1) + "|"]
    metrics = [("episodes", "{}"), ("success_rate", "{:.1%}"), ("avg_collisions", "{:.2f}"),
               ("avg_steps_to_goal", "{}"), ("avg_path_length", "{}"),
               ("avg_decide_ms", "{:.4f} ms"), ("avg_vision_ms", "{:.3f} ms"),
               ("avg_peak_mem_kb", "{:.0f} KB"), ("avg_escapes", "{:.2f}")]
    for key, fmt in metrics:
        cells = []
        for n in names:
            value = summary[n].get(key)
            cells.append("n/a" if value is None else fmt.format(value))
        md.append(f"| {key} | " + " | ".join(cells) + " |")
    p = outdir / "summary.md"
    p.write_text("\n".join(md) + "\n")
    written.append(p)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="results/plots")
    args = parser.parse_args(argv)
    rows = load_rows(Path(args.csv))
    written = make_plots(rows, Path(args.outdir))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
