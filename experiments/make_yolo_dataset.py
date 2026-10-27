"""Synthesize a YOLO training dataset from the simulator itself.

Frames come from randomized maps and random robot poses; the goal bounding
box is derived from the green color mask, so every label is pixel-perfect
and costs nothing to produce. Output is standard YOLO layout:

    images/{train,val}/*.png
    labels/{train,val}/*.txt      # "0 cx cy w h" normalized, goal class only
    data.yaml

Used locally (python -m experiments.make_yolo_dataset) and inside the
Kaggle GPU training kernel (kaggle_run/build_yolo_kernel.py), which calls
generate() directly.
"""
from __future__ import annotations
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flycore.config import ROBOT_RADIUS, WIDTH, HEIGHT
from flycore.environment import Environment
from flycore.maps import make_map
from flycore.perception import COLOR_TOL, GOAL_RGB, _components


def _random_pose(env: Environment, rng: random.Random, aim_at_goal: float = 0.6):
    x = y = 100.0
    for _ in range(200):
        x = rng.uniform(40.0, WIDTH - 40.0)
        y = rng.uniform(40.0, HEIGHT - 40.0)
        if not env.is_solid(x, y, ROBOT_RADIUS):
            break
    if rng.random() < aim_at_goal:
        angle = math.atan2(env.goal_y - y, env.goal_x - x) + rng.uniform(-0.5, 0.5)
    else:
        angle = rng.uniform(-math.pi, math.pi)
    env.robot_x, env.robot_y, env.robot_angle = x, y, angle


def goal_label(frame: np.ndarray) -> str | None:
    """YOLO label line for the goal blob, or None when the goal is unseen."""
    mask = np.all(np.abs(frame.astype(np.int16) - GOAL_RGB) < COLOR_TOL, axis=2)
    boxes = _components(mask, min_px=5)
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    fh, fw = frame.shape[:2]
    return f"0 {(x + w / 2) / fw:.6f} {(y + h / 2) / fh:.6f} {w / fw:.6f} {h / fh:.6f}"


def generate(out_dir, n_train: int = 1400, n_val: int = 250, seed0: int = 1,
             kinds=("open", "rooms", "house")) -> dict:
    from matplotlib.image import imsave   # lazy: core sim must not need it
    out = Path(out_dir)
    rng = random.Random(1234)
    stats = {"train": 0, "val": 0, "with_goal": 0}
    map_index = 0
    for split, target in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        made = 0
        while made < target:
            map_index += 1
            kind = kinds[map_index % len(kinds)]
            env = Environment(make_map(seed0 + map_index, kind=kind))
            for _ in range(min(6, target - made)):   # several poses per map
                _random_pose(env, rng)
                frame = env.camera_frame()
                label = goal_label(frame)
                name = f"{kind}_{map_index}_{made}"
                imsave(out / "images" / split / f"{name}.png", frame)
                if label:   # frames without the goal act as backgrounds
                    (out / "labels" / split / f"{name}.txt").write_text(label + "\n")
                    stats["with_goal"] += 1
                made += 1
        stats[split] = made
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n  0: goal\n")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/yolo_dataset")
    parser.add_argument("--train", type=int, default=1400)
    parser.add_argument("--val", type=int, default=250)
    args = parser.parse_args(argv)
    stats = generate(args.out, args.train, args.val)
    print(f"dataset written to {args.out}: {stats}")


if __name__ == "__main__":
    main()
