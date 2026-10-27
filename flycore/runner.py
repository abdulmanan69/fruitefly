"""Headless episode runner: executes one controller on one map and
collects the metrics named in the project brief — goal success,
collision rate, decision latency, and memory usage.

No pygame, no display: safe for CI machines and Kaggle kernels.
"""
from __future__ import annotations
import time
import tracemalloc

from .environment import Environment
from .maps import MapSpec
from .reflex import ReflexWrapper
from .sensors import get_range_readings
from .vision import detect_green_goal

DEFAULT_MAX_STEPS = 2500


def _percentile(sorted_values, q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def run_episode(controller, map_spec: MapSpec, max_steps: int = DEFAULT_MAX_STEPS,
                track_memory: bool = True, use_reflex: bool = True) -> dict:
    """Run one episode to success or step budget; return a metrics row.

    Every controller gets the same escape-saccade reflex layer (use_reflex)
    so stall recovery does not bias the comparison.
    """
    env = Environment(map_spec)
    if use_reflex and not isinstance(controller, ReflexWrapper):
        controller = ReflexWrapper(controller)
    controller.reset()

    decide_times = []
    vision_times = []
    if track_memory:
        tracemalloc.start()

    for _ in range(max_steps):
        camera = env.camera_frame()
        t0 = time.perf_counter()
        vision_result = detect_green_goal(camera)
        t1 = time.perf_counter()
        ranges = get_range_readings(env)
        pose = (env.robot_x, env.robot_y, env.robot_angle)
        t2 = time.perf_counter()
        if isinstance(controller, ReflexWrapper):
            action = controller.decide(ranges, vision_result, pose)
        else:
            action = controller.decide(ranges, vision_result)
        t3 = time.perf_counter()
        vision_times.append((t1 - t0) * 1000.0)
        decide_times.append((t3 - t2) * 1000.0)
        env.move(action)
        if env.reached:
            break

    if track_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_kb = peak / 1024.0
    else:
        peak_kb = 0.0

    decide_sorted = sorted(decide_times)
    return {
        "controller": getattr(controller, "name", type(controller).__name__),
        "seed": map_spec.seed,
        "success": int(env.reached),
        "steps": env.steps,
        "collisions": env.collisions,
        "path_length": round(env.path_length, 1),
        "avg_decide_ms": round(sum(decide_times) / max(1, len(decide_times)), 4),
        "p95_decide_ms": round(_percentile(decide_sorted, 0.95), 4),
        "avg_vision_ms": round(sum(vision_times) / max(1, len(vision_times)), 4),
        "peak_mem_kb": round(peak_kb, 1),
        "escapes": getattr(controller, "escape_count", 0),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-episode rows into per-controller summary statistics."""
    by_controller: dict[str, list[dict]] = {}
    for row in rows:
        by_controller.setdefault(row["controller"], []).append(row)

    summary = {}
    for name, items in sorted(by_controller.items()):
        n = len(items)
        successes = [r for r in items if r["success"]]
        summary[name] = {
            "episodes": n,
            "success_rate": round(len(successes) / n, 4) if n else 0.0,
            "avg_collisions": round(sum(r["collisions"] for r in items) / n, 2),
            "avg_steps_to_goal": round(
                sum(r["steps"] for r in successes) / len(successes), 1) if successes else None,
            "avg_path_length": round(
                sum(r["path_length"] for r in successes) / len(successes), 1) if successes else None,
            "avg_decide_ms": round(sum(r["avg_decide_ms"] for r in items) / n, 4),
            "p95_decide_ms": round(max(r["p95_decide_ms"] for r in items), 4),
            "avg_vision_ms": round(sum(r["avg_vision_ms"] for r in items) / n, 4),
            "avg_peak_mem_kb": round(sum(r["peak_mem_kb"] for r in items) / n, 1),
            "avg_escapes": round(sum(r.get("escapes", 0) for r in items) / n, 2),
        }
    return summary
