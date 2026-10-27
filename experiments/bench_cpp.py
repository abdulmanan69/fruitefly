"""Python-vs-C++ LIF kernel comparison: correctness parity, then speed.

Feeds both controllers an identical deterministic input stream.
Parity must hold before any speedup claim is made (see README).

Usage:
    python -m experiments.bench_cpp --steps 200000
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flycore.controller import LIFController
from flycore import cpp_bridge


def input_stream(n: int, seed: int = 7):
    rng = random.Random(seed)
    for _ in range(n):
        ranges = (rng.random(), rng.random(), rng.random())
        found = rng.random() < 0.55
        error = rng.uniform(-1.0, 1.0) if found else 0.0
        yield ranges, (found, error, 25 if found else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--out", default="results/bench_cpp.json")
    args = parser.parse_args(argv)

    if not cpp_bridge.is_available():
        print("C++ library not built; nothing to benchmark. "
              "Build instructions: flycore/cpp_bridge.py docstring.")
        return 1

    py = LIFController()
    cpp = cpp_bridge.CppLIFController()

    # 1) Parity on identical inputs.
    mismatches = 0
    first_mismatch = None
    for i, (ranges, vision) in enumerate(input_stream(args.steps)):
        a_py = py.decide(ranges, vision)
        a_cpp = cpp.decide(ranges, vision)
        if a_py != a_cpp:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {"step": i, "python": a_py, "cpp": a_cpp,
                                  "ranges": ranges, "vision": vision}

    # 2) Speed, measured separately per controller on a fresh stream.
    def time_controller(controller):
        controller.reset()
        stream = list(input_stream(args.steps))
        t0 = time.perf_counter()
        for ranges, vision in stream:
            controller.decide(ranges, vision)
        return (time.perf_counter() - t0) / args.steps * 1e9  # ns/step

    py_ns = time_controller(LIFController())
    cpp_ns = time_controller(cpp_bridge.CppLIFController())

    result = {
        "steps": args.steps,
        "parity_ok": mismatches == 0,
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "py_ns_per_step": round(py_ns, 1),
        "cpp_ns_per_step": round(cpp_ns, 1),
        "speedup": round(py_ns / cpp_ns, 2) if cpp_ns else None,
        "note": ("ctypes call overhead is included in the C++ timing; a batched "
                 "API would show a larger kernel-only speedup."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
