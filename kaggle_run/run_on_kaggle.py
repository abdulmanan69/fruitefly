"""Push the FlyCore experiment kernel to Kaggle, wait for it, fetch outputs.

Requires the Kaggle CLI to be authenticated (KAGGLE_API_TOKEN env var or
~/.kaggle/access_token). Never commit tokens to the repository.

Usage:
    python kaggle_run/run_on_kaggle.py --username <kaggle_username>
    python kaggle_run/run_on_kaggle.py --username <u> --no-wait   # push only
    python kaggle_run/run_on_kaggle.py --username <u> --fetch-only
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = Path(__file__).resolve().parent / "kernel"
OUT_DIR = PROJECT_ROOT / "results" / "kaggle"


def _find_kaggle() -> str:
    # The CLI can sit next to the interpreter (venv/system install) or in the
    # pip --user scripts folder (default when site-packages is not writable).
    import sysconfig
    folders = [Path(sys.executable).parent]
    for scheme in ("nt_user", "posix_user"):
        try:
            folders.append(Path(sysconfig.get_path("scripts", scheme)))
        except KeyError:
            pass
    for folder in folders:
        for name in ("kaggle.exe", "kaggle"):
            candidate = folder / name
            if candidate.exists():
                return str(candidate)
    found = shutil.which("kaggle")
    if found:
        return found
    raise FileNotFoundError("kaggle CLI not found; pip install kaggle")


def kaggle_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([_find_kaggle(), *args], capture_output=True, text=True)


def kernel_status(ref: str) -> str:
    result = kaggle_cli("kernels", "status", ref)
    text = (result.stdout + result.stderr).lower()
    for status in ("complete", "error", "cancel", "running", "queued"):
        if status in text:
            return status
    return f"unknown ({text.strip()[:200]})"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--main-episodes", type=int, default=150)
    parser.add_argument("--grid-maps", type=int, default=25)
    parser.add_argument("--val-maps", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=7.0)
    parser.add_argument("--no-wait", action="store_true", help="push and exit")
    parser.add_argument("--fetch-only", action="store_true",
                        help="skip push; poll/download an already-running kernel")
    parser.add_argument("--yolo", action="store_true",
                        help="push the YOLO11 GPU training kernel instead "
                             "(needs internet-enabled Kaggle account)")
    args = parser.parse_args(argv)

    if args.yolo:
        import build_yolo_kernel
        slug = build_yolo_kernel.KERNEL_SLUG
        kernel_dir = build_yolo_kernel.KERNEL_DIR
        out_dir = PROJECT_ROOT / "results" / "kaggle_yolo"

        def do_build():
            return build_yolo_kernel.build(args.username)
    else:
        slug = build_kernel.KERNEL_SLUG
        kernel_dir = KERNEL_DIR
        out_dir = OUT_DIR

        def do_build():
            return build_kernel.build(args.username, args.main_episodes,
                                      args.grid_maps, args.val_maps)

    ref = f"{args.username}/{slug}"

    if not args.fetch_only:
        code_file = do_build()
        print(f"Built kernel: {code_file}")
        push = kaggle_cli("kernels", "push", "-p", str(kernel_dir))
        print(push.stdout.strip() or push.stderr.strip())
        if push.returncode != 0 and "successfully" not in push.stdout.lower():
            print("Push failed.", file=sys.stderr)
            return 1
        print(f"Pushed {ref}; Kaggle is now executing it.")

    if args.no_wait:
        print(f"Not waiting. Check later with:\n  kaggle kernels status {ref}")
        return 0

    deadline = time.time() + args.timeout_hours * 3600
    while time.time() < deadline:
        status = kernel_status(ref)
        print(f"[{time.strftime('%H:%M:%S')}] kernel status: {status}", flush=True)
        if status == "complete":
            break
        if status in ("error", "cancel"):
            print("Kernel did not complete successfully; fetching log anyway.")
            break
        time.sleep(args.poll_seconds)
    else:
        print("Timed out waiting for the kernel.", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = kaggle_cli("kernels", "output", ref, "-p", str(out_dir))
    print(fetched.stdout.strip() or fetched.stderr.strip())
    print(f"Outputs saved under {out_dir}")
    if args.yolo:
        weights = out_dir / "yolo11_goal.pt"
        if weights.exists():
            target = PROJECT_ROOT / "models" / "yolo11_goal.pt"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(weights.read_bytes())
            print(f"Installed trained weights -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
