"""Generate the Kaggle GPU kernel that trains YOLO11 on simulator data.

The kernel synthesizes the dataset IN-KERNEL from the embedded simulator
source (no dataset upload), pip-installs ultralytics, fine-tunes yolo11n,
and writes yolo11_goal.pt + yolo_metrics.json to the kernel output.

Requires GPU and internet enabled for the Kaggle account (internet is
needed for `pip install ultralytics` and the yolo11n.pt checkpoint).

Usage:
    python kaggle_run/build_yolo_kernel.py --username <kaggle_username>
    # or push + wait + fetch in one step:
    python kaggle_run/run_on_kaggle.py --username <u> --yolo
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import build_kernel   # reuse the source bundler

KERNEL_DIR = Path(__file__).resolve().parent / "kernel_yolo"
KERNEL_SLUG = "flycore-yolo-train"

TEMPLATE = '''"""FlyCore Edge - YOLO11 goal detector, trained on synthesized sim data."""
import base64, io, json, subprocess, sys, time, zipfile
from pathlib import Path

T0 = time.time()
SRC_B64 = "@@SRC_B64@@"
src = Path("src")
src.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_B64))) as zf:
    zf.extractall(src)
sys.path.insert(0, str(src / "experiments"))
sys.path.insert(0, str(src))

print("installing ultralytics...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"],
               check=True)

from make_yolo_dataset import generate

# Generate OUTSIDE /kaggle/working: thousands of dataset images must not
# become kernel output artifacts (they crowd out the log and the weights).
stats = generate("/tmp/flycore_dataset", n_train=@@N_TRAIN@@, n_val=@@N_VAL@@)
print(f"[{time.time()-T0:.0f}s] dataset: {stats}", flush=True)

import torch
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="/tmp/flycore_dataset/data.yaml", epochs=@@EPOCHS@@, imgsz=320,
            batch=64, device=0 if torch.cuda.is_available() else "cpu",
            project="/tmp/runs", name="goal")
metrics = model.val(data="/tmp/flycore_dataset/data.yaml", imgsz=320)

out = Path("/kaggle/working")
best = Path("/tmp/runs/goal/weights/best.pt")
(out / "yolo11_goal.pt").write_bytes(best.read_bytes())
(out / "yolo_metrics.json").write_text(json.dumps({
    "map50": float(metrics.box.map50),
    "map50_95": float(metrics.box.map),
    "dataset": stats,
    "epochs": @@EPOCHS@@,
}, indent=2))
print(f"[{time.time()-T0:.0f}s] DONE map50={float(metrics.box.map50):.3f}", flush=True)
'''


def build(username: str, n_train: int = 1400, n_val: int = 250,
          epochs: int = 25) -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    code = (TEMPLATE
            .replace("@@SRC_B64@@", build_kernel.bundle_source())
            .replace("@@N_TRAIN@@", str(n_train))
            .replace("@@N_VAL@@", str(n_val))
            .replace("@@EPOCHS@@", str(epochs)))
    code_file = KERNEL_DIR / "yolo_kernel.py"
    code_file.write_text(code, encoding="utf-8")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{KERNEL_SLUG}",
        # Kaggle derives the slug from the title; it MUST slugify to KERNEL_SLUG
        # or the kernel is created under a different ref than we poll.
        "title": "Flycore Yolo Train",
        "code_file": "yolo_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
    }, indent=2))
    return code_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--train", type=int, default=1400)
    parser.add_argument("--val", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args(argv)
    code_file = build(args.username, args.train, args.val, args.epochs)
    print(f"Built YOLO kernel: {code_file}")


if __name__ == "__main__":
    main()
