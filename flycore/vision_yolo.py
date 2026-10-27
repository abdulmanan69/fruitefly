"""YOLO11 vision backend (Ultralytics).

Trained weights come from the Kaggle GPU kernel
(kaggle_run/build_yolo_kernel.py), which synthesizes a labeled dataset from
the simulator itself - camera frames plus goal bounding boxes derived from
the green mask - and fine-tunes yolo11n on it. Place the trained file at
models/yolo11_goal.pt (the Kaggle fetch step does this automatically).

A COCO-pretrained YOLO knows nothing about this synthetic world, so the
raw yolo11n.pt checkpoint is deliberately NOT used as a fallback: without
trained weights the backend reports itself unavailable instead of
pretending to see.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "yolo11_goal.pt"


def is_available() -> bool:
    if not WEIGHTS.exists():
        return False
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


class YoloVision:
    """Same interface as vision.detect_green_goal, backed by YOLO11."""

    name = "yolo11"

    def __init__(self, conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(str(WEIGHTS))
        self.conf = conf

    def reset(self):
        pass

    def __call__(self, frame: np.ndarray):
        # Ultralytics expects BGR numpy arrays; the camera frame is RGB.
        results = self.model.predict(frame[:, :, ::-1], imgsz=320,
                                     conf=self.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return False, 0.0, 0
        best = int(boxes.conf.argmax())
        x0, y0, x1, y1 = boxes.xyxy[best].tolist()
        cx = (x0 + x1) / 2.0
        width = frame.shape[1]
        error = (cx - (width - 1) / 2.0) / max(1.0, width / 2.0)
        return True, float(np.clip(error, -1.0, 1.0)), int((x1 - x0) * (y1 - y0))
