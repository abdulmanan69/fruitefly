"""Simple color-based visual recognition over a simulated camera frame."""
from __future__ import annotations
import numpy as np

# The simulated goal is rendered bright green. Thresholds tolerate minor display
# differences; this is not intended as a general-purpose real-world vision model.
def detect_green_goal(rgb_frame: np.ndarray):
    """
    Args:
        rgb_frame: H x W x 3 uint8 RGB image.
    Returns:
        (found, normalized_horizontal_error, pixel_count)
        error is -1..1; negative means left of image center.
    """
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
        return False, 0.0, 0
    r = rgb_frame[:, :, 0].astype(np.int16)
    g = rgb_frame[:, :, 1].astype(np.int16)
    b = rgb_frame[:, :, 2].astype(np.int16)
    mask = (g > 150) & (r < 100) & (b < 125) & (g > r * 1.5) & (g > b * 1.25)
    ys, xs = np.where(mask)
    count = int(xs.size)
    if count < 5:
        return False, 0.0, count
    center_x = float(xs.mean())
    width = rgb_frame.shape[1]
    error = (center_x - (width - 1) / 2.0) / max(1.0, width / 2.0)
    return True, float(np.clip(error, -1.0, 1.0)), count
