"""Multi-object perception: detection + Kalman filtering + Hungarian
data association + SORT-style multi-object tracking.

Pipeline (classic SORT, Bewley et al. 2016, implemented from scratch here):

  camera frame -> detections (color segmentation, or YOLO11 via
  flycore.vision_yolo when trained weights are available)
             -> per-track constant-velocity Kalman filters (predict)
             -> Hungarian algorithm on an IoU cost matrix (associate)
             -> track lifecycle management (persistent IDs)

`TrackerVision` wraps the tracker behind the same interface as
flycore.vision.detect_green_goal, with one upgrade: while the goal track
is briefly occluded, its Kalman PREDICTION keeps reporting a bearing, so
the controller does not instantly forget a target that slid behind a wall.

Pure numpy - no OpenCV/scipy dependency in the simulation core.
"""
from __future__ import annotations
import numpy as np

GOAL_RGB = (45, 230, 105)
OBSTACLE_RGB = (218, 105, 78)
CLS_GOAL, CLS_OBSTACLE = 0, 1
COLOR_TOL = 30


# ---------------------------------------------------------------------------
# Detection: connected components over exact-color masks. The simulated
# camera renders solid-color rectangles, so segmentation is the honest
# equivalent of a detector; YOLO11 can be swapped in via detect_fn.
# ---------------------------------------------------------------------------

def _components(mask: np.ndarray, min_px: int = 6):
    """Bounding boxes (x, y, w, h) of 4-connected components in a bool mask."""
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    boxes = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        miny = maxy = y0
        minx = maxx = x0
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            miny, maxy = min(miny, y), max(maxy, y)
            minx, maxx = min(minx, x), max(maxx, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if count >= min_px:
            boxes.append((minx, miny, maxx - minx + 1, maxy - miny + 1))
    return boxes


def detect_objects(frame: np.ndarray):
    """Detections [(x, y, w, h, cls), ...] from a camera frame."""
    detections = []
    signed = frame.astype(np.int16)
    for cls, color in ((CLS_GOAL, GOAL_RGB), (CLS_OBSTACLE, OBSTACLE_RGB)):
        mask = np.all(np.abs(signed - color) < COLOR_TOL, axis=2)
        for box in _components(mask):
            detections.append((*box, cls))
    return detections


# ---------------------------------------------------------------------------
# Hungarian algorithm (Munkres): minimum-cost assignment, from scratch.
# ---------------------------------------------------------------------------

def hungarian(cost_matrix) -> list:
    """Minimum-cost assignment. Returns [(row, col), ...] for a possibly
    rectangular cost matrix (unassigned rows/cols are simply omitted)."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.size == 0:
        return []
    rows, cols = cost.shape
    n = max(rows, cols)
    c = np.full((n, n), float(cost.max()) + 1.0)
    c[:rows, :cols] = cost

    c = c - c.min(axis=1, keepdims=True)
    c = c - c.min(axis=0, keepdims=True)
    EPS = 1e-9
    starred = np.zeros((n, n), dtype=bool)
    primed = np.zeros((n, n), dtype=bool)
    row_cov = np.zeros(n, dtype=bool)
    col_cov = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if c[i, j] < EPS and not row_cov[i] and not col_cov[j]:
                starred[i, j] = True
                row_cov[i] = True
                col_cov[j] = True
    row_cov[:] = False
    col_cov[:] = False

    while True:
        col_cov[:] = starred.any(axis=0)
        if col_cov.sum() >= n:
            break
        while True:
            # Find an uncovered zero.
            free = (c < EPS) & ~row_cov[:, None] & ~col_cov[None, :]
            idx = np.argwhere(free)
            if idx.size == 0:
                # Adjust the matrix by the smallest uncovered value.
                m = c[~row_cov][:, ~col_cov].min()
                c[~row_cov] -= m
                c[:, col_cov] += m
                continue
            i, j = int(idx[0][0]), int(idx[0][1])
            primed[i, j] = True
            star_cols = np.where(starred[i])[0]
            if star_cols.size:
                row_cov[i] = True
                col_cov[star_cols[0]] = False
            else:
                # Augmenting path of alternating stars and primes.
                path = [(i, j)]
                while True:
                    star_rows = np.where(starred[:, path[-1][1]])[0]
                    if star_rows.size == 0:
                        break
                    r = int(star_rows[0])
                    path.append((r, path[-1][1]))
                    pc = int(np.where(primed[r])[0][0])
                    path.append((r, pc))
                for r, cc in path:
                    starred[r, cc] = not starred[r, cc]
                primed[:] = False
                row_cov[:] = False
                col_cov[:] = False
                break

    pairs = []
    for i in range(rows):
        j = np.where(starred[i])[0]
        if j.size and j[0] < cols:
            pairs.append((i, int(j[0])))
    return pairs


# ---------------------------------------------------------------------------
# Constant-velocity Kalman filter over bounding boxes.
# ---------------------------------------------------------------------------

class KalmanBox:
    """State [cx, cy, w, h, vx, vy]; measurements are (cx, cy, w, h)."""

    def __init__(self, bbox):
        cx, cy, w, h = bbox
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=float)
        self.F = np.eye(6)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.H = np.zeros((4, 6))
        self.H[:4, :4] = np.eye(4)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0])
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5, 0.8, 0.8])
        self.R = np.diag([1.0, 1.0, 2.0, 2.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.bbox

    def update(self, bbox):
        z = np.asarray(bbox, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def bbox(self):
        cx, cy, w, h = self.x[:4]
        return float(cx), float(cy), max(1.0, float(w)), max(1.0, float(h))

    @property
    def velocity(self):
        return float(self.x[4]), float(self.x[5])


# ---------------------------------------------------------------------------
# SORT-style multi-object tracker.
# ---------------------------------------------------------------------------

def _iou(a, b):
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id, bbox, cls):
        self.id = track_id
        self.cls = cls
        self.kf = KalmanBox(bbox)
        self.hits = 1
        self.time_since_update = 0

    @property
    def bbox(self):
        return self.kf.bbox


class Sort:
    """Hungarian-matched, Kalman-predicted multi-object tracker."""

    def __init__(self, max_age=8, min_hits=2, iou_gate=0.05):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_gate = iou_gate
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        """detections: [(x, y, w, h, cls), ...] with x,y = top-left corner.
        Returns the list of live confirmed tracks."""
        boxes = [(x + w / 2, y + h / 2, w, h) for x, y, w, h, _ in detections]
        classes = [d[4] for d in detections]
        for t in self.tracks:
            t.kf.predict()

        pairs = []
        cost = None
        if self.tracks and boxes:
            cost = np.ones((len(self.tracks), len(boxes)))
            for i, t in enumerate(self.tracks):
                for j, b in enumerate(boxes):
                    if t.cls == classes[j]:
                        cost[i, j] = 1.0 - _iou(t.bbox, b)
            pairs = hungarian(cost)

        matched_t, matched_d = set(), set()
        for i, j in pairs:
            if 1.0 - cost[i, j] >= self.iou_gate:
                self.tracks[i].kf.update(boxes[j])
                self.tracks[i].hits += 1
                self.tracks[i].time_since_update = 0
                matched_t.add(i)
                matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.time_since_update += 1
        for j, b in enumerate(boxes):
            if j not in matched_d:
                self.tracks.append(Track(self._next_id, b, classes[j]))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits]


# ---------------------------------------------------------------------------
# Drop-in vision backend for the controllers.
# ---------------------------------------------------------------------------

class TrackerVision:
    """Same interface as vision.detect_green_goal, backed by the tracker.

    Upgrade over the raw color filter: when the goal detection drops out
    for a few frames (sliding behind an obstacle edge), the goal track's
    Kalman prediction still provides a bearing, so `found` stays True for
    up to `bridge_frames` frames of occlusion.
    """

    name = "sort"

    def __init__(self, detect_fn=detect_objects, bridge_frames=6):
        self.detect_fn = detect_fn
        self.bridge_frames = bridge_frames
        self.sort = Sort()
        self.last_tracks: list[Track] = []

    def reset(self):
        self.sort.reset()
        self.last_tracks = []

    def __call__(self, frame: np.ndarray):
        self.last_tracks = self.sort.update(self.detect_fn(frame))
        goals = [t for t in self.last_tracks
                 if t.cls == CLS_GOAL and t.time_since_update <= self.bridge_frames]
        if not goals:
            return False, 0.0, 0
        t = max(goals, key=lambda tr: tr.hits)
        cx, _, w, h = t.bbox
        width = frame.shape[1]
        error = (cx - (width - 1) / 2.0) / max(1.0, width / 2.0)
        return True, float(np.clip(error, -1.0, 1.0)), int(max(1.0, w) * max(1.0, h))
