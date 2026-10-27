"""2D world and robot motion with collision checks.

Pure-Python + numpy core: no pygame dependency, so the same simulation
runs in the GUI, in headless batch experiments, and on Kaggle kernels.
Rendering lives in flycore.render (pygame) instead.

A world has an ordered list of targets: zero or more checkpoints followed
by the final goal. Only the ACTIVE target is rendered green in the camera
frame, and the color-based vision only detects green - so checkpoint maps
form a sequential "visit these in order" puzzle without any change to the
controllers.
"""
from __future__ import annotations
import math

import numpy as np

from .config import WIDTH, HEIGHT, ROBOT_RADIUS, ROBOT_SPEED, TURN_RATE, GOAL_RADIUS
from .maps import MapSpec, default_map

COLORS = {
    "background": (21, 27, 38),
    "grid": (34, 43, 57),
    "wall": (105, 119, 139),
    "robot": (85, 165, 255),
    "robot_heading": (235, 245, 255),
    "goal": (45, 230, 105),
    "text": (235, 240, 248),
    "obstacle": (218, 105, 78),
    "camera": (90, 110, 135),
}

CAMERA_BG = (24, 29, 39)
CAMERA_FLOOR = (32, 40, 51)
CAMERA_CENTERLINE = (120, 135, 155)

TRAIL_LIMIT = 900

# Flight mode (GUI toggle): the bee climbs over obstacles whose height < 1.0
# instead of steering around them. Altitude is run by this autopilot, outside
# the controllers, so the LIF/baseline action set and all benchmark numbers
# are untouched when the mode is off.
FLY_CLIMB = 0.05
FLY_DESCEND = 0.03
FLY_MAX = 1.05

CHASE_SPEED = 1.35        # fleeing goal speed (robot moves at 2.1)
CHASE_FLEE_DIST = 280.0   # the goal runs when the bee gets this close


class Environment:
    def __init__(self, map_spec: MapSpec | None = None):
        self.map_spec = map_spec or default_map()
        self.flight = False   # GUI toggle: fly over low obstacles
        self.chase = False    # GUI toggle: the goal flees from the bee
        self.agile = False    # GUI toggle: speed bursts in open air
        self.reset()

    def reset(self, map_spec: MapSpec | None = None):
        if map_spec is not None:
            self.map_spec = map_spec
        spec = self.map_spec
        self.robot_x, self.robot_y, self.robot_angle = spec.start
        self.targets = [tuple(c) for c in spec.checkpoints] + [tuple(spec.goal)]
        self.target_idx = 0
        self.obstacles = list(spec.obstacles)
        if len(spec.heights) == len(spec.obstacles):
            self.obstacle_heights = list(spec.heights)
        else:
            self.obstacle_heights = [1.0] * len(self.obstacles)
        self.altitude = 0.0
        self._goal_heading = 0.0
        self.trail = []
        self.steps = 0
        self.collisions = 0
        self.path_length = 0.0
        self.reached = False
        self.paused = False
        self.last_action = "START"
        return self

    @property
    def goal_x(self) -> float:
        """x of the ACTIVE target (next checkpoint, or the final goal)."""
        return self.targets[self.target_idx][0]

    @property
    def goal_y(self) -> float:
        return self.targets[self.target_idx][1]

    @property
    def checkpoints_remaining(self) -> int:
        return len(self.targets) - 1 - self.target_idx

    def is_solid(self, x, y, radius=0, altitude=None):
        """altitude=None: sensor/general query (flight mode makes flyable
        obstacles invisible to the rays, so the bee aims straight at them
        and the autopilot climbs). altitude=<value>: movement query - an
        obstacle blocks only if taller than that altitude."""
        if x < radius or y < radius or x >= WIDTH - radius or y >= HEIGHT - radius:
            return True
        # Allocation-free equivalent of rect.inflate(2r, 2r).collidepoint(x, y);
        # this runs thousands of times per frame (rays, movement, camera), so
        # building an inflated Rect object per obstacle per call was the main
        # frame-time cost on obstacle-dense maze maps.
        for rect, hgt in zip(self.obstacles, self.obstacle_heights):
            if (rect.x - radius <= x < rect.x + rect.w + radius
                    and rect.y - radius <= y < rect.y + rect.h + radius):
                if altitude is not None:
                    if hgt > altitude:
                        return True
                elif not (self.flight and hgt < 1.0):
                    return True
        return False

    def _blocking_height(self, x, y, radius):
        """Max height blocking this position (2.0 for the world border)."""
        if x < radius or y < radius or x >= WIDTH - radius or y >= HEIGHT - radius:
            return 2.0
        best = 0.0
        for rect, hgt in zip(self.obstacles, self.obstacle_heights):
            if (rect.x - radius <= x < rect.x + rect.w + radius
                    and rect.y - radius <= y < rect.y + rect.h + radius):
                best = max(best, hgt)
        return best

    def add_obstacle(self, rect, height=1.0):
        self.obstacles.append(rect)
        self.obstacle_heights.append(height)

    def remove_obstacle_at(self, x, y) -> bool:
        for i, rect in enumerate(self.obstacles):
            if rect.collidepoint(x, y):
                self.obstacles.pop(i)
                self.obstacle_heights.pop(i)
                return True
        return False

    def move(self, action):
        if self.reached:
            return
        if action == "LEFT":
            self.robot_angle -= TURN_RATE
        elif action == "RIGHT":
            self.robot_angle += TURN_RATE
        elif action == "BRAKE":
            self.steps += 1
            self.last_action = action
            return
        elif action not in ("FORWARD", "SEEK"):
            self.last_action = action
            return

        # Forward and SEEK are deliberately identical movement primitives;
        # the vision input changes the neural controller's choice of action.
        if action in ("FORWARD", "SEEK"):
            speed = ROBOT_SPEED
            if self.agile:
                # Bee habit: burst in open air, slow near obstacles.
                c, s = math.cos(self.robot_angle), math.sin(self.robot_angle)
                if not self.is_solid(self.robot_x + c * 70,
                                     self.robot_y + s * 70, ROBOT_RADIUS):
                    speed = ROBOT_SPEED * 1.6
                elif self.is_solid(self.robot_x + c * 30,
                                   self.robot_y + s * 30, ROBOT_RADIUS):
                    speed = ROBOT_SPEED * 0.75
            nx = self.robot_x + math.cos(self.robot_angle) * speed
            ny = self.robot_y + math.sin(self.robot_angle) * speed
            if self.flight:
                blocked = self.is_solid(nx, ny, ROBOT_RADIUS, altitude=self.altitude)
            else:
                blocked = self.is_solid(nx, ny, ROBOT_RADIUS)
            if blocked:
                if (self.flight
                        and self._blocking_height(nx, ny, ROBOT_RADIUS) < 1.0):
                    # Autopilot: climb instead of colliding with a low wall.
                    self.altitude = min(FLY_MAX, self.altitude + FLY_CLIMB)
                else:
                    self.collisions += 1
                    # Collision is blocked rather than allowing penetration.
            else:
                self.path_length += math.hypot(nx - self.robot_x, ny - self.robot_y)
                self.robot_x, self.robot_y = nx, ny
                self.trail.append((self.robot_x, self.robot_y))
                if len(self.trail) > TRAIL_LIMIT:
                    del self.trail[:TRAIL_LIMIT // 4]
                if self.flight and self.altitude > 0.0:
                    # Settle back toward the ground once the airspace just
                    # ahead is clear at the lower altitude.
                    ax = self.robot_x + math.cos(self.robot_angle) * 45
                    ay = self.robot_y + math.sin(self.robot_angle) * 45
                    lower = max(0.0, self.altitude - 2 * FLY_DESCEND)
                    if (not self.is_solid(self.robot_x, self.robot_y,
                                          ROBOT_RADIUS, altitude=lower)
                            and not self.is_solid(ax, ay, ROBOT_RADIUS,
                                                  altitude=lower)):
                        self.altitude = max(0.0, self.altitude - FLY_DESCEND)
        self.steps += 1
        self.last_action = action
        if self.chase and self.target_idx == len(self.targets) - 1:
            self._move_goal()
        if math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y) < GOAL_RADIUS:
            if self.target_idx >= len(self.targets) - 1:
                self.reached = True
            else:
                self.target_idx += 1  # checkpoint collected; next target activates

    def _move_goal(self):
        """Chase mode: the final goal wanders, and flees when the bee closes
        in. Slower than the robot, so it is always catchable; deterministic
        (seed + step driven), so chase episodes replay identically."""
        gx, gy = self.targets[self.target_idx]
        dx, dy = gx - self.robot_x, gy - self.robot_y
        dist = math.hypot(dx, dy)
        if dist < CHASE_FLEE_DIST and dist > 1e-6:
            heading = math.atan2(dy, dx)          # straight away from the bee
            self._goal_heading = heading
        else:
            self._goal_heading += 0.05 * math.sin(
                self.steps * 0.013 + self.map_spec.seed)
            heading = self._goal_heading
        nx = gx + math.cos(heading) * CHASE_SPEED
        ny = gy + math.sin(heading) * CHASE_SPEED
        if self.is_solid(nx, ny, 14, altitude=0.0):
            for alt_heading in (heading + 1.4, heading - 1.4, heading + math.pi):
                nx2 = gx + math.cos(alt_heading) * CHASE_SPEED
                ny2 = gy + math.sin(alt_heading) * CHASE_SPEED
                if not self.is_solid(nx2, ny2, 14, altitude=0.0):
                    nx, ny = nx2, ny2
                    self._goal_heading = alt_heading
                    break
            else:
                nx, ny = gx, gy   # cornered: stand still, bee catches it
        self.targets[self.target_idx] = (nx, ny)

    def _line_of_sight(self, tx, ty, min_blocking_height=0.6):
        """Is the straight sight line from the robot to (tx, ty) unblocked?

        Only TALL obstacles block sight: the goal flower stays visible over
        low furniture. Without this check the camera saw the goal THROUGH
        walls, and the controller steered into the wall in between.
        """
        dx, dy = tx - self.robot_x, ty - self.robot_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        steps = int(dist // 8) + 1
        sx, sy = dx / steps, dy / steps
        x, y = self.robot_x, self.robot_y
        for _ in range(steps - 1):
            x += sx
            y += sy
            for rect, hgt in zip(self.obstacles, self.obstacle_heights):
                if (hgt >= min_blocking_height
                        and rect.x <= x < rect.x + rect.w
                        and rect.y <= y < rect.y + rect.h):
                    return False
        return True

    def _visible_objects(self, max_range: float, half_fov: float, w: int):
        """Project the active target and obstacles into camera-image space."""
        objects = []
        gx, gy = self.goal_x - self.robot_x, self.goal_y - self.robot_y
        gd = math.hypot(gx, gy)
        gb = math.atan2(gy, gx) - self.robot_angle
        gb = (gb + math.pi) % (2 * math.pi) - math.pi
        if (gd < max_range and abs(gb) < half_fov
                and self._line_of_sight(self.goal_x, self.goal_y)):
            px = int((gb / half_fov * 0.5 + 0.5) * w)
            size_px = max(5, int(38 * (1 - gd / max_range)))
            objects.append((gd, px, size_px, COLORS["goal"]))
        for rect in self.obstacles:
            ox, oy = rect.centerx - self.robot_x, rect.centery - self.robot_y
            d = math.hypot(ox, oy)
            bearing = math.atan2(oy, ox) - self.robot_angle
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            if d < max_range and abs(bearing) < half_fov:
                px = int((bearing / half_fov * 0.5 + 0.5) * w)
                size_px = max(5, int(48 * (1 - d / max_range)))
                objects.append((d, px, size_px, COLORS["obstacle"]))
        return objects

    def camera_frame(self, size=(240, 150)) -> np.ndarray:
        """Render a forward-facing pseudo-camera frame as an H x W x 3 uint8 array."""
        w, h = size
        frame = np.empty((h, w, 3), dtype=np.uint8)
        frame[:, :] = CAMERA_BG
        frame[h // 2:, :] = CAMERA_FLOOR  # floor band adds visual structure

        max_range = 340.0
        half_fov = math.radians(55)
        objects = self._visible_objects(max_range, half_fov, w)
        # Far objects first so near ones overdraw them, same as painter's order.
        for d, px, s, color in sorted(objects, reverse=True):
            rect_h = max(5, int(s * 0.72))
            rect_w = max(4, s)
            py = h // 2 + int((d / max_range) * (h * 0.23))
            x0, x1 = max(0, px - rect_w // 2), min(w, px - rect_w // 2 + rect_w)
            y0, y1 = max(0, py - rect_h // 2), min(h, py - rect_h // 2 + rect_h)
            if x0 < x1 and y0 < y1:
                frame[y0:y1, x0:x1] = color
        frame[h // 2:, w // 2] = CAMERA_CENTERLINE  # heading reference line
        return frame
