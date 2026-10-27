"""Pseudo-3D first-person view of the 2D world, rendered by raycasting.

GUI-only module (imports pygame); the simulation core never imports it.
Each screen column casts one ray; exact ray/AABB slab intersections find
the nearest wall, which is drawn as a distance-shaded, striped vertical
slice - the classic raycaster look. Targets (checkpoints and the goal)
are drawn as depth-tested glowing billboards, so walls correctly occlude
them.
"""
from __future__ import annotations
import math

import pygame

from .config import WIDTH, HEIGHT
from .environment import COLORS

MAX_DEPTH = 700.0
WALL_SCALE = 46.0          # slice height = h * WALL_SCALE / corrected_depth
SKY_TOP = (13, 18, 28)
SKY_BOTTOM = (40, 50, 66)
FLOOR_NEAR = (56, 66, 80)
FLOOR_FAR = (22, 28, 38)
ALT_OBSTACLE = (178, 88, 96)   # alternates with the base obstacle tint

_background_cache: dict = {}


def _lerp(c0, c1, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))


def _background(size):
    """Sky and floor gradients, cached per panel size."""
    surf = _background_cache.get(size)
    if surf is None:
        w, h = size
        surf = pygame.Surface(size)
        half = h // 2
        for y in range(half):
            pygame.draw.line(surf, _lerp(SKY_TOP, SKY_BOTTOM, y / max(1, half - 1)),
                             (0, y), (w, y))
        for y in range(half, h):
            t = (y - half) / max(1, h - half - 1)
            pygame.draw.line(surf, _lerp(FLOOR_FAR, FLOOR_NEAR, t), (0, y), (w, y))
        _background_cache[size] = surf
    return surf


def _ray_rect(px, py, dx, dy, rect):
    """Slab-method ray/AABB intersection.

    Returns (distance, axis) for the entry face (axis 0 = vertical face,
    axis 1 = horizontal face), or None if the ray misses the rect.
    """
    tmin, tmax, axis = 0.0, MAX_DEPTH, 0
    for i, (p, d, lo, hi) in enumerate((
            (px, dx, rect.x, rect.x + rect.w),
            (py, dy, rect.y, rect.y + rect.h))):
        if abs(d) < 1e-9:
            if p < lo or p > hi:
                return None
        else:
            t1, t2 = (lo - p) / d, (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tmin:
                tmin, axis = t1, i
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    if tmin < 0.5:
        return None
    return tmin, axis


def _border_hit(px, py, dx, dy):
    """Distance to the world boundary (the ray always starts inside it)."""
    best, axis = MAX_DEPTH, 0
    if dx > 1e-9:
        best, axis = (WIDTH - px) / dx, 0
    elif dx < -1e-9:
        best, axis = -px / dx, 0
    if dy > 1e-9:
        t = (HEIGHT - py) / dy
        if t < best:
            best, axis = t, 1
    elif dy < -1e-9:
        t = -py / dy
        if t < best:
            best, axis = t, 1
    return best, axis


_frame_cache = {"key": None, "surf": None}


def render_first_person(env, size=(360, 240), fov_deg=80, columns=None,
                        yaw_offset=0.0) -> pygame.Surface:
    """Render the robot's first-person view as a pygame Surface.

    True camera-height projection: walls are vertical slabs 0..height world
    units tall and the camera sits at `cam` (raised by flight altitude), so
    a climbing bee looks DOWN on low walls. `yaw_offset` turns the camera
    relative to the bee's heading - dragging the mouse in immersive mode
    sweeps a full 360 degrees.

    Column count adapts to the obstacle count and panel width, and the
    finished frame is reused while the pose is unchanged (paused) or on odd
    simulation steps - the view runs at half the simulation rate, which is
    imperceptible.
    """
    key = (size, round(env.robot_x, 1), round(env.robot_y, 1),
           round(env.robot_angle, 3), round(yaw_offset, 3),
           round(env.altitude, 2), env.target_idx, len(env.obstacles))
    cached_key = _frame_cache["key"]
    if cached_key is not None and (
            cached_key == key or (env.steps % 2 == 1 and cached_key[0] == size)):
        return _frame_cache["surf"]
    if columns is None:
        base = 110 if len(env.obstacles) <= 22 else 72
        columns = max(base, min(240, size[0] // 6))
    w, h = size
    surf = _background(size).copy()
    half_fov = math.radians(fov_deg) / 2
    px, py = env.robot_x, env.robot_y
    heading = env.robot_angle + yaw_offset
    col_w = w / columns
    depth = [MAX_DEPTH] * columns

    horizon = h // 2
    proj = h * WALL_SCALE                 # projection constant
    cam = 0.5 + 0.55 * (env.altitude if env.flight else 0.0)

    for i in range(columns):
        rel = (i / (columns - 1) - 0.5) * 2 * half_fov
        ang = heading + rel
        dx, dy = math.cos(ang), math.sin(ang)

        d, axis = _border_hit(px, py, dx, dy)
        color = COLORS["wall"]
        hgt = 1.0
        for idx, rect in enumerate(env.obstacles):
            hit = _ray_rect(px, py, dx, dy, rect)
            if hit is not None and hit[0] < d:
                d, axis = hit
                color = COLORS["obstacle"] if idx % 2 == 0 else ALT_OBSTACLE
                hgt = env.obstacle_heights[idx]
        depth[i] = d

        corrected = max(6.0, d * math.cos(rel))   # fish-eye correction
        bottom = min(2 * h, horizon + int(proj * cam / corrected))
        top = max(-h, horizon - int(proj * (hgt - cam) / corrected))
        if bottom <= top:
            continue
        # Distance shading, darker horizontal faces, and a coarse stripe
        # texture keyed to the hit point so walls do not look flat.
        hit_x, hit_y = px + dx * d, py + dy * d
        shade = max(0.14, min(1.0, 1.12 - d / MAX_DEPTH * 1.2))
        if axis == 1:
            shade *= 0.78
        if int((hit_x + hit_y) / 24) % 2:
            shade *= 0.88
        c = tuple(min(255, int(ch * shade)) for ch in color)
        pygame.draw.rect(surf, c, (int(i * col_w), top,
                                   max(1, int(col_w) + 1), bottom - top))

    # Depth-tested target billboards: the active target glows green and
    # pulses; remaining future targets appear as faint gray markers.
    for t_idx in range(env.target_idx, len(env.targets)):
        tx, ty = env.targets[t_idx]
        ox, oy = tx - px, ty - py
        dist = math.hypot(ox, oy)
        bearing = (math.atan2(oy, ox) - heading + math.pi) % (2 * math.pi) - math.pi
        if dist < 24 or dist > MAX_DEPTH or abs(bearing) > half_fov:
            continue
        col = max(0, min(columns - 1, int((bearing / half_fov * 0.5 + 0.5) * (columns - 1))))
        if dist >= depth[col]:
            continue   # occluded by a wall
        cx = int((bearing / half_fov * 0.5 + 0.5) * w)
        corrected = max(10.0, dist * math.cos(bearing))
        active = t_idx == env.target_idx
        pulse = 1.0 + (0.14 * math.sin(env.steps * 0.18) if active else 0.0)
        radius = max(3, int(h * 15.0 / corrected * pulse))
        cy = horizon + max(2, int(proj * (cam - 0.10) / corrected * 0.5))
        color = COLORS["goal"] if active else (110, 126, 118)
        glow = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        for rr, alpha in ((radius * 2, 36), (int(radius * 1.45), 70), (radius, 230)):
            pygame.draw.circle(glow, (*color, alpha), (radius * 2, radius * 2), rr)
        surf.blit(glow, (cx - radius * 2, cy - radius * 2))
    _frame_cache["key"] = key
    _frame_cache["surf"] = surf
    return surf
