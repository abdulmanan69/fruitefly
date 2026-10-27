"""Hand-built sprites for the GUI: an animated top-down bee and flowers.

GUI-only module (imports pygame). The bee is drawn procedurally (vector
shapes, two wing frames), so the project needs no binary assets and stays
license-clean. To use a custom image instead, drop a transparent
`assets/bee.png` (top view, facing RIGHT) into the project root and it is
picked up automatically.
"""
from __future__ import annotations
import math
from pathlib import Path

import pygame

_ASSET_BEE = Path(__file__).resolve().parents[1] / "assets" / "bee.png"

GOLD = (238, 182, 55)
DARK = (58, 42, 20)
FUZZ = (168, 122, 52)
WING = (215, 228, 246, 120)
WING_HL = (240, 246, 252, 150)


def _procedural_frame(wing_phase: float) -> pygame.Surface:
    """One top-down bee frame, facing +x. wing_phase in [0, 1]."""
    w, h = 64, 52
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = 30, 26

    # Legs: three per side, slightly trailing.
    for side in (-1, 1):
        for lx in (-8, -1, 6):
            pygame.draw.line(s, (50, 38, 24, 230),
                             (cx + lx, cy + side * 6),
                             (cx + lx - 4, cy + side * 13), 2)

    # Wings: translucent, swept back; the flap sweeps them forward.
    sweep = 24 + wing_phase * 26           # degrees away from the body axis
    for side in (-1, 1):
        wing = pygame.Surface((30, 15), pygame.SRCALPHA)
        pygame.draw.ellipse(wing, WING, (0, 0, 30, 15))
        pygame.draw.ellipse(wing, WING_HL, (4, 3, 20, 8))
        rotated = pygame.transform.rotozoom(wing, side * sweep, 1.0)
        rect = rotated.get_rect(center=(cx - 10, cy + side * (9 + 3 * wing_phase)))
        s.blit(rotated, rect)

    # Abdomen: gold with dark bands and a sting tip.
    pygame.draw.ellipse(s, GOLD, (cx - 26, cy - 8, 26, 16))
    for off in (5, 12, 19):
        pygame.draw.ellipse(s, DARK, (cx - 26 + off, cy - 7, 5, 14))
    pygame.draw.circle(s, DARK, (cx - 25, cy), 3)

    # Thorax: fuzzy.
    pygame.draw.circle(s, FUZZ, (cx - 1, cy), 9)
    pygame.draw.circle(s, GOLD, (cx - 1, cy), 9, 2)

    # Head, eyes, antennae.
    pygame.draw.circle(s, (48, 36, 26), (cx + 10, cy), 6)
    for side in (-1, 1):
        pygame.draw.circle(s, (12, 12, 14), (cx + 12, cy + side * 3), 2)
        pygame.draw.line(s, (40, 30, 22), (cx + 13, cy + side * 2),
                         (cx + 22, cy + side * 8), 1)
    return s


_frames_cache = None


def bee_frames():
    """Two wing-flap frames; a dropped-in assets/bee.png overrides them."""
    global _frames_cache
    if _frames_cache is None:
        if _ASSET_BEE.exists():
            img = pygame.image.load(str(_ASSET_BEE)).convert_alpha()
            scale = 52 / max(1, img.get_width())
            img = pygame.transform.rotozoom(img, 0, scale)
            _frames_cache = [img, img]
        else:
            _frames_cache = [_procedural_frame(0.0), _procedural_frame(1.0)]
    return _frames_cache


_rot_cache: dict = {}
_shadow_cache = None


def draw_bee(screen, x, y, angle, step, altitude=0.0):
    """Draw the animated bee centered on (x, y), heading = angle (radians).

    Rotations are cached per (4-degree, scale) bucket: rotozoom every frame
    was a measurable share of the frame budget. Altitude grows the sprite
    and pushes the drop shadow away, selling the climb visually.
    """
    global _shadow_cache
    frames = bee_frames()
    fi = (step // 3) % len(frames)
    deg = int(-math.degrees(angle)) % 360
    deg -= deg % 4
    scale = round((1.0 + 0.45 * min(1.0, altitude)) * 20) / 20
    rotated = _rot_cache.get((fi, deg, scale))
    if rotated is None:
        rotated = pygame.transform.rotozoom(frames[fi], deg, scale)
        _rot_cache[(fi, deg, scale)] = rotated
    if _shadow_cache is None:
        _shadow_cache = pygame.Surface((36, 26), pygame.SRCALPHA)
        pygame.draw.ellipse(_shadow_cache, (0, 0, 0, 70), (0, 0, 36, 26))
    off = 3 + int(14 * altitude)
    screen.blit(_shadow_cache, _shadow_cache.get_rect(center=(int(x) + off, int(y) + off)))
    screen.blit(rotated, rotated.get_rect(center=(int(x), int(y))))


_FLOWER_STYLE = {
    # kind: (petal_color, petal_len, core_r, core_color)
    "checkpoint": ((240, 195, 80), 10, 5, (255, 225, 140)),
    "dormant": ((96, 104, 100), 15, 11, (70, 90, 80)),
    "active": ((236, 222, 240), 18, 14, (45, 230, 105)),
}
_flower_cache: dict = {}


def _flower_surface(kind) -> pygame.Surface:
    """Each flower is rendered once and blitted thereafter."""
    surf = _flower_cache.get(kind)
    if surf is None:
        petal_color, petal_len, core_r, core_color = _FLOWER_STYLE[kind]
        size = 2 * (core_r + int(petal_len * 1.8)) + 4
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size // 2
        for k in range(6):
            a = k * math.pi / 3
            petal = pygame.Surface((petal_len * 2, petal_len), pygame.SRCALPHA)
            pygame.draw.ellipse(petal, (*petal_color, 235), (0, 0, petal_len * 2, petal_len))
            rotated = pygame.transform.rotozoom(petal, -math.degrees(a), 1.0)
            px = c + math.cos(a) * (core_r + petal_len * 0.7)
            py = c + math.sin(a) * (core_r + petal_len * 0.7)
            surf.blit(rotated, rotated.get_rect(center=(int(px), int(py))))
        pygame.draw.circle(surf, core_color, (c, c), core_r)
        if kind == "active":
            pygame.draw.circle(surf, (28, 160, 80), (c, c), core_r, 2)
        _flower_cache[kind] = surf
    return surf


def draw_flower(screen, x, y, kind, step):
    """Targets as flowers the bee forages.

    kind: "active" (green core - the hue vision hunts in the camera frame),
    "dormant" (gray, waits for checkpoints), "checkpoint" (small amber bud).
    """
    x, y = int(x), int(y)
    surf = _flower_surface(kind)
    screen.blit(surf, surf.get_rect(center=(x, y)))
    if kind == "active":
        core_r = _FLOWER_STYLE["active"][2]
        pulse = int(3 + 2.5 * math.sin(step * 0.18))
        pygame.draw.circle(screen, (170, 255, 200), (x, y), core_r + 5 + pulse, 2)
