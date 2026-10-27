"""pygame rendering for the FlyCore Edge GUI.

Kept separate from the simulation core so headless experiment runs
(local batches, Kaggle kernels) never import pygame. The window shows the
top-down world plus a sidebar with a raycast first-person 3D view, the
robot's camera (the controller's actual vision input), and live LIF
neuron activity.
"""
from __future__ import annotations
import math

import pygame

from .config import WIDTH, HEIGHT, ROBOT_RADIUS, GOAL_RADIUS
from .environment import Environment, COLORS
from . import sprites
from . import view3d

SIDEBAR_W = 380
PANEL_BG = (10, 14, 21)
PANEL_EDGE = (54, 66, 84)
CHECKPOINT_COLOR = (240, 195, 80)
TRAIL_COLOR = (58, 102, 152)
RAY_COLOR = (238, 190, 85)

ACTION_SHORT = ("FWD", "LFT", "RGT", "SEK", "BRK")

HELP_LINES = (
    "R        reset current map",
    "N        new random map (current kind)",
    "M        cycle map kind: open / maze / rooms",
    "C        toggle checkpoint mode (makes a new map)",
    "B        switch controller (LIF / baseline)",
    "F        flight mode: the bee climbs over LOW obstacles",
    "T        chase mode: the goal flees from the bee",
    "O        tracked vision: Kalman + Hungarian multi-object",
    "         tracking with occlusion bridging (SORT)",
    "TAB      immersive mode: fullscreen 3D + minimap",
    "         (drag mouse to look around 360, right-click recenters)",
    "V        toggle the 3D first-person view",
    "S        save a screenshot to results/screenshots/",
    "SPACE    pause          ESC    quit",
    "",
    "mouse drag             draw a new wall",
    "right-click a wall     remove it",
)


def window_size():
    return WIDTH + SIDEBAR_W, HEIGHT


def _panel(screen, small_font, x, y, w, h, title):
    pygame.draw.rect(screen, PANEL_BG, (x, y, w, h), border_radius=6)
    pygame.draw.rect(screen, PANEL_EDGE, (x, y, w, h), 1, border_radius=6)
    screen.blit(small_font.render(title, True, (150, 165, 185)), (x + 8, y + 5))
    return y + 24


def _draw_targets(screen, env):
    # Targets are flowers the bee forages; only the active one has the
    # green core that the camera-frame vision hunts.
    for idx in range(env.target_idx, len(env.targets)):
        tx, ty = env.targets[idx]
        is_active = idx == env.target_idx
        is_final = idx == len(env.targets) - 1
        kind = "active" if is_active else ("dormant" if is_final else "checkpoint")
        sprites.draw_flower(screen, tx, ty, kind, env.steps)


def _draw_world(screen, env, ranges, drag_rect):
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, COLORS["grid"], (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, COLORS["grid"], (0, y), (WIDTH, y))
    for i, (tx, ty) in enumerate(env.trail):
        if i % 3 == 0:
            pygame.draw.circle(screen, TRAIL_COLOR, (int(tx), int(ty)), 2)
    for rect, hgt in zip(env.obstacles, env.obstacle_heights):
        r = pygame.Rect(int(rect.x), int(rect.y), int(rect.w), int(rect.h))
        if env.flight and hgt < 1.0:
            # Low obstacle the bee can fly over: dimmer, dashed-feel outline.
            pygame.draw.rect(screen, (140, 86, 70), r, border_radius=4)
            pygame.draw.rect(screen, (190, 130, 105), r, 1, border_radius=4)
        else:
            pygame.draw.rect(screen, COLORS["obstacle"], r, border_radius=4)
            pygame.draw.rect(screen, (245, 150, 120), r, 1, border_radius=4)
    _draw_targets(screen, env)
    for deg, dist in zip((-42, 0, 42), ranges):
        a = env.robot_angle + math.radians(deg)
        end = (env.robot_x + math.cos(a) * dist * 125,
               env.robot_y + math.sin(a) * dist * 125)
        pygame.draw.line(screen, RAY_COLOR, (int(env.robot_x), int(env.robot_y)),
                         (int(end[0]), int(end[1])), 2)
    bob = 1.5 * math.sin(env.steps * 0.25)   # hover wobble, visual only
    sprites.draw_bee(screen, env.robot_x, env.robot_y + bob, env.robot_angle,
                     env.steps, altitude=env.altitude if env.flight else 0.0)
    if drag_rect:
        pygame.draw.rect(screen, (255, 230, 140), pygame.Rect(drag_rect), 2)


def _draw_hud(screen, env, small_font, vision_result, ranges, controller,
              controller_name, fps=None, tracker_on=False):
    panel = pygame.Rect(12, 12, 370, 198)
    s = pygame.Surface(panel.size, pygame.SRCALPHA)
    s.fill((12, 16, 24, 215))
    screen.blit(s, panel.topleft)
    escapes = getattr(controller, "escape_count", 0)
    checkpoint_note = (f"   Checkpoints left: {env.checkpoints_remaining}"
                       if len(env.targets) > 1 else "")
    fps_note = f"   FPS: {fps:.0f}" if fps else ""
    modes = ("" + (" FLY" if env.flight else "") + (" CHASE" if env.chase else "")
             + (" SORT" if tracker_on else ""))
    mode_note = f"   [{modes.strip()}]" if modes.strip() else ""
    alt_note = f"   Alt: {env.altitude:.2f}" if env.flight else ""
    lines = [
        "FlyCore Edge | brain-inspired navigation",
        f"Controller: {controller_name}   Map: {env.map_spec.kind} "
        f"#{env.map_spec.seed}{mode_note}{alt_note}",
        f"Step: {env.steps}   Collisions: {env.collisions}   Escapes: {escapes}{fps_note}",
        f"Last action: {env.last_action}{checkpoint_note}",
        f"Goal visually detected: {'YES' if vision_result[0] else 'NO'} ({vision_result[2]} px)",
        f"Range L/F/R: {ranges[0]:.2f} / {ranges[1]:.2f} / {ranges[2]:.2f}",
        "N new | M kind | C checkpoints | B controller",
        "F fly | T chase | R reset | V 3D | S shot | H help",
        "mouse: drag = add wall, right-click = remove",
    ]
    for i, line in enumerate(lines):
        color = COLORS["goal"] if i == 4 and vision_result[0] else COLORS["text"]
        screen.blit(small_font.render(line, True, color), (22, 20 + i * 19))


def _draw_neurons(screen, small_font, x, y, w, h, controller):
    inner = getattr(controller, "inner", controller)
    has_lif = hasattr(inner, "v") and hasattr(inner, "threshold")
    title = "LIF NEURON ACTIVITY (membrane potential)" if has_lif else "CONTROLLER STATE"
    top = _panel(screen, small_font, x, y, w, h, title)
    if not has_lif:
        screen.blit(small_font.render("rule-based baseline: no membrane state",
                                      True, COLORS["text"]), (x + 10, top + 10))
        screen.blit(small_font.render("decisions come from fixed-priority rules",
                                      True, COLORS["text"]), (x + 10, top + 30))
        return
    n = len(inner.v)
    bar_w = 48
    gap = (w - 20 - n * bar_w) / max(1, n - 1)
    area_h = h - (top - y) - 26
    base_y = y + h - 22
    for i, v in enumerate(inner.v):
        ratio = min(1.25, v / max(1e-6, inner.threshold))
        bh = int(area_h * ratio / 1.25)
        bx = int(x + 10 + i * (bar_w + gap))
        color = (90, 200, 130) if ratio >= 1.0 else (86, 140, 205)
        pygame.draw.rect(screen, (28, 36, 48), (bx, base_y - area_h, bar_w, area_h),
                         border_radius=3)
        if bh > 0:
            pygame.draw.rect(screen, color, (bx, base_y - bh, bar_w, bh), border_radius=3)
        thresh_y = base_y - int(area_h / 1.25)   # the spike threshold line
        pygame.draw.line(screen, (230, 120, 110), (bx, thresh_y), (bx + bar_w, thresh_y), 1)
        label = small_font.render(ACTION_SHORT[i] if i < len(ACTION_SHORT) else str(i),
                                  True, COLORS["text"])
        screen.blit(label, (bx + bar_w // 2 - label.get_width() // 2, base_y + 4))


def _draw_tracks(screen, tracks, small_font, px, py, scale_x, scale_y):
    """SORT track overlay on the camera panel: box + persistent ID.
    Solid = detected this frame; amber = Kalman-predicted (occluded)."""
    for t in tracks:
        cx, cy, w, h = t.bbox
        r = pygame.Rect(int(px + (cx - w / 2) * scale_x),
                        int(py + (cy - h / 2) * scale_y),
                        max(3, int(w * scale_x)), max(3, int(h * scale_y)))
        predicted = t.time_since_update > 0
        if predicted:
            color = (250, 210, 100)
        else:
            color = (90, 230, 140) if t.cls == 0 else (250, 150, 90)
        pygame.draw.rect(screen, color, r, 2)
        label = small_font.render(f"#{t.id}", True, color)
        screen.blit(label, (r.x + 2, max(py, r.y - 15)))


def _draw_sidebar(screen, env, small_font, controller, show_3d, camera=None,
                  tracks=None):
    x = WIDTH
    pygame.draw.rect(screen, (8, 11, 17), (x, 0, SIDEBAR_W, HEIGHT))
    pygame.draw.line(screen, PANEL_EDGE, (x, 0), (x, HEIGHT))
    pad = 10
    px = x + pad
    w = SIDEBAR_W - 2 * pad

    top = _panel(screen, small_font, px, 10, w, 268, "FIRST-PERSON 3D VIEW (raycast)")
    if show_3d:
        view = view3d.render_first_person(env, size=(w - 12, 234))
        screen.blit(view, (px + 6, top + 2))
    else:
        screen.blit(small_font.render("off - press V to enable", True, COLORS["text"]),
                    (px + 10, top + 12))

    top2 = _panel(screen, small_font, px, 288, w, 250, "ROBOT CAMERA (vision input)")
    # Reuse the frame the controller already consumed this step rather than
    # rendering the camera a second time.
    cam = env.camera_frame() if camera is None else camera
    cam_surf = pygame.surfarray.make_surface(cam.transpose(1, 0, 2))
    cam_surf = pygame.transform.scale(cam_surf, (w - 12, 218))
    screen.blit(cam_surf, (px + 6, top2 + 2))
    if tracks is not None:
        _draw_tracks(screen, tracks, small_font, px + 6, top2 + 2,
                     (w - 12) / cam.shape[1], 218 / cam.shape[0])

    _draw_neurons(screen, small_font, px, 548, w, HEIGHT - 558, controller)


def _draw_minimap(screen, env, x, y, mw, mh):
    """Scaled top-down overview for the immersive 3D mode."""
    scale = min(mw / WIDTH, mh / HEIGHT)
    s = pygame.Surface((mw, mh), pygame.SRCALPHA)
    s.fill((10, 14, 21, 205))
    screen.blit(s, (x, y))
    for rect, hgt in zip(env.obstacles, env.obstacle_heights):
        color = (140, 86, 70) if (env.flight and hgt < 1.0) else COLORS["obstacle"]
        pygame.draw.rect(screen, color,
                         (int(x + rect.x * scale), int(y + rect.y * scale),
                          max(1, int(rect.w * scale)), max(1, int(rect.h * scale))))
    for idx in range(env.target_idx, len(env.targets)):
        tx, ty = env.targets[idx]
        color = COLORS["goal"] if idx == env.target_idx else (110, 126, 118)
        pygame.draw.circle(screen, color,
                           (int(x + tx * scale), int(y + ty * scale)), 4)
    bx, by = x + env.robot_x * scale, y + env.robot_y * scale
    pygame.draw.circle(screen, COLORS["robot"], (int(bx), int(by)), 4)
    hx = bx + math.cos(env.robot_angle) * 9
    hy = by + math.sin(env.robot_angle) * 9
    pygame.draw.line(screen, (235, 245, 255), (int(bx), int(by)), (int(hx), int(hy)), 2)
    pygame.draw.rect(screen, PANEL_EDGE, (x, y, mw, mh), 1)


def draw_immersive(screen, env: Environment, font, small_font, vision_result,
                   ranges, controller=None, controller_name: str = "lif",
                   yaw_offset: float = 0.0, fps=None, help_on: bool = False):
    """Fullscreen first-person 3D with a minimap - the 'flying in a room' view."""
    w, h = screen.get_size()
    view = view3d.render_first_person(env, size=(w, h), fov_deg=100,
                                      yaw_offset=yaw_offset)
    screen.blit(view, (0, 0))
    _draw_minimap(screen, env, w - 262, 12, 250, 175)
    escapes = getattr(controller, "escape_count", 0)
    lines = [
        f"IMMERSIVE 3D | {controller_name} | {env.map_spec.kind} #{env.map_spec.seed}"
        + (f" | Alt {env.altitude:.2f}" if env.flight else "")
        + (f" | FPS {fps:.0f}" if fps else ""),
        f"step {env.steps}   collisions {env.collisions}   escapes {escapes}   "
        f"goal seen: {'YES' if vision_result[0] else 'no'}",
        "TAB top-down | drag mouse: look around 360 | right-click: recenter view",
    ]
    bg = pygame.Surface((560, 70), pygame.SRCALPHA)
    bg.fill((8, 11, 17, 185))
    screen.blit(bg, (8, h - 76))
    for i, line in enumerate(lines):
        screen.blit(small_font.render(line, True, COLORS["text"]), (16, h - 70 + i * 20))
    if env.reached:
        msg = font.render("DESTINATION REACHED!", True, COLORS["goal"])
        screen.blit(msg, (w // 2 - msg.get_width() // 2, 24))
    if help_on:
        _draw_help(screen, font, small_font)


def _draw_help(screen, font, small_font):
    overlay = pygame.Surface((WIDTH + SIDEBAR_W, HEIGHT), pygame.SRCALPHA)
    overlay.fill((5, 8, 13, 215))
    screen.blit(overlay, (0, 0))
    title = font.render("FLYCORE EDGE - CONTROLS", True, (235, 240, 248))
    screen.blit(title, ((WIDTH + SIDEBAR_W) // 2 - title.get_width() // 2, 150))
    for i, line in enumerate(HELP_LINES):
        screen.blit(small_font.render(line, True, (205, 215, 228)),
                    ((WIDTH + SIDEBAR_W) // 2 - 190, 210 + i * 24))


def draw(screen, env: Environment, font, small_font, vision_result, ranges,
         controller=None, controller_name: str = "lif", show_3d: bool = True,
         drag_rect=None, help_on: bool = False, camera=None, fps=None,
         tracks=None):
    screen.fill(COLORS["background"])
    _draw_world(screen, env, ranges, drag_rect)
    _draw_hud(screen, env, small_font, vision_result, ranges, controller,
              controller_name, fps=fps, tracker_on=tracks is not None)
    _draw_sidebar(screen, env, small_font, controller, show_3d, camera=camera,
                  tracks=tracks)
    if env.reached:
        msg = font.render("DESTINATION REACHED!", True, COLORS["goal"])
        screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 20))
        stats = small_font.render(
            f"steps {env.steps}   path {env.path_length:.0f} px   "
            f"collisions {env.collisions}   (R reset, N new map)",
            True, COLORS["text"])
        screen.blit(stats, (WIDTH // 2 - stats.get_width() // 2, 54))
    if help_on:
        _draw_help(screen, font, small_font)
