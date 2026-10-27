from __future__ import annotations
import argparse
import os
import random
import sys

import pygame

from flycore.config import WIDTH, HEIGHT, FPS, ROBOT_RADIUS
from flycore.environment import Environment
from flycore.geometry import Rect
from flycore.sensors import get_range_readings
from flycore.vision import detect_green_goal
from flycore.controller import LIFController
from flycore.baseline import BaselineController
from flycore.maps import MAP_KINDS, default_map, make_map
from flycore.reflex import ReflexWrapper
from flycore import perception
from flycore import render
from flycore import vision_yolo


def main(argv=None):
    parser = argparse.ArgumentParser(description="FlyCore Edge GUI")
    parser.add_argument("--kind", choices=MAP_KINDS, default=None,
                        help="start on this map kind instead of the demo map")
    parser.add_argument("--seed", type=int, default=7, help="map seed for --kind")
    parser.add_argument("--immersive", action="store_true",
                        help="start in fullscreen 3D view (TAB toggles)")
    parser.add_argument("--fly", action="store_true", help="start with flight mode on")
    parser.add_argument("--chase", action="store_true", help="start with chase mode on")
    args = parser.parse_args(argv)

    pygame.init()
    pygame.display.set_caption("FlyCore Edge — Visual Recognition + Safe Navigation")
    screen = pygame.display.set_mode(render.window_size())
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 28, bold=True)
    small_font = pygame.font.SysFont("consolas", 14)

    env = Environment(make_map(args.seed, kind=args.kind) if args.kind
                      else default_map())
    env.agile = True   # GUI bee flies with speed bursts; benchmarks keep it off
    env.flight = args.fly
    env.chase = args.chase
    controllers = [ReflexWrapper(LIFController()), ReflexWrapper(BaselineController())]
    controller_idx = 0
    map_kind = args.kind or "open"
    checkpoint_mode = False
    show_3d = True
    help_on = False
    paused = False
    drag_start = None
    shot_idx = 0
    running = True
    immersive = args.immersive   # TAB: fullscreen 3D with minimap
    yaw_offset = 0.0      # mouse-look offset (radians) in immersive mode
    mouse_look = False
    tracker_vision = perception.TrackerVision()   # SORT: Kalman + Hungarian MOT
    use_tracker = False   # key O: tracked vision with occlusion bridging
    yolo_vision = None    # key Y: YOLO11 backend (needs Kaggle-trained weights)
    use_yolo = False

    def new_map():
        env.reset(make_map(random.randrange(1, 10_000_000), kind=map_kind,
                           checkpoints=1 if checkpoint_mode else 0))
        controllers[controller_idx].reset()

    while running:
        clock.tick(FPS)
        controller = controllers[controller_idx]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    controller.reset()
                elif event.key == pygame.K_n:
                    new_map()
                elif event.key == pygame.K_m:
                    map_kind = MAP_KINDS[(MAP_KINDS.index(map_kind) + 1) % len(MAP_KINDS)]
                    new_map()
                elif event.key == pygame.K_c:
                    checkpoint_mode = not checkpoint_mode
                    new_map()
                elif event.key == pygame.K_b:
                    controller_idx = (controller_idx + 1) % len(controllers)
                    env.reset()
                    controllers[controller_idx].reset()
                elif event.key == pygame.K_o:
                    use_tracker = not use_tracker
                    tracker_vision.reset()
                elif event.key == pygame.K_y:
                    if vision_yolo.is_available():
                        if yolo_vision is None:
                            yolo_vision = vision_yolo.YoloVision()
                        use_yolo = not use_yolo
                    else:
                        print("YOLO11 weights missing: train them with "
                              "python kaggle_run/run_on_kaggle.py --username <u> --yolo")
                elif event.key == pygame.K_f:
                    env.flight = not env.flight
                    env.altitude = 0.0
                elif event.key == pygame.K_t:
                    env.chase = not env.chase
                elif event.key == pygame.K_TAB:
                    immersive = not immersive
                    yaw_offset = 0.0
                elif event.key == pygame.K_v:
                    show_3d = not show_3d
                elif event.key == pygame.K_h:
                    help_on = not help_on
                elif event.key == pygame.K_s:
                    os.makedirs(os.path.join("results", "screenshots"), exist_ok=True)
                    shot_idx += 1
                    path = os.path.join("results", "screenshots",
                                        f"flycore_{env.map_spec.kind}_{env.map_spec.seed}_{shot_idx}.png")
                    pygame.image.save(screen, path)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if immersive:
                    if event.button == 1:
                        mouse_look = True
                    elif event.button == 3:
                        yaw_offset = 0.0   # recenter the view
                elif mx < WIDTH:
                    if event.button == 1:
                        drag_start = (mx, my)
                    elif event.button == 3:
                        env.remove_obstacle_at(mx, my)
            elif event.type == pygame.MOUSEMOTION and immersive and mouse_look:
                yaw_offset += event.rel[0] * 0.006   # drag = look around 360
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and immersive:
                mouse_look = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and drag_start:
                mx, my = event.pos
                x0, y0 = drag_start
                drag_start = None
                x, y = min(x0, mx), min(y0, my)
                w, h = abs(mx - x0), abs(my - y0)
                if w >= 18 and h >= 18 and mx < WIDTH:
                    new_rect = Rect(x, y, w, h)
                    # Never drop a wall directly on top of the robot.
                    if not new_rect.inflate(2 * ROBOT_RADIUS, 2 * ROBOT_RADIUS) \
                            .collidepoint(env.robot_x, env.robot_y):
                        env.add_obstacle(new_rect)

        drag_rect = None
        if drag_start:
            mx, my = pygame.mouse.get_pos()
            drag_rect = (min(drag_start[0], mx), min(drag_start[1], my),
                         abs(mx - drag_start[0]), abs(my - drag_start[1]))

        camera = env.camera_frame()
        if use_yolo and yolo_vision is not None:
            vision_result = yolo_vision(camera)
            tracks = None
        elif use_tracker:
            vision_result = tracker_vision(camera)
            tracks = tracker_vision.last_tracks
        else:
            vision_result = detect_green_goal(camera)
            tracks = None
        ranges = get_range_readings(env)
        if not paused and not env.reached and not help_on:
            pose = (env.robot_x, env.robot_y, env.robot_angle)
            action = controller.decide(ranges, vision_result, pose)
            env.move(action)
        if immersive:
            render.draw_immersive(screen, env, font, small_font, vision_result,
                                  ranges, controller=controller,
                                  controller_name=controller.name,
                                  yaw_offset=yaw_offset, fps=clock.get_fps(),
                                  help_on=help_on)
        else:
            render.draw(screen, env, font, small_font, vision_result, ranges,
                        controller=controller, controller_name=controller.name,
                        show_3d=show_3d, drag_rect=drag_rect, help_on=help_on,
                        camera=camera, fps=clock.get_fps(), tracks=tracks)
        if paused:
            label = font.render("PAUSED", True, (255, 220, 110))
            screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT - 48))
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
