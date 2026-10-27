"""Record a simulation episode to MP4 with OpenCV (visual analytics).

Renders the GUI offscreen (SDL dummy driver) and writes each frame through
cv2.VideoWriter, so a full demo video is produced without screen capture:

    python -m experiments.record_video --kind house --fly --steps 1500
    python -m experiments.record_video --kind open --chase --immersive
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import numpy as np
import pygame

from flycore.baseline import BaselineController
from flycore.controller import LIFController
from flycore.environment import Environment
from flycore.maps import MAP_KINDS, make_map
from flycore.perception import TrackerVision
from flycore.reflex import ReflexWrapper
from flycore.sensors import get_range_readings
from flycore.vision import detect_green_goal
from flycore import render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=MAP_KINDS, default="house")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--controller", choices=("lif", "baseline"), default="lif")
    parser.add_argument("--fly", action="store_true")
    parser.add_argument("--chase", action="store_true")
    parser.add_argument("--tracker", action="store_true",
                        help="use SORT tracked vision + overlay")
    parser.add_argument("--immersive", action="store_true",
                        help="record the fullscreen 3D view")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    pygame.init()
    screen = pygame.display.set_mode(render.window_size())
    font = pygame.font.SysFont("consolas", 28, bold=True)
    small_font = pygame.font.SysFont("consolas", 14)

    env = Environment(make_map(args.seed, kind=args.kind))
    env.agile = True
    env.flight = args.fly
    env.chase = args.chase
    make_ctrl = LIFController if args.controller == "lif" else BaselineController
    controller = ReflexWrapper(make_ctrl())
    tracker = TrackerVision() if args.tracker else None

    out_path = Path(args.out) if args.out else Path(
        f"results/videos/flycore_{args.kind}_{args.seed}_{args.controller}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, render.window_size())

    for _ in range(args.steps):
        camera = env.camera_frame()
        if tracker:
            vision = tracker(camera)
            tracks = tracker.last_tracks
        else:
            vision = detect_green_goal(camera)
            tracks = None
        ranges = get_range_readings(env)
        if not env.reached:
            action = controller.decide(ranges, vision,
                                       (env.robot_x, env.robot_y, env.robot_angle))
            env.move(action)
        if args.immersive:
            render.draw_immersive(screen, env, font, small_font, vision, ranges,
                                  controller=controller,
                                  controller_name=controller.name, fps=args.fps)
        else:
            render.draw(screen, env, font, small_font, vision, ranges,
                        controller=controller, controller_name=controller.name,
                        camera=camera, fps=args.fps, tracks=tracks)
        pygame.display.flip()
        frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        writer.write(frame[:, :, ::-1])   # RGB -> BGR for OpenCV
        if env.reached:
            # Hold the success frame for a second, then stop.
            for _ in range(args.fps):
                writer.write(frame[:, :, ::-1])
            break

    writer.release()
    pygame.quit()
    print(f"wrote {out_path} (reached={env.reached}, steps={env.steps})")


if __name__ == "__main__":
    main()
