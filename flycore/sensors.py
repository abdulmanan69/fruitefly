"""Virtual range-ray sensors for a top-down robot."""
from __future__ import annotations
import math
from .config import RAY_LENGTH, RAY_ANGLES_DEG, ROBOT_RADIUS

def ray_distance(environment, angle: float, max_distance: float = RAY_LENGTH) -> float:
    """Step a ray forward until it hits a wall/obstacle; return normalized distance.

    The ray tests clearance for the robot's full body radius, not a point:
    otherwise the robot clips corners its sensors never reported.
    """
    step = 4.0
    distance = 0.0
    while distance <= max_distance:
        x = environment.robot_x + math.cos(angle) * distance
        y = environment.robot_y + math.sin(angle) * distance
        if environment.is_solid(x, y, radius=ROBOT_RADIUS):
            return distance / max_distance
        distance += step
    return 1.0

def get_range_readings(environment):
    base = environment.robot_angle
    return [ray_distance(environment, base + math.radians(a)) for a in RAY_ANGLES_DEG]
