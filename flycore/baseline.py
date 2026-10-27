"""Rule-based baseline controller for comparison against the LIF controller.

Same interface as LIFController (reset / decide) so experiment code can
swap controllers freely. Deliberately simple and stateless: fixed-priority
reactive rules with no membrane dynamics.
"""
from __future__ import annotations


class BaselineController:
    """Fixed-priority reactive rules: avoid, then steer to goal, else cruise."""

    ACTIONS = ("FORWARD", "LEFT", "RIGHT", "SEEK", "BRAKE")
    name = "baseline"

    def __init__(self):
        self.avoid_dir = None

    def reset(self):
        self.avoid_dir = None

    def decide(self, ranges, vision_result):
        left, front, right = ranges
        found, goal_error, _pixel_count = vision_result

        # Safety rules first, mirroring the LIF controller's explicit overrides.
        if front < 0.075:
            return "BRAKE"
        if front < 0.25:
            # Commit to one turn direction until the path clears (no rocking).
            if self.avoid_dir is None:
                self.avoid_dir = "LEFT" if left >= right else "RIGHT"
            return self.avoid_dir
        if front >= 0.40:
            # Hysteresis: release only when genuinely clear, so a corner's
            # second wall cannot re-trigger with the opposite direction.
            self.avoid_dir = None
        if left < 0.12:
            return "RIGHT"
        if right < 0.12:
            return "LEFT"

        if found:
            if goal_error < -0.10:
                return "LEFT"
            if goal_error > 0.10:
                return "RIGHT"
        return "FORWARD"
