"""Small leaky-integrate-and-fire-inspired action controller."""
from __future__ import annotations
import math

class LIFController:
    """
    A deliberately small, interpretable LIF-inspired controller.
    Inputs are normalized range readings (0 clear -> 1 very close) and
    a normalized visual goal bearing (-1 left -> +1 right).

    This is an engineering toy model, not a biologically validated fly circuit.
    """
    ACTIONS = ("FORWARD", "LEFT", "RIGHT", "SEEK", "BRAKE")
    name = "lif"

    # Default parameters come from the Kaggle grid search (24 configs x 25 maps,
    # best config confirmed on 100 held-out maps: 83% vs 67% success).
    def __init__(self, decay: float = 0.78, threshold: float = 0.78,
                 side_gain: float = 0.60, forward_drive: float = 0.85,
                 search_gain: float = 0.004, search_cap: float = 0.60):
        self.v = [0.0] * len(self.ACTIONS)
        self.spikes = [0] * len(self.ACTIONS)
        self.decay = decay
        self.threshold = threshold
        self.side_gain = side_gain
        self.forward_drive = forward_drive
        self.search_gain = search_gain
        self.search_cap = search_cap
        self.steps = 0
        self.avoid_dir = None
        self.steps_since_seen = 0

    def reset(self):
        self.v = [0.0] * len(self.ACTIONS)
        self.spikes = [0] * len(self.ACTIONS)
        self.steps = 0
        self.avoid_dir = None
        self.steps_since_seen = 0

    def decide(self, ranges, vision_result):
        left, front, right = ranges
        found, goal_error, pixel_count = vision_result
        self.steps_since_seen = 0 if found else self.steps_since_seen + 1
        # Feature activations. Close obstacle => high avoidance activation.
        left_danger = max(0.0, 1.0-left)
        front_danger = max(0.0, 1.0-front)
        right_danger = max(0.0, 1.0-right)

        # Evidence drives output-neuron membrane potentials. A LEFT turn is
        # excited by danger on the RIGHT (turn away from the obstacle), and
        # vice versa.
        drive = [
            self.forward_drive - 1.15*front_danger - 0.15*(left_danger+right_danger),
            0.05 + 1.25*front_danger + self.side_gain*right_danger - 0.50*left_danger,
            0.05 + 1.25*front_danger + self.side_gain*left_danger - 0.50*right_danger,
            0.10 + (0.90 + 0.40*abs(goal_error) if found
                    # Visual search: the longer the goal stays unseen, the more
                    # SEEK favors long ballistic runs over aimless curving.
                    else min(self.search_cap, self.search_gain * self.steps_since_seen)),
            0.05 + 1.25*front_danger if front < 0.10 else 0.0,
        ]
        # Visual bearing biases turning toward the goal.
        if found and goal_error < -0.10:
            drive[1] += 0.65
            drive[2] -= 0.35
        elif found and goal_error > 0.10:
            drive[2] += 0.65
            drive[1] -= 0.35
        # If the goal is centered, favor forward motion over SEEK.
        if found and abs(goal_error) < 0.12:
            drive[0] += 0.25
            drive[3] -= 0.25

        for i, value in enumerate(drive):
            self.v[i] = self.decay * self.v[i] + max(0.0, value) * 0.25
            self.spikes[i] = 1 if self.v[i] >= self.threshold else 0
        # Action selection reads the membrane potential BEFORE the post-spike
        # reset: resetting first made the winning neuron momentarily lose the
        # argmax every few steps, which produced a slow drift into circles.
        scores = [self.v[i] + drive[i]*0.04 for i in range(4)]
        for i in range(len(self.v)):
            if self.spikes[i]:
                self.v[i] = 0.0  # reset after a spike
        self.steps += 1

        # Safety-biased action selection is explicit, not left to the network.
        if front < 0.075:
            return "BRAKE"
        if front < 0.23:
            # Commit to one turn direction until the path clears; re-picking
            # each step made the robot rock LEFT/RIGHT in corners forever.
            if self.avoid_dir is None:
                self.avoid_dir = "LEFT" if left >= right else "RIGHT"
            return self.avoid_dir
        if front >= 0.40:
            # Hysteresis: release the commitment only when genuinely clear.
            # Releasing at 0.23 let corners re-trigger with the OPPOSITE
            # direction (wall two of the corner), rocking the robot in place.
            self.avoid_dir = None

        best = max(range(4), key=lambda i: scores[i])
        action = self.ACTIONS[best]
        if action == "SEEK":
            if found and goal_error < -0.10:
                return "LEFT"
            if found and goal_error > 0.10:
                return "RIGHT"
            return "FORWARD"
        return action
