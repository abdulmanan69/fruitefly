"""Escape-saccade reflex layer.

Fruit flies recover from blocked flight paths with rapid stereotyped turns
(saccades). This module adds the same idea as a thin wrapper around any
controller: when the robot has made almost no spatial progress over a
window of steps, the reflex takes over, turns toward the more open side,
pushes forward, then hands control back.

The wrapper is applied identically to every controller in experiments,
so the LIF-vs-baseline comparison stays fair.
"""
from __future__ import annotations
import math
from collections import deque


class ReflexWrapper:
    """Wraps a controller; intervenes only when the robot is stalled."""

    STALL_WINDOW = 70        # steps considered for progress measurement
    MIN_PROGRESS = 7.0       # pixels of net displacement expected per window
    # Corner rocking drifts a few px per oscillation, enough to defeat the
    # short stall window; a medium window with a looser bound catches it.
    DRIFT_WINDOW = 150
    DRIFT_MIN_PROGRESS = 24.0
    TURN_STEPS_BASE = 24     # saccade length; varied deterministically per event
    FORWARD_STEPS = 45
    # Exploration reflex: a robot can keep moving (so net displacement never
    # stalls) while orbiting a block or bouncing between the same walls
    # forever. Novelty is measured on a coarse grid of visited cells: if a
    # long window discovers almost no NEW cells while the goal is out of
    # view, a saccade with a long push breaks the cycle. Productive straight
    # runs always enter new cells, so they are never interrupted.
    NOVELTY_CELL = 60.0
    NOVELTY_WINDOW = 420
    NOVELTY_MIN_NEW = 2
    EXPLORE_FORWARD_STEPS = 140
    # Re-acquire saccade: a fly that loses sight of a target turns back
    # toward its last seen azimuth. Fires once per sighting, only when the
    # target slid out the SIDE of the view (large bearing error) and the
    # way ahead is clear enough to turn safely.
    REACQUIRE_MIN_LOST = 25
    REACQUIRE_MAX_LOST = 160
    REACQUIRE_TURN_STEPS = 10

    def __init__(self, controller):
        self.inner = controller
        self.name = getattr(controller, "name", type(controller).__name__)
        self.reset()

    def reset(self):
        self.inner.reset()
        self._history = deque(maxlen=self.STALL_WINDOW)
        self._drift_history = deque(maxlen=self.DRIFT_WINDOW)
        self._visited = set()    # coarse cells seen this episode
        self._novelty = deque(maxlen=self.NOVELTY_WINDOW)  # 1 = new cell
        self._plan = []          # queued escape actions
        self._escapes = 0        # saccade count; also varies the turn length
        self._last_turn_dir = "LEFT"
        self._extend_turns = 0
        self._step_count = 0
        self._last_seen_error = 0.0   # goal bearing when last in view
        self._last_seen_step = -10**9
        self._reacquired_at = -10**9

    @property
    def escape_count(self) -> int:
        return self._escapes

    def _stalled(self) -> bool:
        if len(self._history) == self.STALL_WINDOW:
            x0, y0 = self._history[0]
            x1, y1 = self._history[-1]
            if math.hypot(x1 - x0, y1 - y0) < self.MIN_PROGRESS:
                return True
        if len(self._drift_history) == self.DRIFT_WINDOW:
            x0, y0 = self._drift_history[0]
            x1, y1 = self._drift_history[-1]
            if math.hypot(x1 - x0, y1 - y0) < self.DRIFT_MIN_PROGRESS:
                return True
        return False

    def _start_escape(self, left, right, forward_steps):
        self._escapes += 1
        # Deterministic variation: successive saccades turn longer (up to a
        # near-full rotation) and alternate direction, so repeated escapes
        # from the same region sample genuinely new headings instead of
        # retrying similar ones.
        turn_steps = self.TURN_STEPS_BASE + (self._escapes * 17) % 60
        open_side = "LEFT" if left >= right else "RIGHT"
        if self._escapes % 2 == 0:
            open_side = "RIGHT" if open_side == "LEFT" else "LEFT"
        self._plan = [open_side] * turn_steps + ["FORWARD"] * forward_steps
        self._last_turn_dir = open_side
        self._extend_turns = 0
        self._history.clear()
        self._drift_history.clear()
        self._novelty.clear()
        return self._plan.pop(0)

    def decide(self, ranges, vision_result, pose=None):
        left, front, right = ranges
        self._step_count += 1
        if vision_result[0]:
            self._last_seen_error = vision_result[1]
            self._last_seen_step = self._step_count

        if self._plan:
            action = self._plan.pop(0)
            if action in ("FORWARD", "SEEK"):
                if front < 0.12:
                    # Abort a forward push if the world closes in.
                    self._plan.clear()
                elif front < 0.30 and self._extend_turns < 40:
                    # Still facing a wall after the planned turn (a corner):
                    # keep rotating the same way instead of pushing into it.
                    self._extend_turns += 1
                    self._plan.insert(0, action)
                    return self._last_turn_dir
                else:
                    return action
            else:
                return action

        if pose is not None:
            self._history.append((pose[0], pose[1]))
            self._drift_history.append((pose[0], pose[1]))
            cell = (int(pose[0] // self.NOVELTY_CELL),
                    int(pose[1] // self.NOVELTY_CELL))
            self._novelty.append(0 if cell in self._visited else 1)
            self._visited.add(cell)
            if self._stalled():
                return self._start_escape(left, right, self.FORWARD_STEPS)
            # Exploration escape: only while the goal is out of view, so a
            # final approach is never interrupted.
            if (not vision_result[0]
                    and len(self._novelty) == self.NOVELTY_WINDOW
                    and sum(self._novelty) < self.NOVELTY_MIN_NEW):
                return self._start_escape(left, right, self.EXPLORE_FORWARD_STEPS)

        # Re-acquire saccade: short-term target memory.
        lost_for = self._step_count - self._last_seen_step
        if (not vision_result[0]
                and self.REACQUIRE_MIN_LOST <= lost_for <= self.REACQUIRE_MAX_LOST
                and self._reacquired_at < self._last_seen_step
                and abs(self._last_seen_error) > 0.25
                and front >= 0.3):
            self._reacquired_at = self._step_count
            turn = "LEFT" if self._last_seen_error < 0 else "RIGHT"
            self._plan = [turn] * self.REACQUIRE_TURN_STEPS
            return self._plan.pop(0)

        return self.inner.decide(ranges, vision_result)
