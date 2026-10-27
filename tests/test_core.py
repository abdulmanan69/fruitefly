import math
import unittest

import numpy as np

from flycore.baseline import BaselineController
from flycore.controller import LIFController
from flycore.environment import Environment
from flycore.geometry import Rect
from flycore.maps import MapSpec, default_map, generate_map, make_map, _reachable
from flycore.runner import run_episode, summarize
from flycore.sensors import get_range_readings
from flycore.vision import detect_green_goal


class TestVision(unittest.TestCase):
    def test_detects_green_goal(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        frame[20:30, 65:75] = [20, 240, 50]
        found, error, count = detect_green_goal(frame)
        self.assertTrue(found)
        self.assertGreater(error, 0)
        self.assertGreaterEqual(count, 5)

    def test_no_goal(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        found, error, count = detect_green_goal(frame)
        self.assertFalse(found)

    def test_detects_goal_in_simulated_camera(self):
        env = Environment(default_map())
        # Face the goal directly: it is straight ahead on the default map.
        found, error, count = detect_green_goal(env.camera_frame())
        self.assertFalse(found)  # too far away at spawn (distance > camera range)
        env.robot_x = env.goal_x - 200
        env.robot_y = env.goal_y
        found, error, count = detect_green_goal(env.camera_frame())
        self.assertTrue(found)
        self.assertLess(abs(error), 0.15)


class TestGeometry(unittest.TestCase):
    def test_collidepoint_and_inflate(self):
        rect = Rect(10, 10, 20, 10)
        self.assertTrue(rect.collidepoint(15, 15))
        self.assertFalse(rect.collidepoint(31, 15))
        grown = rect.inflate(10, 10)
        self.assertEqual((grown.x, grown.y, grown.w, grown.h), (5, 5, 30, 20))
        self.assertEqual(grown.centerx, rect.centerx)

    def test_colliderect(self):
        self.assertTrue(Rect(0, 0, 10, 10).colliderect(Rect(5, 5, 10, 10)))
        self.assertFalse(Rect(0, 0, 10, 10).colliderect(Rect(20, 20, 5, 5)))


class TestMaps(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_map(123), generate_map(123))

    def test_generated_maps_are_reachable(self):
        for seed in range(1, 21):
            spec = generate_map(seed)
            self.assertTrue(_reachable(spec.start, spec.goal, spec.obstacles),
                            f"seed {seed} produced an unreachable map")

    def test_start_goal_clearance(self):
        spec = generate_map(5)
        for rect in spec.obstacles:
            self.assertFalse(rect.collidepoint(spec.start[0], spec.start[1]))
            self.assertFalse(rect.collidepoint(spec.goal[0], spec.goal[1]))


class TestEnvironment(unittest.TestCase):
    def test_move_blocks_collisions(self):
        env = Environment(default_map())
        env.robot_angle = math.pi  # face the left wall
        for _ in range(200):
            env.move("FORWARD")
        self.assertGreater(env.collisions, 0)
        self.assertGreaterEqual(env.robot_x, 0)

    def test_sensors_normalized(self):
        env = Environment(default_map())
        for reading in get_range_readings(env):
            self.assertGreaterEqual(reading, 0.0)
            self.assertLessEqual(reading, 1.0)

    def test_camera_frame_shape(self):
        env = Environment(default_map())
        frame = env.camera_frame()
        self.assertEqual(frame.shape, (150, 240, 3))
        self.assertEqual(frame.dtype, np.uint8)


class TestControllers(unittest.TestCase):
    def test_lif_action_is_valid(self):
        controller = LIFController()
        action = controller.decide([1.0, 1.0, 1.0], (False, 0.0, 0))
        self.assertIn(action, controller.ACTIONS)

    def test_lif_close_front_obstacle_brakes(self):
        controller = LIFController()
        action = controller.decide([0.8, 0.02, 0.8], (False, 0.0, 0))
        self.assertEqual(action, "BRAKE")

    def test_baseline_action_is_valid(self):
        controller = BaselineController()
        action = controller.decide([1.0, 1.0, 1.0], (False, 0.0, 0))
        self.assertIn(action, controller.ACTIONS)

    def test_baseline_steers_toward_goal(self):
        controller = BaselineController()
        self.assertEqual(controller.decide([1.0, 1.0, 1.0], (True, -0.5, 30)), "LEFT")
        self.assertEqual(controller.decide([1.0, 1.0, 1.0], (True, 0.5, 30)), "RIGHT")
        self.assertEqual(controller.decide([1.0, 1.0, 1.0], (True, 0.0, 30)), "FORWARD")


class TestRunner(unittest.TestCase):
    def test_episode_returns_metrics(self):
        row = run_episode(BaselineController(), generate_map(3), max_steps=150)
        for key in ("controller", "seed", "success", "steps", "collisions",
                    "path_length", "avg_decide_ms", "peak_mem_kb"):
            self.assertIn(key, row)
        self.assertEqual(row["controller"], "baseline")
        self.assertLessEqual(row["steps"], 150)

    def test_summarize(self):
        rows = [
            {"controller": "lif", "success": 1, "steps": 100, "collisions": 0,
             "path_length": 200.0, "avg_decide_ms": 0.01, "p95_decide_ms": 0.02,
             "avg_vision_ms": 0.3, "peak_mem_kb": 150.0},
            {"controller": "lif", "success": 0, "steps": 500, "collisions": 3,
             "path_length": 900.0, "avg_decide_ms": 0.01, "p95_decide_ms": 0.02,
             "avg_vision_ms": 0.3, "peak_mem_kb": 150.0},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["lif"]["episodes"], 2)
        self.assertAlmostEqual(summary["lif"]["success_rate"], 0.5)
        self.assertEqual(summary["lif"]["avg_steps_to_goal"], 100.0)


class TestAdvancedMaps(unittest.TestCase):
    def test_maze_deterministic_and_solvable(self):
        for seed in range(1, 11):
            m1, m2 = make_map(seed, "maze"), make_map(seed, "maze")
            self.assertEqual(m1, m2)
            self.assertEqual(m1.kind, "maze")
            self.assertGreater(len(m1.obstacles), 10)
            self.assertTrue(_reachable(m1.start, m1.goal, m1.obstacles),
                            f"maze seed {seed} unreachable")

    def test_rooms_solvable(self):
        for seed in range(1, 11):
            m = make_map(seed, "rooms")
            self.assertEqual(m.kind, "rooms")
            self.assertTrue(_reachable(m.start, m.goal, m.obstacles),
                            f"rooms seed {seed} unreachable")

    def test_house_solvable_with_furniture_and_stairs(self):
        for seed in range(1, 9):
            m = make_map(seed, "house")
            self.assertEqual(m.kind, "house")
            self.assertTrue(_reachable(m.start, m.goal, m.obstacles),
                            f"house seed {seed} unreachable")
            self.assertEqual(len(m.heights), len(m.obstacles))
            self.assertTrue(any(h < 1.0 for h in m.heights))  # furniture/stairs

    def test_checkpoints_sit_on_free_path(self):
        spec = make_map(3, "open", checkpoints=1)
        self.assertEqual(len(spec.checkpoints), 1)
        cx, cy = spec.checkpoints[0]
        env = Environment(spec)
        self.assertFalse(env.is_solid(cx, cy, radius=13))
        # Both legs of the sequential task stay reachable.
        self.assertTrue(_reachable(spec.start, (cx, cy), spec.obstacles))
        self.assertTrue(_reachable((cx, cy), spec.goal, spec.obstacles))


class TestCheckpointEnvironment(unittest.TestCase):
    def test_target_advances_in_order(self):
        spec = MapSpec(seed=0, start=(100.0, 350.0, 0.0), goal=(900.0, 350.0),
                       obstacles=(), checkpoints=((300.0, 350.0),))
        env = Environment(spec)
        self.assertEqual(env.target_idx, 0)
        self.assertEqual(env.goal_x, 300.0)   # active target is the checkpoint
        self.assertEqual(env.checkpoints_remaining, 1)
        for _ in range(600):
            env.move("FORWARD")
            if env.reached:
                break
        self.assertTrue(env.reached)
        self.assertEqual(env.target_idx, 1)
        self.assertEqual(env.checkpoints_remaining, 0)


class TestFlightAndChase(unittest.TestCase):
    LOW_WALL = MapSpec(seed=0, start=(100.0, 350.0, 0.0), goal=(900.0, 350.0),
                       obstacles=(Rect(300, 250, 40, 200),), heights=(0.5,))

    def test_flight_climbs_over_low_obstacle(self):
        env = Environment(self.LOW_WALL)
        env.flight = True
        for _ in range(900):
            env.move("FORWARD")
            if env.reached:
                break
        self.assertTrue(env.reached)
        self.assertEqual(env.collisions, 0)
        self.assertEqual(env.altitude, 0.0)   # settled back down

    def test_without_flight_same_wall_blocks(self):
        env = Environment(self.LOW_WALL)
        for _ in range(400):
            env.move("FORWARD")
        self.assertFalse(env.reached)
        self.assertGreater(env.collisions, 0)

    def test_chase_goal_moves_and_stays_legal(self):
        spec = MapSpec(seed=3, start=(100.0, 350.0, 0.0), goal=(600.0, 350.0),
                       obstacles=())
        env = Environment(spec)
        env.chase = True
        g0 = (env.goal_x, env.goal_y)
        for _ in range(300):
            env.move("FORWARD")
            if env.reached:
                break
        self.assertNotEqual(g0, (env.goal_x, env.goal_y))
        self.assertFalse(env.is_solid(env.goal_x, env.goal_y, 14, altitude=0.0))


class TestVisionOcclusion(unittest.TestCase):
    def test_goal_behind_wall_is_not_seen(self):
        wall = MapSpec(seed=0, start=(300.0, 350.0, 0.0), goal=(560.0, 350.0),
                       obstacles=(Rect(420, 250, 40, 200),))
        env = Environment(wall)
        found, _, _ = detect_green_goal(env.camera_frame())
        self.assertFalse(found, "camera saw the goal through a wall")
        clear = MapSpec(seed=0, start=(300.0, 350.0, 0.0), goal=(560.0, 350.0),
                        obstacles=())
        env = Environment(clear)
        found, _, _ = detect_green_goal(env.camera_frame())
        self.assertTrue(found)

    def test_low_furniture_does_not_hide_goal(self):
        spec = MapSpec(seed=0, start=(300.0, 350.0, 0.0), goal=(560.0, 350.0),
                       obstacles=(Rect(420, 250, 40, 200),), heights=(0.45,))
        env = Environment(spec)
        found, _, _ = detect_green_goal(env.camera_frame())
        self.assertTrue(found, "low furniture should not block sight")


class TestCornerEscape(unittest.TestCase):
    def test_bee_escapes_a_room_corner(self):
        from flycore.controller import LIFController
        from flycore.reflex import ReflexWrapper
        from flycore.sensors import get_range_readings
        from flycore.vision import detect_green_goal
        # An L-shaped corner like a room's: vertical wall + horizontal wall
        # meeting at (500, 350); the bee starts aimed straight into it.
        spec = MapSpec(seed=0, start=(430.0, 290.0, 0.7), goal=(900.0, 650.0),
                       obstacles=(Rect(500, 0, 40, 360), Rect(60, 350, 480, 40)))
        env = Environment(spec)
        ctrl = ReflexWrapper(LIFController())
        for _ in range(900):
            vision = detect_green_goal(env.camera_frame())
            ranges = get_range_readings(env)
            env.move(ctrl.decide(ranges, vision,
                                 (env.robot_x, env.robot_y, env.robot_angle)))
        corner_dist = math.hypot(env.robot_x - 500, env.robot_y - 350)
        self.assertGreater(corner_dist, 160,
                           f"still near the corner after 900 steps "
                           f"(dist {corner_dist:.0f}, escapes {ctrl.escape_count})")


class TestPerception(unittest.TestCase):
    def test_hungarian_finds_optimal_assignment(self):
        from flycore.perception import hungarian
        cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
        pairs = dict(hungarian(cost))
        self.assertEqual(len(pairs), 3)
        total = sum(cost[i][j] for i, j in pairs.items())
        self.assertEqual(total, 5)   # (0,1)+(1,0)+(2,2)

    def test_hungarian_rectangular(self):
        from flycore.perception import hungarian
        pairs = hungarian([[1.0, 9.0, 9.0], [9.0, 1.0, 9.0]])
        self.assertEqual(sorted(pairs), [(0, 0), (1, 1)])

    def test_kalman_learns_velocity(self):
        from flycore.perception import KalmanBox
        kf = KalmanBox((10.0, 50.0, 8.0, 8.0))
        for step in range(1, 11):
            kf.predict()
            kf.update((10.0 + 3.0 * step, 50.0, 8.0, 8.0))
        self.assertGreater(kf.velocity[0], 1.5)
        self.assertAlmostEqual(kf.velocity[1], 0.0, delta=0.5)

    def test_sort_keeps_persistent_ids(self):
        from flycore.perception import Sort
        sort = Sort()
        ids_a, ids_b = set(), set()
        for step in range(8):
            tracks = sort.update([(20 + step, 40, 10, 10, 0),
                                  (150, 90 + step, 12, 12, 1)])
            for t in tracks:
                (ids_a if t.cls == 0 else ids_b).add(t.id)
        self.assertEqual(len(ids_a), 1)
        self.assertEqual(len(ids_b), 1)
        self.assertNotEqual(ids_a, ids_b)

    def test_detects_goal_and_obstacle_boxes(self):
        from flycore.perception import detect_objects, CLS_GOAL, CLS_OBSTACLE
        frame = np.zeros((150, 240, 3), dtype=np.uint8)
        frame[40:60, 100:120] = (45, 230, 105)    # goal
        frame[80:110, 30:70] = (218, 105, 78)     # obstacle
        classes = sorted(d[4] for d in detect_objects(frame))
        self.assertEqual(classes, [CLS_GOAL, CLS_OBSTACLE])

    def test_tracker_vision_bridges_occlusion(self):
        from flycore.perception import TrackerVision
        tv = TrackerVision()
        visible = np.zeros((150, 240, 3), dtype=np.uint8)
        visible[60:80, 140:165] = (45, 230, 105)
        empty = np.zeros((150, 240, 3), dtype=np.uint8)
        for _ in range(6):
            found, error, _ = tv(visible)
        self.assertTrue(found)
        self.assertGreater(error, 0.0)   # goal right of center
        for _ in range(4):               # goal occluded: Kalman bridges
            found, error, _ = tv(empty)
        self.assertTrue(found)
        for _ in range(10):              # long loss: track dies honestly
            found, _, _ = tv(empty)
        self.assertFalse(found)


class TestCppParity(unittest.TestCase):
    def test_cpp_matches_python_when_built(self):
        from flycore import cpp_bridge
        if not cpp_bridge.is_available():
            self.skipTest("C++ shared library not built on this machine")
        import random
        py, cpp = LIFController(), cpp_bridge.CppLIFController()
        rng = random.Random(11)
        for _ in range(5000):
            ranges = (rng.random(), rng.random(), rng.random())
            found = rng.random() < 0.5
            vision = (found, rng.uniform(-1, 1) if found else 0.0, 20)
            self.assertEqual(py.decide(ranges, vision), cpp.decide(ranges, vision))


if __name__ == "__main__":
    unittest.main()
