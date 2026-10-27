"""Procedural map generation with guaranteed start-to-goal reachability.

Each map is fully described by a MapSpec so experiments are reproducible:
the same seed always yields the same world on any machine. Three map kinds
are supported:

- "open"  - scattered bars and blocks (the original style)
- "maze"  - a braided recursive-backtracker maze (corridors with loops)
- "rooms" - wall dividers with door gaps plus scattered blocks

A map may also carry checkpoints: intermediate waypoints the robot must
visit in order before the final goal activates (see Environment.targets).
Checkpoints are placed on the BFS path from start to goal, so a checkpoint
map is always solvable in sequence.
"""
from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, replace

from .config import WIDTH, HEIGHT, ROBOT_RADIUS
from .geometry import Rect

START_CLEARANCE = 80.0   # obstacle-free radius around start and goal
GRID_CELL = 20           # BFS grid resolution for the reachability check

MAP_KINDS = ("open", "maze", "rooms", "house")


@dataclass(frozen=True)
class MapSpec:
    seed: int
    start: tuple          # (x, y, heading_radians)
    goal: tuple           # (x, y)
    obstacles: tuple      # tuple[Rect, ...]
    kind: str = "open"
    checkpoints: tuple = ()   # ((x, y), ...) visited in order before the goal
    heights: tuple = ()       # per-obstacle height 0..1; < 1.0 = flyable-over
                              # (only used when the GUI's flight mode is on)


DEFAULT_OBSTACLES = (
    Rect(260, 90, 42, 260),
    Rect(425, 355, 42, 255),
    Rect(590, 65, 42, 255),
    Rect(755, 355, 42, 255),
    Rect(355, 165, 100, 34),
    Rect(655, 500, 90, 34),
)


def default_map() -> MapSpec:
    """The original hand-designed layout, kept for demos and regression tests."""
    return MapSpec(
        seed=0,
        start=(85.0, HEIGHT / 2, 0.0),
        goal=(WIDTH - 85.0, HEIGHT / 2),
        obstacles=DEFAULT_OBSTACLES,
    )


def _clear_of_endpoints(rect: Rect, start, goal) -> bool:
    zone = rect.inflate(2 * START_CLEARANCE, 2 * START_CLEARANCE)
    return not (zone.collidepoint(start[0], start[1]) or zone.collidepoint(goal[0], goal[1]))


def _is_blocked(x: float, y: float, obstacles, radius: float) -> bool:
    if x < radius or y < radius or x >= WIDTH - radius or y >= HEIGHT - radius:
        return True
    # Allocation-free equivalent of rect.inflate(2r, 2r).collidepoint(x, y).
    for rect in obstacles:
        if (rect.x - radius <= x < rect.x + rect.w + radius
                and rect.y - radius <= y < rect.y + rect.h + radius):
            return True
    return False


def _bfs_path(start, goal, obstacles):
    """Grid BFS; returns the list of cells from start to goal, or None."""
    cols, rows = WIDTH // GRID_CELL, HEIGHT // GRID_CELL
    radius = ROBOT_RADIUS + 2

    def cell_of(x, y):
        return int(x // GRID_CELL), int(y // GRID_CELL)

    def free(cx, cy):
        if not (0 <= cx < cols and 0 <= cy < rows):
            return False
        px, py = cx * GRID_CELL + GRID_CELL / 2, cy * GRID_CELL + GRID_CELL / 2
        return not _is_blocked(px, py, obstacles, radius)

    start_cell, goal_cell = cell_of(start[0], start[1]), cell_of(goal[0], goal[1])
    if not free(*start_cell) or not free(*goal_cell):
        return None
    frontier = deque([start_cell])
    parent = {start_cell: None}
    while frontier:
        cell = frontier.popleft()
        if cell == goal_cell:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        cx, cy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cx + dx, cy + dy)
            if nxt not in parent and free(*nxt):
                parent[nxt] = cell
                frontier.append(nxt)
    return None


def _reachable(start, goal, obstacles) -> bool:
    """Coarse grid BFS: is there a robot-width path from start to goal?"""
    return _bfs_path(start, goal, obstacles) is not None


def generate_map(seed: int, min_obstacles: int = 6, max_obstacles: int = 11) -> MapSpec:
    """Generate a random solvable open map. Deterministic for a given seed."""
    rng = random.Random(seed)
    start = (85.0, rng.uniform(120.0, HEIGHT - 120.0), 0.0)
    goal = (WIDTH - 85.0, rng.uniform(120.0, HEIGHT - 120.0))

    n_target = rng.randint(min_obstacles, max_obstacles)
    for attempt in range(60):
        obstacles = []
        tries = 0
        while len(obstacles) < n_target and tries < 300:
            tries += 1
            if rng.random() < 0.55:          # vertical bar
                w, h = rng.randint(34, 52), rng.randint(140, 290)
            elif rng.random() < 0.5:          # horizontal bar
                w, h = rng.randint(90, 200), rng.randint(30, 46)
            else:                             # block
                w, h = rng.randint(55, 110), rng.randint(55, 110)
            x = rng.uniform(140, WIDTH - 140 - w)
            y = rng.uniform(40, HEIGHT - 40 - h)
            rect = Rect(round(x), round(y), w, h)
            if not _clear_of_endpoints(rect, start, goal):
                continue
            if any(rect.inflate(2 * ROBOT_RADIUS + 14, 2 * ROBOT_RADIUS + 14).colliderect(o)
                   for o in obstacles):
                continue
            obstacles.append(rect)
        obstacles = tuple(obstacles)
        if len(obstacles) >= min_obstacles and _reachable(start, goal, obstacles):
            return MapSpec(seed=seed, start=start, goal=goal, obstacles=obstacles)
        # Relax density if placement keeps failing.
        if attempt % 10 == 9 and n_target > min_obstacles:
            n_target -= 1

    # Extremely unlucky seed: fall back to a sparse but valid map.
    return MapSpec(seed=seed, start=start, goal=goal, obstacles=())


# ---------------------------------------------------------------------------
# Maze maps: recursive backtracker over a coarse cell grid, then "braided"
# (a fraction of remaining internal walls removed) so the maze has loops.
# Loops matter for reactive controllers: a perfect maze punishes any
# controller without global memory with long dead-end backtracking.
# ---------------------------------------------------------------------------

MAZE_COLS, MAZE_ROWS = 9, 6
MAZE_MARGIN = 20
MAZE_WALL_T = 14


def generate_maze(seed: int, braid: float = 0.18) -> MapSpec:
    """Generate a braided maze. Solvable by construction (spanning tree)."""
    rng = random.Random(seed)
    cols, rows = MAZE_COLS, MAZE_ROWS
    cw = (WIDTH - 2 * MAZE_MARGIN) / cols
    ch = (HEIGHT - 2 * MAZE_MARGIN) / rows

    # v_walls[(c, r)] = wall to the RIGHT of cell (c, r); h_walls = BELOW.
    v_walls = {(c, r) for c in range(cols - 1) for r in range(rows)}
    h_walls = {(c, r) for c in range(cols) for r in range(rows - 1)}

    stack = [(0, rng.randrange(rows))]
    visited = {stack[0]}
    while stack:
        c, r = stack[-1]
        neighbors = [(c + dc, r + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if (c + dc, r + dr) not in visited
                     and 0 <= c + dc < cols and 0 <= r + dr < rows]
        if not neighbors:
            stack.pop()
            continue
        nc, nr = rng.choice(neighbors)
        if nc != c:
            v_walls.discard((min(c, nc), r))
        else:
            h_walls.discard((c, min(r, nr)))
        visited.add((nc, nr))
        stack.append((nc, nr))

    # Braid: knock out a fraction of the remaining internal walls.
    for wall_set in (v_walls, h_walls):
        for wall in sorted(wall_set):
            if rng.random() < braid:
                wall_set.discard(wall)

    obstacles = []
    for c, r in sorted(v_walls):
        x = MAZE_MARGIN + (c + 1) * cw
        y = MAZE_MARGIN + r * ch
        obstacles.append(Rect(round(x - MAZE_WALL_T / 2), round(y - MAZE_WALL_T / 2),
                              MAZE_WALL_T, round(ch + MAZE_WALL_T)))
    for c, r in sorted(h_walls):
        x = MAZE_MARGIN + c * cw
        y = MAZE_MARGIN + (r + 1) * ch
        obstacles.append(Rect(round(x - MAZE_WALL_T / 2), round(y - MAZE_WALL_T / 2),
                              round(cw + MAZE_WALL_T), MAZE_WALL_T))

    def center(cell):
        return (MAZE_MARGIN + (cell[0] + 0.5) * cw, MAZE_MARGIN + (cell[1] + 0.5) * ch)

    sx, sy = center((0, rng.randrange(rows)))
    gx, gy = center((cols - 1, rng.randrange(rows)))
    return MapSpec(seed=seed, start=(sx, sy, 0.0), goal=(gx, gy),
                   obstacles=tuple(obstacles), kind="maze")


# ---------------------------------------------------------------------------
# Room maps: 2-3 full-height wall dividers with door gaps, plus a few blocks.
# ---------------------------------------------------------------------------

def generate_rooms(seed: int) -> MapSpec:
    """Generate a solvable rooms-and-doors map. Deterministic per seed."""
    rng = random.Random(seed)
    start = (70.0, rng.uniform(100.0, HEIGHT - 100.0), 0.0)
    goal = (WIDTH - 70.0, rng.uniform(100.0, HEIGHT - 100.0))

    for _attempt in range(80):
        xs = sorted(rng.sample(range(220, WIDTH - 220, 20), rng.randint(2, 3)))
        if any(b - a < 170 for a, b in zip(xs, xs[1:])):
            continue
        obstacles = []
        for x in xs:
            # Two doors per divider: with one narrow door, a reactive
            # controller wandering an open room almost never lines up with
            # the gap (observed 0% success over 2,500-step episodes).
            door_ys = sorted(rng.uniform(70.0, HEIGHT - 230.0) for _ in range(2))
            door_h = rng.uniform(110.0, 160.0)
            y0 = 0.0
            for dy in door_ys:
                if dy - y0 > 24:
                    obstacles.append(Rect(x, round(y0), 30, round(dy - y0)))
                y0 = max(y0, dy + door_h)
            if HEIGHT - y0 > 24:
                obstacles.append(Rect(x, round(y0), 30, round(HEIGHT - y0)))
        for _ in range(rng.randint(2, 4)):
            w, h = rng.randint(50, 100), rng.randint(50, 100)
            bx = rng.uniform(140, WIDTH - 140 - w)
            by = rng.uniform(50, HEIGHT - 50 - h)
            rect = Rect(round(bx), round(by), w, h)
            if not _clear_of_endpoints(rect, start, goal):
                continue
            if any(rect.inflate(2 * ROBOT_RADIUS + 14, 2 * ROBOT_RADIUS + 14).colliderect(o)
                   for o in obstacles):
                continue
            obstacles.append(rect)
        obstacles = tuple(obstacles)
        if _reachable(start, goal, obstacles):
            return MapSpec(seed=seed, start=start, goal=goal,
                           obstacles=obstacles, kind="rooms")

    return MapSpec(seed=seed, start=start, goal=goal, obstacles=(), kind="rooms")


# ---------------------------------------------------------------------------
# House maps: a four-room floor plan with doorways, low furniture the bee
# can fly over, and a staircase of ascending blocks. Always ground-solvable
# (BFS through the doors), so flight mode is helpful but never required.
# ---------------------------------------------------------------------------

def generate_house(seed: int) -> MapSpec:
    rng = random.Random(seed)
    wall_t = 26
    for _attempt in range(80):
        obstacles, heights = [], []
        vx = rng.randrange(440, 560, 20)   # vertical interior wall
        hy = rng.randrange(300, 400, 20)   # horizontal interior wall

        # Interior walls, one doorway per segment half.
        for y0, y1 in ((0.0, float(hy)), (float(hy), float(HEIGHT))):
            door = rng.uniform(y0 + 50, y1 - 50 - 130)
            end = door + rng.uniform(110, 140)
            if door - y0 > 20:
                obstacles.append(Rect(vx, round(y0), wall_t, round(door - y0)))
                heights.append(1.0)
            if y1 - end > 20:
                obstacles.append(Rect(vx, round(end), wall_t, round(y1 - end)))
                heights.append(1.0)
        for x0, x1 in ((0.0, float(vx)), (float(vx), float(WIDTH))):
            door = rng.uniform(x0 + 50, x1 - 50 - 130)
            end = door + rng.uniform(110, 140)
            if door - x0 > 20:
                obstacles.append(Rect(round(x0), hy, round(door - x0), wall_t))
                heights.append(1.0)
            if x1 - end > 20:
                obstacles.append(Rect(round(end), hy, round(x1 - end), wall_t))
                heights.append(1.0)

        start = (85.0, rng.uniform(70.0, hy - 70.0), 0.0)
        goal = (WIDTH - 85.0, rng.uniform(hy + 70.0, HEIGHT - 70.0))

        # Staircase: four ascending low blocks in the lower-left room.
        sx = rng.uniform(80, vx - 80 - 4 * 62)
        sy = rng.uniform(hy + 60, HEIGHT - 60 - 95)
        stairs = [Rect(round(sx + i * 62), round(sy), 58, 92) for i in range(4)]
        if not all(_clear_of_endpoints(s, start, goal) for s in stairs):
            continue
        obstacles.extend(stairs)
        heights.extend(0.22 + 0.16 * i for i in range(4))

        # Furniture: low blocks the bee can fly over in flight mode.
        for _ in range(rng.randint(3, 5)):
            w, h = rng.randint(55, 110), rng.randint(45, 95)
            fx = rng.uniform(60, WIDTH - 60 - w)
            fy = rng.uniform(60, HEIGHT - 60 - h)
            rect = Rect(round(fx), round(fy), w, h)
            if not _clear_of_endpoints(rect, start, goal):
                continue
            if any(rect.inflate(2 * ROBOT_RADIUS + 16, 2 * ROBOT_RADIUS + 16)
                   .colliderect(o) for o in obstacles):
                continue
            obstacles.append(rect)
            heights.append(round(rng.uniform(0.35, 0.55), 2))

        obstacles = tuple(obstacles)
        if _reachable(start, goal, obstacles):
            return MapSpec(seed=seed, start=start, goal=goal, obstacles=obstacles,
                           kind="house", heights=tuple(heights))
    return MapSpec(seed=seed, start=(85.0, 120.0, 0.0),
                   goal=(WIDTH - 85.0, HEIGHT - 120.0), obstacles=(), kind="house")


# ---------------------------------------------------------------------------
# Checkpoints and the kind dispatcher.
# ---------------------------------------------------------------------------

def with_checkpoints(spec: MapSpec, n: int = 1) -> MapSpec:
    """Place n checkpoints evenly along the BFS start-to-goal path.

    Checkpoints sit on known-free path cells, so the sequential task is
    solvable whenever the base map is. If the path is too short to space
    them out, the spec is returned unchanged.
    """
    path = _bfs_path(spec.start, spec.goal, spec.obstacles)
    if not path or len(path) < 4 * (n + 1):
        return spec
    checkpoints = []
    for i in range(1, n + 1):
        cx, cy = path[int(len(path) * i / (n + 1))]
        checkpoints.append((cx * GRID_CELL + GRID_CELL / 2,
                            cy * GRID_CELL + GRID_CELL / 2))
    return replace(spec, checkpoints=tuple(checkpoints))


def make_map(seed: int, kind: str = "open", checkpoints: int = 0) -> MapSpec:
    """Generate a map of the given kind, optionally with checkpoints."""
    if kind == "maze":
        spec = generate_maze(seed)
    elif kind == "rooms":
        spec = generate_rooms(seed)
    elif kind == "house":
        spec = generate_house(seed)
    else:
        spec = generate_map(seed)
        # Open maps mix in low obstacles the bee can fly over when the GUI
        # flight mode is on (maze/rooms walls stay full height so corridors
        # keep their meaning). Ignored entirely while flight is off.
        rng = random.Random(seed ^ 0x5EED)
        spec = replace(spec, heights=tuple(
            0.5 if rng.random() < 0.4 else 1.0 for _ in spec.obstacles))
    if checkpoints > 0:
        spec = with_checkpoints(spec, checkpoints)
    return spec
