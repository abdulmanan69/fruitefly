# FlyCore Edge — Project Report

**Project:** Brain-inspired visual navigation in a safe 2D simulation
**Date:** 26 October 2026

---

## 1. Abstract

FlyCore Edge is a 2D robotics simulation in which a small leaky integrate-and-fire (LIF) neural controller navigates a robot to a visually detected goal while avoiding obstacles. This work extends the first prototype into a complete experimental system: procedurally generated solvable worlds, a rule-based baseline controller, a headless measurement pipeline (goal success, collisions, decision latency, memory), cloud-scale experiments on Kaggle, and a C++ implementation of the LIF kernel verified action-for-action against the Python controller. Along the way, three defects in the original prototype were diagnosed from experiment data and fixed, and a Kaggle hyperparameter search tuned the controller — together raising the LIF controller's success rate on randomized maps from 0% to 82% and eliminating collisions entirely, while keeping its paths ~20% shorter than the rule-based baseline's.

## 2. System overview

The simulation core (`flycore/`) is pure Python + numpy and runs headless; pygame is used only by the GUI (`main.py`, `flycore/render.py`). One episode works as follows:

1. The environment renders a forward-facing pseudo-camera frame (numpy array).
2. `vision.py` detects the green goal marker and estimates a horizontal bearing error.
3. `sensors.py` casts three range rays (left/front/right), inflated by the robot's body radius.
4. A controller maps (ranges, vision) to one of five actions: FORWARD, LEFT, RIGHT, SEEK, BRAKE.
5. The environment executes the action; collisions are blocked, never penetrated.

Two controllers share this interface:

- **LIF controller** — five LIF output neurons integrate sensory drive; the strongest pre-spike membrane potential picks the action. Explicit safety rules (braking, committed avoidance turns) remain outside the network. A *visual search* term makes long straight runs increasingly attractive while the goal is unseen.
- **Rule-based baseline** — fixed-priority reactive rules (brake → committed avoid turn → side clearance → goal steering → cruise). Deliberately simple and stateless apart from turn commitment.

Both controllers are wrapped by the same **reflex layer** (`reflex.py`) with two fly-inspired recovery behaviors: an **escape saccade** (net displacement over 70 steps below 7 px → stereotyped turn-and-push), and an **exploration saccade** driven by *novelty* — positions are tracked on a coarse 60 px grid, and if a 420-step window discovers almost no new cells while the goal is out of view, a longer reorienting push breaks the cycle (orbiting a block, bouncing between the same walls). Novelty-gating matters: an earlier fixed-period version interrupted perfectly good long approaches; the novelty version never fires during productive straight runs (0.13 escapes/episode on open maps). Successive saccades turn longer and alternate direction, so repeated escapes sample genuinely new headings. A third behavior adds short-term target memory: a **re-acquire saccade** — when the goal slides out the side of the view, the wrapper turns back toward its last seen azimuth once per sighting, mirroring documented fly pursuit behavior. Applying the identical reflex to both sides keeps the comparison fair.

**Worlds** (`maps.py`) are generated from a seed in four kinds: **open** (scattered bars and blocks), **maze** (a recursive-backtracker maze, *braided* by removing ~18% of internal walls so corridors form loops — a perfect maze would punish any memoryless reactive controller with unbounded dead-end backtracking), **rooms** (full-height dividers with door gaps), and **house** (a four-room floor plan with doorways, low furniture, and a staircase of ascending blocks; obstacle *heights* enable the GUI's flight mode, where an altitude autopilot — outside the network, like the safety rules — climbs the bee over low geometry). In every kind a grid BFS (robot-radius-inflated) proves start→goal reachability before a map is accepted; the same seed reproduces the same world on any machine. Maps may also carry **checkpoints**: waypoints placed on the BFS path that must be visited in order. Only the *active* target is rendered green — and the color-based vision detects only green — so a checkpoint map becomes a sequential puzzle with **zero changes to either controller**.

**GUI** (`render.py`, `view3d.py`): beside the top-down world (with path trail and pulsing targets), a sidebar shows (i) a raycast pseudo-3D first-person view — per-column exact ray/AABB slab intersections, distance shading, darker horizontal faces, stripe texture, and depth-tested glowing target billboards; (ii) the robot's camera frame, i.e. the controller's actual vision input; and (iii) live LIF membrane potentials per output neuron against the spike-threshold line. Maps are editable live with the mouse (drag to draw a wall, right-click to remove one).

## 3. Defects found and fixed (engineering iteration)

Batch experiments on randomized maps exposed three defects that the single hand-designed demo map had masked:

| # | Defect | Symptom | Fix |
|---|--------|---------|-----|
| 1 | Range rays tested a point (radius 2) while the robot body has radius 13 | Robot clipped corners its sensors never reported; ~1,700 blocked collisions per episode | Rays test clearance at full body radius |
| 2 | Action argmax read the membrane potential *after* post-spike reset | The winning FORWARD neuron periodically lost the argmax, producing a drift into circles | Score on pre-reset potential; reset afterwards |
| 3 | LEFT/RIGHT neurons were excited by danger on their *own* side | The network steered **into** obstacles; the safety override then fought it, causing stall loops | Turn-away wiring: LEFT is excited by right-side danger, and vice versa |
| 4 | The camera projected the goal with **no line-of-sight check** (spotted by user observation in the GUI) | The robot "saw" the goal through walls and pressed against the wall in between instead of exploring | Sight-line march over tall obstacles before the goal is drawn; low (flyable) furniture does not hide it |
| 5 | Turn commitment released as soon as the front ray cleared | In room corners, wall #2 re-triggered avoidance with the *opposite* direction — rocking in place below the stall detector's threshold | Hysteresis (release at 0.40, not 0.23) + a medium-window drift detector + escapes that keep rotating while a wall is still ahead |

Two behavioral additions completed the controller: turn **commitment** during avoidance (re-picking the turn direction every step caused corner rocking) and the **visual search** drive (failures were dominated by blind wandering that never brought the goal into view).

Progression on the same 10–15 random maps: LIF 0% → 10% → 40% → **80%** success; baseline 10% → **100%** after the shared sensor fix and reflex.

## 4. Local experiment results

60 randomized maps (seeds 1–60), both controllers, max 2,500 steps per episode, identical reflex wrapper. Machine: Windows 11, Python 3.14. LIF uses the Kaggle-tuned default parameters (§5.2); with the original hand-tuned parameters its success rate was 68.3%.

| metric | baseline | LIF (tuned) |
|---|---|---|
| episodes | 60 | 60 |
| goal success rate | **93.3%** | 81.7% |
| blocked collisions / episode | 0.00 | 0.00 |
| steps to goal (successes) | 1,252 | **1,047** |
| path length (successes) | 2,332 px | **1,854 px** |
| mean decision latency | **4.8 µs** | 53.1 µs |
| p95 decision latency | 12.5 µs | 98.0 µs |
| mean vision latency | 0.44 ms | 0.45 ms |
| peak Python memory / episode | 821 KB | 820 KB |
| reflex escapes / episode | 0.37 | 0.53 |

Figures: `results/plots/success_rate.png`, `steps_to_goal.png`, `decision_latency.png`.

### 4.1 Generalization across map kinds

15 maps per kind, both controllers, final reflex layer (stall + novelty exploration + re-acquire memory). Mazes get a 6,000-step budget (longer solution paths); the rest get 2,500.

Numbers below are with the final physics (line-of-sight vision, defect #4) and the final reflex layer; they supersede interim sweeps run while those fixes were being developed.

| goal success | baseline | **LIF** |
|---|---|---|
| open | **86.7%** | 80.0% |
| rooms | 13.3% | **40.0%** (3×) |
| maze | 26.7% | **46.7%** |
| house | 0.0% | **46.7%** |

| steps to goal (successes) | baseline | **LIF** |
|---|---|---|
| open | 1,176 | **877** |
| maze | 2,505 | **1,387** |
| house | — (no successes) | 1,356 |

Blocked collisions are **0.00 per episode for both controllers on every kind**.

**Reading.** With honest occlusion the division of labor is sharp: the rule-based baseline is marginally more reliable only on open fields (86.7% vs 80.0%), while the LIF controller wins **every structured world** — 3× on rooms, nearly 2× on mazes, and 46.7% vs **zero** baseline successes in the house floor plan. The line-of-sight fix explains the flip: the baseline's old successes partly rode on physically impossible through-wall goal sightings; once vision became honest, only the LIF controller's search behaviors (visual-search drive, novelty escapes, re-acquire memory) reach goals hidden in other rooms. LIF is also consistently faster to the goal where both succeed. Structured worlds remain hard for *any* memoryless reactive controller — the honest gap that §8's memory-based future work targets. Raw data: `results/results_open_check.csv`, `results_rooms.csv`, `results_maze.csv`, `results_house.csv`.

One rooms-generator iteration is worth recording: with a single ~100 px door per divider, *both* controllers scored 0–27% and the baseline never passed a single divider in 2,500 steps (trajectory tracing showed max-x pinned exactly at the first divider). Two wider doors per divider made the task solvable-by-exploration rather than solvable-by-luck.

**Reading.** The baseline remains more *reliable* (93% vs 82%): its rigid wall-hugging rules systematically traverse the world until the goal appears. The LIF controller is more *efficient* — it reaches the goal in 16% fewer steps over 20% shorter paths — because membrane dynamics smooth its trajectories instead of bouncing between rules. Neither controller ever collides: safety never depends on the network. LIF decisions cost ~10× the baseline's (still only ~0.05 ms, far below the 16.7 ms frame budget); vision dominates per-step cost for both.

## 4.2 Perception stack: detection, Kalman filtering, Hungarian association, MOT

`flycore/perception.py` implements the classic SORT tracking pipeline from scratch (no OpenCV/scipy in the core): per-frame detections (connected-component color segmentation over the camera frame — the honest equivalent of a detector for this synthetic world) feed per-track **constant-velocity Kalman filters**; predicted boxes are associated to detections by the **Hungarian algorithm** on an IoU cost matrix; a track lifecycle (min-hits confirmation, max-age deletion) maintains **persistent object identities**. Unit tests verify the Hungarian optimum on known matrices, Kalman velocity estimation, ID persistence, and occlusion behavior.

Used as a vision backend (`TrackerVision`, GUI key **O**), the tracker adds one measurable capability over the raw color filter: when the goal detection drops out for a few frames (sliding behind an obstacle edge), the goal track's **Kalman prediction keeps supplying a bearing**, so the controller does not instantly forget a briefly-occluded target; after ~8 undetected frames the track dies honestly.

**YOLO11** (`flycore/vision_yolo.py`) is a third, learned backend: a Kaggle GPU kernel (`kaggle_run/build_yolo_kernel.py`) synthesizes a labeled dataset from the simulator itself (camera frames + pixel-perfect goal boxes derived from the green mask — 1,650 images across open/rooms/house maps, 363 containing the goal, the rest serving as backgrounds) and fine-tunes `yolo11n` for 25 epochs; the trained weights drop in at `models/yolo11_goal.pt` (GUI key **Y**). A COCO-pretrained YOLO is deliberately not used as a fallback — it knows nothing about this synthetic world.

Trained results (kernel `abdulmananxx/flycore-yolo-train`, completed 2026-10-27; raw outputs incl. PR curves and confusion matrix in `results/kaggle_yolo/`):

| metric | value |
|---|---|
| mAP@50 (held-out val) | **0.995** |
| mAP@50-95 | 0.965 |
| false positives on goal-free frames (local spot check) | none observed |
| local CPU inference | 24.6 ms/frame (vs 0.5 ms for the color filter) |

The accuracy-vs-cost tradeoff is the honest story: on this synthetic world the color filter is already near-perfect and ~50× cheaper, so YOLO11's value is architectural — the vision interface is proven swappable for a learned detector, which is exactly what a real-camera deployment would need. OpenCV renders any episode to MP4 for visual analytics (`experiments/record_video.py`; demos in `results/videos/`).

## 5. Kaggle cloud experiments

The entire project is bundled by `kaggle_run/build_kernel.py` into a single self-contained script kernel (source embedded as a base64 zip; no internet or datasets required on Kaggle) and executed on Kaggle CPU via the Kaggle API (`kaggle_run/run_on_kaggle.py`). The kernel runs four phases:

1. **Comparison sweep** — 150 maps × {LIF, baseline}
2. **Grid search** — 24 LIF hyperparameter configurations (decay × threshold × side-gain × search-gain) × 25 maps
3. **C++ kernel** — g++ build, 200,000-step action parity check against Python, speed benchmark
4. **Validation** — best grid configuration vs default LIF on 100 held-out maps

> Kernel: https://www.kaggle.com/code/abdulmananxx/flycore-edge-experiments (completed 2026-10-27; raw outputs in `results/kaggle/`, figures in `results/plots_kaggle/`)

### 5.1 Comparison sweep (150 maps)

| metric | baseline | LIF (pre-tuning defaults) |
|---|---|---|
| goal success rate | **88.7%** | 68.0% |
| blocked collisions / episode | 0.00 | 0.00 |
| steps to goal (successes) | **1,209** | 1,232 |
| path length (successes) | 2,258 px | **1,995 px** |
| mean decision latency | **13.9 µs** | 95.8 µs |

The pattern matches the local 60-map sweep (§4): the baseline is more reliable, the LIF controller travels shorter paths, and neither ever collides. Absolute latencies are higher than local ones (slower Kaggle CPU); ratios are consistent.

### 5.2 Hyperparameter grid search (24 configs × 25 maps)

Best configuration: **decay 0.78, threshold 0.78, side_gain 0.60, search_gain 0.004** — 88% success and the fastest goal-reaching (903 steps) on the search maps. Lower decay (shorter membrane memory) and weaker side-danger gain consistently outperformed the hand-tuned values; a stronger visual-search ramp helped on sparse maps.

### 5.3 Validation on 100 held-out maps

| metric | LIF default | LIF tuned |
|---|---|---|
| goal success rate | 67.0% | **83.0%** |
| steps to goal (successes) | 1,134 | **1,066** |
| blocked collisions / episode | 0.00 | 0.00 |

The tuned configuration generalizes: **+16 percentage points** success on maps never seen during the search. These parameters are now the defaults in `flycore/controller.py` and `cpp/lif_controller.*`, closing most of the gap to the baseline while keeping the LIF controller's shorter paths.

### 5.4 C++ kernel (Kaggle g++ build)

| check | result |
|---|---|
| action parity vs Python, 200,000 identical steps | **0 mismatches** |
| Python LIF step | 3,465 ns |
| C++ LIF step (via ctypes) | 2,099 ns |
| speedup | **1.65×** (FFI overhead included) |

## 6. C++ optimization path

`cpp/lif_controller.cpp` mirrors the Python LIF controller exactly (including the three fixes above, which were ported line-for-line) and exposes a C ABI consumed through ctypes (`flycore/cpp_bridge.py`). The methodology is parity-first: `experiments/bench_cpp.py` must find **zero** action mismatches over 200,000 identical input steps before any speed numbers are reported. A dedicated unit test (`TestCppParity`) repeats a 5,000-step parity check whenever the library is present.

## 7. Threats to validity and limitations

- The camera is a schematic projection, not rendered 3D; vision results do not transfer to real imagery.
- The LIF model is engineering-inspired, not a biologically validated fly circuit.
- Map generation, while randomized and solvability-checked, samples one obstacle style; results may differ for mazes or concave traps.
- The ctypes benchmark includes per-call FFI overhead; the kernel-only speedup would be larger with a batched API.
- One machine and one Python version (3.14) for local numbers; Kaggle numbers come from a different CPU — absolute latencies are not comparable across the two, ratios are.

## 8. Future work

- Close the reliability gap: add the baseline's systematic coverage as a low-priority LIF drive, or train the drive weights (the controller is already fully parameterized).
- Hardware deployment: an Arduino/ESP32 port of the (parity-tested) C++ kernel ships in `arduino/` — three ultrasonic sensors stand in for the range rays and two photoresistors for vision; running it on a physical chassis and measuring real-world success is the natural next step. The same action set maps to drone yaw/velocity setpoints (the simulation models constant-altitude planar flight).
- Wire the C++ kernel into the GUI loop behind a feature flag.
- Replace color-threshold vision with a tiny CNN trained on rendered frames (Kaggle GPU phase).
- Multi-goal / moving-goal tasks; curriculum over obstacle density.

## 9. Reproducibility

```bash
python -m unittest discover -s tests -v                 # 18 tests
python -m experiments.run_experiments --episodes 60     # local sweep
python -m experiments.analyze --csv results/results.csv # figures
python kaggle_run/run_on_kaggle.py --username <you>     # cloud suite
```

All maps are seed-deterministic; every number in §4 regenerates from the commands above.
