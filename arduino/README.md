# FlyCore Edge on real hardware

The simulation's LIF controller deploys to a physical robot: the decision
logic in `lif_bee.ino` is a line-for-line port of `cpp/lif_controller.cpp`,
which is action-parity-tested against the Python simulation (0 mismatches
over 200,000 steps). The sketch compiles in the Arduino IDE; it has not been
run on hardware as part of this project — treat the `TUNE` constants as
starting points.

## Parts (~$20)

| Part | Qty | Role |
|------|-----|------|
| Arduino Uno/Nano or ESP32 | 1 | runs the LIF controller (~50 Hz) |
| HC-SR04 ultrasonic sensor | 3 | left / front / right range rays |
| LDR photoresistor + 10 kΩ resistor | 2 | "vision": the goal is a bright lamp |
| L298N motor driver | 1 | drives the wheels |
| 2WD robot chassis (2 DC motors + battery holder) | 1 | the bee's body |

## Wiring (matches the pin constants in the sketch)

| Signal | Pin |
|--------|-----|
| Left HC-SR04 TRIG / ECHO | D2 / D3 |
| Front HC-SR04 TRIG / ECHO | D4 / D5 |
| Right HC-SR04 TRIG / ECHO | D6 / D7 |
| L298N ENA / IN1 / IN2 (left motor) | D9 / D8 / D10 |
| L298N ENB / IN3 / IN4 (right motor) | D11 / D12 / D13 |
| Left / right LDR divider | A0 / A1 |

Mount the side sensors angled outward ~40° (the simulation's rays are at
±42°). Each LDR forms a voltage divider with its 10 kΩ resistor between 5V
and GND; aim them slightly left/right of straight ahead.

## How it maps to the simulation

| Simulation | Hardware |
|------------|----------|
| 3 range rays, body-radius inflated, 0..1 | 3 ultrasonic readings / 80 cm, 0..1 |
| Green-goal camera detection + bearing | two LDRs; lamp brightness sum = "found", left/right difference = bearing |
| FORWARD / SEEK | both motors forward |
| LEFT / RIGHT | spin in place (differential) |
| BRAKE | motors off |
| Safety overrides outside the network | identical thresholds (0.075 / 0.23) |

The LIF parameters (decay 0.78, threshold 0.78, side gain 0.60, search gain
0.004) are the Kaggle-tuned simulation defaults, unchanged.

## Raspberry Pi instead

A Pi runs the *Python* package directly — no port needed:

```bash
sudo apt install python3-pip
pip3 install numpy            # simulation core needs numpy only
git clone <your repo> && cd flycore-edge
python3 -m experiments.run_experiments --episodes 10   # headless check
```

For a physical Pi robot, replace `flycore/sensors.py`'s ray casts with GPIO
ultrasonic reads and `flycore/vision.py` with a Pi Camera frame (the
`detect_green_goal()` function already takes any H×W×3 RGB array — point it
at a green object and it just works), then feed `LIFController.decide()`
exactly as `main.py` does. The controller needs nothing but three ranges and
a bearing.

## Drones

A multirotor's flight controller (PX4 / Betaflight) already stabilizes
attitude and holds altitude; a navigation layer on top mostly commands yaw
and forward velocity — exactly this controller's action set. The 2D world is
the top-down view of flight at constant height. Porting = map FORWARD/LEFT/
RIGHT/BRAKE to velocity setpoints over MAVLink (e.g. `pymavlink` on a
companion Pi). Full 3D maneuvering and 360° vision are future work.
