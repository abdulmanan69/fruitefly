/*
 * FlyCore Edge on real hardware: the LIF bee controller for Arduino / ESP32.
 *
 * Mirrors cpp/lif_controller.cpp (which is action-parity-tested against the
 * Python simulation over 200,000 steps). The robot:
 *   - senses left/front/right clearance with three HC-SR04 ultrasonic sensors
 *   - "sees" the goal with two LDR photoresistors (goal = a bright lamp;
 *     the brightness difference gives the bearing, like the camera in sim)
 *   - drives two DC motors through an L298N driver
 *
 * Safety rules (brake, committed avoidance turn) stay OUTSIDE the network,
 * exactly as in the simulation. This sketch compiles in the Arduino IDE for
 * Uno/Nano/ESP32; it has not been run on hardware here - calibrate the
 * constants marked TUNE for your chassis.
 *
 * Wiring (see arduino/README.md for the table):
 *   HC-SR04 x3: TRIG/ECHO pins below, VCC 5V, GND
 *   L298N: IN1..IN4 + ENA/ENB (PWM), motor battery on the driver
 *   LDR x2: voltage dividers into A0 (left) and A1 (right)
 */

// ----- pins (TUNE to your wiring) -------------------------------------------
const int TRIG_L = 2, ECHO_L = 3;
const int TRIG_F = 4, ECHO_F = 5;
const int TRIG_R = 6, ECHO_R = 7;
const int ENA = 9,  IN1 = 8,  IN2 = 10;   // left motor
const int ENB = 11, IN3 = 12, IN4 = 13;   // right motor
const int LDR_L = A0, LDR_R = A1;

// ----- behavior constants (match the simulation / TUNE) ---------------------
const float MAX_RANGE_CM = 80.0;     // ultrasonic reading treated as "fully clear"
const int   LIGHT_FOUND = 650;       // LDR sum above this = goal (lamp) in view  TUNE
const int   DRIVE_PWM = 170;         // cruise speed 0..255                       TUNE
const int   TURN_PWM = 150;          //                                           TUNE
const unsigned long STEP_MS = 20;    // ~50 Hz control loop

// LIF parameters: identical to the Kaggle-tuned simulation defaults.
const float DECAY = 0.78, THRESHOLD = 0.78, SIDE_GAIN = 0.60;
const float FORWARD_DRIVE = 0.85, SEARCH_GAIN = 0.004, SEARCH_CAP = 0.60;

enum Action { FORWARD = 0, LEFT = 1, RIGHT = 2, SEEK = 3, BRAKE = 4 };

float v[5] = {0, 0, 0, 0, 0};   // membrane potentials
int avoidDir = 0;               // 0 none, 1 left, 2 right (turn commitment)
long stepsSinceSeen = 0;        // visual-search drive

float readRange(int trig, int echo) {
  digitalWrite(trig, LOW); delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, 12000UL);   // ~2 m timeout
  if (us == 0) return 1.0;                           // no echo = clear
  float cm = us / 58.0;
  float norm = cm / MAX_RANGE_CM;
  return norm > 1.0 ? 1.0 : norm;                    // 0 close .. 1 clear
}

// The exact decision logic of LifController::step() in cpp/lif_controller.cpp.
Action decide(float left, float front, float right, bool goalFound, float goalError) {
  float ld = 1.0 - left, fd = 1.0 - front, rd = 1.0 - right;
  stepsSinceSeen = goalFound ? 0 : stepsSinceSeen + 1;

  // Turn-away wiring: LEFT is excited by danger on the RIGHT, and vice versa.
  float searchRamp = SEARCH_GAIN * stepsSinceSeen;
  if (searchRamp > SEARCH_CAP) searchRamp = SEARCH_CAP;
  float drive[5] = {
    FORWARD_DRIVE - 1.15f * fd - 0.15f * (ld + rd),
    0.05f + 1.25f * fd + SIDE_GAIN * rd - 0.50f * ld,
    0.05f + 1.25f * fd + SIDE_GAIN * ld - 0.50f * rd,
    0.10f + (goalFound ? 0.90f + 0.40f * fabs(goalError) : searchRamp),
    front < 0.10f ? 1.25f * fd + 0.05f : 0.0f
  };
  if (goalFound && goalError < -0.10f) { drive[1] += 0.65f; drive[2] -= 0.35f; }
  else if (goalFound && goalError > 0.10f) { drive[2] += 0.65f; drive[1] -= 0.35f; }
  if (goalFound && fabs(goalError) < 0.12f) { drive[0] += 0.25f; drive[3] -= 0.25f; }

  for (int i = 0; i < 5; i++) {
    float d = drive[i] > 0 ? drive[i] : 0;
    v[i] = DECAY * v[i] + d * 0.25f;
  }
  // Score on the PRE-reset potential; BRAKE is excluded from the argmax.
  int best = 0; float bestScore = -1e9;
  for (int i = 0; i < 4; i++) {
    float s = v[i] + drive[i] * 0.04f;
    if (s > bestScore) { bestScore = s; best = i; }
  }
  for (int i = 0; i < 5; i++) if (v[i] >= THRESHOLD) v[i] = 0.0f;

  // Explicit safety rules, outside the network.
  if (front < 0.075f) return BRAKE;
  if (front < 0.23f) {
    if (avoidDir == 0) avoidDir = (left >= right) ? 1 : 2;
    return avoidDir == 1 ? LEFT : RIGHT;
  }
  // Hysteresis (matches the simulation): release only when genuinely clear.
  if (front >= 0.40f) avoidDir = 0;
  if (best == 3) {
    if (goalFound && goalError < -0.10f) return LEFT;
    if (goalFound && goalError > 0.10f) return RIGHT;
    return FORWARD;
  }
  return (Action)best;
}

void motors(int lpwm, bool lfwd, int rpwm, bool rfwd) {
  digitalWrite(IN1, lfwd ? HIGH : LOW); digitalWrite(IN2, lfwd ? LOW : HIGH);
  digitalWrite(IN3, rfwd ? HIGH : LOW); digitalWrite(IN4, rfwd ? LOW : HIGH);
  analogWrite(ENA, lpwm); analogWrite(ENB, rpwm);
}

void act(Action a) {
  switch (a) {
    case FORWARD:
    case SEEK:  motors(DRIVE_PWM, true, DRIVE_PWM, true); break;
    case LEFT:  motors(TURN_PWM, false, TURN_PWM, true); break;   // spin left
    case RIGHT: motors(TURN_PWM, true, TURN_PWM, false); break;   // spin right
    case BRAKE: motors(0, true, 0, true); break;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_L, OUTPUT); pinMode(ECHO_L, INPUT);
  pinMode(TRIG_F, OUTPUT); pinMode(ECHO_F, INPUT);
  pinMode(TRIG_R, OUTPUT); pinMode(ECHO_R, INPUT);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
}

void loop() {
  unsigned long t0 = millis();

  float left = readRange(TRIG_L, ECHO_L);
  float front = readRange(TRIG_F, ECHO_F);
  float right = readRange(TRIG_R, ECHO_R);

  // "Vision": the goal is the brightest lamp in the room.
  int ll = analogRead(LDR_L), lr = analogRead(LDR_R);
  bool goalFound = (ll + lr) > LIGHT_FOUND;
  float goalError = goalFound ? (float)(lr - ll) / (float)(ll + lr + 1) : 0.0f;

  Action a = decide(left, front, right, goalFound, goalError);
  act(a);

  Serial.print("L/F/R "); Serial.print(left); Serial.print(" ");
  Serial.print(front); Serial.print(" "); Serial.print(right);
  Serial.print("  goal "); Serial.print(goalFound);
  Serial.print(" err "); Serial.print(goalError);
  Serial.print("  action "); Serial.println((int)a);

  unsigned long dt = millis() - t0;
  if (dt < STEP_MS) delay(STEP_MS - dt);
}
