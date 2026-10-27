#include "lif_controller.hpp"
#include <algorithm>
#include <cmath>

namespace flycore {
LifController::LifController() { reset(); }

void LifController::reset() {
    voltage_.fill(0.0);
    avoid_dir_ = 0;
    steps_since_seen_ = 0;
}

Action LifController::step(const std::array<double, 3>& ranges,
                           bool goal_found, double goal_error) {
    const double left = std::clamp(ranges[0], 0.0, 1.0);
    const double front = std::clamp(ranges[1], 0.0, 1.0);
    const double right = std::clamp(ranges[2], 0.0, 1.0);
    const double ld = 1.0 - left;
    const double fd = 1.0 - front;
    const double rd = 1.0 - right;
    steps_since_seen_ = goal_found ? 0 : steps_since_seen_ + 1;

    // A LEFT turn is excited by danger on the RIGHT (turn away), and vice versa.
    std::array<double, 5> drive = {
        0.85 - 1.15*fd - 0.15*(ld+rd),
        0.05 + 1.25*fd + 0.60*rd - 0.50*ld,
        0.05 + 1.25*fd + 0.60*ld - 0.50*rd,
        // Visual search: unseen goal gradually favors long ballistic runs.
        0.10 + (goal_found ? 0.90 + 0.40*std::abs(goal_error)
                           : std::min(0.60, 0.004 * steps_since_seen_)),
        front < 0.10 ? 1.25*fd + 0.05 : 0.0
    };
    if (goal_found && goal_error < -0.10) {
        drive[1] += 0.65; drive[2] -= 0.35;
    } else if (goal_found && goal_error > 0.10) {
        drive[2] += 0.65; drive[1] -= 0.35;
    }
    if (goal_found && std::abs(goal_error) < 0.12) {
        drive[0] += 0.25; drive[3] -= 0.25;
    }

    for (int i = 0; i < 5; ++i) {
        voltage_[i] = decay_ * voltage_[i] + std::max(0.0, drive[i]) * 0.25;
    }
    // Match the Python controller: score on the PRE-reset membrane potential,
    // and BRAKE (index 4) is excluded from the argmax — it is only ever
    // selected by the explicit safety override below.
    int best = 0;
    double best_score = -1e9;
    for (int i = 0; i < 4; ++i) {
        const double score = voltage_[i] + drive[i] * 0.04;
        if (score > best_score) { best_score = score; best = i; }
    }
    for (int i = 0; i < 5; ++i) {
        if (voltage_[i] >= threshold_) voltage_[i] = 0.0;
    }
    if (front < 0.075) return Action::Brake;
    if (front < 0.23) {
        // Commit to one turn direction until the path clears (no rocking),
        // matching the Python controller.
        if (avoid_dir_ == 0) avoid_dir_ = left >= right ? 1 : 2;
        return avoid_dir_ == 1 ? Action::Left : Action::Right;
    }
    // Hysteresis (matches Python): release only when genuinely clear.
    if (front >= 0.40) avoid_dir_ = 0;
    if (best == 3) {
        if (goal_found && goal_error < -0.10) return Action::Left;
        if (goal_found && goal_error > 0.10) return Action::Right;
        return Action::Forward;
    }
    return static_cast<Action>(best);
}
}

// ---------------------------------------------------------------------------
// C ABI for ctypes: lets Python load the kernel from a shared library and
// compare it against the pure-Python controller on identical inputs.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
#define FLYCORE_EXPORT __declspec(dllexport)
#else
#define FLYCORE_EXPORT
#endif

extern "C" {

FLYCORE_EXPORT void* lif_create() {
    return new flycore::LifController();
}

FLYCORE_EXPORT void lif_destroy(void* handle) {
    delete static_cast<flycore::LifController*>(handle);
}

FLYCORE_EXPORT void lif_reset(void* handle) {
    static_cast<flycore::LifController*>(handle)->reset();
}

FLYCORE_EXPORT int lif_step(void* handle, double left, double front, double right,
                            int goal_found, double goal_error) {
    const std::array<double, 3> ranges{left, front, right};
    return static_cast<int>(
        static_cast<flycore::LifController*>(handle)->step(ranges, goal_found != 0, goal_error));
}

}  // extern "C"
