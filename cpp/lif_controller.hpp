#pragma once
#include <array>

namespace flycore {
enum class Action : int { Forward = 0, Left = 1, Right = 2, Seek = 3, Brake = 4 };

class LifController {
public:
    LifController();
    void reset();
    Action step(const std::array<double, 3>& ranges,
                bool goal_found, double goal_error);
private:
    std::array<double, 5> voltage_{};
    // Defaults match the Python controller's Kaggle-tuned parameters.
    double decay_ = 0.78;
    double threshold_ = 0.78;
    int avoid_dir_ = 0;  // 0 = none, 1 = Left, 2 = Right (turn commitment)
    long steps_since_seen_ = 0;  // visual-search drive for the SEEK neuron
};
}
