#ifndef GRADIENT_ASCENT_OPTIMIZER_HPP
#define GRADIENT_ASCENT_OPTIMIZER_HPP

#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <string>
#include "vector_utils.hpp"
#include "gradient_calculator.hpp"

class StandardGradientAscentOptimizer {
private:
    int total_steps_;

public:
    explicit StandardGradientAscentOptimizer(int total_steps) : total_steps_(total_steps) {
        if (total_steps < 0) {
            throw std::invalid_argument("Total steps must be non-negative.");
        }
    }

    std::vector<double> optimize(
        const MultiVarFunc& func_to_eval,
        const std::vector<double>& initial_point,
        const double fixed_learning_rate) {

        if (fixed_learning_rate <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
        if (initial_point.empty() && total_steps_ > 0) {
            throw std::invalid_argument("Initial point cannot be empty if steps > 0.");
        }
        if (initial_point.empty() && total_steps_ == 0) {
            return {};
        }

        std::vector<double> current_point = initial_point;

        for (int step = 0; step < total_steps_; ++step) {
            double current_func_value = func_to_eval(current_point);

            if (current_point.empty()) {
                break;
            }

            std::vector<double> grad = StandardGradientCalculator::calculate(func_to_eval, current_point);

            if (grad.size() != current_point.size()) {
                throw std::runtime_error("Gradient size mismatch with point size.");
            }

            for (size_t i = 0; i < current_point.size(); ++i) {
                current_point[i] += fixed_learning_rate * grad[i];
            }
        }

        return current_point;
    }
};

#endif