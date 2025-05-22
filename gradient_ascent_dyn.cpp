#include "gradient_ascent_dyn.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>

void print_vector_detail_dyn(const std::string& label, const std::vector<double>& vec, int step_num, StepStrategy current_strategy) {
    if (step_num >= 0) {
        std::cout << "Step " << std::setw(2) << step_num << ": ";
    }
    std::cout << label << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";

    if (step_num >= 0) {
        std::string strat_name = "NORMAL";
        if (current_strategy == StepStrategy::CAUTIOUS) strat_name = "CAUTIOUS";
        else if (current_strategy == StepStrategy::BOLD) strat_name = "BOLD";
        std::cout << " (Strategy: " << strat_name << ")";
    }
}

std::vector<double> calculate_gradient(MultiVarFunc func, const std::vector<double>& point, double h) {
    if (point.empty()) {
        throw std::invalid_argument("Input point vector cannot be empty.");
    }
    if (h <= 0.0) {
        throw std::invalid_argument("Step size h must be positive.");
    }
    size_t num_variables = point.size();
    std::vector<double> gradient(num_variables);
    std::vector<double> perturbed_point = point;

    for (size_t i = 0; i < num_variables; ++i) {
        double original_value_at_i = perturbed_point[i];
        perturbed_point[i] = original_value_at_i + h;
        double f_plus_h = func(perturbed_point);
        perturbed_point[i] = original_value_at_i - h;
        double f_minus_h = func(perturbed_point);
        gradient[i] = (f_plus_h - f_minus_h) / (2.0 * h);
        perturbed_point[i] = original_value_at_i;
    }
    return gradient;
}

std::vector<double> perform_gradient_steps_dyn_strategy(
    MultiVarFunc func,
    std::vector<double> current_point,
    double learning_rate,
    int steps_remaining,
    StepStrategy current_strategy,
    double prev_func_value,
    int current_step_number
) {
    double current_func_value = func(current_point);

    // Print the current state
    print_vector_detail_dyn("Point: ", current_point, current_step_number, current_strategy);
    std::cout << ", f(p): " << std::fixed << std::setprecision(6) << current_func_value << std::endl;

    // Base Case: If no steps are remaining, return the current point.
    if (steps_remaining <= 0) {
        std::cout << "Reached maximum steps. Final point." << std::endl;
        return current_point;
    }

    // 1. Calculate the gradient at the current point.
    std::vector<double> grad = calculate_gradient(func, current_point, 1e-5);

    // 2. Apply current strategy to determine step
    std::vector<double> next_point = current_point;
    double effective_learning_rate = learning_rate;

    switch (current_strategy) {
    case StepStrategy::NORMAL:
        effective_learning_rate = learning_rate;
        break;
    case StepStrategy::CAUTIOUS:
        effective_learning_rate = learning_rate * 0.5;
        break;
    case StepStrategy::BOLD:
        effective_learning_rate = learning_rate * 1.5;
        break;
    }

    for (size_t i = 0; i < current_point.size(); ++i) {
        next_point[i] += effective_learning_rate * grad[i];
    }

    // Dynamic Strategy Switching
    StepStrategy next_strategy = current_strategy;
    double improvement = current_func_value - prev_func_value;

    if (current_step_number > 0) {
        if (current_strategy == StepStrategy::NORMAL) {
            if (current_step_number % 5 == 0 && std::abs(improvement) < 1e-3) {
                next_strategy = StepStrategy::BOLD;
                std::cout << "Switching strategy: NORMAL -> BOLD (slow progress)" << std::endl;
            }
        }
        else if (current_strategy == StepStrategy::BOLD) {
            if (improvement < 0) {
                next_strategy = StepStrategy::CAUTIOUS;
                std::cout << "Switching strategy: BOLD -> CAUTIOUS (overshot)" << std::endl;
            }
            else if (current_step_number % 3 == 0) {
                next_strategy = StepStrategy::NORMAL;
                std::cout << "Switching strategy: BOLD -> NORMAL (duration)" << std::endl;
            }
        }
        else if (current_strategy == StepStrategy::CAUTIOUS) {
            if (current_step_number % 3 == 0 || improvement > 1e-2) {
                next_strategy = StepStrategy::NORMAL;
                std::cout << "Switching strategy: CAUTIOUS -> NORMAL (duration or good progress)" << std::endl;
            }
        }
    }

    // Recursive Call
    return perform_gradient_steps_dyn_strategy(
        func,
        next_point,
        learning_rate,
        steps_remaining - 1,
        next_strategy,
        current_func_value,
        current_step_number + 1
    );
}