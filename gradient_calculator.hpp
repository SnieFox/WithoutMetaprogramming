#ifndef GRADIENT_CALCULATOR_HPP
#define GRADIENT_CALCULATOR_HPP

#include <vector>
#include <stdexcept>
#include <functional>

#ifndef M_PI
#define M_PI (std::acos(-1.0))
#endif

using MultiVarFunc = std::function<double(const std::vector<double>&)>;

class StandardGradientCalculator {
public:
    static std::vector<double> calculate(
        const MultiVarFunc& func_to_eval,
        const std::vector<double>& point,
        double h = 1e-5) {

        if (point.empty()) {
            throw std::invalid_argument("Input point vector cannot be empty.");
        }
        if (h <= 0.0) {
            throw std::invalid_argument("Step size h for finite difference must be positive.");
        }

        size_t num_dimensions = point.size();
        std::vector<double> gradient(num_dimensions);
        std::vector<double> perturbed_point = point;

        for (size_t i = 0; i < num_dimensions; ++i) {
            double original_value_at_i = point[i];

            perturbed_point[i] = original_value_at_i + h;
            double f_plus_h = func_to_eval(perturbed_point);

            perturbed_point[i] = original_value_at_i - h;
            double f_minus_h = func_to_eval(perturbed_point);

            gradient[i] = (f_plus_h - f_minus_h) / (2.0 * h);

            perturbed_point[i] = original_value_at_i;
        }
        return gradient;
    }
};

#endif