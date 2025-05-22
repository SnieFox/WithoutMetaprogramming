#ifndef GRADIENT_ASCENT_DYN_HPP
#define GRADIENT_ASCENT_DYN_HPP

#include <vector>
#include <functional>
#include <string>

// Define PI for trigonometric functions if M_PI is not available
#ifndef M_PI
#define M_PI (std::acos(-1.0))
#endif

// Type alias for a multivariable scalar function
using MultiVarFunc = std::function<double(const std::vector<double>&)>;

// Enum to represent different stepping strategies
enum class StepStrategy {
    NORMAL,
    CAUTIOUS,
    BOLD
};

// Forward declarations
std::vector<double> calculate_gradient(MultiVarFunc func, const std::vector<double>& point, double h);
void print_vector_detail_dyn(const std::string& label, const std::vector<double>& vec, int step_num = -1, StepStrategy current_strategy = StepStrategy::NORMAL);
std::vector<double> perform_gradient_steps_dyn_strategy(
    MultiVarFunc func,
    std::vector<double> current_point,
    double learning_rate,
    int steps_remaining,
    StepStrategy current_strategy,
    double prev_func_value,
    int current_step_number = 0
);

#endif // GRADIENT_ASCENT_DYN_HPP