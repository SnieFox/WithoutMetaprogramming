#ifndef VECTOR_UTILS_HPP
#define VECTOR_UTILS_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

namespace VectorUtils {
    void print_vector(std::ostream& os, const std::string& label, const std::vector<double>& vec, int step_num = -1) {
        if (step_num >= 0) {
            os << "Step " << std::setw(2) << step_num << ": ";
        }
        os << label << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            os << std::fixed << std::setprecision(6) << vec[i];
            if (i < vec.size() - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
}

#endif 