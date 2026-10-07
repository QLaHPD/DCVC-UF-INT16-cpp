// Adapted from SciPy 1.15.3. BSD-3-Clause notice: licenses/SciPy.txt.
#pragma once
#include <cstdint>
namespace uf::transforms::detail {
using npy_intp = std::int64_t;
using npy_double = double;
constexpr int MAX_SPLINE_FILTER_POLES = 2;
enum NI_ExtendMode {
    NI_EXTEND_NEAREST = 0, NI_EXTEND_WRAP = 1, NI_EXTEND_REFLECT = 2,
    NI_EXTEND_MIRROR = 3, NI_EXTEND_CONSTANT = 4,
    NI_EXTEND_GRID_WRAP = 5, NI_EXTEND_GRID_CONSTANT = 6
};
int get_spline_interpolation_weights(double coordinate, int order, double* weights);
int get_filter_poles(int order, int* count, double* poles);
void apply_filter(double* coefficients, npy_intp length, const double* poles, int count, NI_ExtendMode mode);
}  // namespace uf::transforms::detail
