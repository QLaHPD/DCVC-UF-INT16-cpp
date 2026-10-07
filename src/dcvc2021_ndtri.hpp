// Adapted from SciPy 1.15.3's xsf/cephes/ndtri.h and polevl.h.
// Copyright 1984, 1987, 1989 Stephen L. Moshier.
// C++ translation: SciPy developers, 2024.
// SciPy BSD license and Cephes notice: licenses/SciPy.txt.
#pragma once
#include <cstddef>
#include <cmath>
#include <limits>

namespace uf::dcvc2021::image::detail {
template<std::size_t N> inline double polynomial(double x, const double (&coefficients)[N]) {
    double result = coefficients[0];
    for (std::size_t i = 1; i < N; ++i) result = result * x + coefficients[i];
    return result;
}
template<std::size_t N> inline double unit_polynomial(double x, const double (&coefficients)[N]) {
    double result = x + coefficients[0];
    for (std::size_t i = 1; i < N; ++i) result = result * x + coefficients[i];
    return result;
}
inline double normal_quantile(double value) {
    constexpr double p0[] = {-5.99633501014107895267E1, 9.80010754185999661536E1,
        -5.66762857469070293439E1, 1.39312609387279679503E1, -1.23916583867381258016E0};
    constexpr double q0[] = {1.95448858338141759834E0, 4.67627912898881538453E0,
        8.63602421390890590575E1, -2.25462687854119370527E2, 2.00260212380060660359E2,
        -8.20372256168333339912E1, 1.59056225126211695515E1, -1.18331621121330003142E0};
    constexpr double p1[] = {4.05544892305962419923E0, 3.15251094599893866154E1,
        5.71628192246421288162E1, 4.40805073893200834700E1, 1.46849561928858024014E1,
        2.18663306850790267539E0, -1.40256079171354495875E-1, -3.50424626827848203418E-2,
        -8.57456785154685413611E-4};
    constexpr double q1[] = {1.57799883256466749731E1, 4.53907635128879210584E1,
        4.13172038254672030440E1, 1.50425385692907503408E1, 2.50464946208309415979E0,
        -1.42182922854787788574E-1, -3.80806407691578277194E-2, -9.33259480895457427372E-4};
    constexpr double p2[] = {3.23774891776946035970E0, 6.91522889068984211695E0,
        3.93881025292474443415E0, 1.33303460815807542389E0, 2.01485389549179081538E-1,
        1.23716634817820021358E-2, 3.01581553508235416007E-4, 2.65806974686737550832E-6,
        6.23974539184983293730E-9};
    constexpr double q2[] = {6.02427039364742014255E0, 3.67983563856160859403E0,
        1.37702099489081330271E0, 2.16236993594496635890E-1, 1.34204006088543189037E-2,
        3.28014464682127739104E-4, 2.89247864745380683936E-6, 6.79019408009981274425E-9};
    if (value == 0) return -std::numeric_limits<double>::infinity();
    if (value == 1) return std::numeric_limits<double>::infinity();
    if (value < 0 || value > 1) return std::numeric_limits<double>::quiet_NaN();
    bool negative = true;
    if (value > 1.0 - 0.13533528323661269189) { value = 1.0 - value; negative = false; }
    if (value > 0.13533528323661269189) {
        const auto y = value - 0.5, square = y * y;
        return (y + y * (square * polynomial(square, p0) / unit_polynomial(square, q0))) *
               2.50662827463100050242E0;
    }
    const auto x = std::sqrt(-2.0 * std::log(value));
    const auto x0 = x - std::log(x) / x, z = 1.0 / x;
    const auto correction = x < 8.0 ? z * polynomial(z, p1) / unit_polynomial(z, q1)
                                    : z * polynomial(z, p2) / unit_polynomial(z, q2);
    const auto result = x0 - correction;
    return negative ? -result : result;
}
}  // namespace uf::dcvc2021::image::detail
