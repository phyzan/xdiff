/**
 * @file xdiff.hpp
 * @brief XDiff: Advanced automatic differentiation library for computing arbitrary-order partial derivatives.
 *
 * XDiff is a modular, header-only C++20 library that provides compile-time automatic differentiation
 * capabilities for multivariate functions. It computes exact derivatives (not numerical approximations)
 * using dual numbers, expression templates, and lazy evaluation.
 *
 * Key Features:
 * - Arbitrary-order derivatives with compile-time configuration
 * - Multivariate support for functions of any number of independent variables
 * - Two evaluation paths: Direct (Dual) and Lazy (Expression Templates)
 * - Expression simplification and algebraic optimization
 * - GPU support via CUDA, HIP, and SYCL backends
 *
 * @note The number of stored derivatives grows combinatorially as C(Nvars + Norder, Norder),
 *       so high orders with many variables can be memory-intensive.
 *
 * @example
 *     #include <xdiff/xdiff.hpp>
 *
 *     using namespace xdiff;
 *
 *     Symbol<0> x;
 *     Symbol<1> y;
 *
 *     Dual<double, 2, 2> a(x, 3.0);  // a = 3.0, tracking derivatives w.r.t. x
 *     Dual<double, 2, 2> b(y, 2.0);  // b = 2.0, tracking derivatives w.r.t. y
 *
 *     auto f = a * b;
 *     std::cout << f.value() << std::endl;                    // 6.0
 *     std::cout << f.get_diff_wrt(x) << std::endl;            // 2.0 (df/dx)
 *     std::cout << f.get_diff_wrt(y) << std::endl;            // 3.0 (df/dy)
 *     std::cout << f.get_diff_wrt(x, y) << std::endl;         // 1.0 (d²f/dxdy)
 */
#ifndef XDIFF_HPP
#define XDIFF_HPP

// IWYU pragma: begin_exports
#include "src/mathlib/lazy_math.hpp"
// IWYU pragma: end_exports

#endif // XDIFF_HPP