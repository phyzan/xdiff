#ifndef XDIFF_MATH_HPP
#define XDIFF_MATH_HPP

#include <cmath>
#include <lazex/math.hpp>


/**
 * @file math.hpp
 * @brief `XDIFF_USING_MATH` - makes unqualified math calls resolve correctly.
 */
#define XDIFF_USING_MATH                                                       \
    LAZEX_USING_MATH;                                                          \
    using std::abs, std::pow, std::sqrt, std::exp, std::log, std::log10,       \
          std::sin, std::cos, std::tan,                                        \
          std::asin, std::acos, std::atan,                                     \
          std::sinh, std::cosh, std::tanh,                                     \
          std::erf


namespace xdiff{

// So that xdiff's own generic code gets the same overload set it asks users to adopt,
// independently of which headers a translation unit happened to include first.
XDIFF_USING_MATH;

} // namespace xdiff


#endif // XDIFF_MATH_HPP
