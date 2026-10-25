#ifndef XDIFF_DUAL_DECLS_HPP
#define XDIFF_DUAL_DECLS_HPP

#include <ostream>
#include "macros.hpp"
#include "../rules/math.hpp" // IWYU pragma: keep


namespace xdiff::detail{

// Forward declarations of main types
template<typename T, typename GD>
class dual_template;

template<typename T, typename GD, int Order>
class Dual;

template<typename T, typename GD, int Order>
class Seed;

template<typename T, typename GD, int Order>
using SeedState = GD::template state_type<Seed<T, GD, Order>>;

template<typename T, typename GD, int Order>
using DualState = GD::template state_type<Dual<T, GD, Order>>;

template<typename T, typename GD>
using State = GD::template state_type<T>;

template<typename T, typename GD, int Order>
SeedState<T, GD, Order> make_seeds(const State<T, GD>& state);

template<typename U, typename T>
concept converts_to = std::constructible_from<T, U>;

// -------------- Nesting of duals --------------
template<typename T, typename GD, int Order>
struct dual_type_getter{
    static_assert(Order > 0, "Differentiation order must be positive");
    using type = dual_template<Dual<T, GD, Order-1>, GD>;
};

template<typename T, typename GD>
struct dual_type_getter<T, GD, 1>{
    using type = dual_template<T, GD>;
};
// ----------------------------------------------


// The operations live next to Dual, where argument-dependent lookup finds them, the rules' own calls included.
// ----------------------  Unary operations ------------------------



// ------------- Unary operation declarations -------------
XDIFF_DECLARE_UNARY(operator+,  assign_pos)
XDIFF_DECLARE_UNARY(operator-,  assign_neg)
XDIFF_DECLARE_UNARY(abs,        assign_abs)
XDIFF_DECLARE_UNARY(log,        assign_log)
XDIFF_DECLARE_UNARY(log10,      assign_log10)
XDIFF_DECLARE_UNARY(exp,        assign_exp)
XDIFF_DECLARE_UNARY(sqrt,       assign_sqrt)
XDIFF_DECLARE_UNARY(sin,        assign_sin)
XDIFF_DECLARE_UNARY(cos,        assign_cos)
XDIFF_DECLARE_UNARY(tan,        assign_tan)
XDIFF_DECLARE_UNARY(cot,        assign_cot)
XDIFF_DECLARE_UNARY(sec,        assign_sec)
XDIFF_DECLARE_UNARY(csc,        assign_csc)
XDIFF_DECLARE_UNARY(asin,       assign_asin)
XDIFF_DECLARE_UNARY(acos,       assign_acos)
XDIFF_DECLARE_UNARY(atan,       assign_atan)
XDIFF_DECLARE_UNARY(asec,       assign_asec)
XDIFF_DECLARE_UNARY(acsc,       assign_acsc)
XDIFF_DECLARE_UNARY(sinh,       assign_sinh)
XDIFF_DECLARE_UNARY(cosh,       assign_cosh)
XDIFF_DECLARE_UNARY(tanh,       assign_tanh)
XDIFF_DECLARE_UNARY(erf,        assign_erf)

template<int P, typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_power(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/);
template<int P, typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_power(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/);

// returning auto to allow for lazy operations
template<int P, typename T, typename GD, int Order>
XDIFF_HOST_DEVICE auto power(const XDIFF_DUAL& /*arg*/);
template<int P, typename T, typename GD, int Order>
XDIFF_HOST_DEVICE auto power(const XDIFF_SEED& /*arg*/);

// --------------------------------------------------------



// ------------- Binary operation declarations ------------
// Standard operations
XDIFF_DECLARE_BINARY_PACK(operator+, assign_add)
XDIFF_DECLARE_BINARY_PACK(operator-, assign_sub)
XDIFF_DECLARE_BINARY_PACK(operator*, assign_mul)
XDIFF_DECLARE_BINARY_PACK(operator/, assign_div)
XDIFF_DECLARE_BINARY_PACK(pow, assign_pow)

// Comparison operators
XDIFF_DECLARE_COMPARISON(==)
XDIFF_DECLARE_COMPARISON(!=)
XDIFF_DECLARE_COMPARISON(<)
XDIFF_DECLARE_COMPARISON(>)
XDIFF_DECLARE_COMPARISON(<=)
XDIFF_DECLARE_COMPARISON(>=)
// --------------------------------------------------------


// -------------------------------- Stream operator overload --------------------------------
template<typename T, typename GD, int Order>
XDIFF_FORCEINLINE
std::ostream& operator<<(std::ostream& os, const detail::XDIFF_DUAL& a){
    return os << a.value();
}

template<typename T, typename GD, int Order>
XDIFF_FORCEINLINE
std::ostream& operator<<(std::ostream& os, const detail::XDIFF_SEED& a){
    return os << a.value();
}

} // namespace xdiff::detail


namespace xdiff{

using detail::Dual, detail::Seed, detail::SeedState, detail::DualState, detail::State, detail::make_seeds;

// The math functions as xdiff::<some_math_func> etc. 
// The operators need no such line: argument-dependent lookup finds them.
using detail::abs, detail::log, detail::log10, detail::exp, detail::sqrt, detail::sin, detail::cos, detail::tan,
      detail::cot, detail::sec, detail::csc, detail::asin, detail::acos, detail::atan, detail::asec, detail::acsc,
      detail::sinh, detail::cosh, detail::tanh, detail::erf, detail::pow, detail::power;

LAZEX_USING_MATH;

} // namespace xdiff


#define XDIFF_USING_MATH LAZEX_USING_MATH; \
using   ::xdiff::abs, ::xdiff::log, ::xdiff::log10, ::xdiff::exp, \
        ::xdiff::sqrt, ::xdiff::sin, ::xdiff::cos, ::xdiff::tan, \
        ::xdiff::cot, ::xdiff::sec, ::xdiff::csc, ::xdiff::asin, \
        ::xdiff::acos, ::xdiff::atan, ::xdiff::asec, ::xdiff::acsc, \
        ::xdiff::sinh, ::xdiff::cosh, ::xdiff::tanh, ::xdiff::erf;


namespace lazex::detail{

template<typename T, typename GD, int Order>
struct constants<::xdiff::detail::Dual<T, GD, Order>> {
    using type = ::xdiff::detail::Dual<T, GD, Order>;
    static type pi()              { return type(constants<T>::pi()); }
    static type two_over_sqrtpi() { return type(constants<T>::two_over_sqrtpi()); }
};

} // namespace lazex::detail


// =============================================================================
// Standard library specializations
// =============================================================================

namespace std {

/**
 * @brief std::numeric_limits specialization for Dual.
 *
 * Inherits from numeric_limits<T> and wraps the limit values in Dual objects.
 * This enables Dual to work with algorithms that query numeric limits.
 */
template<typename T, typename GD, int Order>
class numeric_limits<::xdiff::detail::XDIFF_DUAL> : public numeric_limits<T>{
public:

    static constexpr ::xdiff::detail::XDIFF_DUAL min() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::min());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL max() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::max());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL lowest() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::lowest());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL epsilon() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::epsilon());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL round_error() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::round_error());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL infinity() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::infinity());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL quiet_NaN() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::quiet_NaN());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL signaling_NaN() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::signaling_NaN());
    }

    static constexpr ::xdiff::detail::XDIFF_DUAL denorm_min() noexcept {
        return ::xdiff::detail::XDIFF_DUAL(numeric_limits<T>::denorm_min());
    }
};

/// @brief Checks if a Dual's value is finite.
template<typename T, typename GD, int Order>
bool isfinite(const ::xdiff::detail::XDIFF_DUAL& x){
    return isfinite(x.value());
}

/// @brief Checks if a Dual's value is finite.
template<typename T, typename GD, int Order>
bool isfinite(const ::xdiff::detail::XDIFF_SEED& x){
    return isfinite(x.value());
}

} // namespace std

#endif // XDIFF_DUAL_DECLS_HPP


// TODO lazy type support