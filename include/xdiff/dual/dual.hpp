#ifndef XDIFF_DUAL_HPP
#define XDIFF_DUAL_HPP


#include "../tools.hpp"
#include <cstdint>
#include <ostream>

#define XDIFF_DUAL Dual<T, NVARS, NORDER, LY>
#define XDIFF_SEED Seed<T, NVARS, NORDER, LY>
#define XDIFF_NO_DEF {static_assert(false, "This function is not defined for this type of Dual");}

namespace xdiff{


enum class Layout : uint8_t {
    Flat,
    Nested
};

template<typename Derived, typename T, size_t NVARS, Layout LY>
class DualBase;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
class Dual;



// For "nvars" and "order", the zero value is used for the default runtime number (for dynamic number of nvars or order) or compile-time number of variables and order, respectively.
// For example, if the compile time of nvars is 3, then passing nvars = 0 will use the compile-time value of 3, and no assertion error will be thrown. If the compile time of nvars is 0 (dynamic number of variables), then passing nvars = 0 will use the default runtime number of variables, and no assertion error will be thrown.
// An assertion error is thrown when the compile-time number of variables "NVARS" is non-zero and the passed nvars is non-zero but different for NVARS. The same applies to the "order" parameter.
struct MakeDual{
    int axis = -1;
    size_t nvars = 0;
    size_t order = 0;
};

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
class SeedVector;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
class Seed;


namespace detail{

/**
 * @brief Satisfied when F is a plain scalar operand of a Dual over T.
 *
 * Every operand that carries a derivative of its own is excluded, because none of them converts
 * to T: a Dual deletes that conversion outright, a Seed never declares one, and a lazy
 * expression standing in for a Dual inherits the deleted one. Constraining the overloads that
 * stand for a scalar this way keeps them from swallowing a differentiable operand and silently
 * treating it as a constant, and leaves the operand to the overload that knows how to
 * differentiate it. It also keeps this header independent of the lazy library.
 */
template<typename F, typename T>
concept isScalarOperand = std::is_convertible_v<std::decay_t<F>, T>;

} // namespace xdiff::detail


template<typename F, typename DF>
struct DiffPair;

template<size_t Axis>
struct Symbol{
    static constexpr char AXIS = Axis;

    /// @brief Implicit conversion to size_t for use in derivative indexing.
    constexpr operator size_t(){
        return Axis;
    }
};

struct ZeroValue{};


namespace traits{

template<typename F>
concept isAnyZero = std::is_base_of_v<ZeroValue, std::decay_t<F>>;

template<typename F>
concept isSymbol = requires {F::AXIS;} && std::is_same_v<Symbol<F::AXIS>, F>;

template<typename F>
concept isAxis = ((isSymbol<F> || std::is_integral_v<F>));

} // namespace xdiff::traits


template<typename F, typename DF>
struct DiffPair{

    F value;   ///< The function value
    DF grad;   ///< The derivative value

    /// @brief Returns true if this derivative is trivially zero (compile-time optimization)
    XDIFF_INLINE_HOST_DEVICE
    static constexpr bool isTrivial() {
        return xdiff::traits::isAnyZero<DF>;
    }
};

/// @brief CTAD guide for DiffPair to preserve reference types
template<typename F, typename DF>
DiffPair(F&&, DF&&)
-> DiffPair<
    std::conditional_t<
        std::is_lvalue_reference_v<F>,
        F,
        std::remove_reference_t<F>
    >,
    std::conditional_t<
        std::is_lvalue_reference_v<DF>,
        DF,
        std::remove_reference_t<DF>
    >
>;


template<typename Derived, typename T, size_t NVARS, Layout LY>
class DualBase {

public:
    using value_type = T;

    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& value() const {
        return XDIFF_THIS->value();
    }

    template<xdiff::traits::isAxis... Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& get_diff_wrt(Int... x) const{
        return XDIFF_THIS->get_diff_wrt(x...);
    }

};


// ----------------------  Unary operations ------------------------

// +Dual
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pos(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pos(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// -Dual
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_neg(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_neg(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// abs(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_abs(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_abs(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// log(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_log(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_log(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// log10(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_log10(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_log10(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// exp(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_exp(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_exp(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// sqrt(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sqrt(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sqrt(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

// sin(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sin(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sin(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// cos(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cos(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cos(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// tan(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_tan(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;
    
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_tan(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// cot(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cot(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cot(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// sec(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sec(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sec(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// csc(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_csc(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_csc(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// asin(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_asin(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_asin(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// acos(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acos(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acos(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// atan(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_atan(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_atan(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// acot(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acot(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acot(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// asec(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_asec(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_asec(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// acsc(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acsc(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acsc(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// sinh(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sinh(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sinh(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// cosh(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cosh(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cosh(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// tanh(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_tanh(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_tanh(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// erf(Dual)
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_erf(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_erf(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

// ----------------------  Binary operations ------------------------


// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Power
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;


// ---------------------- Compound assignment operations ------------------------


// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;




// -------------------------------- Operator overloads --------------------------------

// Unary +
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_DUAL& /*a*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_SEED& /*a*/) XDIFF_NO_DEF;

// Unary -
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_DUAL& /*a*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_SEED& /*a*/) XDIFF_NO_DEF;

// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

// Power
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_SEED& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const F& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;


// -------------------------------- Compound assignment operator overloads --------------------------------

// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator+=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator+=(XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& operator+=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator-=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator-=(XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& operator-=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator*=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator*=(XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& operator*=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator/=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator/=(XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE XDIFF_DUAL& operator/=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;


// -------------------------------- Comparison operator overloads --------------------------------

// ==
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator==(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() == b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator==(const XDIFF_DUAL& a, const XDIFF_SEED& b){
    return a.value() == b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator==(const XDIFF_SEED& a, const XDIFF_DUAL& b){
    return a.value() == b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator==(const XDIFF_SEED& a, const XDIFF_SEED& b){
    return a.value() == b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator==(const XDIFF_DUAL& a, const F& b){
    return a.value() == b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator==(const F& a, const XDIFF_DUAL& b){
    return a == b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator==(const XDIFF_SEED& a, const F& b){
    return a.value() == b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator==(const F& a, const XDIFF_SEED& b){
    return a == b.value();
}

// !=
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() != b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_DUAL& a, const XDIFF_SEED& b){
    return a.value() != b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_SEED& a, const XDIFF_DUAL& b){
    return a.value() != b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_SEED& a, const XDIFF_SEED& b){
    return a.value() != b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_DUAL& a, const F& b){
    return a.value() != b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator!=(const F& a, const XDIFF_DUAL& b){
    return a != b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_SEED& a, const F& b){
    return a.value() != b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator!=(const F& a, const XDIFF_SEED& b){
    return a != b.value();
}

// <
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() < b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<(const XDIFF_DUAL& a, const XDIFF_SEED& b){
    return a.value() < b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<(const XDIFF_SEED& a, const XDIFF_DUAL& b){
    return a.value() < b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<(const XDIFF_SEED& a, const XDIFF_SEED& b){
    return a.value() < b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<(const XDIFF_DUAL& a, const F& b){
    return a.value() < b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<(const F& a, const XDIFF_DUAL& b){
    return a < b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<(const XDIFF_SEED& a, const F& b){
    return a.value() < b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<(const F& a, const XDIFF_SEED& b){
    return a < b.value();
}


// >
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() > b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>(const XDIFF_DUAL& a, const XDIFF_SEED& b){
    return a.value() > b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>(const XDIFF_SEED& a, const XDIFF_DUAL& b){
    return a.value() > b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>(const XDIFF_SEED& a, const XDIFF_SEED& b){
    return a.value() > b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>(const XDIFF_DUAL& a, const F& b){
    return a.value() > b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>(const F& a, const XDIFF_DUAL& b){
    return a > b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>(const XDIFF_SEED& a, const F& b){
    return a.value() > b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>(const F& a, const XDIFF_SEED& b){
    return a > b.value();
}

// <=
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() <= b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_DUAL& a, const XDIFF_SEED& b){
    return a.value() <= b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_SEED& a, const XDIFF_DUAL& b){
    return a.value() <= b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_SEED& a, const XDIFF_SEED& b){
    return a.value() <= b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_DUAL& a, const F& b){
    return a.value() <= b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<=(const F& a, const XDIFF_DUAL& b){
    return a <= b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_SEED& a, const F& b){
    return a.value() <= b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator<=(const F& a, const XDIFF_SEED& b){
    return a <= b.value();
}

// >=
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() >= b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_DUAL& a, const XDIFF_SEED& b){
    return a.value() >= b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_SEED& a, const XDIFF_DUAL& b){
    return a.value() >= b.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_SEED& a, const XDIFF_SEED& b){
    return a.value() >= b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_DUAL& a, const F& b){
    return a.value() >= b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>=(const F& a, const XDIFF_DUAL& b){
    return a >= b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_SEED& a, const F& b){
    return a.value() >= b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
requires (detail::isScalarOperand<F, T>)
XDIFF_HOST_DEVICE bool operator>=(const F& a, const XDIFF_SEED& b){
    return a >= b.value();
}



// -------------------------------- Stream operator overload --------------------------------
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_FORCEINLINE
std::ostream& operator<<(std::ostream& os, const XDIFF_DUAL& a){
    return os << a.value();
}

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_FORCEINLINE
std::ostream& operator<<(std::ostream& os, const XDIFF_SEED& a){
    return os << a.value();
}


// Import standard math functions into xdiff namespace for ADL
using std::abs, std::pow, std::log, std::log10, std::sqrt, std::exp, std::sin, std::cos, std::tan, std::abs, std::sinh, std::cosh, std::tanh, std::erf, std::asin, std::acos, std::atan, std::asinh, std::acosh, std::atanh;

} // namespace xdiff


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
template<typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
class numeric_limits<xdiff::XDIFF_DUAL> : public numeric_limits<T>{
public:

    static constexpr xdiff::XDIFF_DUAL min() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::min());
    }

    static constexpr xdiff::XDIFF_DUAL max() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::max());
    }

    static constexpr xdiff::XDIFF_DUAL lowest() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::lowest());
    }

    static constexpr xdiff::XDIFF_DUAL epsilon() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::epsilon());
    }

    static constexpr xdiff::XDIFF_DUAL round_error() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::round_error());
    }

    static constexpr xdiff::XDIFF_DUAL infinity() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::infinity());
    }

    static constexpr xdiff::XDIFF_DUAL quiet_NaN() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::quiet_NaN());
    }

    static constexpr xdiff::XDIFF_DUAL signaling_NaN() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::signaling_NaN());
    }

    static constexpr xdiff::XDIFF_DUAL denorm_min() noexcept {
        return xdiff::XDIFF_DUAL(numeric_limits<T>::denorm_min());
    }
};

/// @brief Checks if a Dual's value is finite.
template<typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
bool isfinite(const xdiff::XDIFF_DUAL& x){
    return isfinite(x.value());
}

/// @brief Checks if a Dual's value is finite.
template<typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
bool isfinite(const xdiff::XDIFF_SEED& x){
    return isfinite(x.value());
}


} // namespace std

#undef XDIFF_DUAL
#undef XDIFF_SEED
#undef XDIFF_NO_DEF

#endif // XDIFF_DUAL_HPP