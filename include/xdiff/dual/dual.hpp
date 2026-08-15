#ifndef XDIFF_DUAL_HPP
#define XDIFF_DUAL_HPP


#include "../tools.hpp"
#include <cstdint>
#include <ostream>

#define XDIFF_DUAL Dual<T, NVARS, NORDER, LY>
#define XDIFF_NO_DEF {static_assert(false, "This function is not defined for this type of Dual");}

namespace xdiff{

namespace detail{

// All autodiff-related classes should inherit from this class. It acts as an identifier.
template<typename Derived>
struct DualIdentifier{};

} // namespace detail

enum class Layout : uint8_t {
    Flat,
    Nested
};


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

namespace detail{

template<typename T, typename F>
struct isDualHelper{
    static constexpr bool value = false;
};

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
struct isDualHelper<T, Dual<T, NVARS, NORDER, LY>>{
    static constexpr bool value = true;
};

} // namespace xdiff::traits::detail

template<typename F>
concept isAnyZero = std::is_base_of_v<ZeroValue, std::decay_t<F>>;

template<typename F>
concept isSymbol = requires {F::AXIS;} && std::is_same_v<Symbol<F::AXIS>, F>;

template<typename F>
concept isAxis = ((isSymbol<F> || std::is_integral_v<F>));

template<typename F, typename T>
concept isDual = detail::isDualHelper<T, std::decay_t<F>>::value;

template<typename D>
concept isAnyDual = requires { typename D::value_type; } && isDual<D, typename D::value_type>;

} // namespace xdiff::traits


// ----------------------  Unary operations ------------------------

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pos(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_neg(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_abs(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_log(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_log10(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_exp(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sqrt(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sin(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cos(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_tan(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cot(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sec(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_csc(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_asin(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acos(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_atan(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acot(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_asec(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_acsc(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sinh(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_cosh(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_tanh(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_erf(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

// ----------------------  Binary operations ------------------------


// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_add(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_sub(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_mul(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_div(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Power
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_pow(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;


// ---------------------- Compound assignment operations ------------------------


// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& /*out*/, const F& /*arg*/) XDIFF_NO_DEF;




// -------------------------------- Operator overloads --------------------------------

// Unary +
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_DUAL& /*a*/) XDIFF_NO_DEF;

// Unary -
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_DUAL& /*a*/) XDIFF_NO_DEF;
// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator+(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator-(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator*(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL operator/(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

// Power
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL pow(const F& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;


// -------------------------------- Compound assignment operator overloads --------------------------------

// Addition
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator+=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator+=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Subtraction
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator-=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator-=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Multiplication
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator*=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator*=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;

// Division
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator/=(XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/) XDIFF_NO_DEF;
template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE XDIFF_DUAL& operator/=(XDIFF_DUAL& /*a*/, const F& /*b*/) XDIFF_NO_DEF;


// -------------------------------- Comparison operator overloads --------------------------------

// ==
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator==(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() == b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator==(const XDIFF_DUAL& a, const F& b){
    return a.value() == b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator==(const F& a, const XDIFF_DUAL& b){
    return a == b.value();
}

// !=
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() != b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator!=(const XDIFF_DUAL& a, const F& b){
    return a.value() != b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator!=(const F& a, const XDIFF_DUAL& b){
    return a != b.value();
}

// <
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() < b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<(const XDIFF_DUAL& a, const F& b){
    return a.value() < b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<(const F& a, const XDIFF_DUAL& b){
    return a < b.value();
}


// >
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() > b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>(const XDIFF_DUAL& a, const F& b){
    return a.value() > b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>(const F& a, const XDIFF_DUAL& b){
    return a > b.value();
}

// <=
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() <= b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<=(const XDIFF_DUAL& a, const F& b){
    return a.value() <= b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator<=(const F& a, const XDIFF_DUAL& b){
    return a <= b.value();
}

// >=
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return a.value() >= b.value();
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>=(const XDIFF_DUAL& a, const F& b){
    return a.value() >= b;
}

template<typename F, typename T, size_t NVARS, size_t NORDER, Layout LY>
XDIFF_HOST_DEVICE bool operator>=(const F& a, const XDIFF_DUAL& b){
    return a >= b.value();
}



// -------------------------------- Stream operator overload --------------------------------
template<traits::isAnyDual DualNum>
XDIFF_FORCEINLINE
std::ostream& operator<<(std::ostream& os, const DualNum& a){
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

} // namespace std

#undef XDIFF_DUAL
#undef XDIFF_NO_DEF

#endif // XDIFF_DUAL_HPP