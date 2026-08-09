#ifndef XDIFF_DUAL_RECURSIVE_OPERATORS_HPP
#define XDIFF_DUAL_RECURSIVE_OPERATORS_HPP

#include "operator_template.hpp"


#define XDIFF_DUAL Dual<T, NVARS, NORDER, Layout::Nested>

namespace xdiff{



// ----------------------  Compound assignment operations ------------------------
// Addition
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    if constexpr (NVARS == 0) {
        if (out.nvars() != arg.nvars()) {
            out.set_nvars(arg.nvars());
        }
    }
    out.true_value += arg.true_value;
    for (size_t i=0; i < arg.nvars(); i++){
        out[i] += arg[i];
    }
    return out;
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& out, const F& arg){
    out.true_value += arg;
    for (size_t i=0; i < out.nvars(); i++){
        out[i] += arg;
    }
    return out;
}


// Subtraction
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    if constexpr (NVARS == 0) {
        if (out.nvars() != arg.nvars()) {
            out.set_nvars(arg.nvars());
        }
    }
    out.true_value -= arg.true_value;
    for (size_t i=0; i < arg.nvars(); i++){
        out[i] -= arg[i];
    }
    return out;
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& out, const F& arg){
    out.true_value -= arg;
    for (size_t i=0; i < out.nvars(); i++){
        out[i] -= arg;
    }
    return out;
}

// Multiplication
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    if constexpr (NVARS == 0) {
        if (out.nvars() != arg.nvars()) {
            out.set_nvars(arg.nvars());
        }
    }
    using G = typename XDIFF_DUAL::grad_type;
    using DP = xdiff::DiffPair<const G&, const G&>;
    for (size_t i=0; i < out.nvars(); i++){
        out[i] = xdiff::detail::rules::Mul<T>::diff_rule(DP{out.true_value, out[i]}, DP{arg.true_value, arg[i]});
    }
    out.true_value *= arg.true_value;
    return out;
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& out, const F& arg){
    out.true_value *= arg;
    for (size_t i=0; i < out.nvars(); i++){
        out[i] *= arg;
    }
    return out;
}

// Division
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    if constexpr (NVARS == 0) {
        if (out.nvars() != arg.nvars()) {
            out.set_nvars(arg.nvars());
        }
    }
    using G = typename XDIFF_DUAL::grad_type;
    using DP = xdiff::DiffPair<const G&, const G&>;
    for (size_t i=0; i < out.nvars(); i++){
        out[i] = xdiff::detail::rules::Div<T>::diff_rule(DP{out.true_value, out[i]}, DP{arg.true_value, arg[i]});
    }
    out.true_value /= arg.true_value;
    return out;
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& out, const F& arg){
    out.true_value /= arg;
    for (size_t i=0; i < out.nvars(); i++){
        out[i] /= arg;
    }
    return out;
}

// Operator overloads

// operator +=
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator+=(XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return assign_compound_add(a, b);
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator+=(XDIFF_DUAL& a, const F& b){
    return assign_compound_add(a, b);
}

// operator -=
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator-=(XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return assign_compound_sub(a, b);
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator-=(XDIFF_DUAL& a, const F& b){
    return assign_compound_sub(a, b);
}

// operator *=
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator*=(XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return assign_compound_mul(a, b);
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator*=(XDIFF_DUAL& a, const F& b){
    return assign_compound_mul(a, b);
}

// operator /=
template<typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator/=(XDIFF_DUAL& a, const XDIFF_DUAL& b){
    return assign_compound_div(a, b);
}

template<typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& operator/=(XDIFF_DUAL& a, const F& b){
    return assign_compound_div(a, b);
}

XDIFF_DEFINE_RECURSIVE_DUAL_BINARY_OPERATION(operator+, assign_add, xdiff::detail::rules::Add)
XDIFF_DEFINE_RECURSIVE_DUAL_BINARY_OPERATION(operator-, assign_sub, xdiff::detail::rules::Sub)
XDIFF_DEFINE_RECURSIVE_DUAL_BINARY_OPERATION(operator*, assign_mul, xdiff::detail::rules::Mul)
XDIFF_DEFINE_RECURSIVE_DUAL_BINARY_OPERATION(operator/, assign_div, xdiff::detail::rules::Div)
XDIFF_DEFINE_RECURSIVE_DUAL_BINARY_OPERATION(pow, assign_pow, xdiff::detail::rules::Pow)



XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(operator+, assign_pos, xdiff::detail::rules::Pos)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(operator-, assign_neg, xdiff::detail::rules::Neg)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(abs, assign_abs, xdiff::detail::rules::Abs)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(log, assign_log, xdiff::detail::rules::Log)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(log10, assign_log10, xdiff::detail::rules::Log10)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(exp, assign_exp, xdiff::detail::rules::Exp)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(sqrt, assign_sqrt, xdiff::detail::rules::Sqrt)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(sin, assign_sin, xdiff::detail::rules::Sin)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(cos, assign_cos, xdiff::detail::rules::Cos)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(tan, assign_tan, xdiff::detail::rules::Tan)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(cot, assign_cot, xdiff::detail::rules::Cot)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(sec, assign_sec, xdiff::detail::rules::Sec)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(csc, assign_csc, xdiff::detail::rules::Csc)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(asin, assign_asin, xdiff::detail::rules::ArcSin)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(acos, assign_acos, xdiff::detail::rules::ArcCos)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(atan, assign_atan, xdiff::detail::rules::ArcTan)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(acot, assign_acot, xdiff::detail::rules::ArcCot)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(asec, assign_asec, xdiff::detail::rules::ArcSec)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(acsc, assign_acsc, xdiff::detail::rules::ArcCsc)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(sinh, assign_sinh, xdiff::detail::rules::Sinh)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(cosh, assign_cosh, xdiff::detail::rules::Cosh)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(tanh, assign_tanh, xdiff::detail::rules::Tanh)
XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(erf, assign_erf, xdiff::detail::rules::Erf)

} // namespace xdiff


#undef XDIFF_DUAL


#endif // XDIFF_DUAL_RECURSIVE_OPERATORS_HPP