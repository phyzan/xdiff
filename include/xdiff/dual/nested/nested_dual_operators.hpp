#ifndef XDIFF_DUAL_NESTED_OPERATORS_HPP
#define XDIFF_DUAL_NESTED_OPERATORS_HPP

#include "operator_template.hpp"


#define XDIFF_DUAL Dual<T, NVARS, NORDER, Layout::Nested>
#define XDIFF_SEED Seed<T, NVARS, NORDER, Layout::Nested>

namespace xdiff{



// ----------------------  Compound assignment operations ------------------------
// Each one forwards to the corresponding member operator, which owns the implementation and can
// reach the storage directly. That is why none of these has to be a friend of Dual.

template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    return out += arg;
}

template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& out, const XDIFF_SEED& arg){
    return out += arg;
}

template<typename F, typename T, int NVARS, int NORDER>
requires (::xdiff::detail::isScalarOperand<F, T>)
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_add(XDIFF_DUAL& out, const F& arg){
    return out += arg;
}


template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    return out -= arg;
}

template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& out, const XDIFF_SEED& arg){
    return out -= arg;
}

template<typename F, typename T, int NVARS, int NORDER>
requires (::xdiff::detail::isScalarOperand<F, T>)
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_sub(XDIFF_DUAL& out, const F& arg){
    return out -= arg;
}


template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    return out *= arg;
}

template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& out, const XDIFF_SEED& arg){
    return out *= arg;
}

template<typename F, typename T, int NVARS, int NORDER>
requires (::xdiff::detail::isScalarOperand<F, T>)
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_mul(XDIFF_DUAL& out, const F& arg){
    return out *= arg;
}


template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    return out /= arg;
}

template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& out, const XDIFF_SEED& arg){
    return out /= arg;
}

template<typename F, typename T, int NVARS, int NORDER>
requires (::xdiff::detail::isScalarOperand<F, T>)
XDIFF_INLINE_HOST_DEVICE
XDIFF_DUAL& assign_compound_div(XDIFF_DUAL& out, const F& arg){
    return out /= arg;
}


XDIFF_DEFINE_NESTED_DUAL_BINARY_OPERATION(operator+, assign_add, xdiff::detail::rules::Add)
XDIFF_DEFINE_NESTED_DUAL_BINARY_OPERATION(operator-, assign_sub, xdiff::detail::rules::Sub)
XDIFF_DEFINE_NESTED_DUAL_BINARY_OPERATION(operator*, assign_mul, xdiff::detail::rules::Mul)
XDIFF_DEFINE_NESTED_DUAL_BINARY_OPERATION(operator/, assign_div, xdiff::detail::rules::Div)
XDIFF_DEFINE_NESTED_DUAL_BINARY_OPERATION(pow, assign_pow, xdiff::detail::rules::Pow)



XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(operator+, assign_pos, xdiff::detail::rules::Pos)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(operator-, assign_neg, xdiff::detail::rules::Neg)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(abs, assign_abs, xdiff::detail::rules::Abs)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(log, assign_log, xdiff::detail::rules::Log)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(log10, assign_log10, xdiff::detail::rules::Log10)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(exp, assign_exp, xdiff::detail::rules::Exp)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(sqrt, assign_sqrt, xdiff::detail::rules::Sqrt)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(sin, assign_sin, xdiff::detail::rules::Sin)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(cos, assign_cos, xdiff::detail::rules::Cos)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(tan, assign_tan, xdiff::detail::rules::Tan)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(cot, assign_cot, xdiff::detail::rules::Cot)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(sec, assign_sec, xdiff::detail::rules::Sec)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(csc, assign_csc, xdiff::detail::rules::Csc)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(asin, assign_asin, xdiff::detail::rules::ArcSin)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(acos, assign_acos, xdiff::detail::rules::ArcCos)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(atan, assign_atan, xdiff::detail::rules::ArcTan)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(acot, assign_acot, xdiff::detail::rules::ArcCot)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(asec, assign_asec, xdiff::detail::rules::ArcSec)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(acsc, assign_acsc, xdiff::detail::rules::ArcCsc)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(sinh, assign_sinh, xdiff::detail::rules::Sinh)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(cosh, assign_cosh, xdiff::detail::rules::Cosh)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(tanh, assign_tanh, xdiff::detail::rules::Tanh)
XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(erf, assign_erf, xdiff::detail::rules::Erf)

} // namespace xdiff


#undef XDIFF_DUAL
#undef XDIFF_SEED


#endif // XDIFF_DUAL_NESTED_OPERATORS_HPP