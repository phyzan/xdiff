#ifndef XDIFF_DUAL_RECURSIVE_OPERATOR_TEMPLATE_HPP
#define XDIFF_DUAL_RECURSIVE_OPERATOR_TEMPLATE_HPP

#include <iostream>
#include "../../rules/math.hpp"

#define XDIFF_ASSERT_REC_DUAL_BINARY_OPERATION(a, b) \
    assert(a.nvars() == b.nvars() && "All RecursiveDuals must have the same number of derivatives"); \


namespace xdiff::detail {


// ------------------------------ Unary operation implementation ------------------------------
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested>& 
unary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, 
                  const Dual<T, NVARS, NORDER, Layout::Nested>& arg) {
    if constexpr (NVARS == 0) {
        if (out.nvars() != arg.nvars()) {
            out.set_nvars(arg.nvars());
        }
    }
    using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
    using DP = xdiff::DiffPair<const G&, const G&>;

    for (size_t i = 0; i < arg.nvars(); i++){
        out[i] = RuleStruct<T>::diff_rule(DP{arg.true_value, arg[i]});
    }
    out.true_value = RuleStruct<T>::operation(arg.true_value);
    return out;
}

template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested> 
unary_op_impl(const Dual<T, NVARS, NORDER, Layout::Nested>& arg) {
    Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = arg.nvars()});
    unary_assign_impl<RuleStruct>(out, arg);
    return out;
}







// ------------------------------ Binary operation implementation ------------------------------
// Binary: (Dual, Dual)
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested>& 
binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                   const Dual<T, NVARS, NORDER, Layout::Nested>& a,
                   const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
    XDIFF_ASSERT_REC_DUAL_BINARY_OPERATION(a, b);
    if constexpr (NVARS == 0) {
        if (out.nvars() != a.nvars()) {
            out.set_nvars(a.nvars());
        }
    }
    using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
    using DP = xdiff::DiffPair<const G&, const G&>;

    for (size_t i = 0; i < a.nvars(); i++){
        out[i] = RuleStruct<T>::diff_rule(DP{a.true_value, a[i]}, DP{b.true_value, b[i]});
    }
    out.true_value = RuleStruct<T>::operation(a.true_value, b.true_value);
    return out;
}

// Binary: (Scalar, Dual)
template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested>& 
binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                   const F& a,
                   const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
    if constexpr (NVARS == 0) {
        if (out.nvars() != b.nvars()) {
            out.set_nvars(b.nvars());
        }
    }
    using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
    using DP_A = xdiff::DiffPair<const F&, const ZeroValue&>;
    using DP_B = xdiff::DiffPair<const G&, const G&>;

    for (size_t i = 0; i < b.nvars(); i++){
        out[i] = RuleStruct<T>::diff_rule(DP_A{a, ZeroValue{}}, DP_B{b.true_value, b[i]});
    }
    out.true_value = RuleStruct<T>::operation(a, b.true_value);
    return out;
}

// Binary: (Dual, Scalar)
template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested>& 
binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                   const Dual<T, NVARS, NORDER, Layout::Nested>& a,
                   const F& b) {
    if constexpr (NVARS == 0) {
        if (out.nvars() != a.nvars()) {
            out.set_nvars(a.nvars());
        }
    }
    using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
    using DP_A = xdiff::DiffPair<const G&, const G&>;
    using DP_B = xdiff::DiffPair<const F&, const ZeroValue&>;

    for (size_t i = 0; i < a.nvars(); i++){
        out[i] = RuleStruct<T>::diff_rule(DP_A{a.true_value, a[i]}, DP_B{b, ZeroValue{}});
    }
    out.true_value = RuleStruct<T>::operation(a.true_value, b);
    return out;
}

// Free function wrappers
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested> 
binary_op_impl(const Dual<T, NVARS, NORDER, Layout::Nested>& a,
               const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
    Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = a.nvars()});
    binary_assign_impl<RuleStruct>(out, a, b);
    return out;
}

template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested> 
binary_op_impl(const F& a, const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
    Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = b.nvars()});
    binary_assign_impl<RuleStruct>(out, a, b);
    return out;
}

template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
XDIFF_INLINE_HOST_DEVICE
Dual<T, NVARS, NORDER, Layout::Nested> 
binary_op_impl(const Dual<T, NVARS, NORDER, Layout::Nested>& a, const F& b) {
    Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = a.nvars()});
    binary_assign_impl<RuleStruct>(out, a, b);
    return out;
}


} // namespace xdiff::detail







// ------------------------------ Operator macros ------------------------------

// Unary operation
#define XDIFF_DEFINE_RECURSIVE_DUAL_UNARY_OPERATION(NAME, ASSIGN_NAME, STRUCT) \
template<typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE \
Dual<T, NVARS, NORDER, Layout::Nested>& ASSIGN_NAME(Dual<T, NVARS, NORDER, Layout::Nested>& out, \
                    const Dual<T, NVARS, NORDER, Layout::Nested>& arg) { \
    return xdiff::detail::unary_assign_impl<STRUCT>(out, arg); \
} \
\
\
template<typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const Dual<T, NVARS, NORDER, Layout::Nested>& arg) { \
    return xdiff::detail::unary_op_impl<STRUCT>(arg); \
}





// Binary operation
#define XDIFF_DEFINE_RECURSIVE_DUAL_BINARY_OPERATION(NAME, ASSIGN_NAME, STRUCT) \
template<typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME(Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename F, typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME(Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const F& a, const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename F, typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME(Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, const F& b){ \
    return xdiff::detail::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE auto NAME(const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::binary_op_impl<STRUCT>(a, b); \
} \
\
\
template<typename F, typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE auto NAME(const F& a, const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::binary_op_impl<STRUCT>(a, b); \
} \
template<typename F, typename T, size_t NVARS, size_t NORDER> \
XDIFF_INLINE_HOST_DEVICE auto NAME(const Dual<T,NVARS,NORDER,Layout::Nested>& a, const F& b){ \
    return xdiff::detail::binary_op_impl<STRUCT>(a, b); \
}



#endif //XDIFF_DUAL_RECURSIVE_OPERATOR_TEMPLATE_HPP