#ifndef XDIFF_DUAL_NESTED_OPERATOR_TEMPLATE_HPP
#define XDIFF_DUAL_NESTED_OPERATOR_TEMPLATE_HPP

#include <iostream> // IWYU pragma: keep
#include "../../rules/math.hpp" // IWYU pragma: keep

#define XDIFF_ASSERT_REC_DUAL_BINARY_OPERATION(a, b) \
    assert(a.nvars() == b.nvars() && "All RecursiveDuals must have the same number of derivatives"); \


namespace xdiff::detail {


// Resizes 'out' so that it can hold the result of an operation on an operand that has "nvars"
// variables. Only a runtime number of variables can ever be mismatched.
template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
void format_nested(Dual<T, NVARS, NORDER, Layout::Nested>& out, size_t nvars) {

    if constexpr (NVARS == -1) {
        if (out.nvars() != nvars) {
            out.set_nvars(nvars);
        }
    } else {
        assert(out.nvars() == nvars && "All RecursiveDuals must have the same number of derivatives");
        (void)nvars;
    }

    // TODO : When runtime order is needed, handle it here.
}

template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
void format_nested(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                   const Dual<T, NVARS, NORDER, Layout::Nested>& src) {
    format_nested(out, src.nvars());
}

template<typename T, int NVARS, int NORDER>
XDIFF_INLINE_HOST_DEVICE
void format_nested(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                   const Seed<T, NVARS, NORDER, Layout::Nested>& src) {
    format_nested(out, src.nvars());
}


struct NestedDualOperationHelper {



    // ------------------------------ Unary operation implementation ------------------------------

    // f(out, Dual)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>&
    unary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Dual<T, NVARS, NORDER, Layout::Nested>& arg) {
        format_nested(out, arg);
        using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
        using DP = DiffPair<const G&, const G&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < arg.nvars(); i++){
                out[i] = RuleStruct<T>::diff_rule(DP{arg.true_value, arg[i]});
            }
        }
        out.true_value = RuleStruct<T>::operation(arg.true_value);
        return out;
    }

    // f(out, Seed)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>&
    unary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Seed<T, NVARS, NORDER, Layout::Nested>& arg) {
        format_nested(out, arg);

        // A seed variable differentiates to the constant 1 along its own axis, and to 0 along every
        // other one. The vanishing gradient is passed as a ZeroValue so that the rule can drop the
        // term at compile time, exactly as it does for a scalar operand.
        if constexpr (NORDER > 0){
            for (size_t i = 0; i < arg.nvars(); i++){
                if (i == arg.axis()){
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{arg.trimmed(), 1});
                } else {
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{arg.trimmed(), ZeroValue{}});
                }
            }
        }

        out.true_value = RuleStruct<T>::operation(arg.trimmed());

        return out;
    }

    // f(Dual) -> Dual (unary operation)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested> 
    unary_op_impl(const Dual<T, NVARS, NORDER, Layout::Nested>& arg) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(arg.nvars()), .order = int(arg.order())});
        unary_assign_impl<RuleStruct>(out, arg);
        return out;
    }

    // f(Seed) -> Dual (unary operation)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested> 
    unary_op_impl(const Seed<T, NVARS, NORDER, Layout::Nested>& arg) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(arg.nvars()), .order = int(arg.order())});
        unary_assign_impl<RuleStruct>(out, arg);
        return out;
    }





    // ------------------------------ Binary operation implementation ------------------------------

    // Binary: (Dual, Dual)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>& 
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Dual<T, NVARS, NORDER, Layout::Nested>& a,
                    const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
        XDIFF_ASSERT_REC_DUAL_BINARY_OPERATION(a, b);
        format_nested(out, a);
        using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
        using DP = DiffPair<const G&, const G&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < a.nvars(); i++){
                out[i] = RuleStruct<T>::diff_rule(DP{a.true_value, a[i]}, DP{b.true_value, b[i]});
            }
        }
        out.true_value = RuleStruct<T>::operation(a.true_value, b.true_value);
        return out;
    }

    // Binary: (Seed, Dual)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>& 
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Seed<T, NVARS, NORDER, Layout::Nested>& a,
                    const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
        XDIFF_ASSERT_REC_DUAL_BINARY_OPERATION(a, b);
        format_nested(out, a);
        using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
        using DP = DiffPair<const G&, const G&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < a.nvars(); i++){
                if (i == a.axis()){
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), 1}, DP{b.true_value, b[i]});
                }else{
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), ZeroValue{}}, DP{b.true_value, b[i]});
                }
            }
        }
        out.true_value = RuleStruct<T>::operation(a.trimmed(), b.true_value);
        return out;
    }

    // Binary: (Seed, Seed)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>& 
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Seed<T, NVARS, NORDER, Layout::Nested>& a,
                    const Seed<T, NVARS, NORDER, Layout::Nested>& b) {
        XDIFF_ASSERT_REC_DUAL_BINARY_OPERATION(a, b);
        format_nested(out, a);

        // Both operands are seeds, so along any axis but their own the gradient is exactly zero. It
        // is passed as a ZeroValue rather than as a zero-valued scalar so that the rule drops the
        // term at compile time instead of evaluating it, which also keeps rules whose general branch
        // is only defined for a non-vanishing operand (such as Pow) away from an invalid argument.
        if constexpr (NORDER > 0){
            for (size_t i = 0; i < a.nvars(); i++){
                const bool grad_a = (i == a.axis());
                const bool grad_b = (i == b.axis());
                if (grad_a && grad_b){
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), 1}, DiffPair{b.trimmed(), 1});
                } else if (grad_a){
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), 1}, DiffPair{b.trimmed(), ZeroValue{}});
                } else if (grad_b){
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), ZeroValue{}}, DiffPair{b.trimmed(), 1});
                } else {
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), ZeroValue{}}, DiffPair{b.trimmed(), ZeroValue{}});
                }
            }
        }
        out.true_value = RuleStruct<T>::operation(a.trimmed(), b.trimmed());
        return out;
    }

    // Binary: (Dual, Seed)
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>& 
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Dual<T, NVARS, NORDER, Layout::Nested>& a,
                    const Seed<T, NVARS, NORDER, Layout::Nested>& b) {
        XDIFF_ASSERT_REC_DUAL_BINARY_OPERATION(a, b);
        format_nested(out, a);
        using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
        using DP = DiffPair<const G&, const G&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < a.nvars(); i++){
                if (i == b.axis()){
                    out[i] = RuleStruct<T>::diff_rule(DP{a.true_value, a[i]}, DiffPair{b.trimmed(), 1});
                }else{
                    out[i] = RuleStruct<T>::diff_rule(DP{a.true_value, a[i]}, DiffPair{b.trimmed(), ZeroValue{}});
                }
            }
        }
        out.true_value = RuleStruct<T>::operation(a.true_value, b.trimmed());
        return out;
    }

    // Binary: (Scalar, Dual)
    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>& 
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const F& a,
                    const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
        format_nested(out, b);
        using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
        using DP_A = DiffPair<const F&, const ZeroValue&>;
        using DP_B = DiffPair<const G&, const G&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < b.nvars(); i++){
                out[i] = RuleStruct<T>::diff_rule(DP_A{a, ZeroValue{}}, DP_B{b.true_value, b[i]});
            }
        }
        out.true_value = RuleStruct<T>::operation(a, b.true_value);
        return out;
    }

    // Binary: (Dual, Scalar)
    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>& 
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Dual<T, NVARS, NORDER, Layout::Nested>& a,
                    const F& b) {
        format_nested(out, a);
        using G = typename Dual<T, NVARS, NORDER, Layout::Nested>::grad_type;
        using DP_A = DiffPair<const G&, const G&>;
        using DP_B = DiffPair<const F&, const ZeroValue&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < a.nvars(); i++){
                out[i] = RuleStruct<T>::diff_rule(DP_A{a.true_value, a[i]}, DP_B{b, ZeroValue{}});
            }
        }
        out.true_value = RuleStruct<T>::operation(a.true_value, b);
        return out;
    }

    // Binary: (Scalar, Seed)
    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>&
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const F& a,
                    const Seed<T, NVARS, NORDER, Layout::Nested>& b) {
        format_nested(out, b);
        using DP_A = DiffPair<const F&, const ZeroValue&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < b.nvars(); i++){
                if (i == b.axis()){
                    out[i] = RuleStruct<T>::diff_rule(DP_A{a, ZeroValue{}}, DiffPair{b.trimmed(), 1});
                }else{
                    out[i] = RuleStruct<T>::diff_rule(DP_A{a, ZeroValue{}}, DiffPair{b.trimmed(), ZeroValue{}});
                }
            }
        }
        out.true_value = RuleStruct<T>::operation(a, b.trimmed());
        return out;
    }

    // Binary: (Seed, Scalar)
    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>&
    binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out,
                    const Seed<T, NVARS, NORDER, Layout::Nested>& a,
                    const F& b) {
        format_nested(out, a);
        using DP_B = DiffPair<const F&, const ZeroValue&>;

        if constexpr (NORDER > 0){
            for (size_t i = 0; i < a.nvars(); i++){
                if (i == a.axis()){
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), 1}, DP_B{b, ZeroValue{}});
                }else{
                    out[i] = RuleStruct<T>::diff_rule(DiffPair{a.trimmed(), ZeroValue{}}, DP_B{b, ZeroValue{}});
                }
            }
        }
        out.true_value = RuleStruct<T>::operation(a.trimmed(), b);
        return out;
    }

    // Free function wrappers
    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested> 
    binary_op_impl(const Dual<T, NVARS, NORDER, Layout::Nested>& a,
                const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(a.nvars()), .order = int(a.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested> 
    binary_op_impl(const Dual<T, NVARS, NORDER, Layout::Nested>& a,
                const Seed<T, NVARS, NORDER, Layout::Nested>& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(a.nvars()), .order = int(a.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested> 
    binary_op_impl(const Seed<T, NVARS, NORDER, Layout::Nested>& a,
                const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(b.nvars()), .order = int(b.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

    template<template<typename> typename RuleStruct, typename T, int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested> 
    binary_op_impl(const Seed<T, NVARS, NORDER, Layout::Nested>& a,
                const Seed<T, NVARS, NORDER, Layout::Nested>& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(a.nvars()), .order = int(a.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested> 
    binary_op_impl(const F& a, const Dual<T, NVARS, NORDER, Layout::Nested>& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(b.nvars()), .order = int(b.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>
    binary_op_impl(const Dual<T, NVARS, NORDER, Layout::Nested>& a, const F& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(a.nvars()), .order = int(a.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>
    binary_op_impl(const F& a, const Seed<T, NVARS, NORDER, Layout::Nested>& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(b.nvars()), .order = int(b.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

    template<template<typename> typename RuleStruct, typename F, typename T, int NVARS, int NORDER>
    requires (::xdiff::detail::isScalarOperand<F, T>)
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, NVARS, NORDER, Layout::Nested>
    binary_op_impl(const Seed<T, NVARS, NORDER, Layout::Nested>& a, const F& b) {
        Dual<T, NVARS, NORDER, Layout::Nested> out(MakeDual{.nvars = int(a.nvars()), .order = int(a.order())});
        binary_assign_impl<RuleStruct>(out, a, b);
        return out;
    }

};

} // namespace xdiff::detail







// ------------------------------ Operator macros ------------------------------

// Unary operation
#define XDIFF_DEFINE_NESTED_DUAL_UNARY_OPERATION(NAME, ASSIGN_NAME, STRUCT) \
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE \
Dual<T, NVARS, NORDER, Layout::Nested>& ASSIGN_NAME( \
    Dual<T, NVARS, NORDER, Layout::Nested>& out, \
    const Dual<T, NVARS, NORDER, Layout::Nested>& arg) { \
    return xdiff::detail::NestedDualOperationHelper::unary_assign_impl<STRUCT>(out, arg); \
} \
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE \
Dual<T, NVARS, NORDER, Layout::Nested>& ASSIGN_NAME( \
    Dual<T, NVARS, NORDER, Layout::Nested>& out, \
    const Seed<T, NVARS, NORDER, Layout::Nested>& arg) { \
    return xdiff::detail::NestedDualOperationHelper::unary_assign_impl<STRUCT>(out, arg); \
} \
\
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const Dual<T, NVARS, NORDER, Layout::Nested>& arg) { \
    return xdiff::detail::NestedDualOperationHelper::unary_op_impl<STRUCT>(arg); \
}\
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const Seed<T, NVARS, NORDER, Layout::Nested>& arg) { \
    return xdiff::detail::NestedDualOperationHelper::unary_op_impl<STRUCT>(arg); \
}




// Binary operation
#define XDIFF_DEFINE_NESTED_DUAL_BINARY_OPERATION(NAME, ASSIGN_NAME, STRUCT) \
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& a, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& a, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const F& a,\
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const F& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const F& a,\
    const Seed<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE Dual<T,NVARS,NORDER,Layout::Nested>& ASSIGN_NAME( \
    Dual<T,NVARS,NORDER,Layout::Nested>& out, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& a, \
    const F& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_assign_impl<STRUCT>(out, a, b); \
} \
\
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
} \
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const Seed<T,NVARS,NORDER,Layout::Nested>& a, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
} \
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
} \
\
\
template<typename T, int NVARS, int NORDER> \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const Seed<T,NVARS,NORDER,Layout::Nested>& a, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
} \
\
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const F& a, \
    const Dual<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
} \
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const Dual<T,NVARS,NORDER,Layout::Nested>& a, \
    const F& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
} \
\
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const F& a, \
    const Seed<T,NVARS,NORDER,Layout::Nested>& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
} \
template<typename F, typename T, int NVARS, int NORDER> \
requires (::xdiff::detail::isScalarOperand<F, T>) \
XDIFF_INLINE_HOST_DEVICE auto NAME( \
    const Seed<T,NVARS,NORDER,Layout::Nested>& a, \
    const F& b){ \
    return xdiff::detail::NestedDualOperationHelper::binary_op_impl<STRUCT>(a, b); \
}



#endif //XDIFF_DUAL_NESTED_OPERATOR_TEMPLATE_HPP