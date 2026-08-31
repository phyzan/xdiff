#ifndef XDIFF_DUAL_NESTED_HELPERS_HPP
#define XDIFF_DUAL_NESTED_HELPERS_HPP


#include "../dual.hpp"
#include <lazy/lazy.hpp>

namespace xdiff::detail {


template<typename T, typename G, size_t NVARS, typename Derived>
class RecursiveDual;

// Helper struct to extract the dual type (or scalar type) that a recursive dual contains
template<typename U>
struct DualInspector {
    static_assert(!std::is_same_v<U, void>, "Invalid type in DualInspector.");
    using type = U;
};

template<typename T, size_t NVARS, size_t NORDER>
struct DualInspector<Dual<T, NVARS, NORDER, Layout::Nested>> {
    using type = Dual<T, NVARS, NORDER, Layout::Nested>;
};

template<typename T, size_t NVARS, size_t NORDER>
struct DualInspector<lazy::LazyType<Dual<T, NVARS, NORDER, Layout::Nested>>> {
    using type = Dual<T, NVARS, NORDER, Layout::Nested>;
};

// Helper to determine the base RecursiveDual for the Dual class
template<typename Derived, typename T, size_t NVARS, size_t NORDER>
struct RecursiveBaseHelper {
#ifdef XDIFF_LAZY_NESTED_DUAL
    using GradType = std::conditional_t<
        NVARS == 0,
        lazy::LazyType<Dual<T, NVARS, NORDER - 1, Layout::Nested>>,
        Dual<T, NVARS, NORDER - 1, Layout::Nested>
    >;
#else
    using GradType = Dual<T, NVARS, NORDER - 1, Layout::Nested>;
#endif
    using type = RecursiveDual<T, GradType, NVARS, Derived>;
};

template<typename Derived, typename T, size_t NVARS>
struct RecursiveBaseHelper<Derived, T, NVARS, 1> {
    using type = RecursiveDual<T, T, NVARS, Derived>;
};

template<typename Derived, typename T, size_t NVARS, size_t NORDER>
using GetRecursiveBase = typename RecursiveBaseHelper<Derived, T, NVARS, NORDER>::type;


// Defined in operator_template.hpp, which is included at the end of this header. Declared here so that Dual can grant them raw access to its storage.

// Dual -> Dual
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
Dual<T, NVARS, NORDER, Layout::Nested>& unary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Dual<T, NVARS, NORDER, Layout::Nested>& arg);

// Seed -> Dual
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
Dual<T, NVARS, NORDER, Layout::Nested>& unary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Seed<T, NVARS, NORDER, Layout::Nested>& arg);

// (Dual, Dual) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Dual<T, NVARS, NORDER, Layout::Nested>& a, const Dual<T, NVARS, NORDER, Layout::Nested>& b);

// (Dual, Seed) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Dual<T, NVARS, NORDER, Layout::Nested>& a, const Seed<T, NVARS, NORDER, Layout::Nested>& b);

// (Seed, Dual) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Seed<T, NVARS, NORDER, Layout::Nested>& a, const Dual<T, NVARS, NORDER, Layout::Nested>& b);

// (Seed, Seed) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename T, size_t NVARS, size_t NORDER>
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Seed<T, NVARS, NORDER, Layout::Nested>& a, const Seed<T, NVARS, NORDER, Layout::Nested>& b);

// (F, Dual) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
requires (isScalarOperand<F, T>)
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const F& a, const Dual<T, NVARS, NORDER, Layout::Nested>& b);

// (Dual, F) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
requires (isScalarOperand<F, T>)
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Dual<T, NVARS, NORDER, Layout::Nested>& a, const F& b);

// (F, Seed) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
requires (isScalarOperand<F, T>)
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const F& a, const Seed<T, NVARS, NORDER, Layout::Nested>& b);

// (Seed, F) -> Dual (binary operations)
template<template<typename> typename RuleStruct, typename F, typename T, size_t NVARS, size_t NORDER>
requires (isScalarOperand<F, T>)
Dual<T, NVARS, NORDER, Layout::Nested>& binary_assign_impl(Dual<T, NVARS, NORDER, Layout::Nested>& out, const Seed<T, NVARS, NORDER, Layout::Nested>& a, const F& b);


} // namespace xdiff::detail


#endif // XDIFF_DUAL_NESTED_HELPERS_HPP