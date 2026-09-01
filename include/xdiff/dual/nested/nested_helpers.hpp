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


struct NestedDualOperationHelper;

} // namespace xdiff::detail


#endif // XDIFF_DUAL_NESTED_HELPERS_HPP