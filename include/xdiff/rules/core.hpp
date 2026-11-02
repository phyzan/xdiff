#ifndef XDIFF_RULES_CORE_HPP
#define XDIFF_RULES_CORE_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <lazex/lazex.hpp>
#include "../backend.hpp"

#define XDIFF_USING_MATH LAZEX_USING_MATH

namespace xdiff{

// So that xdiff's own generic code gets the same overload set it asks users to adopt,
// independently of which headers a translation unit happened to include first.
LAZEX_USING_MATH;

// Defined in dual/dual.hpp. Declared here for scalar_of, which the rules below call.
enum class Layout : uint8_t;

template<typename T, int NVARS, int NORDER, Layout LY>
class Dual;

template<typename T, int NVARS, int NORDER, Layout LY>
class Seed;

struct ZeroValue{};

namespace traits{

template<typename F>
concept isAnyZero = std::is_base_of_v<ZeroValue, std::decay_t<F>>;

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

namespace detail{

// Recognise a Dual or a Seed, including a class derived from one, such as a lazex::LazyType<Dual>.
template<typename T, int NVARS, int NORDER, Layout LY>
std::true_type is_dual_ptr(const Dual<T, NVARS, NORDER, LY>*);
std::false_type is_dual_ptr(const void*);

template<typename T, int NVARS, int NORDER, Layout LY>
std::true_type is_seed_ptr(const Seed<T, NVARS, NORDER, LY>*);
std::false_type is_seed_ptr(const void*);

} // namespace xdiff::detail

/**
 * @brief Returns a Dual or a Seed as it is, and converts any other argument to T.
 *
 * The rules pass a possibly scalar operand through this before a '-' acts on it, so that an
 * integer scalar is not negated in its own type, where an unsigned one wraps around: -2u in
 * d/dx (2u / x), or 0u - 1 in d/dx pow(x, 0u).
 */
template<typename T, typename U>
XDIFF_INLINE_HOST_DEVICE
constexpr decltype(auto) scalar_of(U&& arg){
    using V = std::remove_cvref_t<U>;
    if constexpr (decltype(detail::is_dual_ptr(std::declval<const V*>()))::value) {
        return std::forward<U>(arg);
    } else if constexpr (decltype(detail::is_seed_ptr(std::declval<const V*>()))::value) {
        return std::forward<U>(arg);
    } else {
        return T(std::forward<U>(arg));
    }
}

} // namespace xdiff


namespace xdiff::detail::rules{

XDIFF_USING_MATH;


template<typename T, typename Derived>
struct BaseOperand{

    using value_type = T;
    
    template<typename... F>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const F&... f){
        return Derived::operation(f...);
    }

    template<typename... F, typename... DF>
    XDIFF_INLINE_HOST_DEVICE
    decltype(auto) diff_rule(const DiffPair<F, DF>&... pairs) const {
        return Derived::diff_rule(pairs...);
    }

};

// =============================================================================
// Concrete operation implementations
// =============================================================================



template<typename T>
struct Add : BaseOperand<T, Add<T>>{

    using Base = BaseOperand<T, Add<T>>;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left + right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static decltype(auto) diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return 0;  // 0 + 0 = 0
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            return b.grad;     // 0 + db = db
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            return a.grad;     // da + 0 = da
        } else {
            return a.grad + b.grad;
        }
    }

};


template<typename T>
struct Sub : BaseOperand<T, Sub<T>>{

    using Base = BaseOperand<T, Sub<T>>;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left - right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static decltype(auto) diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return 0;  // 0 - 0 = 0
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            return -b.grad;    // 0 - db = -db
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            return a.grad;     // da - 0 = da
        } else {
            return a.grad - b.grad;
        }
    }

};


template<typename T>
struct Mul : BaseOperand<T, Mul<T>>{

    using Base = BaseOperand<T, Mul<T>>;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left * right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return 0;  // const * const: derivative is 0
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            return a.value * b.grad;  // const * f: c * df
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            return a.grad * b.value;  // f * const: df * c
        } else {
            return a.grad * b.value + a.value * b.grad;  // Full product rule
        }
    }

};


template<typename T>
struct Div : BaseOperand<T, Div<T>>{

    using Base = BaseOperand<T, Div<T>>;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left / right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return 0;  // const / const: derivative is 0
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            // c / f: d/dx = -c * df / f²
            return -scalar_of<T>(a.value) * b.grad / (b.value * b.value);
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            // f / c: d/dx = df / c
            return a.grad / b.value;
        } else {
            // Full quotient rule
            return (a.grad * b.value - a.value * b.grad) / (b.value * b.value);
        }
    }

};



template<typename T>
struct Pow : BaseOperand<T, Pow<T>>{

    using Base = BaseOperand<T, Pow<T>>;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return pow(left, right);
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return 0;  // const^const: derivative is 0
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            // a^const: d/dx = const * a^(const-1) * da
            return a.grad * b.value * pow(a.value, scalar_of<T>(b.value) - 1);
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            // const^b: d/dx = const^b * log(const) * db
            return a.value * pow(a.value, b.value - 1) * log(a.value) * b.grad;
        } else {
            // General: a^b * (da*b/a + db*log(a))
            return pow(a.value, b.value) * (a.grad * b.value / a.value + b.grad * log(a.value));
        }
    }

};


/**
 * @brief Base class for unary mathematical functions.
 *
 * Provides the chain rule infrastructure for functions f(g(x)):
 *   d/dx f(g) = f'(g) * dg/dx
 *
 * Derived classes must implement:
 * - operation(arg): The function f
 * - special_diff(value, grad): Returns f'(value) * grad
 *
 * @tparam Derived The derived function type (Log, etc.)
 * @tparam T The scalar value type
 */
template<typename Derived, typename T>
struct MathFunc : BaseOperand<T, Derived>{

    using Base = BaseOperand<T, Derived>;

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<F, DF>& a) {
        if constexpr (DiffPair<F, DF>::isTrivial()) {
            return 0;  // Constant argument: derivative is 0
        } else {
            return Derived::special_diff(a.value, a.grad);
        }
    }

    /// @brief Override this in derived classes to define f'(x) * dx
    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return Derived::special_diff(f, df);
    }

};




} // namespace xdiff::detail::rules



#endif // XDIFF_RULES_CORE_HPP