#ifndef XDIFF_RULES_CORE_HPP
#define XDIFF_RULES_CORE_HPP

#include <cmath>
#include "../dual/dual_base.hpp" // IWYU pragma: keep


namespace xdiff::detail::rules{



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
    };

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
    };

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
    };

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
            return -a.value * b.grad / (b.value * b.value);
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            // f / c: d/dx = df / c
            return a.grad / b.value;
        } else {
            // Full quotient rule
            return (a.grad * b.value - a.value * b.grad) / (b.value * b.value);
        }
    };

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
        if constexpr (DiffPair<B, DB>::isTrivial()) {
            // a^const: d/dx = const * a^(const-1) * da
            return a.grad * b.value * pow(a.value, b.value - 1);
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            // const^b: d/dx = const^b * log(const) * db
            return a.value * pow(a.value, b.value - 1) * log(a.value) * b.grad;
        } else {
            // General: a^b * (da*b/a + db*log(a))
            return pow(a.value, b.value) * (a.grad * b.value / a.value + b.grad * log(a.value));
        }
    };

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