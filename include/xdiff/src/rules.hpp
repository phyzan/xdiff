/**
 * @file rules.hpp
 * @brief Differentiation rules and optimized evaluation for Dual operations.
 *
 * This file defines the differentiation rules for basic operations:
 * - Add, Sub: d(a±b) = da ± db
 * - Mul: d(a*b) = a*db + da*b (product rule)
 * - Div: d(a/b) = (da*b - a*db) / b² (quotient rule)
 * - Pow: d(a^b) = a^b * (da*b/a + db*log(a)) (generalized power rule)
 * - Neg: d(-a) = -da
 * - Log: d(log(a)) = da/a
 *
 * Each operation struct provides:
 * - operation(): The forward computation
 * - diff_rule(): The differentiation rule using DiffPair
 * - optimized_eval(): Efficient evaluation into Dual numbers
 * - compound_assign_to(): For compound assignment operators (+=, *=, etc.)
 */
#ifndef XDIFF_OPERATIONS_HPP
#define XDIFF_OPERATIONS_HPP

#include "decl.hpp"
#include "utils.hpp"

namespace xdiff::detail::operations{

// =============================================================================
// Base class for operation structs
// =============================================================================

/**
 * @brief CRTP base class for operation implementations.
 *
 * Provides common infrastructure for all operation types including
 * the recursive algorithm for computing higher-order derivatives.
 *
 * @tparam Derived The derived operation type (Add, Mul, etc.)
 */
template<typename Derived>
struct BaseOperand{

    // =========================================================================
    // Static interface (to be overridden by Derived)
    // =========================================================================

    /**
     * @brief The mathematical operation.
     *
     * Must be overridden by derived classes.
     */
    template<typename... F>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const F&... f){
        return Derived::operation(f...);
    }

    /**
     * @brief The differentiation rule.
     *
     * Takes DiffPair arguments containing (value, derivative) pairs.
     * Must be overridden by derived classes.
     */
    template<typename... F, typename... DF>
    XDIFF_INLINE_HOST_DEVICE
    auto diff_rule(const DiffPair<F, DF>&... pairs) const {
        return Derived::diff_rule(pairs...);
    }

    /**
     * @brief Optimized evaluation into a Dual number.
     *
     * Computes the operation and all derivatives efficiently.
     * Can be overridden for operation-specific optimizations.
     */
    template<typename T, size_t Nvars, size_t Norder, typename... U>
    static void optimized_eval(Dual<T, Nvars, Norder>& out, const U&... f){
        return optimized_eval_impl(out, f...);
    }

    /**
     * @brief Compound assignment implementation.
     *
     * For operations like +=, *=, etc.
     */
    template<typename U, typename T, size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const U& rhs){
        Derived::compound_assign_to(out, rhs);
    }


private:

    /**
     * @brief Implementation of optimized evaluation.
     *
     * Recursively computes all derivatives using the diff_rule.
     * The algorithm:
     * 1. Compute the function value
     * 2. For each variable, compute d/dx_i of the result
     * 3. Assemble derivatives into the output Dual's data array
     *
     * This leverages the recursive structure of higher-order derivatives:
     * d^n/dx^n f(g) can be computed by applying diff_rule to (g, dg, d²g, ...)
     */
    template<typename T, size_t Nvars, size_t Norder, typename... U>
    static void optimized_eval_impl(Dual<T, Nvars, Norder>& out, const U&... f){
        using EV = Evaluator<T, Nvars, Norder>;
        using AD = EV::AD;

        // Compute the function value
        T v = Derived::operation(EV::get_value(f)...);
        out = AD(v);

        if constexpr (Norder > 0) {
            // Compute derivative w.r.t. each variable using the diff_rule
            auto q = [&] XDIFF_DEVICE <size_t I> (auto&&... g) XDIFF_ALWAYS_INLINE {
                return Derived::diff_rule(DiffPair{EV::reduced_value(g), EV::template reduced_diff<I>(g)}...);
            };

            // h[i] contains df/dx_i and all its higher derivatives
            auto h = XDIFF_EXPAND(size_t, Nvars, I,
                return std::array<typename AD::ReducedType, Nvars>{q.template operator()<I>(f...)...};
            );

            size_t n = 1;

            // Assemble derivatives into result array in graded order
            XDIFF_EXPAND(size_t, Norder, Ord,
                auto g = [&] XDIFF_DEVICE <size_t ord>() XDIFF_ALWAYS_INLINE {
                    XDIFF_EXPAND(size_t, Nvars, Ivar,
                        auto R = [&] XDIFF_DEVICE <size_t var>() XDIFF_ALWAYS_INLINE {
                            constexpr size_t Noff_tot = Dual<T, Nvars, AD::REDUCED_ORDER>::offset(ord*(var==Ivar)...);
                            constexpr size_t Nelements = Dual<T, Nvars, AD::REDUCED_ORDER>::ndiffs(ord)-Dual<T, Nvars, AD::REDUCED_ORDER>::local_offset(ord*(var==Ivar)...);
                            utils::copy_array(out.data().data()+n, h[var].data().data()+Noff_tot, Nelements);
                            n+=Nelements;
                        };
                        (R.template operator()<Ivar>(), ...);
                    );
                };
                (g.template operator()<Ord>(), ...);
            );
        }
    }

    /**
     * @brief Helper struct for evaluating arguments in operations.
     *
     * Provides utilities for extracting values and derivatives from
     * different argument types (Dual, Expr, or scalar).
     */
    template<typename T, size_t Nvars, size_t Norder>
    struct Evaluator {
        using AD = Dual<T, Nvars, Norder>;

        /// @brief Identity for Dual arguments.
        XDIFF_INLINE_DEVICE
        static decltype(auto) masked_value(const AD& item){
            return item;
        }

        /// @brief Converts scalars to the appropriate numeric type.
        template<typename U>
        XDIFF_INLINE_DEVICE
        static decltype(auto) masked_value(U&& item){
            return T(item);
        }

        /// @brief Extracts the scalar value from a Dual.
        template<traits::isDual<T> F>
        XDIFF_INLINE_DEVICE
        static T get_value(const F& f) {
            return f.value();
        }

        /// @brief Extracts the scalar value from an Expr.
        template<traits::isExpr<T> F>
        XDIFF_INLINE_DEVICE
        static T get_value(const F& f) {
            return f.value();
        }

        /// @brief Converts a scalar argument to T.
        template<typename ArgType>
        XDIFF_INLINE_DEVICE
        static T get_value(const ArgType& f) {
            static_assert(std::is_convertible_v<ArgType, T>, "Invalid argument");
            return masked_value(f);
        }

        /// @brief Gets the reduced-order representation of a Dual.
        template<size_t Nv, size_t No>
        XDIFF_INLINE_DEVICE
        static auto reduced_value(const Dual<T, Nv, No>& f){
            return f.reduced();
        }

        /// @brief Evaluates an Expr to a reduced-order Dual.
        template<traits::isExpr<T> F>
        XDIFF_INLINE_DEVICE
        static auto reduced_value(const F& f){
            return typename AD::ReducedType(f);
        }

        /// @brief Scalars have trivial reduced representation.
        template<typename U>
        XDIFF_INLINE_DEVICE
        static auto reduced_value(const U& f){
            return T(f);
        }

        /// @brief Gets the reduced derivative w.r.t. variable I from a Dual.
        template<size_t I>
        XDIFF_INLINE_DEVICE
        static auto reduced_diff(const AD& f){
            return f.reduced_diff_wrt(Symbol<I>());
        }

        /// @brief Scalars have zero derivative.
        template<size_t I>
        XDIFF_INLINE_DEVICE
        static auto reduced_diff(const T& /*f*/){
            return Zero<T>();
        }

    };

};

// =============================================================================
// Concrete operation implementations
// =============================================================================

/**
 * @brief Addition operation: a + b
 *
 * Differentiation rule: d(a + b) = da + db
 *
 * Optimizations:
 * - If da = 0: returns db
 * - If db = 0: returns da
 * - If both = 0: returns Zero
 */
template<typename T>
struct Add : BaseOperand<Add<T>>{

    using Base = BaseOperand<Add<T>>;
    using Base::optimized_eval;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left + right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return Zero<T>();  // 0 + 0 = 0
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            return b.grad;     // 0 + db = db
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            return a.grad;     // da + 0 = da
        } else {
            return a.grad + b.grad;
        }
    };

    /// @brief Compound addition: out += rhs (element-wise)
    template<size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const Dual<T, Nvars, Norder>& rhs){
        for (size_t i=0; i<Dual<T, Nvars, Norder>::NTOT; i++){
            out.data()[i] += rhs.data()[i];
        }
    }

    /// @brief Compound addition with scalar: out += scalar (value only)
    template<typename U, size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const U& rhs){
        out.data()[0] += rhs;
    }

};


/**
 * @brief Subtraction operation: a - b
 *
 * Differentiation rule: d(a - b) = da - db
 *
 * Optimizations:
 * - If da = 0: returns -db
 * - If db = 0: returns da
 * - If both = 0: returns Zero
 */
template<typename T>
struct Sub : BaseOperand<Sub<T>>{

    using Base = BaseOperand<Sub<T>>;
    using Base::optimized_eval;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left - right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return Zero<T>();  // 0 - 0 = 0
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            return -b.grad;    // 0 - db = -db
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            return a.grad;     // da - 0 = da
        } else {
            return a.grad - b.grad;
        }
    };

    template<size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const Dual<T, Nvars, Norder>& rhs){
        for (size_t i=0; i<Dual<T, Nvars, Norder>::NTOT; i++){
            out.data()[i] -= rhs.data()[i];
        }
    }

    template<typename U, size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const U& rhs){
        out.data()[0] -= rhs;
    }

};


/**
 * @brief Multiplication operation: a * b
 *
 * Differentiation rule (product rule): d(a * b) = da * b + a * db
 *
 * Optimizations:
 * - If da = 0: returns a * db
 * - If db = 0: returns da * b
 * - If both = 0: returns Zero
 */
template<typename T>
struct Mul : BaseOperand<Mul<T>>{

    using Base = BaseOperand<Mul<T>>;
    using Base::optimized_eval;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left * right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return Zero<T>();  // const * const: derivative is 0
        } else if constexpr (DiffPair<A, DA>::isTrivial()) {
            return a.value * b.grad;  // const * f: c * df
        } else if constexpr (DiffPair<B, DB>::isTrivial()) {
            return a.grad * b.value;  // f * const: df * c
        } else {
            return a.grad * b.value + a.value * b.grad;  // Full product rule
        }
    };

    template<size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const Dual<T, Nvars, Norder>& rhs){
        out = out * rhs;  // Must use full multiplication for correct derivative propagation
    }

    /// @brief Scalar multiplication is element-wise
    template<typename U, size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const U& rhs){
        for (size_t i=0; i<Dual<T, Nvars, Norder>::NTOT; i++){
            out.data()[i] *= rhs;
        }
    }

};


/**
 * @brief Division operation: a / b
 *
 * Differentiation rule (quotient rule): d(a/b) = (da*b - a*db) / b²
 *
 * Optimizations:
 * - If da = 0: returns -a*db / b²
 * - If db = 0: returns da / b
 * - If both = 0: returns Zero
 */
template<typename T>
struct Div : BaseOperand<Div<T>>{

    using Base = BaseOperand<Div<T>>;
    using Base::optimized_eval;

    template<typename A, typename B>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& left, const B& right){
        return left / right;
    }

    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        if constexpr (DiffPair<A, DA>::isTrivial() && DiffPair<B, DB>::isTrivial()) {
            return Zero<T>();  // const / const: derivative is 0
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

    template<size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const Dual<T, Nvars, Norder>& rhs){
        out = out / rhs;
    }

    /// @brief Scalar division is element-wise
    template<typename U, size_t Nvars, size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    static void compound_assign_to(Dual<T, Nvars, Norder>& out, const U& rhs){
        for (size_t i=0; i<Dual<T, Nvars, Norder>::NTOT; i++){
            out.data()[i] /= rhs;
        }
    }

};


/**
 * @brief Power operation: pow(a, b) = a^b
 *
 * Differentiation rule (generalized power rule):
 * - If b is constant: d(a^b) = b * a^(b-1) * da
 * - If a is constant: d(a^b) = a^b * log(a) * db
 * - General: d(a^b) = a^b * (da*b/a + db*log(a))
 */
template<typename T>
struct Pow : BaseOperand<Pow<T>>{

    using Base = BaseOperand<Pow<T>>;
    using Base::optimized_eval;

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
 * @brief Negation operation: -a
 *
 * Differentiation rule: d(-a) = -da
 */
template<typename T>
struct Neg : BaseOperand<Neg<T>>{

    using Base = BaseOperand<Neg<T>>;
    using Base::optimized_eval;

    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return -arg;
    }

    template<typename A, typename DA>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a) {
        if constexpr (DiffPair<A, DA>::isTrivial()) {
            return Zero<T>();
        } else {
            return -a.grad;
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
struct MathFunc : BaseOperand<Derived>{

    using Base = BaseOperand<Derived>;
    using Base::optimized_eval;

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<F, DF>& a) {
        if constexpr (DiffPair<F, DF>::isTrivial()) {
            return Zero<T>();  // Constant argument: derivative is 0
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

/**
 * @brief Natural logarithm: log(a)
 *
 * Differentiation rule: d(log(a)) = da / a
 */
template<typename T>
struct Log : MathFunc<Log<T>, T>{

    using Base = MathFunc<Log<T>, T>;
    using Base::optimized_eval;

    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return log(arg);
    }

    /// @brief d(log(f)) = df / f
    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df / f;
    }

};



} // namespace operations

#endif // XDIFF_OPERATIONS_HPP