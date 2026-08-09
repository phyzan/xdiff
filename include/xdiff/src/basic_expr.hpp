/**
 * @file basic_expr.hpp
 * @brief Expression construction functions with algebraic simplification.
 *
 * This file provides the make_* functions that construct expression trees
 * with compile-time algebraic simplification. These optimizations include:
 *
 * - Identity elimination: 0 + a = a, 1 * a = a, a / 1 = a
 * - Zero propagation: 0 * a = 0, 0 / a = 0
 * - Constant folding: const + const = const
 * - Associativity: (const + (const + x)) = (const+const) + x
 * - Negation simplification: -(-x) = x, -(a - b) = b - a
 * - Division simplification: (a/b)/c = a/(b*c), a/(b/c) = (a*c)/b
 *
 * These optimizations are performed at compile time using template metaprogramming,
 * resulting in more efficient expression trees without runtime overhead.
 */
#ifndef XDIFF_BASIC_EXPR_HPP
#define XDIFF_BASIC_EXPR_HPP


#include <ostream>
#include "decl.hpp"
#include "utils.hpp"


// =============================================================================
// Operator macros for expression templates
// =============================================================================

/**
 * @brief Macro to define unary operators for expression types.
 *
 * @param NAME The operator function name
 * @param MAKE_FUNC The make_* function to use for construction
 */
#define XDIFF_MAKE_EXPR_UNARY_OPERATOR(NAME, MAKE_FUNC) \
template<typename T, typename F> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const detail::Expr<F, T>& arg){ \
    return detail::MAKE_FUNC<T>(static_cast<const F&>(arg)); \
} \

/**
 * @brief Macro to define binary operators for expression types.
 *
 * Generates three overloads:
 * - (Expr, Expr): Both operands are expressions
 * - (Expr, scalar): Left is expression, right is scalar
 * - (scalar, Expr): Left is scalar, right is expression
 *
 * @param NAME The operator function name
 * @param MAKE_FUNC The make_* function to use for construction
 */
#define XDIFF_MAKE_EXPR_BINARY_OPERATOR(NAME, MAKE_FUNC) \
template<typename T, typename A, typename B> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const detail::Expr<A, T>& left, const detail::Expr<B, T>& right){ \
    return detail::MAKE_FUNC<T>(static_cast<const A&>(left), static_cast<const B&>(right)); \
} \
template<typename T, typename A, std::convertible_to<T> B> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const detail::Expr<A, T>& left, const B& right){ \
    return detail::MAKE_FUNC<T>(static_cast<const A&>(left), right); \
} \
template<typename T, std::convertible_to<T> A, typename B> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const A& left, const detail::Expr<B, T>& right){ \
    return detail::MAKE_FUNC<T>(left, static_cast<const B&>(right)); \
} \

namespace xdiff::detail{


// =============================================================================
// Expression construction functions with simplification
// =============================================================================

/**
 * @brief Converts a value to an expression.
 *
 * - Dual -> LazyDual (wraps for lazy evaluation)
 * - Expr -> returns as-is (already an expression)
 * - scalar -> Number (wraps as constant)
 *
 * @tparam T The scalar value type
 * @tparam R The input type
 * @param value The value to convert
 * @return An expression type
 */
template<typename T, typename R>
XDIFF_INLINE_HOST_DEVICE
decltype(auto) make_expr(R&& value){
    if constexpr (traits::isDual<R, T>) {
        return LazyDual<T, R::NVARS, R::NORDER>(std::forward<R>(value));
    } else if constexpr (traits::isExpr<R, T>) {
        return std::forward<R>(value);
    } else {
        return Number<T>(std::forward<R>(value));
    }
}


/**
 * @brief Constructs an addition expression with simplification.
 *
 * Simplifications applied:
 * - 0 + a = a
 * - a + 0 = a
 * - const + const = folded constant
 * - const + (const + x) = (const+const) + x (associativity)
 *
 * @tparam T The scalar value type
 * @tparam L Left operand type
 * @tparam R Right operand type
 * @return Simplified expression (may be AddExpr, or a simpler type)
 */
template<typename T, typename L, typename R>
XDIFF_INLINE_HOST_DEVICE
auto make_add(L&& lhs, R&& rhs){
    auto a = make_expr<T>(std::forward<L>(lhs));
    auto b = make_expr<T>(std::forward<R>(rhs));
    using Ta = decltype(a);
    using Tb = decltype(b);
    if constexpr (traits::isZero<Ta, T>) {
        return b;  // 0 + b = b
    } else if constexpr (traits::isZero<Tb, T>) {
        return a;  // a + 0 = a
    } else if constexpr (traits::isConstant<Ta, T> && traits::isConstant<Tb, T>) {
        return Number<T>(a.value() + b.value());  // Constant folding
    }
    // Associative: const + (const + x) → (const+const) + x
    else if constexpr (traits::isConstant<Ta, T> && traits::isAdd<Tb, T>) {
        if constexpr (traits::isConstant<typename Tb::LeftType, T>) {
            return make_add<T>(a + b.left(), b.right());
        } else if constexpr (traits::isConstant<typename Tb::RightType, T>) {
            return make_add<T>(a + b.right(), b.left());
        } else {
            return AddExpr<T, Ta, Tb>(std::move(a), std::move(b));
        }
    }
    // Associative: (const + x) + const → (const+const) + x
    else if constexpr (traits::isAdd<Ta, T> && traits::isConstant<Tb, T>) {
        if constexpr (traits::isConstant<typename Ta::LeftType, T>) {
            return make_add<T>(a.left() + b, a.right());
        } else if constexpr (traits::isConstant<typename Ta::RightType, T>) {
            return make_add<T>(a.right() + b, a.left());
        } else {
            return AddExpr<T, Ta, Tb>(std::move(a), std::move(b));
        }
    } else {
        return AddExpr<T, Ta, Tb>(std::move(a), std::move(b));
    }
}


/**
 * @brief Constructs a subtraction expression with simplification.
 *
 * Simplifications applied:
 * - 0 - a = -a
 * - a - 0 = a
 * - a - (-b) = a + b
 * - (-a) - b = -(a + b)
 * - const - const = folded constant
 */
template<typename T, typename L, typename R>
XDIFF_INLINE_HOST_DEVICE
auto make_sub(L&& lhs, R&& rhs){
    auto a = make_expr<T>(std::forward<L>(lhs));
    auto b = make_expr<T>(std::forward<R>(rhs));
    using Ta = decltype(a);
    using Tb = decltype(b);
    if constexpr (traits::isZero<Ta, T>) {
        return make_neg<T>(std::move(b));  // 0 - b = -b
    } else if constexpr (traits::isZero<Tb, T>) {
        return a;  // a - 0 = a
    }
    // a - (-b) → a + b
    else if constexpr (traits::isNeg<Tb, T>) {
        return make_add<T>(std::move(a), b.arg());
    }
    // (-a) - b → -(a + b)
    else if constexpr (traits::isNeg<Ta, T>) {
        return make_neg<T>(make_add<T>(a.arg(), std::move(b)));
    } else if constexpr (traits::isConstant<Ta, T> && traits::isConstant<Tb, T>) {
        return Number<T>(a.value() - b.value());  // Constant folding
    } else {
        return SubExpr<T, Ta, Tb>(std::move(a), std::move(b));
    }
}


/**
 * @brief Constructs a multiplication expression with simplification.
 *
 * Simplifications applied:
 * - 0 * a = 0, a * 0 = 0
 * - 1 * a = a, a * 1 = a
 * - (a/b) * c = (a*c) / b
 * - a * (b/c) = (a*b) / c
 * - const * const = folded constant
 * - const * (const * x) = (const*const) * x (associativity)
 */
template<typename T, typename L, typename R>
XDIFF_INLINE_HOST_DEVICE
auto make_mul(L&& lhs, R&& rhs){
    auto a = make_expr<T>(std::forward<L>(lhs));
    auto b = make_expr<T>(std::forward<R>(rhs));
    using Ta = decltype(a);
    using Tb = decltype(b);
    if constexpr (traits::isZero<Ta, T> || traits::isZero<Tb, T>) {
        return Zero<T>();  // 0 * anything = 0
    } else if constexpr (traits::isOne<Ta, T>) {
        return b;  // 1 * b = b
    } else if constexpr (traits::isOne<Tb, T>) {
        return a;  // a * 1 = a
    }
    // (a/b) * c → (a*c) / b
    else if constexpr (traits::isDiv<Ta, T>) {
        return make_div<T>(make_mul<T>(a.left(), std::move(b)), a.right());
    }
    // a * (b/c) → (a*b) / c
    else if constexpr (traits::isDiv<Tb, T>) {
        return make_div<T>(make_mul<T>(std::move(a), b.left()), b.right());
    } else if constexpr (traits::isConstant<Ta, T> && traits::isConstant<Tb, T>) {
        return Number<T>(a.value() * b.value());  // Constant folding
    }
    // Associative: const * (const * x) → (const*const) * x
    else if constexpr (traits::isConstant<Ta, T> && traits::isMul<Tb, T>) {
        if constexpr (traits::isConstant<typename Tb::LeftType, T>) {
            return make_mul<T>(a * b.left(), b.right());
        } else if constexpr (traits::isConstant<typename Tb::RightType, T>) {
            return make_mul<T>(a * b.right(), b.left());
        } else {
            return MulExpr<T, Ta, Tb>(std::move(a), std::move(b));
        }
    }
    // Associative: (const * x) * const → (const*const) * x
    else if constexpr (traits::isMul<Ta, T> && traits::isConstant<Tb, T>) {
        if constexpr (traits::isConstant<typename Ta::LeftType, T>) {
            return make_mul<T>(a.left() * b, a.right());
        } else if constexpr (traits::isConstant<typename Ta::RightType, T>) {
            return make_mul<T>(a.right() * b, a.left());
        } else {
            return MulExpr<T, Ta, Tb>(std::move(a), std::move(b));
        }
    } else {
        return MulExpr<T, Ta, Tb>(std::move(a), std::move(b));
    }
}


/**
 * @brief Constructs a division expression with simplification.
 *
 * Simplifications applied:
 * - 0 / a = 0
 * - a / 1 = a
 * - (a/b) / c = a / (b*c)
 * - a / (b/c) = (a*c) / b
 * - const / const = folded constant
 */
template<typename T, typename L, typename R>
XDIFF_INLINE_HOST_DEVICE
auto make_div(L&& lhs, R&& rhs){
    auto a = make_expr<T>(std::forward<L>(lhs));
    auto b = make_expr<T>(std::forward<R>(rhs));
    using Ta = decltype(a);
    using Tb = decltype(b);
    if constexpr (traits::isZero<Ta, T>) {
        return Zero<T>();  // 0 / b = 0
    } else if constexpr (traits::isOne<Tb, T>) {
        return a;  // a / 1 = a
    }
    // (a/b) / c → a / (b*c)
    else if constexpr (traits::isDiv<Ta, T>) {
        return make_div<T>(a.left(), make_mul<T>(a.right(), std::move(b)));
    }
    // a / (b/c) → (a*c) / b
    else if constexpr (traits::isDiv<Tb, T>) {
        return make_div<T>(make_mul<T>(std::move(a), b.right()), b.left());
    }
    else if constexpr (traits::isConstant<Ta, T> && traits::isConstant<Tb, T>) {
        return Number<T>(a.value() / b.value());  // Constant folding
    }
    else {
        return DivExpr<T, Ta, Tb>(std::move(a), std::move(b));
    }
}


/**
 * @brief Constructs a power expression with simplification.
 *
 * Simplifications applied:
 * - 0^n = 0 (for n != 0)
 * - a^1 = a
 * - const^const = folded constant
 */
template<typename T, typename L, typename R>
XDIFF_INLINE_HOST_DEVICE
auto make_pow(L&& lhs, R&& rhs){
    auto a = make_expr<T>(std::forward<L>(lhs));
    auto b = make_expr<T>(std::forward<R>(rhs));
    using Ta = decltype(a);
    using Tb = decltype(b);
    if constexpr (traits::isZero<Ta, T> && !traits::isZero<Tb, T>) {
        return Zero<T>();  // 0^n = 0
    } else if constexpr (traits::isOne<Tb, T>) {
        return a;  // a^1 = a
    } else if constexpr (traits::isConstant<Ta, T> && traits::isConstant<Tb, T>) {
        return Number<T>(pow(a.value(), b.value()));  // Constant folding
    } else {
        return PowExpr<T, Ta, Tb>(std::move(a), std::move(b));
    }
}


/**
 * @brief Constructs a negation expression with simplification.
 *
 * Simplifications applied:
 * - -0 = 0
 * - -(-x) = x (double negation)
 * - -(a - b) = b - a
 * - -const = folded constant
 */
template<typename T, typename Arg>
XDIFF_INLINE_HOST_DEVICE
auto make_neg(Arg&& arg){
    auto a = make_expr<T>(std::forward<Arg>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isZero<Ta, T>) {
        return Zero<T>();  // -0 = 0
    }
    // -(-x) → x
    else if constexpr (traits::isNeg<Ta, T>) {
        return a.arg();
    }
    // -(a - b) → b - a
    else if constexpr (traits::isSub<Ta, T>) {
        return make_sub<T>(a.right(), a.left());
    } else if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(-a.value());  // Constant folding
    } else {
        return NegExpr<T, Ta>(std::move(a));
    }
}

/// @brief Unary plus (returns the argument unchanged).
template<typename T, traits::isExpr<T> Arg>
XDIFF_INLINE_HOST_DEVICE
auto make_plus(Arg&& arg){
    return std::forward<Arg>(arg);
}

/**
 * @brief Constructs a logarithm expression with simplification.
 *
 * Simplifications applied:
 * - log(const) = folded constant
 * - log(1) = 0
 */
template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_log(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(log(a.value()));  // Constant folding
    } else if constexpr (traits::isOne<Ta, T>) {
        return Zero<T>();  // log(1) = 0
    } else {
        return LogExpr<T, Ta>(std::move(a));
    }
}


// =============================================================================
// Operator overloads for expression templates
// =============================================================================

// Binary operators use the make_* functions with simplification
XDIFF_MAKE_EXPR_BINARY_OPERATOR(operator+, make_add)   ///< Addition operator
XDIFF_MAKE_EXPR_BINARY_OPERATOR(operator-, make_sub)   ///< Subtraction operator
XDIFF_MAKE_EXPR_BINARY_OPERATOR(operator*, make_mul)   ///< Multiplication operator
XDIFF_MAKE_EXPR_BINARY_OPERATOR(operator/, make_div)   ///< Division operator
XDIFF_MAKE_EXPR_BINARY_OPERATOR(pow, make_pow)         ///< Power function
XDIFF_MAKE_EXPR_UNARY_OPERATOR(log, make_log)          ///< Natural logarithm

// Unary operators
XDIFF_MAKE_EXPR_UNARY_OPERATOR(operator+, make_plus)   ///< Unary plus
XDIFF_MAKE_EXPR_UNARY_OPERATOR(operator-, make_neg)    ///< Negation


/// @brief Stream output operator for expressions (outputs the evaluated value).
template<typename T, typename F>
std::ostream& operator<<(std::ostream& os, const detail::Expr<F, T>& expr){
    return os << expr.value();
}


} // namespace xdiff::detail


#endif // XDIFF_BASIC_EXPR_HPP