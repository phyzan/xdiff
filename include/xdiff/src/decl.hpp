/**
 * @file decl.hpp
 * @brief Forward declarations, macros, type traits, and concepts for XDiff.
 *
 * This file provides the foundational building blocks for the XDiff library:
 * - Preprocessor macros for defining operators on Dual types
 * - Forward declarations of all core classes
 * - Type traits and concepts for expression template metaprogramming
 *
 * The type system distinguishes between:
 * - Expr: Base class for all expression types
 * - Node: Expression nodes with arguments (unary or binary operations)
 * - Atom: Leaf expressions (variables, constants)
 * - Dual: Direct evaluation dual numbers storing all derivatives
 * - LazyDual: Lazy wrapper around Dual for expression templates
 */
#ifndef XDIFF_DECL_HPP
#define XDIFF_DECL_HPP


#include <type_traits>
#include <cstdlib>
#include <cmath>


// =============================================================================
// Operator definition macros
// =============================================================================

/**
 * @brief Macro to define binary operators for Dual types.
 *
 * Generates three overloads:
 * - (Dual, Dual): Both operands are Dual numbers
 * - (Dual, scalar): Left operand is Dual, right is scalar
 * - (scalar, Dual): Left operand is scalar, right is Dual
 *
 * @param NAME The operator function name (e.g., operator+)
 * @param OPERAND The operation struct name from xdiff::detail::operations
 */
#define XDIFF_OPERATOR(NAME, OPERAND) \
template<typename T, size_t Nvars, size_t Norder> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const Dual<T, Nvars, Norder>& a, const Dual<T, Nvars, Norder>& b){\
    Dual<T, Nvars, Norder> out;\
    xdiff::detail::operations::OPERAND<T>::optimized_eval(out, a, b);\
    return out;\
}\
template<typename B, typename T, size_t Nvars, size_t Norder> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const Dual<T, Nvars, Norder>& a, const B& b){\
    Dual<T, Nvars, Norder> out;\
    xdiff::detail::operations::OPERAND<T>::optimized_eval(out, a, b);\
    return out;\
}\
\
template<typename A, typename T, size_t Nvars, size_t Norder> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const A& a, const Dual<T, Nvars, Norder>& b){\
    Dual<T, Nvars, Norder> out;\
    xdiff::detail::operations::OPERAND<T>::optimized_eval(out, a, b);\
    return out;\
}

/**
 * @brief Macro to define compound assignment operators (+=, -=, *=, /=) for AutoDiff types.
 */
#define XDIFF_COMPOUND_OPERATOR(NAME, OPERAND) \
template<typename T, size_t Nvars, size_t Norder> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(Dual<T, Nvars, Norder>& a, const Dual<T, Nvars, Norder>& b){\
    xdiff::detail::operations::OPERAND<T>::compound_assign_to(a, b);\
    return a;\
}\
template<typename B, typename T, size_t Nvars, size_t Norder> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(Dual<T, Nvars, Norder>& a, const B& b){\
    xdiff::detail::operations::OPERAND<T>::compound_assign_to(a, b);\
    return a;\
}

/**
 * @brief Macro to define unary mathematical functions (exp, log, etc.) for AutoDiff types.
 */
#define XDIFF_MATHFUNC_DUAL(NAME, OPERAND)\
template<typename T, size_t Nvars, size_t Norder> \
XDIFF_INLINE_HOST_DEVICE \
auto NAME(const Dual<T, Nvars, Norder>& x){ \
    Dual<T, Nvars, Norder> out;\
    xdiff::detail::operations::OPERAND<T>::optimized_eval(out, x);\
    return out;\
}



/**
* @brief Macro to define comparison operators (==, !=, <, <=, >, >=) for AutoDiff types. These compare only the values of the Dual numbers, not their derivatives.
*/
#define XDIFF_DUAL_COMPARISON_OPERATOR(OP) \
template<typename T, size_t Norder, size_t Nvars> \
XDIFF_INLINE_HOST_DEVICE bool operator OP(const Dual<T, Nvars, Norder>& a, const Dual<T, Nvars, Norder>& b){ \
    return a.value() OP b.value(); \
} \
template<typename T, size_t Nvars, size_t Norder, typename U> \
XDIFF_INLINE_HOST_DEVICE bool operator OP(const Dual<T, Nvars, Norder>& a, const U& b){ \
    if constexpr (detail::traits::isExpr<U, T>) { \
        return a.value() OP b.value(); \
    } else { \
        return a.value() OP b; \
    } \
} \
template<typename T, size_t Nvars, size_t Norder, typename U> \
XDIFF_INLINE_HOST_DEVICE bool operator OP(const U& a, const Dual<T, Nvars, Norder>& b){ \
    if constexpr (detail::traits::isExpr<U, T>) { \
        return a.value() OP b.value(); \
    } else { \
        return a OP b.value(); \
    } \
}


namespace xdiff{

// Import standard math functions into xdiff namespace for ADL
using std::pow, std::log, std::log10, std::sqrt, std::exp, std::sin, std::cos, std::tan, std::abs, std::sinh, std::cosh, std::tanh, std::erf, std::asin, std::acos, std::atan, std::asinh, std::acosh, std::atanh;

// =============================================================================
// Forward declarations of core types
// =============================================================================

/**
 * @brief Dual number storing all partial derivatives up to a specified order.
 *
 * The primary type for direct automatic differentiation. Stores the function value
 * and all partial derivatives up to order Nord for Nvars independent variables.
 *
 * @tparam T The scalar type (e.g., double, float)
 * @tparam Nvars Number of independent variables
 * @tparam Nord Maximum derivative order to compute
 */
template<typename T, size_t Nvars, size_t Nord>
class Dual;

/**
 * @brief Lazy wrapper around Dual for use in expression templates.
 *
 * Wraps a Dual and provides the Expr interface, enabling lazy evaluation
 * and expression optimization before final computation.
 *
 * @tparam T The scalar type
 * @tparam Nvars Number of independent variables
 * @tparam Nord Maximum derivative order
 */
template<typename T, size_t Nvars, size_t Nord>
class LazyDual;

/**
 * @brief Runtime or compile-time variable for expression templates.
 *
 * Represents an independent variable in a symbolic expression. Unlike Dual,
 * Variable can be differentiated indefinitely and only stores its value and axis.
 *
 * @tparam T The scalar type
 * @tparam Axis The variable index (-1 for runtime, >= 0 for compile-time)
 */
template<typename T, int Axis>
class Variable;

/**
 * @brief Compile-time symbol for differentiation axes.
 *
 * Used to specify which variable to differentiate with respect to. Implicitly
 * convertible to size_t for derivative indexing. Does not inherit from Expr.
 *
 * @tparam Axis The axis index (0-based)
 *
 * @example
 *     Symbol<0> x;  // First variable
 *     Symbol<1> y;  // Second variable
 *     auto df_dx = f.get_diff_wrt(x);
 */
template<size_t Axis>
struct Symbol;


namespace detail{

// =============================================================================
// Forward declarations of expression template classes
// =============================================================================

/**
 * @brief CRTP base class for all expression types.
 *
 * Provides the common interface for expression templates using the
 * Curiously Recurring Template Pattern (CRTP).
 *
 * @tparam Derived The derived expression type
 * @tparam T The scalar value type
 */
template<typename Derived, typename T>
class Expr;

/**
 * @brief Base class for expression nodes with arguments.
 *
 * Represents operations that take one or more sub-expressions as arguments.
 *
 * @tparam Derived The derived node type
 * @tparam T The scalar value type
 * @tparam Args The argument expression types
 */
template<typename Derived, typename T, typename... Args>
class Node;

/**
 * @brief Base class for unary operations (single argument).
 *
 * @tparam Derived The derived unary operation type
 * @tparam T The scalar value type
 * @tparam Arg The argument expression type
 */
template<typename Derived, typename T, typename Arg>
class Unary;

/**
 * @brief Base class for binary operations (two arguments).
 *
 * @tparam Derived The derived binary operation type
 * @tparam T The scalar value type
 * @tparam L The left operand expression type
 * @tparam R The right operand expression type
 */
template<typename Derived, typename T, typename L, typename R>
class Binary;

/**
 * @brief Base class for atomic (leaf) expressions.
 *
 * Atoms are expressions with no sub-expressions: variables, constants, etc.
 *
 * @tparam Derived The derived atom type
 * @tparam T The scalar value type
 */
template<typename Derived, typename T>
class Atom;

/// @brief Addition expression: left + right
template<typename T, typename L, typename R>
class AddExpr;

/// @brief Subtraction expression: left - right
template<typename T, typename L, typename R>
class SubExpr;

/// @brief Multiplication expression: left * right
template<typename T, typename L, typename R>
class MulExpr;

/// @brief Division expression: left / right
template<typename T, typename L, typename R>
class DivExpr;

/// @brief Power expression: pow(left, right)
template<typename T, typename L, typename R>
class PowExpr;

/// @brief Negation expression: -arg
template<typename T, typename Arg>
class NegExpr;

/// @brief Natural logarithm expression: log(arg)
template<typename T, typename Arg>
class LogExpr;

/**
 * @brief Base class for constant expressions.
 *
 * Constants have zero derivatives with respect to all variables.
 *
 * @tparam Derived The derived constant type
 * @tparam T The scalar value type
 */
template<typename Derived, typename T>
class Constant;

/// @brief Runtime numeric constant wrapping a value of type T
template<typename T>
class Number;

/// @brief Compile-time constant representing zero (enables optimization)
template<typename T>
class Zero;

/// @brief Compile-time constant representing one (enables optimization)
template<typename T>
class One;

/**
 * @brief Pairs a value with its derivative for differentiation rules.
 *
 * Used internally to pass (f, df/dx) pairs through differentiation rules.
 * Enforces the invariant that constants must have zero derivatives.
 *
 * @tparam F The value type
 * @tparam DF The derivative type
 */
template<typename F, typename DF>
struct DiffPair;


// =============================================================================
// Type traits and concepts for expression template metaprogramming
// =============================================================================

namespace traits{

/// @brief Concept: F is an expression type with value_type T
template<typename F, typename T>
concept isExpr = std::is_base_of_v<Expr<std::decay_t<F>, T>, std::decay_t<F>>;

/// @brief Helper for detecting Node types
template<typename F>
constexpr bool HELPER_IS_NODE = false;

template<typename Derived, typename T, typename... Args>
constexpr bool HELPER_IS_NODE<Node<Derived, T, Args...>> = true;

/// @brief Concept: F is a Node expression
template<typename F, typename T>
concept isNode = HELPER_IS_NODE<std::decay_t<F>>;

/// @brief Concept: F is a unary expression (has ArgType member)
template<typename F, typename T>
concept isUnary = requires {typename F::ArgType;} && std::is_base_of_v<Unary<std::decay_t<F>, T, typename F::ArgType>, std::decay_t<F>>;

/// @brief Concept: F is a binary expression (has LeftType and RightType members)
template<typename F, typename T>
concept isBinary = requires {typename F::LeftType; typename F::RightType;} && std::is_base_of_v<Binary<std::decay_t<F>, T, typename F::LeftType, typename F::RightType>, std::decay_t<F>>;

/// @brief Concept: F is an atomic (leaf) expression
template<typename F, typename T>
concept isAtom = std::is_base_of_v<Atom<std::decay_t<F>, T>, std::decay_t<F>>;

/// @brief Concept: F is an AddExpr
template<typename F, typename T>
concept isAdd = std::is_same_v<AddExpr<T, typename F::LeftType, typename F::RightType>, std::decay_t<F>>;

/// @brief Concept: F is a SubExpr
template<typename F, typename T>
concept isSub = std::is_same_v<SubExpr<T, typename F::LeftType, typename F::RightType>, std::decay_t<F>>;

/// @brief Concept: F is a MulExpr
template<typename F, typename T>
concept isMul = std::is_same_v<MulExpr<T, typename F::LeftType, typename F::RightType>, std::decay_t<F>>;

/// @brief Concept: F is a DivExpr
template<typename F, typename T>
concept isDiv = std::is_same_v<DivExpr<T, typename F::LeftType, typename F::RightType>, std::decay_t<F>>;

/// @brief Concept: F is a PowExpr
template<typename F, typename T>
concept isPow = std::is_same_v<PowExpr<T, typename F::LeftType, typename F::RightType>, std::decay_t<F>>;

/// @brief Concept: F is a NegExpr
template<typename F, typename T>
concept isNeg = std::is_same_v<NegExpr<T, typename F::ArgType>, std::decay_t<F>>;

/// @brief Concept: F is a constant expression (derivatives are zero)
template<typename F, typename T>
concept isConstant = std::is_base_of_v<Constant<std::decay_t<F>, T>, std::decay_t<F>>;

/// @brief Concept: F is the Zero constant
template<typename F, typename T>
concept isZero = std::is_same_v<Zero<T>, std::decay_t<F>>;

/// @brief Concept: F is the One constant
template<typename F, typename T>
concept isOne = std::is_same_v<One<T>, std::decay_t<F>>;

/// @brief Concept: F is a Dual type
template<typename F, typename T>
concept isDual = requires {F::NVARS; F::NORDER;} && std::is_base_of_v<Dual<T, F::NVARS, F::NORDER>, std::decay_t<F>>;

/// @brief Concept: F is a LazyDual type
template<typename F, typename T>
concept isLazyDual = requires {F::NVARS; F::NORDER;} && std::is_base_of_v<LazyDual<T, F::NVARS, F::NORDER>, std::decay_t<F>>;

/// @brief Concept: F is a compile-time Symbol
template<typename F>
concept isSymbol = requires {F::AXIS;} && std::is_same_v<Symbol<F::AXIS>, F>;

/// @brief Concept: F can be used as an axis index (Symbol or integral type)
template<typename F>
concept isAxis = ((isSymbol<F> || std::is_integral_v<F>));

/// @brief Concept: F is any expression type (has value_type and satisfies isExpr)
template<typename F>
concept IsAnyExpr = requires { typename std::decay_t<F>::value_type; } && isExpr<std::decay_t<F>, typename std::decay_t<F>::value_type>;

/**
 * @brief Extracts value_type from an expression type, or void if not an expression.
 */
template<typename F, bool = IsAnyExpr<F>>
struct ExtractValueType {
    using type = void;
};

template<typename T>
struct ExtractValueType<T, true> {
    using type = typename T::value_type;
};

/**
 * @brief Deduces the common value_type from two operands.
 *
 * Used by binary operators to determine the result type. Ensures that
 * both operands have compatible value_type if they are expressions.
 *
 * @tparam A First operand type
 * @tparam B Second operand type
 */
template<typename A, typename B>
struct ValueType {

    using typeA = typename ExtractValueType<A>::type;
    using typeB = typename ExtractValueType<B>::type;

    static_assert(
        std::is_void_v<typeA> ||
        std::is_void_v<typeB> ||
        std::is_same_v<typeA, typeB>,
        "A and B have different value_type types"
    );

    using type =
        std::conditional_t<
            !std::is_void_v<typeA>,
            typeA,
            typeB
        >;
};

/// @brief Concept: at least one of A or B is an expression type
template<typename A, typename B>
concept AnyIsExpr = !std::is_void_v<typename ValueType<A, B>::type>;



} // namespace traits

} // namespace detail






} // namespace xdiff




#endif // XDIFF_DECL_HPP