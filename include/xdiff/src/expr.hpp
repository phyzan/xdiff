/**
 * @file expr.hpp
 * @brief Expression template implementations for lazy evaluation and symbolic differentiation.
 *
 * This file implements the expression template system that enables:
 * - Lazy evaluation of arithmetic expressions
 * - Symbolic differentiation via diff_expr()
 * - Efficient assignment to Dual numbers via assign_to()
 *
 * The CRTP (Curiously Recurring Template Pattern) hierarchy:
 *
 *     Expr<Derived, T>           - Base class for all expressions
 *         |
 *         +-- Atom<Derived, T>   - Leaf nodes (no children)
 *         |       |
 *         |       +-- Constant<Derived, T>  - Constants (derivative = 0)
 *         |       |       +-- Number<T>     - Runtime constant value
 *         |       |       +-- Zero<T>       - Compile-time zero
 *         |       |       +-- One<T>        - Compile-time one
 *         |       |
 *         |       +-- Variable<T, Axis>     - Independent variables
 *         |       +-- LazyDual<T, N, O>     - Wrapper around Dual
 *         |
 *         +-- Node<Derived, T, Args...>     - Operations with arguments
 *                 |
 *                 +-- Unary<Derived, T, Arg>       - One argument
 *                 |       +-- NegExpr, LogExpr, ...
 *                 |
 *                 +-- Binary<Derived, T, L, R>    - Two arguments
 *                         +-- AddExpr, SubExpr, MulExpr, DivExpr, PowExpr
 */
#ifndef EXPR_HPP
#define EXPR_HPP


#include <tuple>
#include "decl.hpp"
#include "utils.hpp"
#include "dual.hpp"


namespace xdiff{


namespace detail{

// =============================================================================
// Expression node implementations
// =============================================================================

/**
 * @brief Pairs a value with its derivative for differentiation rules.
 *
 * DiffPair is the core data structure for implementing differentiation rules.
 * It bundles a function value with its derivative, enabling rules like:
 *   d(f*g) = f*dg + df*g
 * to be expressed as operations on DiffPair objects.
 *
 * The isTrivial() method enables compile-time optimization: if the derivative
 * is known to be Zero, many terms can be eliminated entirely.
 *
 * @tparam F The value expression type
 * @tparam DF The derivative expression type
 *
 * @invariant If F is a Constant, DF must be Zero (enforced by static_assert)
 */
template<typename F, typename DF>
struct DiffPair{

    using T = typename traits::ValueType<F, DF>;

    F value;   ///< The function value
    DF grad;   ///< The derivative value

    /// @brief Returns true if this derivative is trivially zero (compile-time optimization)
    XDIFF_INLINE_HOST_DEVICE
    static constexpr bool isTrivial() requires (std::is_same_v<T, void>) {
        return false;
    }

    /// @brief Returns true if this derivative is trivially zero (compile-time optimization)
    XDIFF_INLINE_HOST_DEVICE
    static constexpr bool isTrivial() requires (!std::is_same_v<T, void>) {
        static_assert((traits::isConstant<F, T> && traits::isZero<DF, T>) || (!traits::isConstant<F, T> && !traits::isZero<DF, T>), "If F is a constant, its gradient must be zero");
        return traits::isZero<DF, T>;
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


/**
 * @brief CRTP base class for all expression types.
 *
 * Provides the common interface for expression templates:
 * - value(): Evaluate and return the scalar value
 * - grad(variable): Compute derivative w.r.t. a runtime variable index
 * - diff_expr(Symbol): Return symbolic derivative expression
 * - assign_to(Dual): Evaluate and assign to a Dual number
 *
 * All methods delegate to the derived class via CRTP.
 *
 * @tparam Derived The derived expression type
 * @tparam T The scalar value type
 */
template<typename Derived, typename T>
class Expr{

public:
    using value_type = T;

    // =========================================================================
    // CRTP interface methods (delegate to Derived)
    // =========================================================================

    /// @brief Evaluates and returns the scalar value.
    decltype(auto) value() const {
        return XDIFF_THIS->value();
    }

    /// @brief Evaluates and assigns to a Dual number.
    template<size_t Nargs, size_t Nord>
    void assign_to(Dual<T, Nargs, Nord>& dual) const {
        XDIFF_THIS->assign_to(dual);
    }

    /// @brief Computes the derivative w.r.t. a runtime variable index.
    XDIFF_INLINE_HOST_DEVICE
    decltype(auto) grad(size_t variable) const {
        return XDIFF_THIS->grad(variable);
    }

    /// @brief Returns the symbolic derivative expression w.r.t. a compile-time variable.
    template<size_t I>
    XDIFF_INLINE_HOST_DEVICE
    auto diff_expr(Symbol<I> variable) const {
        return XDIFF_THIS->diff_expr(variable);
    }
};


/**
 * @brief Base class for expression nodes with arguments.
 *
 * Node represents operations that take one or more sub-expressions as arguments
 * (e.g., addition, multiplication, sin, exp). It stores the arguments in a tuple
 * and provides infrastructure for:
 * - Evaluating the operation via value()
 * - Applying differentiation rules via diff_rule()
 * - Efficient assignment to Dual via assign_to()
 *
 * @tparam Derived The derived node type (CRTP)
 * @tparam T The scalar value type
 * @tparam Args The argument expression types
 */
template<typename Derived, typename T, typename... Args>
class Node : public Expr<Derived, T>{

    static constexpr size_t NB = sizeof...(Args);  ///< Number of arguments

#ifdef XDIFF_BACKEND_CPU
    /// Thread-local scratch space for evaluating sub-expressions (CPU only)
    template<size_t Nvars, size_t Nord>
    static inline thread_local std::array<Dual<T, Nvars, Nord>, NB> scratch_dual;
#endif

public:

    /// @brief Evaluates and returns the scalar value.
    T value() const {
        return this->value_impl(std::make_index_sequence<NB>{});
    }

    // =========================================================================
    // Static interface (to be overridden by Derived)
    // =========================================================================

    /**
     * @brief The mathematical operation implemented by this node.
     *
     * Must be overridden by derived classes to define the operation.
     *
     * @param f Evaluated argument values
     * @return The result of the operation
     */
    template<typename... F>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const F&... f){
        static_assert(sizeof...(F)==NB, "Invalid number of arguments for node operation");
        return Derived::operation(f...);
    }

    /**
     * @brief The differentiation rule for this operation.
     *
     * Must be overridden by derived classes. Takes DiffPair arguments containing
     * (value, derivative) pairs and returns the derivative of the operation.
     *
     * @param pairs DiffPair objects for each argument
     * @return The derivative expression or value
     */
    template<typename... F, typename... DF>
    XDIFF_INLINE_HOST_DEVICE
    auto diff_rule(const DiffPair<F, DF>&... pairs) const {
        static_assert(sizeof...(pairs)==NB, "Invalid number of arguments for node grad");
        return Derived::diff_rule(pairs...);
    }

    /// @brief Evaluates and assigns to a Dual number.
    template<size_t Nargs, size_t Nord>
    XDIFF_INLINE_HOST_DEVICE
    void assign_to(Dual<T, Nargs, Nord>& dual) const {
        assign_to_impl(dual, std::make_index_sequence<NB>{});
    }

protected:

    Node(Args... args) : args_(std::move(args)...) {}

    /// @brief Returns const reference to the argument tuple.
    const std::tuple<Args...>& args() const {
        return args_;
    }

private:

    /// @brief Implementation of value() using index sequence.
    template<size_t... I>
    XDIFF_INLINE_HOST_DEVICE
    T value_impl(std::index_sequence<I...> /**/) const {
        return Derived::operation(std::get<I>(args_).value()...);
    }

    /// @brief Implementation of assign_to() using index sequence.
    template<size_t Nargs, size_t Nord, size_t... I>
    XDIFF_INLINE_HOST_DEVICE
    void assign_to_impl(Dual<T, Nargs, Nord>& dual, std::index_sequence<I...> /**/) const {
        Derived::optimized_eval(dual, prim_obj<I>(dual)...);
    }

    /**
     * @brief Converts an argument to a Dual or scalar for optimized_eval.
     *
     * Handles three cases:
     * - LazyDual: Extract the underlying Dual
     * - Node/non-Constant: Evaluate to a Dual (using scratch space on CPU)
     * - Constant: Just return the scalar value
     */
    template<size_t I, size_t Nargs, size_t Nord>
    XDIFF_INLINE_HOST_DEVICE
    decltype(auto) prim_obj(Dual<T, Nargs, Nord>& /**/) const {
        using arg_t = typename std::tuple_element<I, std::tuple<Args...>>::type;
        const auto& arg = std::get<I>(args());
        if constexpr (traits::isLazyDual<arg_t, T>){
            return static_cast<const Dual<T, Nargs, Nord>&>(arg.dual());
        } else if constexpr (traits::isNode<arg_t, T> || not traits::isConstant<arg_t, T>){
#ifdef XDIFF_BACKEND_CPU
            Dual<T, Nargs, Nord>& worker = scratch_dual<Nargs, Nord>[I];
            arg.assign_to(worker);
            return worker;
#else
            return Dual<T, Nargs, Nord>(arg);
#endif
        } else {
            return arg.value();
        }
    }

    std::tuple<Args...> args_;  ///< Stored argument expressions
};


/**
 * @brief Base class for unary operations (single argument).
 *
 * Implements the differentiation pattern for unary functions f(g(x)):
 *   df/dx = f'(g) * dg/dx
 *
 * Derived classes must implement:
 * - operation(arg): The unary function
 * - special_diff(value, grad): The chain rule component f'(g) * dg
 *
 * @tparam Derived The derived unary operation type
 * @tparam T The scalar value type
 * @tparam Arg The argument expression type
 */
template<typename Derived, typename T, typename Arg>
class Unary : public Node<Derived, T, Arg>{

    static_assert(traits::isExpr<Arg, T>, "Unary operator argument must be an expression node.");
    using Base = Node<Derived, T, Arg>;

public:
    using ArgType = Arg;

    XDIFF_HOST_DEVICE Unary(Arg arg) : Base(std::move(arg)){}

    /**
     * @brief Applies the differentiation rule for unary operations.
     *
     * If the argument's derivative is trivially zero, returns Zero.
     * Otherwise, delegates to Derived::special_diff.
     */
    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<F, DF>& a) {
        if constexpr (DiffPair<F, DF>::isTrivial()) {
            return Zero<T>();
        } else {
            return Derived::special_diff(a.value, a.grad);
        }
    }

    /// @brief Computes the derivative w.r.t. a runtime variable index.
    XDIFF_INLINE_HOST_DEVICE
    T grad(size_t variable) const {
        return diff_rule(DiffPair{arg().value(), arg().grad(variable)});
    }

    /// @brief Returns the symbolic derivative expression.
    template<size_t I>
    XDIFF_INLINE_HOST_DEVICE
    auto diff_expr(Symbol<I> variable) const {
        return diff_rule(DiffPair{arg(), arg().diff_expr(variable)});
    }

    /// @brief Returns the argument expression.
    const Arg& arg() const {
        return std::get<0>(this->args());
    }
};


/**
 * @brief Base class for binary operations (two arguments).
 *
 * Implements differentiation for binary functions f(g(x), h(x)).
 * The specific differentiation rule is defined by Derived::diff_rule.
 *
 * @tparam Derived The derived binary operation type
 * @tparam T The scalar value type
 * @tparam L The left operand expression type
 * @tparam R The right operand expression type
 */
template<typename Derived, typename T, typename L, typename R>
class Binary : public Node<Derived, T, L, R>{

    static_assert(traits::isExpr<L, T>, "Binary operator left argument must be an expression node.");
    static_assert(traits::isExpr<R, T>, "Binary operator right argument must be an expression node.");
    using Base = Node<Derived, T, L, R>;

public:
    using LeftType = L;
    using RightType = R;

    XDIFF_HOST_DEVICE Binary(L left, R right) : Base(std::move(left), std::move(right)){}

    /// @brief Returns the symbolic derivative expression.
    template<size_t I>
    XDIFF_INLINE_HOST_DEVICE
    auto diff_expr(Symbol<I> variable) const {
        return diff_rule(DiffPair{left(), left().diff_expr(variable)}, DiffPair{right(), right().diff_expr(variable)});
    }

    /// @brief Computes the derivative w.r.t. a runtime variable index.
    XDIFF_INLINE_HOST_DEVICE
    T grad(size_t variable) const {
        return diff_rule(DiffPair{left().value(), left().grad(variable)}, DiffPair{right().value(), right().grad(variable)});
    }

    /// @brief Applies the differentiation rule (delegates to Derived).
    template<typename A, typename B, typename DA, typename DB>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a, const DiffPair<B, DB>& b) {
        return Derived::diff_rule(a, b);
    }

    /// @brief Returns the left operand expression.
    const L& left() const {
        return std::get<0>(this->args());
    }

    /// @brief Returns the right operand expression.
    const R& right() const {
        return std::get<1>(this->args());
    }

};


/**
 * @brief Specialized base class for binary math operators.
 *
 * Extends Binary with additional interface for mathematical operators.
 * Used by AddExpr, SubExpr, MulExpr, DivExpr, PowExpr.
 */
template<typename Derived, typename T, typename L, typename R>
class BinOp : public Binary<Derived, T, L, R>{

public:

    using Base = Binary<Derived, T, L, R>;

    XDIFF_INLINE_HOST_DEVICE
    BinOp(L left, R right) : Base(std::move(left), std::move(right)){}

};


/**
 * @brief Base class for atomic (leaf) expressions.
 *
 * Atoms are expressions with no sub-expressions: variables, constants, etc.
 * They form the leaves of the expression tree.
 */
template<typename Derived, typename T>
class Atom : public Expr<Derived, T>{};


// =============================================================================
// Concrete expression types
// =============================================================================

/// @brief Addition expression: left + right. Derivative: d(left) + d(right)
template<typename T, typename L, typename R>
class AddExpr : public BinOp<AddExpr<T, L, R>, T, L, R>, public operations::Add<T>{

    using Base = BinOp<AddExpr<T, L, R>, T, L, R>;

public:
    XDIFF_INLINE_HOST_DEVICE
    AddExpr(L left, R right) : Base(std::move(left), std::move(right)){}

    using operations::Add<T>::diff_rule;
    using operations::Add<T>::operation;

};


/// @brief Subtraction expression: left - right. Derivative: d(left) - d(right)
template<typename T, typename L, typename R>
class SubExpr : public BinOp<SubExpr<T, L, R>, T, L, R>, public operations::Sub<T>{

    using Base = BinOp<SubExpr<T, L, R>, T, L, R>;

public:
    XDIFF_INLINE_HOST_DEVICE
    SubExpr(L left, R right) : Base(std::move(left), std::move(right)){}

    using operations::Sub<T>::diff_rule;
    using operations::Sub<T>::operation;
};


/// @brief Multiplication expression: left * right. Derivative: d(left)*right + left*d(right)
template<typename T, typename L, typename R>
class MulExpr : public BinOp<MulExpr<T, L, R>, T, L, R>, public operations::Mul<T>{

public:
    XDIFF_INLINE_HOST_DEVICE
    MulExpr(L left, R right) : BinOp<MulExpr<T, L, R>, T, L, R>(std::move(left), std::move(right)){}

    using operations::Mul<T>::diff_rule;
    using operations::Mul<T>::operation;
};


/// @brief Division expression: left / right. Derivative: (d(left)*right - left*d(right)) / right^2
template<typename T, typename L, typename R>
class DivExpr : public BinOp<DivExpr<T, L, R>, T, L, R>, public operations::Div<T>{

public:
    XDIFF_INLINE_HOST_DEVICE
    DivExpr(L left, R right) : BinOp<DivExpr<T, L, R>, T, L, R>(std::move(left), std::move(right)){}

    using operations::Div<T>::diff_rule;
    using operations::Div<T>::operation;

};


/// @brief Power expression: pow(left, right). Handles both constant and variable exponents.
template<typename T, typename L, typename R>
class PowExpr : public BinOp<PowExpr<T, L, R>, T, L, R>, public operations::Pow<T>{

    using Base = BinOp<PowExpr<T, L, R>, T, L, R>;

public:
    XDIFF_INLINE_HOST_DEVICE
    PowExpr(L left, R right) : BinOp<PowExpr<T, L, R>, T, L, R>(std::move(left), std::move(right)){}

    using operations::Pow<T>::diff_rule;
    using operations::Pow<T>::operation;

};


/// @brief Negation expression: -arg. Derivative: -d(arg)
template<typename T, typename Arg>
class NegExpr : public Unary<NegExpr<T, Arg>, T, Arg>, public operations::Neg<T>{

public:
    XDIFF_INLINE_HOST_DEVICE
    NegExpr(Arg arg) : Unary<NegExpr<T, Arg>, T, Arg>(std::move(arg)){}

    using operations::Neg<T>::diff_rule;
    using operations::Neg<T>::operation;
};


/// @brief Natural logarithm expression: log(arg). Derivative: d(arg) / arg
template<typename T, typename Arg>
class LogExpr : public Unary<LogExpr<T, Arg>, T, Arg>, public operations::Log<T>{

public:
    XDIFF_INLINE_HOST_DEVICE
    LogExpr(Arg arg) : Unary<LogExpr<T, Arg>, T, Arg>(std::move(arg)){}

    using operations::Log<T>::diff_rule;
    using operations::Log<T>::operation;
};


// =============================================================================
// Constant expressions
// =============================================================================

/**
 * @brief Base class for constant expressions.
 *
 * Constants have zero derivatives with respect to all variables.
 * This enables significant compile-time optimization when constants
 * appear in expressions.
 *
 * @tparam Derived The derived constant type (Number, Zero, or One)
 * @tparam T The scalar value type
 */
template<typename Derived, typename T>
class Constant : public Atom<Derived, T>{

public:

    /// @brief Returns zero (constants have zero derivative).
    XDIFF_INLINE_HOST_DEVICE
    T grad(size_t /*axis*/) const {
        return T(0);
    }

    /// @brief Returns Zero expression (compile-time optimization).
    template<size_t I>
    XDIFF_INLINE_HOST_DEVICE
    Zero<T> diff_expr(Symbol<I> /*x*/) const {
        return {};
    }

    /// @brief Assigns to a Dual (value only, derivatives are zero).
    template<size_t Nargs, size_t Nord>
    void assign_to(Dual<T, Nargs, Nord>& dual) const {
        dual = this->value();
    }
};


/**
 * @brief Runtime numeric constant.
 *
 * Wraps a scalar value as an expression. Derivatives are zero.
 */
template<typename T>
class Number : public Constant<Number<T>, T>{

public:
    XDIFF_INLINE_HOST_DEVICE
    Number(T value) : value_(std::move(value)){}

    XDIFF_INLINE_HOST_DEVICE
    const T& value() const {
        return value_;
    }

private:
    T value_;
};


/**
 * @brief Compile-time zero constant.
 *
 * Enables aggressive compile-time optimization. When Zero appears
 * in expressions, entire sub-expressions can be eliminated:
 *   0 + a = a
 *   0 * a = 0
 *   a - 0 = a
 */
template<typename T>
class Zero : public Constant<Zero<T>, T>{

public:
    XDIFF_INLINE_HOST_DEVICE
    Zero() = default;

    XDIFF_INLINE_HOST_DEVICE
    T value() const {
        return 0;
    }
};


/**
 * @brief Compile-time one constant.
 *
 * Enables compile-time optimization:
 *   1 * a = a
 *   a / 1 = a
 *   pow(a, 1) = a
 */
template<typename T>
class One : public Constant<One<T>, T>{

public:
    XDIFF_INLINE_HOST_DEVICE
    One() = default;

    XDIFF_INLINE_HOST_DEVICE
    T value() const {
        return 1;
    }
};



} // namespace detail


// =============================================================================
// Top-level expression types
// =============================================================================

/**
 * @brief Lazy wrapper around Dual for use in expression templates.
 *
 * LazyDual wraps a Dual and provides the Expr interface. This enables
 * Dual numbers to participate in expression templates for lazy evaluation
 * and expression optimization.
 *
 * @tparam T The scalar value type
 * @tparam Nvars Number of independent variables
 * @tparam Nord Maximum derivative order
 */
template<typename T, size_t Nvars, size_t Nord>
class LazyDual : public detail::Atom<LazyDual<T, Nvars, Nord>, std::decay_t<T>>{

    using Base = detail::Atom<LazyDual<T, Nvars, Nord>, T>;

public:

    static constexpr size_t NVARS = Nvars;
    static constexpr size_t NORDER = Nord;

    using DualType = Dual<T, NVARS, NORDER>;

    // =========================================================================
    // Constructors
    // =========================================================================

    /// @brief Default constructor is deleted (must wrap a Dual).
    XDIFF_INLINE_HOST_DEVICE
    LazyDual() = delete;

    /// @brief Constructs from a Dual number.
    XDIFF_INLINE_HOST_DEVICE
    LazyDual(DualType obj) : item_(std::move(obj)){}

    /// @brief Constructs from an expression (evaluates to Dual first).
    template<detail::traits::isExpr<T> F>
    XDIFF_INLINE_HOST_DEVICE
    LazyDual(const F& f) : item_(f){}

    // =========================================================================
    // Accessors
    // =========================================================================

    /// @brief Returns the function value.
    XDIFF_INLINE_HOST_DEVICE
    const T& value() const {
        return item_.value();
    }

    /// @brief Returns a derivative value.
    XDIFF_INLINE_HOST_DEVICE
    const T& grad(size_t variable) const {
        return this->item_.get_diff_wrt(variable);
    }

    /// @brief Returns symbolic derivative (wraps the reduced Dual).
    template<size_t I>
    XDIFF_INLINE_HOST_DEVICE
    auto diff_expr(Symbol<I> variable) const {
        auto f = this->item_.cmpl_reduced_diff(variable);
        return LazyDual<T, decltype(f)::NVARS, decltype(f)::NORDER>(f);
    }

    /// @brief Returns the underlying Dual.
    const DualType& dual() const {
        return item_;
    }

private:
    DualType item_;
};


/**
 * @brief Runtime or compile-time variable for expression templates.
 *
 * Represents an independent variable in a symbolic expression. Unlike Dual,
 * Variable can be differentiated indefinitely and only stores its value and axis.
 *
 * Two modes:
 * - Axis >= 0: Compile-time variable (axis known at compile time)
 * - Axis == -1: Runtime variable (axis specified at construction)
 *
 * @tparam T The scalar value type
 * @tparam Axis The variable index (-1 for runtime, >= 0 for compile-time)
 *
 * @example
 *     // Compile-time variable (more optimized)
 *     Variable<double, 0> x(3.0);  // x = 3.0, axis 0
 *
 *     // Runtime variable (more flexible)
 *     Variable<double, -1> y(2.0, 1);  // y = 2.0, axis 1
 */
template<typename T, int Axis=-1>
class Variable : public detail::Atom<Variable<T, Axis>, T>{

    static_assert(Axis == -1 || Axis >= 0, "Axis must be -1 (for runtime variables) or a non-negative integer (for compile-time variables)");

public:

    /// @brief Constructs a runtime variable (Axis == -1).
    XDIFF_INLINE_HOST_DEVICE
    Variable(T value, size_t axis) requires (Axis == -1): value_(std::move(value)), index_(axis){}

    /// @brief Constructs a compile-time variable (Axis >= 0).
    XDIFF_INLINE_HOST_DEVICE
    Variable(T value) requires (Axis >= 0): value_(std::move(value)){}

    /// @brief Returns the variable's value.
    XDIFF_INLINE_HOST_DEVICE
    const T& value() const {
        return value_;
    }

    /**
     * @brief Returns the symbolic derivative.
     *
     * For compile-time variables: returns One if I == Axis, else Zero.
     * For runtime variables: returns Number(1) or Number(0).
     */
    template<size_t I>
    XDIFF_INLINE_HOST_DEVICE
    auto diff_expr(Symbol<I> /*x*/) const {
        if constexpr (Axis >= 0) {
            if constexpr (I == Axis) {
                return detail::One<T>();
            } else {
                return detail::Zero<T>();
            }
        } else {
            return (I == index_) ? detail::Number<T>(1) : detail::Number<T>(0);
        }
    }

    /// @brief Returns 1 if variable matches this axis, else 0.
    XDIFF_INLINE_HOST_DEVICE
    T grad(size_t variable) const {
        return (variable == index_) ? T(1) : T(0);
    }

    /// @brief Assigns to a Dual, setting the appropriate first derivative to 1.
    template<size_t Nargs, size_t Nord>
    void assign_to(Dual<T, Nargs, Nord>& dual) const {
        if constexpr (Axis >= 0) {
            dual = Dual<T, Nargs, Nord>(Symbol<Axis>{}, value_);
        } else {
            // Runtime axis case
            dual = Dual<T, Nargs, Nord>(value_, index_);
        }
    }

private:
    T value_;
    size_t index_ = Axis >= 0 ? Axis : 0;
};


/**
 * @brief Compile-time symbol for differentiation axes.
 *
 * Used to specify which variable to differentiate with respect to.
 * Implicitly convertible to size_t for derivative indexing.
 *
 * @tparam Axis The axis index (0-based)
 *
 * @example
 *     Symbol<0> x;  // First variable
 *     Symbol<1> y;  // Second variable
 *
 *     Dual<double, 2, 2> f(x, 3.0);  // f = 3.0, df/dx = 1
 *     f.get_diff_wrt(x);              // df/dx
 *     f.get_diff_wrt(x, y);           // d²f/dxdy
 */
template<size_t Axis>
struct Symbol{
    static constexpr char AXIS = Axis;

    /// @brief Implicit conversion to size_t for use in derivative indexing.
    constexpr operator size_t(){
        return Axis;
    }
};



} // namespace xdiff



namespace std {
template<typename T, size_t Nvars, size_t Nord>
class numeric_limits<xdiff::LazyDual<T, Nvars, Nord>> : public numeric_limits<T>{};
}



#endif // EXPR_HPP