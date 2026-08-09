/**
 * @file autodiff.hpp
 * @brief Automatic differentiation library for computing arbitrary-order partial derivatives.
 *
 * This library provides compile-time automatic differentiation capabilities for multivariate
 * functions. It computes exact derivatives (not numerical approximations) using operator
 * overloading and expression templates.
 *
 * @note The number of stored derivatives grows combinatorially as C(Nvars + Norder, Norder),
 *       so high orders with many variables can be memory-intensive.
 */

#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <cassert>
#include <array>
#include <chrono>
#include <numeric>
#include <cmath>
#include "iterators.hpp"

#ifndef AUTODIFF_FAST
#define AUTODIFF_MAYBE_INLINE AUTODIFF_HOST_DEVICE inline
#else
#define AUTODIFF_MAYBE_INLINE AD_INLINE
#endif

#define AUTODIFF AutoDiff<T, Norder, Nvars>

/**
 * @brief Macro to define binary operators for AutoDiff types.
 * Generates three overloads: (AutoDiff, AutoDiff), (AutoDiff, scalar), and (scalar, AutoDiff).
 */
#define OPERATOR(NAME, OPERAND) \
template<typename T, Int Norder, Int Nvars> \
AD_INLINE auto NAME(const AUTODIFF& a, const AUTODIFF& b){\
    return OPERAND::opt_result(a, b);\
}\
template<typename B, typename T, Int Norder, Int Nvars> \
AD_INLINE auto NAME(const AUTODIFF& a, const B& b){\
    return OPERAND::opt_result(a, b);\
}\
\
template<typename A, typename T, Int Norder, Int Nvars> \
AD_INLINE auto NAME(const A& a, const AUTODIFF& b){\
    return OPERAND::opt_result(a, b);\
}

/**
 * @brief Macro to define compound assignment operators (+=, -=, *=, /=) for AutoDiff types.
 */
#define COMPOUND_OPERATOR(NAME, OPERAND) \
template<typename T, Int Norder, Int Nvars> \
AD_INLINE auto NAME(AUTODIFF& a, const AUTODIFF& b){\
    return OPERAND::compound_assign(a, b);\
}\
template<typename B, typename T, Int Norder, Int Nvars> \
AD_INLINE auto NAME(AUTODIFF& a, const B& b){\
    return OPERAND::compound_assign(a, b);\
}

/**
 * @brief Macro to define unary mathematical functions (exp, log, etc.) for AutoDiff types.
 */
#define MATHFUNC(NAME, OPERAND)\
template<typename T, Int Norder, Int Nvars> \
AD_INLINE AUTODIFF NAME(const AUTODIFF& x){ \
    return OPERAND::opt_result(x); \
}

namespace autodiff{ 

using Int = size_t;

/**
 * @brief Finds the index of a value within a parameter pack at compile time.
 * @tparam V The value to search for.
 * @tparam Pack The parameter pack to search within.
 * @return The zero-based index of V in Pack, or sizeof...(Pack) if not found.
 */
template<auto V, auto... Pack>
consteval Int index_of() {
    Int i = 0;
    ((Pack == V ? false : (++i, true)) && ...);
    return i;
}

// Forward declarations for expression types
struct AddExpr;
struct MulExpr;
struct DivExpr;
struct PowExpr;
struct LogExpr;
struct ExpExpr;

/**
 * @brief Compile-time representation of a differentiation variable.
 *
 * Variables are used to specify which partial derivatives to compute.
 * Each variable is identified by a zero-based axis index.
 *
 * @tparam axis The index of this variable (0-based).
 *
 * @example
 *     Variable<0> x;  // First variable
 *     Variable<1> y;  // Second variable
 *     AutoDiff<double, 2, 2> f(3.0, x);  // f = 3.0, tracking derivatives w.r.t. x
 */
template<Int axis>
struct Variable{
    static constexpr char Axis = axis;

    /// @brief Implicit conversion to size_t for use in derivative indexing.
    constexpr operator size_t(){
        return Axis;
    }
};

/// @brief Type trait to detect Variable types.
template<typename T>
inline constexpr bool is_variable_v = false;

template<Int axis>
inline constexpr bool is_variable_v<Variable<axis>> = true;

/**
 * @brief Concept for types usable as variable identifiers.
 * Satisfied by Variable<N> types and integral types.
 */
template<typename... Ts>
concept VarLike = ((is_variable_v<Ts> || std::is_integral_v<Ts>) && ...);

/**
 * @brief Counts occurrences of a variable index within a list.
 * @param x The variable index to count.
 * @param y The list of variable indices to search.
 * @return The number of times x appears in y.
 */
template<VarLike A, VarLike... T>
constexpr Int var_count(A x, T... y){
    return AD_EXPAND(Int, sizeof...(y), I,
        return ((Int(x)==Int(y))+...+0UL);
    );
}


/// @brief Tag type for expression template infrastructure.
template<typename T, Int Norder, Int Nvars>
struct ExprStruct{};

/**
 * @brief Indexing utilities for multivariate derivative storage.
 *
 * Provides methods to compute offsets into the flat derivative array.
 * Derivatives are stored in graded lexicographic order, grouped by total order.
 *
 * @tparam Norder Maximum derivative order.
 * @tparam Nvars Number of independent variables.
 */
template<Int Norder, Int Nvars>
struct MultiDiff{

    /// Total number of unique partial derivatives (including value).
    static constexpr Int Ntot = utils::comb(Nvars+Norder, Norder);

    /**
     * @brief Returns the number of unique derivatives of a given order.
     * @param order The derivative order.
     * @return C(Nvars + order - 1, order), the multiset coefficient.
     */
    AD_INLINE static constexpr Int Ndiffs(Int order);

    /**
     * @brief Computes the local offset within a group of same-order derivatives.
     * @param order Per-variable derivative counts (e.g., {2, 1} means d²/dx² d/dy).
     * @return Index within the group of derivatives of that total order.
     */
    template<std::integral... IntType>
    AD_INLINE static constexpr Int local_offset(IntType... order);

    /**
     * @brief Computes the starting index for derivatives of a given total order.
     * @param order The total derivative order.
     * @return Index where derivatives of this order begin in storage.
     */
    AD_INLINE static constexpr Int global_offset(Int order);

    /**
     * @brief Computes the absolute index of a specific partial derivative.
     * @param order Per-variable derivative counts.
     * @return Index into the flat derivative array.
     */
    template<std::integral... IntType>
    AD_INLINE static constexpr Int offset(IntType... order);

    /// @brief Overload accepting derivative counts as an array.
    AD_INLINE static constexpr Int offset(const std::array<Int, Nvars>& diff_count);

    /**
     * @brief Converts variable indices to per-variable derivative counts.
     * @param x Variable indices (e.g., (0, 1, 1) means d/dx d²/dy²).
     * @return Array where element i is the count of variable i in the input.
     */
    template<VarLike... Var>
    static constexpr std::array<Int, Nvars> diff_count(Var... x){
        return AD_EXPAND(Int, Nvars, I,
            return std::array<Int, Nvars>{var_count(I, x...)...};
        );
    }

};


/**
 * @brief Automatic differentiation type for computing arbitrary-order partial derivatives.
 *
 * Represents a value along with all its partial derivatives up to a specified order.
 * Arithmetic operations automatically propagate derivatives using the chain rule.
 *
 * @tparam T Numeric type (e.g., double, float).
 * @tparam Norder Maximum derivative order to track.
 * @tparam Nvars Number of independent variables.
 *
 * ## Storage Layout
 *
 * Derivatives are stored in a flat array of size C(Nvars + Norder, Norder), grouped
 * by total derivative order. Within each group, derivatives are in graded colexicographic order.
 *
 * For 3 variables (x, y, z) and order 2:
 * ```
 * [f, fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz]
 * ```
 *
 * @example
 *     Variable<0> x;
 *     Variable<1> y;
 *     AutoDiff<double, 2, 2> a(3.0, x);  // a = 3, da/dx = 1, da/dy = 0
 *     AutoDiff<double, 2, 2> b(2.0, y);  // b = 2, db/dx = 0, db/dy = 1
 *     auto f = a * b;                     // f = 6, df/dx = 2, df/dy = 3, d²f/dxdy = 1
 */
template<typename T, Int Norder, Int Nvars>
class AutoDiff : public MultiDiff<Norder, Nvars>{

    using Base = MultiDiff<Norder, Nvars>;

public:

    static constexpr Int ORDER = Norder;     ///< Maximum derivative order.
    static constexpr Int Ntot = MultiDiff<Norder, Nvars>::Ntot;  ///< Total storage size.

    /// Order after one differentiation.
    static constexpr Int ReducedOrder = Norder > 0 ? Norder-1 : 0;

    /// Type with one fewer derivative order.
    using ReducedType = AutoDiff<T, ReducedOrder, Nvars>;

    /// Type after diff_count differentiations.
    template<Int diff_count>
    using Reduced = AutoDiff<T, (Norder > diff_count ? Norder-diff_count : 0), Nvars>;

    using DataType = std::array<T, Ntot>;         ///< Internal storage type.
    using DataStruct = ExprStruct<T, Norder, Nvars>;  ///< Expression template tag.

    template<Int diff_count>
    using ReducedArray = std::array<Int, AutoDiff<T, Norder-diff_count, Nvars>::Ntot>;

    using ValueType = T;  ///< The underlying numeric type.

    /// @brief Default constructor. Initializes value and all derivatives to zero.
    AutoDiff() = default;

    /**
     * @brief Constructs a tracked variable.
     *
     * Creates an AutoDiff representing an independent variable. The first derivative
     * with respect to this variable is set to 1; all others are zero.
     *
     * @tparam I Variable index (0-based, must be < Nvars).
     * @param value Initial value of the variable.
     * @param x Variable tag indicating which variable this represents.
     */
    template<Int I>
    explicit AutoDiff(T value, Variable<I> x);

    /**
     * @brief Constructs from a raw data array.
     *
     * @param data Array containing [value, first derivatives..., second derivatives..., ...].
     *             Must have exactly Ntot elements in the correct storage order.
     */
    explicit AutoDiff(const DataType& data);

    template<std::integral IntType>
    AutoDiff(IntType value);

    /**
     * @brief Constructs a constant (all derivatives zero).
     * @param value The constant value.
     * @note Keep this constructor explicit to avoid accidental conversions that might lead to unintended differentiation or compilation issues.
     */
    explicit AutoDiff(T value);

    /**
     * @brief Returns the function value.
     * @return The value component (zeroth-order derivative).
     */
    AD_INLINE const T& value() const;

    /**
     * @brief Creates a copy with reduced derivative order.
     *
     * Returns an AutoDiff of order Norder-1 containing the value and all
     * derivatives up to that order. Useful for recursive differentiation algorithms.
     *
     * @return AutoDiff with maximum order reduced by 1.
     */
    AD_INLINE ReducedType reduced() const;

    /**
     * @brief Extracts a partial derivative as a new AutoDiff.
     *
     * Returns an AutoDiff whose value is the specified partial derivative of this object,
     * and whose derivatives are the corresponding higher-order derivatives.
     *
     * @tparam IntType Variable index types.
     * @param x Variable indices to differentiate with respect to.
     * @return AutoDiff of reduced order representing the derivative.
     *
     * @example
     *     // For f(x,y,z) stored in obj:
     *     auto dxz = obj.reduced_diff(0, 2);  // Returns d²f/dxdz and its derivatives
     *     // dxz.value() == d²f/dxdz
     *     // dxz.diff_value(0) == d³f/dx²dz
     */
    template<VarLike... IntType>
    requires (sizeof...(IntType)<=Norder)
    AUTODIFF_MAYBE_INLINE Reduced<sizeof...(IntType)> constexpr reduced_diff(IntType... x)const;

    /**
     * @brief Compile-time optimized version of reduced_diff.
     *
     * Same as reduced_diff but with variable indices known at compile time,
     * enabling additional optimizations.
     *
     * @tparam I Variable indices as template parameters.
     * @param x Variable tags (values are ignored, only types matter).
     */
    template<Int... I>
    requires (sizeof...(I)<=Norder)
    AUTODIFF_MAYBE_INLINE Reduced<sizeof...(I)> constexpr cmpl_reduced_diff(Variable<I>... x)const;

    /// @brief Returns a const pointer to the internal storage.
    AD_INLINE const T* data() const;

    /// @brief Returns a mutable pointer to the internal storage.
    AD_INLINE T* data();

    /**
     * @brief Extracts a derivative, returning the same AutoDiff type.
     *
     * Similar to reduced_diff, but returns an object of the same type (same order).
     * Higher-order derivatives that cannot be computed are set to zero.
     * This allows mixing in expressions: `f + f.diff(0, 1)` compiles.
     *
     * @param x Variable indices to differentiate with respect to.
     * @return AutoDiff of same type with derivative as value.
     */
    template<VarLike... IntType>
    AD_INLINE AUTODIFF diff(IntType... x) const;

    /**
     * @brief Returns the numeric value of a specific partial derivative.
     *
     * @param x Variable indices (e.g., diff_value(0, 1) returns d²f/dxdy).
     * @return The derivative value as a scalar.
     */
    template<VarLike... IntType>
    AD_INLINE T diff_value(IntType... x) const;

    operator T() const = delete;  ///< Disable implicit conversion to T to avoid accidental loss of derivative information.

    /// @brief Unary plus (returns copy).
    AUTODIFF operator+()const{
        return *this;
    }

    /// @brief Unary negation (negates all components).
    AUTODIFF operator-() const{
        AUTODIFF res;
        for (Int i=0; i<Ntot; i++){
            res.data()[i] = -_data[i];
        }
        return res;
    }

private:

    /**
     * @brief Retrieves derivative value by per-variable counts.
     * @param order_wrt Derivative count for each variable.
     */
    template<std::integral... IntType>
    AD_INLINE T value_of_diff_counts(IntType... order_wrt)const{
        return _data[Base::offset(order_wrt...)];
    }

    /**
     * @brief Precomputes offsets for reduced_diff extraction.
     * @param x Variables to differentiate with respect to.
     */
    template<VarLike... IntType>
    requires (sizeof...(IntType)<=Norder)
    AUTODIFF_MAYBE_INLINE static constexpr ReducedArray<sizeof...(IntType)> offsets_for_reduced_diff(IntType... x);

    DataType _data{};  ///< Storage for value and all derivatives.

    // Friend declarations for expression types that need private access
    friend struct MulExpr;
};




// ============================================================================
//                         TYPE TRAITS FOR AUTODIFF
// ============================================================================

/// @brief Type trait to detect AutoDiff types.
template<typename>
struct is_expr : std::false_type {};

template<typename T, Int Norder, Int Nvars>
struct is_expr<AutoDiff<T, Norder, Nvars>> : std::true_type {};

/// @brief Concept satisfied by AutoDiff types.
template<typename U>
concept ExprType = is_expr<U>::value;

/**
 * @brief Extracts the first AutoDiff type from a parameter pack.
 *
 * Used by expression templates to determine the result type when
 * mixing AutoDiff with scalars.
 */
template<typename... U>
struct extract_expr;

template<typename First, typename... Rest>
struct extract_expr<First, Rest...> {
    using type = std::conditional_t<
        ExprType<First>,
        First,
        typename extract_expr<Rest...>::type
    >;
};

template<>
struct extract_expr<> {
    using type = void;  ///< Sentinel when no AutoDiff found.
};

// ============================================================================


/**
 * @brief CRTP base class for expression evaluation with automatic differentiation.
 *
 * Provides the recursive differentiation machinery used by all expression types.
 * Derived classes must implement:
 * - `operation(args...)`: The forward computation.
 * - `diff_rule(values..., derivatives...)`: The differentiation rule.
 *
 * @tparam Derived The derived expression type (CRTP pattern).
 */
template<typename Derived>
struct BaseOperand{

    /**
     * @brief Entry point for evaluating an expression.
     *
     * Determines the result type from arguments and dispatches to opt_result_aux.
     * Handles mixed AutoDiff/scalar arguments automatically.
     *
     * @param f Operands (AutoDiff or scalar types).
     * @return AutoDiff containing result value and all derivatives.
     */
    template<typename... U>
    AD_INLINE static auto opt_result(const U&... f){
        using ExprType = extract_expr<U...>::type;
        static_assert(!std::is_same_v<ExprType, void>, "No AutoDiff found in arguments");
        return opt_result_aux(typename ExprType::DataStruct(), masked_value<typename ExprType::ValueType>(f)...);
    }

    /**
     * @brief Recursive differentiation implementation.
     *
     * Computes the function value via Derived::operation, then recursively
     * computes all partial derivatives via Derived::diff_rule.
     */
    template<typename T, Int Norder, Int Nvars, typename... U>
    AUTODIFF_MAYBE_INLINE static auto opt_result_aux(ExprStruct<T, Norder, Nvars>, const U&... f){
        T v = Derived::operation(_value<T, AUTODIFF>(f)...);
        if constexpr (Norder==0) {
            return AUTODIFF(v);
        }
        else{
            // Compute derivative w.r.t. each variable
            auto q = [&] AUTODIFF_DEVICE <Int I> (auto&&... f) AD_LAMBDA_INLINE {
                return Derived::diff_rule(
                    _reduced_value<T, AUTODIFF>(f)...,
                    _reduced_diff<T, AUTODIFF, I>(f)...
                );
            };
            auto h = AD_EXPAND(Int, Nvars, I,
                return std::array<typename AUTODIFF::ReducedType, Nvars>{q.template operator()<I>(f...)...};
            );

            AUTODIFF res(v);
            Int n=1;

            // Assemble derivatives into result array
            AD_EXPAND(Int, Norder, Ord,
                auto g = [&] AUTODIFF_DEVICE <Int ord>() AD_LAMBDA_INLINE {
                    AD_EXPAND(Int, Nvars, Ivar,
                        auto f = [&] AUTODIFF_DEVICE <Int var>() AD_LAMBDA_INLINE {
                            constexpr Int Noff_tot = MultiDiff<AUTODIFF::ReducedOrder, Nvars>::offset(ord*(var==Ivar)...);
                            constexpr Int Nelements = MultiDiff<AUTODIFF::ReducedOrder, Nvars>::Ndiffs(ord)-MultiDiff<AUTODIFF::ReducedOrder, Nvars>::local_offset(ord*(var==Ivar)...);
                            utils::copy_array(res.data()+n, h[var].data()+Noff_tot, Nelements);
                            n+=Nelements;
                        };
                        (f.template operator()<Ivar>(), ...);
                    );
                };
                (g.template operator()<Ord>(), ...);
            );

            return res;
        }
    }

private:

    /// @brief Identity for AutoDiff arguments.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static decltype(auto) masked_value(const AUTODIFF& item){
        return item;
    }

    /// @brief Converts scalars to the appropriate numeric type.
    template<typename T, typename U>
    AD_INLINE static decltype(auto) masked_value(U&& item){
        return T(item);
    }

    /// @brief Extracts the scalar value from an argument.
    template<typename T, typename ExprType, typename ArgType>
    AD_INLINE static T _value(const ArgType& f) {
        static_assert(std::is_same_v<ArgType, ExprType>||std::is_convertible_v<ArgType, T>, "Invalid argument");
        if constexpr (std::is_same_v<ArgType, ExprType>) {
            return f.value();
        }
        else{
            return masked_value<T>(f);
        }
    }

    /// @brief Gets the reduced-order representation of an argument.
    template<typename T, typename ExprType, typename ArgType>
    AD_INLINE static auto _reduced_value(const ArgType& f){
        static_assert(std::is_same_v<ArgType, ExprType>||std::is_convertible_v<ArgType, T>, "Invalid argument");
        if constexpr (std::is_same_v<ArgType, ExprType>) {
            return f.reduced();
        }
        else{
            return f;
        }
    }

    /// @brief Gets the derivative w.r.t. variable I (zero for scalars).
    template<typename T, typename ExprType, Int I, typename ArgType>
    AD_INLINE static auto _reduced_diff(const ArgType& f){
        static_assert(std::is_same_v<ArgType, ExprType>||std::is_convertible_v<ArgType, T>, "Invalid argument");
        if constexpr (std::is_same_v<ArgType, ExprType>) {
            return f.cmpl_reduced_diff(Variable<I>());
        }
        else{
            return T(0);
        }
    }

};

/**
 * @brief Expression type for addition: d(f+g) = df + dg.
 */
struct AddExpr : BaseOperand<AddExpr>{

    using BaseOperand<AddExpr>::opt_result;

    /// @brief Forward operation: f + g.
    template<typename T>
    AD_INLINE static T operation(const T& f, const T& g){
        return f+g;
    }

    /// @brief Compound assignment: f += g (AutoDiff += AutoDiff).
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const AUTODIFF& other){
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            f.data()[i] += other.data()[i];
        }
        return f;
    }

    /// @brief Compound assignment: f += scalar.
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const U& other){
        f.data()[0] += other;
        return f;
    }

#ifdef AUTODIFF_SCALAR_OPTS
    /// @brief Optimized element-wise addition (AutoDiff + AutoDiff).
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const AUTODIFF& g){
        AUTODIFF res;
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            res.data()[i] = f.data()[i]+g.data()[i];
        }
        return res;
    }

    /// @brief Optimized scalar + AutoDiff.
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const U& f, const AUTODIFF& g){
        AUTODIFF res = g;
        res.data()[0] += f;
        return res;
    }

    /// @brief Optimized AutoDiff + scalar.
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const U& g){
        AUTODIFF res = f;
        res.data()[0] += g;
        return res;
    }
#endif

    /// @brief Differentiation rule: d(f+g)/dx = df/dx + dg/dx.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const AUTODIFF& g, const AUTODIFF& df, const AUTODIFF& dg){
        return df+dg;
    }

    /// @brief Differentiation rule when g is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const T& g, const AUTODIFF& df, const T& dg){
        assert(dg==0);
        return df;
    }

    /// @brief Differentiation rule when f is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const T& f, const AUTODIFF& g, const T& df, const AUTODIFF& dg){
        assert(df==0);
        return dg;
    }
};


/**
 * @brief Expression type for subtraction: d(f-g) = df - dg.
 */
struct SubtrExpr : BaseOperand<SubtrExpr>{

    using BaseOperand<SubtrExpr>::opt_result;

    /// @brief Forward operation: f - g.
    template<typename T>
    AD_INLINE static T operation(const T& f, const T& g){
        return f-g;
    }

    /// @brief Compound assignment: f -= g (AutoDiff -= AutoDiff).
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const AUTODIFF& other){
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            f.data()[i] -= other.data()[i];
        }
        return f;
    }

    /// @brief Compound assignment: f -= scalar.
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const U& other){
        f.data()[0] -= other;
        return f;
    }

#ifdef AUTODIFF_SCALAR_OPTS
    /// @brief Optimized element-wise subtraction (AutoDiff - AutoDiff).
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const AUTODIFF& g){
        AUTODIFF res;
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            res.data()[i] = f.data()[i]-g.data()[i];
        }
        return res;
    }

    /// @brief Optimized scalar - AutoDiff.
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const U& f, const AUTODIFF& g){
        AUTODIFF res(f-g.value());
        for (Int i=1; i<AUTODIFF::Ntot; i++){
            res.data()[i] = -g.data()[i];
        }
        return res;
    }

    /// @brief Optimized AutoDiff - scalar.
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const U& g){
        AUTODIFF res = f;
        res.data()[0] -= g;
        return res;
    }
#endif

    /// @brief Differentiation rule: d(f-g)/dx = df/dx - dg/dx.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const AUTODIFF& g, const AUTODIFF& df, const AUTODIFF& dg){
        return df-dg;
    }

    /// @brief Differentiation rule when g is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const T& g, const AUTODIFF& df, const T& dg){
        assert(dg==0);
        return df;
    }

    /// @brief Differentiation rule when f is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const T& f, const AUTODIFF& g, const T& df, const AUTODIFF& dg){
        assert(df==0);
        return -dg;
    }
};


/**
 * @brief Compile-time Leibniz rule coefficients for product differentiation.
 *
 * Precomputes all binomial coefficients and index offsets needed to apply
 * the generalized Leibniz rule for higher-order derivatives of products:
 *
 *     d^n(fg)/dx^n = Σ C(n,k) * (d^k f/dx^k) * (d^(n-k) g/dx^(n-k))
 *
 * For multivariate derivatives, this generalizes to products of multinomial coefficients.
 *
 * @tparam Norder Maximum derivative order.
 * @tparam Nvars Number of independent variables.
 */
template<Int Norder, Int Nvars>
struct LeibnizDiff{

    static constexpr Int NDIFFS = utils::comb(Nvars+Norder, Norder);

    /**
     * @brief Iterates over all Leibniz rule terms.
     * @param f_main Called for each output derivative.
     * @param f_dummy Called for each term in the Leibniz sum.
     */
    template<typename Callable1, typename Callable2>
    static constexpr void iterate(Callable1&& f_main, Callable2&& f_dummy){
        AD_FOR_LOOP(Int, Iord, Norder+1,
            using IterType = utils::MultiSetIterator<Iord, Nvars, true>;
            AD_EXPAND(Int, Nvars, Ivar,
                auto main_func = [&] AUTODIFF_DEVICE (auto&, auto& ord_of_var) AD_LAMBDA_INLINE {
                    auto dummy_func = [&] AUTODIFF_DEVICE (auto... dummy_order) AD_LAMBDA_INLINE {
                        f_dummy(Iord, ord_of_var, std::array<Int, Nvars>({dummy_order...}));
                    };
                    utils::DynamicNDIterator<Nvars>((ord_of_var[Ivar]+1)...).iterate(dummy_func);
                    f_main(Iord, ord_of_var);
                };
                IterType::apply_iter_on(main_func);
            );
        );
    }

    /// @brief Total number of terms across all Leibniz sums.
    static constexpr Int total_cache_count(){
        Int res = 0;
        auto f_dummy = [&] AUTODIFF_DEVICE (Int order, auto order_wrt, auto dummy_order_wrt) AD_LAMBDA_INLINE {
            res++;
        };
        iterate([] AUTODIFF_DEVICE (auto, auto) AD_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    /// @brief Number of terms in each derivative's Leibniz sum.
    static constexpr std::array<Int, NDIFFS> Nsum_per_offset(){
        std::array<Int, NDIFFS> res{};
        Int i=0;
        auto f_main = [&] AUTODIFF_DEVICE (Int, auto) AD_LAMBDA_INLINE {
            i++;
        };
        auto f_dummy = [&] AUTODIFF_DEVICE (Int, auto, auto) AD_LAMBDA_INLINE {
            res[i]++;
        };
        iterate(f_main, f_dummy);
        return res;
    }

    /// @brief Multinomial coefficients for each Leibniz term.
    static constexpr auto cached_coefs(){
        std::array<Int, total_cache_count()> res{};
        Int i=0;
        auto f_dummy = [&] AUTODIFF_DEVICE (Int, auto order_wrt, auto dummy_order_wrt) AD_LAMBDA_INLINE {
            AD_EXPAND(Int, Nvars, Ivar,
                res[i++] = (utils::comb(order_wrt[Ivar], dummy_order_wrt[Ivar])*...);
            );
        };
        iterate([] AUTODIFF_DEVICE (auto, auto) AD_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    /// @brief Offsets into left operand for each Leibniz term.
    static constexpr auto cached_left_offsets(){
        std::array<Int, total_cache_count()> res{};
        Int i=0;
        auto f_dummy = [&] AUTODIFF_DEVICE (Int, auto, auto dummy_order_wrt) AD_LAMBDA_INLINE {
            AD_EXPAND(Int, Nvars, Ivar,
                res[i++] = MultiDiff<Norder, Nvars>::offset(dummy_order_wrt[Ivar]...);
            );
        };
        iterate([] AUTODIFF_DEVICE (auto, auto) AD_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    /// @brief Offsets into right operand for each Leibniz term.
    static constexpr auto cached_right_offsets(){
        std::array<Int, total_cache_count()> res{};
        Int i=0;
        auto f_dummy = [&] AUTODIFF_DEVICE (Int, auto order_wrt, auto dummy_order_wrt) AD_LAMBDA_INLINE {
            AD_EXPAND(Int, Nvars, Ivar,
                res[i++] = MultiDiff<Norder, Nvars>::offset((order_wrt[Ivar]-dummy_order_wrt[Ivar])...);
            );
        };
        iterate([] AUTODIFF_DEVICE (auto, auto) AD_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    static constexpr auto Nsum_of = Nsum_per_offset();   ///< Terms per derivative.
    static constexpr auto coefs = cached_coefs();         ///< Multinomial coefficients.
    static constexpr auto cached_left = cached_left_offsets();   ///< Left operand indices.
    static constexpr auto cached_right = cached_right_offsets(); ///< Right operand indices.

};


/**
 * @brief Expression type for multiplication: d(fg) = f·dg + df·g (product rule).
 *
 * Higher-order derivatives use the generalized Leibniz rule with precomputed
 * multinomial coefficients for efficiency.
 */
struct MulExpr : BaseOperand<MulExpr>{

    using BaseOperand<MulExpr>::opt_result;

    /// @brief Compound assignment: f *= g (AutoDiff *= AutoDiff).
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const AUTODIFF& other){
        f = f*other;
        return f;
    }

    /// @brief Compound assignment: f *= scalar (scales all derivatives).
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const U& other){
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            f.data()[i] *= other;
        }
        return f;
    }

#ifdef AUTODIFF_SCALAR_OPTS
    /// @brief Optimized AutoDiff * scalar.
    template<typename U, typename T, Int Norder, Int Nvars>
    requires (std::convertible_to<U, T>)
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const U& g){
        AUTODIFF res;
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            res.data()[i] = f.data()[i]*g;
        }
        return res;
    }

    /// @brief Optimized scalar * AutoDiff.
    template<typename U, typename T, Int Norder, Int Nvars>
    requires (std::convertible_to<U, T>)
    AD_INLINE static AUTODIFF opt_result(const U& f, const AUTODIFF& g){
        AUTODIFF res;
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            res.data()[i] = f*g.data()[i];
        }
        return res;
    }
#endif

#ifdef AUTODIFF_ITER_MUL_OPT
    /**
     * @brief Optimized multiplication using precomputed Leibniz coefficients.
     *
     * Uses LeibnizDiff to apply the generalized product rule with cached
     * multinomial coefficients and index offsets.
     */
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const AUTODIFF& g){
        using Cache = LeibnizDiff<Norder, Nvars>;

        AUTODIFF res;
        Int n=0;

        for (Int i=0; i<AUTODIFF::Ntot; i++){
            T sum = 0;
            for (Int k=0; k<Cache::Nsum_of[i]; k++){
                sum += T(Cache::coefs[n+k])*f.data()[Cache::cached_left[n+k]]*g.data()[Cache::cached_right[n+k]];
            }
            n += Cache::Nsum_of[i];
            res.data()[i] = sum;
        }

        return res;
    }

#elif defined (AUTODIFF_ITER_MUL)
    /// @brief Iterative multiplication without precomputed cache.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const AUTODIFF& g){
        AUTODIFF res;
        AD_EXPAND(Int, Norder+1, Iord,
            (apply_diffs<Iord>(res, f, g),...);
        );
        return res;
    }
#endif

    /**
     * @brief Applies Leibniz rule for a specific derivative order.
     * @tparam ORDER The total derivative order to compute.
     */
    template<Int ORDER, typename T, Int Norder, Int Nvars>
    AD_INLINE static void apply_diffs(AUTODIFF& res, const AUTODIFF& f, const AUTODIFF& g){
        static_assert(ORDER<=Norder, "ORDER too large");
        constexpr Int glf = MultiDiff<Norder, Nvars>::global_offset(ORDER);
        T* d = res.data()+glf;
        using IterType = utils::MultiSetIterator<ORDER, Nvars, true>;

        [&] AUTODIFF_DEVICE <Int... Ivar>(AD_INTS(Int, Ivar)) AD_LAMBDA_INLINE {
            Int n_iter = 0;
            auto func = [&] AUTODIFF_DEVICE (auto&, auto& ord_of_var) AD_LAMBDA_INLINE {
                d[n_iter++] = diff_element(f, g, ord_of_var[Ivar]...);
            };
            IterType::apply_iter_on(func);
        }(AD_MAKE_INTS(Int, Nvars));
    }

    /**
     * @brief Computes a single derivative element via Leibniz summation.
     * @param order Per-variable derivative counts.
     * @return The derivative value at the specified multi-index.
     */
    template<typename T, Int Norder, Int Nvars, std::integral... IntType>
    AD_INLINE static T diff_element(const AUTODIFF& f, const AUTODIFF& g, IntType... order){
        T res = 0;
        auto func = [&] AUTODIFF_DEVICE (auto... dummy_order) AD_LAMBDA_INLINE {
            res += T((utils::comb(order, dummy_order)*...))*f.value_of_diff_counts(dummy_order...)*g.value_of_diff_counts((order-dummy_order)...);
        };
        DynamicNDIterator<Nvars>((order+1)...).iterate(func);
        return res;
    }

    /// @brief Forward operation: f * g.
    template<typename T>
    AD_INLINE static T operation(const T& f, const T& g){
        return f*g;
    }

    /// @brief Differentiation rule: d(fg)/dx = f·dg/dx + df/dx·g.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const AUTODIFF& g, const AUTODIFF& df, const AUTODIFF& dg){
        return df*g + f*dg;
    }

    /// @brief Differentiation rule when g is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const T& g, const AUTODIFF& df, const T& dg){
        assert(dg==0);
        return df*g;
    }

    /// @brief Differentiation rule when f is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const T& f, const AUTODIFF& g, const T& df, const AUTODIFF& dg){
        assert(df==0);
        return f*dg;
    }

};

/**
 * @brief Expression type for division: d(f/g) = (g·df - f·dg) / g² (quotient rule).
 */
struct DivExpr : BaseOperand<DivExpr>{

    using BaseOperand<DivExpr>::opt_result;

    /// @brief Forward operation: f / g.
    template<typename T>
    AD_INLINE static T operation(const T& f, const T& g){
        return f/g;
    }

    /// @brief Compound assignment: f /= g (AutoDiff /= AutoDiff).
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const AUTODIFF& other){
        f = f/other;
        return f;
    }

    /// @brief Compound assignment: f /= scalar.
    template<typename U, typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF& compound_assign(AUTODIFF& f, const U& other){
        for (Int i=0; i<AUTODIFF::Ntot; i++){
            f.data()[i] /= other;
        }
        return f;
    }

#ifdef AUTODIFF_SCALAR_OPTS
    /// @brief Optimized AutoDiff / scalar.
    template<typename U, typename T, Int Norder, Int Nvars>
    requires (std::convertible_to<U, T>)
    AD_INLINE static AUTODIFF opt_result(const AUTODIFF& f, const U& g){
        if constexpr (Norder == 0) {
            return AUTODIFF(f.value() / g);
        } else {
            typename AUTODIFF::DataType res;
            for (Int i = 0; i < AUTODIFF::Ntot; i++){
                res._data[i] = f.data()[i] / g;
            }
            return AUTODIFF(res);
        }
    }
#endif

    /// @brief Differentiation rule: d(f/g)/dx = (g·df/dx - f·dg/dx) / g².
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const AUTODIFF& g, const AUTODIFF& df, const AUTODIFF& dg){
        return (df*g - f*dg)/(g*g);
    }

    /// @brief Differentiation rule when g is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const T& g, const AUTODIFF& df, const T& dg){
        assert(dg==0);
        return df/g;
    }

    /// @brief Differentiation rule when f is constant.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const T& f, const AUTODIFF& g, const T& df, const AUTODIFF& dg){
        assert(df==0);
        return -f*dg/(g*g);
    }
};




/**
 * @brief Expression type for exponential: d(exp(f)) = exp(f)·df.
 */
struct ExpExpr : BaseOperand<ExpExpr>{

    /// @brief Forward operation: exp(f).
    template<typename T>
    AD_INLINE static T operation(const T& f){
        return exp(f);
    }

    /// @brief Differentiation rule: d(exp(f))/dx = exp(f)·df/dx.
    template<typename T>
    AD_INLINE static T diff_rule(const T& f, const T& df){
        return exp(f)*df;
    }
};

/**
 * @brief Expression type for natural logarithm: d(log(f)) = df/f.
 */
struct LogExpr : BaseOperand<LogExpr>{

    /// @brief Forward operation: log(f).
    template<typename T>
    AD_INLINE static T operation(const T& f){
        return log(f);
    }

    /// @brief Differentiation rule: d(log(f))/dx = df/dx / f.
    template<typename T>
    AD_INLINE static T diff_rule(const T& f, const T& df){
        return df/f;
    }
};

/**
 * @brief Expression type for exponentiation: d(f^g) = f^g · (g'·log(f) + g·f'/f).
 *
 * Handles three cases:
 * - Both base and exponent are AutoDiff
 * - Only base is AutoDiff (power rule)
 * - Only exponent is AutoDiff (exponential derivative)
 */
struct PowExpr : BaseOperand<PowExpr>{

    /// @brief Forward operation: f^g.
    template<typename T>
    AD_INLINE static T operation(const T& f, const T& g){
        return pow(f, g);
    }

    /// @brief General case: d(f^g)/dx = f^g · (dg/dx·log(f) + g·df/dx/f).
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const AUTODIFF& g, const AUTODIFF& df, const AUTODIFF& dg){
        return pow(f, g)*(dg*log(f) + g*df/f);
    }

    /// @brief Power rule: d(f^n)/dx = n·f^(n-1)·df/dx.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const AUTODIFF& f, const T& g, const AUTODIFF& df, const T& dg){
        assert(dg==0);
        return g*pow(f, g-1)*df;
    }

    /// @brief Exponential derivative: d(a^g)/dx = a^g·log(a)·dg/dx.
    template<typename T, Int Norder, Int Nvars>
    AD_INLINE static AUTODIFF diff_rule(const T& f, const AUTODIFF& g, const T& df, const AUTODIFF& dg){
        assert(df==0);
        return pow(f, g)*log(f)*dg;
    }

};


struct SqrtExpr : BaseOperand<SqrtExpr>{

    /// @brief Forward operation: sqrt(f).
    template<typename T>
    AD_INLINE static T operation(const T& f){
        return sqrt(f);
    }

    /// @brief Differentiation rule: d(sqrt(f))/dx = df/(2·sqrt(f)).
    template<typename T>
    AD_INLINE static T diff_rule(const T& f, const T& df){
        return df/(2*sqrt(f));
    }
};


struct AbsExpr : BaseOperand<AbsExpr>{

    /// @brief Forward operation: abs(f).
    template<typename T>
    AD_INLINE static T operation(const T& f){
        return abs(f);
    }

    /// @brief Differentiation rule: d(abs(f))/dx = df/dx·sign(f) (undefined at f=0).
    template<typename T>
    AD_INLINE static T diff_rule(const T& f, const T& df){
        return df*(f>0 ? 1 : (f<0 ? -1 : 0));
    }
};


struct SinExpr : BaseOperand<SinExpr>{

    /// @brief Forward operation: sin(f).
    template<typename T>
    AD_INLINE static T operation(const T& f){
        return sin(f);
    }

    /// @brief Differentiation rule: d(sin(f))/dx = cos(f)·df/dx.
    template<typename T>
    AD_INLINE static T diff_rule(const T& f, const T& df){
        return cos(f)*df;
    }
};

struct CosExpr : BaseOperand<CosExpr>{

    /// @brief Forward operation: cos(f).
    template<typename T>
    AD_INLINE static T operation(const T& f){
        return cos(f);
    }

    /// @brief Differentiation rule: d(cos(f))/dx = -sin(f)·df/dx.
    template<typename T>
    AD_INLINE static T diff_rule(const T& f, const T& df){
        return -sin(f)*df;
    }
};


struct TanExpr : BaseOperand<TanExpr>{

    /// @brief Forward operation: tan(f).
    template<typename T>
    AD_INLINE static T operation(const T& f){
        return tan(f);
    }

    /// @brief Differentiation rule: d(tan(f))/dx = sec²(f)·df/dx.
    template<typename T>
    AD_INLINE static T diff_rule(const T& f, const T& df){
        return (1+tan(f)*tan(f))*df;
    }
};

// ============================================================================
//                         IMPLEMENTATION DETAILS
// ============================================================================

// ----------------------------------------------------------------------------
//                             AutoDiff Constructors
// ----------------------------------------------------------------------------

template<typename T, Int Norder, Int Nvars>
template<Int I>
AUTODIFF::AutoDiff(T value, Variable<I> x) {
    static_assert(I<Nvars, "Variable index must be < Nvars");
    _data[0] = value;
    if constexpr (Norder > 0) {
        constexpr Int offset = I+1;
        _data[offset] = 1;  // d(this)/d(variable I) = 1
    }
}

template<typename T, Int Norder, Int Nvars>
AUTODIFF::AutoDiff(const DataType& data) : _data(data){}

template<typename T, Int Norder, Int Nvars>
template<std::integral IntType>
AUTODIFF::AutoDiff(IntType value) {_data[0] = T(value);}

template<typename T, Int Norder, Int Nvars>
AUTODIFF::AutoDiff(T value) {_data[0] = value;}

template<typename T, Int Norder, Int Nvars>
AD_INLINE const T& AUTODIFF::value() const{
    return _data[0];
}

// ----------------------------------------------------------------------------
//                             MultiDiff Methods
// ----------------------------------------------------------------------------

template<Int Norder, Int Nvars>
AD_INLINE constexpr Int MultiDiff<Norder, Nvars>::Ndiffs(Int order){
    return utils::comb(Nvars+order-1, order);
}

template<Int Norder, Int Nvars>
template<std::integral... IntType>
AD_INLINE constexpr Int MultiDiff<Norder, Nvars>::local_offset(IntType... order){
    if constexpr (!(std::is_same_v<IntType, Int>&&...)){
        assert(((order>=0)&&...));
        return local_offset(order...);
    }
    constexpr Int Nx = sizeof...(order);
    static_assert(Nx>0 && Nx<=Nvars );
    Int total_order = (static_cast<Int>(order) + ...);
    assert(total_order <= Norder && "diff order must be <= Norder");

    // Compute colexicographic offset within the group
    Int colex_offset = AD_EXPAND(Int, Nx, variable,
        return ([&] AUTODIFF_DEVICE () AD_LAMBDA_INLINE {
            constexpr Int v = variable;
            Int truncated_total = AD_EXPAND(Int, Nx, I,
                return ((static_cast<Int>(utils::pack_elem<I>(order...))*(I<v))+...);
            );
            Int res = 0;
            for (Int j = 0; j < static_cast<Int>(utils::pack_elem<v>(order...)); j++){
                Int remaining = total_order - truncated_total - j;
                if (remaining > 0 && Nvars - v - 1 > 0) {
                    res += utils::comb(Nvars - v - 1 + remaining - 1, remaining);
                } else if (remaining == 0) {
                    res += 1;
                }
            }
            return res;
        }()+...);
    );

    Int total_for_order = utils::multiset_coef(Nvars, total_order);
    return total_for_order - colex_offset - 1;
}

template<Int Norder, Int Nvars>
AD_INLINE constexpr Int MultiDiff<Norder, Nvars>::global_offset(Int order) {
    if (std::is_constant_evaluated()) {
        if (order > Norder) {
            throw "order > Norder in constexpr evaluation";
        }
    } else {
        assert(order <= Norder && "Requested order is > Norder");
    }
    return order == 0 ? 0 : utils::comb(Nvars + order - 1, order - 1);
}

template<Int Norder, Int Nvars>
template<std::integral... IntType>
AD_INLINE constexpr Int MultiDiff<Norder, Nvars>::offset(IntType... order){
    return global_offset((static_cast<Int>(order)+...)) + local_offset(order...);
}

template<Int Norder, Int Nvars>
AD_INLINE constexpr Int MultiDiff<Norder, Nvars>::offset(const std::array<Int, Nvars>& diff_count){
    return AD_EXPAND(Int, Nvars, I,
        return global_offset((diff_count[I]+...)) + local_offset(diff_count[I]...);
    );
}

// ----------------------------------------------------------------------------
//                         AutoDiff Derivative Extraction
// ----------------------------------------------------------------------------

template<typename T, Int Norder, Int Nvars>
template<VarLike... IntType>
requires (sizeof...(IntType)<=Norder)
AUTODIFF_MAYBE_INLINE constexpr AUTODIFF::ReducedArray<sizeof...(IntType)> AUTODIFF::offsets_for_reduced_diff(IntType... x) {
    constexpr Int NewOrder = Norder-static_cast<Int>(sizeof...(x));

    std::array<Int, AutoDiff<T, NewOrder, Nvars>::Ntot> res{};
    Int n = 0;

    auto Nx = Base::diff_count(x...);

    auto call_it = [&] AUTODIFF_DEVICE <Int... I>(std::integer_sequence<Int, I...>) AD_LAMBDA_INLINE {
        [&] AUTODIFF_DEVICE <Int... Ord>(AD_INTS(Int, Ord)) AD_LAMBDA_INLINE {
            ([&] AUTODIFF_DEVICE <Int OrdI>() AD_LAMBDA_INLINE {
                using IterType = utils::MultiSetIterator<OrdI+sizeof...(x), Nvars, true>;

                auto f = [&] AUTODIFF_DEVICE (const IterType::SetType&, const IterType::CounterType& order_of_var) AD_LAMBDA_INLINE {
                    if ((((order_of_var[I] >= Nx[I])) &&...)){
                        res[n++] = Base::offset(order_of_var[I]...);
                    }
                };

                IterType::apply_iter_on(f);
            }.template operator()<Ord>(), ...);
        }(AD_MAKE_INTS(Int, NewOrder+1UL));
    };

    call_it(AD_MAKE_INTS(Int, Nvars));
    return res;
}

template<typename T, Int Norder, Int Nvars>
AD_INLINE typename AUTODIFF::ReducedType AUTODIFF::reduced() const{
    if constexpr (Norder>0){
        typename ReducedType::DataType new_data;
        utils::copy_array(new_data.data(), _data.data(), ReducedType::Ntot);
        return ReducedType(new_data);
    }
    else{
        return ReducedType(this->value());
    }
}

template<typename T, Int Norder, Int Nvars>
template<VarLike... IntType>
requires (sizeof...(IntType)<=Norder)
AUTODIFF_MAYBE_INLINE AUTODIFF::Reduced<sizeof...(IntType)> constexpr AUTODIFF::reduced_diff(IntType... I)const{
    using ResType = typename AUTODIFF::Reduced<sizeof...(I)>;
    auto offsets = offsets_for_reduced_diff(I...);
    typename ResType::DataType data{};

    for (Int i=0; i<ResType::Ntot; i++){
        data[i] = _data[offsets[i]];
    }
    return ResType(data);
}

template<typename T, Int Norder, Int Nvars>
template<Int... I>
requires (sizeof...(I)<=Norder)
AUTODIFF_MAYBE_INLINE AUTODIFF::Reduced<sizeof...(I)> constexpr AUTODIFF::cmpl_reduced_diff(Variable<I>...)const{
    using ResType = typename AUTODIFF::Reduced<sizeof...(I)>;
    auto constexpr offsets = offsets_for_reduced_diff(I...);
    typename ResType::DataType data{};

    for (Int i=0; i<ResType::Ntot; i++){
        data[i] = _data[offsets[i]];
    }
    return ResType(data);
}

template<typename T, Int Norder, Int Nvars>
AD_INLINE const T* AUTODIFF::data() const{
    return _data.data();
}

template<typename T, Int Norder, Int Nvars>
AD_INLINE T* AUTODIFF::data(){
    return _data.data();
}

template<typename T, Int Norder, Int Nvars>
template<VarLike... IntType>
AD_INLINE AUTODIFF AUTODIFF::diff(IntType... x) const{
    auto f = this->reduced_diff(x...);
    AUTODIFF res;
    utils::copy_array(res.data(), f.data(), f.Ntot);
    return res;
}

template<typename T, Int Norder, Int Nvars>
template<VarLike... IntType>
AD_INLINE T AUTODIFF::diff_value(IntType... x) const{
    auto Nx = Base::diff_count(x...);
    return _data[Base::offset(Nx)];
}


// ============================================================================
//                         OPERATOR DEFINITIONS
// ============================================================================

// Addition: f + g
OPERATOR(operator+, AddExpr)
COMPOUND_OPERATOR(operator+=, AddExpr)

// Subtraction: f - g
OPERATOR(operator-, SubtrExpr)
COMPOUND_OPERATOR(operator-=, SubtrExpr)

// Multiplication: f * g
OPERATOR(operator*, MulExpr)
COMPOUND_OPERATOR(operator*=, MulExpr)

// Division: f / g
OPERATOR(operator/, DivExpr)
COMPOUND_OPERATOR(operator/=, DivExpr)


// ============================================================================
//                         COMPARISON OPERATORS
// ============================================================================


/// @brief Equality comparison: compares values only (derivatives ignored).
template<typename T, Int Norder, Int Nvars>
AD_INLINE bool operator==(const AUTODIFF& a, const AUTODIFF& b){
    return a.value() == b.value();
}

/// @brief Inequality comparison: compares values only (derivatives ignored).
template<typename T, Int Norder, Int Nvars>
AD_INLINE bool operator!=(const AUTODIFF& a, const AUTODIFF& b){
    return a.value() != b.value();
}

/// @brief Less-than comparison: compares values only (derivatives ignored).
template<typename T, Int Norder, Int Nvars>
AD_INLINE bool operator<(const AUTODIFF& a, const AUTODIFF& b){
    return a.value() < b.value();
}

/// @brief Greater-than comparison: compares values only (derivatives ignored).
template<typename T, Int Norder, Int Nvars>
AD_INLINE bool operator>(const AUTODIFF& a, const AUTODIFF& b){
    return a.value() > b.value();
}

/// @brief Less-than-or-equal comparison: compares values only (derivatives ignored).
template<typename T, Int Norder, Int Nvars>
AD_INLINE bool operator<=(const AUTODIFF& a, const AUTODIFF& b){
    return a.value() <= b.value();
}

/// @brief Greater-than-or-equal comparison: compares values only (derivatives ignored).
template<typename T, Int Norder, Int Nvars>
AD_INLINE bool operator>=(const AUTODIFF& a, const AUTODIFF& b){
    return a.value() >= b.value();
}


// ----------------- templated comparison overloads ----------------------
template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator==(const AUTODIFF& a, const U& b){
    return a.value() == b;
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator!=(const AUTODIFF& a, const U& b){
    return a.value() != b;
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator<(const AUTODIFF& a, const U& b){
    return a.value() < b;
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator>(const AUTODIFF& a, const U& b){
    return a.value() > b;
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator<=(const AUTODIFF& a, const U& b){
    return a.value() <= b;
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator>=(const AUTODIFF& a, const U& b){
    return a.value() >= b;
}

// Allow comparisons with scalar on the left-hand side
template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator==(const U& a, const AUTODIFF& b){
    return a == b.value();
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator!=(const U& a, const AUTODIFF& b){
    return a != b.value();
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator<(const U& a, const AUTODIFF& b){
    return a < b.value();
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator>(const U& a, const AUTODIFF& b){
    return a > b.value();
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator<=(const U& a, const AUTODIFF& b){
    return a <= b.value();
}

template<typename T, Int Norder, Int Nvars, typename U>
AD_INLINE bool operator>=(const U& a, const AUTODIFF& b){
    return a >= b.value();
}

// ============================================================================
//                         MATHEMATICAL FUNCTIONS
// ============================================================================

/// @brief Power function: pow(f, g) = f^g with automatic differentiation.
OPERATOR(pow, PowExpr)

/// @brief Natural logarithm: log(f) with automatic differentiation.
MATHFUNC(log, LogExpr)

/// @brief Exponential function: exp(f) with automatic differentiation.
MATHFUNC(exp, ExpExpr)

/// @brief Square root: sqrt(f) with automatic differentiation.
MATHFUNC(sqrt, SqrtExpr)

/// @brief Sine function: sin(f) with automatic differentiation.
MATHFUNC(sin, SinExpr)

/// @brief Cosine function: cos(f) with automatic differentiation.
MATHFUNC(cos, CosExpr)

/// @brief Tangent function: tan(f) with automatic differentiation.
MATHFUNC(tan, TanExpr)

/// @brief Absolute value function: abs(f) with automatic differentiation.
MATHFUNC(abs, AbsExpr)


// ============================================================================
//                              STREAM OPERATOR
// ============================================================================

/// @brief Stream output for AutoDiff variables (prints value only).
template<typename T, Int Norder, Int Nvars>
inline std::ostream& operator<<(std::ostream& os, const AUTODIFF& x){
    os << x.value();
    return os;
}


} // namespace autodiff

namespace std {
template<typename T, autodiff::Int Norder, autodiff::Int Nvars>
class numeric_limits<autodiff::AUTODIFF> : public numeric_limits<T>{
public:

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF min() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::min());
    }

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF max() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::max());
    }

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF lowest() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::lowest());
    }

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF epsilon() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::epsilon());
    }

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF infinity() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::infinity());
    }

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF quiet_NaN() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::quiet_NaN());
    }

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF signaling_NaN() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::signaling_NaN());
    }

    AUTODIFF_HOST_DEVICE static constexpr autodiff::AUTODIFF denorm_min() noexcept {
        return autodiff::AUTODIFF(numeric_limits<T>::denorm_min());
    }
};

template<typename T, autodiff::Int Norder, autodiff::Int Nvars>
AD_INLINE bool isfinite(const autodiff::AUTODIFF& x){
    return isfinite(x.value());
}

template<typename T, autodiff::Int Norder, autodiff::Int Nvars>
AD_INLINE autodiff::AUTODIFF abs(const autodiff::AUTODIFF& x){
    return autodiff::abs(x);
}
}


#undef AD_THIS
#undef AD_UNIQUE_NAME
#undef AD_CONCAT
#undef AD_CONCAT_IMPL
#undef AD_INTS
#undef AD_MAKE_INTS
#undef AD_EXPAND
#undef AD_FOR_LOOP_IMPL
#undef AD_FOR_LOOP
#undef AD_INLINE
#undef AD_LAMBDA_INLINE



#endif // AUTODIFF_HPP