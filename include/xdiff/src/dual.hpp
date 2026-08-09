/**
 * @file dual.hpp
 * @brief Implementation of the Dual class for automatic differentiation.
 *
 * The Dual class is the primary type for computing derivatives. It stores
 * the function value and all partial derivatives up to order Nord for Nvars
 * independent variables. Derivatives are stored in graded colexicographic order.
 */
#ifndef DUAL_HPP
#define DUAL_HPP


#include "rules.hpp"
#include <ostream>


namespace xdiff{

/**
 * @brief Dual number storing all partial derivatives up to a specified order.
 *
 * The Dual class implements forward-mode automatic differentiation by storing
 * the value of a function along with all its partial derivatives. Arithmetic
 * operations propagate derivatives according to calculus rules.
 *
 * @tparam T The scalar type (e.g., double, float)
 * @tparam Nvars Number of independent variables
 * @tparam Nord Maximum derivative order to compute
 *
 * @note Storage size is C(Nvars + Nord, Nord), which grows combinatorially.
 *       For Nvars=3, Nord=4, this is 35 values.
 *
 * @example
 *     Symbol<0> x;
 *     Symbol<1> y;
 *
 *     Dual<double, 2, 2> a(x, 3.0);  // a = 3.0, da/dx = 1
 *     Dual<double, 2, 2> b(y, 2.0);  // b = 2.0, db/dy = 1
 *
 *     auto f = a * b;  // f = 6.0
 *     f.value();                      // 6.0
 *     f.get_diff_wrt(x);              // 2.0 (df/dx = b)
 *     f.get_diff_wrt(y);              // 3.0 (df/dy = a)
 *     f.get_diff_wrt(x, y);           // 1.0 (d²f/dxdy)
 */
template<typename T, size_t Nvars, size_t Nord>
class Dual {

public:

    static constexpr size_t NVARS = Nvars;   ///< Number of independent variables
    static constexpr size_t NORDER = Nord;   ///< Maximum derivative order
    static constexpr size_t NTOT = utils::comb(NVARS+NORDER, NORDER); ///< Total stored values (value + all derivatives)

    /// Order after one differentiation
    static constexpr size_t REDUCED_ORDER = NORDER > 0 ? NORDER-1 : 0;

    template<typename U, size_t Nv, size_t No>
    friend class Dual;   // every Dual is a friend

    /// Type after DiffCount differentiations
    template<size_t DiffCount>
    using Reduced = Dual<T, NVARS, (NORDER > DiffCount ? NORDER-DiffCount : 0)>;

    /// Type after one differentiation
    using ReducedType = Dual<T, NVARS, REDUCED_ORDER>;

    /// Internal storage type
    using DataType = std::array<T, NTOT>;

    /// Array type for storing offsets after DiffCount differentiations
    template<size_t DiffCount>
    using ReducedArray = std::array<size_t, Dual<T, NVARS, NORDER-DiffCount>::NTOT>;

    /// The scalar value type
    using value_type = T;

    // =========================================================================
    // Constructors
    // =========================================================================

    /// @brief Default constructor. Initializes value and all derivatives to zero.
    XDIFF_INLINE_HOST_DEVICE
    Dual() = default;

    /// @brief Copy constructor.
    XDIFF_INLINE_HOST_DEVICE
    Dual(const Dual& other) = default;

    /// @brief Move constructor.
    XDIFF_INLINE_HOST_DEVICE
    Dual(Dual&& other) = default;

    /**
     * @brief Constructs a Dual representing an independent variable.
     *
     * Creates a Dual with the given value and marks it as the variable
     * at axis I. The first derivative with respect to variable I is set to 1.
     *
     * @tparam I The variable index (compile-time)
     * @param x The Symbol representing the variable axis
     * @param value The initial value
     *
     * @example
     *     Symbol<0> x;
     *     Dual<double, 2, 2> a(x, 3.0);  // a = 3.0, da/dx = 1, da/dy = 0
     */
    template<size_t I>
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(Symbol<I> /*x*/, T value){
        static_assert(I<NVARS, "Variable index must be < NVARS");
        data_[0] = value;
        if constexpr (NORDER > 0) {
            data_[I+1] = 1;  // d(this)/d(variable I) = 1. offset = I+1
        }
    }

    /**
     * @brief Constructs a Dual from an expression.
     *
     * Evaluates the expression and assigns the result to this Dual.
     *
     * @tparam D The derived expression type
     * @param expr The expression to evaluate
     */
    template<typename D>
    XDIFF_INLINE_HOST_DEVICE
    Dual(const detail::Expr<D, T>& expr){
        expr.assign_to(*this);
    }

    /// @brief Constructs a constant Dual from an integral value (all derivatives = 0).
    template<std::integral IntType>
    XDIFF_INLINE_HOST_DEVICE
    Dual(IntType value) {
        data_[0] = T(value);
    }

    /// @brief Constructs a constant Dual from a scalar value (all derivatives = 0).
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(T value) { data_[0] = value; }

    /**
     * @brief Constructs a Dual representing an independent variable (runtime axis).
     *
     * @param value The initial value
     * @param axis The variable index (runtime)
     */
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(T value, size_t axis) {
        data_[0] = value;
        if constexpr (NORDER > 0) {
            data_[axis + 1] = 1;  // d(this)/d(variable axis) = 1. offset = axis+1
        }
    }

    /// @brief Constructs a Dual from raw data array.
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(const DataType& data) : data_(data){}

    // =========================================================================
    // Assignment operators
    // =========================================================================

    /// @brief Copy assignment.
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Dual& other) = default;

    /// @brief Move assignment.
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(Dual&& other) = default;

    /// @brief Assigns from an expression.
    template<detail::traits::isExpr<T> F>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const F& f) {
        f.assign_to(*this);
        return *this;
    }

    /// @brief Assigns a scalar value (all derivatives become zero).
    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const U& value) {
        data_[0] = T(value);
        for (size_t i = 1; i < NTOT; i++) {
            data_[i] = T(0);
        }
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    ~Dual() = default;

    operator T() const = delete;  // Disable implicit conversion to T to avoid accidental loss of derivative information.

    // =========================================================================
    // Unary operators
    // =========================================================================

    /// @brief Unary plus (returns a copy).
    XDIFF_INLINE_HOST_DEVICE
    Dual operator+() const {
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    /// @brief Unary minus (negates value and all derivatives).
    Dual operator-() const {
        Dual result;
        for (size_t i=0; i<NTOT; i++){
            result.data_[i] = -data_[i];
        }
        return result;
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// @brief Returns the function value (zeroth derivative).
    XDIFF_INLINE_HOST_DEVICE
    const T& value() const {
        return data_[0];
    }

    /**
     * @brief Returns a reduced-order Dual truncated to NORDER-1.
     *
     * Creates a new Dual with the same value but one less maximum order.
     * Useful for extracting derivatives as new differentiable objects.
     *
     * @return A Dual with order NORDER-1
     */
    XDIFF_INLINE_HOST_DEVICE
    ReducedType reduced() const {
        if constexpr (NORDER>0){
            ReducedType out;
            utils::copy_array(out.data_.data(), data_.data(), ReducedType::NTOT);
            return out;
        } else{
            return ReducedType(this->value());
        }
    }

    /**
     * @brief Extracts a partial derivative as a new reduced-order Dual (compile-time).
     *
     * Returns a Dual whose value is the specified partial derivative, and whose
     * derivatives are the corresponding higher-order derivatives.
     *
     * @tparam I Variable indices to differentiate with respect to
     * @param x Symbols representing the differentiation variables
     * @return A Dual representing the derivative and its higher derivatives
     *
     * @example
     *     Dual<double, 3, 3> f = ...;
     *     auto df_dx = f.reduced_diff_wrt(Symbol<0>{});      // df/dx and its derivatives
     *     auto d2f_dxdy = f.reduced_diff_wrt(Symbol<0>{}, Symbol<1>{});  // d²f/dxdy
     */
    template<size_t... I>
    XDIFF_HOST_DEVICE
    auto constexpr reduced_diff_wrt(Symbol<I>... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        using ResType = typename Dual::Reduced<sizeof...(I)>;
        auto constexpr OFFSETS = offsets_for_reduced_diff(x...);
        typename ResType::DataType data{};

        for (size_t i=0; i<ResType::NTOT; i++){
            data[i] = data_[OFFSETS[i]];
        }
        return ResType(data);
    }

    /**
     * @brief Extracts a partial derivative as a new reduced-order Dual (runtime or mixed).
     *
     * @tparam IntType Axis index types (Symbol or integral)
     * @param x Variables to differentiate with respect to
     * @return A Dual representing the derivative and its higher derivatives
     */
    template<detail::traits::isAxis... IntType>
    XDIFF_HOST_DEVICE
    auto constexpr reduced_diff_wrt(IntType... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        using ResType = typename Dual::Reduced<sizeof...(x)>;
        auto offsets = offsets_for_reduced_diff(x...);
        typename ResType::DataType data{};
        for (size_t i=0; i<ResType::NTOT; i++){
            data[i] = data_[offsets[i]];
        }
        return ResType(data);
    }

    /**
     * @brief Extracts a partial derivative as a full-order Dual.
     *
     * Similar to reduced_diff_wrt, but returns a Dual of the same order
     * (padded with zeros for unavailable higher derivatives).
     *
     * @tparam IntType Axis index types
     * @param x Variables to differentiate with respect to
     * @return A Dual of the same order containing the derivative
     */
    template<detail::traits::isAxis... IntType>
    XDIFF_INLINE_HOST_DEVICE
    Dual diff_wrt(IntType... x) const{
        auto f = this->reduced_diff_wrt(x...);
        Dual res;
        utils::copy_array(res.data_.data(), f.data_.data(), f.NTOT);
        return res;
    }

    /**
     * @brief Returns the scalar value of a specific partial derivative.
     *
     * @tparam IntType Axis index types
     * @param x Variables to differentiate with respect to
     * @return The derivative value as a scalar
     *
     * @example
     *     Dual<double, 2, 2> f = ...;
     *     f.get_diff_wrt(0);        // df/dx
     *     f.get_diff_wrt(0, 1);     // d²f/dxdy
     *     f.get_diff_wrt(Symbol<0>{}, Symbol<0>{});  // d²f/dx²
     */
    template<detail::traits::isAxis... IntType>
    XDIFF_INLINE_HOST_DEVICE
    const T& get_diff_wrt(IntType... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        auto nx = diff_count(x...);
        return data_[offset(nx)];
    }

    /**
     * @brief Assigns this Dual's data to another Dual (possibly different order).
     *
     * @tparam Norder The target Dual's order
     * @param dual The target Dual to assign to
     */
    template<size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    void assign_to(Dual<T, NVARS, Norder>& dual) const {
        // The number of variables must be the same, but the order can differ.
        // We copy all derivatives that are valid in the target dual.
        constexpr size_t MIN_ORDER = std::min(Norder, NORDER);
        for (size_t i = 0; i < MIN_ORDER; i++) {
            dual.data_[i] = data_[i];
        }
    }

    /// @brief Returns const reference to internal data array.
    const DataType& data() const {
        return data_;
    }

    /// @brief Returns mutable reference to internal data array.
    DataType& data() {
        return data_;
    }

    // =========================================================================
    // Static methods for derivative indexing
    // =========================================================================

    /**
     * @brief Returns the number of unique derivatives of a given order.
     *
     * For Nvars variables and a specific order, returns C(Nvars+order-1, order).
     *
     * @param order The derivative order
     * @return Number of unique partial derivatives of that order
     */
    XDIFF_INLINE_HOST_DEVICE
    static constexpr size_t ndiffs(size_t order){
        return utils::comb(NVARS+order-1, order);
    }

    /**
     * @brief Computes the local (within-order) offset for a derivative.
     *
     * Given the per-variable derivative counts, computes the colexicographic
     * offset within the group of derivatives of the same total order.
     *
     * @tparam IntType Unsigned integral types
     * @param order Per-variable derivative counts
     * @return Local offset within the order group
     */
    template<std::unsigned_integral... IntType>
    XDIFF_INLINE_HOST_DEVICE
    static constexpr size_t local_offset(IntType... order){
        constexpr size_t NX = sizeof...(order);
        static_assert(NX>0 && NX<=Nvars );
        size_t total_order = (static_cast<size_t>(order) + ...);
        assert(total_order <= Nord && "diff order must be <= Norder");

        // Compute colexicographic offset within the group
        size_t colex_offset = XDIFF_EXPAND(size_t, NX, variable,
            return ([&] XDIFF_DEVICE () XDIFF_ALWAYS_INLINE {
                constexpr size_t v = variable;
                size_t truncated_total = XDIFF_EXPAND(size_t, NX, I,
                    return ((static_cast<size_t>(utils::pack_elem<I>(order...))*(I<v))+...);
                );
                size_t res = 0;
                for (size_t j = 0; j < static_cast<size_t>(utils::pack_elem<v>(order...)); j++){
                    size_t remaining = total_order - truncated_total - j;
                    if (remaining > 0 && Nvars - v - 1 > 0) {
                        res += utils::comb(Nvars - v - 1 + remaining - 1, remaining);
                    } else if (remaining == 0) {
                        res += 1;
                    }
                }
                return res;
            }()+...);
        );

        size_t total_for_order = utils::multiset_coef(Nvars, total_order);
        return total_for_order - colex_offset - 1;
    }

    /**
     * @brief Computes the global offset (start index) for derivatives of a given order.
     *
     * Derivatives are stored in graded order: all 0th-order (value), then all
     * 1st-order, then all 2nd-order, etc. This returns the starting index.
     *
     * @param order The derivative order
     * @return Starting index in the data array for derivatives of this order
     */
    XDIFF_INLINE_HOST_DEVICE
    static constexpr size_t global_offset(size_t order){
        if (std::is_constant_evaluated()) {
            if (order > Nord) {
                throw "order > Norder in constexpr evaluation";
            }
        } else {
            assert(order <= Nord && "Requested order is > Norder");
        }
        return order == 0 ? 0 : utils::comb(Nvars + order - 1, order - 1);
    }

    /**
     * @brief Computes the array offset for a specific derivative.
     *
     * @tparam IntType Unsigned integral types
     * @param order Per-variable derivative counts
     * @return Index into the data array
     */
    template<std::unsigned_integral... IntType>
    XDIFF_INLINE_HOST_DEVICE static constexpr size_t offset(IntType... order){
        return global_offset((static_cast<size_t>(order)+...)) + local_offset(order...);
    }

    /// @brief Computes array offset from a diff_count array.
    XDIFF_INLINE_HOST_DEVICE
    static constexpr size_t offset(const std::array<size_t, Nvars>& diff_count){
        return XDIFF_EXPAND(size_t, Nvars, I,
            return global_offset((diff_count[I]+...)) + local_offset(diff_count[I]...);
        );
    }

    /**
     * @brief Converts variable indices to per-variable derivative counts.
     *
     * Given a list of variable indices, counts how many times each variable
     * appears (i.e., the order of differentiation with respect to each).
     *
     * @tparam Var Axis types (Symbol or integral)
     * @param x Variable indices
     * @return Array with count of each variable
     *
     * @example
     *     diff_count(0, 1, 0);  // Returns {2, 1, 0, ...} for d²f/dx²dy
     */
    template<detail::traits::isAxis... Var>
    XDIFF_INLINE_HOST_DEVICE
    static constexpr std::array<size_t, Nvars> diff_count(Var... x){
        return XDIFF_EXPAND(size_t, Nvars, I,
            return std::array<size_t, Nvars>{utils::var_count(I, x...)...};
        );
    }

    /**
     * @brief Retrieves derivative value by per-variable counts.
     *
     * @tparam IntType Integral types
     * @param order_wrt Derivative count for each variable
     * @return The derivative value
     *
     * @example
     *     f.value_of_diff_counts(2, 1, 0);  // d³f/dx²dy
     */
    template<std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE T value_of_diff_counts(IntType... order_wrt)const{
        return data_[offset(order_wrt...)];
    }

    /**
     * @brief Precomputes offsets for reduced_diff_wrt extraction.
     *
     * Used internally to efficiently extract derivatives into a reduced Dual.
     *
     * @tparam IntType Axis types
     * @param x Variables to differentiate with respect to
     * @return Array of offsets for extracting the reduced derivative
     */
    template<detail::traits::isAxis... IntType>
    requires (sizeof...(IntType)<=NORDER)
    XDIFF_HOST_DEVICE
    static constexpr ReducedArray<sizeof...(IntType)> offsets_for_reduced_diff(IntType... x){
        constexpr size_t NEW_ORDER = NORDER-static_cast<size_t>(sizeof...(x));

        std::array<size_t, Dual<T, NVARS, NEW_ORDER>::NTOT> res{};
        size_t n = 0;

        auto nx = diff_count(x...);

        auto call_it = [&] XDIFF_DEVICE <size_t... I>(std::integer_sequence<size_t, I...>) XDIFF_ALWAYS_INLINE {
            [&] XDIFF_DEVICE <size_t... Ord>(XDIFF_INTS(size_t, Ord)) XDIFF_ALWAYS_INLINE {
                ([&] XDIFF_DEVICE <size_t OrdI>() XDIFF_ALWAYS_INLINE {
                    using IterType = utils::MultiSetIterator<OrdI+sizeof...(x), Nvars, true>;

                    auto f = [&] XDIFF_DEVICE (const IterType::SetType&, const IterType::CounterType& order_of_var) XDIFF_ALWAYS_INLINE {
                        if ((((order_of_var[I] >= nx[I])) &&...)){
                            res[n++] = offset(order_of_var[I]...);
                        }
                    };

                    IterType::apply_iter_on(f);
                }.template operator()<Ord>(), ...);
            }(XDIFF_MAKE_INTS(size_t, NEW_ORDER+1UL));
        };

        call_it(XDIFF_MAKE_INTS(size_t, Nvars));
        return res;
    }

private:
    DataType data_ = {};

};




// =============================================================================
// Arithmetic operators for Dual
// =============================================================================

/// @brief Power function: pow(base, exponent)
XDIFF_OPERATOR(pow, Pow)

/// @brief Natural logarithm: log(x)
XDIFF_MATHFUNC_DUAL(log, Log)

/// @brief Addition: f + g
XDIFF_OPERATOR(operator+, Add)
/// @brief Compound addition: f += g
XDIFF_COMPOUND_OPERATOR(operator+=, Add)

/// @brief Subtraction: f - g
XDIFF_OPERATOR(operator-, Sub)
/// @brief Compound subtraction: f -= g
XDIFF_COMPOUND_OPERATOR(operator-=, Sub)

/// @brief Multiplication: f * g
XDIFF_OPERATOR(operator*, Mul)
/// @brief Compound multiplication: f *= g
XDIFF_COMPOUND_OPERATOR(operator*=, Mul)

/// @brief Division: f / g
XDIFF_OPERATOR(operator/, Div)
/// @brief Compound division: f /= g
XDIFF_COMPOUND_OPERATOR(operator/=, Div)


// =============================================================================
// Comparison operators for Dual
// =============================================================================

/**
 * All comparison operators compare values only; derivatives are ignored.
 * This is consistent with the interpretation that a Dual represents a
 * function at a specific point.
 */
XDIFF_DUAL_COMPARISON_OPERATOR(==)
XDIFF_DUAL_COMPARISON_OPERATOR(!=)
XDIFF_DUAL_COMPARISON_OPERATOR(<)
XDIFF_DUAL_COMPARISON_OPERATOR(<=)
XDIFF_DUAL_COMPARISON_OPERATOR(>)
XDIFF_DUAL_COMPARISON_OPERATOR(>=)


/// @brief Stream output operator (outputs the value only).
template<typename T, size_t Nvars, size_t Norder>
std::ostream& operator<<(std::ostream& os, const Dual<T, Nvars, Norder>& dual){
    return os << dual.value();
}


} // namespace xdiff






// =============================================================================
// Standard library specializations
// =============================================================================

namespace std {

/**
 * @brief std::numeric_limits specialization for Dual.
 *
 * Inherits from numeric_limits<T> and wraps the limit values in Dual objects.
 * This enables Dual to work with algorithms that query numeric limits.
 */
template<typename T, size_t Nvars, size_t Norder>
class numeric_limits<xdiff::Dual<T, Nvars, Norder>> : public numeric_limits<T>{
public:

    static constexpr xdiff::Dual<T, Nvars, Norder> min() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::min());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> max() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::max());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> lowest() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::lowest());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> epsilon() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::epsilon());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> round_error() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::round_error());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> infinity() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::infinity());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> quiet_NaN() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::quiet_NaN());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> signaling_NaN() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::signaling_NaN());
    }

    static constexpr xdiff::Dual<T, Nvars, Norder> denorm_min() noexcept {
        return xdiff::Dual<T, Nvars, Norder>(numeric_limits<T>::denorm_min());
    }
};

/// @brief Checks if a Dual's value is finite.
template<typename T, size_t Nvars, size_t Norder>
bool isfinite(const xdiff::Dual<T, Nvars, Norder>& x){
    return isfinite(x.value());
}

} // namespace std

#endif // DUAL_HPP