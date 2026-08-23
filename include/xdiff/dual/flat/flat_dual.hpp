#ifndef XDIFF_DUAL_FLAT_HPP
#define XDIFF_DUAL_FLAT_HPP

#include "itertools.hpp"
#include <array>


#ifdef XDIFF_FAST
#define XDIFF_MAYBE_INLINE XDIFF_INLINE_HOST_DEVICE
#else
#define XDIFF_MAYBE_INLINE inline
#endif


#define XDIFF_DUAL Dual<T, NVARS, NORDER, Layout::Flat>

namespace xdiff{

template<size_t NVARS, size_t NORDER>
struct MultiDiff{
    static constexpr size_t Nvars = NVARS;   ///< Number of independent variables
    static constexpr size_t Norder = NORDER;   ///< Maximum derivative order
    static constexpr size_t Ntot = xdiff::tools::comb(NVARS+NORDER, NORDER); ///< Total stored values (value + all derivatives)

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
        return tools::comb(NVARS+order-1, order);
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
        assert(total_order <= NORDER && "diff order must be <= NORDER");

        // Compute colexicographic offset within the group
        size_t colex_offset = XDIFF_EXPAND(NX, variable,
            return ([&] XDIFF_DEVICE () XDIFF_ALWAYS_INLINE {
                constexpr size_t v = variable;
                size_t truncated_total = XDIFF_EXPAND(NX, I,
                    return ((static_cast<size_t>(tools::pack_elem<I>(order...))*(I<v))+...);
                );
                size_t res = 0;
                for (size_t j = 0; j < static_cast<size_t>(tools::pack_elem<v>(order...)); j++){
                    size_t remaining = total_order - truncated_total - j;
                    if (remaining > 0 && Nvars - v - 1 > 0) {
                        res += tools::comb(Nvars - v - 1 + remaining - 1, remaining);
                    } else if (remaining == 0) {
                        res += 1;
                    }
                }
                return res;
            }()+...);
        );

        size_t total_for_order = tools::multiset_coef(Nvars, total_order);
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
            if (order > NORDER) {
                throw "order > NORDER in constexpr evaluation";
            }
        } else {
            assert(order <= NORDER && "Requested order is > NORDER");
        }
        return order == 0 ? 0 : tools::comb(Nvars + order - 1, order - 1);
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
        return XDIFF_EXPAND(Nvars, I,
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
    template<xdiff::traits::isAxis... Var>
    XDIFF_INLINE_HOST_DEVICE
    static constexpr std::array<size_t, Nvars> diff_count(Var... x){
        return XDIFF_EXPAND(Nvars, I,
            return std::array<size_t, Nvars>{tools::var_count(I, x...)...};
        );
    }
};

template<typename T, size_t NVARS, size_t NORDER>
requires (NVARS > 0)
class Dual<T, NVARS, NORDER, Layout::Flat> : public DualBase<Dual<T, NVARS, NORDER, Layout::Flat>, T, NVARS, Layout::Flat>, public MultiDiff<NVARS, NORDER> {

    using Base = DualBase<Dual<T, NVARS, NORDER, Layout::Flat>, T, NVARS, Layout::Flat>;
    using MDBase = MultiDiff<NVARS, NORDER>;
public:
    static constexpr size_t Nvars = NVARS;   ///< Number of independent variables
    static constexpr size_t Norder = NORDER;   ///< Maximum derivative order
    static constexpr size_t REDUCED_ORDER = NORDER > 0 ? NORDER-1 : 0;

    template<typename U, size_t M, size_t P, Layout LY>
    friend class Dual;   // every Dual is a friend

    /// Type after DiffCount differentiations
    template<size_t DiffCount>
    using Reduced = Dual<T, NVARS, (NORDER > DiffCount ? NORDER-DiffCount : 0), Layout::Flat>;

    /// Type after one differentiation
    using ReducedType = Dual<T, NVARS, REDUCED_ORDER, Layout::Flat>;

    /// Internal storage type
    using DataType = std::array<T, MultiDiff<NVARS, NORDER>::Ntot>;

    /// Array type for storing offsets after DiffCount differentiations
    template<size_t DiffCount>
    using ReducedArray = std::array<size_t, MultiDiff<NVARS, (NORDER > DiffCount ? NORDER-DiffCount : 0)>::Ntot>;

    // Default rule of five
    XDIFF_INLINE_HOST_DEVICE
    Dual() = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual(const Dual&) = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual(Dual&&) = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Dual&) = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(Dual&&) = default;
    ~Dual() = default;

    template<typename U>
    requires (!std::is_same_v<std::decay_t<U>, Dual>)
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(U&& value, MakeDual md = {.axis = -1, .nvars=NVARS, .order=NORDER}) {
        data_[0] = std::forward<U>(value);
        assert((md.nvars == NVARS || md.nvars == 0) && "nvars must match NVARS for compile-time known number of variables in Dual");
        assert((md.order == NORDER || md.order == 0) && "order must match NORDER for compile-time known order in Dual");
        assert(md.axis < int(NVARS) && "Axis index must be within the number of derivatives");
        if (md.axis >= 0) {
            data_[1 + md.axis] = 1;
        }
    }

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, Dual>) {
        Base::operator=(std::forward<U>(other));
        return *this;
    }

    operator T() const = delete; // Disable implicit conversion to T to avoid accidental loss of derivative information.

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    T& operator[](Int idx){
        return data_[idx];
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    const T& operator[](Int idx) const {
        return data_[idx];
    }

    XDIFF_INLINE_HOST_DEVICE
    const T& value() const {
        return data_[0];
    }

    XDIFF_INLINE_HOST_DEVICE
    size_t nvars() const {
        return NVARS;
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
            tools::copy_array(out.data_.data(), data_.data(), ReducedType::Ntot);
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
    XDIFF_MAYBE_INLINE
    auto constexpr reduced_diff_wrt(Symbol<I>... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        using ResType = typename Dual::Reduced<sizeof...(I)>;
        auto constexpr OFFSETS = offsets_for_reduced_diff(x...);
        ResType res;

        for (size_t i=0; i<ResType::Ntot; i++){
            res[i] = data_[OFFSETS[i]];
        }
        return res;
    }

    /**
     * @brief Extracts a partial derivative as a new reduced-order Dual (runtime or mixed).
     *
     * @tparam IntType Axis index types (Symbol or integral)
     * @param x Variables to differentiate with respect to
     * @return A Dual representing the derivative and its higher derivatives
     */
    template<xdiff::traits::isAxis... IntType>
    XDIFF_MAYBE_INLINE
    auto constexpr reduced_diff_wrt(IntType... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        using ResType = typename Dual::Reduced<sizeof...(x)>;
        auto offsets = offsets_for_reduced_diff(x...);
        ResType res;
        for (size_t i=0; i<ResType::Ntot; i++){
            res[i] = data_[offsets[i]];
        }
        return res;
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
    template<xdiff::traits::isAxis... IntType>
    XDIFF_INLINE_HOST_DEVICE
    Dual diff_wrt(IntType... x) const{
        auto f = this->reduced_diff_wrt(x...);
        Dual res;
        tools::copy_array(res.data_.data(), f.data_.data(), f.NTOT);
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
    template<xdiff::traits::isAxis... IntType>
    XDIFF_INLINE_HOST_DEVICE
    const T& get_diff_wrt(IntType... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        auto nx = MDBase::diff_count(x...);
        return data_[MDBase::offset(nx)];
    }

    /**
     * @brief Assigns this Dual's data to another Dual (possibly different order).
     *
     * @tparam Norder The target Dual's order
     * @param dual The target Dual to assign to
     */
    template<size_t Norder>
    XDIFF_INLINE_HOST_DEVICE
    void assign_to(Dual<T, NVARS, Norder, Layout::Flat>& dual) const {
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
        return data_[MDBase::offset(order_wrt...)];
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
    template<xdiff::traits::isAxis... IntType>
    requires (sizeof...(IntType)<=NORDER)
    XDIFF_MAYBE_INLINE
    static constexpr ReducedArray<sizeof...(IntType)> offsets_for_reduced_diff(IntType... x){
        constexpr size_t NEW_ORDER = NORDER-static_cast<size_t>(sizeof...(x));

        std::array<size_t, Dual<T, NVARS, NEW_ORDER, Layout::Flat>::Ntot> res{};
        size_t n = 0;

        std::array<size_t, NVARS> nx = MDBase::diff_count(x...);

        auto call_it = [&] XDIFF_DEVICE <size_t... I>(std::integer_sequence<size_t, I...>) XDIFF_ALWAYS_INLINE {
            [&] XDIFF_DEVICE <size_t... Ord>(std::index_sequence<Ord...>) XDIFF_ALWAYS_INLINE {
                ([&] XDIFF_DEVICE <size_t OrdI>() XDIFF_ALWAYS_INLINE {
                    using IterType = tools::MultiSetIterator<OrdI+sizeof...(x), Nvars, true>;
                    IterType::apply_iter_on(
                        [&] XDIFF_DEVICE (const IterType::SetType&, const IterType::CounterType& order_of_var) XDIFF_ALWAYS_INLINE {
                            if ((((order_of_var[I] >= nx[I])) &&...)){
                                res[n++] = MDBase::offset(order_of_var[I]...);
                            }
                        }
                    );
                }.template operator()<Ord>(), ...);
            }(std::make_index_sequence<NEW_ORDER+1UL>{});
        };

        call_it(std::make_index_sequence<Nvars>{});
        return res;
    }


    template<typename Action>
    inline static constexpr void with_default_nvars([[maybe_unused]] size_t nvars, Action&& action) {
        assert((nvars == NVARS) && "nvars must match NVARS for compile-time known number of variables in flat Dual");
        std::forward<Action>(action)();
    }


private:

    DataType data_ = {};

};


namespace detail{


template<typename STRUCT>
struct HelperBaseOperandEvaluator{

    using T = typename STRUCT::value_type;

    template<size_t Nvars, size_t Norder, typename... U>
    XDIFF_MAYBE_INLINE
    static Dual<T, Nvars, Norder, Layout::Flat>& optimized_eval(Dual<T, Nvars, Norder, Layout::Flat>& out, const U&... f){
        using EV = Evaluator<Nvars, Norder>;
        using AD = typename EV::AD;

        // Compute the function value
        T v = STRUCT::operation(EV::get_value(f)...);
        out = AD(v);

        if constexpr (Norder > 0) {
            // Compute derivative w.r.t. each variable using the diff_rule
            auto q = [&] XDIFF_DEVICE <size_t I> (auto&&... g) XDIFF_ALWAYS_INLINE {
                return STRUCT::diff_rule(DiffPair{EV::reduced_value(g), EV::template reduced_diff<I>(g)}...);
            };

            // h[i] contains df/dx_i and all its higher derivatives
            auto h = XDIFF_EXPAND(Nvars, I,
                return std::array<typename AD::ReducedType, Nvars>{q.template operator()<I>(f...)...};
            );

            size_t n = 1;

            // Assemble derivatives into result array in graded order
            XDIFF_EXPAND(Norder, Ord,
                auto g = [&] XDIFF_DEVICE <size_t ord>() XDIFF_ALWAYS_INLINE {
                    XDIFF_EXPAND(Nvars, Ivar,
                        auto R = [&] XDIFF_DEVICE <size_t var>() XDIFF_ALWAYS_INLINE {
                            constexpr size_t Noff_tot = Dual<T, Nvars, AD::REDUCED_ORDER, Layout::Flat>::offset(ord*(var==Ivar)...);
                            constexpr size_t Nelements = Dual<T, Nvars, AD::REDUCED_ORDER, Layout::Flat>::ndiffs(ord)-Dual<T, Nvars, AD::REDUCED_ORDER, Layout::Flat>::local_offset(ord*(var==Ivar)...);
                            xdiff::tools::copy_array(out.data().data()+n, h[var].data().data()+Noff_tot, Nelements);
                            n+=Nelements;
                        };
                        (R.template operator()<Ivar>(), ...);
                    );
                };
                (g.template operator()<Ord>(), ...);
            );
        }
        return out;
    }


    /**
     * @brief Helper struct for evaluating arguments in operations.
     *
     * Provides utilities for extracting values and derivatives from
     * different argument types (Dual, Expr, or scalar).
     */
    template<size_t Nvars, size_t Norder>
    struct Evaluator {
        using AD = Dual<T, Nvars, Norder, Layout::Flat>;

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
        XDIFF_INLINE_DEVICE
        static T get_value(const Dual<T, Nvars, Norder, Layout::Flat>& f) {
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
        static auto reduced_value(const Dual<T, Nv, No, Layout::Flat>& f){
            return f.reduced();
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
            return ZeroValue{};
        }

    };

};

template<typename STRUCT>
struct BaseOperandEvaluator : public HelperBaseOperandEvaluator<STRUCT>{};



template<size_t Norder, size_t Nvars>
struct LeibnizDiff{

    static constexpr size_t NDIFFS = xdiff::tools::comb(Nvars+Norder, Norder);

    /**
     * @brief Iterates over all Leibniz rule terms.
     * @param f_main Called for each output derivative.
     * @param f_dummy Called for each term in the Leibniz sum.
     */
    template<typename Callable1, typename Callable2>
    static constexpr void iterate(Callable1&& f_main, Callable2&& f_dummy){
        XDIFF_FOR_LOOP(Iord, Norder+1,
            using IterType = xdiff::tools::MultiSetIterator<Iord, Nvars, true>;
            XDIFF_EXPAND(Nvars, Ivar,
                auto main_func = [&] XDIFF_DEVICE (auto&, auto& ord_of_var) XDIFF_LAMBDA_INLINE {
                    auto dummy_func = [&] XDIFF_DEVICE (auto... dummy_order) XDIFF_LAMBDA_INLINE {
                        f_dummy(Iord, ord_of_var, std::array<size_t, Nvars>({dummy_order...}));
                    };
                    xdiff::tools::for_each(dummy_func, (ord_of_var[Ivar]+1)...);
                    f_main(Iord, ord_of_var);
                };
                IterType::apply_iter_on(main_func);
            );
        );
    }

    /// @brief Total number of terms across all Leibniz sums.
    static constexpr size_t total_cache_count(){
        size_t res = 0;
        auto f_dummy = [&] XDIFF_DEVICE (size_t /*order*/, auto /*order_wrt*/, auto /*dummy_order_wrt*/) XDIFF_LAMBDA_INLINE {
            res++;
        };
        iterate([] XDIFF_DEVICE (auto, auto) XDIFF_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    /// @brief Number of terms in each derivative's Leibniz sum.
    static constexpr std::array<size_t, NDIFFS> Nsum_per_offset(){
        std::array<size_t, NDIFFS> res{};
        size_t i=0;
        auto f_main = [&] XDIFF_DEVICE (size_t, auto) XDIFF_LAMBDA_INLINE {
            i++;
        };
        auto f_dummy = [&] XDIFF_DEVICE (size_t, auto, auto) XDIFF_LAMBDA_INLINE {
            res[i]++;
        };
        iterate(f_main, f_dummy);
        return res;
    }

    /// @brief Multinomial coefficients for each Leibniz term.
    static constexpr auto cached_coefs(){
        std::array<size_t, total_cache_count()> res{};
        size_t i=0;
        auto f_dummy = [&] XDIFF_DEVICE (size_t, auto order_wrt, auto dummy_order_wrt) XDIFF_LAMBDA_INLINE {
            XDIFF_EXPAND(Nvars, Ivar,
                res[i++] = (xdiff::tools::comb(order_wrt[Ivar], dummy_order_wrt[Ivar])*...);
            );
        };
        iterate([] XDIFF_DEVICE (auto, auto) XDIFF_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    /// @brief Offsets into left operand for each Leibniz term.
    static constexpr auto cached_left_offsets(){
        std::array<size_t, total_cache_count()> res{};
        size_t i=0;
        auto f_dummy = [&] XDIFF_DEVICE (size_t, auto, auto dummy_order_wrt) XDIFF_LAMBDA_INLINE {
            XDIFF_EXPAND(Nvars, Ivar,
                res[i++] = MultiDiff<Nvars, Norder>::offset(dummy_order_wrt[Ivar]...);
            );
        };
        iterate([] XDIFF_DEVICE (auto, auto) XDIFF_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    /// @brief Offsets into right operand for each Leibniz term.
    static constexpr auto cached_right_offsets(){
        std::array<size_t, total_cache_count()> res{};
        size_t i=0;
        auto f_dummy = [&] XDIFF_DEVICE (size_t, auto order_wrt, auto dummy_order_wrt) XDIFF_LAMBDA_INLINE {
            XDIFF_EXPAND(Nvars, Ivar,
                res[i++] = MultiDiff<Nvars, Norder>::offset((order_wrt[Ivar]-dummy_order_wrt[Ivar])...);
            );
        };
        iterate([] XDIFF_DEVICE (auto, auto) XDIFF_LAMBDA_INLINE {}, f_dummy);
        return res;
    }

    static constexpr auto Nsum_of = Nsum_per_offset();   ///< Terms per derivative.
    static constexpr auto coefs = cached_coefs();         ///< Multinomial coefficients.
    static constexpr auto cached_left = cached_left_offsets();   ///< Left operand indices.
    static constexpr auto cached_right = cached_right_offsets(); ///< Right operand indices.

};

template<typename T>
struct BaseOperandEvaluator<rules::Mul<T>> : public HelperBaseOperandEvaluator<rules::Mul<T>>{
    /**
     * @brief Applies Leibniz rule for a specific derivative order.
     * @tparam ORDER The total derivative order to compute.
     */
    using Base = HelperBaseOperandEvaluator<rules::Mul<T>>;

#ifdef XDIFF_LEIBNIZ_OPT
    using Base::optimized_eval; //use all overloads from Base, but override the next one

    template<size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        using Cache = LeibnizDiff<NORDER, NVARS>;

        size_t n=0;

        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            T sum = 0;
            for (size_t k=0; k<Cache::Nsum_of[i]; k++){
                sum += T(Cache::coefs[n+k])*f[Cache::cached_left[n+k]]*g[Cache::cached_right[n+k]];
            }
            n += Cache::Nsum_of[i];
            out[i] = sum;
        }

        return out;
    }
#endif // XDIFF_LEIBNIZ_OPT

    template<size_t ORDER, size_t Norder, size_t Nvars>
    XDIFF_INLINE_HOST_DEVICE static void apply_diffs(Dual<T, Nvars, Norder, Layout::Flat>& res, const Dual<T, Nvars, Norder, Layout::Flat>& f, const Dual<T, Nvars, Norder, Layout::Flat>& g){
        static_assert(ORDER<=Norder, "ORDER too large");
        constexpr size_t glf = Dual<T, Nvars, Norder, Layout::Flat>::global_offset(ORDER);
        T* d = res.data()+glf;
        using IterType = xdiff::tools::MultiSetIterator<ORDER, Nvars, true>;

        [&] XDIFF_DEVICE <size_t... Ivar>(std::index_sequence<Ivar...>) XDIFF_LAMBDA_INLINE {
            size_t n_iter = 0;
            auto func = [&] XDIFF_DEVICE (auto&, auto& ord_of_var) XDIFF_LAMBDA_INLINE {
                d[n_iter++] = diff_element(f, g, ord_of_var[Ivar]...);
            };
            IterType::apply_iter_on(func);
        }(std::make_index_sequence<Nvars>{});
    }

    /**
     * @brief Computes a single derivative element via Leibniz summation.
     * @param order Per-variable derivative counts.
     * @return The derivative value at the specified multi-index.
     */
    template<size_t Norder, size_t Nvars, std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE static T diff_element(const Dual<T, Nvars, Norder, Layout::Flat>& f, const Dual<T, Nvars, Norder, Layout::Flat>& g, IntType... order){
        T res{0};
        auto func = [&] XDIFF_DEVICE (auto... dummy_order) XDIFF_LAMBDA_INLINE {
            res += T((xdiff::tools::comb(order, dummy_order)*...))*f.value_of_diff_counts(dummy_order...)*g.value_of_diff_counts((order-dummy_order)...);
        };
        xdiff::tools::for_each(func, (order+1)...);
        return res;
    }
};

template<typename STRUCT>
struct OperandEvaluator : public BaseOperandEvaluator<STRUCT>{};


#ifdef XDIFF_SCALAR_OPTIMIZATIONS

template<typename T>
struct OperandEvaluator<rules::Add<T>>{

    template<size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] + g[i];
        }
        return out;
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        out[0] = f[0] + g;
        for (size_t i=1; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i];
        }
        return out;
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const U& f, const XDIFF_DUAL& g){
        out[0] = f + g[0];
        for (size_t i=1; i<XDIFF_DUAL::Ntot; i++){
            out[i] = g[i];
        }
        return out;
    }

};

template<typename T>
struct OperandEvaluator<rules::Sub<T>>{

    template<size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] - g[i];
        }
        return out;
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        out[0] = f[0] - g;
        for (size_t i=1; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i];
        }
        return out;
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const U& f, const XDIFF_DUAL& g){
        out[0] = f - g[0];
        for (size_t i=1; i<XDIFF_DUAL::Ntot; i++){
            out[i] = -g[i];
        }
        return out;
    }
};


template<typename T>
struct OperandEvaluator<rules::Mul<T>> : public BaseOperandEvaluator<rules::Mul<T>>{

    using Base = BaseOperandEvaluator<rules::Mul<T>>;

    template<size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        return Base::optimized_eval(out, f, g);
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] * g;
        }
        return out;
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const U& f, const XDIFF_DUAL& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f * g[i];
        }
        return out;
    }
};


template<typename T>
struct OperandEvaluator<rules::Div<T>> : public BaseOperandEvaluator<rules::Div<T>>{

    using Base = BaseOperandEvaluator<rules::Div<T>>;

    template<size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        return Base::optimized_eval(out, f, g);
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] / g;
        }
        return out;
    }

    template<typename U, size_t NVARS, size_t NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const U& f, const XDIFF_DUAL& g){
        return Base::optimized_eval(out, f, g);
    }
};

#endif // XDIFF_SCALAR_OPTIMIZATIONS

} // namespace xdiff::detail



} // namespace xdiff

#undef XDIFF_DUAL

#include "flat_dual_operators.hpp" // IWYU pragma: keep

#endif //XDIFF_DUAL_FLAT_HPP