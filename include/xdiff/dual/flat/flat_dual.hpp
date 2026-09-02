#ifndef XDIFF_DUAL_FLAT_HPP
#define XDIFF_DUAL_FLAT_HPP

#include "lexicographic.hpp"
#include <array>
#include <concepts>


#ifdef XDIFF_FAST
#define XDIFF_MAYBE_INLINE XDIFF_INLINE_HOST_DEVICE
#else
#define XDIFF_MAYBE_INLINE inline
#endif


#define XDIFF_DUAL Dual<T, NVARS, NORDER, Layout::Flat>

namespace xdiff{


template<typename T, int NVARS, int NORDER>
class Dual<T, NVARS, NORDER, Layout::Flat> : public DualBase<Dual<T, NVARS, NORDER, Layout::Flat>, T, NVARS, Layout::Flat>, public MultiDiff<NVARS, NORDER> {

    static_assert(NVARS >= 0 && NORDER >= 0, "Flat Dual requires a compile-time known NVARS >= 0 and NORDER >= 0");

    using Base = DualBase<Dual<T, NVARS, NORDER, Layout::Flat>, T, NVARS, Layout::Flat>;
    using MDBase = MultiDiff<NVARS, NORDER>;
public:

    static constexpr int REDUCED_ORDER = NORDER > 0 ? NORDER-1 : 0;

    /// Type after DiffCount differentiations
    template<int DiffCount>
    using Reduced = Dual<T, NVARS, (NORDER > DiffCount ? NORDER-DiffCount : 0), Layout::Flat>;

    using ReducedType = Dual<T, NVARS, REDUCED_ORDER, Layout::Flat>;
    using DataType = std::array<T, MDBase::Ntot>; // internal storage

    // =========== Rule of 5 ===========
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
    XDIFF_INLINE_HOST_DEVICE
    ~Dual() = default;

    // =========== Main constructor ===========
    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(U&& value, MakeDual md = {.axis = -1, .nvars=NVARS, .order=NORDER}) : data_{} {

        if (md.nvars != NVARS && md.nvars != -1) {
            throw std::invalid_argument("nvars must be -1 or match NVARS for compile-time known number of variables in Dual");
        } else if (md.order != NORDER && md.order != -1) {
            throw std::invalid_argument("order must be -1 or match NORDER for compile-time known order in Dual");
        } else if (md.axis < -1 || md.axis >= (md.nvars >= 0 ? md.nvars : NVARS)) {
            throw std::invalid_argument("axis must be -1 or within the number of variables in Dual");
        }

        data_[0] = std::forward<U>(value);
        if constexpr (NORDER > 0) {
            if (md.axis >= 0) {
                data_[1 + md.axis] = 1;
            }
        }
    }

    // =========== Assignment from scalar ===========
    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(U&& other){
        std::fill(data_.begin(), data_.end(), 0);
        data_[0] = std::forward<U>(other);
        return *this;
    }

    // =========== Assignment from Seed ===========
    /**
     * @brief Assigns the seed variable a Seed stands for.
     *
     * The result holds the seed's value and a unit derivative along its own axis; every other
     * derivative, of every order, is zero. The seed's order is deduced rather than named,
     * because a Dual of order zero has no Seed to name.
     */
    template<int seed_order>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Seed<T, NVARS, seed_order, Layout::Flat>& seed) {
        // assert(seed.nvars() == this->nvars() && "nvars must match NVARS when assigning a Seed to a Dual");
        if constexpr (NVARS == -1){
            if (seed.nvars() != this->nvars()) {
                throw std::invalid_argument("seed.nvars() must match this->nvars() when assigning a Seed to a Dual");
            }
        }
        *this = seed.value();
        if constexpr (NORDER > 0) {
            data_[1 + seed.axis()] = 1;
        }
        return *this;
    }

    // =========== Compound assignment ===========
    // These are members so that they can reach the storage directly. Every free
    // assign_compound_* forwards to them, which is why none of those needs to be a friend.

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const Dual& arg){
        for (size_t i=0; i < MDBase::Ntot; i++){
            data_[i] += arg.data_[i];
        }
        return *this;
    }

    template<int seed_order>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const Seed<T, NVARS, seed_order, Layout::Flat>& seed){
        // Index 0 holds the value and the first-order derivatives follow it, so the seed's unit
        if constexpr (NVARS == -1){
            if (seed.nvars() != this->nvars()) {
                throw std::invalid_argument("seed.nvars() must match this->nvars() when adding a Seed to a Dual");
            }
        }
        data_[0] += seed.value();
        if constexpr (NORDER > 0) {
            data_[1 + seed.axis()] = data_[1 + seed.axis()] + 1;
        }
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const U& arg){
        data_[0] += arg;   // adding a constant leaves every derivative untouched
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const Dual& arg){
        for (size_t i=0; i < MDBase::Ntot; i++){
            data_[i] -= arg.data_[i];
        }
        return *this;
    }

    template<int seed_order>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const Seed<T, NVARS, seed_order, Layout::Flat>& seed){
        if constexpr (NVARS == -1){
            if (seed.nvars() != this->nvars()) {
                throw std::invalid_argument("seed.nvars() must match this->nvars() when subtracting a Seed from a Dual");
            }
        }
        data_[0] -= seed.value();
        if constexpr (NORDER > 0) {
            data_[1 + seed.axis()] = data_[1 + seed.axis()] - 1;
        }
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const U& arg){
        data_[0] -= arg;   // subtracting a constant leaves every derivative untouched
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const Dual& arg){
        return *this = *this * arg;
    }

    template<int seed_order>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const Seed<T, NVARS, seed_order, Layout::Flat>& seed){
        return *this = *this * seed;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const U& arg){
        for (size_t i=0; i < MDBase::Ntot; i++){
            data_[i] *= arg;
        }
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const Dual& arg){
        return *this = *this / arg;
    }

    template<int seed_order>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const Seed<T, NVARS, seed_order, Layout::Flat>& seed){
        return *this = *this / seed;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const U& arg){
        for (size_t i=0; i < MDBase::Ntot; i++){
            data_[i] /= arg;
        }
        return *this;
    }

    // =========== Accessors ===========

    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& value() const {
        return data_[0];
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t nvars() const {
        return size_t(NVARS);
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t order() const {
        return size_t(NORDER);
    }


    /**
     * @brief Returns a reduced-order Dual truncated to NORDER-1.
     * @return A Dual with order NORDER-1
     */
    XDIFF_INLINE_HOST_DEVICE
    ReducedType trimmed() const {
        ReducedType out;
        std::copy(data_.data(), data_.data() + ReducedType::Ntot, out.data_.data());
        return out;
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
     *     auto df_dx = f.trimmed_diff_wrt(0);      // df/dx and its derivatives
     *     auto d2f_dxdy = f.trimmed_diff_wrt(0, 1);  // d²f/dxdy
     */
    template<size_t... I>
    XDIFF_MAYBE_INLINE
    auto constexpr trimmed_diff_wrt() const{
        static_assert(sizeof...(I)<=NORDER, "Number of differentiations requested must be <= NORDER");
        using ResType = typename Dual::Reduced<sizeof...(I)>;
        auto constexpr OFFSETS = offsets_for_reduced_diff(I...);
        ResType res;

        for (size_t i=0; i<ResType::Ntot; i++){
            res[i] = data_[OFFSETS[i]];
        }
        return res;
    }

    /**
     * @brief Extracts a partial derivative as a new reduced-order Dual (runtime or mixed).
     *
     * @param x Variables to differentiate with respect to
     * @return A Dual representing the derivative and its higher derivatives
     */
    template<std::integral... Int>
    XDIFF_MAYBE_INLINE
    auto constexpr trimmed_diff_wrt(Int... x) const{
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
     * Similar to trimmed_diff_wrt, but returns a Dual of the same order
     * (padded with zeros for unavailable higher derivatives).
     *
     * @tparam IntType Axis index types
     * @param x Variables to differentiate with respect to (integral indices)
     * @return A Dual of the same order containing the derivative
     */
    template<std::integral... Int>
    XDIFF_INLINE_HOST_DEVICE
    Dual diff_wrt(Int... x) const{
        auto f = this->trimmed_diff_wrt(x...);
        Dual res{};
        std::copy(f.data_.data(), f.data_.data() + f.Ntot, res.data_.data());
        return res;
    }

    /**
     * @brief Returns the scalar value of a specific partial derivative.
     *
     * @tparam Int Integral index types
     * @param x Variables to differentiate with respect to
     * @return The derivative value as a scalar
     *
     * @example
     *     Dual<double, 2, 2, Layout::Flat> f = ...;
     *     f.get_diff_wrt(0);        // df/dx
     *     f.get_diff_wrt(0, 1);     // d²f/dxdy
     *     f.get_diff_wrt(0, 0);  // d²f/dx²
     */
    template<std::integral... Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& get_diff_wrt(Int... x) const{
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
    template<int order>
    XDIFF_INLINE_HOST_DEVICE
    void assign_to(Dual<T, NVARS, order, Layout::Flat>& dual) const {
        static_assert(order >= 0, "Order must be non-negative");
        // NVARS is already non-negative

        // The number of variables must be the same, but the order can differ.
        // We copy all derivatives that are valid in the target dual.
        constexpr int MIN_ORDER = std::min(order, NORDER);
        constexpr int ELEM_COUNT = MultiDiff<NVARS, MIN_ORDER>::Ntot;
        for (int i = 0; i < ELEM_COUNT; i++) {
            dual.data_[i] = data_[i];
        }
    }

    /**
     * @brief Retrieves derivative value by per-variable counts.
     *
     * @tparam Int Integral types
     * @param order_wrt Derivative count for each variable
     * @return The derivative value
     *
     * @example
     *     f.value_of_diff_counts(2, 1, 0);  // d³f/dx²dy
     */
    template<std::integral... Int>
    XDIFF_INLINE_HOST_DEVICE T value_of_diff_counts(Int... order_wrt)const{
        return data_[MDBase::offset(order_wrt...)];
    }

    /**
     * @brief Precomputes offsets for trimmed_diff_wrt extraction.
     *
     * Used internally to efficiently extract derivatives into a reduced Dual.
     *
     * @tparam IntType Axis types
     * @param x Variables to differentiate with respect to
     * @return Array of offsets for extracting the reduced derivative
     */
    template<std::integral... Int>
    XDIFF_MAYBE_INLINE
    static constexpr auto offsets_for_reduced_diff(Int... x){
        constexpr int NEW_ORDER = NORDER-sizeof...(x);
        static_assert(NEW_ORDER >= 0, "Too many indices to differentiate wrt, resulting in negative new order.");

        std::array<size_t, Dual<T, NVARS, NEW_ORDER, Layout::Flat>::Ntot> res{};
        size_t n = 0;

        std::array<size_t, NVARS> nx = MDBase::diff_count(x...);

        auto call_it = [&] XDIFF_DEVICE <size_t... I>(std::integer_sequence<size_t, I...>) XDIFF_ALWAYS_INLINE {
            [&] XDIFF_DEVICE <size_t... Ord>(std::index_sequence<Ord...>) XDIFF_ALWAYS_INLINE {
                ([&] XDIFF_DEVICE <size_t OrdI>() XDIFF_ALWAYS_INLINE {
                    using IterType = tools::MultiSetIterator<OrdI+sizeof...(x), NVARS, true>;
                    IterType::apply_iter_on(
                        [&] XDIFF_DEVICE (const IterType::SetType&, const IterType::CounterType& order_of_var) XDIFF_ALWAYS_INLINE {
                            if ((((order_of_var[I] >= nx[I])) &&...)){
                                res[n++] = MDBase::offset(order_of_var[I]...);
                            }
                        }
                    );
                }.template operator()<Ord>(), ...);
            }(std::make_index_sequence<NEW_ORDER+1>{});
        };

        call_it(std::make_index_sequence<NVARS>{});
        return res;
    }


    template<typename Action>
    inline static constexpr void with_default_nvars([[maybe_unused]] size_t nvars, Action&& action) {
        assert((nvars == NVARS) && "nvars must match NVARS for compile-time known number of variables in flat Dual");
        std::forward<Action>(action)();
    }

    operator T() const = delete; // Disable implicit conversion to T to avoid accidental loss of derivative information.

protected:

    // The library internals that operate on the raw storage below. Every other user of a Dual goes through the public interface.
    template<typename STRUCT> friend struct detail::HelperBaseOperandEvaluator;
    template<typename STRUCT> friend struct detail::BaseOperandEvaluator;
    template<typename STRUCT> friend struct detail::OperandEvaluator;





    /// @brief Returns a reference to the element at the given offset in the internal data array. Index 0 is the value, and the derivatives follow in graded order.
    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr T& operator[](Int idx){
        return data_[idx];
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& operator[](Int idx) const {
        return data_[idx];
    }

    /// @brief Returns const reference to internal data array.
    const DataType& data() const {
        return data_;
    }

    /// @brief Returns mutable reference to internal data array.
    DataType& data() {
        return data_;
    }


private:

    template<typename U, int M, int P, Layout LY>
    friend class Dual;   // every Dual is a friend

    DataType data_;

};


namespace detail{


template<typename STRUCT>
struct HelperBaseOperandEvaluator{

    using T = typename STRUCT::value_type;

    template<int Nvars, int Norder, typename... U>
    XDIFF_MAYBE_INLINE
    static Dual<T, Nvars, Norder, Layout::Flat>& optimized_eval(Dual<T, Nvars, Norder, Layout::Flat>& out, const U&... f){
        using EV = Evaluator<Nvars, Norder>;
        using AD = typename EV::AD;

        // Compute the function value
        T v = STRUCT::operation(EV::get_value(f)...);

        if constexpr (!(Norder > 0 && Nvars > 0)) {
            out = AD(v);
        }

        if constexpr (Norder > 0 && Nvars > 0) {
            using RT = typename AD::ReducedType;

            // Compute derivative w.r.t. each variable using the diff_rule
            auto q = [&] XDIFF_DEVICE <size_t I> (auto&&... g) XDIFF_ALWAYS_INLINE {
                return RT(STRUCT::diff_rule(DiffPair{EV::reduced_value(g), EV::template reduced_diff<I>(g)}...));
            };

            // h[i] contains df/dx_i and all its higher derivatives. The explicit conversion is
            // needed because a rule applied to seed variables alone reduces to a bare scalar:
            // a seed's derivative is the constant 1, so no reduced Dual has to be materialized.
            auto h = XDIFF_EXPAND(Nvars, I,
                return std::array<RT, Nvars>{q.template operator()<I>(f...)...};
            );

            out = AD(v);

            size_t n = 1;

            // Assemble derivatives into result array in graded order
            XDIFF_EXPAND(Norder, Ord,
                auto g = [&] XDIFF_DEVICE <size_t ord>() XDIFF_ALWAYS_INLINE {
                    XDIFF_EXPAND(Nvars, Ivar,
                        auto R = [&] XDIFF_DEVICE <size_t var>() XDIFF_ALWAYS_INLINE {
                            constexpr size_t Noff_tot = Dual<T, Nvars, AD::REDUCED_ORDER, Layout::Flat>::offset(ord*(var==Ivar)...);
                            constexpr size_t Nelements = 
                                Dual<T, Nvars, AD::REDUCED_ORDER, Layout::Flat>::ndiffs(ord)
                                -Dual<T, Nvars, AD::REDUCED_ORDER, Layout::Flat>::local_offset(ord*(var==Ivar)...);
                            std::copy(h[var].data().data()+Noff_tot, h[var].data().data()+Noff_tot+Nelements, out.data().data()+n);
                            n += Nelements;
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
    template<int Nvars, int Norder>
    struct Evaluator {
        using AD = Dual<T, Nvars, Norder, Layout::Flat>;

        // Every Seed overload below deduces its order rather than naming Norder, because
        // Evaluator is also instantiated at Norder == 0, where no Seed can exist.

        /// @brief Extracts the scalar value from a Dual.
        XDIFF_INLINE_DEVICE
        static const T& get_value(const AD& f) {
            return f.value();
        }

        /// @brief Extracts the scalar value from a Seed.
        template<int Nv, int No>
        XDIFF_INLINE_DEVICE
        static const T& get_value(const Seed<T, Nv, No, Layout::Flat>& f) {
            return f.value();
        }

        /// @brief A scalar argument is already its own value.
        template<typename U>
        XDIFF_INLINE_DEVICE
        static const U& get_value(const U& f) {
            static_assert(std::is_convertible_v<U, T>, "Invalid argument");
            return f;
        }

        /// @brief Gets the reduced-order representation of a Dual.
        template<int Nv, int No>
        XDIFF_INLINE_DEVICE
        static decltype(auto) reduced_value(const Dual<T, Nv, No, Layout::Flat>& f){
            return f.trimmed();
        }

        /// @brief Gets the reduced-order representation of a Seed.
        template<int Nv, int No>
        XDIFF_INLINE_DEVICE
        static decltype(auto) reduced_value(const Seed<T, Nv, No, Layout::Flat>& f){
            return f.trimmed();
        }

        /// @brief Scalars have trivial reduced representation.
        template<typename U>
        XDIFF_INLINE_DEVICE
        static const U& reduced_value(const U& f){
            return f;
        }

        /// @brief Gets the reduced derivative w.r.t. variable I from a Dual.
        template<size_t I>
        XDIFF_INLINE_DEVICE
        static auto reduced_diff(const AD& f){
            return f.template trimmed_diff_wrt<I>();
        }

        /// @brief Gets the reduced derivative w.r.t. variable I from a Seed. A seed variable
        /// differentiates to the constant 1 along its own axis, and to 0 along every other.
        template<size_t I, int Nv, int No>
        XDIFF_INLINE_DEVICE
        static auto reduced_diff(const Seed<T, Nv, No, Layout::Flat>& f){
            return (f.axis() == I ? 1 : 0);
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



template<int Norder, int Nvars>
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

    template<int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& leibniz_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
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

    template<int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        if (&out == &f || &out == &g){
            XDIFF_DUAL tmp;
            leibniz_eval<NVARS, NORDER>(tmp, f, g);
            out = tmp;
            return out;
        }
        return leibniz_eval<NVARS, NORDER>(out, f, g);
    }
#endif // XDIFF_LEIBNIZ_OPT

    template<size_t ORDER, int Norder, int Nvars>
    XDIFF_INLINE_HOST_DEVICE static void apply_diffs(Dual<T, Nvars, Norder, Layout::Flat>& res, const Dual<T, Nvars, Norder, Layout::Flat>& f, const Dual<T, Nvars, Norder, Layout::Flat>& g){
        static_assert(ORDER<=Norder, "ORDER too large");
        constexpr size_t glf = Dual<T, Nvars, Norder, Layout::Flat>::global_offset(ORDER);
        T* d = res.data().data() + glf;
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
    template<int Norder, int Nvars, std::integral... IntType>
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
struct OperandEvaluator<rules::Add<T>> : public BaseOperandEvaluator<rules::Add<T>>{

    using Base = BaseOperandEvaluator<rules::Add<T>>;
    using Base::optimized_eval;   // generic fallback for operands the fast paths below exclude

    template<int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] + g[i];
        }
        return out;
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        out[0] = f[0] + g;
        for (size_t i=1; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i];
        }
        return out;
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
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
struct OperandEvaluator<rules::Sub<T>> : public BaseOperandEvaluator<rules::Sub<T>>{

    using Base = BaseOperandEvaluator<rules::Sub<T>>;
    using Base::optimized_eval;   // generic fallback for operands the fast paths below exclude

    template<int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] - g[i];
        }
        return out;
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        out[0] = f[0] - g;
        for (size_t i=1; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i];
        }
        return out;
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
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
    using Base::optimized_eval;   // generic fallback for operands the fast paths below exclude

    template<int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        return Base::optimized_eval(out, f, g);
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] * g;
        }
        return out;
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
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
    using Base::optimized_eval;   // generic fallback for operands the fast paths below exclude

    template<int NVARS, int NORDER>
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const XDIFF_DUAL& g){
        return Base::optimized_eval(out, f, g);
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
    XDIFF_INLINE_HOST_DEVICE
    static XDIFF_DUAL& optimized_eval(XDIFF_DUAL& out, const XDIFF_DUAL& f, const U& g){
        for (size_t i=0; i<XDIFF_DUAL::Ntot; i++){
            out[i] = f[i] / g;
        }
        return out;
    }

    template<typename U, int NVARS, int NORDER>
    requires (isScalarOperand<U, T>)
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