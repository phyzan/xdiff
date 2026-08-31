#ifndef XDIFF_DUAL_FLAT_LEXICOGRAPHIC_HPP
#define XDIFF_DUAL_FLAT_LEXICOGRAPHIC_HPP

#include "itertools.hpp"

namespace xdiff {

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
     * @param x Variable indices
     * @return Array with count of each variable
     *
     * @example
     *     diff_count(0, 1, 0);  // Returns {2, 1, 0, ...} for d²f/dx²dy
     */
    template<std::integral... Var>
    XDIFF_INLINE_HOST_DEVICE
    static constexpr std::array<size_t, Nvars> diff_count(Var... x){
        return XDIFF_EXPAND(Nvars, I,
            return std::array<size_t, Nvars>{tools::var_count(I, x...)...};
        );
    }
};

namespace detail{

// Defined further below, after the Dual class. Declared here so that Dual can grant them raw access to its storage.
template<typename STRUCT> struct HelperBaseOperandEvaluator;
template<typename STRUCT> struct BaseOperandEvaluator;
template<typename STRUCT> struct OperandEvaluator;

} // namespace detail

} // namespace xdiff


#endif // XDIFF_DUAL_FLAT_LEXICOGRAPHIC_HPP