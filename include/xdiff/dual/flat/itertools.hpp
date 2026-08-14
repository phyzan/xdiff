#ifndef XDIFF_DUAL_FLAT_ITERTOOLS_HPP
#define XDIFF_DUAL_FLAT_ITERTOOLS_HPP

#include <cassert>
#include <numeric>
#include "../../rules/math.hpp" // IWYU pragma: keep

namespace xdiff::tools{

// =============================================================================
// Combinatorial functions
// =============================================================================

/**
 * @brief Computes binomial coefficient C(n, k) = n! / (k! * (n-k)!)
 *
 * Uses an overflow-safe iterative algorithm with GCD reduction.
 * Constexpr-compatible for compile-time computation.
 *
 * @param n Total count
 * @param k Selection count
 * @return C(n, k)
 */
XDIFF_HOST_DEVICE constexpr size_t comb(size_t n, size_t k) {
    assert(n >= k);

    k = std::min(k, n - k);

    size_t res = 1;
    for (size_t i = 1; i <= k; ++i) {
        size_t num = n - i + 1;
        size_t den = i;

        // Reduce to prevent overflow
        size_t g = std::gcd(num, den);
        num /= g;
        den /= g;

        g = std::gcd(res, den);
        res /= g;
        den /= g;

        res *= num;   // den is now 1
    }
    return res;
}

/**
 * @brief Computes the number of multisets of size k from n categories.
 *
 * This equals C(n+k-1, k), the number of ways to choose k items from
 * n categories with repetition allowed.
 *
 * In the context of autodiff, this is the number of unique partial
 * derivatives of order k for n variables.
 *
 * @param n Number of categories (variables)
 * @param k Size of multiset (derivative order)
 * @return Number of multisets
 *
 * @example
 *     multiset_coef(3, 2);  // = 6 (xx, xy, xz, yy, yz, zz)
 */
XDIFF_INLINE_HOST_DEVICE constexpr size_t multiset_coef(size_t n, size_t k){
    return comb(n+k-1, k);
}


/**
 * @brief Iterator over multisets (combinations with repetition).
 *
 * Iterates over all ways to choose Slots items from Rank categories,
 * where each item can be repeated and order doesn't matter.
 *
 * Used for iterating over partial derivative indices where:
 * - Slots = derivative order (number of differentiations)
 * - Rank = number of variables
 *
 * @tparam Slots Number of items to choose
 * @tparam Rank Number of categories
 * @tparam CountSlots If true, also maintains per-category counts
 */
template<size_t Slots, size_t Rank, bool CountSlots>
class MultiSetIterator{

public:

    using SetType = std::array<size_t, Slots>;       ///< The multiset representation
    using CounterType = std::array<size_t, Rank>;    ///< Per-category counts

    /**
     * @brief Applies a callable to each multiset.
     *
     * @param f Callable taking (const SetType&, const CounterType&)
     */
    template<typename Callable>
    XDIFF_INLINE_DEVICE static constexpr void apply_iter_on(Callable&& f){
        SetType set{};
        CounterType counter{};
        doit<0>(set, counter,std::forward<Callable>(f));
    }

    /**
     * @brief Recursive implementation of multiset iteration.
     *
     * Each slot chooses a value >= the previous slot's value (weak ordering).
     * This ensures we enumerate each multiset exactly once.
     */
    template<size_t Slot, typename Callable>
    XDIFF_INLINE_DEVICE static void constexpr doit(SetType& set, CounterType& counter, Callable&& f){
        if constexpr (Slot < Slots) {
            size_t& var = set[Slot];
            if constexpr (Slot == 0) {
                var = 0;
            }
            else{
                var = set[Slot-1];  // Ensure weak ordering
            }
            if constexpr (CountSlots) {
                counter[var]++;
            }

            do {
                doit<Slot+1>(set, counter, std::forward<Callable>(f));
                if constexpr (CountSlots) {
                    counter[var++]--;
                    if (var < Rank){
                        counter[var]++;
                    }
                }
                else{
                    var++;
                }
            }while (var < Rank);
        }
        else{
            f(static_cast<const SetType&>(set), static_cast<const CounterType&>(counter));
        }
    }

};

} // namespace xdiff::tools

#endif //XDIFF_DUAL_FLAT_ITERTOOLS_HPP