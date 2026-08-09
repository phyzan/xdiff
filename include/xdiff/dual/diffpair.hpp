#ifndef XDIFF_DIFFPAIR_HPP
#define XDIFF_DIFFPAIR_HPP


#include "dual.hpp"


namespace xdiff{

template<typename F, typename DF>
struct DiffPair{

    F value;   ///< The function value
    DF grad;   ///< The derivative value

    /// @brief Returns true if this derivative is trivially zero (compile-time optimization)
    XDIFF_INLINE_HOST_DEVICE
    static constexpr bool isTrivial() {
        return xdiff::traits::isAnyZero<DF>;
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


} // namespace xdiff


#endif // XDIFF_DIFFPAIR_HPP