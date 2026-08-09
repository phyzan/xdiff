#ifndef XDIFF_DUAL_BASE_HPP
#define XDIFF_DUAL_BASE_HPP


#include "diffpair.hpp" // IWYU pragma: keep


namespace xdiff{

template<typename Derived, typename T, size_t NVARS, Layout LY>
struct DualBase : xdiff::detail::DualIdentifier<Derived>{

    using value_type = T;

    XDIFF_INLINE_HOST_DEVICE
    const T& value() const {
        return XDIFF_THIS->value();
    }

    template<xdiff::traits::isAxis... Int>
    XDIFF_INLINE_HOST_DEVICE
    const T& get_diff_wrt(Int... x) const{
        return XDIFF_THIS->get_diff_wrt(x...);
    }

};


} // namespace xdiff
#endif // XDIFF_DUAL_BASE_HPP