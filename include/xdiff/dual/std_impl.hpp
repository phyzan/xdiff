#ifndef XDIFF_DUAL_STD_IMPL_HPP
#define XDIFF_DUAL_STD_IMPL_HPP

#include "dual_impl.hpp" // IWYU pragma: keep


namespace xdiff{

struct DynamicState{

    template<typename U>
    using grad_type = std::vector<U>;

    template<typename F, typename U, typename... Rest>
    static void zip(F&& f, std::vector<U>& out, Rest&&... v){
        if constexpr (sizeof...(Rest) > 0){
            out.resize(std::max({out.size(), v.size()...}));
        }
        for (size_t i = 0; i < out.size(); i++){
            f(out[i], at(v, i)...);
        }
    }

    template<typename U, int N, std::integral Int, std::integral... Rest>
    static const U& get_diff(const xdiff::Dual<U, DynamicState, N>& dual, Int i, Rest... rest) {
        static_assert(sizeof...(Rest) <= N-1);
        decltype(auto) diff_i = DynamicState::at(dual.grad(), i);
        if constexpr (sizeof...(Rest) == 0 && N == 1){
            return diff_i;
        } else if constexpr (sizeof...(Rest) == 0){
            return diff_i.value();
        } else {
            return get_diff(diff_i, rest...);
        }
    }

private:
    
    template<typename U>
    static const U& at(const std::vector<U>& v, size_t i){
        static const U zero{};
        return i < v.size() ? v[i] : zero;
    }

}; // struct DynamicState





template<size_t N>
struct StaticState{

    template<typename U>
    using grad_type = std::array<U, N>;

    template<typename F, typename U, typename... Rest>
    static void zip(F&& f, std::array<U, N>& out, Rest&&... in){
        for (size_t i = 0; i < N; i++){
            f(out[i], in[i]...);
        }
    }

    template<int M, typename U, std::integral Int, std::integral... Rest>
    static const U& get_diff(const xdiff::Dual<U, StaticState, M>& dual, Int i, Rest... rest) {
        static_assert(sizeof...(Rest) <= M-1);
        decltype(auto) diff_i = dual.grad()[i];
        if constexpr (sizeof...(Rest) == 0 && M == 1){
            return diff_i;
        } else if constexpr (sizeof...(Rest) == 0){
            return diff_i.value();
        } else {
            return get_diff(diff_i, rest...);
        }
    }

}; // struct StaticState



template<typename T, int Order>
class DynamicDual : public xdiff::Dual<T, DynamicState, Order>{
    using Base = xdiff::Dual<T, DynamicState, Order>;
public:
    using Base::Base;

    DynamicDual(const Base& other) : Base(other) {}

    DynamicDual(const xdiff::Seed<T, DynamicState, Order>& other) : Base(other) {}

    template<std::integral... Int>
    const T& diff(Int... i) const {
        return DynamicState::get_diff(*this, i...);
    }
};


template<typename T, size_t N, int Order>
class StaticDual : public xdiff::Dual<T, StaticState<N>, Order>{
    using Base = xdiff::Dual<T, StaticState<N>, Order>;
public:
    using Base::Base;

    StaticDual(const Base& other) : Base(other) {}

    StaticDual(const xdiff::Seed<T, StaticState<N>, Order>& other) : Base(other) {}

    template<std::integral... Int>
    const T& diff(Int... i) const {
        return StaticState<N>::get_diff(*this, i...);
    }
};

}; // namespace xdiff

#endif // XDIFF_DUAL_STD_IMPL_HPP