#ifndef ITERATORS_HPP
#define ITERATORS_HPP

#include <cstdlib>
#include <array>
#include <cassert>
#include <numeric>

// =============================================================================
// Backend detection and platform macros
// =============================================================================

// Detect backend
#if defined(__CUDACC__)
#define AUTODIFF_BACKEND_CUDA 1
#elif defined(__HIPCC__)
#define AUTODIFF_BACKEND_HIP 1
#elif defined(SYCL_LANGUAGE_VERSION) || defined(__INTEL_LLVM_COMPILER)
#define AUTODIFF_BACKEND_SYCL 1
#else
#define AUTODIFF_BACKEND_CPU 1
#endif

// =============================================================================
// Device/Host qualifiers
// =============================================================================

#if defined(AUTODIFF_BACKEND_CUDA) || defined(AUTODIFF_BACKEND_HIP)

#define AUTODIFF_DEVICE __device__
#define AUTODIFF_HOST __host__
#define AUTODIFF_HOST_DEVICE __host__ __device__
#define AUTODIFF_FORCEINLINE __forceinline__
#define AUTODIFF_INLINE_DEVICE __forceinline__ __device__
#define AUTODIFF_INLINE_HOST_DEVICE __forceinline__ __host__ __device__

#elif defined(AUTODIFF_BACKEND_SYCL)

#define AUTODIFF_DEVICE
#define AUTODIFF_HOST
#define AUTODIFF_HOST_DEVICE
#define AUTODIFF_FORCEINLINE inline
#define AUTODIFF_INLINE_DEVICE inline
#define AUTODIFF_INLINE_HOST_DEVICE inline

#else // CPU

#define AUTODIFF_DEVICE
#define AUTODIFF_HOST
#define AUTODIFF_HOST_DEVICE

#if defined(_MSC_VER)
#define AUTODIFF_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define AUTODIFF_FORCEINLINE __attribute__((always_inline)) inline
#else
#define AUTODIFF_FORCEINLINE inline
#endif

#define AUTODIFF_INLINE_DEVICE AUTODIFF_FORCEINLINE
#define AUTODIFF_INLINE_HOST_DEVICE AUTODIFF_FORCEINLINE

#endif

// For lambda attributes (flatten only on GCC/Clang, not CUDA/HIP)
#if defined(AUTODIFF_BACKEND_CUDA) || defined(AUTODIFF_BACKEND_HIP)
#define AUTODIFF_ALWAYS_INLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define AUTODIFF_ALWAYS_INLINE __attribute__((always_inline, flatten))
#else
#define AUTODIFF_ALWAYS_INLINE
#endif

// =============================================================================
// Utility macros
// =============================================================================

#define AD_THIS static_cast<std::conditional_t<std::is_void_v<Derived>, \
    std::remove_reference_t<decltype(*this)>, \
    utils::copy_const_t<std::remove_reference_t<decltype(*this)>, Derived>>*>(this)
#define AD_UNIQUE_NAME(base) AD_CONCAT(base, __COUNTER__)
#define AD_CONCAT(a,b) AD_CONCAT_IMPL(a,b)
#define AD_CONCAT_IMPL(a,b) a##b

#define AD_INTS(IntType, I) std::integer_sequence<IntType, I...>

#define AD_MAKE_INTS(IntType, N) std::make_integer_sequence<IntType, N>{}

#define AD_EXPAND(IntType, N, I, ...) [&] AUTODIFF_DEVICE <IntType... I>(AD_INTS(IntType, I)) \
    AUTODIFF_ALWAYS_INLINE { \
    __VA_ARGS__ \
}(AD_MAKE_INTS(IntType, N))

#define AD_FOR_LOOP_IMPL(IntType, I, N, IDUMMY, ...) \
[&] AUTODIFF_DEVICE <IntType... IDUMMY>(AD_INTS(IntType, IDUMMY)) AUTODIFF_ALWAYS_INLINE { \
    ([&] AUTODIFF_DEVICE <IntType I>() AUTODIFF_ALWAYS_INLINE { __VA_ARGS__ }.template operator()<IDUMMY>(), ...); \
}(AD_MAKE_INTS(IntType, N))

#define AD_FOR_LOOP(IntType, I, N, ...) \
    AD_FOR_LOOP_IMPL(IntType, I, N, AD_CONCAT(IDUMMY,__COUNTER__), __VA_ARGS__)

#define AD_INLINE AUTODIFF_INLINE_HOST_DEVICE
#define AD_LAMBDA_INLINE AUTODIFF_ALWAYS_INLINE


namespace autodiff {


namespace utils {

template<typename From, typename To>
using copy_const_t = std::conditional_t<std::is_const_v<From>, const To, To>;

template<std::size_t I, typename FirstType, typename... ArgType>
AD_INLINE constexpr decltype(auto) pack_elem(FirstType&& x0, ArgType&&... x) {
    if constexpr (I == 0) {
        return std::forward<FirstType>(x0);
    } else {
        static_assert(sizeof...(x) > 0, "Index out of bounds");
        return pack_elem<I - 1>(std::forward<ArgType>(x)...);
    }
}

AUTODIFF_HOST_DEVICE constexpr size_t comb(size_t n, size_t k) {
    assert(n >= k);

    k = std::min(k, n - k);

    size_t res = 1;
    for (size_t i = 1; i <= k; ++i) {
        size_t num = n - i + 1;
        size_t den = i;

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

template<typename T>
AD_INLINE void constexpr copy_array(T* dest, const T* src, size_t size){
    // avoid std::copy or memcpy for gpu code compatibility
    for (size_t i = 0; i < size; ++i) {
        dest[i] = src[i];
    }
}

template<typename Derived, size_t ND>
class IndexIterator{

    static_assert(ND>0, "ND>0 in Index Iterator");
public:

    template<std::integral... IntType>
    AD_INLINE bool constexpr iterating(IntType&... idx) const{
        static_assert(sizeof...(idx)==ND, "Invalid number of indices");
        return iterating_impl(idx...);
    }

    template<typename Callable>
    AD_INLINE void constexpr iterate(Callable&& f) const{
        //static override
        return AD_THIS->iterate(std::forward<Callable>(f));
    }

    template<std::integral... IntType>
    AD_INLINE bool iterating_impl(IntType&... idx) const{
        //static override
        return AD_THIS->iterating_impl(idx...);
    }

    inline size_t ndims() const{
        return ND;
    }

protected:

    IndexIterator() = default;

};

template<typename Derived, size_t ND>
class BaseNdIterator : public IndexIterator<Derived, ND>{


public:

    using IdxHolder = std::array<size_t, ND>;

    template<size_t I, typename Callable>
    AD_INLINE static void constexpr iterate_impl(IdxHolder& idx, Callable&& f, const IdxHolder& limit){
        if constexpr (I < ND) {
            idx[I] = 0;
            do {
                iterate_impl<I+1>(idx, std::forward<Callable>(f), limit);
                idx[I]++;
            }while (idx[I] < limit[I]);
        }
        else{
            AD_EXPAND(size_t, ND, J, 
                f(idx[J]...);
            );
        }
    }

protected:

    template<size_t I, size_t... Is, std::integral... IntType>
    AD_INLINE static constexpr bool increment(std::index_sequence<Is...>, const IdxHolder& limit, IntType&... idx){
        if constexpr (I==0) {
            return ((Is==I && (++idx<limit[Is] ? true : (idx=0, false)))||...);
        }
        else{
            return ((Is==I && (++idx<limit[Is] ? true : (idx=0, increment<I-1>(std::index_sequence<Is...>{}, limit, idx...))))||...);
        }
    }


};



template<size_t... Dim>
class StaticNDIterator : public BaseNdIterator<StaticNDIterator<Dim...>, sizeof...(Dim)>{

    static constexpr size_t ND = sizeof...(Dim);
    using Base = BaseNdIterator<StaticNDIterator<Dim...>, ND>;
    

    static constexpr typename Base::IdxHolder SHAPE = {Dim...};

public:

    template<std::integral... IntType>
    AD_INLINE bool iterating_impl(IntType&... idx) const{
        return Base::template increment<ND-1>(std::make_index_sequence<ND>{}, SHAPE, idx...);
    }

    template<typename Callable>
    AD_INLINE void constexpr iterate(Callable&& f) const{
        typename Base::IdxHolder idx{};
        return Base::template iterate_impl<0>(idx, std::forward<Callable>(f), SHAPE);
    }

};


template<size_t ND>
class DynamicNDIterator : public BaseNdIterator<DynamicNDIterator<ND>, ND>{

    using Base = BaseNdIterator<DynamicNDIterator<ND>, ND>;

public:

    template<std::integral... IntType>
    constexpr DynamicNDIterator(IntType... shape) : _shape{shape...}{
        static_assert(sizeof...(shape)==ND, "Invalid shape size");
    }

    template<std::integral... IntType>
    AD_INLINE bool iterating_impl(IntType&... idx) const{
        return Base::template increment<ND-1>(std::make_index_sequence<ND>{}, _shape, idx...);
    }

    template<typename Callable>
    AD_INLINE void constexpr iterate(Callable&& f) const{
        typename Base::IdxHolder idx{};
        return Base::template iterate_impl<0>(idx, std::forward<Callable>(f), _shape);
    }

    typename Base::IdxHolder _shape;

};


template<size_t Slots, size_t Rank, bool CountSlots>
class MultiSetIterator{

public:

    using SetType = std::array<size_t, Slots>;
    using CounterType = std::array<size_t, Rank>;

    template<typename Callable>
    AD_INLINE static constexpr void apply_iter_on(Callable&& f){
        SetType set{};
        CounterType counter{};
        doit<0>(set, counter,std::forward<Callable>(f));
    }

    template<size_t slot, typename Callable>
    AD_INLINE static void constexpr doit(SetType& set, CounterType& counter, Callable&& f){
        if constexpr (slot < Slots) {
            size_t& var = set[slot];
            if constexpr (slot == 0) {
                var = 0;
            }
            else{
                var = set[slot-1];
            }
            if constexpr (CountSlots) {
                counter[var]++;
            }

            do {
                doit<slot+1>(set, counter, std::forward<Callable>(f));
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

AUTODIFF_INLINE_HOST_DEVICE constexpr size_t multiset_coef(size_t n, size_t k){
    //number of weak compositions
    //let A be a vector of n positive integers. 
    //This function returns the number of vectors
    // (combinations of (A_1, ... ,A_n)) that sum exactly to k
    return comb(n+k-1, k);
}

} // namespace utils

} // namespace autodiff


#endif