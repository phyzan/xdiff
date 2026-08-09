/**
 * @file utils.hpp
 * @brief Utility functions, backend detection, and combinatorial helpers.
 *
 * This file provides:
 * - Backend detection (CUDA, HIP, SYCL, CPU)
 * - Device/host qualifiers for GPU compatibility
 * - Compile-time combinatorial functions (comb, multiset_coef)
 * - Array iteration utilities for derivative indexing
 * - Helper macros for constexpr expansion
 */
#ifndef COMB_HPP
#define COMB_HPP

#include <algorithm>
#include <numeric>
#include <cassert>
#include <array>

// =============================================================================
// Backend detection
// =============================================================================

/**
 * Detects the compute backend at compile time.
 * Exactly one of these will be defined:
 * - XDIFF_BACKEND_CUDA: NVIDIA CUDA
 * - XDIFF_BACKEND_HIP: AMD HIP
 * - XDIFF_BACKEND_SYCL: Intel SYCL/DPC++
 * - XDIFF_BACKEND_CPU: Standard C++ (default)
 */
#if defined(__CUDACC__)
#define XDIFF_BACKEND_CUDA 1
#elif defined(__HIPCC__)
#define XDIFF_BACKEND_HIP 1
#elif defined(SYCL_LANGUAGE_VERSION) || defined(__INTEL_LLVM_COMPILER)
#define XDIFF_BACKEND_SYCL 1
#else
#define XDIFF_BACKEND_CPU 1
#endif

// =============================================================================
// Device/Host qualifiers
// =============================================================================

/**
 * These macros provide portable device/host annotations:
 * - XDIFF_DEVICE: Code runs on GPU device only
 * - XDIFF_HOST: Code runs on CPU host only
 * - XDIFF_HOST_DEVICE: Code runs on both host and device
 * - XDIFF_FORCEINLINE: Force inline hint
 * - XDIFF_INLINE_DEVICE: Inline device function
 * - XDIFF_INLINE_HOST_DEVICE: Inline host+device function
 * - XDIFF_ALWAYS_INLINE: Aggressive inlining for lambdas
 */

#if defined(XDIFF_BACKEND_CUDA) || defined(XDIFF_BACKEND_HIP)

#define XDIFF_DEVICE __device__
#define XDIFF_HOST __host__
#define XDIFF_HOST_DEVICE __host__ __device__
#define XDIFF_FORCEINLINE __forceinline__
#define XDIFF_INLINE_DEVICE __forceinline__ __device__
#define XDIFF_INLINE_HOST_DEVICE __forceinline__ __host__ __device__

#elif defined(XDIFF_BACKEND_SYCL)

#define XDIFF_DEVICE
#define XDIFF_HOST
#define XDIFF_HOST_DEVICE
#define XDIFF_FORCEINLINE inline
#define XDIFF_INLINE_DEVICE inline
#define XDIFF_INLINE_HOST_DEVICE inline

#else // CPU

#define XDIFF_DEVICE
#define XDIFF_HOST
#define XDIFF_HOST_DEVICE

#if defined(_MSC_VER)
#define XDIFF_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define XDIFF_FORCEINLINE __attribute__((always_inline)) inline
#else
#define XDIFF_FORCEINLINE inline
#endif

#define XDIFF_INLINE_DEVICE XDIFF_FORCEINLINE
#define XDIFF_INLINE_HOST_DEVICE XDIFF_FORCEINLINE

#endif

// For lambda attributes (flatten only on GCC/Clang, not CUDA/HIP)
#if defined(XDIFF_BACKEND_CUDA) || defined(XDIFF_BACKEND_HIP)
#define XDIFF_ALWAYS_INLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define XDIFF_ALWAYS_INLINE __attribute__((always_inline, flatten))
#else
#define XDIFF_ALWAYS_INLINE
#endif

#define XDIFF_LAMBDA_INLINE XDIFF_ALWAYS_INLINE



// =============================================================================
// Compile-time expansion macros
// =============================================================================

/// @brief Token concatenation helper
#define XDIFF_CONCAT(a,b) XDIFF_CONCAT_IMPL(a,b)
#define XDIFF_CONCAT_IMPL(a,b) a##b

/// @brief Integer sequence type alias
#define XDIFF_INTS(IntType, I) std::integer_sequence<IntType, I...>

/// @brief Creates an integer sequence [0, N)
#define XDIFF_MAKE_INTS(IntType, N) std::make_integer_sequence<IntType, N>{}

/**
 * @brief Expands a lambda over an integer sequence.
 *
 * Creates a lambda that takes an integer sequence parameter pack and
 * executes the provided code with each index available as a compile-time constant.
 *
 * @param IntType The integer type (usually size_t)
 * @param N Number of iterations
 * @param I Parameter pack name for indices
 * @param ... Code to execute (can reference I as a pack)
 */
#define XDIFF_EXPAND(IntType, N, I, ...) [&] XDIFF_DEVICE <IntType... I>(XDIFF_INTS(IntType, I)) \
    XDIFF_ALWAYS_INLINE { \
    __VA_ARGS__ \
}(XDIFF_MAKE_INTS(IntType, N))

/**
 * @brief Compile-time for loop with index as template parameter.
 *
 * Unrolls a loop at compile time, making each index available as a
 * compile-time constant within the loop body.
 */
#define XDIFF_FOR_LOOP_IMPL(IntType, I, N, IDUMMY, ...) \
[&] XDIFF_DEVICE <IntType... IDUMMY>(XDIFF_INTS(IntType, IDUMMY)) XDIFF_ALWAYS_INLINE { \
    ([&] XDIFF_DEVICE <IntType I>() XDIFF_ALWAYS_INLINE { __VA_ARGS__ }.template operator()<IDUMMY>(), ...); \
}(XDIFF_MAKE_INTS(IntType, N))

#define XDIFF_FOR_LOOP(IntType, I, N, ...) \
    XDIFF_FOR_LOOP_IMPL(IntType, I, N, XDIFF_CONCAT(IDUMMY,__COUNTER__), __VA_ARGS__)

/**
 * @brief CRTP helper to cast this pointer to derived type.
 *
 * Used in CRTP base classes to access derived class members.
 * Preserves const-qualification of the this pointer.
 */
#define XDIFF_THIS static_cast<std::conditional_t<std::is_void_v<Derived>, \
    std::remove_reference_t<decltype(*this)>, \
    utils::copy_const_t<std::remove_reference_t<decltype(*this)>, Derived>>*>(this)

namespace utils{

// =============================================================================
// Type utilities
// =============================================================================

/// @brief Copies const qualification from From to To
template<typename From, typename To>
using copy_const_t = std::conditional_t<std::is_const_v<From>, const To, To>;

// =============================================================================
// Array utilities
// =============================================================================

/**
 * @brief GPU-compatible array copy.
 *
 * Avoids std::copy/memcpy for GPU code compatibility.
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Number of elements to copy
 */
template<typename T>
XDIFF_INLINE_HOST_DEVICE constexpr void copy_array(T* dest, const T* src, size_t size){
    for (size_t i = 0; i < size; ++i) {
        dest[i] = src[i];
    }
}

/**
 * @brief Counts occurrences of a variable index within a list.
 *
 * Used to convert variable index lists to per-variable derivative counts.
 *
 * @param x The variable index to count
 * @param y The list of variable indices to search
 * @return The number of times x appears in y
 *
 * @example
 *     var_count(0, 0, 1, 0, 2);  // Returns 2 (index 0 appears twice)
 */
template<typename A, typename... T>
XDIFF_INLINE_HOST_DEVICE
constexpr size_t var_count(A x, T... y){
    return XDIFF_EXPAND(size_t, sizeof...(y), I,
        return ((size_t(x)==size_t(y))+...+0UL);
    );
}

/**
 * @brief Extracts the I-th element from a parameter pack.
 *
 * @tparam I The index to extract (0-based)
 * @param x0 First element
 * @param x Remaining elements
 * @return The I-th element, perfectly forwarded
 */
template<std::size_t I, typename FirstType, typename... ArgType>
XDIFF_INLINE_DEVICE constexpr decltype(auto) pack_elem(FirstType&& x0, ArgType&&... x) {
    if constexpr (I == 0) {
        return std::forward<FirstType>(x0);
    } else {
        static_assert(sizeof...(x) > 0, "Index out of bounds");
        return pack_elem<I - 1>(std::forward<ArgType>(x)...);
    }
}

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





// =============================================================================
// Multi-dimensional iteration utilities
// =============================================================================

/**
 * @brief CRTP base class for multi-dimensional index iterators.
 *
 * Provides infrastructure for iterating over ND-dimensional index spaces.
 * Used for iterating over derivative indices in graded colexicographic order.
 *
 * @tparam Derived The derived iterator type
 * @tparam ND Number of dimensions
 */
template<typename Derived, size_t ND>
class IndexIterator{

    static_assert(ND>0, "ND>0 in Index Iterator");

public:

    /**
     * @brief Advances indices and returns whether iteration should continue.
     *
     * @param idx Reference to index variables
     * @return true if more iterations remain, false if done
     */
    template<std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE bool constexpr iterating(IntType&... idx) const{
        static_assert(sizeof...(idx)==ND, "Invalid number of indices");
        return iterating_impl(idx...);
    }

    /// @brief Iterates over all indices, calling f for each combination.
    template<typename Callable>
    XDIFF_INLINE_HOST_DEVICE void constexpr iterate(Callable&& f) const{
        return XDIFF_THIS->iterate(std::forward<Callable>(f));
    }

    /// @brief Implementation of iterating() (override in derived).
    template<std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE bool constexpr iterating_impl(IntType&... idx) const{
        return XDIFF_THIS->iterating_impl(idx...);
    }

    /// @brief Returns the number of dimensions.
    [[nodiscard]] inline size_t ndims() const{
        return ND;
    }

protected:

    IndexIterator() = default;

};

/**
 * @brief Base class for N-dimensional iterators.
 *
 * Provides common iteration logic for both static and dynamic ND iteration.
 *
 * @tparam Derived The derived iterator type
 * @tparam ND Number of dimensions
 */
template<typename Derived, size_t ND>
class BaseNdIterator : public IndexIterator<Derived, ND>{

public:

    using IdxHolder = std::array<size_t, ND>;

    /**
     * @brief Recursively iterates over all index combinations.
     *
     * @tparam I Current dimension being iterated
     * @param idx Current index values
     * @param f Callable to invoke for each combination
     * @param limit Upper bounds for each dimension
     */
    template<size_t I, typename Callable>
    XDIFF_INLINE_HOST_DEVICE static void constexpr iterate_impl(IdxHolder& idx, Callable&& f, const IdxHolder& limit){
        if constexpr (I < ND) {
            idx[I] = 0;
            do {
                iterate_impl<I+1>(idx, std::forward<Callable>(f), limit);
                idx[I]++;
            }while (idx[I] < limit[I]);
        }
        else{
            XDIFF_EXPAND(size_t, ND, J,
                f(idx[J]...);
            );
        }
    }

protected:

    /**
     * @brief Increments indices in colexicographic order.
     *
     * Carries from dimension I to I-1 when overflow occurs.
     *
     * @return true if increment succeeded, false if iteration complete
     */
    template<size_t I, size_t... Is, std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE static constexpr bool increment(std::index_sequence<Is...> /**/, const IdxHolder& limit, IntType&... idx){
        if constexpr (I==0) {
            return ((Is==I && (++idx<limit[Is] ? true : (idx=0, false)))||...);
        }
        else{
            return ((Is==I && (++idx<limit[Is] ? true : (idx=0, increment<I-1>(std::index_sequence<Is...>{}, limit, idx...))))||...);
        }
    }

};


/**
 * @brief N-dimensional iterator with compile-time dimensions.
 *
 * Iterates over [0, Dim[0]) x [0, Dim[1]) x ... x [0, Dim[N-1]).
 *
 * @tparam Dim The size of each dimension (compile-time)
 */
template<size_t... Dim>
class StaticNDIterator : public BaseNdIterator<StaticNDIterator<Dim...>, sizeof...(Dim)>{

    static constexpr size_t ND = sizeof...(Dim);
    using Base = BaseNdIterator<StaticNDIterator<Dim...>, ND>;

    static constexpr typename Base::IdxHolder SHAPE = {Dim...};

public:

    template<std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE bool iterating_impl(IntType&... idx) const{
        return Base::template increment<ND-1>(std::make_index_sequence<ND>{}, SHAPE, idx...);
    }

    template<typename Callable>
    XDIFF_INLINE_HOST_DEVICE void constexpr iterate(Callable&& f) const{
        typename Base::IdxHolder idx{};
        return Base::template iterate_impl<0>(idx, std::forward<Callable>(f), SHAPE);
    }

};


/**
 * @brief N-dimensional iterator with runtime dimensions.
 *
 * Same as StaticNDIterator but dimensions are specified at construction.
 *
 * @tparam ND Number of dimensions (compile-time)
 */
template<size_t ND>
class DynamicNDIterator : public BaseNdIterator<DynamicNDIterator<ND>, ND>{

    using Base = BaseNdIterator<DynamicNDIterator<ND>, ND>;

public:

    /// @brief Constructs with the given shape.
    template<std::integral... IntType>
    constexpr DynamicNDIterator(IntType... shape) : shape_{shape...}{
        static_assert(sizeof...(shape)==ND, "Invalid shape size");
    }

    template<std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE bool iterating_impl(IntType&... idx) const{
        return Base::template increment<ND-1>(std::make_index_sequence<ND>{}, shape_, idx...);
    }

    template<typename Callable>
    XDIFF_INLINE_HOST_DEVICE void constexpr iterate(Callable&& f) const{
        typename Base::IdxHolder idx{};
        return Base::template iterate_impl<0>(idx, std::forward<Callable>(f), shape_);
    }

private:
    typename Base::IdxHolder shape_;

};




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

} // namespace utils


#endif // COMB_HPP