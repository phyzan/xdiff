#ifndef XDIFF_TOOLS_HPP
#define XDIFF_TOOLS_HPP


#include <type_traits>
#include <cstdlib>
#include <cmath>
#include <array>
#include <vector>
#include <cassert>

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
// =============================================================================
// =============================================================================

// ============================= ITERATION MACROS ==============================

#define XDIFF_FOR_LOOP(I, N, ...) \
xdiff::tools::ForEach<N>([&]<size_t I>() XDIFF_LAMBDA_INLINE { \
    __VA_ARGS__ \
})

#define XDIFF_EXPAND(N, I, ...) \
xdiff::tools::Expand<N>([&]<size_t... I>() XDIFF_LAMBDA_INLINE { \
    __VA_ARGS__ \
})

#define XDIFF_THIS static_cast<std::conditional_t<std::is_void_v<Derived>, \
    std::remove_reference_t<decltype(*this)>, \
    ::xdiff::detail::copy_const_t<std::remove_reference_t<decltype(*this)>, Derived>>*>(this)


namespace xdiff::detail{

template<typename From, typename To>
using copy_const_t = std::conditional_t<std::is_const_v<From>, const To, To>;


template<size_t I, std::size_t N, typename F, typename... Args>
XDIFF_INLINE_HOST_DEVICE constexpr void ForEach_impl(F& f, Args&... args){
    if constexpr (I < N) {
        f.template operator()<I>(args...);
        ForEach_impl<I + 1, N>(f, args...);
    }
}


template<typename F, size_t... I>
XDIFF_INLINE_HOST_DEVICE constexpr decltype(auto) Expand_impl(F&& f, std::index_sequence<I...> /**/){
    return f.template operator()<I...>();
}


template<typename F, std::integral First, std::integral... Int>
XDIFF_FORCEINLINE constexpr void for_each_impl(F&& f, First Ni, Int... Nj){
    if constexpr (sizeof...(Int) == 0){
        for (size_t i = 0; i < Ni; ++i) {
            std::forward<F>(f)(i);
        }
    } else {
        for (size_t i = 0; i < Ni; ++i) {
            for_each_impl([&](auto... Nk) XDIFF_LAMBDA_INLINE {
                f(i, Nk...);
            }, Nj...);
        }
    }
}


template<typename F, size_t Dim, size_t... I>
XDIFF_FORCEINLINE constexpr void make_for_each(F&& f, std::array<size_t, Dim> shape, std::index_sequence<I...> /**/){
    for_each_impl(std::forward<F>(f), shape[I]...);
}


} // namespace xdiff::detail

namespace xdiff::tools{



template<size_t N, typename F, typename... Args>
XDIFF_INLINE_HOST_DEVICE constexpr void ForEach(F&& f, Args&&... args){
    xdiff::detail::ForEach_impl<0, N>(f, args...);
}


template<size_t N, typename F>
XDIFF_INLINE_HOST_DEVICE constexpr decltype(auto) Expand(F&& f){
    return xdiff::detail::Expand_impl(std::forward<F>(f), std::make_index_sequence<N>{});
}


template<typename F, size_t Dim>
XDIFF_FORCEINLINE constexpr void for_each(F&& f, std::array<size_t, Dim> shape){
    xdiff::detail::make_for_each(std::forward<F>(f), shape, std::make_index_sequence<Dim>{});
}

template<typename F, std::integral... Int>
XDIFF_FORCEINLINE constexpr void for_each(F&& f, Int... shape){
    xdiff::detail::for_each_impl(std::forward<F>(f), shape...);
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
    return XDIFF_EXPAND(sizeof...(y), I,
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


template<typename Cls, typename Derived>
using GetDerived = std::conditional_t<(std::is_same_v<Derived, void>), Cls, Derived>;

} // namespace xdiff::tools



namespace xdiff{

template<typename T, size_t N = 0>
class Vector;


template<typename T, size_t N>
class Vector {

public:
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector() = default;
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(const Vector& arr) = default;
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(Vector&& arr) = default;
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector& operator=(const Vector& arr) = default;
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector& operator=(Vector&& arr) = default;
    ~Vector() = default;

    // Custom constructors
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(size_t n) {assert(n == N && "Size mismatch in Vector constructor");}
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(size_t n, const T& value) : Vector(private_tag{}, n, std::make_index_sequence<N>{}, value) { assert(n == N && "Size mismatch in Vector constructor"); }
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(size_t n, T& value) : Vector(n, static_cast<const T&>(value)) {}
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(size_t n, T&& value) : Vector(n, static_cast<const T&>(value)) {}

    template<typename... U>
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(size_t n, const U&... constructor_args) : Vector(private_tag{}, n, std::make_index_sequence<N>{}, constructor_args...) {}

    template<typename Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr T& operator[](Int i){
        return data_[i];
    }

    template<typename Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& operator[](Int i) const{
        return data_[i];
    }


    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t size() const {
        return data_.size();
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr const T* data() const {
        return data_.data();
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr T* data() {
        return data_.data();
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr auto begin() const {
        return data_.begin();
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr auto end() const {
        return data_.end();
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr auto begin() {
        return data_.begin();
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr auto end() {
        return data_.end();
    }

private:

    template<typename A, typename... U>
    friend Vector<A> make_vector(U&&... items);

    struct private_tag{};

    template<typename... U, size_t... I>
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(private_tag, size_t n, std::index_sequence<I...>, const U&... values) : data_{ { ((void)I, T(values...))... } } {
        assert(n == N && "Size mismatch in Vector constructor");
    }

    template<typename... U>
    XDIFF_INLINE_HOST_DEVICE
    constexpr Vector(private_tag, U&&... items) : data_{ std::forward<U>(items)... } {
        static_assert(sizeof...(U) == N, "Size mismatch in Vector constructor");
    }

    std::array<T, N> data_;
};



template<typename T>
class Vector<T, 0> {

public:

    Vector() = default;
    Vector(const Vector& arr) = default;
    Vector(Vector&& arr) = default;
    Vector& operator=(const Vector& arr) = default;
    Vector& operator=(Vector&& arr) = default;
    ~Vector() = default;

    // Custom constructors
    Vector(size_t n) : data_(n) {}

    Vector(size_t n, const T& value) : data_(n, value) {}
    Vector(size_t n, T&& value) : data_(n, static_cast<const T&>(value)) {}
    Vector(size_t n, T& value) : data_(n, static_cast<const T&>(value)) {}

    template<typename... U>
    Vector(size_t n, const U&... constructor_args) {
        data_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            data_.emplace_back(constructor_args...);
        }
    }

    template<std::integral Int>
    inline constexpr T& operator[](Int i){
        return data_[i];
    }

    template<std::integral Int>
    inline constexpr const T& operator[](Int i) const{
        return data_[i];
    }

    inline size_t size() const {
        return data_.size();
    }

    inline const T* data() const {
        return data_.data();
    }

    inline T* data() {
        return data_.data();
    }

    inline auto begin() const {
        return data_.begin();
    }

    inline auto end() const {
        return data_.end();
    }

    inline auto begin() {
        return data_.begin();
    }

    inline auto end() {
        return data_.end();
    }

    inline void resize(size_t n) {
        data_.resize(n);
    }

    inline void reserve(size_t n) {
        data_.reserve(n);
    }

    inline void push_back(const T& value){
        data_.push_back(value);
    }

    inline void push_back(T&& value){
        data_.push_back(std::move(value));
    }

    template<typename... Args>
    inline void emplace_back(Args&&... args){
        data_.emplace_back(std::forward<Args>(args)...);
    }

private:

    template<typename A, typename... U>
    friend Vector<A> make_vector(U&&... items);

    struct private_tag{};

    template<typename... U>
    XDIFF_INLINE_HOST_DEVICE
    Vector(private_tag, U&&... items) {
        data_.reserve(sizeof...(U));
        (data_.emplace_back(std::forward<U>(items)), ...);
    }

    std::vector<T> data_;

};


template<typename T, typename... U>
inline Vector<T> make_vector(U&&... items){
    return Vector<T>(typename Vector<T>::private_tag{}, std::forward<U>(items)...);
}

} // namespace xdiff



#endif // XDIFF_TOOLS_HPP