#ifndef XDIFF_BACKEND_HPP
#define XDIFF_BACKEND_HPP

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


#endif // XDIFF_BACKEND_HPP