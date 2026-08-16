<div align="center">

# XDiff

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)
![Header Only](https://img.shields.io/badge/Header-Only-green.svg?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg?style=flat)
![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20HIP%20%7C%20SYCL-orange.svg?style=flat)

**A modular C++20 library for arbitrary-order automatic differentiation**

</div>

---

## Overview

Numerical differentiation is fundamental in scientific computing, optimization, and machine learning. Finding an appropriate step size for finite differences is tricky, and even with a perfect step size, the result is still an approximation.

**Automatic differentiation** solves this by computing *exact* derivatives through operator overloading, without:
- Explicitly defining derivative functions (tedious for higher-order multivariate cases)
- Using runtime symbolic differentiation

### How It Works

A simple dual number struct:
```cpp
struct Dual {
    double value;      // current scalar value
    double derivative; // current derivative value
};
```

With overloaded operators:
```cpp
Dual operator+(const Dual& a, const Dual& b) {
    return {a.value + b.value, a.derivative + b.derivative};
}

Dual operator*(const Dual& a, const Dual& b) {
    // product rule: d(f*g) = f*dg + df*g
    return {a.value * b.value, a.value * b.derivative + a.derivative * b.value};
}
```

Now `f(x) = x²` differentiates automatically:
```cpp
Dual x{3.0, 1.0}; // x = 3, dx/dx = 1
Dual f = x * x;   // f = x^2
std::cout << "f(3) = " << f.value << "\n";      // 9
std::cout << "df/dx = " << f.derivative << "\n"; // 6
```

XDiff extends this to arbitrary-order derivatives, multivariate functions, and abstract numeric types using template metaprogramming to keep the interface simple while maximizing compiler optimizations.

The key principle: all differentiation information comes from the function itself, as long as it's templated:
```cpp
template<typename T>
T f(const T& x, const T& y) {
    return sin(x - 5/(pow(x, y) - x)) * y - 4*cos(y - x);
}
```

---

## Features

- **Arbitrary-Order Derivatives** — Compute derivatives up to any order
- **Multivariate Support** — Compile-time or runtime number of variables
- **Expression Simplification** — Constant folding, identity elimination
- **GPU Support** — CUDA, HIP, and SYCL backends
- **Modular Architecture** — Clean separation across headers
- **Zero Runtime Overhead** — Optimizations happen at compile time
- **Header-Only** — Just include and use
- **Exact Derivatives** — Analytical, not numerical approximations

---

## Installation

Clone and initialize submodules:
```bash
git submodule update --init --recursive
```

The `lazy` submodule maximizes performance for runtime-sized derivatives (avoiding heap allocation bottlenecks).

**Linking via CMake:**
```cmake
add_subdirectory(path/to/xdiff)
target_link_libraries(your_target PRIVATE xdiff)
```
This gives you `<xdiff/...>` and `<lazy/...>` includes, the required C++20 standard, and the [macros](#macros) below (toggle with e.g. `-DXDIFF_FAST=ON`), all propagated automatically — no need to know xdiff's internal directory layout.

**Syntax highlighting (clangd):** configure the build from the repo root so `compile_commands.json` ends up directly in `build/`, which clangd discovers automatically:
```bash
cmake -S . -B build
```
or, if you want to compile with some optimization macros enabled:
```bash
cmake -S . -B build -DXDIFF_LAZY_NESTED_DUAL=ON -DXDIFF_FAST=ON -DXDIFF_SCALAR_OPTIMIZATIONS=ON
```

**Running the benchmark:**
```bash
cmake --build build
./build/benchmark 100000
```

Then include:
```cpp
#include <xdiff/xdiff.hpp>
```

---

## Quick Start

The main type:
```cpp
template<typename T, size_t Nvars, size_t Norder, Layout LY>
class Dual;
```

**Template parameters:**
| Parameter | Description |
|-----------|-------------|
| `T` | Underlying numeric type (`double`, `float`, etc.) |
| `Nvars` | Number of independent variables (0 = runtime) |
| `Norder` | Maximum differentiation order |
| `LY` | Layout (see below) |

**Layout:**

| Type | Description |
|------|-------------|
| `Layout::Nested` | Straightforward recursive implementation. Longer compile times, more memory. Supports runtime `Nvars` (set `Nvars = 0`). |
| `Layout::Flat` | Compact layout. Near-optimal performance for small `Nvars`/`Norder`. Compile-time only. |

---

### Basic Example

Compile with:
```bash
g++ -std=c++20 -Iinclude -Iexternal/lazy/include test.cpp -o test
```

```cpp
#include <xdiff/xdiff.hpp>

using namespace xdiff;

template<typename T>
T f(const T& x, const T& y, const T& z) {
    return x*(1 - y) + sin(y/(x+z));
}

int main() {
    using D = Dual<double, 3, 2, Layout::Nested>;

    D x(1.0, {.axis = 0}); // gradient: {1, 0, 0}
    D y(2.0, {.axis = 1}); // gradient: {0, 1, 0}
    D z(3.0, {.axis = 2}); // gradient: {0, 0, 1}

    // Constants (no derivatives):
    // D c(5.0);  or  D c(5.0, {.axis = -1});

    D res = f(x, y, z);

    // First derivatives
    std::cout << "df/dx = " << res.get_diff_wrt(0) << "\n";
    std::cout << "df/dy = " << res.get_diff_wrt(1) << "\n";
    std::cout << "df/dz = " << res.get_diff_wrt(2) << "\n";

    // Second derivatives
    std::cout << "d²f/dx² = " << res.get_diff_wrt(0, 0) << "\n";
    std::cout << "d²f/dxdy = " << res.get_diff_wrt(0, 1) << "\n";
    // ... etc

    return 0;
}
```

Results match analytical expressions to machine precision. Requesting derivatives beyond `Norder` triggers a compile-time error.

---

## Runtime Number of Variables

When `Nvars` isn't known at compile time, use `Nvars = 0` with `Layout::Nested` layout:

```cpp
Dual<double, 0, 2, Layout::Nested> x(1.0, {.axis = 0, .nvars = 3}); // stores {value=1.0, gradient={1,0,0}, and hessian={{0,0,0},{0,0,0},{0,0,0}}}
```

**Notes:**
- Uses heap allocation (`std::vector`) (avoid in performance-critical code)
- Not compatible with GPU backends
- If `.nvars` omitted, uses the static default (settable via `Dual::set_default_nvars()`)

> **Warning:** Interacting `Dual` objects must have matching `Nvars`, `Norder`, and `Layout`. Compile-time mismatches cause errors; runtime mismatches trigger assertions.

### Optimizing Runtime Performance

Large expressions with runtime `Nvars` are expensive due to heap allocations. Enable lazy evaluation:

```bash
g++ -DXDIFF_LAZY_NESTED_DUAL ...
```

For maximum performance, also use `lazy::LazyType` instead of `Dual`:
```cpp
lazy::LazyType<Dual<double, 0, 2, Layout::Nested>>
```

See the `lazy` submodule for details.

---

## Macros

| Macro | Effect |
|-------|--------|
| `XDIFF_LAZY_NESTED_DUAL` | Lazy evaluation for `Nested` with `Nvars = 0`. Avoids intermediate heap allocations. |
| `XDIFF_FAST` | Aggressive inlining for `Flat` layout. Recommended for `Norder ≤ 3`. |
| `XDIFF_LEIBNIZ_OPT` | Iterative Leibniz-rule formula for `Flat` higher-order derivatives. It will reduce compile time for large differentiation order, but may decrease performance (mainly when compiling with `g++`) |
| `XDIFF_SCALAR_OPTIMIZATIONS` | Optimized `Dual`-scalar operations for `Flat` layout. |

See useful [macros](external/lazy/README.md##Macros) for the `lazy` submodule.


For higher-order derivatives, prefer using the `clang++` compiler when compiling with `-O3` and `-DXDIFF_FAST`, as it inlines more aggressively and faster than `g++` does, when testing the `Layout::Flat` template parameter.

---

## License

MIT License
