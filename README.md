# AutoDiff

<div align="center">

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)
![Header Only](https://img.shields.io/badge/Header-Only-green.svg?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg?style=flat)

**A high-performance, header-only C++20 library for arbitrary-order automatic differentiation**

</div>

---

## ✨ Features

- **🎯 Arbitrary-Order Derivatives** — Compute derivatives up to any order with compile-time configuration
- **📐 Multivariate Support** — Handle functions of any number of independent variables
- **⚡ Zero Runtime Overhead** — Template metaprogramming ensures computations are optimized at compile time
- **🔧 Header-Only** — Just include and use, no linking required
- **🧮 Exact Derivatives** — Computes analytical derivatives, not numerical approximations
- **🔄 Operator Overloading** — Natural mathematical syntax with automatic derivative propagation

---

## 🚀 Quick Start

All types are in the `autodiff` namespace. You can either use the namespace prefix or `using namespace autodiff;`.

### Basic Example

```cpp
#include <autodiff/autodiff.hpp>
#include <iostream>

using namespace autodiff;

int main() {
    // Define variables (compile-time indices)
    Variable<0> x;
    Variable<1> y;

    // Create AutoDiff objects: <Type, MaxOrder, NumVariables>
    // Order 2 = compute up to second derivatives
    // 2 variables = x and y
    AutoDiff<double, 2, 2> a(3.0, x);  // a = 3, da/dx = 1, da/dy = 0
    AutoDiff<double, 2, 2> b(2.0, y);  // b = 2, db/dx = 0, db/dy = 1

    // Compute f = a * b
    auto f = a * b;

    // Access results
    std::cout << "f(3,2) = " << f.value() << "\n";           // 6
    std::cout << "df/dx  = " << f.diff_value(x) << "\n";     // 2
    std::cout << "df/dy  = " << f.diff_value(y) << "\n";     // 3
    std::cout << "d²f/dxdy = " << f.diff_value(x, y) << "\n"; // 1

    return 0;
}
```

### Higher-Order Derivatives

```cpp
#include <autodiff/autodiff.hpp>

using namespace autodiff;

// Function: f(x,y,z) = x³y²z + xy³
template<typename T>
T compute(const T& x, const T& y, const T& z) {
    return x*x*x * y*y * z + x * y*y*y;
}

int main() {
    Variable<0> X;
    Variable<1> Y;
    Variable<2> Z;

    // Track up to 3rd order derivatives for 3 variables
    using F = AutoDiff<double, 3, 3>;

    F x(2.0, X);
    F y(3.0, Y);
    F z(1.0, Z);

    auto f = compute(x, y, z);

    // First derivatives
    f.diff_value(X);        // df/dx
    f.diff_value(Y);        // df/dy
    f.diff_value(Z);        // df/dz

    // Second derivatives
    f.diff_value(X, X);     // d²f/dx²
    f.diff_value(X, Y);     // d²f/dxdy

    // Third derivatives
    f.diff_value(X, Y, Z);  // d³f/dxdydz
    f.diff_value(X, X, Y);  // d³f/dx²dy

    return 0;
}
```

### Using Mathematical Functions

```cpp
#include <autodiff/autodiff.hpp>

using namespace autodiff;

int main() {
    Variable<0> x;

    AutoDiff<double, 2, 1> a(1.5, x);

    // Supported functions
    auto f1 = exp(a);       // Exponential
    auto f2 = log(a);       // Natural logarithm
    auto f3 = pow(a, 2.0);  // Power (variable base)
    auto f4 = pow(2.0, a);  // Power (variable exponent)
    auto f5 = pow(a, a);    // Power (both variable)

    // Compound expressions
    auto g = exp(a * a) / (1.0 + a);

    return 0;
}
```

---

## 📖 API Reference

All types are in the `autodiff` namespace.

### Core Types

#### `autodiff::Variable<N>`

Compile-time variable identifier used to specify which derivatives to track.

```cpp
using namespace autodiff;

Variable<0> x;  // First variable
Variable<1> y;  // Second variable
Variable<2> z;  // Third variable
```

#### `autodiff::AutoDiff<T, Norder, Nvars>`

Main class representing a value with its derivatives.

| Template Parameter | Description |
|--------------------|-------------|
| `T` | Numeric type (`double`, `float`, etc.) |
| `Norder` | Maximum derivative order to compute |
| `Nvars` | Number of independent variables |

### Constructors

```cpp
using namespace autodiff;

// Default: value = 0, all derivatives = 0
AutoDiff<double, 2, 2> f;

// Constant: value = v, all derivatives = 0
AutoDiff<double, 2, 2> f(5.0);

// Variable: value = v, d/dx_I = 1, others = 0
Variable<0> x;
AutoDiff<double, 2, 2> f(3.0, x);

// From raw data array
std::array<double, 6> data = {1, 2, 3, 4, 5, 6};
AutoDiff<double, 2, 2> f(data);
```

### Member Functions

| Method | Description |
|--------|-------------|
| `value()` | Returns the function value (zeroth derivative) |
| `diff_value(vars...)` | Returns the scalar value of a specific derivative |
| `diff(vars...)` | Returns an AutoDiff representing the derivative |
| `reduced_diff(vars...)` | Returns a reduced-order AutoDiff of the derivative |
| `data()` | Returns pointer to internal storage |

### Supported Operators

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Binary arithmetic |
| `+=`, `-=`, `*=`, `/=` | Compound assignment |
| `+`, `-` (unary) | Unary plus/minus |
| `pow(f, g)` | Exponentiation |
| `exp(f)` | Exponential |
| `log(f)` | Natural logarithm |

---

## ⚙️ Compilation Options

AutoDiff provides several preprocessor macros to optimize performance for different use cases.

### Available Macros

| Macro | Description | Trade-off |
|-------|-------------|-----------|
| `AUTODIFF_FAST` | Enables aggressive inlining with `AD_INLINE` attribute | Faster execution, longer compilation time |
| `AUTODIFF_SCALAR_OPTS` | Enables optimized scalar-AutoDiff operations | Faster scalar ops, more code |
| `AUTODIFF_ITER_MUL_OPT` | Uses precomputed Leibniz coefficients for multiplication | Fastest multiplication, compile-time cost |
| `AUTODIFF_ITER_MUL` | Uses iterative (non-recursive) multiplication | Balanced speed/compile-time |

### Usage

```bash
# Basic compilation
g++ -std=c++20 -O3 main.cpp -o main

# With all optimizations
g++ -std=c++20 -O3 -DAUTODIFF_FAST -DAUTODIFF_SCALAR_OPTS -DAUTODIFF_ITER_MUL_OPT main.cpp -o main

# Balanced (recommended for most cases)
g++ -std=c++20 -O3 -DAUTODIFF_FAST -DAUTODIFF_SCALAR_OPTS main.cpp -o main

# With iterative multiplication (good compile times)
g++ -std=c++20 -O3 -DAUTODIFF_FAST -DAUTODIFF_ITER_MUL main.cpp -o main
```

### Macro Details

#### `AUTODIFF_FAST`

When defined, `AUTODIFF_MAYBE_INLINE` becomes `AD_INLINE` (force inline). Otherwise, it's just `inline`.

```cpp
// Without AUTODIFF_FAST: AUTODIFF_MAYBE_INLINE = inline
// With AUTODIFF_FAST:    AUTODIFF_MAYBE_INLINE = __attribute__((always_inline)) inline
```

#### `AUTODIFF_SCALAR_OPTS`

Enables specialized overloads for operations between AutoDiff and scalars:

```cpp
using namespace autodiff;

// Without AUTODIFF_SCALAR_OPTS: scalar * AutoDiff goes through generic code path
// With AUTODIFF_SCALAR_OPTS: optimized element-wise multiplication
AutoDiff<double, 2, 2> f = a * 2.0;  // Each derivative multiplied by 2
```

#### `AUTODIFF_ITER_MUL_OPT`

Uses `LeibnizDiff` to precompute all multinomial coefficients and array offsets at compile time. Best performance but increases compile time for high orders.

#### `AUTODIFF_ITER_MUL`

Uses an iterative implementation without precomputed caches. Good balance between runtime performance and compile time.

### Recommended Configurations

| Use Case | Flags |
|----------|-------|
| **Development/Debugging** | `-O2` |
| **Production (balanced)** | `-O3 -DAUTODIFF_FAST -DAUTODIFF_SCALAR_OPTS` |
| **Maximum Performance** | `-O3 -DAUTODIFF_FAST -DAUTODIFF_SCALAR_OPTS -DAUTODIFF_ITER_MUL_OPT` |
| **Fast Compile Times** | `-O3 -DAUTODIFF_SCALAR_OPTS` |
| **High-Order Derivatives** | `-O3 -DAUTODIFF_FAST -DAUTODIFF_ITER_MUL` |

---

## ⚡ Performance

### Storage Complexity

The number of stored derivatives grows combinatorially:

```
Storage Size = C(Nvars + Norder, Norder)
```

| Variables | Order 1 | Order 2 | Order 3 | Order 4 |
|-----------|---------|---------|---------|---------|
| 1 | 2 | 3 | 4 | 5 |
| 2 | 3 | 6 | 10 | 15 |
| 3 | 4 | 10 | 20 | 35 |
| 4 | 5 | 15 | 35 | 70 |
| 5 | 6 | 21 | 56 | 126 |

### Storage Layout

Derivatives are stored in graded colexicographic order, grouped by total order:

```
For 3 variables (x, y, z) and order 2:
[f, fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz]
 ↑   ↑-------↑   ↑---------------------------↑
 │   1st order   2nd order derivatives
 value
```

### Benchmarking Tips

```cpp
#include <autodiff/autodiff.hpp>
#include <chrono>

using namespace autodiff;

// Measure overhead vs raw computation
auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < N; i++) {
    auto result = compute(x, y, z);
    volatile auto v = result.value();  // Prevent optimization
}

auto end = std::chrono::high_resolution_clock::now();
```

---

## 🔬 Advanced Usage

### Extracting Derivative Objects

```cpp
using namespace autodiff;

Variable<0> x;
Variable<1> y;

AutoDiff<double, 3, 2> f(2.0, x);
auto g = f * f * f;  // g = x³

// Get df/dx as a new AutoDiff (with its own derivatives)
auto dg_dx = g.reduced_diff(x);  // AutoDiff<double, 2, 2>

// dg_dx.value() = d(x³)/dx = 3x²
// dg_dx.diff_value(x) = d²(x³)/dx² = 6x
```

### Compile-Time Optimization

```cpp
using namespace autodiff;

// Use cmpl_reduced_diff for compile-time known indices
Variable<0> x;
Variable<1> y;
auto dg = g.cmpl_reduced_diff(x, y);

// Equivalent to reduced_diff(0, 1) but with better optimization
```

### Creating Constants

```cpp
using namespace autodiff;

// Explicit constant (derivatives = 0)
AutoDiff<double, 2, 2> c(3.14159);

// In expressions, scalars are automatically treated as constants
auto f = a * 2.0 + 1.0;  // 2.0 and 1.0 are constants
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

</div>
