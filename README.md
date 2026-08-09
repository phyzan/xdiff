# XDiff

<div align="center">

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)
![Header Only](https://img.shields.io/badge/Header-Only-green.svg?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg?style=flat)
![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20HIP%20%7C%20SYCL-orange.svg?style=flat)

**A modular C++20 library for arbitrary-order automatic differentiation**

</div>

---

## Features

- **Arbitrary-Order Derivatives** — Compute derivatives up to any order with compile-time configuration
- **Multivariate Support** — Handle functions of any number of independent variables
- **Dual Evaluation Paths** — Direct evaluation with `Dual` or lazy evaluation with expression templates
- **Expression Simplification** — Compile-time algebraic optimizations (constant folding, identity elimination)
- **GPU Support** — Native support for CUDA, HIP, and SYCL backends
- **Modular Architecture** — Clean separation of concerns across well-documented headers
- **Zero Runtime Overhead** — Template metaprogramming ensures computations are optimized at compile time
- **Header-Only** — Just include and use, no linking required
- **Exact Derivatives** — Computes analytical derivatives, not numerical approximations

---

## Quick Start

All types are in the `xdiff` namespace.

### Basic Example

```cpp
#include <xdiff/xdiff.hpp>
#include <iostream>

using namespace xdiff;

int main() {
    // Define compile-time variable symbols
    Symbol<0> x;
    Symbol<1> y;

    // Create Dual numbers: <Type, NumVariables, MaxOrder>
    // Order 2 = compute up to second derivatives
    // 2 variables = x and y
    Dual<double, 2, 2> a(x, 3.0);  // a = 3.0, da/dx = 1, da/dy = 0
    Dual<double, 2, 2> b(y, 2.0);  // b = 2.0, db/dx = 0, db/dy = 1

    // Compute f = a * b
    auto f = a * b;

    // Access results
    std::cout << "f(3,2) = " << f.value() << "\n";           // 6
    std::cout << "df/dx  = " << f.get_diff_wrt(x) << "\n";   // 2
    std::cout << "df/dy  = " << f.get_diff_wrt(y) << "\n";   // 3
    std::cout << "d²f/dxdy = " << f.get_diff_wrt(x, y) << "\n"; // 1

    return 0;
}
```

### Higher-Order Derivatives

```cpp
#include <xdiff/xdiff.hpp>

using namespace xdiff;

// Function: f(x,y,z) = x³y²z + xy³
template<typename T>
T compute(const T& x, const T& y, const T& z) {
    return x*x*x * y*y * z + x * y*y*y;
}

int main() {
    Symbol<0> X;
    Symbol<1> Y;
    Symbol<2> Z;

    // Track up to 3rd order derivatives for 3 variables
    using F = Dual<double, 3, 3>;

    F x(X, 2.0);
    F y(Y, 3.0);
    F z(Z, 1.0);

    auto f = compute(x, y, z);

    // First derivatives
    f.get_diff_wrt(X);        // df/dx
    f.get_diff_wrt(Y);        // df/dy
    f.get_diff_wrt(Z);        // df/dz

    // Second derivatives
    f.get_diff_wrt(X, X);     // d²f/dx²
    f.get_diff_wrt(X, Y);     // d²f/dxdy

    // Third derivatives
    f.get_diff_wrt(X, Y, Z);  // d³f/dxdydz
    f.get_diff_wrt(X, X, Y);  // d³f/dx²dy

    return 0;
}
```

### Using Expression Templates

XDiff supports lazy evaluation via expression templates for potential optimization:

```cpp
#include <xdiff/xdiff.hpp>

using namespace xdiff;

int main() {
    // Compile-time variable with known axis
    Variable<double, 0> x(3.0);
    Variable<double, 1> y(2.0);

    // Build expression lazily (not yet evaluated)
    auto expr = x * y + log(x);

    // Evaluate to a Dual when needed
    Dual<double, 2, 2> result = expr;

    return 0;
}
```

---

## API Reference

### Core Types

#### `xdiff::Symbol<N>`

Compile-time symbol for differentiation axes.

```cpp
Symbol<0> x;  // First variable
Symbol<1> y;  // Second variable
Symbol<2> z;  // Third variable
```

#### `xdiff::Dual<T, Nvars, Norder>`

Main class representing a value with all its partial derivatives.

| Template Parameter | Description |
|--------------------|-------------|
| `T` | Numeric type (`double`, `float`, etc.) |
| `Nvars` | Number of independent variables |
| `Norder` | Maximum derivative order to compute |

### Constructors

```cpp
// Default: value = 0, all derivatives = 0
Dual<double, 2, 2> f;

// Constant: value = v, all derivatives = 0
Dual<double, 2, 2> f(5.0);

// Variable (compile-time axis): value = v, d/dx_I = 1, others = 0
Symbol<0> x;
Dual<double, 2, 2> f(x, 3.0);

// Variable (runtime axis): value = v, d/dx_axis = 1
Dual<double, 2, 2> f(3.0, 0);  // axis = 0
```

### Member Functions

| Method | Description |
|--------|-------------|
| `value()` | Returns the function value (zeroth derivative) |
| `get_diff_wrt(vars...)` | Returns the scalar value of a specific derivative |
| `diff_wrt(vars...)` | Returns a Dual representing the derivative |
| `reduced_diff_wrt(vars...)` | Returns a reduced-order Dual of the derivative |
| `data()` | Returns reference to internal storage array |

### Supported Operators

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Binary arithmetic |
| `+=`, `-=`, `*=`, `/=` | Compound assignment |
| `+`, `-` (unary) | Unary plus/minus |
| `pow(f, g)` | Exponentiation |
| `log(f)` | Natural logarithm |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Comparison (by value only) |

### Key Design Concepts

#### Two Evaluation Paths

1. **Direct Path (Dual)**: Immediate evaluation with derivative propagation
   ```cpp
   Dual<double, 2, 2> a(x, 3.0);
   Dual<double, 2, 2> b(y, 2.0);
   auto f = a * b;  // Evaluated immediately
   ```

2. **Lazy Path (Expression Templates)**: Build expression tree, optimize, then evaluate
   ```cpp
   Variable<double, 0> x(3.0);
   auto expr = x * x + 2 * x;  // Expression tree (not evaluated)
   Dual<double, 1, 2> result = expr;  // Evaluated here
   ```

#### Expression Simplification

XDiff performs compile-time algebraic simplifications:

```cpp
// These optimizations happen at compile time:
0 + a  →  a           // Identity elimination
1 * a  →  a
a - 0  →  a
0 * a  →  0           // Zero propagation
-(-x)  →  x           // Double negation
(a/b)/c → a/(b*c)     // Algebraic simplification
const + const → const // Constant folding
```

---

## Performance

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

Derivatives are stored in graded colexicographic order:

```
For 3 variables (x, y, z) and order 2:
[f, fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz]
 ↑   └──────┘   └─────────────────────────┘
 │   1st order        2nd order
 value
```

### GPU Optimization

XDiff automatically detects GPU backends and uses:
- Thread-local scratch space on CPU for expression evaluation
- Direct evaluation on GPU to avoid shared state issues
- Aggressive inlining via `__forceinline__` on CUDA/HIP

---

## Advanced Usage

### Extracting Derivative Objects

```cpp
Symbol<0> x;
Symbol<1> y;

Dual<double, 2, 3> f(x, 2.0);
auto g = f * f * f;  // g = x³

// Get dg/dx as a new Dual (with its own derivatives)
auto dg_dx = g.reduced_diff_wrt(x);  // Dual<double, 2, 2>

// dg_dx.value() = d(x³)/dx = 3x² = 12
// dg_dx.get_diff_wrt(x) = d²(x³)/dx² = 6x = 12
```

### Runtime Variable Indices

```cpp
// When axis isn't known at compile time
Dual<double, 3, 2> f(2.0, axis);  // axis is a runtime size_t

// Or use runtime Variable
Variable<double, -1> x(3.0, 0);  // Axis specified at runtime
```

---

## License

This project is licensed under the MIT License.

---