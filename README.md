<div align="center">

# XDiff

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat&logo=c%2B%2B)
![Header Only](https://img.shields.io/badge/Header-Only-green.svg?style=flat)
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
- **Header-Only** — Just include and use


---

## Quick Start

The main type:
```cpp
template<typename T, typename GD, int Order>
class Dual;
```
The class stores a main value, and an abstract object that holds the derivative information. The object can be e.g. a `std::array<double, 4>` when the vector-space is 4-dimensional (4 variable to differentiate w.r.t), but can also be any other object, as long as the information on how to access concrete elements of type `T` is given.

More concretely, the `GD` must be a struct of the form:

```cpp
struct GradientDescriptor{

    template<typanem T>
    using state_type = std::array<T, 4>;

    template<typename F, typename U, typename... Rest>
    static void for_each(F&& f, U&& proto, Rest&&... rest){
        // proto and rest... are state_type's some const&, some non-const
        // Call `f` for all elements of each parameter
        for (size_t i=0; i<4; i++){
            f(proto[i], rest[i]...);
        }
    }

    template<typename To, typename From>
    static void resize_like(state_type<To>& dst, const state_type<From>& ref){
        // nothing to do, std::array has a fixed size
    }
};
```

**Template parameters:**
| Parameter | Description |
|-----------|-------------|
| `T` | Underlying numeric type (`double`, `float`, etc.) |
| `GD` | A struct describing the gradient holder |
| `Order` | Maximum differentiation order |

This means that the GradientDescriptor struct must have a templated `state_type` alias, referencing the type of the derivative holder, and a `for_each` function, that takes as input a function or a lambda `f`, and at least one gradient `proto`. The `rest` parameter pack is also a pack of `state_type` objects, possibly of different template parameter `T`.

The `for_each` function must call the input function `f` for all `S` elements of each `state_type<S>` gradient. Essentially the `for_each` function tells the abstract `Dual` class how to access the elements of a gradient `state_type`. All the gradients passed to a single `for_each` call have the same shape.

The `resize_like` function must give `dst` the shape of `ref` (e.g. the same size for a `std::vector`), where the two may hold elements of different types. It is called whenever the `Dual` class allocates a new gradient, such as for the result of an operation or for a seed, before that gradient is passed to `for_each`. For fixed-size holders it could do nothing.

Concrete implementations using `std::array` and `std::vector` are provided in the `xdiff` namespace.

---

### Basic Example

Compile with:
```bash
g++ -std=c++20 -Iinclude -Iexternal/lazex/include -o test test.cpp
```

```cpp
#include <xdiff/xdiff.hpp>

using T = double;


int main(){

    // Assume the state vector is (x, y, z) = (1, 2, 3)
    // This means that the gradient of x is (1, 0, 0),
    // the gradient of y is (0, 1, 0) and of z is (0, 0, 1)
    // It is important that the gradient and the state vector are
    // the same object types.
    std::array<T, 3> state {1, 2, 3};

    // Now seeds[1] is the same as y, but also holds the information
    // that its gradient is (0, 1, 0)

    // The "4" means storing up to 4th order derivatives
    auto seeds = xdiff::make_seeds<T, xdiff::StaticState<3>, 4>(state);

    auto x = seeds[0];
    auto y = seeds[1];
    auto z = seeds[2];

    auto f = x*x/(1.3 - tan(z)*y) + sin(x-y/(z*z));

    // Convenience alias class that allows for easier access
    // for higher order derivatives
    xdiff::StaticDual<T, 3, 4> g = f;

    std::cout << "d^4f/(dxdxdzdy) = " << g.diff(0, 0, 2, 1) << std::endl;
}
```

Results match analytical expressions to machine precision. Requesting derivatives beyond `Norder` triggers a compile-time error.

---

## Calling math functions: `XDIFF_USING_MATH`

Write `XDIFF_USING_MATH;` in every namespace of yours that calls `abs()`, `sin()`, `pow()` …
**unqualified**. One line per namespace, or once at global scope:

```cpp
#include <xdiff/xdiff.hpp>

namespace myproject {
XDIFF_USING_MATH;

template<typename T>
T f(const T& x){
    // scalar, Dual or a type from the external dependencies
    return abs(x) * sqrt(x);
}

}
```
