#include "../include/autodiff/autodiff.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>

// Test configuration
using namespace autodiff;
constexpr double EPSILON = 1e-10;
int tests_passed = 0;
int tests_failed = 0;

// Helper to check if two doubles are approximately equal
bool approx_equal(double a, double b, double eps = EPSILON) {
    if (std::abs(b) < eps) {
        return std::abs(a - b) < eps;
    }
    return std::abs((a - b) / b) < eps;
}

// Test assertion helper
void check(const std::string& name, double computed, double expected) {
    bool passed = approx_equal(computed, expected);
    if (passed) {
        std::cout << "  [PASS] " << name << ": " << computed << " == " << expected << "\n";
        tests_passed++;
    } else {
        std::cout << "  [FAIL] " << name << ": " << computed << " != " << expected << "\n";
        tests_failed++;
    }
}

void print_section(const std::string& title) {
    std::cout << "\n========================================\n";
    std::cout << title << "\n";
    std::cout << "========================================\n";
}

// ============================================================================
//                              TEST CASES
// ============================================================================

void test_basic_arithmetic() {
    print_section("Basic Arithmetic (2 vars, order 2)");

    Variable<0> x;
    Variable<1> y;

    using F = AutoDiff<double, 2, 2>;

    F a(3.0, x);  // a = x, value = 3
    F b(2.0, y);  // b = y, value = 2

    // Test addition: f = x + y
    {
        auto f = a + b;
        check("(x+y) value", f.value(), 5.0);
        check("(x+y) df/dx", f.diff_value(x), 1.0);
        check("(x+y) df/dy", f.diff_value(y), 1.0);
        check("(x+y) d²f/dx²", f.diff_value(x, x), 0.0);
        check("(x+y) d²f/dxdy", f.diff_value(x, y), 0.0);
    }

    // Test subtraction: f = x - y
    {
        auto f = a - b;
        check("(x-y) value", f.value(), 1.0);
        check("(x-y) df/dx", f.diff_value(x), 1.0);
        check("(x-y) df/dy", f.diff_value(y), -1.0);
    }

    // Test multiplication: f = x * y
    {
        auto f = a * b;
        check("(x*y) value", f.value(), 6.0);
        check("(x*y) df/dx", f.diff_value(x), 2.0);  // y
        check("(x*y) df/dy", f.diff_value(y), 3.0);  // x
        check("(x*y) d²f/dxdy", f.diff_value(x, y), 1.0);
        check("(x*y) d²f/dx²", f.diff_value(x, x), 0.0);
    }

    // Test division: f = x / y at (3, 2)
    {
        auto f = a / b;
        check("(x/y) value", f.value(), 1.5);
        check("(x/y) df/dx", f.diff_value(x), 0.5);       // 1/y = 0.5
        check("(x/y) df/dy", f.diff_value(y), -0.75);     // -x/y² = -3/4
        check("(x/y) d²f/dx²", f.diff_value(x, x), 0.0);
        check("(x/y) d²f/dxdy", f.diff_value(x, y), -0.25); // -1/y² = -1/4
        check("(x/y) d²f/dy²", f.diff_value(y, y), 0.75);   // 2x/y³ = 6/8
    }
}

void test_polynomial() {
    print_section("Polynomial: f(x,y,z) = x³y²z + xy³ (3 vars, order 3)");

    Variable<0> X;
    Variable<1> Y;
    Variable<2> Z;

    using F = AutoDiff<double, 3, 3>;

    // f(x,y,z) = x³y²z + xy³ at (2, 3, 1)
    F x(2.0, X);
    F y(3.0, Y);
    F z(1.0, Z);

    auto f = x*x*x * y*y * z + x * y*y*y;

    // Expected values (computed manually):
    // f = 8*9*1 + 2*27 = 72 + 54 = 126
    // df/dx = 3x²y²z + y³ = 3*4*9*1 + 27 = 108 + 27 = 135
    // df/dy = 2x³yz + 3xy² = 2*8*3*1 + 3*2*9 = 48 + 54 = 102
    // df/dz = x³y² = 8*9 = 72
    // d²f/dx² = 6xy²z = 6*2*9*1 = 108
    // d²f/dxdy = 6x²yz + 3y² = 6*4*3*1 + 3*9 = 72 + 27 = 99
    // d²f/dxdz = 3x²y² = 3*4*9 = 108
    // d²f/dy² = 2x³z + 6xy = 2*8*1 + 6*2*3 = 16 + 36 = 52
    // d²f/dydz = 2x³y = 2*8*3 = 48
    // d²f/dz² = 0
    // d³f/dx³ = 6y²z = 6*9*1 = 54
    // d³f/dx²dy = 12xyz + 0 = 12*2*3*1 = 72
    // d³f/dx²dz = 6xy² = 6*2*9 = 108
    // d³f/dxdy² = 6x²z + 6y = 6*4*1 + 6*3 = 24 + 18 = 42
    // d³f/dxdydz = 6x²y = 6*4*3 = 72
    // d³f/dxdz² = 0
    // d³f/dy³ = 6x = 12
    // d³f/dy²dz = 2x³ = 16
    // d³f/dydz² = 0
    // d³f/dz³ = 0

    check("f value", f.value(), 126.0);

    // First derivatives
    check("df/dx", f.diff_value(X), 135.0);
    check("df/dy", f.diff_value(Y), 102.0);
    check("df/dz", f.diff_value(Z), 72.0);

    // Second derivatives
    check("d²f/dx²", f.diff_value(X, X), 108.0);
    check("d²f/dxdy", f.diff_value(X, Y), 99.0);
    check("d²f/dxdz", f.diff_value(X, Z), 108.0);
    check("d²f/dy²", f.diff_value(Y, Y), 52.0);
    check("d²f/dydz", f.diff_value(Y, Z), 48.0);
    check("d²f/dz²", f.diff_value(Z, Z), 0.0);

    // Third derivatives
    check("d³f/dx³", f.diff_value(X, X, X), 54.0);
    check("d³f/dx²dy", f.diff_value(X, X, Y), 72.0);
    check("d³f/dx²dz", f.diff_value(X, X, Z), 108.0);
    check("d³f/dxdy²", f.diff_value(X, Y, Y), 42.0);
    check("d³f/dxdydz", f.diff_value(X, Y, Z), 72.0);
    check("d³f/dxdz²", f.diff_value(X, Z, Z), 0.0);
    check("d³f/dy³", f.diff_value(Y, Y, Y), 12.0);
    check("d³f/dy²dz", f.diff_value(Y, Y, Z), 16.0);
    check("d³f/dydz²", f.diff_value(Y, Z, Z), 0.0);
    check("d³f/dz³", f.diff_value(Z, Z, Z), 0.0);
}

void test_exp_log() {
    print_section("Exp and Log functions (1 var, order 3)");

    Variable<0> x;
    using F = AutoDiff<double, 3, 1>;

    // Test exp(x) at x = 1
    {
        F a(1.0, x);
        auto f = exp(a);

        double e = std::exp(1.0);
        check("exp(x) value", f.value(), e);
        check("exp(x) df/dx", f.diff_value(x), e);
        check("exp(x) d²f/dx²", f.diff_value(x, x), e);
        check("exp(x) d³f/dx³", f.diff_value(x, x, x), e);
    }

    // Test log(x) at x = 2
    {
        F a(2.0, x);
        auto f = log(a);

        check("log(x) value", f.value(), std::log(2.0));
        check("log(x) df/dx", f.diff_value(x), 0.5);        // 1/x
        check("log(x) d²f/dx²", f.diff_value(x, x), -0.25); // -1/x²
        check("log(x) d³f/dx³", f.diff_value(x, x, x), 0.25); // 2/x³
    }

    // Test exp(2x) at x = 1
    {
        F a(1.0, x);
        auto f = exp(a * 2.0);

        double e2 = std::exp(2.0);
        check("exp(2x) value", f.value(), e2);
        check("exp(2x) df/dx", f.diff_value(x), 2.0 * e2);
        check("exp(2x) d²f/dx²", f.diff_value(x, x), 4.0 * e2);
        check("exp(2x) d³f/dx³", f.diff_value(x, x, x), 8.0 * e2);
    }
}

void test_pow() {
    print_section("Power function (1-2 vars, order 3)");

    Variable<0> x;
    Variable<1> y;

    // Test x² at x = 3
    {
        using F = AutoDiff<double, 3, 1>;
        F a(3.0, x);
        auto f = pow(a, 2.0);

        check("x² value", f.value(), 9.0);
        check("x² df/dx", f.diff_value(x), 6.0);      // 2x
        check("x² d²f/dx²", f.diff_value(x, x), 2.0); // 2
        check("x² d³f/dx³", f.diff_value(x, x, x), 0.0);
    }

    // Test x³ at x = 2
    {
        using F = AutoDiff<double, 3, 1>;
        F a(2.0, x);
        auto f = pow(a, 3.0);

        check("x³ value", f.value(), 8.0);
        check("x³ df/dx", f.diff_value(x), 12.0);       // 3x²
        check("x³ d²f/dx²", f.diff_value(x, x), 12.0);  // 6x
        check("x³ d³f/dx³", f.diff_value(x, x, x), 6.0); // 6
    }

    // Test 2^x (exponential with constant base) at x = 3
    {
        using F = AutoDiff<double, 3, 1>;
        F a(3.0, x);
        auto f = pow(2.0, a);

        double ln2 = std::log(2.0);
        double two_cubed = 8.0;
        check("2^x value", f.value(), two_cubed);
        check("2^x df/dx", f.diff_value(x), two_cubed * ln2);
        check("2^x d²f/dx²", f.diff_value(x, x), two_cubed * ln2 * ln2);
        check("2^x d³f/dx³", f.diff_value(x, x, x), two_cubed * ln2 * ln2 * ln2);
    }

    // Test x^y at (2, 3)
    {
        using F = AutoDiff<double, 2, 2>;
        F a(2.0, x);
        F b(3.0, y);
        auto f = pow(a, b);

        // f = x^y = 8
        // df/dx = y * x^(y-1) = 3 * 4 = 12
        // df/dy = x^y * ln(x) = 8 * ln(2)
        double ln2 = std::log(2.0);
        check("x^y value", f.value(), 8.0);
        check("x^y df/dx", f.diff_value(x), 12.0);
        check("x^y df/dy", f.diff_value(y), 8.0 * ln2);
    }
}

void test_compound_operators() {
    print_section("Compound Assignment Operators (2 vars, order 2)");

    Variable<0> x;
    Variable<1> y;

    using F = AutoDiff<double, 2, 2>;

    // Test +=
    {
        F a(3.0, x);
        F b(2.0, y);
        a += b;  // a = x + y

        check("(x+=y) value", a.value(), 5.0);
        check("(x+=y) df/dx", a.diff_value(x), 1.0);
        check("(x+=y) df/dy", a.diff_value(y), 1.0);
    }

    // Test += with scalar
    {
        F a(3.0, x);
        a += 5.0;

        check("(x+=5) value", a.value(), 8.0);
        check("(x+=5) df/dx", a.diff_value(x), 1.0);
    }

    // Test -=
    {
        F a(3.0, x);
        F b(2.0, y);
        a -= b;  // a = x - y

        check("(x-=y) value", a.value(), 1.0);
        check("(x-=y) df/dx", a.diff_value(x), 1.0);
        check("(x-=y) df/dy", a.diff_value(y), -1.0);
    }

    // Test -= with scalar
    {
        F a(3.0, x);
        a -= 1.0;

        check("(x-=1) value", a.value(), 2.0);
        check("(x-=1) df/dx", a.diff_value(x), 1.0);
    }

    // Test *=
    {
        F a(3.0, x);
        F b(2.0, y);
        a *= b;  // a = x * y

        check("(x*=y) value", a.value(), 6.0);
        check("(x*=y) df/dx", a.diff_value(x), 2.0);
        check("(x*=y) df/dy", a.diff_value(y), 3.0);
    }

    // Test *= with scalar
    {
        F a(3.0, x);
        a *= 4.0;

        check("(x*=4) value", a.value(), 12.0);
        check("(x*=4) df/dx", a.diff_value(x), 4.0);
    }

    // Test /=
    {
        F a(6.0, x);
        F b(2.0, y);
        a /= b;  // a = x / y at (6, 2)

        check("(x/=y) value", a.value(), 3.0);
        check("(x/=y) df/dx", a.diff_value(x), 0.5);
        check("(x/=y) df/dy", a.diff_value(y), -1.5);  // -x/y² = -6/4
    }

    // Test /= with scalar
    {
        F a(6.0, x);
        a /= 2.0;

        check("(x/=2) value", a.value(), 3.0);
        check("(x/=2) df/dx", a.diff_value(x), 0.5);
    }
}

void test_complex_expressions() {
    print_section("Complex Expressions (2-3 vars, order 2-3)");

    Variable<0> x;
    Variable<1> y;
    Variable<2> z;

    // Test f = exp(x) * y at (1, 2)
    {
        using F = AutoDiff<double, 2, 2>;
        F a(1.0, x);
        F b(2.0, y);
        auto f = exp(a) * b;

        double e = std::numbers::e;
        check("exp(x)*y value", f.value(), 2.0 * e);
        check("exp(x)*y df/dx", f.diff_value(x), 2.0 * e);  // y * exp(x)
        check("exp(x)*y df/dy", f.diff_value(y), e);         // exp(x)
        check("exp(x)*y d²f/dxdy", f.diff_value(x, y), e);   // exp(x)
    }

    // Test f = log(x*y) at (2, 3)
    {
        using F = AutoDiff<double, 2, 2>;
        F a(2.0, x);
        F b(3.0, y);
        auto f = log(a * b);

        // f = log(xy) = log(6)
        // df/dx = 1/(xy) * y = 1/x = 0.5
        // df/dy = 1/(xy) * x = 1/y = 1/3
        // d²f/dx² = -1/x² = -0.25
        // d²f/dxdy = 0
        // d²f/dy² = -1/y² = -1/9

        check("log(xy) value", f.value(), std::log(6.0));
        check("log(xy) df/dx", f.diff_value(x), 0.5);
        check("log(xy) df/dy", f.diff_value(y), 1.0/3.0);
        check("log(xy) d²f/dx²", f.diff_value(x, x), -0.25);
        check("log(xy) d²f/dxdy", f.diff_value(x, y), 0.0);
        check("log(xy) d²f/dy²", f.diff_value(y, y), -1.0/9.0);
    }

    // Test f = x / (y + z) at (6, 2, 1)
    {
        using F = AutoDiff<double, 2, 3>;
        F a(6.0, x);
        F b(2.0, y);
        F c(1.0, z);
        auto f = a / (b + c);

        // f = x/(y+z) = 6/3 = 2
        // df/dx = 1/(y+z) = 1/3
        // df/dy = -x/(y+z)² = -6/9 = -2/3
        // df/dz = -x/(y+z)² = -6/9 = -2/3
        // d²f/dxdy = -1/(y+z)² = -1/9
        // d²f/dy² = 2x/(y+z)³ = 12/27 = 4/9
        // d²f/dydz = 2x/(y+z)³ = 4/9

        check("x/(y+z) value", f.value(), 2.0);
        check("x/(y+z) df/dx", f.diff_value(x), 1.0/3.0);
        check("x/(y+z) df/dy", f.diff_value(y), -2.0/3.0);
        check("x/(y+z) df/dz", f.diff_value(z), -2.0/3.0);
        check("x/(y+z) d²f/dxdy", f.diff_value(x, y), -1.0/9.0);
        check("x/(y+z) d²f/dy²", f.diff_value(y, y), 4.0/9.0);
        check("x/(y+z) d²f/dydz", f.diff_value(y, z), 4.0/9.0);
    }

    // Test f = exp(x²) at x = 1
    {
        using F = AutoDiff<double, 3, 1>;
        F a(1.0, x);
        auto f = exp(a * a);

        double e = std::exp(1.0);
        // f = exp(x²)
        // f' = 2x * exp(x²)
        // f'' = 2*exp(x²) + 4x²*exp(x²) = (2 + 4x²)*exp(x²)
        // f''' = 8x*exp(x²) + (2+4x²)*2x*exp(x²) = (8x + 4x + 8x³)*exp(x²) = (12x + 8x³)*exp(x²)
        // At x=1: f=e, f'=2e, f''=6e, f'''=20e

        check("exp(x²) value", f.value(), e);
        check("exp(x²) df/dx", f.diff_value(x), 2.0 * e);
        check("exp(x²) d²f/dx²", f.diff_value(x, x), 6.0 * e);
        check("exp(x²) d³f/dx³", f.diff_value(x, x, x), 20.0 * e);
    }
}

void test_unary_operators() {
    print_section("Unary Operators");

    Variable<0> x;
    using F = AutoDiff<double, 2, 1>;

    F a(3.0, x);

    // Test unary plus
    {
        auto f = +a;
        check("+x value", f.value(), 3.0);
        check("+x df/dx", f.diff_value(x), 1.0);
    }

    // Test unary minus
    {
        auto f = -a;
        check("-x value", f.value(), -3.0);
        check("-x df/dx", f.diff_value(x), -1.0);
    }

    // Test double negation
    {
        auto f = -(-a);
        check("--x value", f.value(), 3.0);
        check("--x df/dx", f.diff_value(x), 1.0);
    }
}

void test_scalar_operations() {
    print_section("Scalar Operations");

    Variable<0> x;
    using F = AutoDiff<double, 2, 1>;

    F a(3.0, x);

    // Scalar + AutoDiff
    {
        auto f = 5.0 + a;
        check("5+x value", f.value(), 8.0);
        check("5+x df/dx", f.diff_value(x), 1.0);
    }

    // AutoDiff + Scalar
    {
        auto f = a + 5.0;
        check("x+5 value", f.value(), 8.0);
        check("x+5 df/dx", f.diff_value(x), 1.0);
    }

    // Scalar - AutoDiff
    {
        auto f = 5.0 - a;
        check("5-x value", f.value(), 2.0);
        check("5-x df/dx", f.diff_value(x), -1.0);
    }

    // AutoDiff - Scalar
    {
        auto f = a - 5.0;
        check("x-5 value", f.value(), -2.0);
        check("x-5 df/dx", f.diff_value(x), 1.0);
    }

    // Scalar * AutoDiff
    {
        auto f = 2.0 * a;
        check("2*x value", f.value(), 6.0);
        check("2*x df/dx", f.diff_value(x), 2.0);
    }

    // AutoDiff * Scalar
    {
        auto f = a * 2.0;
        check("x*2 value", f.value(), 6.0);
        check("x*2 df/dx", f.diff_value(x), 2.0);
    }

    // Scalar / AutoDiff at x = 2
    {
        F b(2.0, x);
        auto f = 6.0 / b;
        check("6/x value", f.value(), 3.0);
        check("6/x df/dx", f.diff_value(x), -1.5);  // -6/x² = -6/4
    }

    // AutoDiff / Scalar
    {
        auto f = a / 2.0;
        check("x/2 value", f.value(), 1.5);
        check("x/2 df/dx", f.diff_value(x), 0.5);
    }
}

void test_reduced_diff() {
    print_section("Reduced Diff Methods");

    Variable<0> x;
    Variable<1> y;

    using F = AutoDiff<double, 3, 2>;

    // f = x²y at (2, 3)
    F a(2.0, x);
    F b(3.0, y);
    auto f = a * a * b;

    // f = x²y = 12
    // df/dx = 2xy = 12
    // df/dy = x² = 4
    // d²f/dx² = 2y = 6
    // d²f/dxdy = 2x = 4
    // d³f/dx²dy = 2

    // Test reduced_diff
    auto df_dx = f.reduced_diff(x);
    check("reduced_diff(x).value()", df_dx.value(), 12.0);
    check("reduced_diff(x).diff_value(x)", df_dx.diff_value(x), 6.0);
    check("reduced_diff(x).diff_value(y)", df_dx.diff_value(y), 4.0);

    auto d2f_dxdy = f.reduced_diff(x, y);
    check("reduced_diff(x,y).value()", d2f_dxdy.value(), 4.0);
    check("reduced_diff(x,y).diff_value(x)", d2f_dxdy.diff_value(x), 2.0);

    // Test diff (returns same type)
    auto df_dy = f.diff(y);
    check("diff(y).value()", df_dy.value(), 4.0);
    check("diff(y).diff_value(x)", df_dy.diff_value(x), 4.0);
}

void test_constants() {
    print_section("Constant Values");

    Variable<0> x;
    using F = AutoDiff<double, 2, 1>;

    // Test constant (no derivatives)
    F c(5.0);

    check("const value", c.value(), 5.0);
    check("const df/dx", c.diff_value(x), 0.0);
    check("const d²f/dx²", c.diff_value(x, x), 0.0);

    // Test constant in expression
    F a(3.0, x);
    auto f = a + c;

    check("x+const value", f.value(), 8.0);
    check("x+const df/dx", f.diff_value(x), 1.0);
}

// ============================================================================
//                              MAIN
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(10);

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              AUTODIFF COMPREHENSIVE TEST SUITE               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    test_basic_arithmetic();
    test_polynomial();
    test_exp_log();
    test_pow();
    test_compound_operators();
    test_complex_expressions();
    test_unary_operators();
    test_scalar_operations();
    test_reduced_diff();
    test_constants();

    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "                         SUMMARY\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "  Total tests: " << (tests_passed + tests_failed) << "\n";
    std::cout << "  Passed:      " << tests_passed << "\n";
    std::cout << "  Failed:      " << tests_failed << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";

    if (tests_failed == 0) {
        std::cout << "  ✓ ALL TESTS PASSED!\n";
    } else {
        std::cout << "  ✗ SOME TESTS FAILED!\n";
    }
    std::cout << "══════════════════════════════════════════════════════════════════\n\n";

    return tests_failed > 0 ? 1 : 0;
}
