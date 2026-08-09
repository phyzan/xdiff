#include <iostream>
#include <chrono>
#include <xdiff/xdiff.hpp>

using std::abs;

template<typename T>
XDIFF_FORCEINLINE void f1(T& out, const T& x, const T& y, const T& z){
    out += -sin(tan(pow(y, 2)) * x) / (4 - log(pow(x, y))) + z;
    out += abs(7 + -(x+10) * (y - 1) / (x * x * x)
             + abs(z * x / (y - 1))
             - x * y * z / (x + z)
             + z * z / (x - z * (y / (x * x)))
             - (-y + z * z) * x / y
             + x / (z + x * y / 4 - 2 * (7 / y))
             - (x * x - y * y) / (x * y - z * z / x)
             + (x + y) * (-(z / x) + y / (x * x)));

    out -= x*y;
    out /= (1 + x*x + y*y + z*z);
    out *= (1.079248 - x*y*z);
}


template<typename T>
XDIFF_FORCEINLINE void f2(T& out, const T& x, const T& y, const T& z){
    out += - x * y * z / (x + z)
             + z * z / (x - z * (y / (x * x)))
             - (-y + z * z) * x / y
             + x / (z + x * y / 4 - 2 * (7 / y))
             - (x * x - y * y) / (x * y - z * z / x)
             + (x + y) * (-(z / x) + y / (x * x));

    out -= x*y;
    out /= (1 + x*x + y*y + z*z);
    out *= (1.079248 - x*y*z);
}


template<typename T>
void run(const char* label, size_t n_times, const T& x, const T& y, const T& z){
    
    // time this loop
    auto start = std::chrono::high_resolution_clock::now();
    T res = 0*x; // instead of 0.0, so that if res is Dual, it will be initialized as a Dual with the same number of variables and order as x
    for (size_t i=0; i<n_times; i++){
        f2(res, x, y, z);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n============= " << label << " =============\n";
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    std::cout << "Result: " << res << "\n";
    if constexpr (not std::is_same_v<T, double>){
        std::cout << "1st Derivatives: " << "\n---------------------\n";
        std::cout << "d/dx: " << res.get_diff_wrt(0) << "\n";
        std::cout << "d/dy: " << res.get_diff_wrt(1) << "\n";
        std::cout << "d/dz: " << res.get_diff_wrt(2) << "\n";
        std::cout << "\n\nSecond Derivatives: " << "\n---------------------\n";
        std::cout << "d^2/dx^2: " << res.get_diff_wrt(0, 0) << "\n";
        std::cout << "d^2/dxdy: " << res.get_diff_wrt(0, 1) << "\n";
        std::cout << "d^2/dxdz: " << res.get_diff_wrt(0, 2) << "\n";
        std::cout << "d^2/dy^2: " << res.get_diff_wrt(1, 1) << "\n";
        std::cout << "d^2/dydz: " << res.get_diff_wrt(1, 2) << "\n";
        std::cout << "d^2/dz^2: " << res.get_diff_wrt(2, 2) << "\n";
    }
}

using namespace xdiff;

int main(int argc, char *argv[]){

#ifdef XDIFF_LAZY_NESTED_DUAL
    std::cout << "XDIFF_LAZY_NESTED_DUAL = ON" << std::endl;
#else
    std::cout << "XDIFF_LAZY_NESTED_DUAL = OFF" << std::endl;
#endif

    using T = double;

    size_t NTIMES = atoi(argv[1]);

    // full compile-time nested
    {
        using D = Dual<T, 3, 2, Layout::Nested>;
        D x(1.0, MakeDual{.axis = 0, .nvars = 3});
        D y(2.0, MakeDual{.axis = 1, .nvars = 3});
        D z(3.0, MakeDual{.axis = 2, .nvars = 3});
        run("Nested + Stack", NTIMES, x, y, z);
    }

    // runtime recursive
    {
        using D = Dual<T, 0, 2, Layout::Nested>;
        D x(1.0, MakeDual{.axis = 0, .nvars = 3});
        D y(2.0, MakeDual{.axis = 1, .nvars = 3});
        D z(3.0, MakeDual{.axis = 2, .nvars = 3});
        run("Nested + Heap", NTIMES, x, y, z);
    }

    // lazy runtime recursive
    {
        using D = lazy::LazyType<Dual<T, 0, 2, Layout::Nested>>;
        D x(1.0, MakeDual{.axis = 0, .nvars = 3});
        D y(2.0, MakeDual{.axis = 1, .nvars = 3});
        D z(3.0, MakeDual{.axis = 2, .nvars = 3});
        run("Nested + Heap + Lazy", NTIMES, x, y, z);
    }

    // compile-time memory-efficient
    {
        using D = Dual<T, 3, 2, Layout::Flat>;
        D x(1.0, MakeDual{.axis = 0, .nvars = 3});
        D y(2.0, MakeDual{.axis = 1, .nvars = 3});
        D z(3.0, MakeDual{.axis = 2, .nvars = 3});
        run("Flat + Stack", NTIMES, x, y, z);
    }

    // plain double
    {
        using D = double;
        D x = 1.0;
        D y = 2.0;
        D z = 3.0;
        run("Plain double", NTIMES, x, y, z);
    }
}


// clang++ -std=c++20 -O3 -DNDEBUG -DXDIFF_FAST -DXDIFF_LEIBNIZ_OPT -DXDIFF_SCALAR_OPTIMIZATIONS -DXDIFF_LAZY_NESTED_DUAL -Iinclude -Iexternal/lazy/include -Iexternal/ndspan/include benchmark.cpp -o benchmark && ./benchmark 1000