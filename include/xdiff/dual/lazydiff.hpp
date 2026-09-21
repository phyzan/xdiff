#ifndef XDIFF_LAZY_HPP
#define XDIFF_LAZY_HPP


#include "dual.hpp" // IWYU pragma: keep
#include "../seed/dualseed.hpp" // IWYU pragma: keep
#include <lazex/lazex.hpp>



namespace lazex::detail {

template<typename T>
concept arithmetic = std::is_arithmetic_v<T>;
/**
 * @brief Unary function specialisations for `mpfr::mpreal`.
 *
 * Overrides `CustomUnaryRules<mpfr::mpreal>::evaluate` for `NEG`, `ABS`, and `SQRT`
 * using the corresponding raw MPFR C library functions, which avoid any overhead from
 * the `mpfr::mpreal` operator overloads and respect the global rounding mode.
 */
template<typename T, int NVARS, int NORDER, xdiff::Layout LY>
struct CustomUnaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>> : public UnaryEvaluator<CustomUnaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>>, xdiff::Dual<T, NVARS, NORDER, LY>>{

    using DualType = xdiff::Dual<T, NVARS, NORDER, LY>;
    using Base = UnaryEvaluator<CustomUnaryEvaluator<DualType>, DualType>;
    using Base::eval_rule;
    using Base::evaluate;

    // neg
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::NEG, DualType){
        xdiff::assign_neg(out, a);
    }

    // abs
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::ABS, DualType){
        xdiff::assign_abs(out, a);
    }

    // sqrt
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::SQRT, DualType){
        xdiff::assign_sqrt(out, a);
    }

    // exp
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::EXP, DualType){
        xdiff::assign_exp(out, a);
    }

    // log
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::LOG, DualType){
        xdiff::assign_log(out, a);
    }

    // sin
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::SIN, DualType){
        xdiff::assign_sin(out, a);
    }

    // cos
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::COS, DualType){
        xdiff::assign_cos(out, a);
    }

    // tan
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::TAN, DualType){
        xdiff::assign_tan(out, a);
    }

    // cot
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::COT, DualType){
        xdiff::assign_cot(out, a);
    }

    // sec
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::SEC, DualType){
        xdiff::assign_sec(out, a);
    }

    // csc
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::CSC, DualType){
        xdiff::assign_csc(out, a);
    }

    // asin
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::ASIN, DualType){
        xdiff::assign_asin(out, a);
    }

    // acos
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::ACOS, DualType){
        xdiff::assign_acos(out, a);
    }

    // atan
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::ATAN, DualType){
        xdiff::assign_atan(out, a);
    }

    // sinh
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::SINH, DualType){
        xdiff::assign_sinh(out, a);
    }

    // cosh
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::COSH, DualType){
        xdiff::assign_cosh(out, a);
    }

    // tanh
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::TANH, DualType){
        xdiff::assign_tanh(out, a);
    }

    // erf
    LAZEX_EVALUATE_FUNC(DualType, a, lazex::tags::ERF, DualType){
        xdiff::assign_erf(out, a);
    }

};


template<typename T, int NVARS, int NORDER, xdiff::Layout LY>
struct CustomBinaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>> : public BinaryEvaluator<CustomBinaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>>, xdiff::Dual<T, NVARS, NORDER, LY>>
{

    using DualType = xdiff::Dual<T, NVARS, NORDER, LY>;
    using Base = BinaryEvaluator<CustomBinaryEvaluator<DualType>, DualType>;
    using Base::evaluate;
    using Base::eval_rule;


    LAZEX_EVALUATE_OPER(DualType, a, b, DualType, lazex::tags::PLUS, DualType){
        xdiff::assign_add(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::PLUS /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const F& b){
        xdiff::assign_add(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::PLUS /**/, DualType& out, Pool<DualType> /**/, const F& a, const DualType& b){
        xdiff::assign_add(out, a, b);
    }

    // The order is deduced rather than named, so that Seed is never instantiated for an
    // order this Dual cannot seed (a Seed always carries at least one derivative order).
    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::PLUS /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const xdiff::Seed<T, NVARS, No, LY>& b){
        xdiff::assign_add(out, a, b);
    }

    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::PLUS /**/, DualType& out, Pool<DualType> /**/, const xdiff::Seed<T, NVARS, No, LY>& a, const DualType& b){
        xdiff::assign_add(out, a, b);
    }


    LAZEX_EVALUATE_OPER(DualType, a, b, DualType, lazex::tags::MINUS, DualType){
        xdiff::assign_sub(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MINUS /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const F& b){
        xdiff::assign_sub(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MINUS /**/, DualType& out, Pool<DualType> /**/, const F& a, const DualType& b){
        xdiff::assign_sub(out, a, b);
    }

    // The order is deduced rather than named, so that Seed is never instantiated for an
    // order this Dual cannot seed (a Seed always carries at least one derivative order).
    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MINUS /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const xdiff::Seed<T, NVARS, No, LY>& b){
        xdiff::assign_sub(out, a, b);
    }

    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MINUS /**/, DualType& out, Pool<DualType> /**/, const xdiff::Seed<T, NVARS, No, LY>& a, const DualType& b){
        xdiff::assign_sub(out, a, b);
    }


    // mul
    LAZEX_EVALUATE_OPER(DualType, a, b, DualType, lazex::tags::MUL, DualType){
        xdiff::assign_mul(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MUL /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const F& b){
        xdiff::assign_mul(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MUL /**/, DualType& out, Pool<DualType> /**/, const F& a, const DualType& b){
        xdiff::assign_mul(out, a, b);
    }

    // The order is deduced rather than named, so that Seed is never instantiated for an
    // order this Dual cannot seed (a Seed always carries at least one derivative order).
    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MUL /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const xdiff::Seed<T, NVARS, No, LY>& b){
        xdiff::assign_mul(out, a, b);
    }

    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::MUL /**/, DualType& out, Pool<DualType> /**/, const xdiff::Seed<T, NVARS, No, LY>& a, const DualType& b){
        xdiff::assign_mul(out, a, b);
    }

    // Division
    LAZEX_EVALUATE_OPER(DualType, a, b, DualType, lazex::tags::DIV, DualType){
        xdiff::assign_div(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::DIV /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const F& b){
        xdiff::assign_div(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::DIV /**/, DualType& out, Pool<DualType> /**/, const F& a, const DualType& b){
        xdiff::assign_div(out, a, b);
    }

    // The order is deduced rather than named, so that Seed is never instantiated for an
    // order this Dual cannot seed (a Seed always carries at least one derivative order).
    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::DIV /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const xdiff::Seed<T, NVARS, No, LY>& b){
        xdiff::assign_div(out, a, b);
    }

    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::DIV /**/, DualType& out, Pool<DualType> /**/, const xdiff::Seed<T, NVARS, No, LY>& a, const DualType& b){
        xdiff::assign_div(out, a, b);
    }

    // Power
    LAZEX_EVALUATE_OPER(DualType, a, b, DualType, lazex::tags::POW, DualType){
        xdiff::assign_pow(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::POW /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const F& b){
        xdiff::assign_pow(out, a, b);
    }

    template<arithmetic F>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::POW /**/, DualType& out, Pool<DualType> /**/, const F& a, const DualType& b){
        xdiff::assign_pow(out, a, b);
    }

    // The order is deduced rather than named, so that Seed is never instantiated for an
    // order this Dual cannot seed (a Seed always carries at least one derivative order).
    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::POW /**/, DualType& out, Pool<DualType> /**/, const DualType& a, const xdiff::Seed<T, NVARS, No, LY>& b){
        xdiff::assign_pow(out, a, b);
    }

    template<int No>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::POW /**/, DualType& out, Pool<DualType> /**/, const xdiff::Seed<T, NVARS, No, LY>& a, const DualType& b){
        xdiff::assign_pow(out, a, b);
    }


};


template<typename T, int NVARS, int NORDER, xdiff::Layout LY>
inline bool isfinite(const xdiff::Dual<T, NVARS, NORDER, LY>& x){
    return isfinite(x.value());
}

}; // namespace lazex::detail

namespace std{
template<typename T, int NVARS, int NORDER, xdiff::Layout LY>
class numeric_limits<lazex::detail::LazyType<xdiff::Dual<T, NVARS, NORDER, LY>>> : public numeric_limits<T>{};
}
// A LazyType<Dual> may interact with a plain scalar, and with the Seed standing for a seed
// variable of the same Dual type: a seed meets a lazy gradient whenever a SeedVector element takes
// part in an operation on a nested Dual with a runtime number of variables. Admitting it here lets
// the seed enter the expression graph, instead of binding to the overloads meant for a scalar and
// losing its unit derivative. A Seed is stored by value in the node (lazex::detail::OtherType),
// so the temporary returned by Seed::trimmed() is copied rather than referenced.
template<typename F, typename T, int NVARS, int NORDER, xdiff::Layout LY>
constexpr bool lazex::traits::lazexConvertCondition<F, xdiff::Dual<T, NVARS, NORDER, LY>> =
    std::is_arithmetic_v<std::decay_t<F>>
    || std::is_same_v<std::decay_t<F>, xdiff::Seed<T, NVARS, NORDER, LY>>;


namespace lazy {

using lazex::detail::isfinite;

// Set command for setting default number of variables for lazex::Dual

} // namespace lazy



#endif // XDIFF_LAZY_HPP