#ifndef XDIFF_LAZY_HPP
#define XDIFF_LAZY_HPP


#include "dual.hpp" // IWYU pragma: keep
#include <lazy/lazy.hpp>



namespace lazy::detail {

template<typename T>
concept arithmetic = std::is_arithmetic_v<T>;
/**
 * @brief Unary function specialisations for `mpfr::mpreal`.
 *
 * Overrides `CustomUnaryRules<mpfr::mpreal>::evaluate` for `NEG`, `ABS`, and `SQRT`
 * using the corresponding raw MPFR C library functions, which avoid any overhead from
 * the `mpfr::mpreal` operator overloads and respect the global rounding mode.
 */
template<typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
struct CustomUnaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>> : public UnaryEvaluator<CustomUnaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>>, xdiff::Dual<T, NVARS, NORDER, LY>>{

    using DualType = xdiff::Dual<T, NVARS, NORDER, LY>;
    using Base = UnaryEvaluator<CustomUnaryEvaluator<DualType>, DualType>;
    using Base::eval_rule;
    using Base::evaluate;

    // neg
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::NEG, DualType){
        xdiff::assign_neg(out, a);
    }

    // abs
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::ABS, DualType){
        xdiff::assign_abs(out, a);
    }

    // sqrt
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::SQRT, DualType){
        xdiff::assign_sqrt(out, a);
    }

    // exp
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::EXP, DualType){
        xdiff::assign_exp(out, a);
    }

    // log
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::LOG, DualType){
        xdiff::assign_log(out, a);
    }

    // sin
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::SIN, DualType){
        xdiff::assign_sin(out, a);
    }

    // cos
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::COS, DualType){
        xdiff::assign_cos(out, a);
    }

    // tan
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::TAN, DualType){
        xdiff::assign_tan(out, a);
    }

    // cot
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::COT, DualType){
        xdiff::assign_cot(out, a);
    }

    // sec
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::SEC, DualType){
        xdiff::assign_sec(out, a);
    }

    // csc
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::CSC, DualType){
        xdiff::assign_csc(out, a);
    }

    // asin
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::ASIN, DualType){
        xdiff::assign_asin(out, a);
    }

    // acos
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::ACOS, DualType){
        xdiff::assign_acos(out, a);
    }

    // atan
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::ATAN, DualType){
        xdiff::assign_atan(out, a);
    }

    // sinh
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::SINH, DualType){
        xdiff::assign_sinh(out, a);
    }

    // cosh
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::COSH, DualType){
        xdiff::assign_cosh(out, a);
    }

    // tanh
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::TANH, DualType){
        xdiff::assign_tanh(out, a);
    }

    // erf
    LAZY_EVALUATE_FUNC(DualType, a, lazy::tags::ERF, DualType){
        xdiff::assign_erf(out, a);
    }

};


template<typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
struct CustomBinaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>> : public BinaryEvaluator<CustomBinaryEvaluator<xdiff::Dual<T, NVARS, NORDER, LY>>, xdiff::Dual<T, NVARS, NORDER, LY>>
{

    using DualType = xdiff::Dual<T, NVARS, NORDER, LY>;
    using Base = BinaryEvaluator<CustomBinaryEvaluator<DualType>, DualType>;
    using Base::evaluate;
    using Base::eval_rule;


    LAZY_EVALUATE_OPER(DualType, a, b, DualType, lazy::tags::PLUS, DualType){
        xdiff::assign_add(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::PLUS /**/, DualType& out, const DualType& a, const F& b){
        xdiff::assign_add(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::PLUS /**/, DualType& out, const F& a, const DualType& b){
        xdiff::assign_add(out, a, b);
    }


    LAZY_EVALUATE_OPER(DualType, a, b, DualType, lazy::tags::MINUS, DualType){
        xdiff::assign_sub(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::MINUS /**/, DualType& out, const DualType& a, const F& b){
        xdiff::assign_sub(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::MINUS /**/, DualType& out, const F& a, const DualType& b){
        xdiff::assign_sub(out, a, b);
    }


    // mul
    LAZY_EVALUATE_OPER(DualType, a, b, DualType, lazy::tags::MUL, DualType){
        xdiff::assign_mul(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::MUL /**/, DualType& out, const DualType& a, const F& b){
        xdiff::assign_mul(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::MUL /**/, DualType& out, const F& a, const DualType& b){
        xdiff::assign_mul(out, a, b);
    }

    // Division
    LAZY_EVALUATE_OPER(DualType, a, b, DualType, lazy::tags::DIV, DualType){
        xdiff::assign_div(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::DIV /**/, DualType& out, const DualType& a, const F& b){
        xdiff::assign_div(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::DIV /**/, DualType& out, const F& a, const DualType& b){
        xdiff::assign_div(out, a, b);
    }

    // Power
    LAZY_EVALUATE_OPER(DualType, a, b, DualType, lazy::tags::POW, DualType){
        xdiff::assign_pow(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::POW /**/, DualType& out, const DualType& a, const F& b){
        xdiff::assign_pow(out, a, b);
    }

    template<arithmetic F>
    LAZY_FORCE_INLINE static void evaluate(lazy::tags::POW /**/, DualType& out, const F& a, const DualType& b){
        xdiff::assign_pow(out, a, b);
    }


};


template<typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
inline bool isfinite(const xdiff::Dual<T, NVARS, NORDER, LY>& x){
    return isfinite(x.value());
}

}; // namespace lazy::detail

namespace std{
template<typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
class numeric_limits<lazy::detail::LazyType<xdiff::Dual<T, NVARS, NORDER, LY>>> : public numeric_limits<T>{};
}
template<typename F, typename T, size_t NVARS, size_t NORDER, xdiff::Layout LY>
constexpr bool lazy::traits::lazyConvertCondition<F, xdiff::Dual<T, NVARS, NORDER, LY>> = std::is_arithmetic_v<std::decay_t<F>>;


namespace lazy {

using lazy::detail::isfinite;

// Set command for setting default number of variables for lazy::Dual

} // namespace lazy



#endif // XDIFF_LAZY_HPP