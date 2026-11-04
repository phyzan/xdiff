#pragma once

#include "../dual/dual_decls.hpp" // IWYU pragma: keep


#define XDIFF_LAZY_UNARY(TAG, ASSIGN) \
LAZEX_FORCE_INLINE static void evaluate(lazex::tags::TAG, DualType& out, Pool<DualType> /*workers*/, const DualType& a){ \
    ::xdiff::detail::ASSIGN(out, a); \
}

#define XDIFF_LAZY_OPERANDS(TAG, ASSIGN, LEFT, RIGHT) \
LAZEX_FORCE_INLINE static void evaluate(lazex::tags::TAG, DualType& out, Pool<DualType> /*workers*/, const LEFT& a, const RIGHT& b){ \
    ::xdiff::detail::ASSIGN(out, a, b); \
}

#define XDIFF_LAZY_BINARY(TAG, ASSIGN) \
XDIFF_LAZY_OPERANDS(TAG, ASSIGN, DualType, DualType) \
XDIFF_LAZY_OPERANDS(TAG, ASSIGN, DualType, SeedType) \
XDIFF_LAZY_OPERANDS(TAG, ASSIGN, SeedType, DualType) \
template<::xdiff::detail::converts_to<T> F> \
XDIFF_LAZY_OPERANDS(TAG, ASSIGN, DualType, F) \
template<::xdiff::detail::converts_to<T> F> \
XDIFF_LAZY_OPERANDS(TAG, ASSIGN, F, DualType)


namespace lazex::detail {

template<typename T, typename GD, int Order>
struct CustomUnaryEvaluator<::xdiff::detail::XDIFF_DUAL> : public UnaryEvaluator<CustomUnaryEvaluator<::xdiff::detail::XDIFF_DUAL>, ::xdiff::detail::XDIFF_DUAL>{

    using DualType = ::xdiff::detail::XDIFF_DUAL;
    using Base = UnaryEvaluator<CustomUnaryEvaluator<DualType>, DualType>;
    using Base::eval_rule;
    using Base::evaluate;

    XDIFF_LAZY_UNARY(NEG,   assign_neg)
    XDIFF_LAZY_UNARY(ABS,   assign_abs)
    XDIFF_LAZY_UNARY(SQRT,  assign_sqrt)
    XDIFF_LAZY_UNARY(EXP,   assign_exp)
    XDIFF_LAZY_UNARY(LOG,   assign_log)
    XDIFF_LAZY_UNARY(LOG10, assign_log10)
    XDIFF_LAZY_UNARY(SIN,   assign_sin)
    XDIFF_LAZY_UNARY(COS,   assign_cos)
    XDIFF_LAZY_UNARY(TAN,   assign_tan)
    XDIFF_LAZY_UNARY(COT,   assign_cot)
    XDIFF_LAZY_UNARY(SEC,   assign_sec)
    XDIFF_LAZY_UNARY(CSC,   assign_csc)
    XDIFF_LAZY_UNARY(ASIN,  assign_asin)
    XDIFF_LAZY_UNARY(ACOS,  assign_acos)
    XDIFF_LAZY_UNARY(ATAN,  assign_atan)
    XDIFF_LAZY_UNARY(ASEC,  assign_asec)
    XDIFF_LAZY_UNARY(ACSC,  assign_acsc)
    XDIFF_LAZY_UNARY(SINH,  assign_sinh)
    XDIFF_LAZY_UNARY(COSH,  assign_cosh)
    XDIFF_LAZY_UNARY(TANH,  assign_tanh)
    XDIFF_LAZY_UNARY(ERF,   assign_erf)

    template<int P>
    LAZEX_FORCE_INLINE static void evaluate(lazex::tags::INTEGRAL_POWER<P>, DualType& out, Pool<DualType> /*workers*/, const DualType& a){
        ::xdiff::detail::assign_power<P>(out, a);
    }
};


template<typename T, typename GD, int Order>
struct CustomBinaryEvaluator<::xdiff::detail::XDIFF_DUAL> : public BinaryEvaluator<CustomBinaryEvaluator<::xdiff::detail::XDIFF_DUAL>, ::xdiff::detail::XDIFF_DUAL>{

    using DualType = ::xdiff::detail::XDIFF_DUAL;
    using SeedType = ::xdiff::detail::XDIFF_SEED;
    using Base = BinaryEvaluator<CustomBinaryEvaluator<DualType>, DualType>;
    using Base::evaluate;
    using Base::eval_rule;

    XDIFF_LAZY_BINARY(PLUS,  assign_add)
    XDIFF_LAZY_BINARY(MINUS, assign_sub)
    XDIFF_LAZY_BINARY(MUL,   assign_mul)
    XDIFF_LAZY_BINARY(DIV,   assign_div)
    XDIFF_LAZY_BINARY(POW,   assign_pow)
    XDIFF_LAZY_BINARY(ATAN2, assign_atan2)
};


template<typename T, typename GD, int Order>
inline bool isfinite(const LazyType<::xdiff::detail::XDIFF_DUAL>& x){
    return std::isfinite(get_value(x));
}

} // namespace lazex::detail


namespace std{

template<typename T, typename GD, int Order>
class numeric_limits<lazex::detail::LazyType<::xdiff::detail::XDIFF_DUAL>> : public numeric_limits<T>{};

} // namespace std


// A LazyType<Dual> may interact with a plain scalar, and with the Seed standing for a seed variable of the same Dual
// type. Admitting a Seed here lets it enter the expression graph, where it is stored by value (lazex::detail::OtherType).
template<typename F, typename T, typename GD, int Order>
constexpr bool lazex::traits::lazexConvertCondition<F, ::xdiff::detail::XDIFF_DUAL> =
    ::xdiff::detail::converts_to<F, T> || std::is_same_v<std::decay_t<F>, ::xdiff::detail::XDIFF_SEED>;


#undef XDIFF_LAZY_UNARY
#undef XDIFF_LAZY_OPERANDS
#undef XDIFF_LAZY_BINARY
