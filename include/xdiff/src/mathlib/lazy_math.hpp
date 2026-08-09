#ifndef XDIFF_LAZY_MATH_HPP
#define XDIFF_LAZY_MATH_HPP

#include "../expr.hpp"
#include "defs.hpp"
#include "../basic_expr.hpp"

namespace xdiff::detail{

// ================ Expression Classes ==========================

template<typename T, typename Arg>
class Log10Expr : public Unary<Log10Expr<T, Arg>, T, Arg>, public operations::Log10<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    Log10Expr(Arg arg) : Unary<Log10Expr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Log10<T>::diff_rule;
    using operations::Log10<T>::operation;
};

template<typename T, typename Arg>
class SqrtExpr : public Unary<SqrtExpr<T, Arg>, T, Arg>, public operations::Sqrt<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    SqrtExpr(Arg arg) : Unary<SqrtExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Sqrt<T>::diff_rule;
    using operations::Sqrt<T>::operation;
};

template<typename T, typename Arg>
class AbsExpr : public Unary<AbsExpr<T, Arg>, T, Arg>, public operations::Abs<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    AbsExpr(Arg arg) : Unary<AbsExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Abs<T>::diff_rule;
    using operations::Abs<T>::operation;
};

template<typename T, typename Arg>
class ExpExpr : public Unary<ExpExpr<T, Arg>, T, Arg>, public operations::Exp<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ExpExpr(Arg arg) : Unary<ExpExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Exp<T>::diff_rule;
    using operations::Exp<T>::operation;
};

template<typename T, typename Arg>
class SinExpr : public Unary<SinExpr<T, Arg>, T, Arg>, public operations::Sin<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    SinExpr(Arg arg) : Unary<SinExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Sin<T>::diff_rule;
    using operations::Sin<T>::operation;
};

template<typename T, typename Arg>
class CosExpr : public Unary<CosExpr<T, Arg>, T, Arg>, public operations::Cos<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    CosExpr(Arg arg) : Unary<CosExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Cos<T>::diff_rule;
    using operations::Cos<T>::operation;
};

template<typename T, typename Arg>
class TanExpr : public Unary<TanExpr<T, Arg>, T, Arg>, public operations::Tan<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    TanExpr(Arg arg) : Unary<TanExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Tan<T>::diff_rule;
    using operations::Tan<T>::operation;
};

template<typename T, typename Arg>
class CotExpr : public Unary<CotExpr<T, Arg>, T, Arg>, public operations::Cot<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    CotExpr(Arg arg) : Unary<CotExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Cot<T>::diff_rule;
    using operations::Cot<T>::operation;
};

template<typename T, typename Arg>
class SecExpr : public Unary<SecExpr<T, Arg>, T, Arg>, public operations::Sec<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    SecExpr(Arg arg) : Unary<SecExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Sec<T>::diff_rule;
    using operations::Sec<T>::operation;
};

template<typename T, typename Arg>
class CscExpr : public Unary<CscExpr<T, Arg>, T, Arg>, public operations::Csc<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    CscExpr(Arg arg) : Unary<CscExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Csc<T>::diff_rule;
    using operations::Csc<T>::operation;
};

template<typename T, typename Arg>
class ArcSinExpr : public Unary<ArcSinExpr<T, Arg>, T, Arg>, public operations::ArcSin<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ArcSinExpr(Arg arg) : Unary<ArcSinExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::ArcSin<T>::diff_rule;
    using operations::ArcSin<T>::operation;
};

template<typename T, typename Arg>
class ArcCosExpr : public Unary<ArcCosExpr<T, Arg>, T, Arg>, public operations::ArcCos<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ArcCosExpr(Arg arg) : Unary<ArcCosExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::ArcCos<T>::diff_rule;
    using operations::ArcCos<T>::operation;
};

template<typename T, typename Arg>
class ArcTanExpr : public Unary<ArcTanExpr<T, Arg>, T, Arg>, public operations::ArcTan<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ArcTanExpr(Arg arg) : Unary<ArcTanExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::ArcTan<T>::diff_rule;
    using operations::ArcTan<T>::operation;
};

template<typename T, typename Arg>
class ArcCotExpr : public Unary<ArcCotExpr<T, Arg>, T, Arg>, public operations::ArcCot<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ArcCotExpr(Arg arg) : Unary<ArcCotExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::ArcCot<T>::diff_rule;
    using operations::ArcCot<T>::operation;
};

template<typename T, typename Arg>
class ArcSecExpr : public Unary<ArcSecExpr<T, Arg>, T, Arg>, public operations::ArcSec<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ArcSecExpr(Arg arg) : Unary<ArcSecExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::ArcSec<T>::diff_rule;
    using operations::ArcSec<T>::operation;
};

template<typename T, typename Arg>
class ArcCscExpr : public Unary<ArcCscExpr<T, Arg>, T, Arg>, public operations::ArcCsc<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ArcCscExpr(Arg arg) : Unary<ArcCscExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::ArcCsc<T>::diff_rule;
    using operations::ArcCsc<T>::operation;
};

template<typename T, typename Arg>
class SinhExpr : public Unary<SinhExpr<T, Arg>, T, Arg>, public operations::Sinh<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    SinhExpr(Arg arg) : Unary<SinhExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Sinh<T>::diff_rule;
    using operations::Sinh<T>::operation;
};

template<typename T, typename Arg>
class CoshExpr : public Unary<CoshExpr<T, Arg>, T, Arg>, public operations::Cosh<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    CoshExpr(Arg arg) : Unary<CoshExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Cosh<T>::diff_rule;
    using operations::Cosh<T>::operation;
};

template<typename T, typename Arg>
class TanhExpr : public Unary<TanhExpr<T, Arg>, T, Arg>, public operations::Tanh<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    TanhExpr(Arg arg) : Unary<TanhExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Tanh<T>::diff_rule;
    using operations::Tanh<T>::operation;
};

template<typename T, typename Arg>
class ErfExpr : public Unary<ErfExpr<T, Arg>, T, Arg>, public operations::Erf<T>{
public:
    XDIFF_INLINE_HOST_DEVICE
    ErfExpr(Arg arg) : Unary<ErfExpr<T, Arg>, T, Arg>(std::move(arg)){}
    using operations::Erf<T>::diff_rule;
    using operations::Erf<T>::operation;
};


// ================ Make Functions ==========================

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_log10(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(log10(a.value()));
    } else {
        return Log10Expr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_sqrt(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(sqrt(a.value()));
    } else {
        return SqrtExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_abs(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(abs(a.value()));
    } else {
        return AbsExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_exp(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(exp(a.value()));
    } else if constexpr (traits::isZero<Ta, T>) {
        return One<T>();
    } else {
        return ExpExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_sin(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(sin(a.value()));
    } else {
        return SinExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_cos(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(cos(a.value()));
    } else {
        return CosExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_tan(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(tan(a.value()));
    } else {
        return TanExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_cot(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(T(1) / tan(a.value()));
    } else {
        return CotExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_sec(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(T(1) / cos(a.value()));
    } else {
        return SecExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_csc(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(T(1) / sin(a.value()));
    } else {
        return CscExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_asin(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(asin(a.value()));
    } else {
        return ArcSinExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_acos(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(acos(a.value()));
    } else {
        return ArcCosExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_atan(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(atan(a.value()));
    } else {
        return ArcTanExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_acot(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(atan(T(1) / a.value()));
    } else {
        return ArcCotExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_asec(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(acos(T(1) / a.value()));
    } else {
        return ArcSecExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_acsc(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(asin(T(1) / a.value()));
    } else {
        return ArcCscExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_sinh(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(sinh(a.value()));
    } else {
        return SinhExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_cosh(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(cosh(a.value()));
    } else {
        return CoshExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_tanh(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(tanh(a.value()));
    } else {
        return TanhExpr<T, Ta>(std::move(a));
    }
}

template<typename T, typename F>
XDIFF_INLINE_HOST_DEVICE
auto make_erf(F&& arg){
    auto a = make_expr<T>(std::forward<F>(arg));
    using Ta = decltype(a);
    if constexpr (traits::isConstant<Ta, T>) {
        return Number<T>(erf(a.value()));
    } else {
        return ErfExpr<T, Ta>(std::move(a));
    }
}


}; // namespace xdiff::detail


namespace xdiff{

// =================== API for math expressions =====================

XDIFF_MAKE_EXPR_UNARY_OPERATOR(log10, make_log10)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(sqrt, make_sqrt)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(abs, make_abs)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(exp, make_exp)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(sin, make_sin)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(cos, make_cos)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(tan, make_tan)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(cot, make_cot)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(sec, make_sec)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(csc, make_csc)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(asin, make_asin)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(acos, make_acos)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(atan, make_atan)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(acot, make_acot)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(asec, make_asec)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(acsc, make_acsc)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(sinh, make_sinh)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(cosh, make_cosh)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(tanh, make_tanh)
XDIFF_MAKE_EXPR_UNARY_OPERATOR(erf, make_erf)


} // namespace xdiff



#endif // XDIFF_LAZY_MATH_HPP
