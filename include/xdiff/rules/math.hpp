#ifndef XDIFF_RULES_MATH_HPP
#define XDIFF_RULES_MATH_HPP

#include "core.hpp"


namespace xdiff::detail::rules{


template<typename T>
struct Pos : BaseOperand<T, Pos<T>>{

    using Base = BaseOperand<T, Pos<T>>;

    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return arg;
    }

    template<typename A, typename DA>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a) {
        if constexpr (DiffPair<A, DA>::isTrivial()) {
            return 0;
        } else {
            return a.grad;
        }
    }
};


template<typename T>
struct Neg : BaseOperand<T, Neg<T>>{

    using Base = BaseOperand<T, Neg<T>>;

    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return -arg;
    }

    template<typename A, typename DA>
    XDIFF_INLINE_HOST_DEVICE
    static auto diff_rule(const DiffPair<A, DA>& a) {
        if constexpr (DiffPair<A, DA>::isTrivial()) {
            return 0;
        } else {
            return -a.grad;
        }
    }
};

template<typename T>
struct Log : MathFunc<Log<T>, T>{

    using Base = MathFunc<Log<T>, T>;

    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return log(arg);
    }

    /// @brief d(log(f)) = df / f
    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df / f;
    }

};



template<typename T>
struct Log10 : MathFunc<Log10<T>, T>{

    using Base = MathFunc<Log10<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return log10(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df / (f * log(T(10)));
    }
};


template<typename T>
struct Sqrt : MathFunc<Sqrt<T>, T>{

    using Base = MathFunc<Sqrt<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return sqrt(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df / (2 * sqrt(f));
    }
};


template<typename T>
struct Abs : MathFunc<Abs<T>, T>{

    using Base = MathFunc<Abs<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return abs(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df*( f > 0 ? 1 : ( f < 0 ? -1 : 0));
    }

};


template<typename T>
struct Exp : MathFunc<Exp<T>, T>{

    using Base = MathFunc<Exp<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return exp(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df * exp(f);
    }

};


template<typename T>
struct Sin : MathFunc<Sin<T>, T>{

    using Base = MathFunc<Sin<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return sin(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df * cos(f);
    }

};


template<typename T>
struct Cos : MathFunc<Cos<T>, T>{

    using Base = MathFunc<Cos<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return cos(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return -df * sin(f);
    }
};


template<typename T>
struct Tan : MathFunc<Tan<T>, T>{
    
    using Base = MathFunc<Tan<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return tan(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        auto tan_f = tan(f);
        return df * (1 + tan_f * tan_f);
    }
};


template<typename T>
struct Cot : MathFunc<Cot<T>, T>{

    using Base = MathFunc<Cot<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return 1 / tan(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        auto sin_f = sin(f);
        return -df / (sin_f * sin_f);
    }
};


template<typename T>
struct Sec : MathFunc<Sec<T>, T>{
    
    using Base = MathFunc<Sec<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return 1 / cos(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df * tan(f) / cos(f);
    }
};


template<typename T>
struct Csc : MathFunc<Csc<T>, T>{

    using Base = MathFunc<Csc<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return 1 / sin(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        auto sin_f = sin(f);
        return -df / (sin_f * sin_f);
    }
};


template<typename T>
struct ArcSin : MathFunc<ArcSin<T>, T>{

    using Base = MathFunc<ArcSin<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return asin(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df / sqrt(1 - (f * f));
    }
};


template<typename T>
struct ArcCos : MathFunc<ArcCos<T>, T>{
    
    using Base = MathFunc<ArcCos<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return acos(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return -df / sqrt(1 - (f * f));
    }
};


template<typename T>
struct ArcTan : MathFunc<ArcTan<T>, T>{

    using Base = MathFunc<ArcTan<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return atan(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df / (1 + (f * f));
    }
};


template<typename T>
struct ArcCot : MathFunc<ArcCot<T>, T>{

    using Base = MathFunc<ArcCot<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return atan(1 / arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return -df / (1 + f * f);
    }
};


template<typename T>
struct ArcSec : MathFunc<ArcSec<T>, T>{

    using Base = MathFunc<ArcSec<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return acos(1 / arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df / (abs(f) * sqrt((f * f) - 1));
    }
};


template<typename T>
struct ArcCsc : MathFunc<ArcCsc<T>, T>{

    using Base = MathFunc<ArcCsc<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return asin(1 / arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return -df / (abs(f) * sqrt((f * f) - 1));
    }
};


template<typename T>
struct Sinh : MathFunc<Sinh<T>, T>{

    using Base = MathFunc<Sinh<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return sinh(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df * cosh(f);
    }
};


template<typename T>
struct Cosh : MathFunc<Cosh<T>, T>{
    
    using Base = MathFunc<Cosh<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return cosh(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df * sinh(f);
    }
};


template<typename T>
struct Tanh : MathFunc<Tanh<T>, T>{
    
    using Base = MathFunc<Tanh<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return tanh(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        auto th = tanh(f);
        return df * (1 - th * th);
    }
};


template<typename T>
struct Erf : MathFunc<Erf<T>, T>{
    
    using Base = MathFunc<Erf<T>, T>;

    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return erf(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return M_2_SQRTPI * df * exp(-f * f);
    }
};








} // namespace xdiff::detail::rules

#endif // XDIFF_RULES_MATH_HPP