#ifndef XDIFF_MATH_DEFS_HPP
#define XDIFF_MATH_DEFS_HPP


#include "../rules.hpp"


namespace xdiff{


namespace detail::operations{

template<typename T>
struct Log10 : MathFunc<Log10<T>, T>{

    using Base = MathFunc<Log10<T>, T>;
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return tan(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return df * (1 + tan(f)*tan(f));
    }
};


template<typename T>
struct Cot : MathFunc<Cot<T>, T>{

    using Base = MathFunc<Cot<T>, T>;
    using Base::optimized_eval;
    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return 1 / tan(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return -df / (sin(f) * sin(f));
    }
};


template<typename T>
struct Sec : MathFunc<Sec<T>, T>{
    
    using Base = MathFunc<Sec<T>, T>;
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
    template<typename A>
    XDIFF_INLINE_HOST_DEVICE
    static auto operation(const A& arg){
        return 1 / sin(arg);
    }

    template<typename F, typename DF>
    XDIFF_INLINE_HOST_DEVICE
    static auto special_diff(const F& f, const DF& df) {
        return -df * cos(f) / (sin(f) * sin(f));
    }
};


template<typename T>
struct ArcSin : MathFunc<ArcSin<T>, T>{

    using Base = MathFunc<ArcSin<T>, T>;
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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
    using Base::optimized_eval;
    
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

} // namespace detail::operations



// ============= Apply the definitions to the public API for the Dual class =================

XDIFF_MATHFUNC_DUAL(abs, Abs)
XDIFF_MATHFUNC_DUAL(log10, Log10)
XDIFF_MATHFUNC_DUAL(sqrt, Sqrt)
XDIFF_MATHFUNC_DUAL(exp, Exp)
XDIFF_MATHFUNC_DUAL(sin, Sin)
XDIFF_MATHFUNC_DUAL(cos, Cos)
XDIFF_MATHFUNC_DUAL(tan, Tan)
XDIFF_MATHFUNC_DUAL(cot, Cot)
XDIFF_MATHFUNC_DUAL(sec, Sec)
XDIFF_MATHFUNC_DUAL(csc, Csc)
XDIFF_MATHFUNC_DUAL(asin, ArcSin)
XDIFF_MATHFUNC_DUAL(acos, ArcCos)
XDIFF_MATHFUNC_DUAL(atan, ArcTan)
XDIFF_MATHFUNC_DUAL(acot, ArcCot)
XDIFF_MATHFUNC_DUAL(asec, ArcSec)
XDIFF_MATHFUNC_DUAL(acsc, ArcCsc)
XDIFF_MATHFUNC_DUAL(sinh, Sinh)
XDIFF_MATHFUNC_DUAL(cosh, Cosh)
XDIFF_MATHFUNC_DUAL(tanh, Tanh)
XDIFF_MATHFUNC_DUAL(erf, Erf)


} // namespace xdiff



#endif // XDIFF_MATH_DEFS_HPP