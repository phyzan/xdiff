#ifndef XDIFF_DUAL_IMPL_HPP
#define XDIFF_DUAL_IMPL_HPP

#include "dual_decls.hpp"
#include <type_traits>


namespace xdiff::detail{


// ------------- Core Dual container ------------
template<typename T, typename GD>
class dual_template {
public:
    ~dual_template() = default;
protected:
    using value_type = T;
    using state_type = State<T, GD>;

    XDIFF_INLINE_HOST_DEVICE
    dual_template() = default;
    XDIFF_INLINE_HOST_DEVICE
    dual_template(const dual_template& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    dual_template(dual_template&& other) noexcept = default;
    XDIFF_INLINE_HOST_DEVICE
    dual_template& operator=(const dual_template& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    dual_template& operator=(dual_template&& other) noexcept = default;

    operator T() const = delete;

    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    explicit dual_template(U&& value) : value_(std::forward<U>(value)), grad_{} {}

    template<typename U, typename G>
    XDIFF_INLINE_HOST_DEVICE
    explicit dual_template(U&& value, G&& grad) :
        value_(std::forward<U>(value)),
        grad_(std::forward<G>(grad)) {}

    // Assign from value
    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    dual_template& operator=(U&& v){
        value_ = std::forward<U>(v);
        GD::for_each([](T& d){ d = 0; }, grad_);// a constant has no derivative
        return *this;
    }

    // Accessors
    const T& value() const {
        return value_;
    }

    const state_type& grad() const {
        return grad_;
    }

    value_type  value_;
    state_type   grad_;
};
// ----------------------------------------------


template<typename T, typename GD, int Order>
class Dual : private dual_type_getter<T, GD, Order>::type {
    static_assert(Order > 0, "Differentiation order must be > 0");
    using Base = detail::dual_type_getter<T, GD, Order>::type;
public:
    using value_type = T;
    using state_type = Base::state_type;

    // level relative to `Order`
    // Returns the scalar type the value and
    // the elements of state_type are made of
    template<int level>
    using Scalar = std::conditional_t<
        (Order+level-1 > 0),
        Dual<T, GD, Order+level-1>,
        std::conditional_t<
            (Order+level-1 == 0),
            T, void>>;
            
    XDIFF_INLINE_HOST_DEVICE
    Dual() = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual(const Dual& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual(Dual&& other) noexcept = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Dual& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(Dual&& other) noexcept = default;
    ~Dual() = default;

    /*
    Explicitly stated. Conversion to the scalar type
    will lose derivative information and can result in
    bad overload resolution of operators.
    */
    operator T() const = delete;

    // ------------- Construct from scalar --------------
    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(U&& value) : Base(std::forward<U>(value)) {}
    // --------------------------------------------------

    // ----- Construct from scalar + gradient -----------
    template<converts_to<T> U, converts_to<state_type> G>
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(U&& value, G&& grad) :
        Base(std::forward<U>(value), std::forward<G>(grad)) {}
    // --------------------------------------------------

    // -------- Construct from lower order Dual ---------
    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(const Dual<T, GD, N>& lower) requires (N < Order) :
        Base(lower) {
            this->resize_grad_like(lower);
            this->fill_gradient();
        }
    
    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(Dual<T, GD, N>&& lower) requires (N < Order) :
        Base(std::move(lower)) {
            this->resize_grad_like(this->value_);
            this->fill_gradient();
        }
    // --------------------------------------------------


    // ------------ Resize-free assignments -------------
    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& assign(U&& v){
        return *this = v;
    }

    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& assign(const Seed<T, GD, N>& seed){
        if constexpr (Order == 1){
            this->value_ = seed.value();
        } else {
            this->value_.assign(seed);
        }
        GD::for_each([&seed](Scalar<0>& d, const T& s){
            d = static_cast<int>(&s == &seed.value());
        }, this->grad_, seed.state());
        return *this;
    }

    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& assign(const Dual<T, GD, N>& lower) requires (N < Order){
        //Do NOT call Base, it performs unnecesary work calling GD::for_each
        this->value_.assign(lower);
        this->fill_gradient();
        return *this;
    }

    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& assign(Dual<T, GD, N>&& lower) requires (N < Order){
        //Do NOT call Base, it performs unnecesary work calling GD::for_each
        this->value_.assign(std::move(lower));
        this->fill_gradient();
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& assign(const Dual& other){
        // for recursive fallbacks.
        return *this = other;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& assign(Dual&& other){
        // for recursive fallbacks.
        return *this = std::move(other);
    }
    // --------------------------------------------------


    // ----------- Assign from scalar value -------------
    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(U&& v){
        Base::operator=(std::forward<U>(v));
        return *this;
    }
    // --------------------------------------------------

    // ----------- Assign from Seed -------------
    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Seed<T, GD, N>& seed){
        this->resize_like(seed.state());
        return this->assign(seed);
    }
    // --------------------------------------------------

    // --------- Assign from lower order Dual -----------
    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Dual<T, GD, N>& lower) requires (N < Order){
        this->resize_grad_like(lower);
        this->value_ = lower;
        this->fill_gradient();
        return *this;
    }

    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(Dual<T, GD, N>&& lower) requires (N < Order){
        this->resize_grad_like(lower);
        this->value_ = std::move(lower);
        this->fill_gradient();
        return *this;
    }
    // --------------------------------------------------

    // ---------------- Dual += other -------------------
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const Dual& other){
        this->value_ += other.value_;
        GD::for_each([](Scalar<0>& d_this, const Scalar<0>& d_other){
            d_this += d_other;
        }, this->grad_, other.grad());
        return *this;
    }

    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(U&& v){
        this->value_ += std::forward<U>(v);
        return *this;
    }
    // --------------------------------------------------

    // ---------------- Dual -= other -------------------
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const Dual& other){
        this->value_ -= other.value_;
        GD::for_each([](Scalar<0>& d_this, const Scalar<0>& d_other){
            d_this -= d_other;
        }, this->grad_, other.grad_);
        return *this;
    }

    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(U&& v){
        this->value_ -= std::forward<U>(v);
        return *this;
    }
    // --------------------------------------------------

    // ---------------- Dual *= other -------------------
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const Dual& other){
        GD::for_each([&](Scalar<0>& d_this, const Scalar<0>& d_other){
            d_this = rules::Mul<T>::diff_rule(
                DiffPair{this->value_, d_this},
                DiffPair{other.value_, d_other}
            );
        }, this->grad_, other.grad_);
        this->value_ *= other.value_;
        return *this;
    }
    
    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(U&& v){
        // scaling by a constant scales every derivative
        GD::for_each([&](Scalar<0>& d){ d *= v; }, this->grad_);
        this->value_ *= v;
        return *this;
    }
    // --------------------------------------------------

    // ---------------- Dual /= other -------------------
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const Dual& other){
        GD::for_each([&](Scalar<0>& d_this, const Scalar<0>& d_other){
            d_this = rules::Div<T>::diff_rule(
                DiffPair{this->value_, d_this},
                DiffPair{other.value_, d_other}
            );
        }, this->grad_, other.grad_);
        this->value_ /= other.value_;
        return *this;
    }

    template<converts_to<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(U&& v){
        GD::for_each([&](Scalar<0>& d){
            d /= v;
        }, this->grad_);
        this->value_ /= v;
        return *this;
    }
    // --------------------------------------------------

    XDIFF_INLINE_HOST_DEVICE
    Dual& set_zero_grad(){
        GD::for_each([](Scalar<0>& d){
            d = 0;
        }, this->grad_);
        return *this;
    }

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& resize_like(const GD::template state_type<U>& proto){
        GD::resize_like(this->grad_, proto);

        // For deep resizing
        if constexpr (Order > 1){
            this->value_.resize_like(proto);
            GD::for_each([&proto](Scalar<0>& d){
                d.resize_like(proto);
            }, this->grad_);
        }
        return *this;
    }

    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& resize_like(const Dual<T, GD, N>& proto){
        return this->resize_like(proto.grad());     
    }

    // -------------------- Accessors -------------------
    const T& value() const {
        if constexpr (Order == 1){
            return this->value_;
        } else {
            return this->value_.value();
        }
    }

    const Scalar<0>& full_value() const {
        // Could be T or Dual<T, GR, Order-1>
        return this->value_;
    }

    const state_type& grad() const {
        return this->grad_;
    }
    // --------------------------------------------------

private:
    friend struct dual_operator_handler;

    template<typename A, typename B, int N>
    friend class Seed;

    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual& resize_grad_like(const Dual<T, GD, N>& proto){
        GD::resize_like(this->grad_, proto.grad());
        if constexpr (Order > 1){
            GD::for_each([&proto](Scalar<0>& d){
                d.resize_like(proto);
            }, this->grad_); 
        }
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    void fill_gradient(){
        GD::for_each([](Scalar<0>& d, const Scalar<-1>& dv){
            if constexpr (Order == 1){
                d = dv;
            } else {
                d.assign(dv);
            }
        }, this->grad_, this->value_.grad());
    }
};



template<typename T, typename GD, int Order>
class Seed{
public:
    using state_type = State<T, GD>;
    using dual_type = Dual<T, GD, Order>;

    XDIFF_INLINE_HOST_DEVICE
    Seed() = default;

    XDIFF_INLINE_HOST_DEVICE
    Seed(const T& value, const state_type& state) : value_(&value), state_(&state) {}

    XDIFF_INLINE_HOST_DEVICE
    Seed(const T& value, state_type&& state) = delete;

    XDIFF_INLINE_HOST_DEVICE
    Seed(T&& value, const state_type& state) = delete;

    operator T() = delete;

    /// @brief The element this seed stands for.
    XDIFF_INLINE_HOST_DEVICE
    const T& value() const {
        return *value_;
    }

    /// @brief The state this seed stands for.
    XDIFF_INLINE_HOST_DEVICE
    const state_type& state() const {
        return *state_;
    }

    /// @brief The Dual this seed stands for.
    XDIFF_INLINE_HOST_DEVICE
    dual_type dual() const {
        return make<Order>();
    }

    XDIFF_INLINE_HOST_DEVICE
    operator dual_type() const {
        return dual();
    }

private:

    template<int N>
    XDIFF_INLINE_HOST_DEVICE
    Dual<T, GD, N> make() const {
        Dual<T, GD, N> out;
        out = *this;
        return out;
    }

    const T* value_ = nullptr;
    const state_type* state_ = nullptr; // container where *value_ lives in
};


struct dual_operator_handler {

    template<typename Rule, typename T, typename GD, int Order, typename... A>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, GD, Order>& assign(Dual<T, GD, Order>& out, const A&... args){
        // The gradient goes first, since it still needs the old value when "out" aliases an operand.
        GD::for_each([&](auto& d_out, const auto&... d){
            d_out = Rule::diff_rule(pair_of(args, d, out)...);
        }, out.grad_, grad_of(args, out)...);
        out.value_ = Rule::operation(value_of(args, out)...);
        return out;
    }

    template<typename T, typename GD, int Order>
    XDIFF_INLINE_HOST_DEVICE
    static Dual<T, GD, Order> empty_like(const Dual<T, GD, Order>& proto){
        Dual<T, GD, Order> out;
        out.resize_like(proto);
        return out;
    }

private:

    template<typename T, typename GD, int Order>
    XDIFF_INLINE_HOST_DEVICE
    static const auto& value_of(const Dual<T, GD, Order>& x, const Dual<T, GD, Order>& /*out*/){ return x.full_value(); }

    template<typename F, typename D>
    XDIFF_INLINE_HOST_DEVICE
    static const F& value_of(const F& x, const D& /*out*/){ return x; }

    // d is a component of x's gradient (a Dual one order lower, or T), and x.full_value() is the value stored at that
    // level, which is not Dual::value_type (always T) above order 1.
    template<typename T, typename GD, int Order, typename D>
    XDIFF_INLINE_HOST_DEVICE
    static auto pair_of(const Dual<T, GD, Order>& x, const D& d, const Dual<T, GD, Order>& /*out*/){
        return DiffPair{x.full_value(), d};
    }

    template<typename F, typename D, typename O>
    XDIFF_INLINE_HOST_DEVICE
    static DiffPair<const F&, ZeroValue> pair_of(const F& x, const D& /*d*/, const O& /*out*/){
        return {x, {}};
    }

    template<typename T, typename GD, int Order>
    XDIFF_INLINE_HOST_DEVICE
    static const auto& grad_of(const Dual<T, GD, Order>& x, const Dual<T, GD, Order>& /*out*/){
        return x.grad();
    }

    template<typename F, typename D>
    XDIFF_INLINE_HOST_DEVICE
    static const auto& grad_of(const F& /*x*/, const D& out){
        return out.grad();
    }
};


template<typename T, typename GD, int Order>
SeedState<T, GD, Order> make_seeds(const State<T, GD>& state){
    SeedState<T, GD, Order> seeds{};
    GD::resize_like(seeds, state);
    GD::for_each([&state](Seed<T, GD, Order>& seed, const T& x){
        seed = Seed<T, GD, Order>(x, state);
    }, seeds, state);
    return seeds;
}

template<size_t NFACTORS, typename F, typename... U>
XDIFF_INLINE_HOST_DEVICE
auto power_fold(const F& f, const U&... copies){
    if constexpr (sizeof...(copies) < NFACTORS){
        return power_fold<NFACTORS>(f, copies..., f);
    } else {
        return (copies*...);
    }
}



#define XDIFF_DEFINE_UNARY(OPERATOR, ASSIGN, RULE) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_DUAL& arg){ \
    return dual_operator_handler::assign<RULE<T>>(out, arg); \
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_SEED& arg){\
    return ASSIGN(out, arg.dual());\
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL OPERATOR(const XDIFF_DUAL& arg){ \
    XDIFF_DUAL out = dual_operator_handler::empty_like(arg);\
    ASSIGN(out, arg);\
    return out;\
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL OPERATOR(const XDIFF_SEED& arg){ \
    return OPERATOR(arg.dual());\
}\




#define XDIFF_DEFINE_ASSIGN_BINARY(ASSIGN, RULE) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_DUAL& a, const XDIFF_DUAL& b){ \
    return dual_operator_handler::assign<RULE<T>>(out, a, b); \
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_DUAL& a, const XDIFF_SEED& b){ \
    return ASSIGN(out, a, b.dual()); \
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_SEED& a, const XDIFF_DUAL& b){ \
    return ASSIGN(out, a.dual(), b); \
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_SEED& a, const XDIFF_SEED& b){ \
    return ASSIGN(out, a.dual(), b.dual()); \
}\
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_DUAL& a, const F& b){ \
    return dual_operator_handler::assign<RULE<T>>(out, a, b); \
}\
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const F& a, const XDIFF_DUAL& b){ \
    return dual_operator_handler::assign<RULE<T>>(out, a, b); \
}\
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const XDIFF_SEED& a, const F& b){ \
    return ASSIGN(out, a.dual(), b); \
}\
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& out, const F& a, const XDIFF_SEED& b){ \
    return ASSIGN(out, a, b.dual()); \
}\




// A new result gets its gradient shaped like that of a Dual operand.
#define XDIFF_DEFINE_BINARY_OPERATOR(NAME, ASSIGN) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_DUAL& a, const XDIFF_DUAL& b){ \
    XDIFF_DUAL out = dual_operator_handler::empty_like(a); \
    ASSIGN(out, a, b); \
    return out; \
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_DUAL& a, const XDIFF_SEED& b){ \
    return NAME(a, b.dual()); \
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_SEED& a, const XDIFF_DUAL& b){ \
    return NAME(a.dual(), b); \
}\
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_SEED& a, const XDIFF_SEED& b){ \
    return NAME(a.dual(), b.dual()); \
}\
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_DUAL& a, const F& b){\
    XDIFF_DUAL out = dual_operator_handler::empty_like(a); \
    ASSIGN(out, a, b); \
    return out; \
}\
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const F& a, const XDIFF_DUAL& b){ \
    XDIFF_DUAL out = dual_operator_handler::empty_like(b); \
    ASSIGN(out, a, b); \
    return out; \
}\
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_SEED& a, const F& b){ \
    return NAME(a.dual(), b); \
} \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const F& a, const XDIFF_SEED& b){ \
    return NAME(a, b.dual()); \
}\


#define XDIFF_DEFINE_BINARY_PACK(NAME, ASSIGN, RULE) \
XDIFF_DEFINE_ASSIGN_BINARY(ASSIGN, RULE) \
XDIFF_DEFINE_BINARY_OPERATOR(NAME, ASSIGN)


// Comparisons look at the value alone, as they do for every other Dual: a seed's own element, without building its Dual.
#define XDIFF_DEFINE_COMPARISON(OP) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_DUAL& a, const XDIFF_DUAL& b){ return a.value() OP b.value(); } \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_DUAL& a, const XDIFF_SEED& b){ return a.value() OP b.value(); } \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_SEED& a, const XDIFF_DUAL& b){ return a.value() OP b.value(); } \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_SEED& a, const XDIFF_SEED& b){ return a.value() OP b.value(); } \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_DUAL& a, const F& b){ return a.value() OP b; } \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const F& a, const XDIFF_DUAL& b){ return a OP b.value(); } \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_SEED& a, const F& b){ return a.value() OP b; } \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const F& a, const XDIFF_SEED& b){ return a OP b.value(); }



XDIFF_DEFINE_UNARY(operator+,  assign_pos,   rules::Pos)
XDIFF_DEFINE_UNARY(operator-,  assign_neg,   rules::Neg)
XDIFF_DEFINE_UNARY(abs,        assign_abs,   rules::Abs)
XDIFF_DEFINE_UNARY(log,        assign_log,   rules::Log)
XDIFF_DEFINE_UNARY(log10,      assign_log10, rules::Log10)
XDIFF_DEFINE_UNARY(exp,        assign_exp,   rules::Exp)
XDIFF_DEFINE_UNARY(sqrt,       assign_sqrt,  rules::Sqrt)
XDIFF_DEFINE_UNARY(sin,        assign_sin,   rules::Sin)
XDIFF_DEFINE_UNARY(cos,        assign_cos,   rules::Cos)
XDIFF_DEFINE_UNARY(tan,        assign_tan,   rules::Tan)
XDIFF_DEFINE_UNARY(cot,        assign_cot,   rules::Cot)
XDIFF_DEFINE_UNARY(sec,        assign_sec,   rules::Sec)
XDIFF_DEFINE_UNARY(csc,        assign_csc,   rules::Csc)
XDIFF_DEFINE_UNARY(asin,       assign_asin,  rules::ArcSin)
XDIFF_DEFINE_UNARY(acos,       assign_acos,  rules::ArcCos)
XDIFF_DEFINE_UNARY(atan,       assign_atan,  rules::ArcTan)
XDIFF_DEFINE_UNARY(asec,       assign_asec,  rules::ArcSec)
XDIFF_DEFINE_UNARY(acsc,       assign_acsc,  rules::ArcCsc)
XDIFF_DEFINE_UNARY(sinh,       assign_sinh,  rules::Sinh)
XDIFF_DEFINE_UNARY(cosh,       assign_cosh,  rules::Cosh)
XDIFF_DEFINE_UNARY(tanh,       assign_tanh,  rules::Tanh)
XDIFF_DEFINE_UNARY(erf,        assign_erf,   rules::Erf)


// power<P>(f), with the integer exponent fixed at compile time
template<int P, typename T, typename GD, int Order>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_power(XDIFF_DUAL& out, const XDIFF_DUAL& arg){
    return dual_operator_handler::assign<rules::Power<T, P>>(out, arg);
}

template<int P, typename T, typename GD, int Order>
XDIFF_HOST_DEVICE XDIFF_DUAL& assign_power(XDIFF_DUAL& out, const XDIFF_SEED& arg){
    return assign_power<P>(out, arg.dual());
}

template<int P, typename T, typename GD, int Order>
XDIFF_HOST_DEVICE auto power(const XDIFF_DUAL& arg){
    XDIFF_DUAL out = dual_operator_handler::empty_like(arg);
    assign_power<P>(out, arg);
    return out;
}

template<int P, typename T, typename GD, int Order>
XDIFF_HOST_DEVICE auto power(const XDIFF_SEED& arg){
    return power<P>(arg.dual());
}


XDIFF_DEFINE_BINARY_PACK(operator+, assign_add, rules::Add)
XDIFF_DEFINE_BINARY_PACK(operator-, assign_sub, rules::Sub)
XDIFF_DEFINE_BINARY_PACK(operator*, assign_mul, rules::Mul)
XDIFF_DEFINE_BINARY_PACK(operator/, assign_div, rules::Div)
XDIFF_DEFINE_BINARY_PACK(pow,       assign_pow, rules::Pow)

XDIFF_DEFINE_COMPARISON(==)
XDIFF_DEFINE_COMPARISON(!=)
XDIFF_DEFINE_COMPARISON(<)
XDIFF_DEFINE_COMPARISON(>)
XDIFF_DEFINE_COMPARISON(<=)
XDIFF_DEFINE_COMPARISON(>=)

} // namespace xdiff::detail


namespace xdiff{

/// @brief f^P as a fold of |P| factors, reciprocated when P < 0: the integer power of a plain scalar, which
/// rules::Power reaches at the bottom of a Dual. A Dual itself takes the more specialized overload above.
template<int P, typename F>
XDIFF_INLINE_HOST_DEVICE
decltype(auto) power(const F& f){
    static_assert(P != 0, "Integer power must be non-zero");
    if constexpr (P > 0) {
        return detail::power_fold<size_t(P)>(f);
    } else {
        return 1 / detail::power_fold<size_t(-P)>(f);
    }
}

} // namespace xdiff

#undef XDIFF_DEFINE_UNARY
#undef XDIFF_DEFINE_ASSIGN_BINARY
#undef XDIFF_DEFINE_BINARY_OPERATOR
#undef XDIFF_DEFINE_BINARY_PACK
#undef XDIFF_DEFINE_COMPARISON
#undef XDIFF_DUAL
#undef XDIFF_SEED
#undef XDIFF_NO_DEF

#endif // XDIFF_DUAL_IMPL_HPP
