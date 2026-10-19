#ifndef XDIFF_DUAL_MACROS_HPP
#define XDIFF_DUAL_MACROS_HPP

#include "../backend.hpp" // IWYU pragma: keep


#define XDIFF_DUAL Dual<T, GD, Order>
#define XDIFF_SEED Seed<T, GD, Order>

#define XDIFF_DECLARE_UNARY(OPERATOR, ASSIGN) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*arg*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*arg*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL OPERATOR(const XDIFF_DUAL& /*arg*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL OPERATOR(const XDIFF_SEED& /*arg*/); \




#define XDIFF_DECLARE_ASSIGN_BINARY(ASSIGN) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_DUAL& /*a*/, const F& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const XDIFF_SEED& /*a*/, const F& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL& ASSIGN(XDIFF_DUAL& /*out*/, const F& /*a*/, const XDIFF_SEED& /*b*/);




#define XDIFF_DECLARE_BINARY_OPERATOR(NAME) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_DUAL& /*a*/, const F& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const F& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const XDIFF_SEED& /*a*/, const F& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE XDIFF_DUAL NAME(const F& /*a*/, const XDIFF_SEED& /*b*/); \


#define XDIFF_DECLARE_BINARY_PACK(NAME, ASSIGN) \
XDIFF_DECLARE_ASSIGN_BINARY(ASSIGN) \
XDIFF_DECLARE_BINARY_OPERATOR(NAME)


// Comparisons look at the value alone, so they return bool.
#define XDIFF_DECLARE_COMPARISON(OP) \
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_DUAL& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_DUAL& /*a*/, const XDIFF_SEED& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_SEED& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_SEED& /*a*/, const XDIFF_SEED& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_DUAL& /*a*/, const F& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const F& /*a*/, const XDIFF_DUAL& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const XDIFF_SEED& /*a*/, const F& /*b*/); \
\
template<typename T, typename GD, int Order, detail::converts_to<T> F> \
XDIFF_HOST_DEVICE bool operator OP(const F& /*a*/, const XDIFF_SEED& /*b*/);


#endif // XDIFF_DUAL_MACROS_HPP