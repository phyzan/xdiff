#ifndef XDIFF_SEED_DUALSEED_HPP
#define XDIFF_SEED_DUALSEED_HPP

#include "../dual/dual.hpp"
#include <concepts>
#include <cstddef>
#include <utility>


namespace xdiff {

namespace detail{

// Empty stand-in for a size that is already known at compile time, so that a Seed
// with a compile-time number of variables does not pay for storing it.
struct UnusedSize{
    XDIFF_INLINE_HOST_DEVICE
    constexpr UnusedSize(size_t /*size*/) {}
};

} // namespace xdiff::detail


/**
 * @brief A lightweight stand-in for a Dual that holds an independent variable.
 *
 * A seed variable x_i has a trivial derivative structure: dx_i/dx_j is 1 for j == i and 0
 * otherwise, and every higher-order derivative vanishes. A Seed therefore stores only
 * the value and the axis, and the operators reconstruct the derivatives on the fly. It takes
 * part in every operation a Dual does, and any such operation yields a Dual.
 */
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
class Seed{

    static_assert(NORDER > 0, "A Seed must carry at least one derivative order");

public:
    using DualType = Dual<T, NVARS, NORDER, LY>;

    /// Type after one differentiation: one order lower, or a plain scalar once no order is left.
    using TrimmedType = std::conditional_t<(NORDER > 1), Seed<T, NVARS, NORDER-1, LY>, T>;

    XDIFF_INLINE_HOST_DEVICE
    Seed(T value, size_t axis, size_t nvars) : value_(std::move(value)), axis_(axis), nvars_(nvars) {
        assert((NVARS == 0 || nvars == NVARS) && "nvars must match NVARS for a compile-time known number of variables in Seed");
        assert(axis < this->nvars() && "Axis out of bounds");
    }

    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& value() const{
        return value_;
    }

    /// @brief Builds the Dual this seed stands for.
    XDIFF_INLINE_HOST_DEVICE
    DualType to_dual() const{
        return DualType{value_, MakeDual{.axis = int(axis_), .nvars = nvars(), .order = NORDER}};
    }

    /**
     * @brief Returns the reduced-order representation of this seed.
     * @return A Seed of order NORDER-1, or the scalar value when no order is left.
     */
    XDIFF_INLINE_HOST_DEVICE
    decltype(auto) trimmed() const{
        if constexpr (NORDER > 1){
            return TrimmedType{value_, axis_, nvars()};
        } else {
            return (value_);
        }
    }

    /// @brief The variable this seed differentiates to 1 along.
    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t axis() const{
        return axis_;
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t nvars() const{
        if constexpr (NVARS > 0) {
            return NVARS;
        } else {
            return nvars_;
        }
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t order() const{
        return NORDER;
    }

private:

    T value_;
    size_t axis_;
    [[no_unique_address]] std::conditional_t<NVARS == 0, size_t, detail::UnusedSize> nvars_;
};


/**
 * @brief A cursor over a state vector whose elements are seen as independent variables.
 *
 * Dereferencing or indexing a SeedVector yields the Seed for that element, which behaves
 * as the Dual holding that value with a unit derivative along its own axis. The state vector
 * is only referenced, never copied, so writing to it is immediately visible through the seed.
 *
 * SeedVector is both the range and its own iterator, so it supports range-for as well as the
 * pointer-like interface (`*`, `[]`, `++`, `--`, `+`, `-`).
 */
template<typename T, size_t NVARS, size_t NORDER, Layout LY>
class SeedVector{

    static_assert(NORDER > 0, "A SeedVector must carry at least one derivative order");

    using seed_t = Seed<T, NVARS, NORDER, LY>;

public:
    using value_type = seed_t;
    using iterator = SeedVector;
    using const_iterator = SeedVector;

    // "nvars" and "order" default to the compile-time NVARS / NORDER, and must match them wherever
    // they are known. Only a runtime number of variables (NVARS == 0) carries a value of its own,
    // and there the zero default selects the Dual's default number of variables.
    // "nvars" is validated by nv(), the one place that interprets it.
    XDIFF_INLINE_HOST_DEVICE
    SeedVector(const T* seed, size_t nvars = NVARS, size_t order = NORDER) : seed_(seed), size_(nv(nvars)) {
        assert(order == NORDER && "order must match NORDER for a compile-time known order in SeedVector");
        (void)order;
    }

    XDIFF_INLINE_HOST_DEVICE
    seed_t operator*() const{
        check_is_valid();
        return seed_t{seed_[idx_], idx_, size_};
    }

    /// @brief Returns the seed "i" elements past the current one.
    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    seed_t operator[](Int i) const{
        const size_t idx = offset_index(static_cast<std::ptrdiff_t>(i));
        assert(idx < size_ && "Index out of upper bound");
        return seed_t{seed_[idx], idx, size_};
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    SeedVector operator+(Int i) const{
        return SeedVector{*this, offset_index(static_cast<std::ptrdiff_t>(i))};
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    SeedVector operator-(Int i) const{
        return SeedVector{*this, offset_index(-static_cast<std::ptrdiff_t>(i))};
    }

    // Prefix ++obj
    XDIFF_INLINE_HOST_DEVICE
    SeedVector& operator++() {
        ++idx_;
        return *this;
    }

    // Postfix obj++
    XDIFF_INLINE_HOST_DEVICE
    SeedVector operator++(int) {
        SeedVector old{*this};
        idx_++;
        return old;
    }

    // Prefix --obj
    XDIFF_INLINE_HOST_DEVICE
    SeedVector& operator--() {
        --idx_;
        return *this;
    }

    // Postfix obj--
    XDIFF_INLINE_HOST_DEVICE
    SeedVector operator--(int) {
        SeedVector old{*this};
        idx_--;
        return old;
    }

    XDIFF_INLINE_HOST_DEVICE
    bool operator==(const SeedVector& other) const {
        return seed_ == other.seed_ && idx_ == other.idx_;
    }

    XDIFF_INLINE_HOST_DEVICE
    const_iterator begin() const {
        return SeedVector{*this, 0};
    }

    XDIFF_INLINE_HOST_DEVICE
    const_iterator end() const {
        return SeedVector{*this, size_};
    }

    XDIFF_INLINE_HOST_DEVICE
    const_iterator cbegin() const {
        return begin();
    }

    XDIFF_INLINE_HOST_DEVICE
    const_iterator cend() const {
        return end();
    }

    /// @brief The state vector this seed reads from. It is never copied, so it stays live.
    XDIFF_INLINE_HOST_DEVICE
    const T* data() const{
        return seed_;
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t nvars() const{
        return size();
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t size() const{
        if constexpr (NVARS > 0){
            return NVARS;
        } else {
            return size_;
        }
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t order() const{
        return NORDER;
    }

    template<size_t Order>
    XDIFF_INLINE_HOST_DEVICE
    SeedVector<T, NVARS, Order, LY> with_order() const {
        return SeedVector<T, NVARS, Order, LY>{seed_, nvars(), Order} + idx_;
    }

private:

    XDIFF_INLINE_HOST_DEVICE
    SeedVector(const SeedVector& s, size_t idx) : SeedVector(s) {
        idx_ = idx;
    }

    XDIFF_INLINE_HOST_DEVICE
    void check_is_valid() const{
        assert(idx_ < size_ && "Cannot dereference SeedVector, its index has exceeded the state vector's length");
    }

    XDIFF_INLINE_HOST_DEVICE
    size_t offset_index(std::ptrdiff_t i) const{
        // For a negative offset, idx_ must be at least -i. Written as -(i+1) < idx_ so that
        // the most negative representable offset cannot overflow while being negated.
        assert((i >= 0 || size_t(-(i+1)) < idx_) && "Index out of lower bound");
        return idx_ + size_t(i);
    }

    XDIFF_INLINE_HOST_DEVICE
    static size_t nv(size_t nvars){
        if constexpr (NVARS > 0){
            assert(nvars == NVARS && "nvars must match NVARS for a compile-time known number of variables in SeedVector");
            (void)nvars;
            return NVARS;
        } else {
            return nvars > 0 ? nvars : Dual<T, NVARS, NORDER, LY>::get_default_nvars();
        }
    }

    const T* seed_;
    size_t size_;

    // Starting index of seed_.
    // The true state vector starts from `seed_`, but the current
    // SeedVector might be starting from seed_ + 1, e.g. due to incrementing
    // its pointer value.
    size_t idx_ = 0;
};

} // namespace xdiff

#endif // XDIFF_SEED_DUALSEED_HPP
