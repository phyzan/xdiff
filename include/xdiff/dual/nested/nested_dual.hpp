#ifndef XDIFF_DUAL_NESTED_HPP
#define XDIFF_DUAL_NESTED_HPP


#include "../dual_base.hpp"
#include <lazy/lazy.hpp>



namespace xdiff::detail{


template<typename T, typename G, size_t NVARS, typename Derived>
class RecursiveDual;

// Helper struct to extract the dual type (or scalar type) that a recursive dual contains
template<typename U>
struct DualInspector {
    static_assert(!std::is_same_v<U, void>, "Invalid type in DualInspector.");
    using type = U;
};

template<typename T, size_t NVARS, size_t NORDER>
struct DualInspector<Dual<T, NVARS, NORDER, Layout::Nested>> {
    using type = Dual<T, NVARS, NORDER, Layout::Nested>;
};

template<typename T, size_t NVARS, size_t NORDER>
struct DualInspector<lazy::LazyType<Dual<T, NVARS, NORDER, Layout::Nested>>> {
    using type = Dual<T, NVARS, NORDER, Layout::Nested>;
};

// Helper to determine the base RecursiveDual for the Dual class
template<typename Derived, typename T, size_t NVARS, size_t NORDER>
struct RecursiveBaseHelper {
#ifdef XDIFF_LAZY_NESTED_DUAL
    using GradType = std::conditional_t<
        NVARS == 0,
        lazy::LazyType<Dual<T, NVARS, NORDER - 1, Layout::Nested>>,
        Dual<T, NVARS, NORDER - 1, Layout::Nested>
    >;
#else
    using GradType = Dual<T, NVARS, NORDER - 1, Layout::Nested>;
#endif
    using type = RecursiveDual<T, GradType, NVARS, Derived>;
};

template<typename Derived, typename T, size_t NVARS>
struct RecursiveBaseHelper<Derived, T, NVARS, 1> {
    using type = RecursiveDual<T, T, NVARS, Derived>;
};

template<typename Derived, typename T, size_t NVARS, size_t NORDER>
using GetRecursiveBase = typename RecursiveBaseHelper<Derived, T, NVARS, NORDER>::type;


// Struct holding the default number of variables for recursive dual numbers, with thread-local storage.
template<typename T>
struct DefaultNvarsHolder{
    inline static thread_local size_t default_nvars = 1;
};

// Base class for recursive dual numbers, defining the core interface and common functionality.
template<typename Derived, typename T, typename G, size_t NVARS>
class RecursiveDualBase : public DualBase<Derived, T, NVARS, Layout::Nested>{    

    using Base = DualBase<Derived, T, NVARS, Layout::Nested>;

public:
    // Base already defines value_type.
    using grad_type = G;

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase(const RecursiveDualBase& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase(RecursiveDualBase&& other) noexcept = default;
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase& operator=(const RecursiveDualBase& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase& operator=(RecursiveDualBase&& other) noexcept = default;

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, RecursiveDualBase> && std::is_convertible_v<U, T>) {
        this->true_value = std::forward<U>(other);
        return *this;
    }
    
    ~RecursiveDualBase() = default;

    const T& value() const {
        if constexpr (std::is_same_v<T, G>){
            return this->true_value;
        } else{
            return this->true_value.value();
        }
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    const T& grad(Int i) const {
        if constexpr (std::is_same_v<T, G>){
            return (*this)[i];
        } else{
            return (*this)[i].value();
        }
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    size_t nvars() const{
        return XDIFF_THIS->nvars();
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    const G& operator[](Int i) const {
        return XDIFF_THIS->operator[](i);
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    G& operator[](Int i){
        return XDIFF_THIS->operator[](i);
    }

    G true_value = {}; // The scalar value of the dual number

protected:

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase() = default;

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    explicit RecursiveDualBase(U&& value, MakeDual /**/) requires (std::is_same_v<T, G>) : true_value(std::forward<U>(value)) {}

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    explicit RecursiveDualBase(U&& value, MakeDual md) requires (!std::is_same_v<T, G>) : true_value(std::forward<U>(value), md) {}

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase(MakeDual /*md*/) requires (std::is_same_v<T, G>) {}

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase(MakeDual md) requires (!std::is_same_v<T, G>) : true_value(md) {}

};


// RecursiveDual for compile-time known number of variables.
template<typename T, typename G, size_t NVARS, typename Derived = void>
class  RecursiveDual : public RecursiveDualBase<xdiff::tools::GetDerived<RecursiveDual<T, G, NVARS, Derived>, Derived>, T, G, NVARS> {

    static_assert(NVARS > 0, "NVARS must be positive for compile-time known number of variables in RecursiveDual");

    using Base = RecursiveDualBase<xdiff::tools::GetDerived<RecursiveDual<T, G, NVARS, Derived>, Derived>, T, G, NVARS>;

public:

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual() = default;
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual(const RecursiveDual& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual(RecursiveDual&& other) noexcept = default;

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    explicit RecursiveDual(U&& value, MakeDual md) : Base(std::forward<U>(value), md) {
        assert(md.nvars == NVARS || md.nvars == 0 && "nvars must match NVARS for compile-time known number of variables in RecursiveDual");
        assert(md.axis < int(NVARS) && "Axis index must be within the number of derivatives");
        if (md.axis >= 0) {
            diffs_[md.axis] = 1; // Set the derivative for the specified axis to 1
        }
    }

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual(MakeDual md) : Base(md) {
        assert(md.nvars == NVARS || md.nvars == 0 && "nvars must match NVARS for compile-time known number of variables in RecursiveDual");
        assert(md.axis == -1 && "For uninitialized Dual, the axis is not used and should be -1");
    }

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual& operator=(const RecursiveDual& other) = default;
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual& operator=(RecursiveDual&& other) = default;

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, RecursiveDual>) {
        Base::operator=(std::forward<U>(other));
        std::fill(diffs_.begin(), diffs_.end(), 0);
        return *this;
    }

    ~RecursiveDual() = default;

    operator T() const = delete; // Disable implicit conversion to T to avoid accidental loss of derivative information.

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t nvars() const{ //number of partial derivatives stored
        return NVARS;
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    G& operator[](Int i){
        assert(i < NVARS && "Gradient index out of bounds");
        return diffs_[i];
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    const G& operator[](Int i) const{
        assert(i < NVARS && "Gradient index out of bounds");
        return diffs_[i];
    }

private:

    Vector<G, NVARS> diffs_ = {}; // The derivative values of the dual number

};


// RecursiveDual for runtime known number of variables.
template<typename T, typename G, typename Derived>
class RecursiveDual<T, G, 0, Derived> : public RecursiveDualBase<xdiff::tools::GetDerived<RecursiveDual<T, G, 0, Derived>, Derived>, T, G, 0> {

    using Base = RecursiveDualBase<xdiff::tools::GetDerived<RecursiveDual<T, G, 0, Derived>, Derived>, T, G, 0>;

public:

    static void set_default_nvars(size_t nvars){
        DefaultNvarsHolder<T>::default_nvars = nvars;
    }

    static size_t get_default_nvars(){
        return DefaultNvarsHolder<T>::default_nvars;
    }

    RecursiveDual() requires(!std::is_same_v<T, G>): Base(), diffs_(get_default_nvars(), MakeDual{.axis = -1, .nvars = get_default_nvars()}) {}

    RecursiveDual() requires (std::is_same_v<T, G>): Base(), diffs_(get_default_nvars()) {}

    template<typename U>
    explicit RecursiveDual(U&& value, MakeDual md) requires (!std::is_same_v<T, G>) : Base(std::forward<U>(value), md), diffs_(nv(md.nvars), MakeDual{.axis = -1, .nvars = md.nvars, .order = md.order}) {
        if (md.axis >= 0) {
            diffs_[md.axis] = 1;
        }
    }

    template<typename U>
    explicit RecursiveDual(U&& value, MakeDual md) requires (std::is_same_v<T, G>) : Base(std::forward<U>(value), md), diffs_(nv(md.nvars), 0) {
        if (md.axis >= 0) {
            diffs_[md.axis] = 1; // Set the derivative for the specified axis to 1
        }
    }

    RecursiveDual(MakeDual md) requires (std::is_same_v<T, G>): Base(md), diffs_(nv(md.nvars)) {}

    RecursiveDual(MakeDual md) requires (!std::is_same_v<T, G>): Base(md), diffs_(nv(md.nvars), md) {
        assert(md.axis == -1 && "For uninitialized Dual, the axis is not used and should be -1");
    }

    // ================ Copy and move constructors ============================
    RecursiveDual(const RecursiveDual&) = default;
    RecursiveDual(RecursiveDual&&) noexcept = default;
    RecursiveDual& operator=(const RecursiveDual&) = default;
    RecursiveDual& operator=(RecursiveDual&&) noexcept = default;
    ~RecursiveDual() = default;

    template<typename U>
    RecursiveDual& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, RecursiveDual>) {
        Base::operator=(std::forward<U>(other));
        std::fill(diffs_.begin(), diffs_.end(), 0);
        return *this;
    }

    operator T() const = delete; // Disable implicit conversion to T to avoid accidental loss of derivative information.

    void set_nvars(size_t nvars){
        diffs_ = Vector<G, 0>(nvars);
        if constexpr (!std::is_same_v<T, G>){
            diffs_.resize(nvars);
            for (size_t i = 0; i < nvars; ++i) {
                (*this)[i].set_nvars(nvars);
            }
        }
    }

    [[nodiscard]] size_t nvars() const{ //number of partial derivatives stored
        return diffs_.size();
    }

    template<std::integral Int>
    inline G& operator[](Int i){
        assert(i < nvars() && "Gradient index out of bounds");
        return diffs_[i];
    }

    template<std::integral Int>
    inline const G& operator[](Int i) const{
        assert(i < nvars() && "Gradient index out of bounds");
        return diffs_[i];
    }

private:

    static size_t nv(size_t nvars){
        return nvars > 0 ? nvars : DefaultNvarsHolder<T>::default_nvars;
    }

    Vector<G, 0> diffs_; // The derivative values of the dual number
};


} // namespace xdiff::detail


namespace xdiff{

template<typename T, size_t NVARS, size_t NORDER>
requires (NORDER > 0)
class Dual<T, NVARS, NORDER, Layout::Nested> : public xdiff::detail::GetRecursiveBase<Dual<T, NVARS, NORDER, Layout::Nested>, T, NVARS, NORDER> {

    using Base = xdiff::detail::GetRecursiveBase<Dual<T, NVARS, NORDER, Layout::Nested>, T, NVARS, NORDER>;
public:

    XDIFF_INLINE_HOST_DEVICE
    Dual() = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual(const Base& base) : Base(base) {} // Constructor from the base class
    XDIFF_INLINE_HOST_DEVICE
    Dual(Base&& base) noexcept : Base(std::move(base)) {}
    XDIFF_INLINE_HOST_DEVICE
    Dual(const Dual&) = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual(Dual&&) noexcept = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Dual&) = default;
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(Dual&&) noexcept = default;
    XDIFF_INLINE_HOST_DEVICE
    ~Dual() = default;

    template<typename U>
    requires (!std::is_same_v<std::decay_t<U>, Dual>)
    XDIFF_INLINE_HOST_DEVICE
    explicit Dual(U&& value, MakeDual md = {.axis = -1, .nvars=NVARS, .order=NORDER}) : Base(std::forward<U>(value), md) {}

    // TODO : For internal use only
    XDIFF_INLINE_HOST_DEVICE
    Dual(MakeDual md): Base(md) {}

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, Dual>) {
        Base::operator=(std::forward<U>(other));
        return *this;
    }

    operator T() const = delete; // Disable implicit conversion to T to avoid accidental loss of derivative information.

    template<xdiff::traits::isAxis... Int>
    XDIFF_INLINE_HOST_DEVICE
    const T& get_diff_wrt(Int... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        return diff_accessor(*this, x...);
    }

    static void set_default_nvars(size_t nvars) requires (NVARS == 0) {
        Base::set_default_nvars(nvars);
    }

    std::array<size_t, NORDER> get_vmat_shape() const{
        std::array<size_t, NORDER> shape;
        shape.fill(this->nvars()+1);
        return shape;
    }

    template<typename ArrayType>
    void build_diff_matrix(ArrayType& mat) const{
        std::array<size_t, NORDER> shape = get_vmat_shape();
        xdiff::tools::for_each([&](auto... idx){
            mat(idx...) = get_virtual_matrix_element(idx...);
        }, shape);
    }

    template<typename Action>
    inline static constexpr void with_default_nvars([[maybe_unused]] size_t nvars, Action&& action) {
        if constexpr (NVARS == 0) {
            size_t old_nvars = Base::get_default_nvars();
            Base::set_default_nvars(nvars);
            try{
                action();
            } catch(...) {
                Base::set_default_nvars(old_nvars);
                throw;
            }
            Base::set_default_nvars(old_nvars);
            return;
        } else {
            assert((nvars == NVARS) && "nvars must match NVARS for compile-time known number of variables in nested Dual");
            action();
            return;
        }
    }

private:

    template<typename U, size_t Nv, size_t Nord, Layout St>
    friend class Dual;   // every Dual is a friend

    template<size_t NORD, xdiff::traits::isAxis First,xdiff::traits::isAxis... Int>
    XDIFF_INLINE_HOST_DEVICE
    static const T& diff_accessor(const Dual<T, NVARS, NORD, Layout::Nested>& dual, First x0, Int... x) {
        static_assert(sizeof...(x) < NORD, "Number of differentiations requested must be < NORDER");
        if constexpr (sizeof...(x) == 0){
            return dual.grad(x0);
        } else {
            return diff_accessor(dual[x0], x...);
        }
    }

    template<std::integral First, std::integral... Int>
    const T& get_virtual_matrix_element(First x0, Int... x) const{
        static_assert(sizeof...(x) + 1 == NORDER, "Number of differentiations requested must be <= NORDER");
        if constexpr (NORDER == 1){
            if (x0 == 0){
                return this->value();
            } else {
                return this->grad(x0-1);
            }
        } else {
            if (x0 == 0){
                return this->true_value.get_virtual_matrix_element(x...);
            } else {
                return (*this)[x0-1].get_virtual_matrix_element(x...);
            }
        }
    }
};


} // namespace xdiff


#include "nested_dual_operators.hpp" // IWYU pragma: keep


#endif // XDIFF_DUAL_NESTED_HPP