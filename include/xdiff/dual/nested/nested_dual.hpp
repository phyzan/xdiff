#ifndef XDIFF_DUAL_NESTED_DUAL_HPP
#define XDIFF_DUAL_NESTED_DUAL_HPP


#include "nested_helpers.hpp"
#include "../../rules/math.hpp" // IWYU pragma: keep

namespace xdiff::detail{


// Struct holding the default number of variables for recursive dual numbers, with thread-local storage.
template<typename T>
struct DefaultNvarsHolder{
    inline static thread_local size_t default_nvars = 1;
};

// Base class for recursive dual numbers, defining the core interface and common functionality.
template<typename Derived, typename T, typename G, int NVARS>
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
    ~RecursiveDualBase() = default;

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, RecursiveDualBase> && std::is_convertible_v<U, T>) {
        this->true_value = std::forward<U>(other);
        return *this;
    }

    constexpr const T& value() const {
        if constexpr (std::is_same_v<T, G>){
            return this->true_value;
        } else {
            return this->true_value.value();
        }
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& grad(Int i) const {
        if constexpr (std::is_same_v<T, G>){
            return (*this)[i];
        } else{
            return (*this)[i].value();
        }
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t nvars() const{
        return XDIFF_THIS->nvars();
    }

protected:

    // operator[] and true_value are raw storage: the Dual that derives from this base names the library internals that may reach them.
    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const G& operator[](Int i) const {
        return XDIFF_THIS->operator[](i);
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr G& operator[](Int i){
        return XDIFF_THIS->operator[](i);
    }

    G true_value; // The scalar value of the dual number

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase() = default;

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    explicit RecursiveDualBase(U&& value, MakeDual /**/) requires (std::is_same_v<T, G>) : true_value(std::forward<U>(value)) {}

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    explicit RecursiveDualBase(U&& value, MakeDual md) requires (!std::is_same_v<T, G>) : true_value(std::forward<U>(value), MakeDual{.axis = md.axis, .nvars = md.nvars, .order = -1}) {}

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase(MakeDual /*md*/) requires (std::is_same_v<T, G>) : true_value{} {}

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDualBase(MakeDual md) requires (!std::is_same_v<T, G>) : true_value(MakeDual{.axis = md.axis, .nvars = md.nvars, .order = -1}) {}

};


// RecursiveDual for compile-time known number of variables.
template<typename T, typename G, int NVARS, typename Derived>
class  RecursiveDual : public RecursiveDualBase<xdiff::tools::GetDerived<RecursiveDual<T, G, NVARS, Derived>, Derived>, T, G, NVARS> {

    static_assert(NVARS >= 0, "NVARS must be non-negative for a compile-time known number of variables in RecursiveDual");

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
    explicit RecursiveDual(U&& value, MakeDual md) : Base(std::forward<U>(value), md), diffs_{} {
        assert((md.nvars == NVARS || md.nvars == -1) && "nvars must match NVARS for compile-time known number of variables in RecursiveDual");
        assert(md.axis < int(NVARS) && "Axis index must be within the number of derivatives");
        if (md.axis >= 0) {
            diffs_[md.axis] = 1; // Set the derivative for the specified axis to 1
        }
    }

    XDIFF_INLINE_HOST_DEVICE
    RecursiveDual(MakeDual md) : Base(md), diffs_{} {
        assert((md.nvars == NVARS || md.nvars == -1) && "nvars must match NVARS for compile-time known number of variables in RecursiveDual");
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

protected:

    // Raw access to the gradient entries. Base forwards its own operator[] to these overloads, and the Dual that derives from it names the library internals that may reach them.
    friend Base;

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr G& operator[](Int i){
        assert(size_t(i) < NVARS && "Gradient index out of bounds");
        return diffs_[i];
    }

    template<std::integral Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const G& operator[](Int i) const{
        assert(size_t(i) < NVARS && "Gradient index out of bounds");
        return diffs_[i];
    }

private:

    Vector<G, NVARS> diffs_; // The derivative values of the dual number

};


// RecursiveDual for runtime known number of variables.
template<typename T, typename G, typename Derived>
class RecursiveDual<T, G, -1, Derived> : public RecursiveDualBase<xdiff::tools::GetDerived<RecursiveDual<T, G, -1, Derived>, Derived>, T, G, -1> {

    using Base = RecursiveDualBase<xdiff::tools::GetDerived<RecursiveDual<T, G, -1, Derived>, Derived>, T, G, -1>;

public:

    static void set_default_nvars(size_t nvars){
        DefaultNvarsHolder<T>::default_nvars = nvars;
    }

    static size_t get_default_nvars(){
        return DefaultNvarsHolder<T>::default_nvars;
    }

    RecursiveDual() requires(!std::is_same_v<T, G>): Base(), diffs_(get_default_nvars(), MakeDual{.axis = -1, .nvars = int(get_default_nvars())}) {}

    RecursiveDual() requires (std::is_same_v<T, G>): Base(), diffs_(get_default_nvars()) {}

    template<typename U>
    explicit RecursiveDual(U&& value, MakeDual md) requires (!std::is_same_v<T, G>) : Base(std::forward<U>(value), md), diffs_(nv(md.nvars), MakeDual{.axis = -1, .nvars = md.nvars, .order = -1}) {
        assert(md.axis < int(diffs_.size()) && "Axis index must be within the number of derivatives");
        if (md.axis >= 0) {
            diffs_[md.axis] = 1;
        }
    }

    template<typename U>
    explicit RecursiveDual(U&& value, MakeDual md) requires (std::is_same_v<T, G>) : Base(std::forward<U>(value), md), diffs_(nv(md.nvars), 0) {
        assert(md.axis < int(diffs_.size()) && "Axis index must be within the number of derivatives");
        if (md.axis >= 0) {
            diffs_[md.axis] = 1; // Set the derivative for the specified axis to 1
        }
    }

    RecursiveDual(MakeDual md) requires (std::is_same_v<T, G>): Base(md), diffs_(nv(md.nvars)) {}

    RecursiveDual(MakeDual md) requires (!std::is_same_v<T, G>): Base(md), diffs_(nv(md.nvars), MakeDual{.axis = -1, .nvars = md.nvars, .order = -1}) {
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
        diffs_ = Vector<G, -1>(nvars);
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

protected:

    // Raw access to the gradient entries. Base forwards its own operator[] to these overloads, and the Dual that derives from it names the library internals that may reach them.
    friend Base;

    template<std::integral Int>
    constexpr inline G& operator[](Int i){
        assert(size_t(i) < nvars() && "Gradient index out of bounds");
        return diffs_[i];
    }

    template<std::integral Int>
    constexpr inline const G& operator[](Int i) const{
        assert(size_t(i) < nvars() && "Gradient index out of bounds");
        return diffs_[i];
    }

private:

    static size_t nv(int nvars){
        return nvars >= 0 ? size_t(nvars) : DefaultNvarsHolder<T>::default_nvars;
    }

    Vector<G, -1> diffs_; // The derivative values of the dual number
};


} // namespace xdiff::detail


namespace xdiff{

template<typename T, int NVARS, int NORDER>
requires (NORDER > 0)
class Dual<T, NVARS, NORDER, Layout::Nested> : public xdiff::detail::GetRecursiveBase<Dual<T, NVARS, NORDER, Layout::Nested>, T, NVARS, NORDER> {

    using Base = xdiff::detail::GetRecursiveBase<Dual<T, NVARS, NORDER, Layout::Nested>, T, NVARS, NORDER>;

    static_assert(NVARS >= -1, "Nested Dual requires NVARS >= -1");
public:

    using ReducedType = Dual<T, NVARS, NORDER-1, Layout::Nested>;

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
    explicit Dual(U&& value, MakeDual md = {.axis = -1, .nvars=NVARS, .order=NORDER}) : Base(std::forward<U>(value), validate(md)) {}

    // TODO : For internal use only
    XDIFF_INLINE_HOST_DEVICE
    Dual(MakeDual md): Base(validate(md)) {}

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, Dual> && detail::isScalarOperand<U, T>) {
        Base::operator=(std::forward<U>(other));
        return *this;
    }

    /**
     * @brief Assigns the seed variable a Seed stands for.
     *
     * The result holds the seed's value and a unit derivative along its own axis; every other
     * derivative, of every order, is zero. Assigning the scalar value first clears the whole
     * gradient recursively, so only that one entry is left to set.
     */
    template<int No>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Seed<T, NVARS, No, Layout::Nested>& seed) {
        if constexpr (NVARS == -1) {
            if (this->nvars() != seed.nvars()) {
                this->set_nvars(seed.nvars());
            }
        } else {
            assert(seed.nvars() == NVARS && "nvars must match NVARS when assigning a Seed to a Dual");
        }
        *this = seed.value();
        (*this)[seed.axis()] = 1;
        return *this;
    }

    // =========== Compound assignment ===========

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const Dual& arg){
        format_nested(*this, arg);
        this->true_value += arg.true_value;
        for (size_t i=0; i < arg.nvars(); i++){
            (*this)[i] += arg[i];
        }
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const Seed<T, NVARS, NORDER, Layout::Nested>& seed){
        format_nested(*this, seed);
        // A seed variable adds its value, and a unit derivative along its own axis only.
        this->true_value += seed.trimmed();
        (*this)[seed.axis()] += 1;
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const U& arg){
        this->true_value += arg;   // adding a constant leaves every derivative untouched
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const Dual& arg){
        format_nested(*this, arg);
        this->true_value -= arg.true_value;
        for (size_t i=0; i < arg.nvars(); i++){
            (*this)[i] -= arg[i];
        }
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const Seed<T, NVARS, NORDER, Layout::Nested>& seed){
        format_nested(*this, seed);
        // A seed variable subtracts its value, and a unit derivative along its own axis only.
        this->true_value -= seed.trimmed();
        (*this)[seed.axis()] -= 1;
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const U& arg){
        this->true_value -= arg;   // subtracting a constant leaves every derivative untouched
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const Dual& arg){
        format_nested(*this, arg);
        using G = typename Base::grad_type;
        using DP = DiffPair<const G&, const G&>;
        // The gradient is updated before the value, since the product rule needs the old value.
        for (size_t i=0; i < this->nvars(); i++){
            (*this)[i] = detail::rules::Mul<T>::diff_rule(DP{this->true_value, (*this)[i]}, DP{arg.true_value, arg[i]});
        }
        this->true_value *= arg.true_value;
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const Seed<T, NVARS, NORDER, Layout::Nested>& seed){
        return *this = *this * seed;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const U& arg){
        this->true_value *= arg;
        for (size_t i=0; i < this->nvars(); i++){
            (*this)[i] *= arg;
        }
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const Dual& arg){
        format_nested(*this, arg);
        using G = typename Base::grad_type;
        using DP = DiffPair<const G&, const G&>;
        // The gradient is updated before the value, since the quotient rule needs the old value.
        for (size_t i=0; i < this->nvars(); i++){
            (*this)[i] = detail::rules::Div<T>::diff_rule(DP{this->true_value, (*this)[i]}, DP{arg.true_value, arg[i]});
        }
        this->true_value /= arg.true_value;
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const Seed<T, NVARS, NORDER, Layout::Nested>& seed){
        return *this = *this / seed;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const U& arg){
        this->true_value /= arg;
        for (size_t i=0; i < this->nvars(); i++){
            (*this)[i] /= arg;
        }
        return *this;
    }

    operator T() const = delete; // Disable implicit conversion to T to avoid accidental loss of derivative information.

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t order() const{
        return NORDER;
    }

    template<std::integral... Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& get_diff_wrt(Int... x) const{
        static_assert(sizeof...(x)<=NORDER, "Number of differentiations requested must be <= NORDER");
        if constexpr (sizeof...(x) == 0){
            return this->value();
        } else {
            return diff_accessor(*this, x...);
        }
    }

    XDIFF_INLINE_HOST_DEVICE
    ReducedType trimmed() const{
        ReducedType out;
        out = this->true_value;
        return out;
    }

    template<std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE
    auto constexpr trimmed_diff_wrt(IntType... x) const{
        // Returning `auto` and not `const auto&` for compatibility with Dual<Layout=Flat>
        // TODO : should be able to do <= NORDER.
        // This must be fixed by allowing NORDER=0 as a template parameter, without that meaning dynamic size
        // Dynamic size should be NORDER = -1 by changing from size_t -> int
        static_assert(sizeof...(x) <= NORDER, "Number of differentiations requested must be <= NORDER");
        if constexpr (sizeof...(x) == 0){
            return (*this);
        } else {
            return this->trimmed_diff_wrt_helper(x...);
        }
    }

    static void set_default_nvars(size_t nvars) requires (NVARS == -1) {
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
        if constexpr (NVARS == -1) {
            size_t old_nvars = Base::get_default_nvars();
            Base::set_default_nvars(nvars);
            try{
                std::forward<Action>(action)();
            } catch(...) {
                Base::set_default_nvars(old_nvars);
                throw;
            }
            Base::set_default_nvars(old_nvars);
        } else {
            assert((nvars == NVARS) && "nvars must match NVARS for compile-time known number of variables in nested Dual");
            std::forward<Action>(action)();
        }
    }

private:

    XDIFF_INLINE_HOST_DEVICE
    static MakeDual validate(MakeDual md){
        if constexpr (NVARS >= 0){
            if (md.nvars != NVARS && md.nvars != -1) {
                throw std::invalid_argument("nvars must be -1 or match NVARS for compile-time known number of variables in Dual");
            }
        } else {
            if (md.nvars < -1) {
                throw std::invalid_argument("nvars must be -1 or non-negative for a runtime number of variables in Dual");
            }
        }
        if (md.order != NORDER && md.order != -1) {
            throw std::invalid_argument("order must be -1 or match NORDER for compile-time known order in Dual");
        }
        if (md.axis < -1 || md.axis >= effective_nvars(md)) {
            throw std::invalid_argument("axis must be -1 or within the number of variables in Dual");
        }
        return md;
    }

    XDIFF_INLINE_HOST_DEVICE
    static int effective_nvars(MakeDual md){
        if (md.nvars >= 0) {
            return md.nvars;
        }
        if constexpr (NVARS >= 0){
            return NVARS;
        } else {
            return int(Base::get_default_nvars());
        }
    }

    template<typename U, int Nv, int Nord, Layout St>
    friend class Dual;   // every Dual is a friend

    friend struct detail::NestedDualOperationHelper;

    template<int NORD, std::integral First, std::integral... Int>
    XDIFF_INLINE_HOST_DEVICE
    static constexpr const T& diff_accessor(const Dual<T, NVARS, NORD, Layout::Nested>& dual, First x0, Int... x) {
        static_assert(sizeof...(x) < NORD, "Number of differentiations requested must be < NORDER");
        if constexpr (sizeof...(x) == 0){
            return dual.grad(x0);
        } else {
            return diff_accessor(dual[x0], x...);
        }
    }

    template<std::integral First, std::integral... IntType>
    XDIFF_INLINE_HOST_DEVICE
    constexpr auto trimmed_diff_wrt_helper(First x0, IntType... x) const{
        if constexpr (sizeof...(x) == 0){
            ReducedType out;
            out = (*this)[x0];
            return out;
        } else {
            return (*this)[x0].trimmed_diff_wrt(x...);
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


template<typename T, int NVARS>
class Dual<T, NVARS, 0, Layout::Nested> : public xdiff::detail::RecursiveDualBase<Dual<T, NVARS, 0, Layout::Nested>, T, T, NVARS> {

    using Base = xdiff::detail::RecursiveDualBase<Dual<T, NVARS, 0, Layout::Nested>, T, T, NVARS>;

    static_assert(NVARS >= -1, "Nested Dual requires NVARS >= -1");
public:

    using ReducedType = Dual<T, NVARS, 0, Layout::Nested>;

    XDIFF_INLINE_HOST_DEVICE
    Dual() : Base(), nvars_(nv(-1)) {}
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
    explicit Dual(U&& value, MakeDual md = {.axis = -1, .nvars = NVARS, .order = 0}) : Base(std::forward<U>(value), validate(md)), nvars_(nv(md.nvars)) {}

    XDIFF_INLINE_HOST_DEVICE
    Dual(MakeDual md) : Base(validate(md)), nvars_(nv(md.nvars)) {}

    template<typename U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(U&& other) requires (!std::is_same_v<std::decay_t<U>, Dual> && detail::isScalarOperand<U, T>) {
        Base::operator=(std::forward<U>(other));
        return *this;
    }

    template<int No>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator=(const Seed<T, NVARS, No, Layout::Nested>& seed) {
        this->true_value = seed.value();
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const Dual& arg){
        this->true_value += arg.true_value;
        return *this;
    }

    template<int No>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const Seed<T, NVARS, No, Layout::Nested>& seed){
        this->true_value += seed.value();
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator+=(const U& arg){
        this->true_value += arg;
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const Dual& arg){
        this->true_value -= arg.true_value;
        return *this;
    }

    template<int No>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const Seed<T, NVARS, No, Layout::Nested>& seed){
        this->true_value -= seed.value();
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator-=(const U& arg){
        this->true_value -= arg;
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const Dual& arg){
        this->true_value *= arg.true_value;
        return *this;
    }

    template<int No>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const Seed<T, NVARS, No, Layout::Nested>& seed){
        this->true_value *= seed.value();
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator*=(const U& arg){
        this->true_value *= arg;
        return *this;
    }

    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const Dual& arg){
        this->true_value /= arg.true_value;
        return *this;
    }

    template<int No>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const Seed<T, NVARS, No, Layout::Nested>& seed){
        this->true_value /= seed.value();
        return *this;
    }

    template<detail::isScalarOperand<T> U>
    XDIFF_INLINE_HOST_DEVICE
    Dual& operator/=(const U& arg){
        this->true_value /= arg;
        return *this;
    }

    operator T() const = delete;

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t nvars() const{
        if constexpr (NVARS >= 0){
            return size_t(NVARS);
        } else {
            return nvars_;
        }
    }

    [[nodiscard]]
    XDIFF_INLINE_HOST_DEVICE
    constexpr size_t order() const{
        return 0;
    }

    template<std::integral... Int>
    XDIFF_INLINE_HOST_DEVICE
    constexpr const T& get_diff_wrt(Int... x) const{
        static_assert(sizeof...(x) == 0, "Number of differentiations requested must be <= NORDER");
        return this->true_value;
    }

    XDIFF_INLINE_HOST_DEVICE
    ReducedType trimmed() const{
        return *this;
    }

    template<std::integral... Int>
    XDIFF_INLINE_HOST_DEVICE
    auto constexpr trimmed_diff_wrt(Int... x) const{
        static_assert(sizeof...(x) == 0, "Number of differentiations requested must be <= NORDER");
        return *this;
    }

    static void set_default_nvars(size_t nvars) requires (NVARS == -1) {
        xdiff::detail::DefaultNvarsHolder<T>::default_nvars = nvars;
    }

    XDIFF_INLINE_HOST_DEVICE
    void set_nvars(size_t nvars) requires (NVARS == -1) {
        nvars_ = nvars;
    }

    std::array<size_t, 0> get_vmat_shape() const{
        return {};
    }

    template<typename Action>
    inline static constexpr void with_default_nvars([[maybe_unused]] size_t nvars, Action&& action) {
        if constexpr (NVARS == -1) {
            size_t old_nvars = xdiff::detail::DefaultNvarsHolder<T>::default_nvars;
            xdiff::detail::DefaultNvarsHolder<T>::default_nvars = nvars;
            try{
                std::forward<Action>(action)();
            } catch(...) {
                xdiff::detail::DefaultNvarsHolder<T>::default_nvars = old_nvars;
                throw;
            }
            xdiff::detail::DefaultNvarsHolder<T>::default_nvars = old_nvars;
        } else {
            assert((nvars == NVARS) && "nvars must match NVARS for compile-time known number of variables in nested Dual");
            std::forward<Action>(action)();
        }
    }

private:

    XDIFF_INLINE_HOST_DEVICE
    static MakeDual validate(MakeDual md){
        if constexpr (NVARS >= 0){
            if (md.nvars != NVARS && md.nvars != -1) {
                throw std::invalid_argument("nvars must be -1 or match NVARS for compile-time known number of variables in Dual");
            }
        } else {
            if (md.nvars < -1) {
                throw std::invalid_argument("nvars must be -1 or non-negative for a runtime number of variables in Dual");
            }
        }
        if (md.order != 0 && md.order != -1) {
            throw std::invalid_argument("order must be -1 or match NORDER for compile-time known order in Dual");
        }
        if (md.axis < -1 || md.axis >= effective_nvars(md)) {
            throw std::invalid_argument("axis must be -1 or within the number of variables in Dual");
        }
        return md;
    }

    XDIFF_INLINE_HOST_DEVICE
    static int effective_nvars(MakeDual md){
        if (md.nvars >= 0) {
            return md.nvars;
        }
        if constexpr (NVARS >= 0){
            return NVARS;
        } else {
            return int(xdiff::detail::DefaultNvarsHolder<T>::default_nvars);
        }
    }

    XDIFF_INLINE_HOST_DEVICE
    static size_t nv(int nvars){
        if constexpr (NVARS >= 0){
            return size_t(NVARS);
        } else {
            return nvars >= 0 ? size_t(nvars) : xdiff::detail::DefaultNvarsHolder<T>::default_nvars;
        }
    }

    [[no_unique_address]] std::conditional_t<NVARS == -1, size_t, xdiff::detail::NoNvars> nvars_;

    friend struct detail::NestedDualOperationHelper;
};


} // namespace xdiff


#include "nested_dual_operators.hpp" // IWYU pragma: keep


#endif // XDIFF_DUAL_NESTED_DUAL_HPP